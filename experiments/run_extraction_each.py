#!/usr/bin/env python
"""
单层全注意力矩阵提取脚本

提取单层的完整 frame×frame 注意力矩阵（组合所有 block）。

用法：
    PYTHONPATH=. python experiments/run_extraction_each.py \
        --layer_index 3 \
        --output_path cache/layer3.pt \
        --checkpoint_path checkpoints/self_forcing_dmd.pt

    PYTHONPATH=. python experiments/run_extraction_each.py \
        --layer_index 15 \
        --output_path cache/layer15_wan.pt \
        --no_checkpoint
"""

import argparse
import torch
import os
import sys


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        sys.exit(1)

    from omegaconf import OmegaConf
    from pipeline.causal_inference import CausalInferencePipeline
    from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE
    from utils.misc import set_seed

    set_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    # 加载配置
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    print("Initializing inference pipeline...")
    torch.set_grad_enabled(False)

    pipeline = CausalInferencePipeline(args=config, device=device)

    # 加载 checkpoint
    if args.no_checkpoint:
        print("Using original Wan2.1 base model (no checkpoint)")
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            pipeline.generator.load_state_dict(state_dict['generator'])
    else:
        print("Warning: checkpoint not found, using original Wan2.1 base model")

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval()

    num_frame_per_block = config.get('num_frame_per_block', 3)
    num_frames = args.num_frames
    frame_seq_length = 1560
    layer_index = args.layer_index

    # 计算 block 结构
    independent_first_frame = config.get('independent_first_frame', True)
    if independent_first_frame:
        num_blocks = (num_frames - 1) // num_frame_per_block + 1
        block_sizes = [1] + [num_frame_per_block] * ((num_frames - 1) // num_frame_per_block)
    else:
        num_blocks = num_frames // num_frame_per_block
        block_sizes = [num_frame_per_block] * num_blocks

    print(f"\nLayer to capture: {layer_index}")
    print(f"Num frames: {num_frames}")
    print(f"Block structure: {block_sizes} (total {num_blocks} blocks)")
    print(f"Prompt: {args.prompt}")

    # 创建输入噪声
    noise = torch.randn(
        [1, num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # 获取模型层数
    num_layers = len(pipeline.generator.model.blocks)
    print(f"Model layers: {num_layers}")

    if layer_index >= num_layers:
        print(f"Error: layer_index {layer_index} >= num_layers {num_layers}")
        sys.exit(1)

    # 每个 block 有 2 次 attention 调用：
    #   - 偶数索引 = self-attention (video tokens)
    #   - 奇数索引 = cross-attention (text tokens)
    # layer N → 调用索引 2*N
    self_attn_idx = 2 * layer_index
    print(f"Layer {layer_index} → self-attn call index: {self_attn_idx}")

    # 启用 attention 捕获
    ATTENTION_WEIGHT_CAPTURE.enable(
        layer_indices=[self_attn_idx],
        capture_logits=True,
        num_layers=num_layers * 2
    )

    try:
        print("\nRunning inference...")
        output = pipeline.inference(
            noise=noise,
            text_prompts=[args.prompt],
            return_latents=True,
        )
        if isinstance(output, tuple):
            output_latent = output[1]
        else:
            output_latent = output
        print(f"Output latent shape: {output_latent.shape}")

    finally:
        captured_weights = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
        ATTENTION_WEIGHT_CAPTURE.disable()

    print(f"\nCaptured {len(captured_weights)} attention matrices")

    if not captured_weights:
        print("Error: No attention captured!")
        sys.exit(1)

    # ========== 构建完整的 frame×frame 注意力矩阵 ==========
    print("\n" + "=" * 60)
    print("Building full attention matrix")
    print("=" * 60)

    # 按 K 长度排序（升序），对应 block 顺序
    attns_sorted = sorted(captured_weights, key=lambda x: x['k_shape'][1])

    # 获取 head 数量
    num_heads = attns_sorted[0]['attn_weights'].shape[1]
    print(f"Num heads: {num_heads}")

    # 初始化完整的 frame-level 注意力矩阵: [num_heads, num_frames, num_frames]
    full_frame_attn = torch.zeros(num_heads, num_frames, num_frames, dtype=torch.float32)

    # 处理每个 block
    processed_k_frames = set()
    current_q_start = 0

    for block_idx, block_size in enumerate(block_sizes):
        expected_k_frames = sum(block_sizes[:block_idx + 1])

        # 找到匹配的 attention
        matching_attn = None
        for a in attns_sorted:
            k_frames = a['k_shape'][1] // frame_seq_length
            if k_frames == expected_k_frames and k_frames not in processed_k_frames:
                matching_attn = a
                processed_k_frames.add(k_frames)
                break

        if matching_attn is None:
            print(f"  Warning: No matching attention for block {block_idx}")
            current_q_start += block_size
            continue

        # 提取注意力数据
        attn_logits = matching_attn['attn_weights'][0].float()  # [num_heads, Lq, Lk]
        q_tokens = attn_logits.shape[1]
        k_tokens = attn_logits.shape[2]
        q_frames_in_block = q_tokens // frame_seq_length
        k_frames_total = k_tokens // frame_seq_length

        print(f"  Block {block_idx}: Q frames {current_q_start}-{current_q_start + q_frames_in_block - 1}, "
              f"K frames 0-{k_frames_total - 1}")

        # 计算 frame-level 注意力
        for h in range(num_heads):
            head_attn = attn_logits[h]  # [Lq, Lk]

            for qf_local in range(q_frames_in_block):
                qf_global = current_q_start + qf_local
                q_start_tok = qf_local * frame_seq_length
                q_end_tok = (qf_local + 1) * frame_seq_length

                for kf in range(k_frames_total):
                    k_start_tok = kf * frame_seq_length
                    k_end_tok = (kf + 1) * frame_seq_length

                    # 平均所有 token pair 的注意力
                    frame_attn_val = head_attn[q_start_tok:q_end_tok, k_start_tok:k_end_tok].mean()
                    full_frame_attn[h, qf_global, kf] = frame_attn_val

        current_q_start += block_size

    print(f"\nFull attention matrix shape: {full_frame_attn.shape}")
    print(f"Range: [{full_frame_attn.min():.4f}, {full_frame_attn.max():.4f}]")

    # ========== 计算最后一个 block 的 per-head 帧注意力（用于柱状图） ==========
    # 最后一个 block 的 Q frames
    last_block_q_start = sum(block_sizes[:-1])
    last_block_q_end = num_frames
    last_block_q_frames = list(range(last_block_q_start, last_block_q_end))

    # 计算每个 head 对每个 key frame 的平均注意力（只看最后一个 block 的 query）
    last_block_frame_attn = full_frame_attn[:, last_block_q_start:last_block_q_end, :].mean(dim=1)  # [num_heads, num_frames]

    # ========== 保存数据 ==========
    save_data = {
        'layer_index': layer_index,
        'full_frame_attention': full_frame_attn.to(torch.float16),  # [num_heads, num_frames, num_frames]
        'last_block_frame_attention': last_block_frame_attn.to(torch.float16),  # [num_heads, num_frames]
        'is_logits': True,
        'prompt': args.prompt,
        'num_frames': num_frames,
        'frame_seq_length': frame_seq_length,
        'num_frame_per_block': num_frame_per_block,
        'num_heads': num_heads,
        'block_sizes': block_sizes,
        'query_frames': list(range(num_frames)),
        'key_frames': list(range(num_frames)),
        'last_block_query_frames': last_block_q_frames,
    }

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    torch.save(save_data, args.output_path)
    print(f"\nSaved to: {args.output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Layer: {layer_index}")
    print(f"Full attention: {tuple(full_frame_attn.shape)} [num_heads, Q_frames, K_frames]")
    print(f"Last block attention: {tuple(last_block_frame_attn.shape)} [num_heads, K_frames]")
    print(f"Last block Q frames: {last_block_q_frames}")

    # 分析对角线和 sink
    diag_mean = torch.diagonal(full_frame_attn, dim1=1, dim2=2).mean().item()
    first_col = full_frame_attn[:, :, 0]
    first_col_mean = first_col[first_col != 0].mean().item() if (first_col != 0).any() else 0
    print(f"Diagonal mean: {diag_mean:.4f}")
    print(f"First frame (sink) mean: {first_col_mean:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="cache/attention_layer.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--layer_index", type=int, default=0, help="Layer index to capture (0-29)")
    parser.add_argument("--no_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
