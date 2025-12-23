#!/usr/bin/env python
"""
Figure 4 注意力权重提取脚本（v5 - 复用原版推理代码）

核心思路：直接使用模型的 ATTENTION_WEIGHT_CAPTURE 机制在推理过程中捕获真实的 attention。
这样可以确保与模型的实际计算完全一致。

用法：
    python run_extraction_figure4.py \
        --config_path configs/self_forcing_dmd.yaml \
        --output_path cache/attention_cache_figure4.pt \
        --layer_indices 0 4 \
        --no_checkpoint  # 使用原始 Wan 模型
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
        print("使用原始 Wan2.1 基础模型（不加载任何 checkpoint）")
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            pipeline.generator.load_state_dict(state_dict['generator'])
    else:
        print("警告: checkpoint 不存在，使用原始 Wan2.1 基础模型")

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval()

    num_frame_per_block = config.get('num_frame_per_block', 3)

    print(f"\nNum frames: {args.num_frames}")
    print(f"Num frames per block: {num_frame_per_block}")
    print(f"Layer indices to capture: {args.layer_indices}")
    print(f"Prompt: {args.prompt}")

    # 创建输入噪声
    batch_size = 1
    noise = torch.randn(
        [batch_size, args.num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # ========== 使用 ATTENTION_WEIGHT_CAPTURE 在推理过程中捕获 ==========
    print("\n" + "=" * 60)
    print("在推理过程中捕获 attention（使用模型原生机制）")
    print("=" * 60)

    # 获取模型层数
    num_layers = len(pipeline.generator.model.blocks)
    print(f"模型层数: {num_layers}")

    # 重要：每个 block 有 2 次 attention 调用：
    #   - 偶数索引 (0, 2, 4, ...) = self-attention (video tokens)
    #   - 奇数索引 (1, 3, 5, ...) = cross-attention (text tokens)
    # 用户请求的 layer_indices 需要映射到 self-attention 的调用索引
    # layer N → 调用索引 2*N
    if args.layer_indices is not None:
        self_attn_indices = [2 * layer_idx for layer_idx in args.layer_indices]
        print(f"请求层: {args.layer_indices} → 映射到 self-attn 调用索引: {self_attn_indices}")
    else:
        # 如果没有指定，捕获所有层的 self-attention
        self_attn_indices = [2 * i for i in range(num_layers)]
        print(f"捕获所有 {num_layers} 层的 self-attention")

    # 启用 attention 捕获
    # num_layers 设为 2*num_layers，因为每个 block 有 2 次调用
    ATTENTION_WEIGHT_CAPTURE.enable(
        layer_indices=self_attn_indices,
        capture_logits=True,  # 捕获 pre-softmax logits
        num_layers=num_layers * 2  # 每个 block 有 2 次 attention 调用
    )

    try:
        # 执行推理
        print("\n执行推理...")
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
        # 获取捕获的 attention
        captured_weights = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
        ATTENTION_WEIGHT_CAPTURE.disable()

    print(f"\n捕获到 {len(captured_weights)} 个 attention 矩阵")

    if not captured_weights:
        print("Error: No attention captured!")
        sys.exit(1)

    # ========== 处理和保存捕获的数据 ==========
    # 按层分组，找到最后一个 block 的 attention（K 长度最大的）
    # 注意：layer_idx 是调用索引（0, 2, 4, ...），需要转换回 block 索引（0, 1, 2, ...）
    layer_attentions = {}
    for attn in captured_weights:
        call_idx = attn['layer_idx']  # 调用索引 (0, 2, 4, ...)
        block_idx = call_idx // 2     # 转换回 block 索引 (0, 1, 2, ...)
        if block_idx not in layer_attentions:
            layer_attentions[block_idx] = []
        layer_attentions[block_idx].append(attn)

    # 每层选择 K 长度最大的（对应最后一个 temporal block，包含所有历史帧）
    final_attentions = []
    for block_idx in sorted(layer_attentions.keys()):
        attns = layer_attentions[block_idx]
        # 按 K 的长度排序，取最大的
        attns_sorted = sorted(attns, key=lambda x: x['k_shape'][1], reverse=True)
        selected = attns_sorted[0]
        # 更新 layer_idx 为 block 索引
        selected = {**selected, 'layer_idx': block_idx}
        final_attentions.append(selected)
        print(f"Layer {block_idx}: selected attention with Q={selected['q_shape']}, K={selected['k_shape']}")

    # 计算 frame_seq_length
    # 从 K 的形状推断：K shape 应该是 [B, L_k, N, D]
    # L_k = num_key_frames * frame_seq_length
    first_attn = final_attentions[0]
    k_len = first_attn['k_shape'][1]
    q_len = first_attn['q_shape'][1]

    # 推断 frame_seq_length
    # 最后一个 block 的 Q 有 num_frame_per_block 帧
    frame_seq_length = q_len // num_frame_per_block
    num_key_frames = k_len // frame_seq_length

    print(f"\n推断的参数:")
    print(f"  frame_seq_length: {frame_seq_length}")
    print(f"  num_key_frames: {num_key_frames}")
    print(f"  num_query_frames: {num_frame_per_block}")

    # 保存数据
    save_data = {
        'attention_weights': [],
        'prompt': args.prompt,
        'num_frames': args.num_frames,
        'frame_seq_length': frame_seq_length,
        'num_frame_per_block': num_frame_per_block,
        'layer_indices': args.layer_indices,
        'is_logits': True,
        'capture_method': 'native_inference_capture',
        'query_frames': list(range(args.num_frames - num_frame_per_block, args.num_frames)),
        'key_frames': list(range(num_key_frames)),
    }

    for attn in final_attentions:
        layer_idx = attn['layer_idx']
        attn_data = attn['attn_weights']  # [B, N, Q, K]

        save_data['attention_weights'].append({
            'layer_idx': layer_idx,
            'attn_logits': attn_data.to(torch.float16),
            'is_logits': attn['is_logits'],
            'num_frames': args.num_frames,
            'frame_seq_length': frame_seq_length,
            'num_heads': attn_data.shape[1],
            'num_query_frames': num_frame_per_block,
            'num_key_frames': num_key_frames,
        })

    torch.save(save_data, args.output_path)
    print(f"\nSaved to: {args.output_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("ATTENTION DATA SUMMARY")
    print("=" * 60)
    for w in save_data['attention_weights']:
        layer = w['layer_idx']
        logits = w['attn_logits'].float()
        shape = tuple(logits.shape)
        print(f"Layer {layer}:")
        print(f"  Shape: {shape}")
        print(f"  Range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")

        # 计算帧间统计（使用 softmax）
        import torch.nn.functional as F
        weights = F.softmax(logits, dim=-1)

        fsl = w['frame_seq_length']
        num_heads = w['num_heads']
        num_kf = w['num_key_frames']

        sink_ratios = []
        for h in range(num_heads):
            head_w = weights[0, h].mean(dim=0)  # [K]
            frame_w = torch.tensor([head_w[f*fsl:(f+1)*fsl].mean().item() for f in range(num_kf)])
            first = frame_w[0].item()
            mid = frame_w[1:-1].mean().item() if num_kf > 2 else frame_w.mean().item()
            sink_ratios.append(first / mid if mid > 0 else 0)

        print(f"  Sink ratios (softmax): min={min(sink_ratios):.2f}x, max={max(sink_ratios):.2f}x, avg={sum(sink_ratios)/len(sink_ratios):.2f}x")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="cache/attention_cache_figure4.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--layer_indices", type=int, nargs='+', default=[0, 4])
    parser.add_argument("--no_checkpoint", action="store_true", default=False,
                        help="不加载 checkpoint，使用原始 Wan2.1 基础模型")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
