#!/usr/bin/env python
"""
Attention Extraction Script for Wan2.1

Extracts attention weights during inference and saves them for visualization.

Usage:
    python experiments/extract_attention_wan.py \
        --ckpt_dir ./Wan2.1-T2V-1.3B \
        --layer_index 20 \
        --output_path cache/layer20.pt \
        --prompt "A majestic eagle soaring through a cloudy sky"
"""

import argparse
import os
import sys

import torch


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("Error: CUDA is required")
        sys.exit(1)

    # Patch attention before importing wan
    from wan.modules.attention import ATTENTION_WEIGHT_CAPTURE


    # Now import wan
    import wan
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Load config
    task = args.task
    config = WAN_CONFIGS[task]

    print(f"\nTask: {task}")
    print(f"Checkpoint: {args.ckpt_dir}")
    print(f"Layer to capture: {args.layer_index}")
    print(f"Prompt: {args.prompt}")

    # Initialize model
    print("\nInitializing WanT2V...")
    wan_t2v = wan.WanT2V(
        config=config,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.gpu_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=args.t5_cpu,
    )

    # Get model info
    num_layers = len(wan_t2v.model.blocks)
    num_heads = config.num_heads
    print(f"Model layers: {num_layers}")
    print(f"Num heads: {num_heads}")

    if args.layer_index >= num_layers:
        print(f"Error: layer_index {args.layer_index} >= num_layers {num_layers}")
        sys.exit(1)

    # Each block has 2 attention calls: self-attention (even) and cross-attention (odd)
    # Layer N -> self-attention call index = 2*N
    self_attn_idx = 2 * args.layer_index
    print(f"Layer {args.layer_index} -> self-attn call index: {self_attn_idx}")

    # Enable attention capture
    ATTENTION_WEIGHT_CAPTURE.enable(
        layer_indices=[self_attn_idx],
        capture_logits=True,
        num_layers=num_layers * 2  # *2 for self+cross attention
    )

    # Parse size
    size = tuple(map(int, args.size.split('*')))

    try:
        print(f"\nRunning inference with size={size}, frames={args.num_frames}...")
        output = wan_t2v.generate(
            input_prompt=args.prompt,
            size=size,
            frame_num=args.num_frames,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            offload_model=args.offload_model,
        )
        print(f"Generation complete!")

    finally:
        captured_weights = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
        ATTENTION_WEIGHT_CAPTURE.disable()

    print(f"\nCaptured {len(captured_weights)} attention matrices")

    if not captured_weights:
        print("Error: No attention captured!")
        print("This may happen if flash_attention is being used instead of the patched version.")
        sys.exit(1)

    # Process captured attention into frame-level attention matrix
    print("\n" + "=" * 60)
    print("Processing attention data")
    print("=" * 60)

    # Calculate frame sequence length based on size and model config
    vae_stride = config.vae_stride  # e.g., (4, 8, 8)
    patch_size = config.patch_size  # e.g., (1, 2, 2)

    # Latent dimensions
    h_latent = size[1] // vae_stride[1]
    w_latent = size[0] // vae_stride[2]
    tokens_per_frame = (h_latent // patch_size[1]) * (w_latent // patch_size[2])

    print(f"Tokens per frame: {tokens_per_frame}")
    print(f"Num frames: {args.num_frames}")

    # Group by denoising step
    # Each denoising step has 2 model calls (cond + uncond for CFG)
    # We capture self-attention from each

    # Use the first captured attention to determine structure
    first_attn = captured_weights[0]
    q_shape = first_attn['q_shape']
    k_shape = first_attn['k_shape']

    print(f"First attention Q shape: {q_shape}")
    print(f"First attention K shape: {k_shape}")

    # Calculate frame-level attention
    # For T2V, Q and K have same length = num_frames * tokens_per_frame
    total_tokens = q_shape[1]
    inferred_frames = total_tokens // tokens_per_frame

    if inferred_frames != args.num_frames:
        print(f"Warning: inferred frames ({inferred_frames}) != specified frames ({args.num_frames})")
        print(f"Using inferred frame count.")

    num_frames = inferred_frames

    # Build full frame attention matrix from captured data
    # Use the last denoising step's conditional attention (typically cleaner)
    # Find attention with most tokens (should be full sequence)
    best_attn = max(captured_weights, key=lambda x: x['k_shape'][1])

    attn_logits = best_attn['attn_weights'].float()  # [B, num_heads, Lq, Lk]
    if attn_logits.dim() == 4:
        attn_logits = attn_logits[0]  # [num_heads, Lq, Lk]

    print(f"Selected attention shape: {attn_logits.shape}")

    # Compute frame-level attention by averaging over token pairs
    full_frame_attn = torch.zeros(num_heads, num_frames, num_frames, dtype=torch.float32)

    for h in range(num_heads):
        head_attn = attn_logits[h]  # [Lq, Lk]

        for qf in range(num_frames):
            q_start = qf * tokens_per_frame
            q_end = (qf + 1) * tokens_per_frame

            for kf in range(num_frames):
                k_start = kf * tokens_per_frame
                k_end = (kf + 1) * tokens_per_frame

                # Average attention over all token pairs
                frame_attn_val = head_attn[q_start:q_end, k_start:k_end].mean()
                full_frame_attn[h, qf, kf] = frame_attn_val

    print(f"Full frame attention shape: {full_frame_attn.shape}")
    print(f"Range: [{full_frame_attn.min():.4f}, {full_frame_attn.max():.4f}]")

    # Compute last block attention (for bar chart)
    # Assume last 3 frames as "last block"
    last_block_size = min(3, num_frames)
    last_block_q_start = num_frames - last_block_size
    last_block_q_frames = list(range(last_block_q_start, num_frames))

    last_block_frame_attn = full_frame_attn[:, last_block_q_start:, :].mean(dim=1)  # [num_heads, num_frames]

    # Save data
    save_data = {
        'layer_index': args.layer_index,
        'full_frame_attention': full_frame_attn.to(torch.float16),
        'last_block_frame_attention': last_block_frame_attn.to(torch.float16),
        'is_logits': True,
        'prompt': args.prompt,
        'num_frames': num_frames,
        'tokens_per_frame': tokens_per_frame,
        'num_heads': num_heads,
        'block_sizes': [last_block_size] * (num_frames // last_block_size),
        'query_frames': list(range(num_frames)),
        'key_frames': list(range(num_frames)),
        'last_block_query_frames': last_block_q_frames,
    }

    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    torch.save(save_data, args.output_path)
    print(f"\nSaved to: {args.output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Layer: {args.layer_index}")
    print(f"Full attention: {tuple(full_frame_attn.shape)} [num_heads, Q_frames, K_frames]")
    print(f"Last block attention: {tuple(last_block_frame_attn.shape)} [num_heads, K_frames]")

    # Analyze diagonal and sink
    diag_mean = torch.diagonal(full_frame_attn, dim1=1, dim2=2).mean().item()
    first_col = full_frame_attn[:, :, 0]
    first_col_mean = first_col.mean().item()
    print(f"Diagonal mean (self-attention): {diag_mean:.4f}")
    print(f"First frame mean (sink): {first_col_mean:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--output_path", type=str, default="cache/attention_layer.pt",
                        help="Output path for attention data")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--task", type=str, default="t2v-1.3B",
                        choices=["t2v-1.3B", "t2v-14B"],
                        help="Model task/size")
    parser.add_argument("--size", type=str, default="832*480",
                        help="Output size (width*height)")
    parser.add_argument("--num_frames", type=int, default=21,
                        help="Number of frames to generate")
    parser.add_argument("--layer_index", type=int, default=20,
                        help="Layer index to capture (0 to num_layers-1)")
    parser.add_argument("--sample_steps", type=int, default=50,
                        help="Number of sampling steps")
    parser.add_argument("--sample_shift", type=float, default=8.0,
                        help="Sampling shift")
    parser.add_argument("--guide_scale", type=float, default=6.0,
                        help="Guidance scale")
    parser.add_argument("--sample_solver", type=str, default="unipc",
                        choices=["unipc", "dpm++"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                        help="Place T5 on CPU to save VRAM")
    parser.add_argument("--offload_model", action="store_true", default=False,
                        help="Offload model to CPU after each forward")
    return parser.parse_args()


if __name__ == "__main__":
    main()
