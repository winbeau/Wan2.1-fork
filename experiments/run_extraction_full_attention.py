#!/usr/bin/env python
"""
Full Attention Matrix Extraction Script

Captures attention weights from ALL blocks during autoregressive inference,
then assembles them into a full (num_frames × num_frames) attention matrix.

This allows visualization of how each frame attends to all previous frames
(lower-triangular due to causal attention).

Usage:
    PYTHONPATH=. python experiments/run_extraction_full_attention.py \
        --config_path configs/self_forcing_dmd.yaml \
        --output_path cache/attention_full_matrix.pt \
        --layer_indices 0 4 \
        --num_frames 21 \
        --no_checkpoint
"""

import argparse
import torch
import os
import sys
from collections import defaultdict


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

    # Load config
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)

    print("Initializing inference pipeline...")
    torch.set_grad_enabled(False)

    pipeline = CausalInferencePipeline(args=config, device=device)

    # Load checkpoint
    if args.no_checkpoint:
        print("Using base Wan2.1 model (no checkpoint loaded)")
    elif args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
        key = 'generator_ema' if args.use_ema else 'generator'
        if key in state_dict:
            pipeline.generator.load_state_dict(state_dict[key])
        else:
            pipeline.generator.load_state_dict(state_dict['generator'])
    else:
        print("Warning: checkpoint does not exist, using base Wan2.1 model")

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.eval()

    num_frame_per_block = config.get('num_frame_per_block', 3)
    num_frames = args.num_frames
    frame_seq_length = 1560  # tokens per frame

    # Calculate number of blocks
    # With independent_first_frame=True: [1, 3, 3, 3, 3, 3, 3] for 21 frames
    # Block structure depends on config
    independent_first_frame = config.get('independent_first_frame', True)
    if independent_first_frame:
        num_blocks = (num_frames - 1) // num_frame_per_block + 1
        block_sizes = [1] + [num_frame_per_block] * ((num_frames - 1) // num_frame_per_block)
    else:
        num_blocks = num_frames // num_frame_per_block
        block_sizes = [num_frame_per_block] * num_blocks

    print(f"\nNum frames: {num_frames}")
    print(f"Num frames per block: {num_frame_per_block}")
    print(f"Independent first frame: {independent_first_frame}")
    print(f"Block structure: {block_sizes} (total {num_blocks} blocks)")
    print(f"Layer indices to capture: {args.layer_indices}")
    print(f"Prompt: {args.prompt}")

    # Create input noise
    batch_size = 1
    noise = torch.randn(
        [batch_size, num_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )

    # Get number of model layers
    num_layers = len(pipeline.generator.model.blocks)
    print(f"Model layers: {num_layers}")

    # Enable attention capture for ALL calls (we'll filter later)
    ATTENTION_WEIGHT_CAPTURE.enable(
        layer_indices=args.layer_indices,
        capture_logits=True,
        num_layers=num_layers
    )

    try:
        # Run inference
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
        # Get captured attention
        captured_weights = ATTENTION_WEIGHT_CAPTURE.captured_weights.copy()
        ATTENTION_WEIGHT_CAPTURE.disable()

    print(f"\nCaptured {len(captured_weights)} attention matrices")

    if not captured_weights:
        print("Error: No attention captured!")
        sys.exit(1)

    # ========== Organize captured attention by layer and block ==========
    # Group by layer index
    layer_attentions = defaultdict(list)
    for attn in captured_weights:
        layer_idx = attn['layer_idx']
        layer_attentions[layer_idx].append(attn)

    print("\n" + "=" * 60)
    print("Organizing attention data by block")
    print("=" * 60)

    # For each layer, we should have attention from each block
    # The K dimension grows as blocks progress
    for layer_idx in sorted(layer_attentions.keys()):
        attns = layer_attentions[layer_idx]
        print(f"\nLayer {layer_idx}: {len(attns)} attention captures")

        # Sort by K length to understand block progression
        attns_sorted = sorted(attns, key=lambda x: x['k_shape'][1])
        for i, a in enumerate(attns_sorted[:5]):  # Show first 5
            q_len = a['q_shape'][1]
            k_len = a['k_shape'][1]
            q_frames = q_len // frame_seq_length
            k_frames = k_len // frame_seq_length
            print(f"  [{i}] Q={q_len} ({q_frames} frames), K={k_len} ({k_frames} frames)")

    # ========== Build full attention matrix ==========
    print("\n" + "=" * 60)
    print("Building full attention matrix")
    print("=" * 60)

    save_data = {
        'attention_weights': [],
        'prompt': args.prompt,
        'num_frames': num_frames,
        'frame_seq_length': frame_seq_length,
        'num_frame_per_block': num_frame_per_block,
        'layer_indices': args.layer_indices,
        'is_logits': True,
        'capture_method': 'full_matrix_all_blocks',
        'block_sizes': block_sizes,
    }

    for layer_idx in sorted(layer_attentions.keys()):
        attns = layer_attentions[layer_idx]

        # Sort by K length (ascending) to get block order
        attns_sorted = sorted(attns, key=lambda x: x['k_shape'][1])

        # Get number of heads from first attention
        num_heads = attns_sorted[0]['attn_weights'].shape[1]

        # Initialize full frame-level attention matrix: [num_heads, num_frames, num_frames]
        full_frame_attn = torch.zeros(num_heads, num_frames, num_frames, dtype=torch.float32)

        # Track which blocks we've processed
        processed_k_frames = set()

        # Process each captured attention (should correspond to different blocks)
        current_q_start = 0
        for block_idx, block_size in enumerate(block_sizes):
            # Find the attention that matches this block
            expected_k_frames = sum(block_sizes[:block_idx + 1])

            # Find matching attention by K length
            matching_attn = None
            for a in attns_sorted:
                k_frames = a['k_shape'][1] // frame_seq_length
                if k_frames == expected_k_frames and k_frames not in processed_k_frames:
                    matching_attn = a
                    processed_k_frames.add(k_frames)
                    break

            if matching_attn is None:
                print(f"  Warning: No matching attention for block {block_idx} (expected K={expected_k_frames} frames)")
                current_q_start += block_size
                continue

            # Extract attention data
            attn_logits = matching_attn['attn_weights'][0].float()  # [num_heads, Lq, Lk]
            q_tokens = attn_logits.shape[1]
            k_tokens = attn_logits.shape[2]
            q_frames_in_block = q_tokens // frame_seq_length
            k_frames_total = k_tokens // frame_seq_length

            print(f"  Block {block_idx}: Q frames {current_q_start}-{current_q_start + q_frames_in_block - 1}, "
                  f"K frames 0-{k_frames_total - 1}")

            # Compute frame-level attention by averaging over tokens
            for h in range(num_heads):
                head_attn = attn_logits[h]  # [Lq, Lk]

                for qf_local in range(q_frames_in_block):
                    qf_global = current_q_start + qf_local
                    q_start_tok = qf_local * frame_seq_length
                    q_end_tok = (qf_local + 1) * frame_seq_length

                    for kf in range(k_frames_total):
                        k_start_tok = kf * frame_seq_length
                        k_end_tok = (kf + 1) * frame_seq_length

                        # Average attention over all token pairs for this frame pair
                        frame_attn_val = head_attn[q_start_tok:q_end_tok, k_start_tok:k_end_tok].mean()
                        full_frame_attn[h, qf_global, kf] = frame_attn_val

            current_q_start += block_size

        print(f"  Full attention matrix shape: {full_frame_attn.shape}")
        print(f"  Range: [{full_frame_attn.min():.4f}, {full_frame_attn.max():.4f}]")

        # Save layer data
        save_data['attention_weights'].append({
            'layer_idx': layer_idx,
            'full_frame_attention': full_frame_attn.to(torch.float16),  # [num_heads, num_frames, num_frames]
            'is_logits': True,
            'num_frames': num_frames,
            'num_heads': num_heads,
        })

    # Save data
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(save_data, args.output_path)
    print(f"\nSaved to: {args.output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FULL ATTENTION MATRIX SUMMARY")
    print("=" * 60)
    for w in save_data['attention_weights']:
        layer = w['layer_idx']
        attn = w['full_frame_attention'].float()
        print(f"Layer {layer}:")
        print(f"  Shape: {tuple(attn.shape)} [num_heads, num_query_frames, num_key_frames]")
        print(f"  Range: [{attn.min().item():.4f}, {attn.max().item():.4f}]")

        # Check diagonal pattern (self-attention)
        diag_mean = torch.diagonal(attn, dim1=1, dim2=2).mean().item()
        off_diag = attn.clone()
        for i in range(attn.shape[1]):
            off_diag[:, i, i] = 0
        off_diag_mean = off_diag.sum().item() / (attn.shape[0] * (attn.shape[1] * attn.shape[2] - attn.shape[1]))
        print(f"  Diagonal mean: {diag_mean:.4f}, Off-diagonal mean: {off_diag_mean:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--output_path", type=str, default="cache/attention_full_matrix.pt")
    parser.add_argument("--prompt", type=str,
                        default="A majestic eagle soaring through a cloudy sky, cinematic lighting")
    parser.add_argument("--num_frames", type=int, default=21)
    parser.add_argument("--layer_indices", type=int, nargs='+', default=[0, 4])
    parser.add_argument("--no_checkpoint", action="store_true", default=False,
                        help="Don't load checkpoint, use base Wan2.1 model")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_id", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
