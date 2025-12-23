#!/bin/bash
# Attention Extraction Script
#
# Usage:
#   bash experiments/extract_attention.sh <ckpt_dir> [layer_index] [output_dir]
#
# Example:
#   bash experiments/extract_attention.sh ./Wan2.1-T2V-1.3B 20 cache

set -e

CKPT_DIR="${1:?Usage: $0 <ckpt_dir> [layer_index] [output_dir]}"
LAYER_INDEX="${2:-20}"
OUTPUT_DIR="${3:-cache}"

# Determine task based on checkpoint dir name
if [[ "$CKPT_DIR" == *"1.3B"* ]]; then
    TASK="t2v-1.3B"
    SIZE="832*480"
    GUIDE_SCALE=6.0
    SAMPLE_SHIFT=8.0
else
    TASK="t2v-14B"
    SIZE="1280*720"
    GUIDE_SCALE=5.0
    SAMPLE_SHIFT=5.0
fi

OUTPUT_PATH="${OUTPUT_DIR}/layer${LAYER_INDEX}.pt"

echo "============================================"
echo "Attention Extraction"
echo "============================================"
echo "Checkpoint: $CKPT_DIR"
echo "Task: $TASK"
echo "Layer: $LAYER_INDEX"
echo "Output: $OUTPUT_PATH"
echo "============================================"

mkdir -p "$OUTPUT_DIR"

python experiments/extract_attention_wan.py \
    --ckpt_dir "$CKPT_DIR" \
    --task "$TASK" \
    --layer_index "$LAYER_INDEX" \
    --output_path "$OUTPUT_PATH" \
    --size "$SIZE" \
    --guide_scale "$GUIDE_SCALE" \
    --sample_shift "$SAMPLE_SHIFT" \
    --num_frames 21 \
    --sample_steps 50 \
    --seed 42 \
    --t5_cpu \
    --offload_model

echo ""
echo "Done! Output saved to: $OUTPUT_PATH"
echo ""
echo "To visualize, open notebooks/extract_all_attention.ipynb and set:"
echo "  DATA_PATH = '../${OUTPUT_PATH}'"
