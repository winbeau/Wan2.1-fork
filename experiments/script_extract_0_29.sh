#!/bin/bash

# 循环 30 次，变量 i 从 0 到 29
for i in {0..29}; do
  idx=$((i))

  echo "正在执行层级: $idx ..."

  python experiments/run_extraction_each.py \
      --layer_index $idx \
      --output_path cache/layer${idx}.pt \
      --checkpoint_path checkpoints/self_forcing_dmd.pt
done
