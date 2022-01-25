#!/usr/bin/env bash

# CLI args with defaults
BATCH_SIZE=${1:-100}
AMP=${2:-false}
NUM_EPOCHS=${3:-100}
LEARNING_RATE=${4:-0.001}
WEIGHT_DECAY=${5:-0.1}

python -m se3_transformer.runtime.training \
  --amp "$AMP" \
  --batch_size "$BATCH_SIZE" \
  --epochs "$NUM_EPOCHS" \
  --lr "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --use_layer_norm \
  --norm \
  --save_ckpt_path model_ani1x.pth \
  --precompute_bases \
  --seed 42 \
  --gradient_clip 10.0 \
