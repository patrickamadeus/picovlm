#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=,
export TOKENIZERS_PARALLELISM=false
export HF_HOME=
export WANDB_DISABLED=
export MASTER_ADDR=
export MASTER_PORT=
export OMP_NUM_THREADS=

torchrun \
  --nproc_per_node=2 \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  train.py \
  --config nanovlm
