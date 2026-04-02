#!/usr/bin/env bash

set -euo pipefail

export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=
export TOKENIZERS_PARALLELISM=false
export HF_HOME=

python evaluate.py \
  --checkpoint "lusxvr/nanoVLM-230M-8k" \
  --tasks "mmstar" \
  --batch_size 8 \
  --device cuda \
  --output_path "eval_results/mmstar.json"
