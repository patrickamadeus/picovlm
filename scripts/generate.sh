#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=
export TOKENIZERS_PARALLELISM=false
export HF_HOME=

python generate.py \
  --checkpoint "lusxvr/nanoVLM-230M-8k" \
  --image "assets/cat.png" \
  --prompt "Describe the image." \
  --generations 3 \
  --max_new_tokens 64 \
  --top_k 50 \
  --top_p 0.9 \
  --temperature 0.7
