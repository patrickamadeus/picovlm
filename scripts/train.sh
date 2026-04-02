#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES=
export TOKENIZERS_PARALLELISM=false
export HF_HOME=
export WANDB_DISABLED=

python train.py --config nanovlm
