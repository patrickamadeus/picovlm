# picoVLM

A small, readable vision-language model repo for training, generation, and evaluation in plain PyTorch.

`picoVLM` is a trimmed fork in the spirit of [`nanoVLM`](https://github.com/huggingface/nanoVLM): keep the codepath short, keep the scripts direct, and keep only the pieces needed for practical local work.

## What This Repo Keeps

- one main VLM implementation
- one default YAML config
- one direct training script
- one direct generation script
- one direct `lmms-eval` evaluation script
- lightweight structured logging and process logs

## What This Repo Trims

- no notebook-first workflow
- no extra README assets
- no large experimental surface
- no extra framework abstraction beyond the essentials

## Quick Start

```bash
git clone https://github.com/patrickamadeus/picovlm.git
cd picovlm
uv sync --python 3.12 --extra eval
source .venv/bin/activate
```

This creates a repo-local environment in `.venv/`.

## Train

```bash
python train.py --config nanovlm
```

Default config:

- [`configs/yaml/nanovlm.yml`](./configs/yaml/nanovlm.yml)

Training outputs:

- `results/run_<timestamp>/loss.jsonl`
- `results/run_<timestamp>/samples.jsonl`
- `results/run_<timestamp>/loss.png`
- `results/run_<timestamp>/resolved_config.yml`

Process log:

- `logs/train_<timestamp>.log`

The training progress bar reports:

- `step`
- `batch_loss`
- `toks`

`consumed_tokens` tracks visible non-masked tokens from `attention_mask`, not just supervised target tokens.

## Generate

```bash
python generate.py \
  --checkpoint /path/to/checkpoint \
  --image assets/cat.png \
  --prompt "Describe the image."
```

Process log:

- `logs/generate_<timestamp>.log`

## Evaluate

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint \
  --tasks mmstar \
  --batch_size 8
```

Optional saved results:

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint \
  --tasks mmstar \
  --batch_size 8 \
  --output_path eval_results/mmstar.json
```

Process log:

- `logs/evaluate_<timestamp>.log`

`lmms-eval` is installed from source through the `eval` extra in [`pyproject.toml`](./pyproject.toml).

## Main Files

- [`train.py`](./train.py)
- [`generate.py`](./generate.py)
- [`evaluate.py`](./evaluate.py)
- [`models/nanovlm.py`](./models/nanovlm.py)
- [`configs/config.py`](./configs/config.py)
- [`utils/train_helper.py`](./utils/train_helper.py)
- [`utils/generation_helper.py`](./utils/generation_helper.py)
- [`utils/eval_wrapper.py`](./utils/eval_wrapper.py)

## Validation

The repo-local `uv` environment has been tested for:

- training-related tests
- generation helper tests
- evaluation tests
- logging-path tests
- `torch`, `transformers`, `datasets`, `yaml`, `matplotlib`, and `lmms_eval` imports

## Acknowledgement

This repo is inspired by:

- https://github.com/huggingface/nanoVLM

Upstream citation:

```bibtex
@misc{wiedmann2025nanovlm,
  author = {Luis Wiedmann and Aritra Roy Gosthipaty and Andrés Marafioti},
  title = {nanoVLM},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/huggingface/nanoVLM}}
}
```
