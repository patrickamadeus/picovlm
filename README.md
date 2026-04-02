# picoVLM

The simplest trimmed-down repository in this workspace for training, generating with, and evaluating a small vision-language model in plain PyTorch.

`picoVLM` is a focused fork derived from the `nanoVLM` style of repository design: keep the codepath short, readable, and hackable; keep the training loop explicit; keep Hub and benchmark integration available; cut away everything that is not essential for this repo's immediate use case.

This repository is intentionally small in scope:

- one main VLM implementation
- one default YAML config
- one direct training script
- one direct generation script
- one direct evaluation script
- lightweight fixed-sample qualitative logging during training

It is meant to be a clean baseline you can actually read end to end, then adapt.

## What Is picoVLM?

`picoVLM` is a compact single-image vision-language training and inference repo built around:

- a vision encoder backbone
- a lightweight modality projection bridge
- a decoder-only language backbone
- a plain PyTorch training loop

The default setup in this fork uses:

- `google/siglip2-base-patch16-512` as the vision backbone
- `HuggingFaceTB/SmolLM2-135M-Instruct` as the language backbone/tokenizer
- `lusxvr/nanoVLM-230M-8k` as the default pretrained checkpoint source for warm start

The spirit of the repo is practical rather than maximal:

- readable code over framework abstraction
- explicit scripts over orchestration layers
- local debugability over feature sprawl
- enough logging to understand what training is doing without adding a full experiment platform dependency

## What This Repo Keeps

This fork keeps the pieces that are essential for day-to-day small-VLM work:

- end-to-end local training with a readable loop in [`train.py`](./train.py)
- direct checkpointed generation in [`generate.py`](./generate.py)
- direct `lmms-eval` evaluation in [`evaluate.py`](./evaluate.py)
- Hugging Face model loading and saving through the model class
- dataset loading through `datasets`
- training-time qualitative sample generation on fixed local images
- JSONL loss/sample records and process-level `.log` files
- a `uv`-managed local environment with optional `lmms-eval` install from source

## What This Repo Trims

This fork intentionally trims away a lot of surrounding surface area:

- no notebook-first workflow
- no extra benchmark helper scripts beyond the direct evaluation entrypoint
- no extra repo assets for README presentation
- no deep training framework abstraction
- no separate packaging of submodules
- no attempt to support every training mode or experimental branch from the larger workspace

In short: this repo is for running, inspecting, and modifying one practical baseline cleanly.

## Quick Start

Clone the repo and enter it:

```bash
git clone https://github.com/patrickamadeus/picovlm.git
cd picovlm
```

Set up the local environment with `uv`:

```bash
uv sync --python 3.12 --extra eval
source .venv/bin/activate
```

This creates a repo-local environment at:

```text
.venv/
```

The `eval` extra installs `lmms-eval` from source so that the evaluation script works in the same environment as training and generation.

## Environment Setup

This repository is configured for `uv` and Python `3.12`.

Files involved:

- [`pyproject.toml`](./pyproject.toml)
- [`uv.lock`](./uv.lock)
- [`.python-version`](./.python-version)

Recommended setup:

```bash
uv sync --python 3.12 --extra eval
source .venv/bin/activate
```

If you want to rerun commands without activating first, use:

```bash
uv run python train.py --config nanovlm
uv run python generate.py --checkpoint /path/to/checkpoint --image assets/cat.png --prompt "Describe the image."
uv run python evaluate.py --checkpoint /path/to/checkpoint --tasks mmstar --batch_size 8
```

### Main Dependencies

Core runtime:

- `torch`
- `torchvision`
- `numpy`
- `pillow`
- `datasets`
- `transformers`
- `huggingface-hub`
- `safetensors`
- `pyyaml`
- `einops`
- `matplotlib`
- `tqdm`
- `wandb`
- `termcolor`

Evaluation:

- `lmms-eval` from source

## Repository Layout

Core files:

- [`train.py`](./train.py): training loop
- [`generate.py`](./generate.py): image-conditioned text generation
- [`evaluate.py`](./evaluate.py): `lmms-eval` entrypoint
- [`models/nanovlm.py`](./models/nanovlm.py): main model definition and Hub save/load helpers
- [`configs/config.py`](./configs/config.py): dataclass configs
- [`configs/yaml/nanovlm.yml`](./configs/yaml/nanovlm.yml): default training/model config

Support code:

- [`utils/datasets.py`](./utils/datasets.py): dataset loading and collation
- [`utils/generation_helper.py`](./utils/generation_helper.py): prompt/image preparation and fixed training samples
- [`utils/eval_wrapper.py`](./utils/eval_wrapper.py): `lmms-eval` model wrapper
- [`utils/train_helper.py`](./utils/train_helper.py): logging, plotting, checkpointing, helper utilities
- [`utils/processor_transforms.py`](./utils/processor_transforms.py): tokenizer/image preprocessing helpers

Tests:

- [`tests/test_train.py`](./tests/test_train.py)
- [`tests/test_evaluate.py`](./tests/test_evaluate.py)
- [`tests/test_generation_helper.py`](./tests/test_generation_helper.py)
- [`tests/test_logging_paths.py`](./tests/test_logging_paths.py)

## Configuration

The default YAML config lives at:

- [`configs/yaml/nanovlm.yml`](./configs/yaml/nanovlm.yml)

It defines two top-level sections:

- `vlm`
- `train`

The default config currently assumes:

- pretrained warm start from `lusxvr/nanoVLM-230M-8k`
- The Cauldron as the training dataset source
- single-image examples
- fp32 training by default
- periodic validation during training

To use the default config:

```bash
python train.py --config nanovlm
```

## Training

Run training with:

```bash
python train.py --config nanovlm
```

The training script:

- loads model and train config from YAML
- builds train/validation dataloaders
- optionally resumes from a checkpoint
- runs a plain PyTorch optimizer loop
- performs periodic validation
- logs JSONL metrics and qualitative samples
- optionally saves/pushes checkpoints

### What Gets Logged During Training

There are two kinds of logging in this repo:

1. structured run artifacts in `results/`
2. process-level console logs in `logs/`

#### `results/`

Each training run creates a timestamped directory under:

```text
results/run_<date>_<time>/
```

Typical contents:

- `resolved_config.yml`
- `loss.jsonl`
- `samples.jsonl`
- `loss.png`

`loss.jsonl` stores stepwise structured records such as:

- train loss
- batch loss
- seen tokens for the step
- target tokens for the step
- cumulative consumed tokens
- validation loss events
- checkpoint events

`samples.jsonl` stores qualitative generations on fixed example prompts from local images in `assets/`.

#### `logs/`

Every process also writes a plain `.log` file under:

```text
logs/
```

For example:

- `logs/train_<timestamp>.log`
- `logs/generate_<timestamp>.log`
- `logs/evaluate_<timestamp>.log`

These logs capture stdout/stderr so you can inspect exactly what happened during training, generation, or evaluation without relying only on terminal scrollback.

### Progress Bar Fields

The training progress bar now reports:

- `step`
- `batch_loss`
- `toks`

Example:

```text
step=120 batch_loss=1.84 toks=3.5M
```

Token formatting uses compact suffixes:

- `K`
- `M`
- `B`

### Seen Tokens vs Target Tokens

This repo distinguishes between:

- `seen tokens`: all non-masked tokens in the batch according to `attention_mask`
- `target tokens`: tokens supervised by the loss, typically where `labels != -100`

This matters because a VLM sees more context than it is always directly trained to predict. The cumulative `consumed_tokens` metric is based on the visible training tokens, not just the supervised target positions.

### Fixed Sample Generation During Training

At each stats logging interval, the training loop runs qualitative generation on five fixed local examples defined in:

- [`utils/generation_helper.py`](./utils/generation_helper.py)

These samples use images from `assets/` and prompts such as:

- describe the image
- identify text in the image
- answer a relationship question
- count players
- identify shirt color

The outputs are:

- printed to the training log
- appended to `samples.jsonl`

This gives you a lightweight qualitative signal during training without needing a separate notebook or dashboard.

## Generation

Generate from a checkpoint with:

```bash
python generate.py \
  --checkpoint /path/to/checkpoint \
  --image assets/cat.png \
  --prompt "Describe the image."
```

Useful flags:

- `--generations`
- `--max_new_tokens`
- `--top_k`
- `--top_p`
- `--temperature`
- `--greedy`

Example:

```bash
python generate.py \
  --checkpoint lusxvr/nanoVLM-230M-8k \
  --image assets/cat.png \
  --prompt "What is in the image?" \
  --generations 3 \
  --max_new_tokens 64
```

Generation also writes a process log to `logs/`.

## Evaluation

This repo supports benchmark evaluation through `lmms-eval`.

Run evaluation with:

```bash
python evaluate.py \
  --checkpoint /path/to/checkpoint \
  --tasks mmstar \
  --batch_size 8
```

Example with multiple tasks:

```bash
python evaluate.py \
  --checkpoint lusxvr/nanoVLM-230M-8k \
  --tasks mmstar,mme \
  --batch_size 8 \
  --device cuda
```

The evaluation script:

- wraps the model with [`utils/eval_wrapper.py`](./utils/eval_wrapper.py)
- passes the model into `lmms-eval`
- optionally saves JSON outputs and samples
- writes a process log under `logs/`

If you want saved raw results:

```bash
python evaluate.py \
  --checkpoint lusxvr/nanoVLM-230M-8k \
  --tasks mmstar \
  --batch_size 8 \
  --output_path eval_results/mmstar.json
```

## Saving and Loading Models

The model class supports local and Hub-style save/load patterns.

Load from a Hub repo or local folder:

```python
from models.nanovlm import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("lusxvr/nanoVLM-230M-8k")
```

Save locally:

```python
model.save_pretrained("path/to/model")
```

This repo uses Hugging Face-style `save_pretrained` and `from_pretrained` flows to keep checkpoint handling simple and portable.

## Validation Status

The current repo has been validated in its own local `uv` environment.

Verified in the repo-local `.venv`:

- imports for `torch`, `transformers`, `datasets`, `yaml`, `matplotlib`, and `lmms_eval`
- unit tests for training helpers
- unit tests for evaluation
- unit tests for generation helper logic
- unit tests for logging paths

This means the repo-local environment is functional for:

- training codepaths
- generation codepaths
- evaluation integration
- process logging

What this does not automatically mean:

- a full long-running production training job has already been executed in this README workflow
- a full benchmark sweep has been run on remote checkpoints and datasets

Those are longer runtime validations, not documentation/runtime bootstrapping checks.

## Design Goals

This repo is opinionated.

What it optimizes for:

- low-friction local iteration
- minimal mental overhead
- readability over framework magic
- explicit logging
- direct debuggability

What it does not optimize for:

- every possible VLM architecture variant
- every distributed training strategy
- highly abstract multi-repo reuse
- feature completeness for unrelated experiments

If you want a wider experimental surface, use the larger workspace repos. If you want a clean baseline you can read and edit quickly, use this one.

## Acknowledgement

`picoVLM` is heavily inspired by the repository design, training ergonomics, and educational clarity of `nanoVLM` from Hugging Face:

- https://github.com/huggingface/nanoVLM

This fork keeps that same general energy:

- directness
- small codepaths
- practical scripts
- educational readability

while trimming the repo down to the essentials needed here.

## Citation

If you use `picoVLM`, please also acknowledge the upstream `nanoVLM` project that inspired this fork. Their repository cites the project as:

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

If you want to cite this fork separately, add your own project-level citation alongside the upstream one.
