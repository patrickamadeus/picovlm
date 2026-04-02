from pathlib import Path
from datetime import datetime
import contextlib
import json
import os
import random
import shutil
import sys
import tempfile
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

try:
    from termcolor import colored as _termcolor_colored
except ImportError:
    _termcolor_colored = None

from utils.distributed import sum_scalar, unwrap_model


def create_run_dir(base_dir="results", tz_name="Asia/Bangkok"):
    timestamp = datetime.now(ZoneInfo(tz_name)).strftime("run_%d-%m-%y_%H-%M-%S")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_process_log_path(base_dir, process_name, *, timestamp=None, tz_name="Asia/Bangkok"):
    base_dir = Path(base_dir)
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now(ZoneInfo(tz_name)).strftime("%Y%m%d_%H%M%S")
    return log_dir / f"{process_name}_{stamp}.log"


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextlib.contextmanager
def tee_run_log(log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with open(log_path, "a", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def colorize_text(text, color=None, *, attrs=None):
    if _termcolor_colored is None:
        return str(text)
    return _termcolor_colored(str(text), color=color, attrs=attrs)


def format_consumed_tokens(count):
    count = int(count)
    for suffix, scale in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if abs(count) >= scale:
            return f"{count / scale:.1f}{suffix}"
    return f"{count:,}"


def format_train_postfix(*, step, batch_loss, consumed_tokens):
    return f"step={int(step)} batch_loss={float(batch_loss):.2f} toks={format_consumed_tokens(consumed_tokens)}"


def should_log_train_outputs(*, step, stats_log_interval, effective_stop_step):
    interval = max(1, int(stats_log_interval))
    step = int(step)
    return bool(step == 1 or step % interval == 0 or step == int(effective_stop_step))


def count_batch_tokens(batch):
    if batch is None:
        return 0
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        return int(attention_mask.sum().item())
    input_ids = batch.get("input_ids")
    if input_ids is not None:
        return int(input_ids.numel())
    return 0


def resolve_stop_after_step(train_cfg):
    stop_after_step = getattr(train_cfg, "stop_after_step", None)
    if stop_after_step is None:
        return int(train_cfg.max_training_steps)
    return min(int(stop_after_step), int(train_cfg.max_training_steps))


def capture_rng_state():
    state = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not state:
        return
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("numpy_random_state") is not None:
        np.random.set_state(state["numpy_random_state"])
    if state.get("torch_cpu_rng_state") is not None:
        torch.set_rng_state(state["torch_cpu_rng_state"])
    if torch.cuda.is_available() and state.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state_all"])


def load_training_state(checkpoint_path_or_repo, *, revision=None, map_location="cpu"):
    if os.path.exists(checkpoint_path_or_repo):
        state_path = Path(checkpoint_path_or_repo) / "training_state.pt"
    else:
        from huggingface_hub import hf_hub_download

        state_path = hf_hub_download(repo_id=checkpoint_path_or_repo, filename="training_state.pt", revision=revision)
    if not Path(state_path).exists():
        raise FileNotFoundError(f"training_state.pt not found at {state_path}")
    return torch.load(state_path, map_location=map_location, weights_only=False)


def advance_train_iterator(loader, total_batches_seen):
    train_iter = iter(loader)
    if len(loader) == 0:
        return train_iter
    for _ in range(int(total_batches_seen) % len(loader)):
        try:
            next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
    return train_iter


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def cast_optimizer_state(optimizer, *, device, dtype=None):
    for state in optimizer.state.values():
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            target = value.to(device)
            if dtype is not None and torch.is_floating_point(target):
                target = target.to(dtype=dtype)
            state[key] = target


def resolve_train_dtype(train_cfg, device):
    precision = str(getattr(train_cfg, "precision", "fp32")).lower()
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training is only supported on cuda devices")
        return torch.float16
    if precision == "bf16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 training is not supported on this cuda device")
        return torch.bfloat16
    raise ValueError(f"unsupported precision: {train_cfg.precision}")


def _valid_batch(batch):
    return isinstance(batch, dict) and "input_ids" in batch and batch["input_ids"].numel() > 0


def _set_lr(optimizer, step, train_cfg):
    warmup_steps = max(1, int(train_cfg.max_training_steps * train_cfg.warmup_ratio))
    for group in optimizer.param_groups:
        peak = group["max_lr"]
        group["lr"] = peak * min(1.0, step / warmup_steps) if step <= warmup_steps else peak


@torch.inference_mode()
def estimate_val_loss(model, loader, device, max_batches=32):
    raw_model = unwrap_model(model)
    raw_model.eval()
    total_loss, total_tokens = 0.0, 0
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        if not _valid_batch(batch):
            continue
        logits, _ = model(
            input_ids=batch["input_ids"].to(device),
            images=batch["images"],
            attention_mask=batch["attention_mask"].to(device),
            targets=None,
        )
        logits = raw_model.decoder.head(logits) if not raw_model.decoder.lm_use_tokens else logits
        labels = batch["labels"].to(device)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
        total_loss += float(loss.item())
        total_tokens += int((labels != -100).sum().item())
    total_loss = sum_scalar(total_loss, device)
    total_tokens = int(sum_scalar(total_tokens, device))
    raw_model.train()
    return None if total_tokens == 0 else total_loss / total_tokens


def save_plot(path, train_steps, train_losses, val_steps, val_losses):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_losses, label="train")
    if val_steps:
        ax.plot(val_steps, val_losses, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(path, model, optimizer, step, train_cfg=None, training_state=None, include_training_state=True):
    path.mkdir(parents=True, exist_ok=True)
    unwrap_model(model).save_pretrained(str(path))
    if train_cfg is not None:
        with open(path / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(train_cfg.__dict__, f, indent=2)
    if include_training_state:
        state = {"optimizer": optimizer.state_dict(), "step": int(step)}
        if train_cfg is not None:
            state["train_cfg"] = train_cfg.__dict__
        if training_state:
            state.update(training_state)
        torch.save(state, path / "training_state.pt")


def push_checkpoint_to_hub(path, repo_id, *, private=False, commit_message=None, include_training_state=True):
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id=repo_id, private=private, exist_ok=True)
    if include_training_state:
        upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(path),
            commit_message=commit_message or f"Upload checkpoint {Path(path).name}",
        )
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for src in Path(path).iterdir():
            if src.name == "training_state.pt":
                continue
            if src.is_dir():
                shutil.copytree(src, tmp_path / src.name)
            else:
                shutil.copy2(src, tmp_path / src.name)
        upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(tmp_path),
            commit_message=commit_message or f"Upload checkpoint {Path(path).name}",
        )
