import argparse
import contextlib
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from configs.config import TrainConfig, load_train_config, load_vlm_config
from models.stackvlm import StackVLM
from utils.datasets import build_dataloaders
from utils.distributed import (
    barrier,
    destroy_distributed,
    get_device,
    get_rank,
    get_world_size,
    gather_objects,
    init_distributed,
    is_distributed,
    is_master,
    sum_scalar,
    wrap_model,
)
from utils.generation_helper import generate_fixed_samples
from utils.train_helper import (
    _set_lr,
    _valid_batch,
    advance_train_iterator,
    append_jsonl,
    cast_optimizer_state,
    capture_rng_state,
    count_batch_tokens,
    colorize_text,
    create_process_log_path,
    create_run_dir,
    load_training_state,
    estimate_val_loss,
    format_train_postfix,
    push_checkpoint_to_hub,
    resolve_train_dtype,
    resolve_stop_after_step,
    restore_rng_state,
    save_checkpoint,
    save_plot,
    should_log_train_outputs,
    tee_run_log,
)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="nanovlm", help="YAML config name under configs/yaml, without extension")
    return parser.parse_args()


def _load_yaml_config(config_name: str):
    config_path = Path(__file__).resolve().parent / "configs" / "yaml" / f"{config_name}.yml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config '{config_name}' not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return config_path, yaml.safe_load(f) or {}


def _apply_freeze_flags(model, train_cfg):
    modules = (
        (model.vision_encoder, train_cfg.lr_vision_backbone <= 0),
        (model.MP, train_cfg.lr_mp <= 0),
        (model.decoder, train_cfg.lr_language_backbone <= 0),
        (getattr(model, "full_decoder", None), getattr(train_cfg, "lr_full_decoder", train_cfg.lr_language_backbone) <= 0),
    )
    for module, freeze in modules:
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = not freeze


def initialize_model_for_training(model_cfg, train_cfg, *, device: torch.device, train_dtype: torch.dtype):
    if train_cfg.resume_checkpoint_path:
        model = StackVLM.from_pretrained(train_cfg.resume_checkpoint_path)
    elif train_cfg.resume_from_vlm_checkpoint:
        if not model_cfg.vlm_checkpoint_path:
            raise ValueError("resume_from_vlm_checkpoint=True requires vlm.vlm_checkpoint_path")
        model = StackVLM.from_pretrained(model_cfg.vlm_checkpoint_path)
    else:
        model = StackVLM(model_cfg, load_backbone=model_cfg.vlm_load_backbone_weights)
    _apply_freeze_flags(model, train_cfg)
    return model.to(device=device, dtype=train_dtype)


def build_optimizer_groups(model, train_cfg):
    groups = []
    for name, module, lr in (
        ("mp", model.MP, train_cfg.lr_mp),
        ("vision", model.vision_encoder, train_cfg.lr_vision_backbone),
        ("decoder", model.decoder, train_cfg.lr_language_backbone),
        ("full_decoder", getattr(model, "full_decoder", None), getattr(train_cfg, "lr_full_decoder", train_cfg.lr_language_backbone)),
    ):
        if module is None:
            continue
        params = [param for param in module.parameters() if param.requires_grad]
        if params and lr > 0:
            groups.append({"params": params, "lr": lr, "max_lr": lr, "name": name})
    if not groups:
        raise ValueError("no trainable parameter groups found; set a positive learning rate for at least one NanoVLM module")
    return groups


def build_model_forward_kwargs(batch: dict, device: torch.device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "images": batch["images"],
        "attention_mask": batch["attention_mask"].to(device),
        "targets": None,
    }


def project_output_logits(raw_model, hidden_states):
    if hasattr(raw_model, "output_logits"):
        return raw_model.output_logits(hidden_states)
    return raw_model.decoder.head(hidden_states) if not raw_model.decoder.lm_use_tokens else hidden_states


def maybe_log_train_samples(*, raw_model, device, base_dir, sample_log_path, step):
    was_training = raw_model.training
    raw_model.eval()
    try:
        sample_outputs = generate_fixed_samples(
            raw_model,
            device=device,
            base_dir=base_dir,
        )
    finally:
        raw_model.train(was_training)

    print(colorize_text(f"[samples] step={step}", "magenta", attrs=["bold"]))
    for sample in sample_outputs:
        print(colorize_text(f"  [{sample['name']}] {sample['prompt']}", "magenta"))
        for idx, generation in enumerate(sample["generations"], start=1):
            print(colorize_text(f"    >> generation {idx}: {generation}", "white"))
        append_jsonl(
            sample_log_path,
            {
                "kind": "sample",
                "step": int(step),
                "name": sample["name"],
                "image": sample["image"],
                "prompt": sample["prompt"],
                "generations": sample["generations"],
            },
        )


def main():
    init_distributed()
    try:
        args = _parse_args()
        config_path, cfg = _load_yaml_config(args.config)
        model_cfg = load_vlm_config(cfg)
        train_cfg = load_train_config(cfg)
        repo_dir = Path(__file__).resolve().parent
        run_dir = create_run_dir(Path(__file__).resolve().parent / "results") if is_master() else None
        plot_path = run_dir / "loss.png" if run_dir is not None else None
        loss_log_path = run_dir / "loss.jsonl" if run_dir is not None else None
        sample_log_path = run_dir / "samples.jsonl" if run_dir is not None else None
        process_log_path = create_process_log_path(repo_dir, "train") if is_master() else None
        run_log_ctx = tee_run_log(process_log_path) if process_log_path is not None else contextlib.nullcontext()
        with run_log_ctx:
            if process_log_path is not None:
                print(colorize_text(f"[log] file={process_log_path}", "cyan"))
            if run_dir is not None:
                with open(run_dir / "resolved_config.yml", "w", encoding="utf-8") as f:
                    yaml.safe_dump(
                        {
                            "source_config": config_path.name,
                            "vlm": model_cfg.__dict__,
                            "train": train_cfg.__dict__,
                        },
                        f,
                        sort_keys=False,
                    )

            _, train_loader, val_loader = build_dataloaders(train_cfg, model_cfg)
            device = get_device()
            train_dtype = resolve_train_dtype(train_cfg, device)

            raw_model = initialize_model_for_training(
                model_cfg,
                train_cfg,
                device=device,
                train_dtype=train_dtype,
            )

            groups = build_optimizer_groups(raw_model, train_cfg)
            optimizer = torch.optim.AdamW(groups)
            model = wrap_model(raw_model)

            train_steps, train_losses, val_steps, val_losses = [], [], [], []
            start_step = 1
            total_batches_seen = 0
            consumed_tokens = 0
            if train_cfg.resume_checkpoint_path:
                state = load_training_state(train_cfg.resume_checkpoint_path, map_location="cpu")
                optimizer.load_state_dict(state["optimizer"])
                cast_optimizer_state(optimizer, device=device, dtype=train_dtype)
                rng_state = state.get("rng_state")
                if state.get("rng_state_by_rank") is not None:
                    rng_state = state["rng_state_by_rank"][get_rank()]
                restore_rng_state(rng_state)
                start_step = int(state.get("step", 0)) + 1
                total_batches_seen = int(state.get("total_batches_seen", 0))
                consumed_tokens = int(state.get("consumed_tokens", 0))
                train_steps = list(state.get("train_steps", []))
                train_losses = list(state.get("train_losses", []))
                val_steps = list(state.get("val_steps", []))
                val_losses = list(state.get("val_losses", []))
                if is_master():
                    print(colorize_text(f"[resume] step={start_step} checkpoint={train_cfg.resume_checkpoint_path}", "cyan"))
                    append_jsonl(loss_log_path, {"kind": "resume", "checkpoint": train_cfg.resume_checkpoint_path, "step": start_step})
            else:
                val0 = estimate_val_loss(model, val_loader, device, max_batches=train_cfg.max_val_batches) if train_cfg.do_eval else None
                if val0 is not None and is_master():
                    val_steps.append(0)
                    val_losses.append(val0)
                    print(colorize_text(f"[val] step=0 loss={val0:.2f}", "yellow"))
                    append_jsonl(loss_log_path, {"kind": "val", "step": 0, "loss": val0})
            train_iter = advance_train_iterator(train_loader, total_batches_seen)
            optimizer.zero_grad(set_to_none=True)
            effective_stop_step = resolve_stop_after_step(train_cfg)

            step_iter = range(start_step, effective_stop_step + 1)
            if is_master():
                step_iter = tqdm(step_iter, desc="train", dynamic_ncols=True)
            for step in step_iter:
                _set_lr(optimizer, step, train_cfg)
                loss_sum, target_token_sum, seen_token_sum = 0.0, 0, 0
                for micro_step in range(train_cfg.gradient_accumulation_steps):
                    while True:
                        try:
                            batch = next(train_iter)
                        except StopIteration:
                            train_iter = iter(train_loader)
                            batch = next(train_iter)
                        total_batches_seen += 1
                        if _valid_batch(batch):
                            break
                    sync_context = model.no_sync if is_distributed() and micro_step < train_cfg.gradient_accumulation_steps - 1 else contextlib.nullcontext
                    with sync_context():
                        logits, _ = model(**build_model_forward_kwargs(batch, device))
                        logits = project_output_logits(raw_model, logits)
                        labels = batch["labels"].to(device)
                        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
                        target_tokens = int((labels != -100).sum().item())
                        seen_tokens = count_batch_tokens(batch)
                        if target_tokens == 0:
                            continue
                        loss.backward()
                        loss_sum += float(loss.item())
                        target_token_sum += target_tokens
                        seen_token_sum += seen_tokens
                global_target_token_sum = int(sum_scalar(target_token_sum, device))
                global_seen_token_sum = int(sum_scalar(seen_token_sum, device))
                global_loss_sum = float(sum_scalar(loss_sum, device))
                if global_target_token_sum == 0:
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scale = get_world_size() / global_target_token_sum
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(scale)

                if train_cfg.max_grad_norm and train_cfg.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_loss = global_loss_sum / global_target_token_sum
                consumed_tokens += global_seen_token_sum
                should_log_outputs = should_log_train_outputs(
                    step=step,
                    stats_log_interval=train_cfg.stats_log_interval,
                    effective_stop_step=effective_stop_step,
                )
                if is_master():
                    step_iter.set_postfix_str(
                        format_train_postfix(
                            step=step,
                            batch_loss=train_loss,
                            consumed_tokens=consumed_tokens,
                        )
                    )
                    train_steps.append(step)
                    train_losses.append(train_loss)
                    append_jsonl(
                        loss_log_path,
                        {
                            "kind": "train",
                            "step": step,
                            "loss": train_loss,
                            "batch_loss": train_loss,
                            "tokens": global_seen_token_sum,
                            "target_tokens": global_target_token_sum,
                            "consumed_tokens": consumed_tokens,
                            "log_interval": should_log_outputs,
                        },
                    )

                if step % train_cfg.eval_interval == 0:
                    val_loss = estimate_val_loss(model, val_loader, device, max_batches=train_cfg.max_val_batches) if train_cfg.do_eval else None
                    if val_loss is not None and is_master():
                        val_steps.append(step)
                        val_losses.append(val_loss)
                        print(colorize_text(f"[val] step={step} loss={val_loss:.2f}", "yellow"))
                        append_jsonl(loss_log_path, {"kind": "val", "step": step, "loss": val_loss})
                    checkpoint_rng_state = capture_rng_state()
                    checkpoint_rng_state_by_rank = gather_objects(checkpoint_rng_state)
                    if train_cfg.push_checkpoints_to_hub and is_master():
                        repo_id = train_cfg.checkpoint_repo_pattern.format(i=step, step=step)
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            ckpt_path = Path(tmp_dir) / f"step_{step}"
                            save_checkpoint(
                                ckpt_path,
                                model,
                                optimizer,
                                step,
                                train_cfg,
                                training_state={
                                    "rng_state": checkpoint_rng_state_by_rank[0],
                                    "rng_state_by_rank": checkpoint_rng_state_by_rank,
                                    "world_size": get_world_size(),
                                    "total_batches_seen": total_batches_seen,
                                    "consumed_tokens": consumed_tokens,
                                    "train_steps": train_steps,
                                    "train_losses": train_losses,
                                    "val_steps": val_steps,
                                    "val_losses": val_losses,
                                },
                                include_training_state=train_cfg.save_training_state_to_hub,
                            )
                            push_checkpoint_to_hub(
                                ckpt_path,
                                repo_id,
                                private=train_cfg.hf_private,
                                commit_message=f"Upload step {step} checkpoint",
                                include_training_state=train_cfg.save_training_state_to_hub,
                            )
                        print(colorize_text(f"[checkpoint] step={step} repo={repo_id}", "green"))
                        append_jsonl(loss_log_path, {"kind": "checkpoint", "step": step, "repo_id": repo_id})
                    barrier()
                
                if is_master() and should_log_outputs:
                    save_plot(plot_path, train_steps, train_losses, val_steps, val_losses)
                    maybe_log_train_samples(
                        raw_model=raw_model,
                        device=device,
                        base_dir=Path(__file__).resolve().parent,
                        sample_log_path=sample_log_path,
                        step=step,
                    )

            final_rng_state = capture_rng_state()
            final_rng_state_by_rank = gather_objects(final_rng_state)
            if train_cfg.push_final_model_to_hub and is_master():
                repo_id = train_cfg.checkpoint_repo_pattern.format(i="final", step=effective_stop_step)
                with tempfile.TemporaryDirectory() as tmp_dir:
                    final_path = Path(tmp_dir) / "final"
                    save_checkpoint(
                        final_path,
                        model,
                        optimizer,
                        effective_stop_step,
                        train_cfg,
                        training_state={
                            "rng_state": final_rng_state_by_rank[0],
                            "rng_state_by_rank": final_rng_state_by_rank,
                            "world_size": get_world_size(),
                            "total_batches_seen": total_batches_seen,
                            "consumed_tokens": consumed_tokens,
                            "train_steps": train_steps,
                            "train_losses": train_losses,
                            "val_steps": val_steps,
                            "val_losses": val_losses,
                        },
                        include_training_state=train_cfg.save_training_state_to_hub,
                    )
                    push_checkpoint_to_hub(
                        final_path,
                        repo_id,
                        private=train_cfg.hf_private,
                        commit_message="Upload final checkpoint",
                        include_training_state=train_cfg.save_training_state_to_hub,
                    )
                print(colorize_text(f"[final] repo={repo_id}", "green", attrs=["bold"]))
                append_jsonl(loss_log_path, {"kind": "checkpoint", "step": effective_stop_step, "repo_id": repo_id})
            barrier()

            if is_master():
                save_plot(plot_path, train_steps, train_losses, val_steps, val_losses)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
