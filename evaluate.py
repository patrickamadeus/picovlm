import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from models.stackvlm import StackVLM
from utils.eval_wrapper import NanoVLMWrapper
from utils.train_helper import create_process_log_path, tee_run_log


def parse_args():
    parser = argparse.ArgumentParser(description="Run lmms-eval for NanoVLM")
    parser.add_argument("--config", type=str, default=None, help="optional yaml file for evaluation args")
    parser.add_argument("--checkpoint", type=str, required=False, default=None, help="local checkpoint dir or HF repo id")
    parser.add_argument("--tasks", type=str, required=False, default=None, help="comma-separated lmms-eval tasks")
    parser.add_argument("--batch_size", type=int, default=None, help="batch size per gpu")
    parser.add_argument("--device", type=str, default=None, help="device string")
    parser.add_argument("--limit", type=float, default=None, help="example limit")
    parser.add_argument("--num_fewshot", type=int, default=None, help="fewshot examples")
    parser.add_argument("--output_path", type=str, default=None, help="json file to save raw results")
    parser.add_argument("--gen_kwargs", type=str, default=None, help="lmms-eval generation kwargs string")
    parser.add_argument("--write_out", action="store_true", help="print the prompt for the first few documents via lmms-eval")
    parser.add_argument("--log_samples", action="store_true", help="save sample outputs if lmms-eval supports it")
    parser.add_argument("--verbosity", type=str, default=None, help="lmms-eval verbosity")
    return parser.parse_args()


def load_yaml_args(path: str | None):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded.get("evaluation", loaded)


def _ensure_output_paths(output_path: str | None):
    if output_path is None:
        return None, None
    output = Path(output_path)
    if output.suffix.lower() == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        sample_dir = output.parent / f"{output.stem}_samples"
    else:
        output.mkdir(parents=True, exist_ok=True)
        output = output / "results.json"
        sample_dir = output.parent / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return output, sample_dir


def _save_outputs(output_path: str | None, results: dict, samples: dict | None):
    output_file, sample_dir = _ensure_output_paths(output_path)
    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
    if sample_dir is not None and samples:
        for task_name, task_samples in samples.items():
            with open(sample_dir / f"{task_name}.json", "w", encoding="utf-8") as f:
                json.dump(task_samples, f, indent=2, default=str)


def cli_evaluate(args=None):
    cfg_args = parse_args() if args is None else args
    repo_dir = Path(__file__).resolve().parent
    log_path = create_process_log_path(repo_dir, "evaluate")
    with tee_run_log(log_path):
        print(f"[log] file={log_path}")
        yaml_args = load_yaml_args(getattr(cfg_args, "config", None))

        checkpoint = cfg_args.checkpoint or yaml_args.get("checkpoint")
        tasks = cfg_args.tasks or yaml_args.get("tasks")
        if not checkpoint or not tasks:
            raise ValueError("--checkpoint and --tasks are required unless provided by --config")
        batch_size = int(cfg_args.batch_size or yaml_args.get("batch_size", 8))
        device = cfg_args.device or yaml_args.get("device", "cuda")
        limit = cfg_args.limit if cfg_args.limit is not None else yaml_args.get("limit")
        num_fewshot = cfg_args.num_fewshot if cfg_args.num_fewshot is not None else yaml_args.get("num_fewshot")
        output_path = cfg_args.output_path or yaml_args.get("output_path")
        gen_kwargs = cfg_args.gen_kwargs if cfg_args.gen_kwargs is not None else yaml_args.get("gen_kwargs", "")
        write_out = bool(getattr(cfg_args, "write_out", False) or yaml_args.get("write_out", False))
        log_samples = bool(cfg_args.log_samples or yaml_args.get("log_samples", False))
        verbosity = cfg_args.verbosity or yaml_args.get("verbosity", "INFO")

        try:
            from lmms_eval import evaluator, utils
            from lmms_eval.tasks import TaskManager
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise ImportError("lmms_eval is required to run evaluate.py") from exc

        task_manager = TaskManager()
        task_names = task_manager.match_tasks(tasks.split(","))
        model = StackVLM.from_pretrained(checkpoint)
        wrapped_model = NanoVLMWrapper(model=model, device=device, batch_size=batch_size)
        cli_args = SimpleNamespace(
            output_path=output_path,
            process_with_media=False,
        )

        results = evaluator.simple_evaluate(
            model=wrapped_model,
            model_args="",
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            write_out=write_out,
            log_samples=log_samples,
            gen_kwargs=gen_kwargs,
            task_manager=task_manager,
            verbosity=verbosity,
            cli_args=cli_args,
        )
        samples = results.pop("samples") if results and log_samples and "samples" in results else None
        _save_outputs(output_path, results, samples)

        if results and "results" in results:
            print(utils.make_table(results))
        return results


if __name__ == "__main__":
    cli_evaluate()
