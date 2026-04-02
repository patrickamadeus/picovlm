import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluate import _ensure_output_paths, _save_outputs, cli_evaluate


class TestEvaluateOutputs(unittest.TestCase):
    def test_ensure_output_paths_supports_json_file_and_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_output, file_samples = _ensure_output_paths(str(Path(tmp_dir) / "results.json"))
            self.assertEqual(file_output.name, "results.json")
            self.assertEqual(file_samples.name, "results_samples")

            dir_output, dir_samples = _ensure_output_paths(str(Path(tmp_dir) / "eval_dir"))
            self.assertEqual(dir_output.name, "results.json")
            self.assertEqual(dir_samples.name, "samples")

    def test_save_outputs_writes_aggregate_and_sample_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = str(Path(tmp_dir) / "aggregate.json")
            results = {"results": {"mmstar": {"score": 1.0}}}
            samples = {"mmstar": [{"doc_id": 1, "response": "A"}]}
            _save_outputs(output_path, results, samples)

            with open(output_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            self.assertEqual(saved, results)

            sample_file = Path(tmp_dir) / "aggregate_samples" / "mmstar.json"
            self.assertTrue(sample_file.exists())
            with open(sample_file, "r", encoding="utf-8") as f:
                saved_samples = json.load(f)
            self.assertEqual(saved_samples, samples["mmstar"])

    def test_cli_evaluate_passes_cli_args_with_required_fields_to_lmms_eval(self):
        fake_args = SimpleNamespace(
            config=None,
            checkpoint="repo-or-path",
            tasks="coco_cap",
            batch_size=1,
            device="cpu",
            limit=1,
            num_fewshot=0,
            output_path="eval_results/coco.json",
            gen_kwargs="",
            write_out=False,
            log_samples=False,
            verbosity="INFO",
        )
        recorded = {}

        def fake_simple_evaluate(**kwargs):
            recorded.update(kwargs)
            return {"results": {"coco_cap": {"Bleu_4": 0.0}}}

        fake_evaluator = types.SimpleNamespace(simple_evaluate=fake_simple_evaluate)
        fake_utils = types.SimpleNamespace(make_table=lambda results: "table")
        fake_task_manager_module = types.SimpleNamespace(TaskManager=lambda: types.SimpleNamespace(match_tasks=lambda tasks: tasks))
        fake_lmms_eval = types.SimpleNamespace(evaluator=fake_evaluator, utils=fake_utils)

        with patch.dict(
            sys.modules,
            {
                "lmms_eval": fake_lmms_eval,
                "lmms_eval.tasks": fake_task_manager_module,
            },
        ), patch("evaluate.NanoVLMWrapper", return_value="wrapped-model"), patch("builtins.print"):
            cli_evaluate(fake_args)

        self.assertIn("cli_args", recorded)
        self.assertEqual(recorded["cli_args"].output_path, "eval_results/coco.json")
        self.assertFalse(recorded["cli_args"].process_with_media)
        self.assertEqual(recorded["model"], "wrapped-model")

    def test_cli_evaluate_wraps_execution_in_logs_file(self):
        fake_args = SimpleNamespace(
            config=None,
            checkpoint="repo-or-path",
            tasks="coco_cap",
            batch_size=1,
            device="cpu",
            limit=1,
            num_fewshot=0,
            output_path=None,
            gen_kwargs="",
            write_out=False,
            log_samples=False,
            verbosity="INFO",
        )

        fake_evaluator = types.SimpleNamespace(simple_evaluate=lambda **kwargs: {"results": {"coco_cap": {"Bleu_4": 0.0}}})
        fake_utils = types.SimpleNamespace(make_table=lambda results: "table")
        fake_task_manager_module = types.SimpleNamespace(TaskManager=lambda: types.SimpleNamespace(match_tasks=lambda tasks: tasks))
        fake_lmms_eval = types.SimpleNamespace(evaluator=fake_evaluator, utils=fake_utils)

        with patch.dict(
            sys.modules,
            {
                "lmms_eval": fake_lmms_eval,
                "lmms_eval.tasks": fake_task_manager_module,
            },
        ), patch("evaluate.NanoVLMWrapper", return_value="wrapped-model"), patch(
            "evaluate.create_process_log_path",
            return_value=Path("/repo/logs/evaluate_20260402_181500.log"),
        ) as mock_log_path, patch("evaluate.tee_run_log") as mock_tee, patch("builtins.print"):
            mock_tee.return_value.__enter__ = lambda self=None: None
            mock_tee.return_value.__exit__ = lambda exc_type, exc, tb, self=None: False
            cli_evaluate(fake_args)

        mock_log_path.assert_called_once()
        mock_tee.assert_called_once_with(Path("/repo/logs/evaluate_20260402_181500.log"))
