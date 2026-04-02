import tempfile
import unittest
from pathlib import Path

from utils.train_helper import create_process_log_path, tee_run_log


class TestLoggingPaths(unittest.TestCase):
    def test_create_process_log_path_creates_gitignored_logs_dir_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = create_process_log_path(Path(tmp_dir), "evaluate", timestamp="20260402_181500")
            self.assertEqual(path, Path(tmp_dir) / "logs" / "evaluate_20260402_181500.log")

    def test_tee_run_log_writes_stream_output_to_log_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = Path(tmp_dir) / "logs" / "generate_20260402_181500.log"
            with tee_run_log(log_path):
                print("hello from generate")
            self.assertTrue(log_path.exists())
            self.assertIn("hello from generate", log_path.read_text(encoding="utf-8"))
