import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils.generation_helper import generate_fixed_samples, resolve_train_sample_specs


class _FakeTokenizer:
    def batch_decode(self, generated, skip_special_tokens=True):
        del skip_special_tokens
        return [f"decoded-{int(generated.reshape(-1)[0].item())}"]


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return torch.tensor([[len(self.calls)]], dtype=torch.long)


class TestGenerationHelper(unittest.TestCase):
    def test_resolve_train_sample_specs_has_expected_assets_and_prompts(self):
        specs = resolve_train_sample_specs(Path("/repo"))
        self.assertEqual(len(specs), 5)
        self.assertTrue(specs[0]["image"].endswith("assets/cat.png"))
        self.assertEqual(specs[0]["prompt"], "Describe the image.")
        self.assertEqual(specs[-1]["prompt"], "What is the color of their shirt?")

    def test_generate_fixed_samples_uses_nanovlm_signature_only(self):
        tokenizer = _FakeTokenizer()
        model = _FakeModel()

        def fake_inputs(_model, image_path, prompt, device):
            del _model, image_path, prompt, device
            return tokenizer, torch.tensor([[1, 7, 7]], dtype=torch.long), torch.ones(1, 3, dtype=torch.long), [torch.zeros(1, 3, 4, 4)]

        with patch("utils.generation_helper.build_generation_inputs", side_effect=fake_inputs):
            outputs = generate_fixed_samples(model, device=torch.device("cpu"), base_dir=Path("/repo"))

        self.assertEqual(len(outputs), 5)
        self.assertEqual(len(model.calls), 5)
        self.assertTrue(all("last_img_idx" not in call for call in model.calls))
        self.assertEqual(outputs[0]["generations"], ["decoded-1"])
        self.assertEqual(outputs[-1]["generations"], ["decoded-5"])
