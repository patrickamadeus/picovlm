import unittest
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
from unittest.mock import patch

import torch

from configs.config import TrainConfig, VLMConfig, load_train_config, load_vlm_config
from models.nanovlm import VisionLanguageModel
from train import build_model_forward_kwargs, build_optimizer_groups, maybe_log_train_samples
from utils.datasets import VQACollator
from utils.processor_transforms import get_tokenizer
from utils.train_helper import (
    _valid_batch,
    count_batch_tokens,
    create_process_log_path,
    format_consumed_tokens,
    format_train_postfix,
    should_log_train_outputs,
)


class _FakeTokenizer:
    pad_token_id = 0


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_encoder = torch.nn.Linear(4, 4)
        self.MP = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 4)


class TestTrainUtilities(unittest.TestCase):
    @staticmethod
    def _resolve_local_snapshot(org, repo):
        hub_root = os.environ.get("HUGGINGFACE_HUB_CACHE") or (
            str(Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None
        )
        candidates = [
            Path.home() / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}" / "snapshots",
        ]
        if hub_root is not None:
            candidates.append(Path(hub_root) / f"models--{org}--{repo}" / "snapshots")
        for base in candidates:
            if base.exists():
                snapshots = sorted(p for p in base.iterdir() if p.is_dir())
                if snapshots:
                    return snapshots[-1]
        raise FileNotFoundError(f"Local snapshot for {org}/{repo} not found")

    def test_config_loaders_ignore_unknown_fields(self):
        payload = {
            "vlm": {"vit_img_size": 256, "unknown": 1},
            "train": {"batch_size": 2, "another_unknown": 7},
        }
        vlm_cfg = load_vlm_config(payload)
        train_cfg = load_train_config(payload)
        self.assertIsInstance(vlm_cfg, VLMConfig)
        self.assertIsInstance(train_cfg, TrainConfig)
        self.assertEqual(vlm_cfg.vit_img_size, 256)
        self.assertEqual(train_cfg.batch_size, 2)

    def test_build_optimizer_groups_uses_only_positive_learning_rates(self):
        model = _FakeModel()
        train_cfg = SimpleNamespace(lr_mp=1e-3, lr_vision_backbone=0.0, lr_language_backbone=2e-3)
        groups = build_optimizer_groups(model, train_cfg)
        self.assertEqual([group["name"] for group in groups], ["mp", "decoder"])

    def test_build_optimizer_groups_rejects_all_frozen_configuration(self):
        model = _FakeModel()
        train_cfg = SimpleNamespace(lr_mp=0.0, lr_vision_backbone=0.0, lr_language_backbone=0.0)
        with self.assertRaisesRegex(ValueError, "no trainable parameter groups"):
            build_optimizer_groups(model, train_cfg)

    def test_build_model_forward_kwargs_matches_nanovlm_inputs(self):
        batch = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "images": [torch.zeros(1, 3, 4, 4)],
        }
        kwargs = build_model_forward_kwargs(batch, torch.device("cpu"))
        self.assertEqual(set(kwargs), {"input_ids", "images", "attention_mask", "targets"})
        self.assertIsNone(kwargs["targets"])

    def test_valid_batch_no_longer_requires_legacy_split_metadata(self):
        self.assertTrue(_valid_batch({"input_ids": torch.ones(1, 2, dtype=torch.long)}))
        self.assertFalse(_valid_batch({"input_ids": torch.empty(0, 0, dtype=torch.long)}))

    def test_format_consumed_tokens_uses_k_m_b_suffixes(self):
        self.assertEqual(format_consumed_tokens(999), "999")
        self.assertEqual(format_consumed_tokens(12_300), "12.3K")
        self.assertEqual(format_consumed_tokens(4_500_000), "4.5M")
        self.assertEqual(format_consumed_tokens(6_700_000_000), "6.7B")

    def test_format_train_postfix_includes_step_batch_loss_and_tokens(self):
        self.assertEqual(
            format_train_postfix(step=42, batch_loss=1.2345, consumed_tokens=12_300),
            "step=42 batch_loss=1.23 toks=12.3K",
        )

    def test_count_batch_tokens_uses_attention_mask_not_supervision_mask(self):
        batch = {
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0]], dtype=torch.long),
            "labels": torch.tensor([[-100, -100, 7, 8, -100, -100]], dtype=torch.long),
        }
        self.assertEqual(count_batch_tokens(batch), 5)

    def test_should_log_train_outputs_fires_on_first_interval_and_last_step(self):
        self.assertTrue(should_log_train_outputs(step=1, stats_log_interval=100, effective_stop_step=250))
        self.assertTrue(should_log_train_outputs(step=100, stats_log_interval=100, effective_stop_step=250))
        self.assertTrue(should_log_train_outputs(step=250, stats_log_interval=100, effective_stop_step=250))
        self.assertFalse(should_log_train_outputs(step=101, stats_log_interval=100, effective_stop_step=250))

    def test_create_process_log_path_writes_under_logs_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_path = create_process_log_path(Path(tmp_dir), "train", timestamp="20260402_180000")
            self.assertEqual(log_path, Path(tmp_dir) / "logs" / "train_20260402_180000.log")

    def test_maybe_log_train_samples_uses_fixed_generation_specs(self):
        sample_log_records = []
        sample_outputs = [
            {"name": "image_1", "image": "/repo/assets/cat.png", "prompt": "Describe the image.", "generations": ["cat"]},
            {"name": "image_2", "image": "/repo/assets/clinic.png", "prompt": "What is the name of the clinic?", "generations": ["clinic"]},
            {"name": "image_3", "image": "/repo/assets/case.png", "prompt": "relationship", "generations": ["case"]},
            {"name": "image_4_count", "image": "/repo/assets/soccer.png", "prompt": "How many players are there?", "generations": ["11"]},
            {"name": "image_4_color", "image": "/repo/assets/soccer.png", "prompt": "What is the color of their shirt?", "generations": ["red"]},
        ]
        raw_model = torch.nn.Linear(1, 1)
        raw_model.train()

        with patch("train.generate_fixed_samples", return_value=sample_outputs) as mock_generate, patch(
            "train.append_jsonl", side_effect=lambda path, record: sample_log_records.append((path, record))
        ):
            maybe_log_train_samples(
                raw_model=raw_model,
                device=torch.device("cpu"),
                base_dir=Path("/repo"),
                sample_log_path=Path("/repo/results/samples.jsonl"),
                step=100,
            )

        mock_generate.assert_called_once()
        self.assertTrue(raw_model.training)
        self.assertEqual(len(sample_log_records), 5)
        self.assertEqual(sample_log_records[0][1]["step"], 100)
        self.assertEqual(sample_log_records[-1][1]["name"], "image_4_color")

    def test_vqa_collator_left_pads_without_last_img_idx(self):
        collator = VQACollator(_FakeTokenizer(), max_length=8)
        short = {
            "input_ids": torch.tensor([5], dtype=torch.long),
            "attention_mask": torch.tensor([1], dtype=torch.long),
            "labels": torch.tensor([7], dtype=torch.long),
            "images": [],
        }
        long = {
            "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([21, 22, 23], dtype=torch.long),
            "images": [],
        }

        batch = collator([short, long])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 3))
        self.assertEqual(batch["attention_mask"][0].tolist(), [0, 0, 1])
        self.assertEqual(batch["labels"][0].tolist(), [-100, -100, 7])
        self.assertNotIn("last_img_idx", batch)

    def test_nanovlm_end_to_end_finetune_step_runs_on_gpu1(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            self.skipTest("cuda:1 is not available")

        try:
            tokenizer_path = self._resolve_local_snapshot("HuggingFaceTB", "SmolLM2-135M-Instruct")
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

        cfg = VLMConfig(
            vit_hidden_dim=8,
            vit_inter_dim=16,
            vit_patch_size=2,
            vit_img_size=4,
            vit_n_heads=2,
            vit_dropout=0.0,
            vit_n_blocks=1,
            vit_cls_flag=False,
            lm_hidden_dim=8,
            lm_inter_dim=16,
            lm_rms_eps=1e-5,
            lm_re_base=10000,
            lm_max_position_embeddings=32,
            lm_base_vocab_size=50000,
            extra_token_amount=66,
            lm_vocab_size=50066,
            lm_n_heads=2,
            lm_n_kv_heads=1,
            lm_dropout=0.0,
            lm_n_blocks=1,
            lm_attn_scaling=1.0,
            lm_max_length=32,
            lm_use_tokens=False,
            lm_tie_weights=True,
            lm_model_type=str(tokenizer_path),
            lm_tokenizer=str(tokenizer_path),
            mp_pixel_shuffle_factor=1,
            mp_image_token_length=4,
            max_img_size=4,
            resize_to_max_side_len=False,
            vlm_load_backbone_weights=False,
            vlm_checkpoint_path=None,
        )
        tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template, model_max_length=cfg.lm_max_length)
        device = torch.device("cuda:1")
        model = VisionLanguageModel(cfg, load_backbone=False).to(device)
        optimizer = torch.optim.AdamW(build_optimizer_groups(model, TrainConfig(lr_mp=1e-3, lr_vision_backbone=1e-3, lr_language_backbone=1e-3)))

        image_token_id = tokenizer.image_token_id
        input_ids = torch.tensor(
            [[image_token_id, image_token_id, image_token_id, image_token_id, 5, 6, 7]],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor([[-100, -100, -100, -100, 6, 7, -100]], dtype=torch.long, device=device)
        images = [torch.arange(3 * 4 * 4, dtype=torch.float32, device=device).view(1, 3, 4, 4)]

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(input_ids=input_ids, images=images, attention_mask=attention_mask, targets=None)
        logits = model.decoder.head(logits) if not model.decoder.lm_use_tokens else logits
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )
        self.assertTrue(torch.isfinite(loss).item())

        tracked_param = model.decoder.token_embedding.weight.detach().clone()
        loss.backward()
        grad_norm = model.decoder.token_embedding.weight.grad.norm().item()
        self.assertGreater(grad_norm, 0.0)
        optimizer.step()

        self.assertFalse(torch.equal(tracked_param, model.decoder.token_embedding.weight.detach()))
