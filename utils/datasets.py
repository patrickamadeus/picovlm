import logging

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from utils.distributed import get_rank, get_world_size, is_distributed
from utils.image_utils import coerce_image_to_pil
from utils.processor_transforms import get_image_processor, get_image_string, get_tokenizer


def _normalize_source_name(value):
    return str(value).strip().lower()


def apply_runtime_source_filter(ds, train_cfg, dataset_name: str, split_name: str):
    if not bool(train_cfg.enable_source_filter):
        return ds
    if not train_cfg.allowed_dataset_sources:
        logging.warning("source_filter enabled but allowed_dataset_sources is empty; skipping source filter")
        return ds

    column_names = getattr(ds, "column_names", None)
    if column_names is not None and "source" not in column_names:
        logging.warning("%s:%s has no source column; skipping source filter", dataset_name, split_name)
        return ds

    allowed_sources = {_normalize_source_name(src) for src in train_cfg.allowed_dataset_sources}

    def _keep_example(example):
        source_value = example.get("source")
        if source_value is None:
            return False
        if isinstance(source_value, str):
            return _normalize_source_name(source_value) in allowed_sources
        if isinstance(source_value, list):
            return any(_normalize_source_name(item) in allowed_sources for item in source_value if item is not None)
        return _normalize_source_name(source_value) in allowed_sources

    if bool(train_cfg.stream_dataset):
        return ds.filter(_keep_example)

    return ds.filter(_keep_example, desc=f"filtering {dataset_name}:{split_name} by source")


class VQADataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length, train_cfg):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.max_images_per_example = train_cfg.max_images_per_example
        self.prefix_len = self._get_prefix_len()
        self.min_ratings = {
            "relevance_ratings": train_cfg.relevance_min_rating,
            "image_correspondence_ratings": train_cfg.image_correspondence_min_rating,
            "visual_dependency_ratings": train_cfg.visual_dependency_min_rating,
            "formatting_ratings": train_cfg.formatting_min_rating,
        }

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        probe = "xzyvd"
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": probe}],
            tokenize=False,
            add_special_tokens=False,
        )
        probe_idx = rendered.find(probe)
        return len(self.tokenizer.encode(rendered[:probe_idx]))

    def _keep_text(self, item, idx):
        for key, minimum in self.min_ratings.items():
            values = item.get(key)
            if values is not None and values[idx] is not None and values[idx] < minimum:
                return False
        return True

    def _process_images(self, images):
        processed_images, split_counts = [], []
        for image in images:
            processed, split_count = self.image_processor(coerce_image_to_pil(image))
            if not hasattr(self.tokenizer, "global_image_token") and split_count[0] * split_count[1] == len(processed) - 1:
                processed = processed[1:]
            processed_images.append(processed)
            split_counts.append(split_count)
        return processed_images, split_counts

    def _build_messages(self, item, split_counts):
        messages = []
        for idx, text in enumerate(item["texts"]):
            try:
                if not self._keep_text(item, idx):
                    continue
            except Exception as exc:
                logging.warning("rating filter failed on sample %s: %s", idx, exc)
            messages.append({"role": "user", "content": text["user"].replace(self.tokenizer.image_token, "")})
            messages.append({"role": "assistant", "content": text["assistant"].replace(self.tokenizer.image_token, "")})
        if not messages:
            return []
        if split_counts:
            messages[0]["content"] = get_image_string(self.tokenizer, split_counts, self.mp_image_token_length) + messages[0]["content"]
        return messages

    def _prepare_inputs(self, messages):
        conv = self.tokenizer.apply_chat_template(messages, tokenize=True, add_special_tokens=False, return_dict=True)
        mask = [0] * len(conv["input_ids"])
        cursor = 0
        for msg in messages:
            segment = self.tokenizer.apply_chat_template([msg], tokenize=True, add_special_tokens=False)
            seg_len = len(segment)
            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                mask[start:cursor + seg_len] = [1] * max(0, cursor + seg_len - start)
            cursor += seg_len
        input_ids = torch.tensor(conv["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(conv["attention_mask"], dtype=torch.long)
        labels = input_ids.clone().masked_fill(~torch.tensor(mask, dtype=torch.bool), -100).roll(-1)
        labels[-1] = -100
        return input_ids, attention_mask, labels

    def __getitem__(self, idx):
        item = self.dataset[idx]
        images = item["images"] if item.get("images") is not None else []
        if not isinstance(images, list):
            images = [images]
        if self.max_images_per_example and len(images) > self.max_images_per_example:
            return None
        processed_images, split_counts = self._process_images(images) if images else ([], [])
        messages = self._build_messages(item, split_counts)
        if not messages:
            return None
        input_ids, attention_mask, labels = self._prepare_inputs(messages)
        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class VQACollator:
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        rows = []
        for sample in batch:
            if sample is None or int(sample["input_ids"].size(0)) > int(self.max_length):
                continue
            rows.append(sample)
        if not rows:
            empty = torch.empty((0, 0), dtype=torch.long)
            return {"input_ids": empty, "attention_mask": empty, "labels": empty, "images": []}

        max_len = max(int(row["input_ids"].size(0)) for row in rows)
        padded = {"input_ids": [], "attention_mask": [], "labels": [], "images": []}
        for row in rows:
            for key, pad_value in (("input_ids", self.tokenizer.pad_token_id), ("attention_mask", 0), ("labels", -100)):
                padded[key].append(F.pad(row[key], (max_len - int(row[key].size(0)), 0), value=pad_value))
            padded["images"].append(row["images"])
        return {
            "input_ids": torch.stack(padded["input_ids"]),
            "attention_mask": torch.stack(padded["attention_mask"]),
            "labels": torch.stack(padded["labels"]),
            "images": padded["images"],
        }


def build_dataloaders(train_cfg, vlm_cfg):
    dataset_name = None if not train_cfg.train_dataset_name or train_cfg.train_dataset_name[0] in {"", "default", None} else train_cfg.train_dataset_name[0]
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template, model_max_length=train_cfg.max_sample_length)
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    train_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, split=train_cfg.train_split, streaming=False, on_bad_files="warn").shuffle(seed=0)
    val_ds = load_dataset(train_cfg.train_dataset_path, dataset_name, split=train_cfg.val_split, streaming=False, on_bad_files="warn")
    train_ds = apply_runtime_source_filter(train_ds, train_cfg, str(dataset_name), train_cfg.train_split)
    val_ds = apply_runtime_source_filter(val_ds, train_cfg, str(dataset_name), train_cfg.val_split)
    if is_distributed():
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())
        val_ds = val_ds.shard(num_shards=get_world_size(), index=get_rank())
    collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    train_dataset = VQADataset(train_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length, train_cfg)
    val_dataset = VQADataset(val_ds, tokenizer, image_processor, vlm_cfg.mp_image_token_length, train_cfg)
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=False, persistent_workers=False, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collator, num_workers=4, pin_memory=False, persistent_workers=False, drop_last=True)
    return tokenizer, train_loader, val_loader
