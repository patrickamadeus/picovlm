from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

from models.nanovlm import VisionLanguageModel
from utils.processor_transforms import get_image_processor, get_image_string, get_tokenizer

try:
    from lmms_eval import utils as lmms_utils
    from lmms_eval.api.instance import Instance
    from lmms_eval.api.model import lmms
except ImportError:  # pragma: no cover - optional runtime dependency
    lmms_utils = None
    Instance = object

    class lmms:  # type: ignore[override]
        pass


def _flatten_visuals(items):
    flattened = []
    for sublist in items:
        if sublist is None:
            flattened.append(None)
        else:
            flattened.extend(sublist)
    return flattened


def _get_benchmark_formatting(task_name: str) -> dict:
    benchmark_formats = {
        ("ai2d", "mmstar", "seedbench", "scienceqa"): {
            "text_replacements": {
                "\nOptions:": "\nChoices:",
                "\nA. ": "\nChoices:\nA. ",
                "Please select the correct answer from the options above.": "Answer with the letter.",
                "Answer with the option's letter from the given choices directly": "Answer with the letter directly",
            },
            "assistant_prefix": "Answer:",
            "user_prefix": "",
            "user_suffix": "",
        },
        ("docvqa_val", "docvqa_test"): {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": "Give a short and terse answer to the following question. Do not paraphrase or reformat the text you see in the image. Do not include any full stops. Just give the answer without additional explanation. Question: ",
            "user_suffix": "",
        },
        "chartvqa": {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": "For the question below, follow the following instructions:\n-The answer should contain as few words as possible.\n-Don't paraphrase or reformat the text you see in the image.\n-Answer a binary question with Yes or No.\n-When asked to give a numerical value, provide a number like 2 instead of Two.\n-If the final answer has two or more items, provide it in the list format like [1, 2].\n-When asked to give a ratio, give out the decimal value like 0.25 instead of 1:4.\n-When asked to give a percentage, give out the whole value like 17 instead of decimal like 0.17%.\n-Don't include any units in the answer.\n-Do not include any full stops at the end of the answer.\n-Try to include the full label from the graph when asked about an entity.\nQuestion: ",
            "user_suffix": "",
        },
        ("textvqa_val", "textvqa_test"): {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": "Answer the following question about the image using as few words as possible. Follow these additional instructions:\n-Always answer a binary question with Yes or No.\n-When asked what time it is, reply with the time seen in the image.\n-Do not put any full stops at the end of the answer.\n-Do not put quotation marks around the answer.\n-An answer with one or two words is favorable.\n-Do not apply common sense knowledge. The answer can be found in the image.\nQuestion: ",
            "user_suffix": "",
        },
        ("mmmu_val", "mmmu_test"): {
            "text_replacements": {
                "Question:": "",
                "Answer with the option's letter from the given choices directly.": "Answer with the letter directly.",
                "\nA. ": "\nChoices:\nA. ",
            },
            "assistant_prefix": "Answer:",
            "user_prefix": "",
            "user_suffix": "",
        },
        ("infovqa_val", "mme", "ocrbench"): {
            "text_replacements": {},
            "assistant_prefix": "",
            "user_prefix": "",
            "user_suffix": "\nGive a very brief answer.",
        },
    }
    if task_name in benchmark_formats:
        return benchmark_formats[task_name]
    for key, formatting in benchmark_formats.items():
        if isinstance(key, (list, tuple)) and task_name in key:
            return formatting
    return {"text_replacements": {}, "assistant_prefix": "", "user_prefix": "", "user_suffix": ""}


def _apply_benchmark_formatting(context_str: str, prompt: str, task_name: str) -> tuple[str, str]:
    formatting = _get_benchmark_formatting(task_name)
    if formatting["user_prefix"]:
        context_str = formatting["user_prefix"] + context_str
    for old_text, new_text in formatting["text_replacements"].items():
        context_str = context_str.replace(old_text, new_text)
    if formatting["user_suffix"]:
        context_str = context_str + formatting["user_suffix"]
    if formatting["assistant_prefix"]:
        prompt = prompt + formatting["assistant_prefix"]
    return context_str, prompt


class NanoVLMWrapper(lmms):
    def __init__(self, model="lusxvr/nanoVLM-230M-8k", device: str = "cuda", batch_size: int = 8, **kwargs):
        if lmms_utils is None:
            raise ImportError("lmms_eval is required to use NanoVLMWrapper")
        super().__init__()
        self.model = VisionLanguageModel.from_pretrained(model).to(device) if isinstance(model, str) else model.to(device)
        self.model.eval()
        self.device = device
        self.batch_size = batch_size
        self._rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        self.tokenizer = get_tokenizer(self.model.cfg.lm_tokenizer, self.model.cfg.vlm_extra_tokens, self.model.cfg.lm_chat_template)
        self.image_processor = get_image_processor(self.model.cfg.max_img_size, self.model.cfg.vit_img_size, self.model.cfg.resize_to_max_side_len)

    def _prepare_visual_input(self, visual_list: List[Image.Image]) -> tuple[Optional[list], Optional[list]]:
        if not visual_list:
            return [], []
        images = []
        split_counts = []
        for visual in visual_list:
            if visual is None:
                continue
            if isinstance(visual, Image.Image):
                image = visual
            elif isinstance(visual, str):
                image = Image.open(visual).convert("RGB")
            elif isinstance(visual, np.ndarray):
                image = Image.fromarray(visual)
            else:
                raise ValueError(f"unsupported visual type: {type(visual)}")
            processed_images, split_count = self.image_processor(image)
            if not hasattr(self.tokenizer, "global_image_token") and split_count[0] * split_count[1] == len(processed_images) - 1:
                processed_images = processed_images[1:]
            images.append(processed_images)
            split_counts.append(split_count)
        return images, split_counts

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("loglikelihood is not implemented for NanoVLM")

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = lmms_utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            visuals = [dtv(self.task_dict[t][s][i]) for dtv, i, t, s in zip(doc_to_visual, doc_id, task, split)]
            images, split_counts = self._prepare_visual_input(_flatten_visuals(visuals))
            messages = []
            split_idx = 0
            for idx, context_str in enumerate(contexts):
                current_context, _ = _apply_benchmark_formatting(context_str, "", task[idx])
                image_count = 0 if visuals[idx] is None else len(visuals[idx])
                image_string = ""
                for _ in range(image_count):
                    image_string += get_image_string(self.tokenizer, [split_counts[split_idx]], self.model.cfg.mp_image_token_length)
                    split_idx += 1
                messages.append([{"role": "user", "content": image_string + current_context}])

            prompts = [self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
            for idx in range(len(prompts)):
                _, prompts[idx] = _apply_benchmark_formatting("", prompts[idx], task[idx])

            old_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "right"
            try:
                inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_length)
            finally:
                self.tokenizer.padding_side = old_padding_side

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            current_gen_kwargs = all_gen_kwargs[0] if all_gen_kwargs else {}
            max_new_tokens = current_gen_kwargs.get("max_new_tokens", 50)
            temperature = current_gen_kwargs.get("temperature", 0.0)
            top_p = current_gen_kwargs.get("top_p", 1.0)
            top_k = current_gen_kwargs.get("top_k", 50)
            greedy = current_gen_kwargs.get("do_sample", False) is False or temperature == 0.0
            kwargs = {
                "input_ids": input_ids,
                "images": images or [],
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "greedy": greedy,
                "temperature": None if greedy else temperature,
                "top_p": None if greedy else top_p,
                "top_k": top_k,
            }
            generated_ids = self.model.generate(**kwargs)
            res.extend(self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
            pbar.update(len(contexts))

        pbar.close()
        return re_ords.get_original(res)

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("multi-round generation is not implemented for NanoVLM")

    @property
    def max_length(self):
        return self.model.cfg.lm_max_position_embeddings

    @property
    def batch_size_per_gpu(self):
        return self.batch_size

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size
