import math
import torch
from transformers import AutoTokenizer
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, InterpolationMode
from einops import rearrange
from typing import Tuple, Union
from PIL import Image
import re

TOKENIZERS_CACHE = {}


class DynamicResize(torch.nn.Module):
    """
    Resize so that:
      * the longer side ≤ `max_side_len` **and** is divisible by `patch_size`
      * the shorter side keeps aspect ratio and is also divisible by `patch_size`
    Optionally forbids up-scaling.

    Works on PIL Images, (C, H, W) tensors, or (B, C, H, W) tensors.
    Returns the same type it receives.
    """
    def __init__(
        self,
        patch_size: int,
        max_side_len: int,
        resize_to_max_side_len: bool = False,
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ) -> None:
        super().__init__()
        self.p = int(patch_size)
        self.m = int(max_side_len)
        self.interpolation = interpolation
        print(f"Resize to max side len: {resize_to_max_side_len}")
        self.resize_to_max_side_len = resize_to_max_side_len

    # ------------------------------------------------------------
    def _get_new_hw(self, h: int, w: int) -> Tuple[int, int]:
        """Compute target (h, w) divisible by patch_size."""
        long, short = (w, h) if w >= h else (h, w)

        # 1) upscale long side
        target_long = self.m if self.resize_to_max_side_len else min(self.m, math.ceil(long / self.p) * self.p)

        # 2) scale factor
        scale = target_long / long

        # 3) compute short side with ceil → never undershoot
        target_short = math.ceil(short * scale / self.p) * self.p
        target_short = max(target_short, self.p)  # just in case

        return (target_short, target_long) if w >= h else (target_long, target_short)

    # ------------------------------------------------------------
    def forward(self, img: Union[Image.Image, torch.Tensor]):
        if isinstance(img, Image.Image):
            w, h = img.size
            new_h, new_w = self._get_new_hw(h, w)
            return resize(img, [new_h, new_w], interpolation=self.interpolation)

        if not torch.is_tensor(img):
            raise TypeError(
                "DynamicResize expects a PIL Image or a torch.Tensor; "
                f"got {type(img)}"
            )

        # tensor path ---------------------------------------------------------
        batched = img.ndim == 4
        if img.ndim not in (3, 4):
            raise ValueError(
                "Tensor input must have shape (C,H,W) or (B,C,H,W); "
                f"got {img.shape}"
            )

        # operate batch-wise
        imgs = img if batched else img.unsqueeze(0)
        _, _, h, w = imgs.shape
        new_h, new_w = self._get_new_hw(h, w)
        out = resize(imgs, [new_h, new_w], interpolation=self.interpolation)

        return out if batched else out.squeeze(0)


class SplitImage(torch.nn.Module):
    """Split (B, C, H, W) image tensor into square patches.

    Returns:
        patches: (B·n_h·n_w, C, patch_size, patch_size)
        grid:    (n_h, n_w)  - number of patches along H and W
    """
    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.p = patch_size

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim == 3:            # add batch dim if missing
            x = x.unsqueeze(0)

        b, c, h, w = x.shape
        if h % self.p or w % self.p:
            raise ValueError(f'Image size {(h,w)} not divisible by patch_size {self.p}')

        n_h, n_w = h // self.p, w // self.p
        patches = rearrange(x, 'b c (nh ph) (nw pw) -> (b nh nw) c ph pw',
                            ph=self.p, pw=self.p)
        return patches, (n_h, n_w)


class GlobalAndSplitImages(torch.nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()
        self.p = patch_size
        self.splitter = SplitImage(patch_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        if x.ndim == 3:
            x = x.unsqueeze(0)

        patches, grid = self.splitter(x)

        if grid == (1, 1):
            return patches, grid  # Dont add global patch if there is only one patch

        global_patch = resize(x, [self.p, self.p])
        return torch.cat([global_patch, patches], dim=0), grid

def get_tokenizer(name, extra_special_tokens=None, chat_template=None, model_max_length=None):
    if name not in TOKENIZERS_CACHE:
        tokenizer_init_kwargs = {"use_fast": True}
        if extra_special_tokens is not None:
            tokenizer_init_kwargs["extra_special_tokens"] = extra_special_tokens
        if chat_template is not None:
            tokenizer_init_kwargs["chat_template"] = chat_template
        tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_init_kwargs,)
        tokenizer.pad_token = tokenizer.eos_token
        TOKENIZERS_CACHE[name] = tokenizer
    tokenizer = TOKENIZERS_CACHE[name]
    if model_max_length is not None:
        tokenizer.model_max_length = model_max_length
    return tokenizer

def get_image_processor(max_img_size, splitted_image_size, resize_to_max_side_len=False):
    return transforms.Compose([
        DynamicResize(splitted_image_size, max_img_size, resize_to_max_side_len),
        transforms.ToTensor(),
        GlobalAndSplitImages(splitted_image_size),
    ])

def get_image_string(tokenizer, splitted_image_counts, mp_image_token_length):
    image_string = ""
    # splitted_image_counts is a list of tuples (n_h, n_w)
    for idx, (n_h, n_w) in enumerate(splitted_image_counts):
        if len(splitted_image_counts) > 1:
            image_string += f"<image: {idx}>"
        if hasattr(tokenizer, "global_image_token"):
            image_string += tokenizer.global_image_token
            image_string += tokenizer.image_token * mp_image_token_length
            if n_h == 1 and n_w == 1:  # If there is only one patch, treat it as the global image
                continue
        for i in range(n_h):
            for j in range(n_w):
                image_string += getattr(tokenizer, f'r{i+1}c{j+1}')
                image_string += tokenizer.image_token * mp_image_token_length
    return image_string


# Used to check our models performance on multiple choice tasks. This can also be done in a more involved way with e.g. LLM-as-a-judge
def check_multiple_choice_with_regex(model_outputs, correct_answers):
    results = []
    for model_output, correct_answer in zip(model_outputs, correct_answers):
        # Strip any trailing newlines and convert to uppercase
        correct_answer = correct_answer.rstrip('\n').upper()

        # Look for the answer letter at the beginning of a line or as the last word
        patterns = [
            rf"\b{correct_answer}\b",  # Word boundary around the answer letter
            rf"\b{correct_answer}[.,)]",  # Answer followed by punctuation
            rf"\(.*{correct_answer}.*\)",  # Answer within parentheses
        ]

        match_found = False
        for pattern in patterns:
            if re.search(pattern, model_output):
                match_found = True
                break  # Exit inner loop once a match is found
        results.append(match_found)
    return results


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf')):
    """
    Apply top-k and/or nucleus (top-p) filtering to logits.
    """
    top_k = min(top_k, logits.size(-1))  # Safety

    if top_k > 0:
        # Remove all tokens with a probability less than the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p

        # Always keep the first token
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits
