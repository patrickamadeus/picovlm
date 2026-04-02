from pathlib import Path

import torch
from PIL import Image

from utils.processor_transforms import get_image_processor, get_image_string, get_tokenizer


TRAIN_SAMPLE_SPECS = (
    {"name": "image_1", "image": "assets/cat.png", "prompt": "Describe the image."},
    {"name": "image_2", "image": "assets/clinic.png", "prompt": "What is the name of the clinic?"},
    {
        "name": "image_3",
        "image": "assets/case.png",
        "prompt": (
            "Which option describe the object relationship in the image correctly?\n"
            "Options: A: The suitcase is on the book., B: The suitcase is beneath the cat., "
            "C: The suitcase is beneath the bed., D: The suitcase is beneath the book."
        ),
    },
    {"name": "image_4_count", "image": "assets/soccer.png", "prompt": "How many players are there?"},
    {"name": "image_4_color", "image": "assets/soccer.png", "prompt": "What is the color of their shirt?"},
)


def resolve_train_sample_specs(base_dir):
    base_dir = Path(base_dir)
    return [{**spec, "image": str((base_dir / spec["image"]).resolve())} for spec in TRAIN_SAMPLE_SPECS]


def build_generation_inputs(model, image_path: str, prompt: str, device: torch.device):
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, model.cfg.resize_to_max_side_len)

    image = Image.open(image_path).convert("RGB")
    processed_image, split_counts = image_processor(image)
    if not hasattr(tokenizer, "global_image_token") and split_counts[0] * split_counts[1] == len(processed_image) - 1:
        processed_image = processed_image[1:]
    image_string = get_image_string(tokenizer, [split_counts], model.cfg.mp_image_token_length)

    messages = [{"role": "user", "content": image_string + prompt}]
    prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return tokenizer, input_ids, attention_mask, [processed_image]


@torch.inference_mode()
def generate_fixed_samples(
    model,
    *,
    device,
    base_dir,
    generations_per_prompt=1,
    max_new_tokens=64,
    top_k=50,
    top_p=0.9,
    temperature=0.7,
    greedy=False,
):
    sample_outputs = []
    for spec in resolve_train_sample_specs(base_dir):
        tokenizer, input_ids, attention_mask, images = build_generation_inputs(model, spec["image"], spec["prompt"], device)
        generations = []
        for _ in range(int(generations_per_prompt)):
            generated = model.generate(
                input_ids=input_ids,
                images=images,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                greedy=greedy,
            )
            generations.append(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])
        sample_outputs.append(
            {
                "name": spec["name"],
                "image": spec["image"],
                "prompt": spec["prompt"],
                "generations": generations,
            }
        )
    return sample_outputs
