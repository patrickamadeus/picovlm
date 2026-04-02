import argparse

import torch
import yaml

from models.nanovlm import VisionLanguageModel
from utils.generation_helper import build_generation_inputs


torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text from an image with NanoVLM")
    parser.add_argument("--config", type=str, default=None, help="optional YAML config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="local checkpoint directory or HF repo id")
    parser.add_argument("--image", type=str, default="./assets/cat.png", help="path to input image")
    parser.add_argument("--prompt", type=str, default="What is in the image?", help="text prompt")
    parser.add_argument("--generations", type=int, default=5, help="number of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="maximum number of new tokens")
    parser.add_argument("--top_k", type=int, default=None, help="top-k sampling")
    parser.add_argument("--top_p", type=float, default=None, help="top-p sampling")
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="enable greedy decoding")
    return parser.parse_args()


def load_yaml_config(path: str | None):
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    return loaded.get("generation", loaded)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    yaml_cfg = load_yaml_config(args.config)

    checkpoint = args.checkpoint or yaml_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("--checkpoint is required")
    image_path = args.image or yaml_cfg.get("image")
    prompt = args.prompt or yaml_cfg.get("prompt")
    if not image_path or not prompt:
        raise ValueError("--image and --prompt are required unless provided by --config")
    generations = args.generations if args.generations is not None else int(yaml_cfg.get("generations", 1))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(yaml_cfg.get("max_new_tokens", 50))
    top_k = args.top_k if args.top_k is not None else int(yaml_cfg.get("top_k", 50))
    top_p = args.top_p if args.top_p is not None else float(yaml_cfg.get("top_p", 0.9))
    temperature = args.temperature if args.temperature is not None else float(yaml_cfg.get("temperature", 0.7))
    greedy = bool(args.greedy or yaml_cfg.get("greedy", False))

    device = pick_device()
    model = VisionLanguageModel.from_pretrained(checkpoint).to(device)
    model.eval()

    tokenizer, input_ids, attention_mask, images = build_generation_inputs(model, image_path, prompt, device)

    print(f"using device: {device}")
    print(f"loading weights from: {checkpoint}")
    print(f"\ninput:\n  {prompt}\n\noutput:")

    for idx in range(generations):
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
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        print(f"  >> generation {idx + 1}: {decoded}")


if __name__ == "__main__":
    main()
