from dataclasses import dataclass, field, fields


@dataclass
class VLMConfig:
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072
    vit_patch_size: int = 16
    vit_img_size: int = 512
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    vit_cls_flag: bool = False
    vit_model_type: str = "google/siglip2-base-patch16-512"

    lm_hidden_dim: int = 576
    lm_inter_dim: int = 1536
    lm_rms_eps: float = 1e-5
    lm_re_base: int = 100000
    lm_max_position_embeddings: int = 1024
    lm_base_vocab_size: int = 49152
    extra_token_amount: int = 66
    lm_vocab_size: int = 49218
    lm_n_heads: int = 9
    lm_n_kv_heads: int = 3
    lm_dropout: float = 0.0
    lm_n_blocks: int = 30
    lm_attn_scaling: float = 1.0
    lm_pad_aware_rope: bool = False
    lm_max_length: int = 1024
    lm_use_tokens: bool = False
    lm_tie_weights: bool = True
    lm_model_type: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    lm_chat_template: str = (
        "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
        "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )

    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64
    max_img_size: int = 2048
    resize_to_max_side_len: bool = False

    vlm_extra_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "image_token": "<|image|>",
            "global_image_token": "<|global_image|>",
            **{f"r{r}c{c}": f"<row_{r}_col_{c}>" for r in range(1, 9) for c in range(1, 9)},
        }
    )
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str | None = "lusxvr/nanoVLM-230M-8k"


@dataclass
class TrainConfig:
    lr_mp: float = 5e-5
    lr_vision_backbone: float = 0.0
    lr_language_backbone: float = 5e-5
    lr_full_decoder: float = 5e-5

    batch_size: int = 16
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    max_training_steps: int = 20_000
    stop_after_step: int | None = None
    warmup_ratio: float = 0.03
    stats_log_interval: int = 100
    precision: str = "fp32"
    compile: bool = False

    do_eval: bool = True
    eval_interval: int = 500
    max_val_batches: int = 32

    max_images_per_example: int = 1
    max_sample_length: int = 1024

    train_dataset_path: str = "patrickamadeus/the_cauldron"
    train_dataset_name: tuple[str, ...] = ("all",)
    train_split: str = "train"
    val_split: str = "validation"
    stream_dataset: bool = False
    enable_source_filter: bool = False
    allowed_dataset_sources: tuple[str, ...] = ("ocrvqa",)
    relevance_min_rating: int = 1
    image_correspondence_min_rating: int = 1
    visual_dependency_min_rating: int = 1
    formatting_min_rating: int = 1

    wandb_entity: str = "HuggingFace"
    log_wandb: bool = False
    push_checkpoints_to_hub: bool = False
    save_training_state_to_hub: bool = False
    checkpoint_repo_pattern: str = "patrickamadeus/nanovlm-{i}"
    hf_private: bool = False
    push_final_model_to_hub: bool = False
    resume_from_vlm_checkpoint: bool = True
    resume_checkpoint_path: str | None = None


def filter_config_payload(config_cls, payload: dict):
    valid_keys = {field.name for field in fields(config_cls)}
    return {key: value for key, value in payload.items() if key in valid_keys}


def instantiate_config(config_cls, payload: dict):
    return config_cls(**filter_config_payload(config_cls, payload))


def load_vlm_config(payload: dict):
    return instantiate_config(VLMConfig, payload.get("vlm", payload))


def load_train_config(payload: dict):
    return instantiate_config(TrainConfig, payload.get("train", payload))
