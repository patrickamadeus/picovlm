import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model
from configs.config import VLMConfig

from utils.processor_transforms import get_tokenizer


def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits = logits.masked_fill(indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim
        self.conv = nn.Conv2d(3, self.embd_dim, kernel_size=self.patch_size, stride=self.patch_size, padding="valid")

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        return x + self.position_embedding


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.vit_dropout
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.sdpa:
            print("Warning: scaled dot product attention not available. Using standard attention in ViT.")

    def forward(self, x):
        bsz, seq_len, channels = x.size()
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(channels, dim=2)
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.sdpa:
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        y = self.out_proj(y)
        return self.resid_dropout(y)


class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate="tanh")
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        return self.dropout(x)


class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            return self.layer_norm(x[:, 0])
        return self.layer_norm(x)

    @classmethod
    def from_pretrained(cls, cfg):
        from huggingface_hub import hf_hub_download
        from transformers import SiglipVisionConfig
        import safetensors

        hf_config = SiglipVisionConfig.from_pretrained(cfg.vit_model_type)
        cfg.vit_dropout = hf_config.attention_dropout
        cfg.vit_hidden_dim = hf_config.hidden_size
        cfg.vit_img_size = hf_config.image_size
        cfg.vit_inter_dim = hf_config.intermediate_size
        cfg.vit_ln_eps = hf_config.layer_norm_eps
        cfg.vit_n_heads = hf_config.num_attention_heads
        cfg.vit_n_blocks = hf_config.num_hidden_layers
        cfg.vit_patch_size = hf_config.patch_size
        model = cls(cfg)
        safetensors_file = hf_hub_download(repo_id=cfg.vit_model_type, filename="model.safetensors")

        sd = model.state_dict()
        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias": "layer_norm.bias",
        }

        for i in range(cfg.vit_n_blocks):
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = f"blocks.{i}.ln1.weight"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = f"blocks.{i}.ln1.bias"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = f"blocks.{i}.ln2.weight"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = f"blocks.{i}.ln2.bias"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = f"blocks.{i}.attn.out_proj.weight"
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = f"blocks.{i}.attn.out_proj.bias"

        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    elif "position_embedding" in hf_key:
                        sd[our_key].copy_(tensor.unsqueeze(0))
                    else:
                        print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")

            for i in range(model.cfg.vit_n_blocks):
                q_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight")
                k_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight")
                v_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight")
                sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(torch.cat((q_weight, k_weight, v_weight), dim=0))

                q_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias")
                k_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias")
                v_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias")
                sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(torch.cat((q_bias, k_bias, v_bias), dim=0))

        model.load_state_dict(sd)
        print(f"Successfully loaded {cfg.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model


class ModalityProjector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_dim = cfg.vit_hidden_dim * (cfg.mp_pixel_shuffle_factor ** 2)
        self.output_dim = cfg.lm_hidden_dim
        self.scale_factor = cfg.mp_pixel_shuffle_factor
        self.proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        seq_root = int(seq ** 0.5)
        assert seq_root ** 2 == seq, "Sequence length must be a perfect square for pixel shuffle"
        assert seq_root % self.scale_factor == 0, "Sequence root must be divisible by scale factor"

        x = x.view(bsz, seq_root, seq_root, embed_dim)
        h_out = seq_root // self.scale_factor
        w_out = seq_root // self.scale_factor
        x = x.reshape(bsz, h_out, self.scale_factor, w_out, self.scale_factor, embed_dim)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.reshape(bsz, h_out * w_out, embed_dim * self.scale_factor ** 2)

    def forward(self, x):
        return self.proj(self.pixel_shuffle(x))


class RMSNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(cfg.lm_hidden_dim))
        self.eps = cfg.lm_rms_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        irms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x * irms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.lm_hidden_dim % cfg.lm_n_heads == 0, "Hidden dimension must be divisible by number of heads"
        self.dim = cfg.lm_hidden_dim // cfg.lm_n_heads
        self.base = cfg.lm_re_base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self.original_max_seq_len = cfg.lm_max_position_embeddings
        self.attention_scaling = cfg.lm_attn_scaling

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = position_ids.shape
        max_seq = position_ids.max() + 1
        inv_freq = self.inv_freq / (max_seq / self.original_max_seq_len) if max_seq > self.original_max_seq_len else self.inv_freq
        freqs = position_ids.reshape(-1).float().unsqueeze(-1) * inv_freq.unsqueeze(0)
        freqs = freqs.reshape(batch_size, seq_len, -1)
        emb = torch.cat([freqs, freqs], dim=-1)
        return torch.cos(emb) * self.attention_scaling, torch.sin(emb) * self.attention_scaling


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_embd(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    # Keep rotary factors aligned with q/k dtype so SDPA receives q, k, and v
    # in the same precision during half/bfloat16 execution.
    cos = cos.to(device=q.device, dtype=q.dtype).unsqueeze(unsqueeze_dim)
    sin = sin.to(device=q.device, dtype=q.dtype).unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LanguageModelGroupedQueryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.lm_n_heads
        self.n_kv_heads = cfg.lm_n_kv_heads
        self.embd_dim = cfg.lm_hidden_dim
        self.dropout = cfg.lm_dropout

        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"

        self.n_kv_groups = self.n_heads // self.n_kv_heads
        self.head_dim = self.embd_dim // self.n_heads
        self.q_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.k_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.v_proj = nn.Linear(self.embd_dim, self.head_dim * self.n_kv_heads, bias=False)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)
        self.sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.sdpa:
            print("Warning: scaled dot product attention not available, using standard attention in LM.")

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask=None, block_kv_cache=None) -> tuple[torch.Tensor, dict]:
        is_prefill = block_kv_cache is None
        bsz, t_curr, channels = x.size()

        q_curr = self.q_proj(x).view(bsz, t_curr, self.n_heads, self.head_dim).transpose(1, 2)
        k_curr = self.k_proj(x).view(bsz, t_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_curr = self.v_proj(x).view(bsz, t_curr, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k_rotated = apply_rotary_pos_embd(q_curr, k_curr, cos, sin)

        if not is_prefill and block_kv_cache["key"] is not None:
            k = torch.cat([block_kv_cache["key"], k_rotated], dim=2)
            v = torch.cat([block_kv_cache["value"], v_curr], dim=2)
            block_kv_cache["key"] = k
            block_kv_cache["value"] = v
        else:
            k = k_rotated
            v = v_curr
            block_kv_cache = {"key": k, "value": v}

        k_exp = k.repeat_interleave(self.n_kv_groups, dim=1)
        v_exp = v.repeat_interleave(self.n_kv_groups, dim=1)
        t_kv = k_exp.size(2)

        additive_attn_mask = None
        if attention_mask is not None:
            mask_for_keys = attention_mask[:, :t_kv]
            mask_for_keys = mask_for_keys.unsqueeze(1).unsqueeze(2).to(dtype=q.dtype)
            additive_attn_mask = (1.0 - mask_for_keys) * torch.finfo(q.dtype).min

        if self.sdpa and x.device.type != "mps":
            is_causal = t_curr == t_kv and t_curr > 1
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k_exp,
                v_exp,
                attn_mask=additive_attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            attn = torch.matmul(q, k_exp.transpose(2, 3)) / math.sqrt(self.head_dim)
            if t_curr == t_kv and t_curr > 1:
                causal_mask_val = torch.tril(torch.ones(t_curr, t_curr, device=x.device, dtype=torch.bool)).view(1, 1, t_curr, t_curr)
                attn = attn.masked_fill(~causal_mask_val, float("-inf"))
            if additive_attn_mask is not None:
                attn = attn + additive_attn_mask
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v_exp

        y = y.transpose(1, 2).contiguous().view(bsz, t_curr, channels)
        y = self.out_proj(y)
        return self.resid_dropout(y), block_kv_cache


class LanguageModelMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = F.silu
        self.gate_proj = nn.Linear(cfg.lm_hidden_dim, cfg.lm_inter_dim, bias=False)
        self.up_proj = nn.Linear(cfg.lm_hidden_dim, cfg.lm_inter_dim, bias=False)
        self.down_proj = nn.Linear(cfg.lm_inter_dim, cfg.lm_hidden_dim, bias=False)

    def forward(self, x):
        gate = self.activation_fn(self.gate_proj(x))
        x = self.up_proj(x)
        return self.down_proj(gate * x)


class LanguageModelBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = LanguageModelMLP(cfg)
        self.attn = LanguageModelGroupedQueryAttention(cfg)
        self.norm1 = RMSNorm(cfg)
        self.norm2 = RMSNorm(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attention_mask: torch.Tensor = None, block_kv_cache: dict = None):
        res = x
        x = self.norm1(x)
        x, block_kv_cache = self.attn(x, cos, sin, attention_mask, block_kv_cache)
        x = res + x
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        return res + x, block_kv_cache


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lm_use_tokens = cfg.lm_use_tokens
        self.lm_tie_weights = cfg.lm_tie_weights
        self.token_embedding = nn.Embedding(cfg.lm_vocab_size, cfg.lm_hidden_dim)
        self.rotary_embd = RotaryEmbedding(cfg)
        self.blocks = nn.ModuleList([LanguageModelBlock(cfg) for _ in range(cfg.lm_n_blocks)])
        self.norm = RMSNorm(cfg)
        self.head = nn.Linear(cfg.lm_hidden_dim, cfg.lm_vocab_size, bias=False)
        if self.lm_tie_weights:
            self.head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None, kv_cache: list[dict] = None, start_pos: int = 0):
        if self.lm_use_tokens:
            x = self.token_embedding(x)

        bsz, t_curr, _ = x.size()
        current_position_ids = torch.arange(start_pos, start_pos + t_curr, device=x.device).unsqueeze(0).expand(bsz, -1)
        cos, sin = self.rotary_embd(current_position_ids)

        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        for i, block in enumerate(self.blocks):
            x, kv_cache[i] = block(x, cos, sin, attention_mask, kv_cache[i])

        x = self.norm(x)
        if self.lm_use_tokens:
            x = self.head(x)
        return x, kv_cache

    @torch.inference_mode()
    def generate(self, inputs: torch.Tensor, max_new_tokens: int = 20):
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
        generated_outputs = inputs.clone()
        prompt_output, kv_cache_list = self.forward(generated_outputs, attention_mask=None, kv_cache=None, start_pos=0)
        last_output = prompt_output[:, -1, :]

        for i in range(max_new_tokens):
            next_output = torch.argmax(last_output, dim=-1, keepdim=True) if self.lm_use_tokens else last_output.unsqueeze(1)
            generated_outputs = torch.cat((generated_outputs, next_output), dim=1)
            current_token_start_pos = generated_outputs.size(1) - 1

            if i == max_new_tokens - 1:
                break

            decode_step_output, kv_cache_list = self.forward(next_output, attention_mask=None, kv_cache=kv_cache_list, start_pos=current_token_start_pos)
            last_output = decode_step_output[:, -1, :]

        return generated_outputs

    @classmethod
    def from_pretrained(cls, cfg):
        import json
        import safetensors
        import torch.nn.init as init
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(cfg.lm_model_type)
        original_vocab_size = hf_config.vocab_size
        cfg.lm_hidden_dim = hf_config.hidden_size
        cfg.lm_inter_dim = hf_config.intermediate_size
        cfg.lm_rms_eps = hf_config.rms_norm_eps
        cfg.lm_re_base = hf_config.rope_theta
        cfg.lm_max_position_embeddings = hf_config.max_position_embeddings
        if hasattr(cfg, "lm_vocab_size"):
            if cfg.lm_vocab_size < original_vocab_size:
                raise ValueError(f"Config vocab size ({cfg.lm_vocab_size}) is smaller than pretrained model vocab size ({original_vocab_size})")
        else:
            cfg.lm_vocab_size = original_vocab_size
        cfg.lm_n_heads = hf_config.num_attention_heads
        cfg.lm_n_kv_heads = hf_config.num_key_value_heads
        cfg.lm_dropout = hf_config.attention_dropout
        cfg.lm_n_blocks = hf_config.num_hidden_layers

        model = cls(cfg)
        try:
            index_path = hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors.index.json")
            with open(index_path, "r") as f:
                index = json.load(f)
            safetensors_filenames = sorted(list(set(index["weight_map"].values())))
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename=fn) for fn in safetensors_filenames]
        except EntryNotFoundError:
            safetensors_files = [hf_hub_download(repo_id=cfg.lm_model_type, filename="model.safetensors")]

        sd = model.state_dict()
        mapping = {"model.embed_tokens.weight": "token_embedding.weight", "model.norm.weight": "norm.weight"}
        for i in range(cfg.lm_n_blocks):
            layer_prefix = f"model.layers.{i}."
            block_prefix = f"blocks.{i}."
            mapping.update({
                f"{layer_prefix}self_attn.q_proj.weight": f"{block_prefix}attn.q_proj.weight",
                f"{layer_prefix}self_attn.k_proj.weight": f"{block_prefix}attn.k_proj.weight",
                f"{layer_prefix}self_attn.v_proj.weight": f"{block_prefix}attn.v_proj.weight",
                f"{layer_prefix}self_attn.o_proj.weight": f"{block_prefix}attn.out_proj.weight",
                f"{layer_prefix}mlp.gate_proj.weight": f"{block_prefix}mlp.gate_proj.weight",
                f"{layer_prefix}mlp.up_proj.weight": f"{block_prefix}mlp.up_proj.weight",
                f"{layer_prefix}mlp.down_proj.weight": f"{block_prefix}mlp.down_proj.weight",
                f"{layer_prefix}input_layernorm.weight": f"{block_prefix}norm1.weight",
                f"{layer_prefix}post_attention_layernorm.weight": f"{block_prefix}norm2.weight",
            })

        has_extended_embeddings = False
        loaded_keys = set()
        for safetensors_file in safetensors_files:
            with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                for hf_key, our_key in mapping.items():
                    if our_key in loaded_keys:
                        continue
                    if hf_key in f.keys() and our_key in sd:
                        tensor = f.get_tensor(hf_key)
                        if hf_key == "model.embed_tokens.weight" and tensor.shape[0] != sd[our_key].shape[0]:
                            has_extended_embeddings = True
                            print(f"Extending token embeddings from {tensor.shape} to {sd[our_key].shape}")
                            sd[our_key][:tensor.shape[0]].copy_(tensor)
                            init.normal_(sd[our_key][tensor.shape[0]:], mean=0.0, std=0.02)
                            sd["head.weight"].copy_(sd[our_key])
                        elif tensor.shape == sd[our_key].shape:
                            sd[our_key].copy_(tensor)
                        else:
                            print(f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}")
                        loaded_keys.add(our_key)

        model.load_state_dict(sd)

        if has_extended_embeddings and hasattr(model, "head") and "head.weight" in sd:
            for safetensors_file in safetensors_files:
                with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
                    if "lm_head.weight" in f.keys():
                        lm_head = f.get_tensor("lm_head.weight")
                        if lm_head.shape[0] != sd["head.weight"].shape[0]:
                            print(f"Extending LM head from {lm_head.shape} to {sd['head.weight'].shape}")
                            sd["head.weight"][:lm_head.shape[0]].copy_(lm_head)
                            init.normal_(sd["head.weight"][lm_head.shape[0]:], mean=0.0, std=0.02)
                            model.load_state_dict(sd)
                        break

        if cfg.lm_tie_weights and hasattr(model, "head") and hasattr(model, "token_embedding"):
            model.head.weight = model.token_embedding.weight

        print(f"Successfully loaded {cfg.lm_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters.")
        return model


class VisionLanguageModel(nn.Module):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)
        self.MP = ModalityProjector(cfg)
        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(cfg.lm_tokenizer, cfg.vlm_extra_tokens, cfg.lm_chat_template)

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        updated_token_embd = token_embd.clone()
        mask = input_ids == self.tokenizer.image_token_id
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)
        return updated_token_embd

    def _process_images(self, images, device):
        if isinstance(images, list):
            if images and isinstance(images[0], list):
                images = [img for sublist in images for img in sublist]
            if not images:
                return None
            return torch.cat(images, dim=0).to(device)
        return images

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        logits, _ = self.decoder(token_embd, attention_mask=attention_mask)

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0)
        prefill_output, kv_cache_list = self.decoder(token_embd, attention_mask=attention_mask, kv_cache=None, start_pos=0)
        last_token_output_from_prefill = prefill_output[:, -1, :]
        current_logits = self.decoder.head(last_token_output_from_prefill) if not self.decoder.lm_use_tokens else last_token_output_from_prefill
        newly_generated_ids_list = []

        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)
            next_token_embed = self.decoder.token_embedding(next_token_id)
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )
            last_token_output = decode_step_output[:, -1, :]
            current_logits = self.decoder.head(last_token_output) if not self.decoder.lm_use_tokens else last_token_output

        if not newly_generated_ids_list:
            return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        if self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0:
            seq_len = generated_ids.size(1)
            device = generated_ids.device
            eos_mask = generated_ids == self.tokenizer.eos_token_id
            col_indices_for_min = torch.arange(seq_len, device=device)
            masked_col_indices = torch.where(eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1)
            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)
            col_indices_for_comparison = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)
            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids

    @classmethod
    def from_pretrained(cls, repo_id_or_path: str, *, revision: Optional[str] = None) -> "VisionLanguageModel":
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")
            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found at {config_path}. Please provide a valid path.")
            if not os.path.exists(weights_path):
                raise ValueError(f"Weights file not found at {weights_path}. Please provide a valid path.")
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=repo_id_or_path, filename="config.json", revision=revision)
            weights_path = hf_hub_download(repo_id=repo_id_or_path, filename="model.safetensors", revision=revision)

        with open(config_path, "r") as f:
            raw_cfg = json.load(f)
        valid_keys = {field.name for field in fields(VLMConfig)}
        cfg = VLMConfig(**{key: value for key, value in raw_cfg.items() if key in valid_keys})

        model = cls(cfg, load_backbone=False)
        load_model(model, weights_path)
        return model

    def save_pretrained(self, save_directory: str) -> None:
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        from huggingface_hub import create_repo, upload_folder

        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            self.save_pretrained(save_path)
            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))
            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM using push_to_hub",
            )



MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: image-text-to-text
tags:
  - vision-language
  - multimodal
  - research
---

**nanoVLM** is a minimal and lightweight Vision-Language Model (VLM) designed for efficient training and experimentation. Built using pure PyTorch, the entire model architecture and training logic fits within ~750 lines of code. It combines a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M), resulting in a compact 222M parameter model.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.nanovlm import VisionLanguageModel

model = VisionLanguageModel.from_pretrained("{repo_id}")
```
"""
