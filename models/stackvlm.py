from models.nanovlm import VisionLanguageModel, LanguageModel, VLMConfig, top_k_top_p_filtering
import torch.nn.functional as F
import torch
from typing import Optional
import os
from safetensors.torch import load_file
from dataclasses import fields
import json

class StackVLM(VisionLanguageModel):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__(cfg, load_backbone)

        if load_backbone:
            self.full_decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.full_decoder = LanguageModel(cfg)

    def _encode_images_with_llm(self, images_tensor):
        image_embd = self.vision_encoder(images_tensor)
        image_embd = self.MP(image_embd)
        image_embd, _ = self.decoder(image_embd, attention_mask=None)
        return image_embd

    def output_logits(self, hidden_states):
        return self.full_decoder.head(hidden_states) if not self.full_decoder.lm_use_tokens else hidden_states

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.full_decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self._encode_images_with_llm(images_tensor)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
     
        hidden_states, _ = self.full_decoder(token_embd, attention_mask=attention_mask)
        loss = None
        if targets is not None:
            logits = self.output_logits(hidden_states)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
            return logits, loss

        return hidden_states, loss

    @torch.inference_mode()
    def generate(self, input_ids, images, attention_mask=None, max_new_tokens=5, top_k=50, top_p=0.9, temperature=0.5, greedy=False):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.full_decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self._encode_images_with_llm(images_tensor)
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        current_total_seq_len = token_embd.size(1)
        batch_size = input_ids.size(0)
        prefill_output, kv_cache_list = self.full_decoder(token_embd, attention_mask=attention_mask, kv_cache=None, start_pos=0)
        last_token_output_from_prefill = prefill_output[:, -1, :]
        current_logits = self.output_logits(last_token_output_from_prefill)
        newly_generated_ids_list = []

        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)
            next_token_embed = self.full_decoder.token_embedding(next_token_id)
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            decode_step_output, kv_cache_list = self.full_decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )
            last_token_output = decode_step_output[:, -1, :]
            current_logits = self.output_logits(last_token_output)

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
        state_dict = load_file(weights_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if any(key.startswith("full_decoder.") for key in missing_keys):
            decoder_state = model.decoder.state_dict()
            model.full_decoder.load_state_dict(decoder_state, strict=False)
        if unexpected_keys:
            print(f"Ignored unexpected checkpoint keys: {unexpected_keys}")
        return model
