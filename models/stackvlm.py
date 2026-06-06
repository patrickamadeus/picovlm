from nanovlm import VisionLanguageModel, LanguageModel, VLMConfig
import torch.nn.functional as F
import torch

class StackVLM(VisionLanguageModel):
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__(cfg, load_backbone)
        self.full_decoder = LanguageModel(cfg)

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        images_tensor = self._process_images(images, input_ids.device)
        token_embd = self.decoder.token_embedding(input_ids)

        if images_tensor is not None:
            image_embd = self.vision_encoder(images_tensor)
            image_embd = self.MP(image_embd)
            
            # NOTE: implementation starts here, add language model on top of the vision encoder
            image_embd_len = image_embd.size(1)
            causal_mask = torch.tril(torch.ones((image_embd_len, image_embd_len), device=image_embd.device)).unsqueeze(0)
            image_embd = self.decoder(image_embd, attention_mask=causal_mask)
            # end of implementation
            
            token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)
     
        logits, _ = self.full_decoder(token_embd, attention_mask=attention_mask)
        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)

        return logits, loss