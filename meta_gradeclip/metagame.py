"""Meta-Grad-ECLIP attribution: Metagame and MetagameTokens classes plus the
shared `explain_image` helper used by both `pointing_game.py` and `example.py`.
"""

import math
import warnings

import shapiq
import torch
import torch.nn.functional as F
from PIL import Image

import gradeclip
import gradesiglip


device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore")


def explain_image(input_image, input_text, model, processor, mode="gradeclip"):
    """Compute an image explanation map.

    Args:
        input_image: PIL Image or preprocessed tensor
        input_text:  text string
        model:       CLIPModel or SiglipModel instance
        processor:   matching AutoProcessor instance
        mode:        explanation method —
                       CLIP:   "gradeclip", "genericattention", "maskclip", "gradcam", "selfattention", "attentionrollout"
                       SigLIP: "gradesiglip", "genericattention", "masksiglip", "gradcam", "selfattention", "attentionrollout"

    Returns:
        emap: [H, W] explanation heat map
    """
    if isinstance(input_image, Image.Image):
        image_preprocessed = processor(images=input_image, return_tensors="pt")["pixel_values"].to(device)
    else:
        image_preprocessed = input_image.to(device)

    if "clip" in str(type(model)):
        text_inputs    = processor(text=[input_text], return_tensors="pt", padding=True).to(device)
        text_embedding = F.normalize(model.get_text_features(**text_inputs), dim=-1)

        if mode in ("gradeclip", "maskclip", "gradcam", "selfattention"):
            outputs, v_final, last_input, v, q_out, k_out, attn, att_output, map_size = \
                gradeclip.clip_encode_image_dense(image_preprocessed, model)
            cosines = (F.normalize(outputs[:, 0], dim=-1) @ text_embedding.T)[0]

        if mode == "gradeclip":
            emap = [gradeclip.gradeclip_image(c, q_out, k_out, v, att_output, map_size) for c in cosines]
            emap = torch.stack(emap, dim=0).sum(0)
        elif mode == "maskclip":
            emap = gradeclip.maskclip(text_embedding[0], v_final, k_out, map_size)
        elif mode == "gradcam":
            emap = [gradeclip.gradcam(c, last_input, map_size) for c in cosines]
            emap = torch.stack(emap, dim=0).sum(0)
        elif mode == "selfattention":
            emap = attn[0, 0, 0, 1:].detach().reshape(*map_size)
        elif mode == "genericattention":
            emap = gradeclip.genericattention(
                image_preprocessed, text_inputs["input_ids"], model,
                text_inputs.get("attention_mask"),
            ).sum(0)
        elif mode == "attentionrollout":
            with torch.no_grad():
                attn_layers = model.vision_model(image_preprocessed, output_attentions=True).attentions
            matrices = [a.mean(1) for a in attn_layers]  # [B, N, N] per layer
            emap = gradeclip.rolloutattention(matrices)[0]

    elif "siglip" in str(type(model)):
        text_inputs    = processor(text=[input_text], return_tensors="pt", padding="max_length").to(device)
        text_embedding = F.normalize(model.get_text_features(**text_inputs), dim=-1)

        if mode in ("gradesiglip", "masksiglip", "gradcam", "selfattention"):
            image_emb, v_final, last_input, v, q_out, k_out, attn, att_output, map_size = \
                gradesiglip.siglip_encode_image_dense(image_preprocessed, model)
            cosines = (F.normalize(image_emb, dim=-1) @ text_embedding.T)[0]

        if mode == "gradesiglip":
            emap = [gradesiglip.gradesiglip_image(c, q_out, k_out, v_final, map_size) for c in cosines]
            emap = torch.stack(emap, dim=0).sum(0)
        elif mode == "masksiglip":
            emap = gradesiglip.masksiglip(text_embedding[0], v_final, model, map_size)
        elif mode == "gradcam":
            emap = [gradesiglip.gradcam(c, last_input, map_size) for c in cosines]
            emap = torch.stack(emap, dim=0).sum(0)
        elif mode == "selfattention":
            emap = attn[0, 0].mean(0).detach().reshape(*map_size)
        elif mode == "genericattention":
            emap = gradesiglip.genericattention(
                image_preprocessed, text_inputs["input_ids"], model,
                text_inputs.get("attention_mask"),
            ).squeeze(0)
        elif mode == "attentionrollout":
            with torch.no_grad():
                attn_layers, _ = gradesiglip._extract_vision_attentions(model, image_preprocessed)
            matrices = [a.mean(1) for a in attn_layers]  # [B, N, N] per layer
            emap = gradesiglip.rolloutattention(matrices)[0]

    return emap.detach().cpu().numpy()


class Metagame:
    def __init__(self, model, processor, n_image_tokens, mode="gradeclip"):
        self.model = model
        self.processor = processor
        self.n_image_tokens = n_image_tokens
        self.mode = mode
        self.first_order_explanations = None
        self.second_order_explanations = None

    def explain(self, input_image, input_text):
        input_text_words = input_text.split(" ")
        n_text_tokens = len(input_text_words)
        player_indices = tuple(range(n_text_tokens))
        player_sets = shapiq.utils.powerset(player_indices)

        first_order_explanations = {}
        for s in player_sets:
            input_text_subset = " ".join([input_text_words[i] for i in s])
            first_order_explanations[s] = explain_image(input_image, input_text_subset, self.model, self.processor, mode=self.mode)
        self.first_order_explanations = first_order_explanations

        second_order_explanations = {}
        for p in player_indices:
            shapley_value = 0
            for s in self.first_order_explanations.keys():
                if p not in s:
                    s_with_p = tuple(sorted(s + (p,)))
                    s_len = len(s)
                    w = (math.factorial(s_len) * math.factorial(n_text_tokens - s_len - 1)) / math.factorial(n_text_tokens)
                    v_s_with_p = self.first_order_explanations[s_with_p]
                    v_s = self.first_order_explanations[s]
                    shapley_value += w * (v_s_with_p - v_s)
            second_order_explanations[p] = shapley_value
        self.second_order_explanations = second_order_explanations

        meta_values = {}
        for p in player_indices:
            flat_shapley = second_order_explanations[p].reshape(-1)
            for i_token in range(self.n_image_tokens):
                val = flat_shapley[i_token]
                meta_values[(i_token, self.n_image_tokens + p)] = val.item() if hasattr(val, 'item') else val

        for i in range(self.n_image_tokens + n_text_tokens):
            meta_values[(i, )] = 0

        iv = shapiq.InteractionValues(
            values=meta_values,
            index=self.mode,
            n_players=self.n_image_tokens + n_text_tokens,
            max_order=2,
            min_order=1,
            baseline_value=0
        )

        return iv


class MetagameTokens:
    """Same as Metagame, but players are model tokens (from processor.tokenizer) instead of whitespace-split words."""
    def __init__(self, model, processor, n_image_tokens, mode="gradeclip"):
        self.model = model
        self.processor = processor
        self.n_image_tokens = n_image_tokens
        self.mode = mode
        self.first_order_explanations = None
        self.second_order_explanations = None
        self.text_tokens = None

    def explain(self, input_image, input_text):
        tokenizer = self.processor.tokenizer
        token_ids = tokenizer(input_text, add_special_tokens=False)["input_ids"]
        self.text_tokens = [tokenizer.decode([tid]) for tid in token_ids]
        n_text_tokens = len(token_ids)
        player_indices = tuple(range(n_text_tokens))
        player_sets = shapiq.utils.powerset(player_indices)

        first_order_explanations = {}
        for s in player_sets:
            input_text_subset = tokenizer.decode(
                [token_ids[i] for i in s], skip_special_tokens=True
            )
            first_order_explanations[s] = explain_image(input_image, input_text_subset, self.model, self.processor, mode=self.mode)
        self.first_order_explanations = first_order_explanations

        second_order_explanations = {}
        for p in player_indices:
            shapley_value = 0
            for s in self.first_order_explanations.keys():
                if p not in s:
                    s_with_p = tuple(sorted(s + (p,)))
                    s_len = len(s)
                    w = (math.factorial(s_len) * math.factorial(n_text_tokens - s_len - 1)) / math.factorial(n_text_tokens)
                    v_s_with_p = self.first_order_explanations[s_with_p]
                    v_s = self.first_order_explanations[s]
                    shapley_value += w * (v_s_with_p - v_s)
            second_order_explanations[p] = shapley_value
        self.second_order_explanations = second_order_explanations

        meta_values = {}
        for p in player_indices:
            flat_shapley = second_order_explanations[p].reshape(-1)
            for i_token in range(self.n_image_tokens):
                val = flat_shapley[i_token]
                meta_values[(i_token, self.n_image_tokens + p)] = val.item() if hasattr(val, 'item') else val

        for i in range(self.n_image_tokens + n_text_tokens):
            meta_values[(i, )] = 0

        iv = shapiq.InteractionValues(
            values=meta_values,
            index=self.mode,
            n_players=self.n_image_tokens + n_text_tokens,
            max_order=2,
            min_order=1,
            baseline_value=0
        )

        return iv
