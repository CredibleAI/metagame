import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def _attention_layer_single_head(q, k, v, attn_mask=None):
    """Single-head scaled dot product attention (LND format).

    Args:
        q, k, v:   [seq_len, batch, embed_dim]
        attn_mask: optional additive mask [batch, seq_len, seq_len]
    Returns:
        attn_output:  [seq_len, batch, embed_dim]
        attn_weights: [batch, 1, seq_q, seq_k]
    """
    embed_dim = q.shape[2]
    q_t = (q * (float(embed_dim) ** -0.5)).transpose(0, 1)
    k_t, v_t = k.transpose(0, 1), v.transpose(0, 1)
    attn_weights = torch.bmm(q_t, k_t.transpose(1, 2))
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask
    attn_weights = F.softmax(attn_weights, dim=-1)
    return torch.bmm(attn_weights, v_t).transpose(0, 1), attn_weights.unsqueeze(1)


def siglip_encode_image_dense(pixel_values, siglip_model):
    """SigLIP-2 vision encoder decomposed at the last layer and MAP head.

    Args:
        pixel_values: [B, 3, H, W]
        siglip_model: SiglipModel
    Returns (9-tuple):
        image_embedding: [B, D]       final image embedding
        v_final:         [B, N, D]    post-layernorm encoder output (for masksiglip)
        last_input:      [B, N, D]    input to last encoder layer (for gradcam)
        v_lnd:           [N, B, D]    raw value features
        q_out:           [N, B, D]    q projected through out_proj
        k_out:           [N, B, D]    k projected through out_proj
        attn_weights:    [B, 1, N, N]
        att_output:      [N, B, D]    attention output before out_proj
        map_size:        (H_patches, W_patches)
    """
    model     = siglip_model.vision_model
    hidden_states = model.embeddings(pixel_values)
    patch_size    = model.embeddings.patch_size
    map_size      = (pixel_values.shape[-2] // patch_size, pixel_values.shape[-1] // patch_size)

    for layer in model.encoder.layers[:-1]:
        hidden_states = layer(hidden_states, attention_mask=None)

    last_input  = hidden_states.detach().requires_grad_(True)
    hidden_states = last_input
    last_layer  = model.encoder.layers[-1]
    attn_module = last_layer.self_attn
    x_normed    = last_layer.layer_norm1(hidden_states)

    q_lnd = attn_module.q_proj(x_normed).permute(1, 0, 2)
    k_lnd = attn_module.k_proj(x_normed).permute(1, 0, 2)
    v_lnd = attn_module.v_proj(x_normed).permute(1, 0, 2)
    attn_output_lnd, attn_weights = _attention_layer_single_head(q_lnd, k_lnd, v_lnd)

    hidden_states = hidden_states + attn_module.out_proj(attn_output_lnd.permute(1, 0, 2))
    hidden_states = hidden_states + last_layer.mlp(last_layer.layer_norm2(hidden_states))
    hidden_states = model.post_layernorm(hidden_states)

    head  = model.head
    probe = head.probe.repeat(hidden_states.shape[0], 1, 1)
    map_attn_out, _ = head.attention(probe, hidden_states, hidden_states)
    image_embedding = (map_attn_out + head.mlp(head.layernorm(map_attn_out)))[:, 0]

    with torch.no_grad():
        qkv   = torch._C._nn.linear(
            torch.stack((q_lnd, k_lnd, v_lnd)), attn_module.out_proj.weight, attn_module.out_proj.bias,
        )
        q_out, k_out = qkv[0], qkv[1]

    return image_embedding, hidden_states, last_input, v_lnd, q_out, k_out, attn_weights, attn_output_lnd, map_size


def siglip_encode_text_dense(input_ids, attention_mask, siglip_model, n=8):
    """SigLIP-2 text encoder decomposed at the last n layers.

    Args:
        input_ids:      [B, seq_len]
        attention_mask: [B, seq_len]
        siglip_model:   SiglipModel
        n:              number of last layers to decompose
    Returns:
        text_embedding: [B, D]
        (q_outs, k_outs, vs): per-layer intermediates
        attns:          attention weights per layer
        attn_outputs:   attention outputs per layer
    """
    text_model    = siglip_model.text_model
    hidden_states = text_model.embeddings(input_ids=input_ids)

    from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
    expanded_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype) if attention_mask is not None else None

    for layer in text_model.encoder.layers[:-n]:
        hidden_states = layer(hidden_states, attention_mask=expanded_mask)

    attns, attn_outputs, vs, q_outs, k_outs = [], [], [], [], []
    x_in = hidden_states
    for layer in text_model.encoder.layers[-n:]:
        attn_module = layer.self_attn
        x_normed    = layer.layer_norm1(x_in)
        q_lnd = attn_module.q_proj(x_normed).permute(1, 0, 2)
        k_lnd = attn_module.k_proj(x_normed).permute(1, 0, 2)
        v_lnd = attn_module.v_proj(x_normed).permute(1, 0, 2)

        # SigLIP-2 text uses no causal mask
        attn_output_lnd, attn_w = _attention_layer_single_head(q_lnd, k_lnd, v_lnd)
        attn_outputs.append(attn_output_lnd)
        attns.append(attn_w)
        vs.append(v_lnd)

        x    = x_in + attn_module.out_proj(attn_output_lnd.permute(1, 0, 2))
        x_in = x + layer.mlp(layer.layer_norm2(x))

        with torch.no_grad():
            qk = torch._C._nn.linear(
                torch.stack((q_lnd, k_lnd)), attn_module.out_proj.weight, attn_module.out_proj.bias,
            )
            q_outs.append(qk[0])
            k_outs.append(qk[1])

    x = text_model.final_layer_norm(x_in)
    return text_model.head(x[:, -1, :]), (q_outs, k_outs, vs), attns, attn_outputs


def gradesiglip_image(c, q_out, k_out, v_final, map_size, withksim=True):
    """Grad-ECLIP image explanation map (SigLIP-2).

    Uses v_final (post-layernorm features seen by MAP head) instead of raw
    attention outputs.  Mean gradient across patches serves as global channel
    weight — the SigLIP analog of CLIP's CLS gradient.
    Spatial weight: mean-q to patch cosine similarity from last encoder layer.

    Args:
        c:          scalar similarity score
        q_out:      [N, B, D]  q projected through out_proj
        k_out:      [N, B, D]  k projected through out_proj
        v_final:    [B, N, D]  post-layernorm encoder output (MAP head input)
        map_size:   (H, W)
        withksim:   use loosened spatial weight
    Returns:
        emap: [H, W]
    """
    grad = torch.autograd.grad(c, v_final, retain_graph=True)[0].detach()  # [B, N, D]
    channel_w = grad.mean(1)  # [B, D] — global channel importance
    if withksim:
        w = (F.normalize(k_out[:, 0, :], dim=-1) * F.normalize(q_out.mean(0)[0, :], dim=-1)).sum(-1)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        emap = F.relu_((channel_w[0] * v_final.detach()[0] * w[:, None]).sum(-1))
    else:
        emap = F.relu_((channel_w[0] * v_final.detach()[0]).sum(-1))
    return emap.reshape(*map_size)



def masksiglip(txt_feats, v_final, siglip_model, map_size):
    """MaskCLIP explanation map (SigLIP-2).

    Per-patch embeddings are obtained by projecting each patch through the MAP
    head's value projection, out_proj, and MLP+LayerNorm — the same path the
    MAP attention output takes to become the image embedding.  This places each
    patch feature in the joint text-image embedding space, enabling direct
    cosine similarity with the text embedding (analogous to CLIP's approach of
    projecting value features through out_proj → MLP → LN → visual.proj).

    The per-patch embedding for patch i corresponds to treating patch i as the
    sole key/value in the MAP cross-attention (attention weight = 1), so:
        patch_emb[i] = out_proj(V_proj[i]) + MLP(LN(out_proj(V_proj[i])))
    where V_proj[i] = in_proj_weight[2D:] @ v_final[i].

    Args:
        txt_feats:    [D] or [1, D]  normalized text embedding
        v_final:      [B, N, D]  post-layernorm encoder output (MAP head input)
        siglip_model: SiglipModel  used to access vision_model.head
        map_size:     (H, W)
    Returns:
        emap: [H, W]
    """
    head = siglip_model.vision_model.head
    mha  = head.attention
    W, b = mha.in_proj_weight, mha.in_proj_bias
    D    = v_final.shape[-1]
    with torch.no_grad():
        V     = F.linear(v_final, W[2*D:], b[2*D:] if b is not None else None)  # [B, N, D]
        V_out = F.linear(V, mha.out_proj.weight, mha.out_proj.bias)             # [B, N, D]
        patch_embs = V_out + head.mlp(head.layernorm(V_out))                    # [B, N, D]
        sim = (F.normalize(patch_embs[0], dim=-1) @
               F.normalize(txt_feats.reshape(1, -1), dim=-1).T).squeeze(-1)     # [N]
    return sim.detach().reshape(*map_size)


def gradcam(c, layer_feat, map_size):
    """Grad-CAM using the last encoder layer input (SigLIP-2).

    Args:
        c:          scalar similarity score
        layer_feat: [B, N, D]  last_input from siglip_encode_image_dense
        map_size:   (H, W)
    Returns:
        cam: [H, W]
    """
    grad = torch.autograd.grad(c, layer_feat, retain_graph=True)[0].detach()
    return F.relu_((grad[0].mean(0) * layer_feat.detach()[0]).sum(-1)).reshape(*map_size)


def _extract_vision_attentions(siglip_model, pixel_values):
    """Run vision encoder manually, collecting per-layer attention weights.

    SigLIP-2's encoder discards attention weights, so we re-run each layer's
    self-attention to capture them.  Requires attn_implementation="eager" set
    on model.config.vision_config before calling.

    Args:
        siglip_model: SiglipModel
        pixel_values: [B, 3, H, W]
    Returns:
        attentions: list of [B, num_heads, N, N] tensors (one per layer)
        hidden_states: final encoder hidden states [B, N, D]
    """
    model = siglip_model.vision_model
    hidden_states = model.embeddings(pixel_values)
    attentions = []
    for layer in model.encoder.layers:
        attn_module = layer.self_attn
        residual = hidden_states
        x_normed = layer.layer_norm1(hidden_states)
        attn_output, attn_weights = attn_module(hidden_states=x_normed, attention_mask=None, output_attentions=True)
        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = residual + layer.mlp(layer.layer_norm2(hidden_states))
        attentions.append(attn_weights)
    return attentions, hidden_states


def _map_attention_forward(head, probe, encoder_output):
    """Manually compute MAP head cross-attention so weights stay in the autograd graph.

    PyTorch's nn.MultiheadAttention returns attention weights that are not
    connected to the output's computation graph, so gradients cannot be computed
    through them.  This function replicates the same computation explicitly.

    Args:
        head:           SiglipMultiheadAttentionPoolingHead
        probe:          [B, 1, D]  learnable probe (query)
        encoder_output: [B, N, D]  post-layernorm encoder features (key/value)
    Returns:
        attn_output:  [B, 1, D]  output before MLP
        attn_weights: [B, num_heads, 1, N]  per-head attention weights in graph
    """
    mha = head.attention
    B, N, D = encoder_output.shape
    num_heads = mha.num_heads
    head_dim = D // num_heads
    W, b = mha.in_proj_weight, mha.in_proj_bias  # [3D, D] and [3D]

    q = F.linear(probe,           W[:D],    b[:D]    if b is not None else None)  # [B, 1, D]
    k = F.linear(encoder_output,  W[D:2*D], b[D:2*D] if b is not None else None)  # [B, N, D]
    v = F.linear(encoder_output,  W[2*D:],  b[2*D:]  if b is not None else None)  # [B, N, D]

    q = q.reshape(B, 1, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, 1, d]
    k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, d]
    v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, d]

    attn_weights = F.softmax(q @ k.transpose(-2, -1) * (head_dim ** -0.5), dim=-1)  # [B, H, 1, N]
    attn_output = (attn_weights @ v).transpose(1, 2).reshape(B, 1, D)               # [B, 1, D]
    attn_output = F.linear(attn_output, mha.out_proj.weight, mha.out_proj.bias)
    return attn_output, attn_weights


def genericattention(pixel_values, input_ids, siglip_model, attention_mask, start_layer=-1, flag="image"):
    """Generic Attention-Model Explainability (SigLIP-2).

    Applies the Generic Attention update rule (Chefer et al. 2021) to all
    encoder self-attention layers AND to the MAP cross-attention head.
    SigLIP-2 has no CLS token; the MAP probe plays its role, so the
    gradient-weighted MAP attention weights serve as the final aggregator
    (analogous to R[:, 0, 1:] in CLS-based models).

    Requires model.config.vision_config._attn_implementation = "eager".

    Args:
        pixel_values:   [B, 3, H, W]
        input_ids:      [B, seq_len]
        siglip_model:   SiglipModel
        attention_mask: [B, seq_len]
        start_layer:    first encoder layer to include (-1 = last layer only)
        flag:           "image" or "text"
    Returns:
        image: [B, H, W]
        text:  [B, seq_len, seq_len]
    """
    attentions, encoder_output = _extract_vision_attentions(siglip_model, pixel_values)

    # Forward through post-layernorm and MAP head (manually, to keep attn weights in graph)
    hidden_states = siglip_model.vision_model.post_layernorm(encoder_output)
    head = siglip_model.vision_model.head
    probe = head.probe.repeat(hidden_states.shape[0], 1, 1)
    map_attn_out, map_attn_weights = _map_attention_forward(head, probe, hidden_states)
    image_emb = (map_attn_out + head.mlp(head.layernorm(map_attn_out)))[:, 0]
    image_features = F.normalize(image_emb, dim=-1)

    text_outputs = siglip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    text_features = F.normalize(text_outputs.pooler_output, dim=-1)

    logits = image_features @ text_features.T * siglip_model.logit_scale.exp() + siglip_model.logit_bias
    batch_size = logits.shape[0]
    one_hot = torch.zeros_like(logits)
    one_hot[torch.arange(batch_size), torch.arange(batch_size)] = 1
    one_hot_sum = (one_hot * logits).sum()
    siglip_model.zero_grad()

    # Generic Attention rollout through encoder layers
    sl = len(attentions) - 1 if start_layer == -1 else start_layer
    num_tokens = attentions[0].shape[-1]
    R = torch.eye(num_tokens, dtype=attentions[0].dtype, device=logits.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    for i, attn in enumerate(attentions):
        if i < sl:
            continue
        grad = torch.autograd.grad(one_hot_sum, [attn], retain_graph=True)[0].detach()
        R = R + torch.bmm((grad.mean(1) * attn.detach().mean(1)).clamp(min=0), R)

    if flag == "image":
        n = int(num_tokens ** 0.5)
        # Apply the Generic Attention update to the MAP cross-attention:
        # Ā_map = mean_heads((∇A_map ⊙ A_map)^+)  [B, 1, N]
        # final_rel = Ā_map @ R  [B, 1, N] — analogous to R[:, 0, 1:] in CLS models
        grad_map = torch.autograd.grad(one_hot_sum, [map_attn_weights], retain_graph=False)[0].detach()
        A_bar_map = (grad_map * map_attn_weights.detach()).clamp(min=0).mean(1)  # [B, 1, N]
        final_rel = torch.bmm(A_bar_map, R)  # [B, 1, N]
        return final_rel.squeeze(1).reshape(batch_size, n, n)
    return R


def rolloutattention(all_layer_matrices, start_layer=0, flag="image"):
    """Attention rollout across layers (SigLIP-2).

    Args:
        all_layer_matrices: list of [B, N, N] attention matrices
        start_layer:        first layer to include
        flag:               "image" or "text"
    Returns:
        image: [B, H, W]
        text:  [B, seq_len, seq_len]
    """
    B, N = all_layer_matrices[0].shape[0], all_layer_matrices[0].shape[-1]
    eye  = torch.eye(N, device=all_layer_matrices[0].device).unsqueeze(0).expand(B, -1, -1)
    mats = [(m + eye) / (m + eye).sum(dim=-1, keepdim=True) for m in all_layer_matrices]
    joint = mats[start_layer]
    for m in mats[start_layer + 1:]:
        joint = m.bmm(joint)
    if flag == "image":
        n = int(N ** 0.5)
        return joint.mean(1).reshape(B, n, n)
    return joint