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


def _make_causal_mask(bsz, seq_len, dtype, device):
    """Upper-triangular additive causal mask (0 = keep, -inf = masked)."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), dtype=dtype, device=device), diagonal=1)
    return mask.unsqueeze(0).expand(bsz, -1, -1)


def clip_encode_image_dense(pixel_values, clip_model):
    """CLIP vision encoder decomposed at the last layer.

    Applies post_layernorm and visual_projection to ALL tokens so that
    outputs[:, 0] is the CLS image embedding.

    Args:
        pixel_values: [B, 3, H, W]
        clip_model:   CLIPModel
    Returns (9-tuple):
        outputs:     [B, N+1, D_proj]  all projected tokens; [:, 0] = CLS
        v_final:     [B, N, D_proj]    patch value features bypassing attention (for maskclip)
        last_input:  [B, N+1, D]       input to last encoder layer (for gradcam)
        v_lnd:       [N+1, B, D]       raw value features
        q_out:       [N+1, B, D]       q projected through out_proj
        k_out:       [N+1, B, D]       k projected through out_proj
        attn_weights:[B, 1, N+1, N+1]
        att_output:  [N+1, B, D]       attention output before out_proj
        map_size:    (H_patches, W_patches)
    """
    model = clip_model.vision_model
    hidden_states = model.pre_layrnorm(model.embeddings(pixel_values))
    num_patches = hidden_states.shape[1] - 1
    map_size = (int(num_patches ** 0.5),) * 2

    for layer in model.encoder.layers[:-1]:
        hidden_states = layer(hidden_states, attention_mask=None, causal_attention_mask=None)[0]

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
    all_projected = clip_model.visual_projection(model.post_layernorm(hidden_states))

    with torch.no_grad():
        qkv = torch._C._nn.linear(
            torch.stack((q_lnd, k_lnd, v_lnd)), attn_module.out_proj.weight, attn_module.out_proj.bias,
        )
        q_out, k_out = qkv[0], qkv[1]
        v_hs    = last_input + qkv[2].permute(1, 0, 2)
        v_hs    = v_hs + last_layer.mlp(last_layer.layer_norm2(v_hs))
        v_final = clip_model.visual_projection(model.post_layernorm(v_hs))[:, 1:]

    return all_projected, v_final, last_input, v_lnd, q_out, k_out, attn_weights, attn_output_lnd, map_size


def clip_encode_text_dense(input_ids, clip_model, n=8):
    """CLIP text encoder decomposed at the last n layers.

    Args:
        input_ids:  [B, seq_len]
        clip_model: CLIPModel
        n:          number of last layers to decompose
    Returns:
        text_embedding: [B, D_proj]
        (q_outs, k_outs, vs): per-layer intermediates
        attns:        attention weights per layer
        attn_outputs: attention outputs per layer
    """
    text_model    = clip_model.text_model
    hidden_states = text_model.embeddings(input_ids=input_ids)
    bsz, seq_len  = input_ids.shape
    causal_mask   = _make_causal_mask(bsz, seq_len, hidden_states.dtype, hidden_states.device)

    for layer in text_model.encoder.layers[:-n]:
        hidden_states = layer(hidden_states, attention_mask=None, causal_attention_mask=causal_mask)[0]

    attns, attn_outputs, vs, q_outs, k_outs = [], [], [], [], []
    x_in = hidden_states
    for layer in text_model.encoder.layers[-n:]:
        attn_module = layer.self_attn
        x_normed    = layer.layer_norm1(x_in)
        q_lnd = attn_module.q_proj(x_normed).permute(1, 0, 2)
        k_lnd = attn_module.k_proj(x_normed).permute(1, 0, 2)
        v_lnd = attn_module.v_proj(x_normed).permute(1, 0, 2)

        attn_output_lnd, attn_w = _attention_layer_single_head(q_lnd, k_lnd, v_lnd, attn_mask=causal_mask)
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

    x   = text_model.final_layer_norm(x_in)
    eos = input_ids.argmax(dim=-1)
    return clip_model.text_projection(x[torch.arange(x.shape[0]), eos]), (q_outs, k_outs, vs), attns, attn_outputs


def gradeclip_image(c, q_out, k_out, v, att_output, map_size, withksim=True):
    """Grad-ECLIP image explanation map (CLIP).

    Channel weight: CLS gradient. Spatial weight: loosened CLS-to-patch cosine similarity.

    Args:
        c:          scalar similarity score
        q_out:      [N+1, B, D]  q projected through out_proj
        k_out:      [N+1, B, D]  k projected through out_proj
        v:          [N+1, B, D]  raw value features
        att_output: [N+1, B, D]  attention output before out_proj
        map_size:   (H, W)
        withksim:   use loosened spatial weight
    Returns:
        emap: [H, W]
    """
    grad_cls = torch.autograd.grad(c, att_output, retain_graph=True)[0].detach()[:1, 0, :]
    if withksim:
        w = (F.normalize(q_out[:1, 0, :], dim=-1) * F.normalize(k_out[1:, 0, :], dim=-1)).sum(-1)
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        emap = F.relu_((grad_cls * v[1:, 0, :] * w[:, None]).detach().sum(-1))
    else:
        emap = F.relu_((grad_cls * v[1:, 0, :]).detach().sum(-1))
    return emap.reshape(*map_size)




def maskclip(txt_feats, v_final, k_out, map_size):
    """MaskCLIP explanation map (CLIP).

    Local text-patch cosine similarity weighted by CLS-to-patch key similarity.

    Args:
        txt_feats: [D_proj] or [1, D_proj]
        v_final:   [B, N, D_proj]  from clip_encode_image_dense
        k_out:     [N+1, B, D]     position 0 is the CLS key
        map_size:  (H, W)
    Returns:
        emap: [H, W]
    """
    sim = (F.normalize(v_final[0], dim=-1) @ F.normalize(txt_feats.reshape(1, -1), dim=-1).T).squeeze(-1)
    w   = (F.normalize(k_out[1:, 0, :], dim=-1) @ F.normalize(k_out[0, 0, :], dim=-1).unsqueeze(-1)).squeeze(-1)
    return (sim * w).detach().reshape(*map_size)


def gradcam(c, layer_feat, map_size):
    """Grad-CAM using the last encoder layer input (CLIP).

    Args:
        c:          scalar similarity score
        layer_feat: [B, N+1, D]  last_input from clip_encode_image_dense
        map_size:   (H, W)
    Returns:
        cam: [H, W]
    """
    grad = torch.autograd.grad(c, layer_feat, retain_graph=True)[0].detach()
    return F.relu_((grad[0].mean(0) * layer_feat.detach()[0, 1:]).sum(-1)).reshape(*map_size)


def genericattention(pixel_values, input_ids, clip_model, attention_mask=None, start_layer=-1, flag="image"):
    """Generic Attention-Model Explainability (CLIP).

    Uses HuggingFace output_attentions — requires attn_implementation="eager".

    Args:
        pixel_values:   [B, 3, H, W]
        input_ids:      [B, seq_len]
        clip_model:     CLIPModel
        attention_mask: [B, seq_len] or None
        start_layer:    first layer to include (-1 = last layer only)
        flag:           "image" or "text"
    Returns:
        image: [B, H, W]            CLS-to-patch relevance
        text:  [B, seq_len, seq_len]
    """
    outputs = clip_model(
        pixel_values=pixel_values, input_ids=input_ids,
        attention_mask=attention_mask, output_attentions=True,
    )
    logits     = outputs.logits_per_image
    batch_size = logits.shape[0]
    one_hot    = torch.zeros_like(logits)
    one_hot[torch.arange(batch_size), torch.arange(batch_size)] = 1
    one_hot_sum = (one_hot * logits).sum()
    clip_model.zero_grad()

    attentions = outputs.vision_model_output.attentions if flag == "image" else outputs.text_model_output.attentions
    sl         = len(attentions) - 1 if start_layer == -1 else start_layer
    num_tokens = attentions[0].shape[-1]
    R = torch.eye(num_tokens, dtype=attentions[0].dtype, device=logits.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    for i, attn in enumerate(attentions):
        if i < sl:
            continue
        grad = torch.autograd.grad(one_hot_sum, [attn], retain_graph=True)[0].detach()
        R = R + torch.bmm((grad.mean(1) * attn.detach().mean(1)).clamp(min=0), R)

    if flag == "image":
        n = int(R[:, 0, 1:].shape[-1] ** 0.5)
        return R[:, 0, 1:].reshape(batch_size, n, n)
    return R


def rolloutattention(all_layer_matrices, start_layer=0, flag="image"):
    """Attention rollout across layers (CLIP).

    Args:
        all_layer_matrices: list of [B, N, N] attention matrices
        start_layer:        first layer to include
        flag:               "image" or "text"
    Returns:
        image: [B, H, W]  CLS-to-patch rollout
        text:  [B, seq_len, seq_len]
    """
    B, N = all_layer_matrices[0].shape[0], all_layer_matrices[0].shape[-1]
    eye  = torch.eye(N, device=all_layer_matrices[0].device).unsqueeze(0).expand(B, -1, -1)
    mats = [(m + eye) / (m + eye).sum(dim=-1, keepdim=True) for m in all_layer_matrices]
    joint = mats[start_layer]
    for m in mats[start_layer + 1:]:
        joint = m.bmm(joint)
    if flag == "image":
        n = int(joint[:, 0, 1:].shape[-1] ** 0.5)
        return joint[:, 0, 1:].reshape(B, n, n)
    return joint


def _shapiq_jet_lut():
    """Diverging LUT (256, 3) in RGB: shapiq blue (−1) → black (0) → shapiq red (+1)."""
    BLUE_RGB = ( 30, 136, 229)
    RED_RGB  = (255,  13,  87)
    stops = [
        (0.0, BLUE_RGB),
        (0.5, (0, 0, 0)),
        (1.0, RED_RGB),
    ]
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    rgb = np.array([s[1] for s in stops], dtype=np.float32)
    grid = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    lut = np.stack([np.interp(grid, xs, rgb[:, c]) for c in range(3)], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8)


def save_explanation(image, explanation, path, tag, theta=0.5, quantile=0.99, scale=None, png=True, npy=False):
    """Overlay a diverging heatmap (blue→black→red, −1 to +1) on an image and save.

    The explanation is rescaled symmetrically by `scale` (if provided), otherwise
    by the `quantile`-th quantile of |explanation| (default 0.99). Values above
    the scale saturate at ±1. Pass an explicit `scale` to make multiple
    heatmaps comparable on a shared colormap.

    Args:
        image:       [3, H, W] float numpy array (e.g. pixel_values[0])
        explanation: [h, w] float numpy array (patch-grid sized; resized internally)
        path:        output directory
        tag:         filename stem
        theta:       image/heatmap blend weight (1.0 = image only, 0.0 = heatmap only)
        quantile:    quantile of |explanation| for per-map rescaling (ignored if `scale` is set)
        scale:       optional shared scale; if None, computed from `quantile`
    """
    img_np = np.transpose(image, (1, 2, 0))  # [3, H, W] -> [H, W, 3]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    img_np = (img_np * 255).astype(np.uint8)

    if scale is None:
        scale = float(np.quantile(np.abs(explanation), quantile)) + 1e-8
    emap = np.clip(explanation / scale, -1.0, 1.0)          # [-1, 1]
    emap = cv2.resize(emap, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    idx  = ((emap + 1.0) * 0.5 * 255.0).astype(np.uint8)     # 0 -> 128 (black)
    color = _shapiq_jet_lut()[idx]                             # (H, W, 3) RGB
    c_ret = np.clip(img_np * theta + color * (1 - theta), 0, 255).astype(np.uint8)
    if npy:
        np.save(os.path.join(path, "explanation_{}.npy".format(tag)), emap)
    if png:
        cv2.imwrite(os.path.join(path, "explanation_{}.png".format(tag)), c_ret[:, :, ::-1])


def save_explanation3(image, explanation, path, tag, quantile=0.99, scale=None, background=None, png=True, npy=False):
    """Transparency-based heatmap following src/plot.py `interactions_to_heatmap`.

    Each patch gets a solid color — shapiq RED for value >= 0, shapiq BLUE for
    value < 0 — blended onto the background with `alpha = |value| / scale`
    clipped to [0, 1]. Low-magnitude patches are transparent (background shows
    through); high-magnitude patches cover it entirely.

    Args:
        image:       [3, H, W] float numpy array (e.g. pixel_values[0]);
                     used as the background when `background` is None
        explanation: [h, w] float numpy array (patch-grid sized; resized internally)
        path:        output directory
        tag:         filename stem
        quantile:    quantile of |explanation| used as the alpha denominator if
                     `scale` is not provided (default 0.99)
        scale:       optional shared scale; if None, computed from `quantile`
        background:  None = use `image` underneath; int 0–255 = flat gray canvas
    """
    RED_RGB  = np.array([255,  13,  87], dtype=np.float32)
    BLUE_RGB = np.array([ 30, 136, 229], dtype=np.float32)

    img_np = np.transpose(image, (1, 2, 0))
    if background is None:
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        img_np = (img_np * 255).astype(np.float32)
    else:
        img_np = np.full_like(img_np, float(background), dtype=np.float32)

    if scale is None:
        scale = float(np.quantile(np.abs(explanation), quantile)) + 1e-8

    alpha = np.clip(np.abs(explanation) / scale, 0.0, 1.0).astype(np.float32)
    sign  = (explanation >= 0).astype(np.float32)

    alpha_full = cv2.resize(alpha, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)[..., None]
    sign_full  = cv2.resize(sign,  (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST)[..., None]
    color      = np.where(sign_full >= 0.5, RED_RGB, BLUE_RGB)

    out = np.ascontiguousarray(
        (img_np * (1.0 - alpha_full) + color * alpha_full).clip(0, 255).astype(np.uint8)
    )

    if npy:
        np.save(os.path.join(path, "explanation_{}.npy".format(tag)), explanation)
    if png:
        cv2.imwrite(os.path.join(path, "explanation_{}.png".format(tag)), out[:, :, ::-1])