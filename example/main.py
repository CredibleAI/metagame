"""Single-block transformer on a + b / a − b. Trains, then renders the
panels of paper Figure 1 + App. Figures 5, 6."""

import argparse
import itertools
import math
import os
import random
import re
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["axes.unicode_minus"] = True
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "XCharter"
matplotlib.rcParams["mathtext.bf"] = "XCharter:bold"
matplotlib.rcParams["mathtext.it"] = "XCharter:italic"
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seeds(seed_value=42):
    """Set seeds for random, numpy, and torch."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Vocabulary & sequence layout
# =============================================================================

UNK, PLUS, MINUS, EQ = "<unk>", "+", "-", "="
VOCAB = [UNK] + [str(d) for d in range(-9, 10)] + [PLUS, MINUS, EQ]
TOK2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOK = {i: t for t, i in TOK2ID.items()}
VOCAB_SIZE = len(VOCAB)
UNK_ID = TOK2ID[UNK]
SEQ_LEN = 4
EXPLAIN_POSITIONS = (0, 1, 2)            # a, op, b
CONSTANT_POSITIONS = (3,)                # '=' — never masked, never attributed
OPS = (PLUS, MINUS)


def tokenize(a, b, op):
    assert op in OPS
    t = lambda x: UNK if x is None else str(x)
    return [TOK2ID[tok] for tok in [t(a), op, t(b), EQ]]


def evaluate(a, b, op):
    assert op in OPS
    a, b = (x if x is not None else 0 for x in (a, b))
    return (a + b) if op == PLUS else (a - b)


def decode(ids, pretty=False):
    out = [ID2TOK[int(i)] for i in ids]
    return ["∅" if t == UNK else t for t in out] if pretty else out


# =============================================================================
# Dataset
# =============================================================================

def _records_for_triples(pairs):
    records = []
    for a, b in pairs:
        for op in OPS:
            base_ids = tokenize(a, b, op)
            records.append((base_ids, evaluate(a, b, op)))
            for pos_idx in range(2):
                vals = [a, b]; vals[pos_idx] = None
                records.append((tokenize(*vals, op), evaluate(*vals, op)))
        op_masked = tokenize(a, b, PLUS)
        op_masked[1] = UNK_ID
        # mean over op of (a+b, a-b) = a
        records.append((op_masked, a))
    return records


def _to_arrays(records):
    X = np.asarray([r[0] for r in records], dtype=np.int64)
    y = np.asarray([r[1] for r in records], dtype=np.float32)
    return X, y


def build_dataset(seed=0, holdout_per_a=2, val_frac=0.1):
    """Stratify holdout by `a`: each a ∈ [-9, 9] keeps `holdout_per_a` test b's."""
    rng = np.random.default_rng(seed)
    holdout_pairs, in_dist_pairs = [], []
    for a in range(-9, 10):
        bs = list(range(-9, 10))
        rng.shuffle(bs)
        holdout_pairs.extend((a, b) for b in bs[:holdout_per_a])
        in_dist_pairs.extend((a, b) for b in bs[holdout_per_a:])
    rng.shuffle(in_dist_pairs)
    rng.shuffle(holdout_pairs)
    X_in, y_in = _to_arrays(_records_for_triples(in_dist_pairs))
    perm = rng.permutation(len(X_in))
    X_in, y_in = X_in[perm], y_in[perm]
    n_train = int((1 - val_frac) * len(X_in))
    X_tr, y_tr = X_in[:n_train], y_in[:n_train]
    X_va, y_va = X_in[n_train:], y_in[n_train:]
    X_te, y_te = _to_arrays(_records_for_triples(holdout_pairs))
    perm_te = rng.permutation(len(X_te))
    X_te, y_te = X_te[perm_te], y_te[perm_te]
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


# =============================================================================
# AttnLRP primitives + Model
# =============================================================================

def identity_rule(x, fn):
    y = fn(x)
    ratio = (y / torch.where(x != 0, x, torch.full_like(x, 1e-10))).detach()
    return x * ratio + (y - x * ratio).detach()


def div_grad(x, factor):
    return x / factor + (x - x / factor).detach()


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=8, n_heads=1,
                 d_ff=16, seq_len=SEQ_LEN, activation="gelu"):
        super().__init__()
        assert activation in ("relu", "gelu")
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.activation = activation
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.ln1 = RMSNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff, bias=False)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.ln_final = RMSNorm(d_model)
        self.head = nn.Linear(d_model, 1, bias=False)

    def _ln(self, x, ln, lrp):
        if not lrp:
            return ln(x)
        rms = (x.pow(2).mean(-1, keepdim=True) + ln.eps).sqrt().detach()
        return x / rms * ln.weight

    def _act(self, x, lrp):
        fn = F.gelu if self.activation == "gelu" else F.relu
        return identity_rule(x, fn) if lrp else fn(x)

    def _attn(self, x, lrp):
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.q(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v(x).view(B, L, H, Dh).transpose(1, 2)
        if lrp:
            # AttnLRP (Achtibat et al. 2024): q,k → /4; v → /2 on the gradient.
            q = div_grad(q, 4.0)
            k = div_grad(k, 4.0)
            v = div_grad(v, 2.0)
        scores = q @ k.transpose(-1, -2) / (Dh ** 0.5)
        attn = torch.softmax(scores, -1)
        self._last_attn = attn.detach()
        ctx = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.o(ctx)

    def forward(self, ids=None, *, embeds=None, lrp_mode=False):
        assert (ids is None) != (embeds is None)
        if embeds is None:
            embeds = self.token_embed(ids)
        B, L, _ = embeds.shape
        pos = torch.arange(L, device=embeds.device).unsqueeze(0).expand(B, -1)
        x = embeds + self.pos_embed(pos)
        x = x + self._attn(self._ln(x, self.ln1, lrp_mode), lrp_mode)
        x = x + self.ff2(self._act(self.ff1(self._ln(x, self.ln2, lrp_mode)), lrp_mode))
        x = self._ln(x, self.ln_final, lrp_mode)
        return self.head(x.mean(1)).squeeze(-1)


# =============================================================================
# Training
# =============================================================================

def _eval_loss(model, X, y, batch_size=4096):
    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            losses.append(F.mse_loss(model(ids=X[i:i + batch_size]),
                                     y[i:i + batch_size], reduction="sum").item())
    return sum(losses) / len(X)


def train(model, train_data, val_data, test_data,
          epochs=100, batch_size=1024, lr=1e-2, device=torch.device("cpu")):
    X_tr, y_tr = (torch.tensor(a, device=device) for a in train_data)
    X_va, y_va = (torch.tensor(a, device=device) for a in val_data)
    X_te, y_te = (torch.tensor(a, device=device) for a in test_data)
    base_seq = [UNK_ID] * SEQ_LEN
    for p in CONSTANT_POSITIONS:
        base_seq[p] = TOK2ID[EQ]
    base_ids = torch.tensor([base_seq], dtype=torch.long, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 10)
    history = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        tr_loss = tr_n = 0
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.mse_loss(model(ids=X_tr[idx]), y_tr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * len(idx); tr_n += len(idx)
        va_loss = _eval_loss(model, X_va, y_va)
        te_loss = _eval_loss(model, X_te, y_te)
        with torch.no_grad():
            v_empty = model(ids=base_ids).item()
        history.append((tr_loss / tr_n, va_loss, te_loss))
        print(f"  epoch {ep+1:3d}/{epochs}  train {tr_loss/tr_n:7.4f}  "
              f"val {va_loss:7.4f}  test {te_loss:7.4f}  "
              f"v(∅)={v_empty:+.4f}  lr {opt.param_groups[0]['lr']:.2e}", flush=True)
        sched.step()
    return history


# =============================================================================
# Attribution helpers
# =============================================================================

def powerset(players):
    out = []
    for r in range(len(players) + 1):
        out.extend(itertools.combinations(players, r))
    return out


def _eval_masked(model, ids, coalitions, players, baseline_id=UNK_ID):
    device = next(model.parameters()).device
    batch = np.tile(np.asarray(ids, dtype=np.int64), (len(coalitions), 1))
    for n, s in enumerate(coalitions):
        absent = [p for p in players if p not in s]
        if absent:
            batch[n, absent] = baseline_id
    model.eval()
    with torch.no_grad():
        return model(ids=torch.tensor(batch, device=device)).cpu().numpy()


def _sv_from_cache(cache, players, n_out):
    Np = len(players)
    sv = np.zeros(n_out)
    for i in players:
        for s in cache:
            if i in s:
                continue
            w = math.factorial(len(s)) * math.factorial(Np - len(s) - 1) / math.factorial(Np)
            sv[i] += w * (cache[tuple(sorted(s + (i,)))] - cache[s])
    return sv


def _sv_sii_from_cache(cache, players, n_out):
    Np = len(players)
    sv = _sv_from_cache(cache, players, n_out)
    sii = np.zeros((n_out, n_out))
    for i, j in itertools.combinations(players, 2):
        total = 0.0
        for s in cache:
            if i in s or j in s:
                continue
            w = math.factorial(len(s)) * math.factorial(Np - len(s) - 2) / math.factorial(Np - 1)
            total += w * (
                cache[tuple(sorted(s + (i, j)))] - cache[tuple(sorted(s + (i,)))]
                - cache[tuple(sorted(s + (j,)))] + cache[s]
            )
        sii[i, j] = sii[j, i] = total
    return sv, sii


def _baseline_ids(ids, baseline_id=UNK_ID):
    out = list(ids)
    for p in EXPLAIN_POSITIONS:
        out[p] = baseline_id
    return out


def shapley(model, ids, baseline_id=UNK_ID):
    L = len(ids)
    players = EXPLAIN_POSITIONS
    coals = powerset(players)
    ys = _eval_masked(model, ids, coals, players, baseline_id)
    cache = {s: float(ys[i]) for i, s in enumerate(coals)}
    sv, _ = _sv_sii_from_cache(cache, players, L)
    return sv, cache


def attnlrp(model, ids, baseline_id=UNK_ID):
    device = next(model.parameters()).device
    ids_t = torch.tensor([list(ids)], dtype=torch.long, device=device)
    embeds = model.token_embed(ids_t).detach().requires_grad_(True)
    model(embeds=embeds, lrp_mode=True)[0].backward()
    out = (embeds * embeds.grad).sum(-1).detach().cpu().numpy()[0]
    out[list(CONSTANT_POSITIONS)] = 0.0
    return out


def integrated_gradients(model, ids, baseline_id=UNK_ID, steps=32):
    device = next(model.parameters()).device
    ids_t = torch.tensor([list(ids)], dtype=torch.long, device=device)
    xb_ids = torch.tensor([_baseline_ids(ids, baseline_id)], dtype=torch.long, device=device)
    x0 = model.token_embed(ids_t).detach()
    xb = model.token_embed(xb_ids).detach()
    delta = x0 - xb
    grads = torch.zeros_like(x0)
    for s in range(steps):
        alpha = (s + 0.5) / steps
        xt = (xb + alpha * delta).clone().requires_grad_(True)
        model(embeds=xt)[0].backward()
        grads = grads + xt.grad.detach()
    return (delta * grads / steps).sum(-1).cpu().numpy()[0]


def integrated_hessians(model, ids, baseline_id=UNK_ID, steps=12):
    """Integrated Hessians (Janizek et al. 2021): completeness-preserving
    pairwise attributions. Diagonal = direct, off-diagonal = pairwise."""
    device = next(model.parameters()).device
    ids_t = torch.tensor([list(ids)], dtype=torch.long, device=device)
    xb_ids = torch.tensor([_baseline_ids(ids, baseline_id)], dtype=torch.long, device=device)
    x0 = model.token_embed(ids_t).detach()
    xb = model.token_embed(xb_ids).detach()
    delta = x0 - xb
    L = x0.shape[1]
    ih = torch.zeros(L, L, device=device, dtype=torch.float32)
    inv_scale = 1.0 / (steps * steps)
    qs = [(s + 0.5) / steps for s in range(steps)]
    for alpha in qs:
        for beta in qs:
            xt = (xb + alpha * beta * delta).clone().requires_grad_(True)
            y = model(embeds=xt)
            g1 = torch.autograd.grad(y[0], xt, create_graph=True)[0]
            diag_lin = (g1[0] * delta[0]).sum(-1).detach()
            for i in range(L):
                ih[i, i] += diag_lin[i].float() * inv_scale
                gi = (g1[0, i] * delta[0, i]).sum()
                g2 = torch.autograd.grad(gi, xt, retain_graph=True, allow_unused=True)[0]
                if g2 is None:
                    continue
                ih[i] += alpha * beta * (g2[0] * delta[0]).sum(-1).float() * inv_scale
    ih_np = ih.cpu().numpy()
    return (ih_np + ih_np.T) / 2


def shapley_taylor(cache, players, n_out):
    """Shapley-Taylor Interaction Index at order 2 (Sundararajan et al. 2020)."""
    Np = len(players)
    out = np.zeros((n_out, n_out))
    v_empty = cache[tuple()]
    for i in players:
        out[i, i] = cache[(i,)] - v_empty
    if Np >= 2:
        scale = 2.0 / Np
        for i, j in itertools.combinations(players, 2):
            total = 0.0
            for s in cache:
                if i in s or j in s:
                    continue
                w = math.factorial(len(s)) * math.factorial(Np - len(s) - 1) / math.factorial(Np - 1)
                total += w * (
                    cache[tuple(sorted(s + (i, j)))] - cache[tuple(sorted(s + (i,)))]
                    - cache[tuple(sorted(s + (j,)))] + cache[s]
                )
            out[i, j] = out[j, i] = scale * total
    return out


# =============================================================================
# Meta-attribution: outer Shapley over an inner attribution to the target.
# Directed M[i, j] = SV_j(target=i). Inner ∈ {Shapley, IG, AttnLRP}.
# =============================================================================

INNER_IG_STEPS = 16


def _inner_sv(model, mids, t, baseline_id):
    return shapley(model, mids, baseline_id=baseline_id)[0][t]


def _inner_ig(model, mids, t, baseline_id):
    return integrated_gradients(model, mids, baseline_id=baseline_id, steps=INNER_IG_STEPS)[t]


def _inner_lrp(model, mids, t, baseline_id):
    return attnlrp(model, mids, baseline_id=baseline_id)[t]


_INNER_FN = {"sv": _inner_sv, "ig": _inner_ig, "lrp": _inner_lrp}
META_VARIANTS = ("meta_sv", "meta_ig", "meta_lrp")


def metagame(model, ids, target_pos, variant, baseline_id=UNK_ID):
    """φ_{j→i} = outer Shapley over the (d−1)-player game (target_pos fixed
    as present). Diagonal := pure individual effect ϕ_i({i}; f, x)."""
    inner_fn = _INNER_FN[variant.split("_", 1)[1]]
    L = len(ids)
    players = tuple(p for p in EXPLAIN_POSITIONS if p != target_pos)
    cache = {}
    for s in powerset(players):
        masked = np.asarray(ids, dtype=np.int64).copy()
        absent = [p for p in players if p not in s]
        if absent:
            masked[absent] = baseline_id
        cache[s] = float(inner_fn(model, masked, target_pos, baseline_id))
    sv = _sv_from_cache(cache, players, L)
    sv[target_pos] = cache[()]  # Janizek et al., Def. 2
    return sv


# =============================================================================
# Plotting
# =============================================================================

def plot_combined(rows, title, path, fontsize=9):
    """Stack attribution methods as rows in one figure.
    Each row: (name, cells, group, [custom_norm], [custom_cmap]).
    Cells: [(slot_idx, label, value, width)]."""
    n_rows = len(rows)
    label_width = 1.5
    cell_w_self, cell_w_pair = 0.40, 0.58
    cell_h = 0.26
    box_inset_x = 0.085
    title_h = 0.22 if title else 0.04
    slot_x = [label_width + i * cell_w_self for i in range(3)]
    slot_x += [label_width + 3 * cell_w_self + i * cell_w_pair for i in range(6)]
    slot_unit_w = [cell_w_self] * 3 + [cell_w_pair] * 6
    total_cells_w = 3 * cell_w_self + 6 * cell_w_pair
    xlim_right = label_width + total_cells_w - box_inset_x
    fig_w = max(4.0, xlim_right)
    fig_h = cell_h * n_rows + title_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, xlim_right)
    ax.set_ylim(0, n_rows * cell_h)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - (title_h / fig_h))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_div", ["#1B7BD0", "#ffffff", "#FF1D62"])
    group_vmax = {}
    for row in rows:
        group = row[2]
        custom_norm = row[3] if len(row) > 3 else None
        if group is None or custom_norm is not None:
            continue
        vs = [abs(c[2]) for c in row[1]]
        group_vmax[group] = max(group_vmax.get(group, 0.0), max(vs) if vs else 0.0)
    for r_idx, row in enumerate(rows):
        name, cells, group = row[0], row[1], row[2]
        custom_norm = row[3] if len(row) > 3 else None
        row_cmap = row[4] if len(row) > 4 else cmap
        y = (n_rows - 1 - r_idx) * cell_h
        ax.text(label_width - 0.08, y + cell_h * 0.63, name,
                ha="right", va="center", fontsize=fontsize - 1, fontfamily="XCharter")
        values = np.asarray([c[2] for c in cells], dtype=float)
        if custom_norm is not None:
            norm = custom_norm
        elif group is not None:
            vmax = max(group_vmax[group], 1e-8)
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        else:
            vmax = max(float(np.abs(values).max()), 1e-8)
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        for slot_idx, lab, v, width in cells:
            x = slot_x[slot_idx]
            full_w = sum(slot_unit_w[slot_idx:slot_idx + width])
            rgba = row_cmap(norm(v))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            single_slot_w = slot_unit_w[slot_idx]
            box_w = single_slot_w - 2 * box_inset_x
            box_x = x + (full_w - single_slot_w) / 2 + box_inset_x
            ax.add_patch(mpatches.FancyBboxPatch(
                (box_x, y + cell_h * 0.42),
                box_w, cell_h * 0.42,
                boxstyle="round,pad=0.003",
                facecolor=rgba, edgecolor="none", alpha=0.9))
            text_color = "black" if lum > 0.58 else "white"
            ax.text(x + full_w / 2, y + cell_h * 0.61, lab,
                    ha="center", va="center", fontsize=fontsize - 2,
                    fontfamily="JetBrains Mono", weight="bold", color=text_color)
            ax.text(x + full_w / 2, y + cell_h * 0.14, f"{v:+.2f}",
                    ha="center", va="center", fontsize=fontsize - 3,
                    fontfamily="JetBrains Mono", color="#333")
    if title:
        input_str, pred_val = title
        inp = " ".join(input_str.split()).replace("-", "−")
        pred_s = f"{pred_val:.2f}".replace("-", "−")
        title_tex = (rf"$\bf{{Input\ sequence\!:}}$ {inp}      "
                     rf"$\bf{{Predicted\ output\!:}}$ {pred_s}")
        fig.text(0.01, 1 - 0.04 / fig_h, title_tex,
                 ha="left", va="top", fontsize=fontsize, fontfamily="XCharter")
    fig.savefig(path, dpi=140)
    plt.close(fig)


# Slot layout: slots 0–2 = singletons (width 1); slots 3+2k, 4+2k = pair zone k.
# Matrix convention: mat[target, source]; label "src → tgt".
PAIRS = [(0, 1), (0, 2), (1, 2)]


def _cells_diag(tokens, mat_or_vec):
    """Width-1 singleton cells at slots 0–2, from a vector or a matrix diagonal."""
    get = (lambda k: mat_or_vec[k, k]) if np.ndim(mat_or_vec) == 2 else (lambda k: mat_or_vec[k])
    return [(k, tokens[k], float(get(k)), 1) for k in range(3)]


def _cells_sym_full(tokens, mat):
    return _cells_diag(tokens, mat) + [
        (3 + 2 * k, f"{tokens[i]} , {tokens[j]}", float(mat[i, j]), 2)
        for k, (i, j) in enumerate(PAIRS)
    ]


def _cells_sym_full_split(tokens, mat):
    """Two width-1 undirected cells per pair (mirrors asym layout, comma label)."""
    out = _cells_diag(tokens, mat)
    for k, (i, j) in enumerate(PAIRS):
        out.append((3 + 2 * k, f"{tokens[i]} , {tokens[j]}", float(mat[i, j]), 1))
        out.append((4 + 2 * k, f"{tokens[j]} , {tokens[i]}", float(mat[i, j]), 1))
    return out


def _cells_asym_full(tokens, mat):
    out = _cells_diag(tokens, mat)
    for k, (i, j) in enumerate(PAIRS):
        out.append((3 + 2 * k, f"{tokens[i]} → {tokens[j]}", float(mat[j, i]), 1))
        out.append((4 + 2 * k, f"{tokens[j]} → {tokens[i]}", float(mat[i, j]), 1))
    return out


# =============================================================================
# Pipeline
# =============================================================================

def compute_attributions(model, ids, ig_steps=32, ih_steps=12):
    sv, cache = shapley(model, ids)
    f_full = cache[tuple(EXPLAIN_POSITIONS)]
    f_base = cache[tuple()]
    stii = shapley_taylor(cache, EXPLAIN_POSITIONS, len(ids))
    lrp = attnlrp(model, ids)
    # attnlrp just ran a forward; _last_attn matches a plain forward (div_grad
    # is forward-identity).
    attn_w = model._last_attn[0].mean(0).cpu().numpy()
    ig = integrated_gradients(model, ids, steps=ig_steps)
    ih = integrated_hessians(model, ids, steps=ih_steps)
    L = len(ids)
    meta = {}
    for variant in META_VARIANTS:
        M = np.zeros((L, L))
        for i in EXPLAIN_POSITIONS:
            M[i] = metagame(model, ids, target_pos=i, variant=variant)
        meta[variant] = M
    return {
        "tokens": decode(ids),
        "sv": sv, "stii": stii,
        "attnlrp": lrp, "attention": attn_w,
        "ig": ig, "ih": ih,
        "metagame": meta,
        "f_full": f_full, "f_base": f_base,
    }


def explain(model, ids, plot_dir, tag, label="", ig_steps=32, ih_steps=12):
    disp = decode(ids, pretty=True)
    print(f"\n--- {tag}: {label} ---", flush=True)
    ep = list(EXPLAIN_POSITIONS)
    disp_ep = [disp[p] for p in ep]
    crop_v = lambda v: np.asarray(v)[ep]
    crop_m = lambda m: np.asarray(m)[np.ix_(ep, ep)]
    r = compute_attributions(model, ids, ig_steps=ig_steps, ih_steps=ih_steps)
    sv, stii = r["sv"], r["stii"]
    lrp, attn_w = r["attnlrp"], r["attention"]
    ig, ih = r["ig"], r["ih"]
    meta = r["metagame"]
    f_full = r["f_full"]

    # Group keys: rows sharing a non-None group share a diverging color scale.
    G_SHAP, G_IH, G_MIG, G_MLRP = "shap", "ih", "mig", "mlrp"
    ATTN_NORM = mcolors.Normalize(vmin=0.0, vmax=1.0)
    ATTN_CMAP = mcolors.LinearSegmentedColormap.from_list(
        "gray_lb", ["#ececec", "#000000"])
    R = {
        "attn":     ("Attention",                  _cells_asym_full(disp_ep, crop_m(attn_w)),         None, ATTN_NORM, ATTN_CMAP),
        "sv":       ("Shapley values",             _cells_diag(disp_ep, crop_v(sv)),                  G_SHAP),
        "ig":       ("Integrated gradients",       _cells_diag(disp_ep, crop_v(ig)),                  G_MIG),
        "lrp":      (r"AttnLRP ($\approx$input$\times$gradient)", _cells_diag(disp_ep, crop_v(lrp)), G_MLRP),
        "stii":     ("Shapley interactions",       _cells_sym_full(disp_ep, crop_m(stii)),            G_SHAP),
        "ih":       ("Integrated Hessians",        _cells_sym_full_split(disp_ep, crop_m(ih)),        G_IH),
        "meta_sv":  (r"$\bf{Meta{-}}$Shapley values",         _cells_asym_full(disp_ep, crop_m(meta["meta_sv"])),  G_SHAP),
        "meta_ig":  (r"$\bf{Meta{-}}$Integrated gradients",   _cells_asym_full(disp_ep, crop_m(meta["meta_ig"])),  G_MIG),
        "meta_lrp": (r"$\bf{Meta{-}}$AttnLRP",                _cells_asym_full(disp_ep, crop_m(meta["meta_lrp"])), G_MLRP),
    }
    input_str = (label.split("=")[0].strip() + " =") if "=" in (label or "") else (label or tag)
    # Wrap a negative `b` operand so "a op -b =" renders as "a op (-b) =".
    input_str = re.sub(r'([+\-])\s+(-\d+)(\s*=)', r'\1 (\2)\3', input_str)

    order = ["attn",
             "sv", "meta_sv", "stii",
             "ig", "meta_ig", "ih",
             "lrp", "meta_lrp"]
    plot_combined([R[k] for k in order], title=(input_str, f_full),
                  path=f"{plot_dir}/{tag}.pdf")
    return r


# =============================================================================
# Examples — paper Figure 1 + App. Figures 5, 6
# =============================================================================

EXAMPLES = [
    ("3_m5_m",   3, -5, MINUS, "3 - -5 = 8"),
    ("7_m5_m",   7, -5, MINUS, "7 - -5 = 12"),
    ("4_m4_p",   4, -4, PLUS,  "4 + -4 = 0"),
    ("m6_8_p",  -6,  8, PLUS,  "-6 +  8 =  2"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activation", choices=["relu", "gelu"], default="gelu")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--ih-steps", type=int, default=12)
    ap.add_argument("--output-dir", type=str, default="./results")
    ap.add_argument("--train-only", action="store_true",
                    help="Skip the attribution suite; train + sanity check only.")
    args = ap.parse_args()

    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    run_dir = args.output_dir
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "run.log")
    log_file = open(log_path, "w", buffering=1)
    class _Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
    sys.stdout = _Tee(sys.__stdout__, log_file)
    print(f"logging to {log_path}", flush=True)

    print("building dataset...", flush=True)
    train_data, val_data, test_data = build_dataset(seed=args.seed)
    print(f"  train {len(train_data[0]):,}  val {len(val_data[0]):,}  "
          f"test {len(test_data[0]):,}  (test = held-out triples)", flush=True)

    model = SimpleTransformer(activation=args.activation).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} params  "
          f"(activation={args.activation})", flush=True)

    ckpt_path = os.path.join(run_dir, "model.pt")
    if os.path.exists(ckpt_path):
        print(f"loading checkpoint from {ckpt_path} (delete it to retrain)", flush=True)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        print("training...", flush=True)
        history = train(model, train_data, val_data, test_data,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, device=device)
        torch.save({"state_dict": model.state_dict(), "activation": args.activation,
                    "history": history}, ckpt_path)

    model.eval()
    with torch.no_grad():
        y_hat_tr = model(ids=torch.tensor(train_data[0], device=device)).cpu().numpy()
        y_hat_va = model(ids=torch.tensor(val_data[0], device=device)).cpu().numpy()
        y_hat_te = model(ids=torch.tensor(test_data[0], device=device)).cpu().numpy()
    for name, yh, yt in (("train", y_hat_tr, train_data[1]),
                         ("valid", y_hat_va, val_data[1]),
                         ("test ", y_hat_te, test_data[1])):
        mse = float(np.mean((yh - yt) ** 2))
        print(f"eval [{name.strip():<5s}]  MSE={mse:7.4f}  "
              f"mean(y_hat)={yh.mean():+.4f}  mean(y)={yt.mean():+.4f}", flush=True)

    if args.train_only:
        print("\n=== --train-only: skipping attribution suite ===", flush=True)
        return

    results = {}
    for tag, a, b, op, label in EXAMPLES:
        ids = np.asarray(tokenize(a, b, op), dtype=np.int64)
        results[tag] = explain(model, ids, run_dir, tag, label=label,
                               ig_steps=args.ig_steps, ih_steps=args.ih_steps)

    flat = {}
    scalar_keys = ("sv", "stii", "attnlrp", "attention", "ig", "ih")
    for tag, r in results.items():
        for k in scalar_keys:
            flat[f"{tag}__{k}"] = np.asarray(r[k], dtype=np.float32)
        for variant, M in r["metagame"].items():
            flat[f"{tag}__{variant}"] = np.asarray(M, dtype=np.float32)
    np.savez_compressed(os.path.join(run_dir, "results.npz"), **flat)

    print(f"\n=== Done. Outputs in {run_dir}/ ===", flush=True)


if __name__ == "__main__":
    main()
