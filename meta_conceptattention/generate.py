"""Meta-ConceptAttention qualitative example for paper Figure 4 (bike).

Generates a FLUX.1 [schnell] image from the prompt "a photo of a vintage bike
on a city street", then computes:
  - paper CA at 2 / 3 / N=20 coalitions (`pipeline.generate_image(softmax=True)`),
  - Meta-CA Shapley diagonal sv[t] AND directional φ_{j→i} for the visualised
    target concepts via a single streaming sweep over coalition sizes 1..N
    (`pipeline.generate_image(softmax=False)`, GPU softmax per chunk).

The player set is the same 5-bg + 20-thing benchmark used by `main.py`'s Pascal VOC
evaluation (Pascal VOC 20 classes with the paper's T5-friendly remapping).
We visualise the three concepts {car, street, bike} from Figure 4.

Outputs.
  results/bike/
    input.png                  — generated image
    grid.{png,pdf}             — input + paper-CA at 2 / 3 / N + Meta-CA
                                 Shapley (rows × {car, street, bike} columns)
    directional.{png,pdf}      — K×N grid of directional Meta-Shapley
                                 φ_{j→i} (rows = visualised target i, cols =
                                 every player j; diagonal = sv[i])

Usage (conda env: metaconceptattention):
  python generate.py
"""
import itertools
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from concept_attention import ConceptAttentionFluxPipeline


HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results", "bike")
os.makedirs(OUT_DIR, exist_ok=True)

# Paper Figure 4 example.
PROMPT = "a photo of a vintage bike on a city street"
VIZ_CONCEPTS = ["car", "street", "bike"]
SEED = 31

BG_CONCEPTS = ["background"]
# 20 single-T5-token concepts forming the player set (N = 20).
THING_CONCEPTS = [
    "plane", "bike", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
    "cow", "table", "dog", "horse", "motorcycle", "person", "pot", "sheep",
    "sofa", "train", "street",
]

WIDTH = HEIGHT = 1024
NUM_STEPS = 4
LAYER_INDICES = list(range(9, 19))
DEVICE = "cuda:0"

# Paper colour palette.
_PAPER_BLUE = "#1B7BD0"
_PAPER_RED  = "#FF1D62"
_CMAP_POS = matplotlib.colors.LinearSegmentedColormap.from_list(
    "paper_pos", ["white", _PAPER_RED])
_CMAP_SIGNED = matplotlib.colors.LinearSegmentedColormap.from_list(
    "paper_signed", [_PAPER_BLUE, "white", _PAPER_RED])


def _hide(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _norm01(x):
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(x)


def _imshow_pos(ax, data, title, vmax=None):
    if vmax is None:
        ax.imshow(_norm01(data), cmap=_CMAP_POS, vmin=0, vmax=1, interpolation="bilinear")
    else:
        ax.imshow(data, cmap=_CMAP_POS, vmin=0, vmax=vmax, interpolation="bilinear")
    ax.set_title(title, fontsize=9); _hide(ax)


def _imshow_signed(ax, data, title, vmax=None):
    v = float(np.abs(data).max()) if vmax is None else float(vmax)
    if v < 1e-8:
        ax.imshow(np.zeros_like(data), cmap=_CMAP_SIGNED, vmin=-1, vmax=1, interpolation="bilinear")
    else:
        ax.imshow(data, cmap=_CMAP_SIGNED, vmin=-v, vmax=v, interpolation="bilinear")
    ax.set_title(title, fontsize=9); _hide(ax)


def _row_vmax(d, ps):
    vals = [d[p].max() for p in ps if p in d]
    return max(vals) if vals else 1.0


def _row_signed_vmax(dir_off, target_i, players, q_lo=0.001, q_hi=0.999):
    pooled = np.concatenate([dir_off[(target_i, j)].ravel()
                             for j in players
                             if j != target_i and (target_i, j) in dir_off])
    if pooled.size == 0:
        return 1.0
    return max(abs(float(np.quantile(pooled, q_lo))),
               abs(float(np.quantile(pooled, q_hi))))


def streaming_meta_ca(raw_np, players, ctx, viz_idx, device, chunk_size=2000):
    """Single streaming sweep producing both the Shapley diagonal sv[i] for
    every i in `viz_idx` AND directional φ_{j→i} for every ordered (i, j) pair
    in viz_idx × players (i ≠ j). Math (cf. Definition 2 of the paper):

        sv[i]     = Σ_{S ∋ i} (|S|-1)!(n-|S|)!/n! · softmax_{S∪ctx}(raw)[i]
        φ_{j→i}  = Σ_{S ∋ i, j ∈ S}     (|S|-2)!(n-|S|)!/(n-1)!     · ν_i(S)
                 − Σ_{S ∋ i, j ∉ S}     (|S|-1)!(n-|S|-1)!/(n-1)!   · ν_i(S)
        with ν_i(S) := softmax_{S∪ctx}(raw)[i].
    """
    import torch
    n = len(players)
    H, W = raw_np.shape[1:]
    raw_t = torch.from_numpy(raw_np).to(device)
    sv = {t: torch.zeros((H, W), dtype=torch.float32, device=device)
          for t in viz_idx}
    dir_off = {(i, j): torch.zeros((H, W), dtype=torch.float32, device=device)
               for i in viz_idx for j in players if i != j}
    ctx_arr = np.array(sorted(ctx), dtype=np.int64)

    for k in range(1, n + 1):
        coalitions = list(itertools.combinations(players, k))
        diag_w = math.factorial(k - 1) * math.factorial(n - k) / math.factorial(n)
        dir_with    = (math.factorial(k - 2) * math.factorial(n - k)     / math.factorial(n - 1)) if k >= 2     else 0.0
        dir_without = (math.factorial(k - 1) * math.factorial(n - k - 1) / math.factorial(n - 1)) if k <= n - 1 else 0.0

        for ck in range(0, len(coalitions), chunk_size):
            chunk = coalitions[ck:ck + chunk_size]
            n_chunk = len(chunk)
            T_arr = np.array(chunk, dtype=np.int64)
            if ctx_arr.size > 0:
                ctx_block = np.broadcast_to(ctx_arr, (n_chunk, ctx_arr.size))
                members = np.concatenate([T_arr, ctx_block], axis=1)
            else:
                members = T_arr
            local_t = torch.from_numpy(members).to(device)
            soft = torch.softmax(raw_t[local_t], dim=1)

            for i in viz_idx:
                t_mask_np = (T_arr == i).any(axis=1)
                if not t_mask_np.any():
                    continue
                positions_np = (T_arr[t_mask_np] == i).argmax(axis=1).astype(np.int64)
                t_mask = torch.from_numpy(t_mask_np).to(device)
                positions = torch.from_numpy(positions_np).to(device)
                soft_with_t = soft[t_mask]
                idx = positions.view(-1, 1, 1, 1).expand(-1, 1, H, W)
                t_softmax = torch.gather(soft_with_t, dim=1, index=idx).squeeze(1)
                sv[i] = sv[i] + diag_w * t_softmax.sum(dim=0)

                T_with_i = T_arr[t_mask_np]
                for j in players:
                    if j == i:
                        continue
                    j_in_np = (T_with_i == j).any(axis=1)
                    if dir_with > 0 and j_in_np.any():
                        j_in = torch.from_numpy(j_in_np).to(device)
                        dir_off[(i, j)] = dir_off[(i, j)] + dir_with * t_softmax[j_in].sum(dim=0)
                    j_out_np = ~j_in_np
                    if dir_without > 0 and j_out_np.any():
                        j_out = torch.from_numpy(j_out_np).to(device)
                        dir_off[(i, j)] = dir_off[(i, j)] - dir_without * t_softmax[j_out].sum(dim=0)
            del soft, local_t

    sv_out  = {t: sv[t].cpu().numpy() for t in viz_idx}
    dir_out = {pair: dir_off[pair].cpu().numpy() for pair in dir_off}
    return sv_out, dir_out


def main():
    import torch

    full_concepts = list(BG_CONCEPTS) + list(THING_CONCEPTS)
    n_bg = len(BG_CONCEPTS)
    ctx = tuple(range(n_bg))
    players = tuple(range(n_bg, len(full_concepts)))
    viz_idx = tuple(full_concepts.index(c) for c in VIZ_CONCEPTS)
    viz_2 = viz_idx[:2]
    N = len(players)

    print(f"=== {PROMPT}  (seed={SEED}, N={N}) ===", flush=True)

    print("loading FLUX pipeline (~5min)…", flush=True)
    pipeline = ConceptAttentionFluxPipeline(model_name="flux-schnell", device=DEVICE)
    print("  pipeline ready", flush=True)

    def _flux(concepts_list, softmax_flag):
        return pipeline.generate_image(
            prompt=PROMPT, concepts=concepts_list,
            width=WIDTH, height=HEIGHT,
            layer_indices=LAYER_INDICES,
            seed=SEED,
            num_inference_steps=NUM_STEPS,
            softmax=softmax_flag,
            return_pil_heatmaps=False,
        )

    # 1. softmax=False → raw scores cached for the streaming Meta-CA sweep.
    out = _flux(full_concepts, False)
    pil_image = out.image
    raw_np = np.asarray(out.concept_heatmaps, dtype=np.float32)
    print(f"  raw: {raw_np.shape}", flush=True)
    pil_image.save(os.path.join(OUT_DIR, "input.png"))
    del out; torch.cuda.empty_cache()

    # 2. paper CA (AveragedSoftmax) at 2 / 3 / N concept coalitions.
    viz_2_concepts = [full_concepts[p] for p in viz_2]
    viz_3_concepts = [full_concepts[p] for p in viz_idx]
    out2 = _flux(list(BG_CONCEPTS) + viz_2_concepts, True)
    raw2 = np.asarray(out2.concept_heatmaps, dtype=np.float32)
    paper_ca_2 = {p: raw2[n_bg + i] for i, p in enumerate(viz_2)}
    del out2; torch.cuda.empty_cache()

    out3 = _flux(list(BG_CONCEPTS) + viz_3_concepts, True)
    raw3 = np.asarray(out3.concept_heatmaps, dtype=np.float32)
    paper_ca_3 = {p: raw3[n_bg + i] for i, p in enumerate(viz_idx)}
    del out3; torch.cuda.empty_cache()

    out_full = _flux(full_concepts, True)
    paper_ca_full = np.asarray(out_full.concept_heatmaps, dtype=np.float32)
    paper_ca_N = {p: paper_ca_full[p] for p in viz_idx}
    del out_full; torch.cuda.empty_cache()
    print(f"  paper CA (2/3/N) ready", flush=True)

    # 3. streaming Meta-CA sweep: Shapley diagonal + directional φ_{j→i}.
    sv_diag, dir_off = streaming_meta_ca(
        raw_np, players, ctx, viz_idx=viz_idx, device=DEVICE)
    print(f"  Meta-CA over {N} players done", flush=True)

    # ----- Figure 1: 5-row grid (input + CA at 2/3/N + Meta-CA) -----
    K = len(viz_idx)
    rmax_pca2 = _row_vmax(paper_ca_2, viz_2)
    rmax_pca3 = _row_vmax(paper_ca_3, viz_idx)
    rmax_pcaN = _row_vmax(paper_ca_N, viz_idx)
    rmax_meta = _row_vmax(sv_diag,    viz_idx)

    fig, axes = plt.subplots(5, K + 1, figsize=(2.4 * (K + 1), 5 * 2.4))
    axes[0, 0].imshow(pil_image)
    axes[0, 0].set_title("FLUX.1 [schnell]", fontsize=10); _hide(axes[0, 0])
    for k, p in enumerate(viz_idx):
        axes[0, k + 1].text(0.5, 0.5, full_concepts[p], fontsize=13,
                            ha="center", va="center", fontweight="bold")
        _hide(axes[0, k + 1])

    H, W = next(iter(sv_diag.values())).shape

    def _draw_row(row_idx, label, heatmaps, vmax):
        _hide(axes[row_idx, 0])
        axes[row_idx, 0].text(0.5, 0.5, label, fontsize=9, ha="center", va="center")
        for k, p in enumerate(viz_idx):
            if p in heatmaps:
                _imshow_pos(axes[row_idx, k + 1], heatmaps[p],
                            full_concepts[p], vmax=vmax)
            else:
                axes[row_idx, k + 1].imshow(np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
                axes[row_idx, k + 1].set_title("(out of coalition)", fontsize=8)
                _hide(axes[row_idx, k + 1])

    _draw_row(1, f"CA\n(2 concepts:\n{full_concepts[viz_2[0]]} +\n{full_concepts[viz_2[1]]})",
              paper_ca_2, rmax_pca2)
    _draw_row(2, "CA\n(3 concepts)",     paper_ca_3, rmax_pca3)
    _draw_row(3, f"CA\n(Pascal VOC: {N}\nconcepts)", paper_ca_N, rmax_pcaN)
    _draw_row(4, f"Meta-CA\nShapley\n(N={N} players)", sv_diag, rmax_meta)

    fig.suptitle(f'"{PROMPT}"', fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "grid.png"), dpi=140, bbox_inches="tight")
    fig.savefig(os.path.join(OUT_DIR, "grid.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote grid", flush=True)

    # ----- Figure 2: K × N directional Meta-Shapley -----
    fig2, axes2 = plt.subplots(K, N, figsize=(1.3 * N, 1.6 * K), squeeze=False)
    for r, i in enumerate(viz_idx):
        rmax_dir_i = _row_signed_vmax(dir_off, i, players)
        for c, j in enumerate(players):
            ax = axes2[r, c]
            if i == j:
                _imshow_pos(ax, sv_diag[i], full_concepts[i])
            else:
                _imshow_signed(ax, dir_off[(i, j)], full_concepts[j],
                               vmax=rmax_dir_i)
        axes2[r, 0].set_ylabel(f"target:\n{full_concepts[i]}", fontsize=9,
                               rotation=0, ha="right", va="center", labelpad=20)
    fig2.suptitle(f'Directional Meta-Shapley φ_(j→i) over {N}-player game   "{PROMPT}"',
                  fontsize=11, y=1.00)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "directional.png"), dpi=140, bbox_inches="tight")
    fig2.savefig(os.path.join(OUT_DIR, "directional.pdf"), bbox_inches="tight")
    plt.close(fig2)
    print(f"  wrote directional", flush=True)


if __name__ == "__main__":
    main()
