"""Visualization of Meta-AttnLRP results — emits a per-(model,
domain) `combined.pdf` next to each `results.npz`, plus a 4-panel
`recall_curve.pdf` aggregated across all runs."""

import argparse
import json
import os
from collections import defaultdict, namedtuple
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "XCharter"
matplotlib.rcParams["mathtext.bf"] = "XCharter:bold"
matplotlib.rcParams["mathtext.it"] = "XCharter:italic"
matplotlib.rcParams["font.family"] = "XCharter"
matplotlib.rcParams["axes.unicode_minus"] = True
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.textpath import TextPath


RED = "#ff0d57"
BLUE = "#1e88e5"
CMAP_SEQ = mcolors.LinearSegmentedColormap.from_list("wr_custom", ["#ffffff", RED])
PT_PER_INCH = 72

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(HERE, "results")
DEFAULT_PROMPTS_JSON = os.path.join(HERE, "prompts.json")
HF_CACHE = os.environ.get(
    "HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
)
IT_MODELS = ("gemma-3-1b-it", "gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it")
PT_MODELS = ("gemma-3-1b-pt", "gemma-3-4b-pt", "gemma-3-12b-pt", "gemma-3-27b-pt")
ALL_MODELS = IT_MODELS + PT_MODELS

Result = namedtuple("Result", ["interaction_matrix", "relevance_mean",
                               "real_idx", "tokens_clean", "generated_ids"])

SPECIAL_TOKENS = {
    "<bos>", "<start_of_turn>", "<end_of_turn>",
    r"<start\_of\_turn>", r"<end\_of\_turn>",
    "user", "model", r"\#", "\\#", r"\#\#", r"\\#\\#",
    "\n", r"\\n",
}


def _mpl_safe(s):
    return s.replace("$", r"\$")


def _norm(s):
    return str(s).strip().lower()


def discover_runs(root, models):
    runs = {}
    root = Path(root)
    for m in models:
        md = root / m
        if not md.is_dir():
            continue
        for dd in sorted(md.iterdir()):
            p = dd / "results.npz"
            if dd.is_dir() and p.is_file():
                runs[(m, dd.name)] = p
    return runs


def load_result(path):
    d = np.load(path, allow_pickle=True)
    return Result(**{f: d[f] for f in Result._fields})


def load_prompts(path):
    with open(path) as f:
        return {d["domain"]: d for d in sorted(json.load(f), key=lambda x: x["id"])}


def _load_gemma_tokenizer():
    """Cheap tokenizer to decode generated_ids; matches all Gemma-3 variants."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("google/gemma-3-1b-it", cache_dir=HF_CACHE)


# --- Pair matching --------------------------------------------------------

def match_pair_tokens(pair_tokens, tokens_clean, real_idx):
    """Resolve annotated surface strings to real-token index pairs."""
    if len(pair_tokens) < 2:
        return None
    surface = [_norm(tokens_clean[i]) for i in real_idx]
    n = len(surface)

    def find_spans(target):
        t = _norm(target).replace(" ", "")
        if not t:
            return []
        exact, contain = [], []
        for s in range(n):
            acc = ""
            for e in range(s, n):
                acc += surface[e]
                if not acc:
                    continue
                if acc == t:
                    exact.append((s, e - s)); break
                if t in acc:
                    contain.append((s, e - s)); break
        chosen = exact or contain
        if not chosen:
            return []
        m_len = min(x[1] for x in chosen)
        return [x[0] for x in chosen if x[1] == m_len]

    sa, sb = find_spans(pair_tokens[0]), find_spans(pair_tokens[1])
    if not sa or not sb:
        return None
    return sorted({(a, b) for a in sa for b in sb if a != b})


def _off_diag_abs(M):
    return np.abs(M[~np.eye(M.shape[0], dtype=bool)])


def _edge_rank(abs_M, off_abs, i, j):
    return int((off_abs > abs_M[i, j]).sum() + 1)


# --- Recall@K curve -------------------------------------------------------

def plot_recall_curve_extended(runs, prompts, path,
                               sizes=("1b", "4b", "12b", "27b"),
                               pct_range=range(0, 16)):
    """4-panel recall@K curve (one per model size). Each panel overlays IT
    (solid) vs PT (dashed) for both AttnLRP (blue) and Meta-AttnLRP (red)."""
    doms_per = defaultdict(set)
    for (m, d) in runs:
        doms_per[m].add(d)
    if not doms_per:
        return
    common = set.intersection(*[doms_per[m] for m in doms_per])

    def _entries_for(model_id):
        out = []
        for d in common:
            if d not in prompts:
                continue
            pth = runs.get((model_id, d))
            if pth is None:
                continue
            r = load_result(pth)
            pairs = prompts[d].get("pairs", [])
            if not pairs:
                continue
            M_int = r.interaction_matrix
            rel = r.relevance_mean[r.real_idx]
            M_attr = rel[:, None] + rel[None, :]
            int_abs, int_off = np.abs(M_int), _off_diag_abs(M_int)
            attr_abs, attr_off = np.abs(M_attr), _off_diag_abs(M_attr)
            n_real = M_int.shape[0]
            int_ranks, attr_ranks = [], []
            for pair_obj in pairs:
                matched = match_pair_tokens(pair_obj["tokens"],
                                            r.tokens_clean, r.real_idx)
                if not matched:
                    continue
                best_ir, best_ar = None, None
                for a, b in matched:
                    ir = min(_edge_rank(int_abs, int_off, a, b),
                             _edge_rank(int_abs, int_off, b, a))
                    ar = _edge_rank(attr_abs, attr_off, a, b)
                    if best_ir is None or ir < best_ir:
                        best_ir, best_ar = ir, ar
                int_ranks.append(best_ir); attr_ranks.append(best_ar)
            if int_ranks:
                out.append({
                    "int_ranks": np.asarray(int_ranks),
                    "attr_ranks": np.asarray(attr_ranks),
                    "n_edges": n_real * (n_real - 1),
                })
        return out

    pcts = list(pct_range)
    FS = 9
    fig, axes = plt.subplots(1, len(sizes), figsize=(6.0, 1.6), sharey=True)
    if len(sizes) == 1:
        axes = [axes]

    PT_DASHES = (3, 2)
    for ax, size in zip(axes, sizes):
        for variant, ls_kw in (
                ("it", {"linestyle": "-"}),
                ("pt", {"dashes": PT_DASHES}),
        ):
            entries = _entries_for(f"gemma-3-{size}-{variant}")
            if not entries:
                continue
            attr_m, attr_s, int_m, int_s = [], [], [], []
            for pct in pcts:
                if pct == 0:
                    attr_m.append(0.0); int_m.append(0.0)
                    attr_s.append(0.0); int_s.append(0.0)
                    continue
                ah, ih = [], []
                for e in entries:
                    K = max(1, round(pct / 100.0 * e["n_edges"]))
                    ah.append(float(np.mean(e["attr_ranks"] <= K)))
                    ih.append(float(np.mean(e["int_ranks"] <= K)))
                attr_m.append(np.mean(ah)); int_m.append(np.mean(ih))
                attr_s.append(np.std(ah, ddof=1) / np.sqrt(len(ah))
                              if len(ah) > 1 else 0.0)
                int_s.append(np.std(ih, ddof=1) / np.sqrt(len(ih))
                             if len(ih) > 1 else 0.0)
            attr_m = np.asarray(attr_m); attr_s = np.asarray(attr_s)
            int_m  = np.asarray(int_m);  int_s  = np.asarray(int_s)
            ax.plot(pcts, attr_m, color=BLUE, lw=1.4, **ls_kw)
            ax.fill_between(pcts, attr_m - 2 * attr_s, attr_m + 2 * attr_s,
                            color=BLUE, alpha=0.10, linewidth=0)
            ax.plot(pcts, int_m, color=RED, lw=1.4, **ls_kw)
            ax.fill_between(pcts, int_m - 2 * int_s, int_m + 2 * int_s,
                            color=RED, alpha=0.10, linewidth=0)
        ax.set_title(f"Gemma-3-{size.upper()}", fontsize=FS)
        ax.set_xlim(pcts[0], pcts[-1]); ax.set_ylim(0, 1)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xticklabels(["0%", "5%", "10%", "15%"], fontsize=FS - 2)
        ax.get_xticklabels()[-1].set_horizontalalignment("right")
        ax.tick_params(axis="y", labelsize=FS - 2)
        ax.grid(alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel(r"Recall @$K$", fontsize=FS, y=0.45)
    color_handles = [
        Line2D([0], [0], color=RED, lw=1.4),
        Line2D([0], [0], color=BLUE, lw=1.4),
    ]
    color_labels = [r"$\bf{Meta{-}}$AttnLRP", "AttnLRP"]
    style_handles = [
        Line2D([0], [0], color="black", lw=1.4, linestyle="-"),
        Line2D([0], [0], color="black", lw=1.4, dashes=PT_DASHES),
    ]
    style_labels = ["Instruction-tuned", "Pre-trained"]
    color_kw = dict(loc="center", bbox_to_anchor=(0.5, 0.5),
                    fontsize=FS - 3, frameon=False, handlelength=1.5,
                    handletextpad=0.4, borderaxespad=0.0)
    style_kw = dict(color_kw); style_kw["bbox_to_anchor"] = (0.62, 0.5)
    for idx in (0, 2):
        if idx < len(axes):
            axes[idx].legend(color_handles, color_labels, **color_kw)
    for idx in (1, 3):
        if idx < len(axes):
            axes[idx].legend(style_handles, style_labels, **style_kw)
    fig.tight_layout(pad=0.3, w_pad=0.5, h_pad=0.2)
    x_center = (axes[0].get_position().x0 + axes[-1].get_position().x1) / 2
    y_bottom = min(ax.xaxis.get_tightbbox(fig.canvas.get_renderer())
                   .transformed(fig.transFigure.inverted()).y0 for ax in axes)
    fig.text(x_center, y_bottom - 0.01, r"$K$ (% of all interactions)",
             ha="center", va="top", fontsize=FS)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)


def plot_combined(tokens, relevance, interaction_matrix, pair_labels,
                            path, caption="", figwidth=6.5, fontsize=11,
                            drop_special=True, k_pairs=10):
    """Two-column figure per example.
    Left column: top colorbar (AttnLRP) + prompt token heatmap +
                 generated-text caption.
    Right column: "Meta-AttnLRP" label (top, same y as AttnLRP) +
                  top-`k_pairs` directed pair cells stacked vertically,
                  sorted by |interaction value|.
    Both columns share a body region whose height fits the taller column.
    """
    relevance_np = (relevance.numpy() if hasattr(relevance, "numpy")
                    else np.asarray(relevance))

    BR_SENTINEL = "\x00BR\x00"

    def _is_newline_tok(t):
        s = t.strip()
        return (s == "" and ("\n" in t or "\\n" in t)) or t in ("\n", "\\n", r"\n")

    if drop_special:
        kept_tokens, kept_rel = [], []
        for tok, rel in zip(tokens, relevance_np):
            if _is_newline_tok(tok):
                kept_tokens.append(BR_SENTINEL)
                kept_rel.append(0.0)
            elif (tok not in SPECIAL_TOKENS
                    and tok.strip() not in SPECIAL_TOKENS):
                kept_tokens.append(tok)
                kept_rel.append(rel)
        tokens = kept_tokens
        relevance_np = np.array(kept_rel)
    else:
        BR_SENTINEL = None

    vmax = max(float(np.abs(relevance_np).max())
               if len(relevance_np) else 1e-8, 1e-8)
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    colormap = plt.get_cmap(CMAP_SEQ)

    # Top-K pair cells.
    n_t = interaction_matrix.shape[0]
    edges = [(i, j, float(interaction_matrix[i, j]))
             for i in range(n_t) for j in range(n_t) if i != j]
    edges.sort(key=lambda e: abs(e[2]), reverse=True)
    edges = edges[:k_pairs]
    n_cells = len(edges)

    cmap_pair = mcolors.LinearSegmentedColormap.from_list(
        "custom_div", ["#1B7BD0", "#ffffff", "#FF1D62"])
    pair_vmax = max((abs(v) for _, _, v in edges), default=1e-12)
    pair_norm = mcolors.TwoSlopeNorm(vmin=-pair_vmax, vcenter=0, vmax=pair_vmax)

    label_fontsize = fontsize - 2
    cell_inset_x_pts = 0.02 * PT_PER_INCH
    inner_pad_pts = label_fontsize * 0.155
    val_gap_pts = 5

    cell_label_fp = FontProperties(family="JetBrains Mono", weight="bold",
                                   size=fontsize - 2)
    cell_value_fp = FontProperties(family="JetBrains Mono",
                                   size=fontsize - 3)
    cell_label_strs = [f"{pair_labels[j]} → {pair_labels[i]}"
                       for i, j, _ in edges]
    cell_value_strs = [f"{v:+.2f}" for _, _, v in edges]
    label_widths_pts = [TextPath((0, 0), s,
                                 prop=cell_label_fp).get_extents().width
                        for s in cell_label_strs]
    value_widths_pts = [TextPath((0, 0), s,
                                 prop=cell_value_fp).get_extents().width
                        for s in cell_value_strs]
    box_widths_pts = [w + 2 * inner_pad_pts for w in label_widths_pts]
    cell_widths_pts = [bw + val_gap_pts + vw + 2 * cell_inset_x_pts
                       for bw, vw in zip(box_widths_pts, value_widths_pts)]

    box_h_pts = label_fontsize * 1.123
    cell_h_pts = box_h_pts + 5
    cell_row_gap_pts = 2
    right_body_h_pts = (cell_h_pts * n_cells
                        + cell_row_gap_pts * max(0, n_cells - 1))
    max_box_w_pts = max(box_widths_pts) if box_widths_pts else 0
    max_value_w_pts = max(value_widths_pts) if value_widths_pts else 0
    right_col_w_pts = (cell_inset_x_pts + max_box_w_pts + val_gap_pts
                       + max_value_w_pts + cell_inset_x_pts)

    col_gap_pts = 18
    col_gap_in = col_gap_pts / PT_PER_INCH
    right_col_w_in = right_col_w_pts / PT_PER_INCH
    left_col_w_in = max(2.0, figwidth - right_col_w_in - col_gap_in)
    left_col_w_pts = left_col_w_in * PT_PER_INCH

    prompt_fontsize = fontsize - 1
    char_fp = FontProperties(family="XCharter", size=prompt_fontsize)
    prompt_bold_fp = FontProperties(family="XCharter", size=prompt_fontsize,
                                    weight="bold")
    prompt_label_str = "Prompt:"
    prompt_label_w_pts = TextPath((0, 0), prompt_label_str,
                                  prop=prompt_bold_fp).get_extents().width
    probe = "X" * 20
    char_width_pts = TextPath((0, 0), probe,
                              prop=char_fp).get_extents().width / len(probe)

    line_height = prompt_fontsize * 1.6
    box_height = prompt_fontsize * 1.4
    box_descent = prompt_fontsize * 0.35
    box_pad_x = 2
    gap_pts = char_width_pts

    while len(tokens) and tokens[0] == BR_SENTINEL:
        tokens = tokens[1:]
        relevance_np = relevance_np[1:]

    lines = []
    cur = [(prompt_label_str, None, prompt_label_w_pts)]
    cur_w = prompt_label_w_pts
    for tok, rel in zip(tokens, relevance_np):
        if tok == BR_SENTINEL:
            lines.append(cur); cur, cur_w = [], 0
            continue
        segments = tok.replace("\\n", "\n").split("\n")
        for seg_idx, segment in enumerate(segments):
            if seg_idx > 0:
                lines.append(cur); cur, cur_w = [], 0
            stripped = segment.strip()
            if not stripped:
                continue
            tw = TextPath((0, 0), stripped, prop=char_fp).get_extents().width
            extra = tw + (gap_pts if cur else 0)
            if cur_w + extra > left_col_w_pts and cur:
                lines.append(cur); cur, cur_w = [], 0
                extra = tw
            cur.append((stripped, rel, tw)); cur_w += extra
    if cur:
        lines.append(cur)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    text_h_pts = (((len(lines) - 1) * line_height + box_height)
                  if lines else 0)

    caption_fontsize = fontsize - 1
    caption_lines = []
    if caption:
        italic_fp = FontProperties(family="XCharter", size=caption_fontsize,
                                   style="italic")
        bold_fp = FontProperties(family="XCharter", size=caption_fontsize,
                                 weight="bold")

        def _ww(w, fp):
            return TextPath((0, 0), w, prop=fp).get_extents().width

        space_w = _ww("x x", italic_fp) - _ww("xx", italic_fp)
        words = []
        body = caption
        if caption.startswith("Generated: "):
            words.append((r"$\mathbf{Generated\!:}$",
                          _ww("Generated:", bold_fp)))
            body = caption[len("Generated: "):]
        for w in body.split():
            words.append((w, _ww(w, italic_fp)))
        cur_l, cur_lw = [], 0.0
        for disp, w in words:
            extra = w + (space_w if cur_l else 0)
            if cur_l and cur_lw + extra > left_col_w_pts:
                caption_lines.append(" ".join(d for d, _ in cur_l))
                cur_l = [(disp, w)]; cur_lw = w
            else:
                cur_l.append((disp, w)); cur_lw += extra
        if cur_l:
            caption_lines.append(" ".join(d for d, _ in cur_l))

    caption_h_pts = (caption_fontsize * 1.3 * len(caption_lines)
                     if caption_lines else 0)
    caption_top_gap_pts = prompt_fontsize * 0.5
    left_body_h_pts = text_h_pts + caption_top_gap_pts + caption_h_pts

    body_h_pts = max(left_body_h_pts, right_body_h_pts)
    body_h_in = body_h_pts / PT_PER_INCH
    top_cbar_bar_h_in = 0.10
    cbar_gap_above_body_in = 0.04
    top_section_h_in = 0.20
    fig_height = top_section_h_in + body_h_in + 0.02

    fig = plt.figure(figsize=(figwidth, fig_height))

    cbar_w_in = min(0.22 * figwidth, left_col_w_in * 0.45)
    cbar_left_in = left_col_w_in - cbar_w_in - 0.25
    cbar_left_frac = cbar_left_in / figwidth
    cbar_w_frac = cbar_w_in / figwidth
    cbar_h_frac = top_cbar_bar_h_in / fig_height
    cbar_y_frac = (body_h_in + cbar_gap_above_body_in) / fig_height

    cax = fig.add_axes([cbar_left_frac, cbar_y_frac, cbar_w_frac, cbar_h_frac])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_ticks([])
    cbar.ax.text(-0.04, 0.5, "0", transform=cbar.ax.transAxes,
                 ha="right", va="center", fontsize=fontsize - 1)
    rounded = round(vmax, 1) if vmax >= 1 else round(vmax, 2)
    cbar.ax.text(1.04, 0.5, f"{rounded:g}", transform=cbar.ax.transAxes,
                 ha="left", va="center", fontsize=fontsize - 1)
    fig.text(cbar_left_frac - 0.04, cbar_y_frac + cbar_h_frac / 2,
             "AttnLRP", ha="right", va="center",
             fontsize=fontsize, fontfamily="XCharter")

    right_col_x_frac = (left_col_w_in + col_gap_in) / figwidth
    meta_x_frac = right_col_x_frac + (right_col_w_in / figwidth) / 2
    fig.text(meta_x_frac, cbar_y_frac + cbar_h_frac / 2,
             r"$\bf{Meta{-}}$AttnLRP", ha="center", va="center",
             fontsize=fontsize, fontfamily="XCharter")

    body_h_frac = body_h_in / fig_height
    ax_left = fig.add_axes([0, 0, left_col_w_in / figwidth, body_h_frac])
    ax_left.set_xlim(0, left_col_w_pts)
    ax_left.set_ylim(0, body_h_pts)
    ax_left.axis("off")

    ax_right = fig.add_axes([right_col_x_frac, 0,
                             right_col_w_in / figwidth, body_h_frac])
    ax_right.set_xlim(0, right_col_w_pts)
    ax_right.set_ylim(0, body_h_pts)
    ax_right.axis("off")

    y_top = body_h_pts - (box_height - box_descent)
    for line_idx, line in enumerate(lines):
        line_w = sum(tw for _, _, tw in line) + max(0, len(line) - 1) * gap_pts
        x = (left_col_w_pts - line_w) / 2
        y = y_top - line_idx * line_height
        for i, (tok, rel, tw) in enumerate(line):
            if i > 0:
                x += gap_pts
            if rel is None:
                ax_left.text(x, y, r"$\mathbf{Prompt\!:}$",
                             fontsize=prompt_fontsize, fontfamily="XCharter",
                             color="black", verticalalignment="baseline")
            else:
                rect = mpatches.Rectangle(
                    (x - box_pad_x, y - box_descent),
                    tw + 2 * box_pad_x, box_height,
                    facecolor=colormap(norm(rel)), edgecolor="none", alpha=0.75,
                )
                ax_left.add_patch(rect)
                ax_left.text(x, y, tok, fontsize=prompt_fontsize,
                             fontfamily="XCharter",
                             color="white" if norm(rel) > 0.65 else "black",
                             verticalalignment="baseline")
            x += tw

    if caption_lines:
        cap_y_top = body_h_pts - text_h_pts - caption_top_gap_pts
        ax_left.text(left_col_w_pts / 2, cap_y_top,
                     "\n".join(caption_lines),
                     fontsize=caption_fontsize, fontfamily="XCharter",
                     fontstyle="italic", ha="center", va="top",
                     color="#555555", linespacing=1.3)

    box_right_x = cell_inset_x_pts + max_box_w_pts
    value_left_x = box_right_x + val_gap_pts
    for cell_idx, ((i, j, v), bw, lbl) in enumerate(
            zip(edges, box_widths_pts, cell_label_strs)):
        cell_top_y = body_h_pts - cell_idx * (cell_h_pts + cell_row_gap_pts)
        cell_bot_y = cell_top_y - cell_h_pts
        cy = (cell_top_y + cell_bot_y) / 2
        box_x = box_right_x - bw
        box_y = cy - box_h_pts / 2
        rgba = cmap_pair(pair_norm(v))
        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = "black" if lum > 0.63 else "white"
        ax_right.add_patch(mpatches.FancyBboxPatch(
            (box_x, box_y), bw, box_h_pts,
            boxstyle="round,pad=0.3",
            facecolor=rgba, edgecolor="none", alpha=0.9))
        ax_right.text(box_x + bw / 2, cy, lbl,
                      ha="center", va="center", fontsize=fontsize - 2,
                      fontfamily="JetBrains Mono", weight="bold",
                      color=text_color)
        ax_right.text(value_left_x, cy, f"{v:+.2f}",
                      ha="left", va="center",
                      fontsize=fontsize - 3, fontfamily="JetBrains Mono",
                      color="#333")

    fig.savefig(path, bbox_inches="tight", pad_inches=0.02, dpi=150)
    plt.close(fig)


def write_combined_per_run(runs, prompts, output_root):
    """For every (model, domain) run, write `<output_root>/<model>/<domain>/combined.pdf`."""
    output_root = Path(output_root)
    tokenizer = _load_gemma_tokenizer()
    for (m, d), npz_path in sorted(runs.items()):
        if d not in prompts:
            continue
        r = load_result(npz_path)
        labels = [_norm(r.tokens_clean[i]) or str(r.tokens_clean[i])
                  for i in r.real_idx]
        generated_text = tokenizer.decode(list(r.generated_ids))
        caption = (f"Generated: "
                   f"{_mpl_safe(generated_text.replace('<end_of_turn>', '').strip())}")
        plot_combined(
            list(r.tokens_clean), r.relevance_mean,
            r.interaction_matrix, labels,
            output_root / m / d / "combined.pdf",
            caption=caption, drop_special=True, k_pairs=8,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--prompts-json", default=DEFAULT_PROMPTS_JSON)
    args = ap.parse_args()

    output_root = Path(args.output_root)
    prompts = load_prompts(args.prompts_json)
    runs = discover_runs(output_root, ALL_MODELS)
    if not runs:
        print(f"[error] no runs under {output_root}", flush=True)
        return

    plot_recall_curve_extended(runs, prompts, output_root / "recall_curve.pdf")
    print(f"[info] wrote {output_root}/recall_curve.pdf", flush=True)
    write_combined_per_run(runs, prompts, output_root)
    print(f"[info] wrote combined.pdf per (model, domain) under {output_root}/", flush=True)


if __name__ == "__main__":
    main()
