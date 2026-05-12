# `meta_attnlrp/` — Figure 3 + App. Figures 10, 11, 12

Token-interaction analysis of Gemma-3 (1B / 4B / 12B / 27B, both PT and IT)
via AttnLRP and its meta extension.

## Quick start

```bash
conda env create -f env.yml
conda run -n metaattnlrp python analyze.py
```

Reads `results/<model>/<domain>/results.npz` for all 8 Gemma-3 variants ×
21 prompt domains and writes:

- `results/recall_curve.pdf` — 4-panel recall@K curve, one panel per
  model size, IT (solid) vs PT (dashed). Reproduces paper Fig 3 / App. Fig 10.
- `results/<model>/<domain>/combined.pdf` — per-(model, domain) figure with
  the prompt-token AttnLRP heatmap on the left and the top-K directed
  Meta-AttnLRP interactions on the right. Reproduces paper App. Figs 11 / 12.

## Files

| File | Purpose |
|---|---|
| `main.py` | Per-(model, prompt) sweep: load Gemma-3 → AttnLRP relevance per generation step → masked-coalition Shapley over real input tokens → directed n×n interaction matrix. Writes `results/<model>/<run_name>/results.npz`. |
| `analyze.py` | Loads the per-prompt `results.npz` files and emits `recall_curve.pdf` plus the per-(model, domain) `combined.pdf` panels. |
| `run_main.sh` | sbatch wrapper around `main.py` with `--array=0-20%6` for the 21 prompt domains. |
| `sampler.py` | `CoalitionSampler` — used by `compute_shapley_attnlrp` in `main.py` to draw masked input-token coalitions with paired Bernoulli(p=0.5) sampling and Shapley sampling weights, the inputs of the XGBoost regressor that estimates per-target Shapley values. |
| `prompts.json` | The 21 paper prompt domains with annotated token pairs (App. D.1). |
| `env.yml` | Conda env spec (`metaattnlrp`). Note: `transformers<4.56` is pinned for AttnLRP compatibility — incompatible with the `metagame` / `metagradeclip` envs. |

## Cached results

```
results/
├── recall_curve.pdf                                  # paper Fig 3 / App. Fig 10
└── <gemma-3-{1b,4b,12b,27b}-{it,pt}>/
    └── <prompt domain>/
        ├── results.npz                                # raw payload (numerical)
        └── combined.pdf                               # paper App. Figs 11 / 12
```

## Reproducing each artifact

| Artifact | Command |
|---|---|
| **Fig 3 / App. Fig 10 + App. Figs 11 & 12** | `python analyze.py` |
| **Single (model, prompt) cell** (regenerates `results.npz`) | `python main.py --model-id google/gemma-3-4b-it --prompt "<text>" --run-name <domain>` |
| **Full Gemma-3 sweep** (8 models × 21 prompts) | `for M in google/gemma-3-{1b,4b,12b,27b}-{it,pt}; do sbatch run_main.sh "$M"; done` (slurm) |

## Environment variables

- `HF_HOME` — Hugging Face cache for model weights. Defaults to
  `~/.cache/huggingface`. Override if your shared cache lives elsewhere.
