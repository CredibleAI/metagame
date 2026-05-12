# `meta_gradeclip/` — Tables 2 & 4 + Figures 2, 7, 8, 9

Pointing-game evaluation across CLIP, SigLIP-2, MetaCLIP-2; plus the App. C.3
adaptations of generic Attention / MaskCLIP / Grad-ECLIP to SigLIP-2.

## Quick start: reproduce Tables 2 & 4 from the cached CSV

```bash
conda env create -f env.yml
conda run -n metagradeclip python pointing_game.py --table-only
```

Reads `results/pointing_game_results.csv` (50 600 rows, 4.7 MB) and prints
the per-(model, method, meta, n_objects) `mass_ratio` mean ± 2·SE. No GPU,
no dataset, no model download.

## Files

| File | Purpose |
|---|---|
| `pointing_game.py` | Pointing-game pipeline only: GPU sweep over CLIP / SigLIP-2 / MetaCLIP-2 with gradeclip / genericattention / maskclip / their meta variants, then aggregation into `pointing_game_results.csv`. Pass `--table-only` to skip the heavy section and read the cached CSV directly. |
| `pointing_game_fixlip.py` | FIxLIP Shapley-interaction baseline for one (model, scene). Invoked per slurm task by `run_pointing_game_fixlip.sh`. |
| `run_pointing_game_fixlip.py` | Submits the FIxLIP sweep over the 10 paper games × 4 model patch sizes via `sbatch run_pointing_game_fixlip.sh`. |
| `run_pointing_game_fixlip.sh` | Slurm wrapper. |
| `example.py` | Qualitative figures: pointing-game Fig 7 panels (cached JSONs) plus Figs 2, 8, 9 (`dog_and_hydrant_{1,2,3}`, `dog_hot_dog`) rendered from scratch via `MetagameTokens`. |
| `metagame.py` | `Metagame`, `MetagameTokens`, and the shared `explain_image` helper, imported by both `pointing_game.py` and `example.py`. |
| `data.py` | Builds the pointing-game dataset from ImageNet-1k validation: per-class image dumps and 4-class 2×2 composites. Set `IMAGENET_PATH` env var to point at a local arrow dump, or leave unset to pull from HuggingFace. |
| `gradeclip.py`, `gradesiglip.py` | Per-method explanation kernels for CLIP/SigLIP families (App. C.3). |
| `fixlip/` | Helpers package: `utils`, `plot`, `sampler`, `game_huggingface`, `fixlip` (the FIxLIP class). |
| `env.yml` | Conda env spec (`metagradeclip`). |

## Cached results

```
results/
├── pointing_game_results.csv          # Tables 2 & 4 source
├── dog_and_hydrant_1/                 # Fig 2 (the paper's primary example)
│   ├── explanation_gradeclip.png      # raw Grad-ECLIP attribution
│   ├── first_order/                   # f(S) maps for text-token coalitions (paper-visible 4)
│   ├── second_order/                  # Meta-Grad-ECLIP "Shapley values (2nd effects)": one map per text token
│   └── third_order/                   # Meta-Grad-ECLIP "Shapley interactions (3rd)": paper-visible token pairs
├── dog_and_hydrant_2/                 # Fig 8a (green-hydrant variant)
├── dog_and_hydrant_3/                 # Fig 8b (yorkshire / red-hydrant)
├── dog_hot_dog/                       # Fig 9 (synergy + antisynergy)
└── pointing_game/                     # Fig 7 panels: <game>_7.pdf
```

The `images/` directory holds the 5 input photos referenced by the
qualitative scripts (`dog_and_hydrant_{1,2,3}`, `dog_hot_dog`,
`pointing_game_new`).

## Reproducing each artifact

| Artifact | Command / Path |
|---|---|
| **Tables 2 & 4** | `python pointing_game.py --table-only` |
| **Fig 7 + Figs 2, 8, 9** (qualitative panels) | `python example.py` — Part 1 renders Fig 7 from cached pointing-game JSONs into `results/pointing_game/`; Part 2 loads MetaCLIP-2 huge and produces Figs 2 / 8 / 9 in `results/<example>/`. GPU required for Part 2. |
| **Full sweep** (regenerates the CSV) | `python pointing_game.py` end-to-end + `python run_pointing_game_fixlip.py` for the FIxLIP baseline. Needs the dataset built by `python data.py` first, plus access to GPUs (~hours per model × method). |

## Environment variables

- `POINTING_GAME_DATASET` (default `pointing_game`) — root for the
  generated 4-image composites; consumed by `data.py`, `pointing_game.py`,
  and `run_pointing_game_fixlip.py`.
- `POINTING_GAME_OUTPUT` (default `results`) — root for written
  attributions and the aggregated CSV.
- `IMAGENET_PATH` (optional) — if set, `data.py` reads ImageNet-1k from a
  local arrow dump instead of pulling from HuggingFace.