# `meta_conceptattention/` — Table 3 + Figures 4, 13

ConceptAttention vs. Meta-ConceptAttention on FLUX.1 [schnell] across Pascal
VOC, MS COCO, and ImageNet-Segmentation.

## Quick start: reproduce Table 3 from the cached metrics

```bash
conda env create -f env.yml
conda run -n metaconceptattention python -c "
import json
m = json.load(open('results/mscoco_n20/metrics.json'))
print('CA       single:', m['paper_tracks']['ca']['single'])
print('Meta-CA  single:', m['paper_tracks']['meta_ca']['single'])
print('CA       multi: ', m['paper_tracks']['ca']['multi'])
print('Meta-CA  multi: ', m['paper_tracks']['meta_ca']['multi'])
"
```

Reads `results/mscoco_n20/metrics.json` (the default Table 3 variant) and
prints CA and Meta-CA for both the single-concept and multiple-concepts
tracks. No GPU, no dataset, no model download. Substitute `pascalvoc_n20` or
`imagenetseg_n20` for the other Table 3 rows.

## Files

| File | Purpose |
|---|---|
| `main.py` | Evaluation pipeline: MS COCO / Pascal VOC / ImageNet-Seg sweep over CA + Meta-CA, sharded inference + `--aggregate` step that writes `results/<variant>/metrics.json`. |
| `generate.py` | Figure 4 qualitative example: FLUX.1 [schnell] on `"a photo of a vintage bike on a city street"`, rendering the 5-row grid (input + CA at 2/3/N concept coalitions + Meta-CA Shapley) and the K×N directional Meta-Shapley figure. Outputs to `results/bike/`. |
| `run_main.sh` / `run_generate.sh` | Slurm wrappers. |
| `concept_attention/` | Bundled trimmed upstream package — only the parts imported by `main.py` / `generate.py`. Picked up automatically as a sibling Python package when running scripts from this directory; no `pip install` needed. |
| `env.yml` | Conda env spec (`metaconceptattention`). |

## Cached results

```
results/
├── bike/                              # Figure 4 (the paper's qualitative example)
│   ├── input.png                      # FLUX.1-generated image
│   ├── grid.{pdf,png}                 # CA at 2/3/N + Meta-CA Shapley
│   └── directional.{pdf,png}          # K×N directional Meta-Shapley φ_(j→i)
├── mscoco_n20/metrics.json            # Table 3 default (MS COCO)
├── pascalvoc_n20/metrics.json         # Table 3 default (Pascal VOC)
├── imagenetseg_n20/metrics.json       # Table 3 default (ImageNet-Seg)
├── <dataset>_<n>/metrics.json         # baseline at each player-set size:
│                                      #   n0, d10, n20, n24 — Fig 13 x-axis
└── <dataset>_<n>_<ablation>/metrics.json
                                       # 3 paper Fig 13 ablation styles:
                                       #   _decoupled   = without cross-concept attention
                                       #   _l14to18     = last 5 layers only
                                       #   _prompt      = with artificial prompt
```

44 ablation cells in total. Each `metrics.json` carries `paper_tracks.<method>.<track>`
where method ∈ `{ca, meta_ca}` and track ∈ `{single, multi}` (multi omitted
for ImageNet-Seg, whose binary masks make it equivalent to the single track).

## Reproducing each artifact

| Artifact | Command / Path |
|---|---|
| **Table 3** (Acc/mIoU/mAP) | `python -c "import json; print(json.load(open('results/<dataset>_n20/metrics.json'))['paper_tracks'])"` for `<dataset>` ∈ `{mscoco, pascalvoc, imagenetseg}`. |
| **Fig 13** (player-set + ablation curves) | All 44 `results/<variant>/metrics.json` files; the figure plots the four `paper_tracks.<method>.<track>.{mAcc,mIoU,mAP}` values across the player-set grid for each ablation style. |
| **Fig 4** (bike qualitative panels) | `python generate.py` → `results/bike/{input,grid,directional}.{png,pdf}`. Requires GPU + the FLUX.1 [schnell] model (~5 min loading + a few minutes per coalition pass). |
| **Full sweep** (regenerates the metrics) | `sbatch run_main.sh` per `(DATASET, ablation env vars)` combination — see `run_main.sh` header for the 44-cell grid. Each cell aggregates ~hours on a single H100. |

## `main.py` flags (paper App. Fig 13)

- `--total-players N` (default `20`) — pad each image's player set to
  exactly N single-token thing classes (GT-present + deterministic random
  distractors). N=0 disables padding.
- `--n-distractors d` — variable-size: per-image total = n_present + d.
  Overrides `--total-players`. Used for the `d10` cells.
- `--prompt` — artificial `'a {cls1}, a {cls2}, …'` prompt (default: empty).
- `--decoupled` — `concept_self_attention=False` (= without cross-concept attention).
- `--layer-indices 14 15 16 17 18` — last 5 layers only (default: `range(9, 19)`).

## Environment variables

- `HF_HOME` (default `~/.cache/huggingface`) — used as the parent dir for the
  MS COCO, Pascal VOC, and ImageNet-Seg downloads. On first run, `main.py` `wget`s and
  extracts each archive into `$HF_HOME/{mscoco,pascalvoc,imagenetseg}/`
  (idempotent — existing files are kept). Also respected implicitly by
  diffusers as the FLUX.1 [schnell] model cache.
