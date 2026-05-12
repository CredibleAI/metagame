# `example/` — Figure 1 + App. Figures 5, 6

A 760-parameter single-block transformer trained on `a + b` and `a − b`
(with `a, b ∈ {−9, …, 9}`). After training, the script renders the four
attribution panels shown in paper Figures 1 & 5 (`3_m5_m`) and App. Figure 6
(`7_m5_m`, `4_m4_p`, `m6_8_p`).

## Files

| File | Purpose |
|---|---|
| `main.py` | End-to-end script: dataset → model → train (or load `results/model.pt`) → 9-row attribution panel per example. Self-contained — no internal imports. |
| `env.yml` | Conda env spec (`metagame`). |

## Quick start

```bash
conda env create -f env.yml
conda run -n metagame python main.py --output-dir results
```

## CLI flags

- `--activation {gelu,relu}` (default `gelu`)
- `--epochs`, `--batch-size`, `--lr`, `--seed` — training hyperparameters
- `--ig-steps` (default 32), `--ih-steps` (default 12) — attribution budget
- `--output-dir` (default `./results`)
- `--train-only` — skip the interpretation, run training only

## Algorithms shown (9 rows per panel)

| Row | Source |
|---|---|
| Attention | `model._last_attn`, mean over heads |
| Shapley values | `shapley()` — exact via 2^3 powerset |
| Meta-Shapley values | `metagame(variant="meta_sv")` |
| Shapley interactions | `shapley_taylor()` (STII, k=2) |
| Integrated gradients | `integrated_gradients(steps=32)` |
| Meta-Integrated gradients | `metagame(variant="meta_ig")` |
| Integrated Hessians | `integrated_hessians(steps=12)` |
| AttnLRP | `attnlrp()` — input × gradient with AttnLRP rules |
| Meta-AttnLRP | `metagame(variant="meta_lrp")` |

## Cached results

```
results/
├── model.pt          # trained model checkpoint
├── results.npz       # attributions for the 4 examples (sv/stii/ih/ig/attnlrp/attention + meta_*)
├── run.log           # script stdout from the most recent run
├── 3_m5_m.pdf        # paper Figures 1 & 5
├── 7_m5_m.pdf        # paper Figure 6, top
├── 4_m4_p.pdf        # paper Figure 6, middle
└── m6_8_p.pdf        # paper Figure 6, bottom
```