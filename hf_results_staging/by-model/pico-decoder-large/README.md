# pico-decoder-large

**pico decoder ~500M, largest of the scale ladder** — [`pico-lm/pico-decoder-large`](https://huggingface.co/pico-lm/pico-decoder-large) · dense commit-history checkpoints.

21 checkpoints, steps 0–125000.

> ## ⚠️ The brain alignment numbers on this page are confounded
>
> `brain_alignment.csv`, every `brain_*` column in `checkpoints.csv`, and
> `ablation_alignment.csv` are **not interpretable as a result about this model.**
>
> In ds003604 each stimulus appears in exactly one scanner run, so run membership is
> perfectly confounded with stimulus identity. "Different run" predicts brain
> dissimilarity at Spearman **+0.49 to +0.87** across all 12 task × session cells, while
> no stimulus property predicts it at all (trial type ≈ −0.02, length ≈ 0, lexical
> overlap ≈ 0). A language model cannot represent a scanner run, so its RSA against that
> structure is ≈0 by construction.
>
> Ruled out as explanations: **cohort size** (40 vs 98 subjects agree at rho 0.928) and
> **layer choice** (flat across all 12–14 layers). Within-run normalisation removes the
> confound (+0.562 → −0.041) and leaves alignment at **+0.022, not significant**.
>
> **`interp_*`, `localisation_*`, `behaviour` and `ablation_behaviour` on this page are
> unaffected** — they are computed from the model alone and never touch the fMRI data.
> See the [repo README](../../README.md) for the full account.


## Headline numbers

| metric | value | safe to use? |
|---|---|---|
| brain RSA (mean over checkpoints × 12 cells) | -0.0124 | ❌ confounded |
| brain RSA vs training step | rho 0.006, p 0.927 (n=252) | ❌ confounded |
| interp: participation ratio (effective dim / hidden) | 0.1042 | ✅ |
| interp: gini (activation sparsity) | 0.2206 | ✅ |
| interp: hoyer sparsity | 0.4837 | ✅ |
| interp: CKA to previous checkpoint | 0.7775 | ✅ |
| localisation: selectivity index | 0.5391 | ✅ |
| localisation: mean overlap with other phenomena | 0.0068 | ✅ |
| localisation: active layers | 11.9286 | ✅ |
| localisation: layer centre of mass | 0.5685 | ✅ |
| behaviour: minimal-pair accuracy (chance 0.5) | 0.6873 | ✅ |
| causal selectivity (localized vs random ablation) | -0.0057 | ✅ |

## What this model shows

- Minimal-pair accuracy **0.687**, clearly above chance — the model has learned the contrasts being probed.
- Participation ratio **0.104**: the representation uses ~10% of its available dimensions. Very compressed relative to the other families here.
- Selectivity **0.539** — like every model in this release, phenomena are only moderately isolated. Nothing here shows sharp, dedicated linguistic circuitry.
- Brain alignment: **no claim is made**, see the warning above.

## Files

| file | contents | confounded? |
|---|---|---|
| `ablation_alignment.csv` | brain RSA with the localized circuit ablated | **yes — whole file** |
| `ablation_behaviour.csv` | accuracy under localized vs random ablation | no |
| `behaviour.csv` | minimal-pair accuracy per phenomenon per checkpoint | no |
| `brain_alignment.csv` | RSA per task × session × checkpoint | **yes — whole file** |
| `checkpoints.csv` | this model's rows of the main per-checkpoint table (all three axes) | brain_* columns only |
| `interp_layerwise.csv` | the same metrics per layer per checkpoint | no |
| `interp_mechanistic.csv` | per-checkpoint representation metrics | no |
| `localisation_isolation.csv` | per-phenomenon circuit localisation | no |
| `localisation_onset.csv` | localisation onset step per phenomenon | no |

![overview](figures/pico-decoder-large_overview.png)
