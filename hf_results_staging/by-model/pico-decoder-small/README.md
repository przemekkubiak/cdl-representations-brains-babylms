# pico-decoder-small

**pico decoder ~65M (12 layers, width 384)** — [`pico-lm/pico-decoder-small`](https://huggingface.co/pico-lm/pico-decoder-small) · dense checkpoints -- the full 126-checkpoint trajectory, the densest in this release.

126 checkpoints, steps 0–125000.

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
| brain RSA (mean over checkpoints × 12 cells) | 0.0012 | ❌ confounded |
| brain RSA vs training step | rho 0.164, p 0.000 (n=1512) | ❌ confounded |
| interp: participation ratio (effective dim / hidden) | 0.1839 | ✅ |
| interp: gini (activation sparsity) | 0.2427 | ✅ |
| interp: hoyer sparsity | 0.2817 | ✅ |
| interp: CKA to previous checkpoint | 0.9607 | ✅ |
| localisation: selectivity index | 0.5509 | ✅ |
| localisation: mean overlap with other phenomena | 0.0066 | ✅ |
| localisation: active layers | 11.4583 | ✅ |
| localisation: layer centre of mass | 0.5465 | ✅ |
| behaviour: minimal-pair accuracy (chance 0.5) | 0.6819 | ✅ |
| causal selectivity (localized vs random ablation) | -0.0089 | ✅ |

## What this model shows

- Minimal-pair accuracy **0.682**, clearly above chance — the model has learned the contrasts being probed.
- Participation ratio **0.184**: the representation uses ~18% of its available dimensions. Mid-range for this release.
- Selectivity **0.551** — like every model in this release, phenomena are only moderately isolated. Nothing here shows sharp, dedicated linguistic circuitry.
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

![overview](figures/pico-decoder-small_overview.png)
