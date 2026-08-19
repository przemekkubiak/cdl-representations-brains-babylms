# beetle-humanscale-eng

**English at a developmentally plausible ('human-scale') data budget** — [`Beetle-HumanScale/beetle-monolingual-humanscale-eng`](https://huggingface.co/Beetle-HumanScale/beetle-monolingual-humanscale-eng) · 35 step branches.

18 checkpoints, steps 0–9500.

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
| brain RSA (mean over checkpoints × 12 cells) | -0.0158 | ❌ confounded |
| brain RSA vs training step | rho -0.025, p 0.710 (n=216) | ❌ confounded |
| interp: participation ratio (effective dim / hidden) | 0.2995 | ✅ |
| interp: gini (activation sparsity) | 0.0770 | ✅ |
| interp: hoyer sparsity | 0.0120 | ✅ |
| interp: CKA to previous checkpoint | 0.9101 | ✅ |
| localisation: selectivity index | 0.5303 | ✅ |
| localisation: mean overlap with other phenomena | 0.0046 | ✅ |
| localisation: active layers | 14.0000 | ✅ |
| localisation: layer centre of mass | 0.4572 | ✅ |
| behaviour: minimal-pair accuracy (chance 0.5) | 0.5096 | ✅ |
| causal selectivity (localized vs random ablation) | 0.0077 | ✅ |

## What this model shows

- **Behaviour is close to chance** (0.510 vs 0.5). Read every other number here with that in mind: a model that barely discriminates minimal pairs is not a strong test of anything.
- Participation ratio **0.300**: the representation uses ~30% of its available dimensions. High-dimensional relative to the other families here.
- Selectivity **0.530** — like every model in this release, phenomena are only moderately isolated. Nothing here shows sharp, dedicated linguistic circuitry.
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

![overview](figures/beetle-humanscale-eng_overview.png)
