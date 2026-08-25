# Noise ceiling + within-run correction (added 2026-08-25)

This directory is **additive**. It does not replace or correct any file elsewhere
in the dataset; the tables under `overall/`, `by-model/` and `diagnostics/` are
unchanged and still carry the run confound described in the top-level README.

What is new here is the measurement that was missing: a **noise ceiling**, and
alignment expressed as a fraction of it.

## Why it matters

The top-level README says of the brain-alignment tables: *"these tables do not
show that language models align with the brain, and they do not show that they
fail to. The measurement cannot answer it."* That was correct for those tables.
With a ceiling, the question becomes answerable, and the answer is now a
**well-powered null** rather than an uninterpretable one.

## The numbers

ds003604, task Phon, ses-5, 43 subjects, 72 stimuli (perceptual controls
excluded). Both RDMs are built from the *same* per-subject patterns and differ
only by the correction.

|                                   | raw          | within-run normalised |
|-----------------------------------|--------------|-----------------------|
| noise ceiling (LOO lower / upper) | 0.774 / 0.792| **0.849 / 0.859**     |
| "different run" predicts dissimilarity | **+0.557** | **−0.041**       |

Best layer per model against the corrected RDM:

| model | layer | Spearman rho | % of noise ceiling |
|---|---|---|---|
| beetle-humanscale-eng | 2 | +0.022 | **2.6%** |
| babylm-gpt2-3 | 5 | +0.013 | **1.5%** |
| pico-decoder-large | 0 | −0.001 | **−0.1%** |

**Read it this way.** Inter-subject reliability is 0.85 — well above the 0.2–0.4
typical in this literature — so there is a large, replicable target available to
predict. Three models spanning three architectures capture 0–3% of it.

Two supporting points:

- **The ceiling rises after correction** (0.774 → 0.849). If the raw reliability
  had been mostly the run artefact, removing the artefact would have *lowered*
  the ceiling. It rose, so the correction removes noise rather than signal.
- **The corrected layer profile is sensible** — positive at early/middle layers,
  negative at the output — whereas the raw profile is not. The correction makes
  the measurement behave, and it still finds nothing.

## Scope, stated plainly

One task × session cell, one session, 43 subjects. Whether this replicates across
all twelve cells is exactly what the full corrected sweep is running to answer;
those results will be added here as a separate, clearly-labelled set. Do not read
this single cell as the whole dataset.

## Files

| file | what |
|---|---|
| `ceiling_Phon_ses-5.csv` | ceiling + run-confound, raw vs corrected |
| `alignment_vs_ceiling_Phon_ses-5.csv` | per model per layer, rsa raw/corrected and fraction of ceiling |
| `summary_Phon_ses-5.json` | the same as a single record |
| `fig_ceiling_layerwise.png/pdf` | alignment vs the ceiling band, per model, with a zoomed layer profile |
| `fig_ceiling_confound.png/pdf` | run confound and ceiling, before vs after correction |

## Reproducing

```bash
python scripts/ceiling_report.py --pattern-dir <patterns> --task Phon --session ses-5
python scripts/make_ceiling_figures.py --dir paper_results/ceiling
```

Session RDMs built after 2026-08-25 carry their per-subject RDMs and a
`within_run_normalized` flag, so the ceiling is recomputable from the `.npz`
alone and a corrected RDM can never be confused with an uncorrected one.
`scripts/collect_ceilings.py` tabulates ceilings for a whole tree;
`scripts/verify_rdm_provenance.py` fails loudly on a tree with mixed provenance.
