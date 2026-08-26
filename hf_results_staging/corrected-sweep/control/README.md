# Positive control — and why the ds003604 null cannot be published

Written by `scripts/positive_control.py` and `scripts/rdm_dimensionality.py`.
**Read this before quoting any alignment number from this project.**

## The question

Every alignment result we have is a null. A null is worth something only if the
instrument has demonstrated power. The noise ceiling (0.85) shows the brain RDMs
are *reliable* — two halves of the subject pool agree. It does not show they
carry *stimulus-driven* signal an RSA can recover.

So: does anything at all correlate with these RDMs?

## The test has power — that part passes

The same permutation test, run on the **uncorrected** RDMs, recovers acquisition
structure easily:

| control | uncorrected | corrected |
|---|---|---|
| run identity | **ρ = +0.666** (best cell +0.866) | −0.119 |
| presentation order | **ρ = +0.468** | −0.092 |

Two things follow. The permutation test detects real structure at ρ ≈ 0.7 when it
is there, so a null from it is a real null. And the within-run normalisation did
what it claims — the run confound is gone (slightly over-corrected).

## Nothing stimulus-driven survives

Ten stimulus controls × 12 cells on the corrected RDMs, permutation-tested
(5000 permutations) and Holm-corrected within the family:

**0 / 108 significant.**

| control | best ρ | best z | best p (uncorrected) |
|---|---|---|---|
| acoustic spectrum (log-mel) | +0.061 | 3.53 | 0.005 |
| n phonemes | +0.050 | 2.87 | 0.009 |
| n syllables | +0.056 | 2.73 | 0.014 |
| intensity | +0.080 | 2.45 | 0.028 |
| acoustic envelope | +0.050 | 2.11 | 0.037 |
| condition (the design's own contrast) | +0.036 | 1.85 | 0.054 |
| word length | +0.053 | 1.70 | 0.049 |
| duration | +0.036 | 1.54 | 0.055 |
| text edit distance | +0.027 | 1.44 | 0.116 |
| log frequency | −0.000 | −0.01 | 0.106 |

ds003604 is an auditory design — subjects heard 352 `.wav` files — so the
acoustic spectrum of the audio they actually heard is the textbook control. It
does not correlate. Neither does the study's own experimental contrast.

## Why: the RDMs are near-degenerate and track whole-brain signal level

| | |
|---|---|
| 72-stimulus RDMs | live in **4 dimensions** (90% of the spectrum) |
| 60-stimulus RDMs | live in **7 dimensions** |
| per-subject RDMs | same rank, 4–5 and 7 |
| voxels per pattern | **917,504** — whole-brain, no anatomical restriction |
| voxels near-constant across stimuli | 0.0% (so these are in-brain voxels) |
| pattern effective rank | 6 of ~40 stimuli; top 2 components = 61% |
| PC1 vs the pattern's global mean signal | \|ρ\| = **0.85** |
| RDM vs a pure global-amplitude RDM | ρ = **+0.43** |

The patterns are whole-brain GLM betas with no anatomical restriction — no grey
matter, ROI or language-network mask — and their leading component tracks the
pattern's overall signal level rather than where activity was. A 72-item RDM
with four dimensions cannot express stimulus-level structure regardless of what
it is compared against.

That resolves the paradox of a reliable but empty RDM. Whole-brain signal level
is extremely consistent across subjects — same scanner, same sequence, same
paradigm timing — so the ceiling is high. It is also stimulus-independent, so no
stimulus property correlates and no language model can align with it.

**A correction to an earlier version of this file.** It reported "100% non-zero
— no brain mask", concluding the patterns were unmasked whole-volume with air
included. That inference was wrong and the statistic was vacuous: a pattern
vector *is* the masked voxels, so its non-zero fraction is ~1 by construction.
The real test is a near-constant (air/skull) population, and there is none —
0.0%. So a mask was applied; it is simply whole-brain rather than anatomically
targeted. Everything else here is unchanged.

## What this means for the results

**The ds003604 language-model null is vacuous.** It is not evidence that models
fail to align with the developing brain; it is a measurement of whole-brain
signal level that no model could have matched. Everything downstream inherits
this — the 15-family grid, the Pythia scale ladder, the PARC seed-null, the
training trends. Those analyses are correct as analyses; their input is not
a representational geometry.

Note this also explains the training trends in
`paper_results/corrected/README.md`: untrained models scored *higher* than
trained ones. Against an amplitude-dominated RDM that is what you would expect.

## What has to happen before any alignment claim

1. **Restrict the voxels anatomically.** 917k whole-brain voxels per pattern is
   the root cause: a stimulus-specific response in language cortex is a rounding
   error against whole-brain signal level. `run_analysis.py --aal-rois` and
   `scripts/run_roi_pipeline.py` already support this and the grid has never
   used them (TODO §3).
2. **Remove global signal per pattern**, or model it explicitly. Correlation
   distance does not remove it — it still leaves ρ = 0.43 with amplitude.
3. **Keep the GLM** — per-stimulus betas are already what is extracted
   (`use_glm=True`, SPM HRF, cosine drift), so that part is sound. The problem is
   the voxel set and the global component, not the estimator.
4. **Re-run this control.** It is cheap, it is the gate, and it must come back
   positive — the acoustic spectrum should correlate with auditory cortex —
   before any alignment number is reported again.
5. Only then re-run the model grid.

The raw BOLD is gone (the streaming design deletes it, by necessity — see
`PICKUP.md`), so steps 1–3 mean re-downloading and re-preprocessing ds003604.
That is a tier-0 job, ~2 h on this box at the observed rate.

Cheaper first probe, no download needed: 249 pattern files survive under
`data/processed/fmri_wrn/ds003604/`. Removing the global component from those and
re-running this control would test the diagnosis before committing to the re-run.

## Files

| file | what |
|---|---|
| `control_by_cell.csv` | every control × cell × RDM-variant, with permutation p |
| `control_summary.csv` | per control, across cells |
| `acquisition_controls.csv` | run identity and presentation order, both variants |
| `rdm_dimensionality.csv` | RDM rank per cell |
| `pattern_dimensionality.csv` | raw pattern rank, mask status, global-signal share |
| `summary.json`, `dimensionality_summary.json` | headline numbers |
| `fig_positive_control.*` | what the pipeline can and cannot detect |
