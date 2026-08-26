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

## The probe: it is not the voxel set, and not the global component

`scripts/probe_global_signal.py` tested the obvious fix on the 249 surviving
pattern files, so the answer cost minutes rather than a re-download:

| variant | effective rank | RDM vs amplitude | best \|ρ\| vs duration | p < .05 |
|---|---|---|---|---|
| raw | 3 | +0.418 | 0.077 | 0/40 |
| per-pattern mean removed | 3 | +0.418 | 0.077 | 0/40 |
| leading component removed | 3 | +0.191 | 0.059 | 0/40 |

Removing the global component halves its footprint in the RDM and recovers
**nothing**. (Mean-removal is identical to raw, as it must be — correlation
distance already subtracts each pattern's mean. That the numbers match exactly
is a check that the probe is measuring what it claims.)

So the diagnosis moves upstream. The voxel set and the global signal are real
features of these patterns but they are not what is destroying the stimulus
information: **the per-stimulus response estimates are near-degenerate on their
own terms** — effective rank ~3 of 40–48 stimuli per run.

The likely cause is the beta estimation itself. `extract_stimulus_activity_glm`
puts every stimulus in as its own regressor in one design matrix (LSA). In a
fast auditory design with one presentation per stimulus, those regressors are
strongly collinear, and LSA single-trial betas collapse toward a few shared
components — exactly the rank-3 structure measured here. The standard fix is
**LSS** (least-squares-separate: one model per stimulus, that stimulus against
everything else), which is far more stable under collinearity.

That is a methods change, not a parameter change, and it needs the raw BOLD —
which the streaming design has deleted. So it implies the tier-0 re-download
either way.

## What has to happen before any alignment claim

1. **Re-estimate the per-stimulus responses with LSS**, not LSA. This is the
   root cause as far as the evidence reaches: rank ~3 of 40–48 stimuli, and it
   survives every post-hoc correction tried.
2. **Restrict the voxels anatomically** while re-estimating — grey matter, or
   the AAL/language ROIs. `run_analysis.py --aal-rois` and
   `scripts/run_roi_pipeline.py` already support it and the grid has never used
   them (TODO §3). Not the root cause, but 917k whole-brain voxels dilutes any
   real effect and costs nothing to fix at the same time.
3. **Check the events**: confirm stimulus onsets, durations and ISI are what the
   GLM is being told they are. A design whose regressors are collinear enough to
   produce rank 3 deserves that check before the model is blamed.
4. **Re-run this control.** It is cheap, it is the gate, and it must come back
   positive — the acoustic spectrum should correlate with auditory cortex —
   before any alignment number is reported again.
5. Only then re-run the model grid.

The raw BOLD is gone (the streaming design deletes it, by necessity — see
`PICKUP.md`), so steps 1–3 mean re-downloading and re-preprocessing ds003604.
That is a tier-0 job, ~2 h on this box at the observed rate.

That probe has now been run (above) and rules out the cheap fixes, so the
re-download is the only remaining path.

## Files

| file | what |
|---|---|
| `control_by_cell.csv` | every control × cell × RDM-variant, with permutation p |
| `control_summary.csv` | per control, across cells |
| `acquisition_controls.csv` | run identity and presentation order, both variants |
| `rdm_dimensionality.csv` | RDM rank per cell |
| `pattern_dimensionality.csv` | raw pattern rank, mask status, global-signal share |
| `summary.json`, `dimensionality_summary.json` | headline numbers |
| `global_signal_probe.csv`, `.json` | the cheap-fix probe: rank and RSA before/after removal |
| `fig_positive_control.*` | what the pipeline can and cannot detect |
