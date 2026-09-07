# Brain–language-model alignment: ds003604 (roi-phonology)

The flagship dataset of this project — auditory sentence/word-pair listening in children aged 5, 7 and 9, four phenomena (semantic, phonological, grammatical, plausibility).

- Paper: https://openneuro.org/datasets/ds003604
- Data: https://openneuro.org/datasets/ds003604
- Generated: 2026-08-30
- Pipeline: https://github.com/suchirsalhan/cdl-representations-brains-babylms
- Masking: **roi-phonology** -- see DATASETS.md section 10 for the three-level standard (phonology/language/all) this is part of, and how it differs from the whole-brain reference.

## Read this first: does the measurement work?

Every alignment number in this dataset is only as meaningful as the brain
RDMs it was computed against. So before any model result, the same
pipeline is asked whether *anything* stimulus-driven correlates with those
RDMs — stimulus duration, intensity, word length, frequency, phoneme and
syllable counts, an acoustic model of the audio where the stimuli are
audio, and the study's own condition contrast — each tested by a
permutation test that shuffles stimulus identity.

**GATE: FAILED. 0/108 stimulus tests are significant** after Holm
correction — not the acoustic model of the audio the children actually
heard, not the study's own experimental contrast.

**No alignment numbers exist in this results directory** -- the
language-model grid produced zero rows for this run (see this run's
own logs for why: a real crash, an environment problem, or simply
never having been run). That is independent of the gate result
above, which is real either way.

Measured cause, from `control/`:

- RDM effective rank: **5** of 66 stimuli
- voxels per pattern: 8,909
- leading component vs the pattern's global signal: |ρ| = 0.92

This reproduces what was found on ds003604: the per-stimulus GLM
betas are near-degenerate, so the RDM cannot express stimulus-level
structure regardless of what it is compared against. The estimator
is shared across datasets, which is why the failure repeats.

## What was built

12 task × session cells, each an RDM over the stimuli
shared by that cell's subjects, with voxel patterns z-scored **within
run** before aggregation (without that, the RDM measures scanner drift
rather than language) and an inter-subject noise ceiling.

| task   | session   |   n_stim |   ceiling_lower |   ceiling_upper |   ceiling_n |
|:-------|:----------|---------:|----------------:|----------------:|------------:|
| Gram   | ses-5     |       60 |        0.644986 |        0.808684 |           3 |
| Gram   | ses-7     |       60 |      nan        |      nan        |         nan |
| Gram   | ses-9     |       60 |        0.810485 |        0.911715 |           3 |
| Phon   | ses-5     |       72 |        0.724946 |        0.861042 |           3 |
| Phon   | ses-7     |       72 |        0.846016 |        0.911894 |           3 |
| Phon   | ses-9     |       72 |        0.727362 |        0.87046  |           3 |
| Plaus  | ses-5     |       60 |        0.753902 |        0.869767 |           3 |
| Plaus  | ses-7     |       60 |        0.751198 |        0.851014 |           4 |
| Plaus  | ses-9     |       60 |        0.860116 |        0.914215 |           4 |
| Sem    | ses-5     |       72 |        0.776346 |        0.897154 |           3 |
| Sem    | ses-7     |       72 |        0.829125 |        0.893889 |           4 |
| Sem    | ses-9     |       72 |        0.788227 |        0.901604 |           3 |

## Dataset-specific notes

The only dataset with a longitudinal DEVELOPMENTAL axis across three discrete ages rather than a continuous one (ses-5/ses-7/ses-9). Each stimulus is presented in exactly one scanner run, which is the source of the run confound the within-run normalisation in this pipeline corrects (see the HF repo README for BrainAlign/ds003604-session-rdms for the measured before/after). Every other dataset here was added to generalise past this one, not to replace it — treat its numbers as the reference point the others are compared against, not as one dataset among four.

## Files

**No alignment or figure files exist in this results directory** --
the table below is what a completed run produces; this run's own
logs say why these are absent.

| path | what | present here |
|---|---|---|
| `alignment_by_checkpoint.csv` | every model × checkpoint × cell, with ceiling | — |
| `alignment_by_family.csv` | per family, with equivalence tests | — |
| `alignment_by_cell.csv` | per task × session | — |
| `ceilings_ds003604.csv` | noise ceiling per cell | ✓ |
| `control/` | the positive control and RDM dimensionality — the gate | ✓ |
| `scale_ladder.csv` | the Pythia 70M→1.4B scale test | — |
| `fig_*.pdf, fig_*.png` | figures | — |

## Method

Representational similarity analysis. For each cell, a brain RDM over
stimuli (correlation distance between per-stimulus GLM beta patterns,
within-run z-scored, aggregated across subjects) is compared by Spearman
correlation with a model RDM over the same stimuli, taken from each
checkpoint's hidden states. Alignment is reported raw and as a fraction of
the inter-subject noise ceiling, and judged against a null built from the
PARC suite — 18 models differing only by random seed, which is what 'no
effect' looks like on this measurement.

Null and fixation trials are excluded from the stimulus set. For paired
designs the stimulus identity is the pair, not either word alone.