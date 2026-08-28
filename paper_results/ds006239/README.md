# Brain–language-model alignment: ds006239

Wang et al. 2025 — word-level phonological and semantic reading tasks in children and adolescents aged 10–17.

- Paper: https://www.sciencedirect.com/science/article/pii/S2352340925009692
- Data: https://openneuro.org/datasets/ds006239/versions/1.0.5
- Generated: 2026-08-28
- Pipeline: https://github.com/suchirsalhan/cdl-representations-brains-babylms

## Read this first: does the measurement work?

Every alignment number in this dataset is only as meaningful as the brain
RDMs it was computed against. So before any model result, the same
pipeline is asked whether *anything* stimulus-driven correlates with those
RDMs — stimulus duration, intensity, word length, frequency, phoneme and
syllable counts, an acoustic model of the audio where the stimuli are
audio, and the study's own condition contrast — each tested by a
permutation test that shuffles stimulus identity.

**GATE: NOT RUN.** Treat everything below as provisional.

## What was built

8 task × session cells, each an RDM over the stimuli
shared by that cell's subjects, with voxel patterns z-scored **within
run** before aggregation (without that, the RDM measures scanner drift
rather than language) and an inter-subject noise ceiling.

| task     | session   |   n_stim |   ceiling_lower |   ceiling_upper |   ceiling_n |
|:---------|:----------|---------:|----------------:|----------------:|------------:|
| Orth     | ses-11+   |       96 |        0.562195 |        0.61485  |          22 |
| Orth     | ses-11    |       96 |        0.522525 |        0.589811 |          18 |
| Phon     | ses-11+   |       96 |        0.562195 |        0.61485  |          22 |
| Phon     | ses-11    |       96 |        0.522525 |        0.589811 |          18 |
| Sem      | ses-11+   |       48 |        0.356634 |        0.438561 |          22 |
| Sem      | ses-11    |       48 |        0.289344 |        0.394831 |          18 |
| SemLocal | ses-11+   |       48 |        0.305571 |        0.38963  |          23 |
| SemLocal | ses-11    |       48 |        0.230569 |        0.364206 |          15 |

## Dataset-specific notes

Contains **LocalSem**, the only genuinely run/stimulus-CROSSED language cell across all four datasets in this project: its stimuli recur across runs, so run identity and stimulus identity are separable and the scanner-run confound that invalidated the first ds003604 analysis cannot arise. Per-subject age is NOT recoverable from the release — participants.tsv has birthdate but no scan date and there are no *_scans.tsv files — so this dataset is cohort-level only and cannot carry the developmental axis as published.

## Files

| path | what |
|---|---|
| `alignment_by_checkpoint.csv` | every model × checkpoint × cell, with ceiling |
| `alignment_by_family.csv` | per family, with equivalence tests |
| `alignment_by_cell.csv` | per task × session |
| `ceilings_*.csv` | noise ceiling per cell |
| `control/` | the positive control and RDM dimensionality — the gate |
| `scale_ladder.csv` | the Pythia 70M→1.4B scale test |
| `fig_*.pdf`, `fig_*.png` | figures |

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