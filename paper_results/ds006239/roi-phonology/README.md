# Brain–language-model alignment: ds006239 (roi-phonology)

Wang et al. 2025 — word-level phonological and semantic reading tasks in children and adolescents aged 10–17.

- Paper: https://www.sciencedirect.com/science/article/pii/S2352340925009692
- Data: https://openneuro.org/datasets/ds006239/versions/1.0.5
- Generated: 2026-09-07
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

**GATE: FAILED. 0/38 stimulus tests are significant** after Holm
correction — not the acoustic model of the audio the children actually
heard, not the study's own experimental contrast.

**The alignment numbers below are therefore uninterpretable as
evidence about language models.** They measure a representational
geometry that does not demonstrably encode the stimuli. They are
published for completeness and for whoever fixes the estimator, not as
a result. Do not cite them as evidence that models fail to align with
the developing brain.

Measured cause, from `control/`:

- RDM effective rank: **53** of 84 stimuli


Note that this is NOT ds003604's failure mode. There, the RDM
effective rank was ~3 of 40-48 stimuli -- near-degenerate betas
that could not express stimulus-level structure at all. The rank
recorded above is a large fraction of the stimulus count, so these
RDMs do carry stimulus structure and the control failing here means
the specific controls tested did not reach significance, not that
the measurement is uninterpretable. Check `control/` for which
controls ran: an acoustic or visual control needs the dataset's
stimulus files present, and reports zero features if they are not.

## What was built

8 task × session cells, each an RDM over the stimuli
shared by that cell's subjects, with voxel patterns z-scored **within
run** before aggregation (without that, the RDM measures scanner drift
rather than language) and an inter-subject noise ceiling.

| task     | session   |   n_stim |   ceiling_lower |   ceiling_upper |   ceiling_n |
|:---------|:----------|---------:|----------------:|----------------:|------------:|
| Orth     | ses-11+   |       96 |        0.527509 |        0.760826 |           3 |
| Orth     | ses-11    |       96 |        0.395701 |        0.694777 |           3 |
| Phon     | ses-11+   |       96 |        0.527509 |        0.760826 |           3 |
| Phon     | ses-11    |       96 |        0.395701 |        0.694777 |           3 |
| Sem      | ses-11+   |       72 |        0.313565 |        0.658019 |           3 |
| Sem      | ses-11    |       72 |        0.185478 |        0.586945 |           3 |
| SemLocal | ses-11+   |       48 |        0.208521 |        0.601744 |           3 |
| SemLocal | ses-11    |       48 |        0.216461 |        0.607091 |           3 |

Model grid: **15 families**, 2096 alignment rows across 8 cells.

| | |
|---|---|
| mean noise ceiling | 0.346 |
| best alignment anywhere | 0.0653 |
| as a fraction of ceiling | 28.2% |
| families equivalent to zero (TOST ±0.05) | 15/15 |
| Pythia scale trend | ρ = -0.129, p = 0.43 |

### Per family

| family                |   n_checkpoints |   rsa_mean |   rsa_sd |   rsa_abs_max |   frac_of_ceiling_abs_max |   p_equivalence_tost |
|:----------------------|----------------:|-----------:|---------:|--------------:|--------------------------:|---------------------:|
| pico-decoder-tiny     |              21 |     0.0055 |   0.0039 |        0.0434 |                    0.2038 |               0      |
| beetle-humanscale-eng |              18 |     0.0008 |   0.0052 |        0.0398 |                    0.1841 |               0      |
| pico-decoder-large    |              21 |    -0.0013 |   0.0084 |        0.0521 |                    0.2498 |               0      |
| pico-decoder-small    |              21 |    -0.0028 |   0.0115 |        0.0523 |                    0.2822 |               0      |
| pythia-70m-full       |              21 |    -0.0036 |   0.0067 |        0.0519 |                    0.2396 |               0      |
| pico-decoder-medium   |              21 |    -0.0039 |   0.0073 |        0.0351 |                    0.1587 |               0      |
| pythia-410m-full      |              21 |    -0.0049 |   0.0086 |        0.038  |                    0.1899 |               0      |
| pythia-1b-full        |              21 |    -0.007  |   0.0106 |        0.0482 |                    0.2179 |               0      |
| beetle-fineweb3-eng   |              19 |    -0.0074 |   0.0051 |        0.0569 |                    0.2628 |               0      |
| pythia-1.4b-full      |              21 |    -0.0102 |   0.0073 |        0.0482 |                    0.1831 |               0      |
| babylm-gpt2           |               9 |    -0.0122 |   0.0124 |        0.042  |                    0.2016 |               0      |
| pythia-160m-full      |              21 |    -0.0141 |   0.0077 |        0.0568 |                    0.241  |               0      |
| babylm-gpt2-5         |               9 |    -0.0308 |   0.02   |        0.0644 |                    0.1395 |               0.015  |
| babylm-gpt2-7         |               9 |    -0.0309 |   0.0201 |        0.0623 |                    0.1217 |               0.0156 |
| babylm-gpt2-3         |               9 |    -0.0311 |   0.0205 |        0.0653 |                    0.1318 |               0.0176 |

## Dataset-specific notes

Contains **LocalSem**, the only genuinely run/stimulus-CROSSED language cell across all four datasets in this project: its stimuli recur across runs, so run identity and stimulus identity are separable and the scanner-run confound that invalidated the first ds003604 analysis cannot arise. Per-subject age is NOT recoverable from the release — participants.tsv has birthdate but no scan date and there are no *_scans.tsv files — so this dataset is cohort-level only and cannot carry the developmental axis as published.

## Files

| path | what | present here |
|---|---|---|
| `alignment_by_checkpoint.csv` | every model × checkpoint × cell, with ceiling | ✓ |
| `alignment_by_family.csv` | per family, with equivalence tests | ✓ |
| `alignment_by_cell.csv` | per task × session | ✓ |
| `ceilings_ds006239.csv` | noise ceiling per cell | ✓ |
| `control/` | the positive control and RDM dimensionality — the gate | ✓ |
| `scale_ladder.csv` | the Pythia 70M→1.4B scale test | ✓ |
| `fig_*.pdf, fig_*.png` | figures | ✓ |

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