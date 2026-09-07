# Brain–language-model alignment: ds001894 (roi-phonology)

Lytle et al. 2019 — longitudinal word-level phonological processing in children scanned twice, at roughly 10 and 12 years old.

- Paper: https://www.nature.com/articles/s41597-019-0338-5
- Data: https://openneuro.org/datasets/ds001894/versions/1.4.2
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

**GATE: FAILED. 0/32 stimulus tests are significant** after Holm
correction — not the acoustic model of the audio the children actually
heard, not the study's own experimental contrast.

**The alignment numbers below are therefore uninterpretable as
evidence about language models.** They measure a representational
geometry that does not demonstrably encode the stimuli. They are
published for completeness and for whoever fixes the estimator, not as
a result. Do not cite them as evidence that models fail to align with
the developing brain.

Measured cause, from `control/`:

- RDM effective rank: **60** of 96 stimuli


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

| task   | session   |   n_stim |   ceiling_lower |   ceiling_upper |   ceiling_n |
|:-------|:----------|---------:|----------------:|----------------:|------------:|
| Orth   | ses-11+   |       96 |        0.174262 |        0.545734 |           4 |
| Orth   | ses-11    |       96 |        0.213686 |        0.52405  |           5 |
| Orth   | ses-7     |       96 |      nan        |      nan        |         nan |
| Orth   | ses-9     |       96 |        0.165076 |        0.607179 |           3 |
| Phon   | ses-11+   |       96 |        0.174262 |        0.545734 |           4 |
| Phon   | ses-11    |       96 |        0.213686 |        0.52405  |           5 |
| Phon   | ses-7     |       96 |      nan        |      nan        |         nan |
| Phon   | ses-9     |       96 |        0.165076 |        0.607179 |           3 |

Model grid: **15 families**, 2096 alignment rows across 8 cells.

| | |
|---|---|
| mean noise ceiling | 0.184 |
| best alignment anywhere | 0.0493 |
| as a fraction of ceiling | 29.8% |
| families equivalent to zero (TOST ±0.05) | 15/15 |
| Pythia scale trend | ρ = +0.300, p = 0.06 |

### Per family

| family                |   n_checkpoints |   rsa_mean |   rsa_sd |   rsa_abs_max |   frac_of_ceiling_abs_max |   p_equivalence_tost |
|:----------------------|----------------:|-----------:|---------:|--------------:|--------------------------:|---------------------:|
| pythia-1b-full        |              21 |     0.0096 |   0.0076 |        0.042  |                    0.1966 |               0      |
| pythia-160m-full      |              21 |     0.0092 |   0.0036 |        0.0302 |                    0.1534 |               0      |
| pythia-1.4b-full      |              21 |     0.0079 |   0.0069 |        0.0323 |                    0.1511 |               0      |
| pico-decoder-tiny     |              21 |     0.0055 |   0.0062 |        0.0331 |                    0.2003 |               0      |
| beetle-humanscale-eng |              18 |     0.0038 |   0.0028 |        0.0276 |                    0.1585 |               0      |
| pythia-410m-full      |              21 |     0.0017 |   0.0058 |        0.0407 |                    0.1904 |               0      |
| pythia-70m-full       |              21 |     0.0016 |   0.0012 |        0.0327 |                    0.1979 |               0      |
| pico-decoder-small    |              21 |     0.0016 |   0.0025 |        0.0299 |                    0.1686 |               0      |
| babylm-gpt2           |               9 |     0.0009 |   0.0128 |        0.023  |                    0.1079 |               0      |
| pico-decoder-large    |              21 |     0.0006 |   0.0055 |        0.0264 |                    0.1433 |               0      |
| pico-decoder-medium   |              21 |    -0      |   0.0057 |        0.0317 |                    0.192  |               0      |
| beetle-fineweb3-eng   |              19 |    -0.0015 |   0.0085 |        0.0214 |                    0.1298 |               0      |
| babylm-gpt2-3         |               9 |    -0.0185 |   0.0107 |        0.0493 |                    0.2985 |               0      |
| babylm-gpt2-7         |               9 |    -0.0187 |   0.0114 |        0.0484 |                    0.2933 |               0.0001 |
| babylm-gpt2-5         |               9 |    -0.0202 |   0.0113 |        0.0486 |                    0.2942 |               0.0001 |

## Dataset-specific notes

The only longitudinal dataset here: the same children at two timepoints (ses-T1, ses-T2), which is the closest real analogue to a language model's checkpoint trajectory. Per-subject age at scan is available. Trial types cross orthographic with phonological similarity (O+P+/O+P-/O-P+/O-P-), so Phon and Orth contrasts are decorrelated by design. ses-T2 has only the VV tasks.

## Files

| path | what | present here |
|---|---|---|
| `alignment_by_checkpoint.csv` | every model × checkpoint × cell, with ceiling | ✓ |
| `alignment_by_family.csv` | per family, with equivalence tests | ✓ |
| `alignment_by_cell.csv` | per task × session | ✓ |
| `ceilings_ds001894.csv` | noise ceiling per cell | ✓ |
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