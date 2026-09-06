# Brain–language-model alignment: ds001894 (whole-brain)

Lytle et al. 2019 — longitudinal word-level phonological processing in children scanned twice, at roughly 10 and 12 years old.

- Paper: https://www.nature.com/articles/s41597-019-0338-5
- Data: https://openneuro.org/datasets/ds001894/versions/1.4.2
- Generated: 2026-09-06
- Pipeline: https://github.com/suchirsalhan/cdl-representations-brains-babylms

## Read this first: does the measurement work?

Every alignment number in this dataset is only as meaningful as the brain
RDMs it was computed against. So before any model result, the same
pipeline is asked whether *anything* stimulus-driven correlates with those
RDMs — stimulus duration, intensity, word length, frequency, phoneme and
syllable counts, an acoustic model of the audio where the stimuli are
audio, and the study's own condition contrast — each tested by a
permutation test that shuffles stimulus identity.

**GATE: FAILED. 0/16 stimulus tests are significant** after Holm
correction — not the acoustic model of the audio the children actually
heard, not the study's own experimental contrast.

**The alignment numbers below are therefore uninterpretable as
evidence about language models.** They measure a representational
geometry that does not demonstrably encode the stimuli. They are
published for completeness and for whoever fixes the estimator, not as
a result. Do not cite them as evidence that models fail to align with
the developing brain.

Measured cause, from `control/`:

- RDM effective rank: **64** of 96 stimuli
- voxels per pattern: 123,043
- leading component vs the pattern's global signal: |ρ| = 0.20

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

4 task × session cells, each an RDM over the stimuli
shared by that cell's subjects, with voxel patterns z-scored **within
run** before aggregation (without that, the RDM measures scanner drift
rather than language) and an inter-subject noise ceiling.

| task   | session   |   n_stim |   ceiling_lower |   ceiling_upper |   ceiling_n |
|:-------|:----------|---------:|----------------:|----------------:|------------:|
| Phon   | ses-11+   |       96 |        0.261613 |        0.573154 |           4 |
| Phon   | ses-11    |       96 |        0.33555  |        0.574596 |           5 |
| Phon   | ses-7     |       96 |      nan        |      nan        |         nan |
| Phon   | ses-9     |       96 |        0.183971 |        0.58974  |           3 |

Model grid: **11 families**, 356 alignment rows across 2 cells.

| | |
|---|---|
| mean noise ceiling | 0.260 |
| best alignment anywhere | 0.0431 |
| as a fraction of ceiling | 23.4% |
| families equivalent to zero (TOST ±0.05) | 0/11 |
| Pythia scale trend | ρ = +0.222, p = 0.54 |

### Per family

| family                |   n_checkpoints |   rsa_mean |   rsa_sd |   rsa_abs_max |   frac_of_ceiling_abs_max |   p_equivalence_tost |
|:----------------------|----------------:|-----------:|---------:|--------------:|--------------------------:|---------------------:|
| pythia-1b-full        |              21 |     0.004  |   0.0096 |        0.0257 |                    0.096  |                  nan |
| pythia-160m-full      |              21 |     0.0021 |   0.009  |        0.033  |                    0.1547 |                  nan |
| pythia-1.4b-full      |              21 |     0.0002 |   0.0078 |        0.0262 |                    0.1423 |                  nan |
| pythia-410m-full      |              21 |    -0.0007 |   0.013  |        0.0265 |                    0.1376 |                  nan |
| beetle-humanscale-eng |              18 |    -0.003  |   0.0092 |        0.0347 |                    0.1884 |                  nan |
| beetle-fineweb3-eng   |              19 |    -0.0035 |   0.0012 |        0.0202 |                    0.1    |                  nan |
| pythia-70m-full       |              21 |    -0.0035 |   0.0107 |        0.0339 |                    0.1842 |                  nan |
| babylm-gpt2           |               9 |    -0.0057 |   0.0019 |        0.0185 |                    0.1004 |                  nan |
| babylm-gpt2-3         |               9 |    -0.0342 |   0.0021 |        0.0413 |                    0.2247 |                  nan |
| babylm-gpt2-5         |               9 |    -0.0348 |   0.0035 |        0.0418 |                    0.2271 |                  nan |
| babylm-gpt2-7         |               9 |    -0.0359 |   0.0034 |        0.0431 |                    0.2344 |                  nan |

## Dataset-specific notes

The only longitudinal dataset here: the same children at two timepoints (ses-T1, ses-T2), which is the closest real analogue to a language model's checkpoint trajectory. Per-subject age at scan is available. Trial types cross orthographic with phonological similarity (O+P+/O+P-/O-P+/O-P-), so Phon and Orth contrasts are decorrelated by design. ses-T2 has only the VV tasks.

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