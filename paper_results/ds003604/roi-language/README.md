# Brain–language-model alignment: ds003604 (roi-language)

The flagship dataset of this project — auditory sentence/word-pair listening in children aged 5, 7 and 9, four phenomena (semantic, phonological, grammatical, plausibility).

- Paper: https://openneuro.org/datasets/ds003604
- Data: https://openneuro.org/datasets/ds003604
- Generated: 2026-09-06
- Pipeline: https://github.com/suchirsalhan/cdl-representations-brains-babylms
- Masking: **roi-language** -- see DATASETS.md section 10 for the three-level standard (phonology/language/all) this is part of, and how it differs from the whole-brain reference.

## Read this first: does the measurement work?

Every alignment number in this dataset is only as meaningful as the brain
RDMs it was computed against. So before any model result, the same
pipeline is asked whether *anything* stimulus-driven correlates with those
RDMs — stimulus duration, intensity, word length, frequency, phoneme and
syllable counts, an acoustic model of the audio where the stimuli are
audio, and the study's own condition contrast — each tested by a
permutation test that shuffles stimulus identity.

**GATE: FAILED. 0/10 stimulus tests are significant** after Holm
correction — not the acoustic model of the audio the children actually
heard, not the study's own experimental contrast.

**The alignment numbers below are therefore uninterpretable as
evidence about language models.** They measure a representational
geometry that does not demonstrably encode the stimuli. They are
published for completeness and for whoever fixes the estimator, not as
a result. Do not cite them as evidence that models fail to align with
the developing brain.

Measured cause, from `control/`:

- RDM effective rank: **4** of 72 stimuli
- voxels per pattern: 18,370
- leading component vs the pattern's global signal: |ρ| = 0.75

This reproduces what was found on ds003604: the per-stimulus GLM
betas are near-degenerate, so the RDM cannot express stimulus-level
structure regardless of what it is compared against. The estimator
is shared across datasets, which is why the failure repeats.

## What was built

1 task × session cells, each an RDM over the stimuli
shared by that cell's subjects, with voxel patterns z-scored **within
run** before aggregation (without that, the RDM measures scanner drift
rather than language) and an inter-subject noise ceiling.

| task   | session   |   n_stim |   ceiling_lower |   ceiling_upper |   ceiling_n |
|:-------|:----------|---------:|----------------:|----------------:|------------:|
| Sem    | ses-5     |       72 |        0.818641 |        0.827832 |          70 |

Model grid: **11 families**, 178 alignment rows across 1 cells.

| | |
|---|---|
| mean noise ceiling | 0.819 |
| best alignment anywhere | 0.0320 |
| as a fraction of ceiling | 3.9% |
| families equivalent to zero (TOST ±0.05) | 0/11 |
| Pythia scale trend | ρ = -0.100, p = 0.87 |

### Per family

| family                |   n_checkpoints |   rsa_mean |   rsa_sd |   rsa_abs_max |   frac_of_ceiling_abs_max |   p_equivalence_tost |
|:----------------------|----------------:|-----------:|---------:|--------------:|--------------------------:|---------------------:|
| pythia-1b-full        |              21 |     0.0063 |      nan |        0.0246 |                    0.0301 |                  nan |
| beetle-fineweb3-eng   |              19 |     0.0006 |      nan |        0.032  |                    0.0391 |                  nan |
| pythia-160m-full      |              21 |    -0.0013 |      nan |        0.0182 |                    0.0222 |                  nan |
| pythia-410m-full      |              21 |    -0.0017 |      nan |        0.0084 |                    0.0103 |                  nan |
| babylm-gpt2           |               9 |    -0.0042 |      nan |        0.0076 |                    0.0092 |                  nan |
| beetle-humanscale-eng |              18 |    -0.0042 |      nan |        0.0159 |                    0.0194 |                  nan |
| pythia-70m-full       |              21 |    -0.005  |      nan |        0.0255 |                    0.0311 |                  nan |
| pythia-1.4b-full      |              21 |    -0.0064 |      nan |        0.0145 |                    0.0177 |                  nan |
| babylm-gpt2-7         |               9 |    -0.0122 |      nan |        0.0176 |                    0.0214 |                  nan |
| babylm-gpt2-5         |               9 |    -0.0124 |      nan |        0.0167 |                    0.0204 |                  nan |
| babylm-gpt2-3         |               9 |    -0.0131 |      nan |        0.0183 |                    0.0224 |                  nan |

## Dataset-specific notes

The only dataset with a longitudinal DEVELOPMENTAL axis across three discrete ages rather than a continuous one (ses-5/ses-7/ses-9). Each stimulus is presented in exactly one scanner run, which is the source of the run confound the within-run normalisation in this pipeline corrects (see the HF repo README for BrainAlign/ds003604-session-rdms for the measured before/after). Every other dataset here was added to generalise past this one, not to replace it — treat its numbers as the reference point the others are compared against, not as one dataset among four.

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