# Brain–language-model alignment: ds002236 (roi-auditory)

Lytle et al. 2020 — orthographic, phonological and semantic word processing in school-aged children (8.7–15.5), auditory and visual.

- Paper: https://pubmed.ncbi.nlm.nih.gov/31956678/
- Data: https://openneuro.org/datasets/ds002236/versions/1.0.1
- Generated: 2026-09-07
- Pipeline: https://github.com/suchirsalhan/cdl-representations-brains-babylms
- Masking: **roi-auditory** -- see DATASETS.md section 10 for the three-level standard (phonology/language/all) this is part of, and how it differs from the whole-brain reference.

## Read this first: does the measurement work?

Every alignment number in this dataset is only as meaningful as the brain
RDMs it was computed against. So before any model result, the same
pipeline is asked whether *anything* stimulus-driven correlates with those
RDMs — stimulus duration, intensity, word length, frequency, phoneme and
syllable counts, an acoustic model of the audio where the stimuli are
audio, and the study's own condition contrast — each tested by a
permutation test that shuffles stimulus identity.

**GATE: FAILED. 0/30 stimulus tests are significant** after Holm
correction — not the acoustic model of the audio the children actually
heard, not the study's own experimental contrast.

**The alignment numbers below are therefore uninterpretable as
evidence about language models.** They measure a representational
geometry that does not demonstrably encode the stimuli. They are
published for completeness and for whoever fixes the estimator, not as
a result. Do not cite them as evidence that models fail to align with
the developing brain.

Measured cause, from `control/`:

- RDM effective rank: **44** of 84 stimuli


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

6 task × session cells, each an RDM over the stimuli
shared by that cell's subjects, with voxel patterns z-scored **within
run** before aggregation (without that, the RDM measures scanner drift
rather than language) and an inter-subject noise ceiling.

| task   | session   |   n_stim |   ceiling_lower |   ceiling_upper |   ceiling_n |
|:-------|:----------|---------:|----------------:|----------------:|------------:|
| Phon   | ses-11+   |       96 |        0.243964 |        0.652672 |           3 |
| Phon   | ses-11    |       96 |      nan        |      nan        |         nan |
| Phon   | ses-9     |       96 |      nan        |      nan        |         nan |
| Sem    | ses-11+   |       72 |        0.239779 |        0.655166 |           3 |
| Sem    | ses-11    |       72 |      nan        |      nan        |         nan |
| Sem    | ses-9     |       72 |      nan        |      nan        |         nan |

Model grid: **15 families**, 1284 alignment rows across 6 cells.

| | |
|---|---|
| mean noise ceiling | 0.242 |
| best alignment anywhere | 0.0700 |
| as a fraction of ceiling | 27.0% |
| families equivalent to zero (TOST ±0.05) | 13/15 |
| Pythia scale trend | ρ = +0.287, p = 0.20 |

### Per family

| family                |   n_checkpoints |   rsa_mean |   rsa_sd |   rsa_abs_max |   frac_of_ceiling_abs_max |   p_equivalence_tost |
|:----------------------|----------------:|-----------:|---------:|--------------:|--------------------------:|---------------------:|
| pythia-1b-full        |              21 |     0.0201 |   0.0078 |        0.0489 |                  nan      |             nan      |
| pythia-1.4b-full      |              21 |     0.0177 |   0.0101 |        0.07   |                  nan      |             nan      |
| babylm-gpt2-3         |               9 |     0.0147 |   0.0215 |        0.0462 |                    0.17   |               0.0051 |
| babylm-gpt2-7         |               9 |     0.0143 |   0.022  |        0.048  |                    0.1607 |               0.0053 |
| babylm-gpt2-5         |               9 |     0.0134 |   0.0221 |        0.049  |                    0.1578 |               0.0049 |
| pythia-410m-full      |               1 |     0.0134 |   0.0218 |        0.0398 |                    0.0998 |               0.0046 |
| babylm-gpt2           |               9 |     0.0113 |   0.0209 |        0.041  |                    0.1679 |               0.0031 |
| pico-decoder-medium   |              21 |     0.0104 |   0.019  |        0.066  |                    0.2703 |               0.0019 |
| beetle-humanscale-eng |              18 |     0.0091 |   0.01   |        0.0486 |                    0.2005 |               0.0001 |
| pico-decoder-large    |              21 |     0.0087 |   0.0199 |        0.0634 |                    0.2599 |               0.0019 |
| pythia-160m-full      |              21 |     0.0073 |   0.0108 |        0.0592 |                    0.2427 |               0.0001 |
| pico-decoder-small    |              21 |     0.0069 |   0.019  |        0.0573 |                    0.2243 |               0.0013 |
| pythia-70m-full       |              21 |     0.0068 |   0.0128 |        0.0446 |                    0.1798 |               0.0002 |
| beetle-fineweb3-eng   |              19 |     0.0029 |   0.0069 |        0.0399 |                    0.1152 |               0      |
| pico-decoder-tiny     |              21 |     0.0016 |   0.0162 |        0.0581 |                    0.2383 |               0.0004 |

## Dataset-specific notes

The accession is not stated in the data article; it was resolved to ds002236 by matching OpenNeuro's own dataset name ("Cross-Sectional Multidomain Lexical Processing") AND the per-subject age range in participants.tsv (8.67–15.5) against the range the article reports. Best developmental axis of the four datasets: explicit per-subject age at scan, continuous rather than binned. Six tasks crossing modality (auditory/visual) with judgement (rhyme/spelling/semantic) — a modality control no other dataset here provides. A third of trials are coded null (Tones/nullsilence.WAV) and are excluded from the stimulus set.

## Files

| path | what | present here |
|---|---|---|
| `alignment_by_checkpoint.csv` | every model × checkpoint × cell, with ceiling | ✓ |
| `alignment_by_family.csv` | per family, with equivalence tests | ✓ |
| `alignment_by_cell.csv` | per task × session | ✓ |
| `ceilings_ds002236.csv` | noise ceiling per cell | ✓ |
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