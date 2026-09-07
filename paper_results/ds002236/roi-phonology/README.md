# Brain–language-model alignment: ds002236 (roi-phonology)

Lytle et al. 2020 — orthographic, phonological and semantic word processing in school-aged children (8.7–15.5), auditory and visual.

- Paper: https://pubmed.ncbi.nlm.nih.gov/31956678/
- Data: https://openneuro.org/datasets/ds002236/versions/1.0.1
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

- RDM effective rank: **48** of 84 stimuli


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
| Phon   | ses-11+   |       96 |        0.282138 |        0.668909 |           3 |
| Phon   | ses-11    |       96 |      nan        |      nan        |         nan |
| Phon   | ses-9     |       96 |      nan        |      nan        |         nan |
| Sem    | ses-11+   |       72 |        0.273181 |        0.661182 |           3 |
| Sem    | ses-11    |       72 |      nan        |      nan        |         nan |
| Sem    | ses-9     |       72 |      nan        |      nan        |         nan |

Model grid: **15 families**, 1572 alignment rows across 6 cells.

| | |
|---|---|
| mean noise ceiling | 0.278 |
| best alignment anywhere | 0.0720 |
| as a fraction of ceiling | 25.5% |
| families equivalent to zero (TOST ±0.05) | 15/15 |
| Pythia scale trend | ρ = +0.008, p = 0.97 |

### Per family

| family                |   n_checkpoints |   rsa_mean |   rsa_sd |   rsa_abs_max |   frac_of_ceiling_abs_max |   p_equivalence_tost |
|:----------------------|----------------:|-----------:|---------:|--------------:|--------------------------:|---------------------:|
| babylm-gpt2-3         |               9 |     0.0141 |   0.0167 |        0.0491 |                    0.1741 |               0.0016 |
| babylm-gpt2-7         |               9 |     0.014  |   0.0171 |        0.0493 |                    0.1746 |               0.0018 |
| babylm-gpt2-5         |               9 |     0.0125 |   0.0172 |        0.0488 |                    0.173  |               0.0015 |
| pythia-1b-full        |              21 |     0.0107 |   0.0111 |        0.0599 |                    0.2121 |               0.0002 |
| pico-decoder-medium   |              21 |     0.0105 |   0.0176 |        0.072  |                    0.255  |               0.0013 |
| babylm-gpt2           |               9 |     0.01   |   0.0144 |        0.0306 |                    0.1086 |               0.0005 |
| pico-decoder-large    |              21 |     0.01   |   0.0166 |        0.0696 |                    0.2387 |               0.001  |
| pico-decoder-small    |              21 |     0.0092 |   0.0152 |        0.0623 |                    0.214  |               0.0006 |
| beetle-humanscale-eng |              18 |     0.0091 |   0.0087 |        0.0547 |                    0.1489 |               0      |
| pythia-410m-full      |              21 |     0.0071 |   0.0068 |        0.0454 |                    0.1609 |               0      |
| pythia-70m-full       |              21 |     0.0071 |   0.0096 |        0.0607 |                    0.2153 |               0.0001 |
| pythia-160m-full      |              21 |     0.0065 |   0.0072 |        0.0532 |                    0.1885 |               0      |
| pythia-1.4b-full      |              21 |     0.0062 |   0.0133 |        0.0599 |                    0.2078 |               0.0002 |
| pico-decoder-tiny     |              21 |     0.0016 |   0.015  |        0.06   |                    0.1646 |               0.0003 |
| beetle-fineweb3-eng   |              19 |     0.0007 |   0.0067 |        0.046  |                    0.163  |               0      |

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