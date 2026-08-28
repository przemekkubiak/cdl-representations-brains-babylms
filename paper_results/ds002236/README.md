# Brain–language-model alignment: ds002236

Lytle et al. 2020 — orthographic, phonological and semantic word processing in school-aged children (8.7–15.5), auditory and visual.

- Paper: https://pubmed.ncbi.nlm.nih.gov/31956678/
- Data: https://openneuro.org/datasets/ds002236/versions/1.0.1
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

**GATE: FAILED. 0/6 stimulus tests are significant** after Holm
correction — not the acoustic model of the audio the children actually
heard, not the study's own experimental contrast.

**The alignment numbers below are therefore uninterpretable as
evidence about language models.** They measure a representational
geometry that does not demonstrably encode the stimuli. They are
published for completeness and for whoever fixes the estimator, not as
a result. Do not cite them as evidence that models fail to align with
the developing brain.

Measured cause, from `control/`:

- RDM effective rank: **54** of 72 stimuli


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
| Phon   | ses-11+   |       96 |        0.302747 |        0.40391  |          20 |
| Phon   | ses-11    |       96 |        0.234711 |        0.411819 |          10 |
| Phon   | ses-9     |       96 |        0.255574 |        0.441516 |           9 |
| Sem    | ses-11+   |       48 |        0.436333 |        0.506525 |          21 |
| Sem    | ses-11    |       48 |        0.390186 |        0.542448 |           8 |
| Sem    | ses-9     |       48 |        0.344433 |        0.482067 |          11 |

Model grid: **15 families**, 524 alignment rows across 2 cells.

| | |
|---|---|
| mean noise ceiling | 0.327 |
| best alignment anywhere | 0.1139 |
| as a fraction of ceiling | 44.5% |
| families equivalent to zero (TOST ±0.05) | 0/15 |
| Pythia scale trend | ρ = +0.148, p = 0.68 |

### Per family

| family                |   n_checkpoints |   rsa_mean |   rsa_sd |   rsa_abs_max |   frac_of_ceiling_abs_max |   p_equivalence_tost |
|:----------------------|----------------:|-----------:|---------:|--------------:|--------------------------:|---------------------:|
| babylm-gpt2           |               9 |     0.0287 |   0.0059 |        0.0534 |                    0.1689 |                  nan |
| pico-decoder-medium   |              21 |     0.0276 |   0.0346 |        0.1139 |                    0.4455 |                  nan |
| pythia-1b-full        |              21 |     0.0217 |   0.0268 |        0.0762 |                    0.298  |                  nan |
| pico-decoder-small    |              21 |     0.0213 |   0.0167 |        0.1044 |                    0.4086 |                  nan |
| babylm-gpt2-7         |               9 |     0.0207 |   0.0065 |        0.0445 |                    0.1292 |                  nan |
| pythia-410m-full      |              21 |     0.0188 |   0.0139 |        0.0741 |                    0.2901 |                  nan |
| babylm-gpt2-5         |               9 |     0.0186 |   0.0034 |        0.0368 |                    0.116  |                  nan |
| babylm-gpt2-3         |               9 |     0.018  |   0.007  |        0.0432 |                    0.1346 |                  nan |
| pico-decoder-large    |              21 |     0.0179 |   0.0334 |        0.0796 |                    0.3116 |                  nan |
| beetle-fineweb3-eng   |              19 |     0.0177 |   0.0213 |        0.0778 |                    0.3044 |                  nan |
| pythia-160m-full      |              21 |     0.0155 |   0.012  |        0.0716 |                    0.2801 |                  nan |
| pico-decoder-tiny     |              21 |     0.014  |   0.0121 |        0.0758 |                    0.2967 |                  nan |
| pythia-1.4b-full      |              21 |     0.0125 |   0.0271 |        0.0946 |                    0.37   |                  nan |
| pythia-70m-full       |              21 |     0.0121 |   0.0155 |        0.0753 |                    0.2947 |                  nan |
| beetle-humanscale-eng |              18 |     0.0101 |   0.006  |        0.0477 |                    0.1865 |                  nan |

## Dataset-specific notes

The accession is not stated in the data article; it was resolved to ds002236 by matching OpenNeuro's own dataset name ("Cross-Sectional Multidomain Lexical Processing") AND the per-subject age range in participants.tsv (8.67–15.5) against the range the article reports. Best developmental axis of the four datasets: explicit per-subject age at scan, continuous rather than binned. Six tasks crossing modality (auditory/visual) with judgement (rhyme/spelling/semantic) — a modality control no other dataset here provides. A third of trials are coded null (Tones/nullsilence.WAV) and are excluded from the stimulus set.

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