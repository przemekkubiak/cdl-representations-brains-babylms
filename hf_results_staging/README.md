---
license: cc-by-4.0
tags:
  - neuroscience
  - fmri
  - brain-alignment
  - interpretability
  - language-models
configs:
  - config_name: summary_by_checkpoint
    default: true
    data_files: "overall/by_checkpoint.csv"
  - config_name: summary_by_family
    data_files: "overall/summary_by_family.csv"
  - config_name: brain_alignment
    data_files: "by-model/*/brain_alignment.csv"
  - config_name: interp_mechanistic
    data_files: "by-model/*/interp_mechanistic.csv"
  - config_name: interp_layerwise
    data_files: "by-model/*/interp_layerwise.csv"
  - config_name: localisation_isolation
    data_files: "by-model/*/localisation_isolation.csv"
  - config_name: localisation_onset
    data_files: "by-model/*/localisation_onset.csv"
  - config_name: behaviour
    data_files: "by-model/*/behaviour.csv"
  - config_name: ablation_alignment
    data_files: "by-model/*/ablation_alignment.csv"
  - config_name: ablation_behaviour
    data_files: "by-model/*/ablation_behaviour.csv"
  - config_name: claim_tests
    data_files: "overall/claim_tests.csv"
  - config_name: heldout_predictor
    data_files: "overall/heldout_predictor.csv"
  - config_name: diagnostics_layerwise
    data_files: "diagnostics/layerwise_alignment.csv"
  - config_name: diagnostics_run_confound
    data_files: "diagnostics/run_confound_check.csv"
---

# CDL DevAI results — brain × interpretability × localisation, per model per checkpoint

Developmental analysis of 10 language-model families against the **ds003604** auditory
language fMRI dataset. For every training checkpoint of every model we measured three
things and here report them side by side:

| axis | what it asks | source tables |
|---|---|---|
| **brain** | does the model's representational geometry match the brain's? | `brain_alignment` |
| **interp** | how is the representation organised internally? | `interp_mechanistic`, `interp_layerwise` |
| **localisation** | are linguistic phenomena isolated into dedicated units? | `localisation_isolation`, `localisation_onset` |

**Start with the `summary_by_checkpoint` config** (the default, and what the viewer shows
first): one row per model × checkpoint, with all three axes as columns. 262 rows, 10 models.

---

## ⚠️ READ THIS BEFORE USING THE BRAIN ALIGNMENT NUMBERS

**The brain alignment columns are confounded by scanner run and must not be read as a
result about language models.** This affects every `rsa*` column in `brain_alignment`,
every `brain_*` column in the summary tables, and the `ablation_alignment` table.

In ds003604 **each stimulus is presented in exactly one scanner run** (for Phon, run-01
carries 48 of the 96 stimuli and run-02 the other 48). Run membership is therefore
perfectly confounded with stimulus identity, and every cross-run stimulus pair inherits
that run's drift, baseline shift and scaling. Measured:

```
"different run" predicts brain dissimilarity, Spearman rho, all 12 task × session cells:
    Gram   +0.866  +0.812  +0.828        Plaus  +0.741  +0.689  +0.761
    Phon   +0.562  +0.488  +0.624        Sem    +0.511  +0.510  +0.601
```

By comparison **no stimulus property predicts these RDMs at all**: trial type ≈ −0.02,
text length ≈ 0, lexical overlap ≈ 0. The RDMs look highly reliable across independent
subject cohorts (rho 0.74–0.92 between sessions of the same task) — but that is largely
the reliability of an *acquisition artefact*, because run assignment is fixed by the
protocol and so repeats identically for every subject.

A language model cannot represent which scanner run a stimulus appeared in, so its RSA
against this structure is ≈0 by construction, and slightly negative in practice.

**Two obvious explanations were tested and ruled out:**

- *Cohort size.* Sem/ses-7 was built from 40 subjects; rebuilt from **98** it agrees with
  the 40-subject version at **rho = 0.928**, identical stimuli. Bigger cohorts change
  nothing.
- *Layer choice.* Alignment is flat across **every** layer — −0.012 to −0.021 across all
  12 layers of babylm-gpt2-3 and all 14 of beetle-humanscale-eng. There is no
  middle-layer peak being missed. See `diagnostics_layerwise`.

**What fixes it, and what happens when you do.** Z-scoring each voxel *within run* before
combining runs removes the confound cleanly — run predictiveness falls from **+0.562 to
−0.041**. Alignment against the corrected RDM then peaks at **+0.022** (layer 2,
beetle-humanscale-eng, 2,556 stimulus pairs), which is **not significant**. The layer
*profile* becomes sensible — early/middle layers positive, output layers negative — but
the magnitude is ≈0. See `diagnostics_run_confound`.

**Bottom line:** these tables do not show that language models align with the brain, and
they do not show that they fail to. The measurement cannot answer it. The corrected
analysis, on one cell and one model, finds no detectable alignment. Do not cite the raw
numbers in either direction.

### Which columns are safe

| axis | affected by the run confound? |
|---|---|
| `brain_*`, `rsa*`, `ablation_alignment` | ❌ **Confounded. Do not use as a model result.** |
| `interp_*` (norm, gini, hoyer, per, condition_number, cka_to_prev) | ✅ Safe — computed from LM activations only. The fMRI data is not involved. |
| `loc_*` / `localisation_*` (selectivity_index, overlap, gini, entropy, layer_com, n_active_layers) | ✅ Safe — LM-internal localisation against text contrasts. No fMRI. |
| `behaviour` (mp_accuracy) | ✅ Safe — minimal-pair accuracy, text only. |
| `ablation_behaviour` (causal_selectivity) | ✅ Safe — ablation vs behaviour, no fMRI. |

The **one uncontaminated positive result** in this release is the causal behaviour test:
ablating a phenomenon's localized circuit costs **1.13%** minimal-pair accuracy versus
**0.55%** for a random circuit of the same size — selectivity +0.0058, **t = 1.98,
p = 0.049, n = 316**, driven mostly by Phon (+0.021). That is *borderline* and should be
described as suggestive, not established.

---

## Layout

```
overall/
  by_checkpoint.csv        <- THE MAIN TABLE. one row per (family, step),
                              brain + interp + localisation side by side
  summary_by_family.csv    <- one row per model: means, ranges, step-vs-alignment trend
  claim_tests.csv          <- per-family claim tests (claim, stat, value, p, n)
  heldout_predictor.csv    <- cross-family held-out predictive validation
  localisation_onset.csv
by-model/<family>/
  README.md                <- what this model is, its numbers, what is odd about it
  checkpoints.csv          <- this model's rows of the main table
  brain_alignment.csv      <- full per task × session × step detail  (CONFOUNDED)
  interp_mechanistic.csv   interp_layerwise.csv
  localisation_isolation.csv  localisation_onset.csv
  behaviour.csv            ablation_alignment.csv  ablation_behaviour.csv
  figures/<family>_overview.png
diagnostics/
  layerwise_alignment.csv  <- alignment at every layer (rules out the layer explanation)
  run_confound_check.csv   <- raw vs run-partialled alignment, per layer
figures/                   <- cross-model figures (fig1-fig8, tables)
superseded/early_tier1/    <- an earlier PARTIAL pass (7 families, 8/12 cells). Kept for
                              completeness, deliberately NOT a viewer config. Do not use.
provenance_tier_ledger.json <- the run record: per-tier status, exit code, duration, peak GPU
```

Every table carries `family` and `model_ref`, so you can filter by model in the viewer
without downloading anything.

---

## The models

10 families, 262 checkpoints total. `brain_rsa_mean` is shown **only** so you can see it
is flat and near zero; per the warning above it is not interpretable.

| family | ckpts | steps | brain RSA (confounded) | trend ρ (p) | interp PR | interp gini | loc selectivity | behaviour acc |
|---|---|---|---|---|---|---|---|---|
| pico-decoder-tiny | 21 | 0–126k | −0.017 | −0.02 (0.80) | 0.221 | 0.204 | 0.546 | 0.631 |
| pico-decoder-small | 126 | 0–125k | +0.001 | +0.16 (<0.001) | 0.184 | 0.243 | 0.551 | 0.682 |
| pico-decoder-medium | 21 | 0–125k | −0.011 | −0.10 (0.13) | 0.191 | 0.206 | 0.544 | 0.679 |
| pico-decoder-large | 21 | 0–125k | −0.012 | +0.01 (0.93) | 0.104 | 0.221 | 0.539 | 0.687 |
| beetle-humanscale-eng | 18 | 0–9.5k | −0.016 | −0.03 (0.71) | 0.300 | 0.077 | 0.530 | 0.510 |
| beetle-fineweb3-eng | 19 | 0–65k | −0.015 | −0.05 (0.44) | 0.241 | 0.090 | 0.558 | 0.590 |
| babylm-gpt2-3 | 9 | 191–1908 | −0.021 | +0.17 (0.07) | 0.065 | 0.326 | 0.519 | 0.693 |
| babylm-gpt2-5 | 9 | 191–1908 | −0.022 | +0.14 (0.14) | 0.066 | 0.324 | 0.531 | 0.687 |
| babylm-gpt2-7 | 9 | 191–1908 | −0.024 | +0.12 (0.21) | 0.065 | 0.328 | 0.520 | 0.707 |
| babylm-gpt2 | 9 | 598–2990 | −0.032 | −0.03 (0.73) | 0.072 | 0.293 | 0.530 | 0.707 |

pico-decoder-small's trend is p<0.001 only because n = 1,512 stimulus-level rows; ρ = 0.16
on a confounded measure is not a finding. No other family reaches p < 0.05, which across
ten tests is what chance looks like. The held-out cross-family predictor scores mean
**R² = −2.74** — worse than predicting the mean.

---

## How to read the metrics

**interp** (safe). `per` = participation ratio, the effective dimensionality of the
representation as a fraction of hidden size; **lower = more compressed**. Here 0.06–0.30,
and the babylm models (0.065) are far more compressed than Beetle (0.24–0.30). `gini` and
`hoyer` are sparsity of activation mass, **higher = sparser**. `condition_number` is the
spread of the activation covariance spectrum. `cka_to_prev` is representational similarity
to the previous checkpoint — **near 1 means training has stopped changing the geometry**.

**localisation** (safe). `selectivity_index` is how strongly a phenomenon's top units
prefer it over other phenomena; ~0.5 across all models here, i.e. **moderate and
strikingly constant** — no model isolates phenomena sharply. `mean_overlap_with_others` is
how much a phenomenon's circuit is shared with other phenomena; **high overlap means
little specialisation**. `n_active_layers` is how many layers contribute; `layer_com` is
the centre of mass over depth (**low = early layers**).

**behaviour** (safe). `mp_accuracy` is minimal-pair accuracy, chance = 0.5. Beetle
humanscale at 0.51 is essentially at chance; the babylm and pico models reach 0.63–0.71.

**brain** (confounded — see the warning). `rsa` is Spearman between the LM RDM and the
brain RDM over stimulus pairs; `rsa_pearson`/`rsa_kendall` are the same with different
rank treatments. `n_stim` is 72 for Sem/Phon and 60 for Gram/Plaus (controls excluded).

---

## Provenance

Produced by [`suchirsalhan/cdl-representations-brains-babylms`](https://github.com/suchirsalhan/cdl-representations-brains-babylms),
tiers 0–3, 2026-08-19. Brain-side session RDMs are cached separately at
[`BrainAlign/ds003604-session-rdms`](https://huggingface.co/datasets/BrainAlign/ds003604-session-rdms)
(12 of 12 task × session cells). The run confound described above applies to those RDMs
too, and the recommended fix is to normalise voxel patterns within run before aggregating
across runs.
