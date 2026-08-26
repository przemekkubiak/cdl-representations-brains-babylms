# The corrected grid — 15 model families against confound-free ds003604 RDMs

Written by `scripts/corrected_sweep_summary.py`. Supersedes every alignment
number in `hf_results_staging/by-model/` and `hf_results_staging/overall/`,
which were computed against RDMs carrying the scanner-run confound.

## What changed since the confounded tables

The old RDMs correlated with run identity at **ρ = 0.56**; after per-run voxel
z-scoring that drops to **−0.04**, and the inter-subject noise ceiling *rises*
to **0.85** (it was 0.77 raw). So the correction removed the confound without
costing reliability — the corrected RDMs carry more real stimulus structure than
the confounded ones did, not less.

That matters for how the null reads. A null against unreliable RDMs is
uninterpretable. A null against RDMs where two halves of the subject pool agree
at ρ = 0.85 means the measurement had room to find something and did not.

## The result

| | |
|---|---|
| families × checkpoints × cells | 15 × (9–21) × 12 = **2964 rows** |
| noise ceiling (mean of 12 cells) | **0.855** |
| best alignment anywhere in the grid | **0.056** — **6.7 %** of the ceiling |
| across-seed sd of the PARC null | **0.0079** |
| families whose best cell beats the matched noise maximum | **0 / 15** |
| families statistically equivalent to zero (TOST, SESOI ±0.05) | **15 / 15** |
| Pythia scale trend (96 M → 1.5 B) | ρ = **+0.012**, p = 0.93, n = 60 cells |

## How the best cell is judged — read this before quoting a number

A family's best cell is a **maximum over 100–250 comparisons**. Comparing that
maximum against the across-seed sd at a *fixed* cell (0.0079) makes every family
look like a 5–7 σ detection, and that is an artefact: the seed sd holds task,
session and step constant while a maximum ranges over all of them and collects
each cell's own idiosyncratic bias.

The matched yardstick is the same maximum statistic computed on data known to
contain no effect. The PARC suite supplies it — 18 runs (3 architectures × 6
seeds) that differ **only by initialisation**, each spanning 204 cells. Their
per-run maxima run **0.033–0.070**, i.e. the same size as the families' maxima.
`null_max_p50 ≈ 0.048`, `null_max_p95 ≈ 0.065`, and no family's best cell reaches
the 95th percentile.

Where the comparison *is* like-for-like — a family's per-cell mean over ~20
checkpoints against the across-seed sd at that same cell — the largest deviation
across all 15 families is **1.0 σ**.

## Does it develop over training?

No. Spearman ρ(step, alignment) computed per cell, then combined across the 12
cells per family (pooling all rows would count one trajectory twelve times):

- Pure-noise PARC runs give trends of **−0.163 … +0.236** on the same test.
- **6/15** families fall outside that range — and **5 of them are DECREASING**
  (pythia-410m −0.53, pythia-160m −0.47, pico-small −0.38, pico-medium −0.34,
  pico-large −0.31). The one positive, babylm-gpt2-5 at +0.29, barely clears the
  noise maximum and has 9 checkpoints over 1908 steps.
- Untrained models align **better** than trained ones: step 0 gives +0.0056,
  trained checkpoints +0.0010 (p = 0.005). The declines survive dropping step 0,
  so this is not one anomalous point.

Whatever small correlation exists is present at random initialisation and
training removes it — consistent with surface form (token identity, word length)
rather than anything linguistic. That makes the pending positive control doubly
informative: if surface-form structure is what the untrained models pick up, an
acoustic / word-length control should come back **positive**.

## Is any of the per-cell structure a model effect?

No. Some cells return a consistently positive alignment for every checkpoint of
every family — Gram/ses-5 sits at **+0.028** across all 247. That looks like an
effect until you measure the same cell on the PARC runs, which have no
relationship to these models: they give **+0.027**.

Across all 12 cells, per-cell means from our 15 families correlate with per-cell
means from 18 pure-noise runs at **r = +0.987** (p = 3.4e-09). Variance
decomposition agrees: **cell identity 47.6%**, **model family 2.9%**,
checkpoint-level residual 49.5%.

So the only reliable structure in the alignment numbers belongs to the stimulus
set and the RDM, not to any model. Quote a per-cell value only alongside its
noise-run counterpart.

## Scale

The Pythia ladder exists to answer "your models are just undertrained." Real
parameter counts (from safetensors metadata, not nominal names — pythia-70m is
95.6 M): 96 M → 213 M → 506 M → 1.08 B → 1.52 B. Alignment does not move
(ρ = +0.012 against parameters, p = 0.93). Sixteen-fold scale buys nothing.

Caveat: the ladder ran at `MAX_CKPT=20`, so 18 of Pythia's 154 checkpoints, and
the other families at `MAX_CKPT=25`. This is a subsampled trajectory, not the
dense one. It spans step 0 → 143 000 log-uniformly, so it is adequate for a
trend test and would not hide a monotone effect.

## Files

| file | what |
|---|---|
| `alignment_by_checkpoint.csv` | every row + ceiling, `frac_of_ceiling`, params |
| `alignment_by_family.csv` | per family: mean/sd, best cell, TOST, null-max comparison |
| `alignment_by_cell.csv` | family × task × session |
| `scale_ladder.csv` | the Pythia ladder with its trend test |
| `seed_null_comparison.csv` | every family against the matched pure-noise null |
| `training_trend.csv` | per family: does alignment grow with training? |
| `training_trend_null.csv` | the same test on the 18 pure-noise runs |
| `cell_vs_noise.csv` | per-cell means, our families vs noise runs |
| `model_params.csv` | exact parameter counts, cached from the Hub |
| `summary.json` | the headline numbers above |
| `fig_corrected_scale_ladder.*` | alignment vs parameters, against the ceiling |
| `fig_corrected_family.*` | all 15 families vs the seed null |
| `fig_corrected_trajectory.*` | alignment vs training step |
| `fig_corrected_null_checks.*` | training trend vs noise; per-cell vs noise |

## What this does not yet establish

The measurement is shown to be *reliable* (ceiling 0.85) but not yet shown to be
*sensitive to alignment that exists*. The positive control in `TODO.md` §0 — a
low-level acoustic / word-length RDM that ought to correlate with auditory
cortex — has not been run. Until it is, the honest claim is "no LM alignment
detectable in RDMs of demonstrated inter-subject reliability", not "LMs do not
align with the developing brain."

`ds006239/LocalSem` remains the single confound-free language cell available and
is the right target for that control.
