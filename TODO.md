# TODO — notes from OpenReview

Reviewer-driven work plan. Each item says what to do, where it lands in the
codebase, and what currently blocks it. Status: `[ ]` todo, `[~]` partial,
`[x]` done.

**READ THIS FIRST — the ds003604 null is NOT publishable, and as of 2026-08-26
the reason is known.** The positive control failed: nothing stimulus-driven
correlates with these brain RDMs (0/108 tests) — not the acoustic spectrum of
the audio subjects actually heard, not the study's own condition contrast. The
cause is measured, not guessed: the voxel patterns are **unmasked whole-volume**
(917,504 voxels, 100% non-zero), their leading component tracks the volume's
**global signal** (|ρ| = 0.85), and the 72-item RDMs live in **4 dimensions**.
The RDMs largely encode scanner brightness, which is why the 0.85 noise ceiling
is so high — brightness is very consistent across subjects — and why no model
can align with them.

So the null is a measurement artefact, not a finding about language models.
Everything downstream of these RDMs is on hold: the 15-family grid, the Pythia
ladder, the PARC seed-null, the training trends. See
`paper_results/control/README.md`. Fix order: brain mask → global-signal
handling → GLM betas → re-run the control → only then re-run the grid.

The scanner-run confound described below WAS real and IS fixed (run correlation
0.56 → −0.04); it was simply not the only thing wrong.

---

## 0. Blocking — do these before any new sweep

These decide whether the claim is publishable at all. Adding models and
datasets on top of the current pipeline just multiplies an untrustworthy null.

- [x] **Compute noise ceilings and normalise all alignment against them.** DONE 2026-08-25.
  `src/rsa/noise_ceiling.py` exists and is *never called* by the tier driver.
  `scripts/checkpoint_alignment_trajectory.py:287` already tries to load
  `noise_ceiling_by_session.csv`; that file does not exist anywhere.
  - Wire the estimator into `run_devai_bare.sh` tier 0, after session RDMs.
  - Report both raw and ceiling-normalised RSA in every output table.
  - Reviewer note: ceilings from inter-subject reliability are typically
    **0.2–0.4** in this literature. If ours come back near zero, the paper is
    about the confound in developmental fMRI designs, not about LMs.
  - Cross-check against Schrimpf et al. 2024 cross-subject prediction accuracy
    as the comparison point for what a healthy ceiling looks like.

- [x] **Re-run the full grid with within-run normalisation.** DONE 2026-08-26.
  All 12 task × session cells rebuilt with per-run voxel z-scoring; **15
  families × 2964 rows** run against them (`devai_grid_wrn/`), summarised by
  `scripts/corrected_sweep_summary.py` into `paper_results/corrected/` and
  published to `corrected-sweep/grid/` on the Hub.
  - The correction works: run-identity correlation **0.56 → −0.04**, and the
    noise ceiling *rises* 0.77 → **0.85**, so it removed the confound without
    costing reliability.
  - The null survives it. Best alignment anywhere in the grid **0.056 = 6.7% of
    the ceiling**; **0/15** families beat a matched pure-noise maximum;
    **15/15** equivalent to zero (TOST, ±0.05).
  - Judge a best cell against the **matched max statistic** (PARC per-run
    maxima, 0.033–0.070), never against the fixed-cell across-seed sd — the
    latter turns every family into a fake 5–7σ detection. See
    `paper_results/corrected/README.md`.
  - Confounded tables (`hf_results_staging/by-model/`, `overall/`) are
    superseded by `corrected-sweep/grid/`.

- [x] **Positive controls — RUN 2026-08-26, AND THEY FAILED.** This is now the
  project's headline finding, not a checkbox. `scripts/positive_control.py` and
  `scripts/rdm_dimensionality.py`; write-up in `paper_results/control/README.md`.
  - The TEST has power: it recovers run identity at ρ = +0.67 (best cell +0.87)
    on the uncorrected RDMs, and the correction removes it (−0.12). So a null
    from this test is a real null.
  - **0/108 stimulus × cell tests significant** on the corrected RDMs, after
    Holm correction, with 5000 permutations each.
  - Cause: unmasked whole-volume patterns dominated by global signal; RDM
    effective rank 4 (72 stimuli) / 7 (60 stimuli); RDM vs a pure amplitude RDM
    ρ = +0.43.
  - **Therefore the ds003604 LM null is vacuous.** Do not publish it.
  - Fix order: brain mask → remove/model global signal → GLM betas → re-run this
    control → only then re-run the grid. Needs a tier-0 re-download, ~2 h.
  - Original plan, for the re-run:
  - Low-level acoustic / word-length RDM vs auditory cortex. If a spectrogram
    RDM does not correlate, the RDMs carry no usable stimulus signal and the LM
    null is vacuous.
  - [x] Brain-to-brain split-half (doubles as the ceiling, item 1) — 0.84–0.89
    across all 12 cells.
  - Replicate a published alignment effect on a dataset where one is
    established, and confirm we recover it.

---

## 1. More neuroimaging datasets

Reviewer: "use a few neuroimaging datasets." Note the deeper problem is not the
*count* but the *design*: ds003604 presents each stimulus in exactly one run, so
run and stimulus are perfectly nested — that is what created the confound. A
second dataset with the same structure reproduces the same failure. Prioritise
designs where run and stimulus are **crossed**, or naturalistic continuous
listening.

- [x] **Make the data path dataset-aware.** DONE 2026-08-25.
  Was blocking everything below: `scripts/batch_download_bold.py` built every
  URL from `OpenNeuroDatasets/ds003604` with no way to change it.
  - `configs/neuro_datasets.yaml` — dataset registry (accession, snapshot,
    tasks, sessions, age metadata, run/stimulus structure, contrast spec).
  - `src/datasets/registry.py` — `DatasetSpec`, URL construction, per-task
    accessors. Refuses to fall back to a default accession: an unresolved
    dataset raises rather than silently downloading ds003604 under another name.
  - `scripts/batch_download_bold.py --dataset <key>` — verified to reproduce the
    exact previous ds003604 URLs (backward compatible; 1052 Phon files found).
  - `prepare_brain_rdms.sh` already took `DATASET=`; it now lines up.

- [x] **Generalise the contrast spec per dataset.** DONE 2026-08-25.
  `src/contrast_spec.py` now holds `CONTRAST_SPECS` keyed by accession, with
  `get_contrast_spec(dataset)`. The old `CONTRAST_SPEC` / `PHENOMENA` names still
  export ds003604 unchanged, so existing importers are untouched. Added
  `text_from_stim_filename` / `reconstruct_pair_text` for datasets whose stimuli
  are filenames, and `normalise_trial_type` for the float-coding quirk.

- [x] **`scripts/inspect_dataset.py`** — new. Reads structure, per-subject age
  availability, trial-type codes (decoded via the events.json `Levels` sidecar),
  and **run/stimulus crossing** off a metadata-only checkout, before any BOLD
  download. Validated against ds003604, where it independently reproduces the
  known answer (100% of stimuli in exactly one run, all four tasks).

### KEY FINDING FROM THE INSPECTION — read before planning the sweep

**The run confound is not a ds003604 quirk. It is a property of this class of
blocked word-pair design.** Measured on the actual checkouts:

| dataset | task(s) | stimuli in exactly one run | verdict |
|---|---|---|---|
| ds003604 | Sem, Phon, Gram, Plaus | 100% | nested |
| ds001894 | all six | 98–99% | nested |
| ds006239 | ReadPhon, ReadMean | 96–97% | nested |
| ds006239 | LocalEng | 95% | partial |
| **ds006239** | **LocalSem, LocalASL, LocalSR0** | **0–47%** | **crossed** |

So "use more datasets" does **not** fix the confound — two of the three new
sources inherit it. Within-run normalisation is mandatory across the board.

**`ds006239/LocalSem` is the only confound-free language cell we have.** Its
stimuli recur across runs, so run and stimulus identity are separable and the
ds003604 failure cannot arise. It is therefore the single best target for a
clean alignment estimate and for the §0 positive control — prioritise it.

### Datasets — status after inspection

- [x] **Lytle et al. 2019 — `ds001894`** — checkout bootstrapped, spec verified,
  `status: ready`. 188 subjects, ses-T1/ses-T2, 6 tasks.
  - **Best age metadata of the three**: explicit age at scan, per subject, per
    run, in `participants.tsv`. Longitudinal — the same children at ~10 and ~12,
    which is the closest real analogue we have to an LM checkpoint trajectory.
  - Trial types verified from the sidecar: a **2×2 crossing of orthographic ×
    phonological similarity** (O+P+/O+P-/O-P+/O-P-). This gives a `Phon` and an
    `Orth` contrast that are decorrelated by design — and `Orth` is a phenomenon
    ds003604 cannot provide at all.
  - AA / AV / VV task variants are a built-in modality control, directly useful
    for §3 (auditory vs higher-level regions).
  - Caveat: ses-T2 has only the VV tasks. Quirk: `task-AVWord` writes some
    trial_types as floats; handled, but do not parse raw.

- [x] **Wang et al. 2025 — `ds006239`** — checkout bootstrapped, spec verified,
  `status: ready`. 89 subjects, single session, 6 tasks.
  - Trial types verified: ReadPhon is the same 2×2 O×P design; ReadMean mirrors
    ds003604's Sem (high / low / unrelated association); LocalSem is a semantic
    picture-matching localizer.
  - **Per-subject age is NOT recoverable.** `participants.tsv` has birthdate but
    no scan date, and the release contains **zero** `*_scans.tsv`. The 10–17
    range is documented only in the paper. This dataset is **cohort-level only**
    and cannot carry the developmental axis as published — request per-subject
    age from the authors if the axis is needed. It remains fully usable for
    alignment magnitude, which given LocalSem is the more valuable role anyway.
  - Quirk: run entity is `run-1`/`run-2`, not zero-padded.

- [ ] **Lytle et al. 2020** — still **unresolved**, and deliberately so.
  The linked database is a Data in Brief article, not an OpenNeuro accession.
  Registered in `configs/neuro_datasets.yaml` with `accession: null` and a
  blocker; the downloader refuses it by name with that explanation rather than
  guessing. **Action: resolve the real accession from the data article**, then
  run `scripts/inspect_dataset.py --dataset lytle-2020 --bootstrap`.
  - Would add an **orthographic** task — though note ds001894 and ds006239 both
    turned out to supply `Orth` already, so this is now less critical than it
    looked when the review notes were written.

- [ ] Consider one naturalistic continuous-listening dataset (e.g. Narratives)
  purely as a confound-free design where run/stimulus nesting cannot arise.

---

## 2. Alignment metric and methodology

- [ ] **Justify the alignment metric.** We currently report Spearman RSA
  (plus Pearson/Kendall) with no argument for why. Need an explicit defence of
  the choice, and ideally a demonstration that the conclusion is invariant to it.
- [ ] **Add voxelwise encoding models alongside RSA.** RSA over 60–72 stimuli is
  a weak instrument; cross-validated ridge encoding R² is the field standard and
  is more sensitive. Tier 2's description in `run_devai_bare.sh:19` mentions
  "encoding" but no encoding outputs exist. A null that holds under *both*
  methods is far harder to dismiss.
- [ ] **Multiple benchmarks, and control for model dimensionality.** Reviewer
  explicitly flags that model differences (e.g. hidden dimensionality) can drive
  benchmark performance. Our families span 11M–1B with very different
  participation ratios (0.065 for babylm-gpt2 vs 0.30 for Beetle —
  `hf_results_staging/overall/summary_by_family.csv`), so this is a live
  confound in our own results. Needs dimensionality-matched comparison or
  explicit covariate control.
- [ ] **Equivalence testing.** `p > 0.05` is not evidence of absence. Report
  TOST or Bayes factors against a pre-specified smallest effect of interest —
  ideally the effect size from whatever prior work we are contradicting.

---

## 3. Anatomical specificity

- [ ] **Report alignment by region, not pooled.** Reviewer: check whether gains
  are larger in regions underlying **low-level** processing (auditory cortex)
  than in **higher-level semantic** regions (STG, IFG, AG). This is the standard
  test for whether an alignment effect is linguistic or acoustic.
  - `run_analysis.py --aal-rois` and `scripts/run_roi_pipeline.py` already
    support ROI restriction; the grid does not use them.
  - Add an ROI axis to the alignment output so every result is reported
    per-region.
  - If any residual alignment turns out to be auditory-cortex-only, that is
    itself the finding.

---

## 4. Individual-subject robustness

- [ ] **Demonstrate reproducibility at the individual-subject level.** Current
  RDMs are cohort-aggregated (hyperalignment across subjects). Reviewer wants
  per-subject results.
  - Per-subject RDMs and per-subject alignment, with the distribution reported,
    not just the group mean.
  - Split-half / leave-one-subject-out stability.
  - We know cohort size is not the issue (Sem/ses-7 at n=40 vs n=98 agree at
    ρ = 0.928), but that is aggregate agreement, not individual reproducibility.

---

## 5. Framing

- [ ] **Be explicit about the neuroscientific framing.** Reviewer asks directly:
  are the proposed links between the NLP system and the brain analogous to
  processes already used in NLP (RNN dynamics, signal propagation), or are they
  something with **no current equivalent in NLP architectures**? Right now the
  paper does not commit. Write this section explicitly.
- [ ] **Include standard quantitative evaluation.** Claims of SOTA performance
  or broader significance need conventional metrics — **perplexity** at minimum,
  per family per checkpoint. We currently report only minimal-pair accuracy
  (`behav_mp_accuracy`, 0.51–0.71). Beetle-humanscale at 0.51 is at chance,
  which needs explaining and perplexity would explain it.

---

## 6. Model scale

Not raised in the review, but reviewers will read a null over 11M–1B models as
"these models are undertrained," which is a much weaker claim than "LMs lack
brain alignment."

- [x] **PARC suite added** (18 models). DONE 2026-08-25.
  https://huggingface.co/collections/jmichaelov/parc-models — Pythia 160M,
  Mamba 130M, RWKV 169M on OpenWebText, 4000 steps, **6 seeds each**, 73 shared
  checkpoints (verified identical across all 18 repos). All three load and
  extract correctly; Mamba (`backbone.layers`, 24 blocks) and RWKV
  (`rwkv.blocks`, 12) are now explicit in `discover_block_layer_names`.
  - **Seed axis = the null distribution.** Six seeds differing only by
    initialisation tell us what "no effect" looks like on this measurement. An
    alignment counts only if it clears that spread — far stronger than a p-value
    against zero, and it supplies the equivalence test asked for in §2.
    `scripts/parc_seed_null.py` does seed spread, TOST, and the architecture
    ANOVA.
  - **Architecture axis answers §5 directly.** Transformer vs state-space vs
    RNN-like at matched data, scale and steps: whether brain-LM links resemble
    recurrence or signal propagation becomes measurable rather than arguable.
  - Launched via `launch_parc_sweep.sh` (GPUs 4–7), armed by
    `scripts/parc_watcher.sh` to start when stage 1's 12 corrected cells land.

- [x] **Run the Pythia ladder.** DONE 2026-08-26. All five rungs ran against the
  corrected RDMs (stage 3 of `launch_full_sweep.sh`). Real parameter counts, not
  nominal: 96M → 213M → 506M → 1.08B → 1.52B. **No trend** — Spearman ρ = +0.012
  vs parameters (p = 0.93, n = 60 cells). 16× scale buys nothing, which is the
  answer to "your models are just undertrained". `paper_results/corrected/scale_ladder.csv`.
  - Caveat to state in the paper: `MAX_CKPT=20`, so 18 of Pythia's 154
    checkpoints — log-uniform over step 0–143 000, fine for a trend test but not
    the dense trajectory.
- [ ] Include at least one ~7B model outside the developmental framing, since
  published alignment effects typically grow with scale.

---

## 7. Then: the full sweep

Only after §0 is green.

- [ ] Full RSA sweep: {ds003604, ds001894, ds006239, Lytle-2020} ×
  {pico ×4, beetle ×2, babylm ×4, pythia ×5, one large} × all checkpoints ×
  per-ROI × {RSA, encoding}, all ceiling-normalised.
- [ ] Budget and stage this properly — see `PICKUP.md` for the disk floor
  (350 GB, shared overlay with another project) and the GPU pin (cards 0–2
  only). The streaming design in `prepare_brain_rdms.sh` is load-bearing; four
  datasets makes it more so, not less.
