# Masking — what changed, why, and how to run it

Written 2026-08-26, for whoever runs this next (no GPU access on the writing
end — see "How this was verified" below). Answers three things: what images
the pipeline downloads and how it preprocessed them before today, what was
wrong with that, and exactly what to run now.

## 1. What was on the GPU box, and how it was preprocessed

`scripts/batch_download_bold.py` pulls straight from `OpenNeuroDatasets/ds003604`'s
raw BIDS tree (`configs/neuro_datasets.yaml`) — there is no `derivatives/`
imaging path anywhere in the pipeline, and ds003604's own `derivatives/` folder
only holds behavioural accuracy/RT tables, not preprocessed imaging. So every
BOLD file the GPU box ever had was **raw, as-acquired, native-space** — no
motion correction, no slice-timing correction, no coregistration, no spatial
normalisation to any template. `src/preprocessing/fmri_preprocessing.py` then:

1. loads that raw BOLD + its `events.tsv`,
2. applies spatial smoothing only (6mm FWHM Gaussian, no motion correction —
   none is estimated anywhere in this pipeline),
3. fits a GLM (`nilearn.glm.first_level.FirstLevelModel`, SPM canonical HRF,
   cosine drift/high-pass) with one regressor per stimulus, and pulls a
   per-stimulus beta map — this part was already correct; the "re-derive with
   GLM betas" item in `TODO.md` §0.3 is done and was done in every run so far,
4. masks the result with a `NiftiMasker`.

Step 4 is what was broken, and why: with no mask ever supplied
(`brainprep_subject.sh` never passed `--mask-path`), nilearn fell back to its
own default `mask_strategy='background'`, which assumes near-zero voxels
outside the brain. ds003604's raw BOLD does not have that property, so the
auto-mask silently kept ~100% of the raw acquisition volume — 917,504 voxels,
air included (`paper_results/control/README.md`).

Cross-subject aggregation (`src/rsa/session_based_rsa.py`) uses BrainIAK's
Shared Response Model, not anatomical alignment — which is exactly why no
normalisation existed anywhere upstream: SRM was chosen so subjects being in
different native spaces wouldn't matter. That also means a template-space
(MNI) mask cannot simply be resampled onto raw ds003604 BOLD — resampling only
reconciles voxel *grids* via each image's own affine, it doesn't register two
genuinely different spaces. Doing that would silently place a mask in the
wrong anatomical location for almost every subject.

## 2. What changed

**A. The whole-brain mask bug is fixed, unconditionally.** `_get_run_mask` in
`fmri_preprocessing.py` now builds a real per-run mask with
`mask_strategy='epi'` (nilearn's strategy for functional images without a
clean zero background) whenever no explicit mask is given. This applies to
*every* run regardless of anything else below.

**B. Real auditory-cortex and motor-cortex ROIs, via actual per-subject
registration.** ds003604 ships a T1w anatomical per subject-session (in
`anat/`, two acquisitions as of the 2026-08-26 checkout — `find_t1w` picks the
first deterministically and logs it). New code:

| file | what |
|---|---|
| `src/preprocessing/roi_atlas.py` | Named ROI sets (`auditory`, `motor`, `language`) as AAL region-**name** substrings, not numeric codes — see the module docstring for why: the existing `--aal-rois` numeric list silently falls back to the *wrong* regions on the current atlas, and name-based lookup can't have that failure mode (an unmatched name raises). |
| `src/preprocessing/spatial_normalization.py` | Per-subject-session registration: rigid EPI→T1, affine T1→MNI152 (via `dipy`, mutual information — deliberately not diffeomorphic; see the module docstring for the reasoning), used to warp a named ROI from MNI space into that subject's own native functional space. Every registration is sanity-checked (Dice overlap + translation bounds) and **falls back to the whole-brain mask, never crashes**, if a T1 is missing or a registration fails. |
| `fmri_preprocessing.py` | New `roi_set` / `mask_cache_dir` constructor args, wired through `batch_preprocessing.py` (`--roi-set`, `--mask-cache-dir`) and the shell scripts (`ROI_SET`, `MASK_CACHE_DIR` env vars). |

Registration is cached **per (subject, session)**, not per task, under
`MASK_CACHE_DIR` (default `$RDM_ROOT/_masks`) — computed once, reused across
Sem/Phon/Gram/Plaus.

## 3. How this was verified without a GPU

I have no GPU or downloaded BOLD data available. Everything above was tested
against realistic **synthetic** data — a real, brain-shaped, cropped copy of
nilearn's own bundled MNI152 template standing in for a subject's anatomy,
with known planted transforms — in `tests/test_masking_pipeline.py`. This
caught and fixed four real bugs before they could reach a real run: a 4D/3D
input mix-up that failed deep inside dipy with an unreadable error, an
early-return that skipped the registration cache whenever no EPI reference was
passed (breaking exactly the "second ROI set reuses the registration" case
this whole caching design exists for), and two variations of the same
temp-file-naming bug (`Path.with_suffix()` mishandling `.nii.gz`) that made
every write fail. All 23 tests pass now (~60s, no GPU, no BOLD data, no
downloads beyond nilearn's own bundled MNI template and a one-time AAL atlas
fetch it needs anyway).

**Run this first, before spending any GPU time:**

```bash
pip install -r requirements.txt   # adds dipy
pytest tests/test_masking_pipeline.py -v
```

If this doesn't pass in your environment, something about the environment
differs from what was tested against (nilearn/dipy version drift is the most
likely culprit) — fix that before running anything real, not after.

## 4. What to actually run

Whole-brain fix only (no ROI restriction) — this is the default, nothing to set:

```bash
bash prepare_brain_rdms.sh
```

With auditory + motor cortex ROI masks added:

```bash
ROI_SET=auditory,motor bash prepare_brain_rdms.sh
```

Combine with the existing language-network ROI too if useful:

```bash
ROI_SET=language,auditory,motor bash prepare_brain_rdms.sh
```

`MASK_CACHE_DIR` defaults to `$RDM_ROOT/_masks` and does not need to be set
explicitly — just don't override it differently per task if you do set it.

## 5. How to check it worked, without re-running anything

- **`$RDM_ROOT/_masks/roi_mask_status.csv`** — one row per subject-session:
  `status` is `ok`, `no_anat` (no T1 found), `registration_failed`,
  `warp_failed`, or `empty_after_warp`. Anything other than `ok` means that
  subject-session fell back to the whole-brain mask alone — expected for a
  few subjects, worth a glance if it's most of them.
- **`$RDM_ROOT/_masks/<subject>/<subject>_<session>_roi-<set>_qc.png`** — one
  image per successfully-registered subject-session: the warped ROI overlaid
  on that subject's own T1. This is the thing to actually look at to judge
  registration quality, not just the automated Dice/translation checks.
- **Re-run the existing diagnostics** on the new patterns (no pipeline
  changes needed, they already exist):
  ```bash
  python scripts/rdm_dimensionality.py --rdm-root data/processed/fmri/ds003604
  ```
  The number to watch: `frac_voxels_nonzero`. It was ~1.0 (no brain mask)
  before this fix; it should now be well below that. If it's still ~1.0,
  `mask_strategy='epi'` did not fix the problem on the real data the way it
  did on the synthetic test data, and that's the first thing to report back.
  Then:
  ```bash
  python scripts/positive_control.py --rdm-root data/processed/fmri_wrn/ds003604
  ```
  This is the actual gate (`paper_results/control/README.md`) — it needs to
  come back with at least the acoustic-spectrum control significant before
  any alignment number is reported again.

## 6. What this does NOT do

- No diffeomorphic (nonlinear) registration — affine only. Adequate for
  centimeter-scale ROIs like auditory/motor cortex; not adequate if a future
  need calls for finer anatomical precision. See
  `spatial_normalization.py`'s module docstring for the reasoning.
- No global-signal removal. Masking removes the air/skull voxels that were
  dominating the RDMs; it does not remove global-signal structure that may
  remain within the (now smaller, real) brain mask. That's `TODO.md` §0 step
  2, still open.
- Does not touch `extract_stimulus_activity_simple` (the `--no-glm` path) —
  it's unused in practice (`use_glm=True` is the default everywhere in this
  pipeline) and raises clearly if `roi_set` is combined with `--no-glm` rather
  than silently ignoring the ROI restriction.
