# Multi-dataset infrastructure — what changed, how to run it, what's verified

Written 2026-08-26, alongside [MASKING.md](MASKING.md) (read that first for the
masking/ROI-registration story — this document assumes it). Answers: how to
run any of the four registered datasets, how the three ROI groupings work,
how the age-group taxonomy replaces "session" for cross-sectional datasets,
and what's verified vs. still ds003604-only.

## 1. The four datasets

| key | paper | status | sessions | age |
|---|---|---|---|---|
| `ds003604` | Wang et al. 2022 | ready (flagship) | ses-5/7/9 | per-subject, discrete |
| `ds001894` | Lytle et al. 2019 | ready | ses-T1/T2 | per-subject, continuous (7.4–16.4) |
| `ds006239` | Wang et al. 2025 | ready | ses-1 | per-subject, continuous (10.1–16.9) — corrected 2026-08-26, was wrongly recorded as cohort-only |
| `ds002236` | Lytle et al. 2020 | ready | none (session-less) | per-subject, continuous (8.7–15.5) |

Full detail — trial-type codes, run/stimulus structure, quirks — is in
`configs/neuro_datasets.yaml`, each entry's own comments.

## 2. ROI groupings

Three named sets in `src/preprocessing/roi_atlas.py`, each a real per-subject
registered mask (MASKING.md), not a whole-brain default:

| name | AAL regions | use |
|---|---|---|
| `language` | the original 12-region set | language-network alignment |
| `phonology` | Heschl's gyrus + superior temporal gyrus + precentral gyrus (auditory ∪ motor) | phonological/articulatory processing |
| `all` | union of every set above | everything at once |

Run the same task three times, once per grouping, to get all three — each
writes to its own subdirectory so they never overwrite each other (see §4).

```bash
ROI_SET=language  DATASET=ds002236 bash prepare_brain_rdms.sh
ROI_SET=phonology DATASET=ds002236 bash prepare_brain_rdms.sh
ROI_SET=all       DATASET=ds002236 bash prepare_brain_rdms.sh
```

Omit `ROI_SET` for the whole-brain default (still gets the `mask_strategy=
'epi'` fix from MASKING.md regardless).

## 3. Age groups replace "session" for three of the four datasets

ds003604's three BIDS sessions already correspond to three developmental
timepoints (`configs/age_groups.yaml` `verified_against`). The other three
are cross-sectional: a single BIDS session (or none, for ds002236) can span
multiple age-group bins — ds001894's `ses-T1` alone runs 7.36–14.54 years.
Building one RDM per BIDS session for those would average children years
apart in development, so `scripts/regroup_patterns_by_age.py` relabels each
subject's pattern file by their *real* age-group bin
(5/7/9/11/11+, `configs/age_groups.yaml`) before any RDM is built.
`prepare_brain_rdms.sh` runs this automatically for every dataset except
ds003604 (untouched by design) — nothing to invoke manually.

**What this means for output**: for ds001894/ds006239/ds002236, the
`session_rdm_ses-*.npz` files you get are keyed by age-group bin
(`ses-9`, `ses-11`, `ses-11+`, ...), not by the dataset's own BIDS session
label. `ses-5`/`ses-7` will be empty for these three — nobody in that age
range exists in their cohorts; that's real, not a bug (see the coverage
table below).

| dataset | 5 | 7 | 9 | 11 | 11+ |
|---|---|---|---|---|---|
| ds003604 | ✅ | ✅ | ✅ | — | — |
| ds001894 | — | (3 obs) | ✅ | ✅ | ✅ |
| ds006239 | — | — | — | ✅ | ✅ |
| ds002236 | — | — | ✅ | ✅ | ✅ |

## 4. Running a new dataset end to end

```bash
DATASET=ds002236 PHENOMENA="Phon Sem" ROI_SET=phonology WITHIN_RUN_NORM=1 \
    bash prepare_brain_rdms.sh
```

Notes specific to non-ds003604 datasets:

- **`PHENOMENA` must be set explicitly.** The default (`Sem Phon Gram Plaus`)
  is ds003604's four; the other three only have `Phon`/`Sem` registered
  (`configs/neuro_datasets.yaml` → `phenomena`).
- **`WITHIN_RUN_NORM=1` is required for all three** — every one of them is
  `nested` (each stimulus in exactly one run), same confound class as
  ds003604. Without it the RDM measures scanner run, not language.
- **Output layout**: `data/processed/fmri/<DATASET>/[roi-<SET>/]<PHENOMENON>/session_rdm_ses-<BIN>.npz`.
  The `roi-<SET>/` segment only appears when `ROI_SET` is set, so the
  whole-brain default's path is unchanged from before ROI support existed.
- **Disk floor caveat**: for these three datasets, patterns for *every*
  on-disk session of a task stay on disk simultaneously until every age-group
  RDM for that task is built (they have to, to be relabeled correctly) —
  this is the one place these datasets depart from ds003604's strict
  per-session reclaim discipline (`PICKUP.md`). All three are much smaller
  than ds003604 (91–322 subjects vs. ds003604's ~3,851 runs), so this should
  stay well inside the disk floor, but it hasn't been measured on the real
  box — watch `free_gb` in the log on a first run.
- **Resumption is coarser** than ds003604's: a re-run after an interruption
  redoes that task's preprocessing (bounded, cheap relative to a fresh
  sweep), but never redoes an already-built (or already Hub-cached) age-group
  RDM — see the comment in `prepare_brain_rdms.sh` above the age-group block.

## 5. How this was verified

No GPU, no BOLD download, on the writing end — same constraint as MASKING.md.
Verified instead against:

- **Real metadata checkouts** of all four datasets (`scripts/inspect_dataset.py
  --bootstrap`) — real `events.tsv`/`participants.tsv`, no synthetic
  substitutes for anything dataset-structural.
- **Real events + synthetic-but-brain-shaped BOLD** through the actual GLM
  fit (`FirstLevelModel.fit` → `compute_contrast` → pattern extraction) for
  both a new dataset (ds002236: 48/48 trials → correctly keyed word-pair
  patterns) and ds003604 (48/48 patterns, identical keys to the pre-existing
  behaviour — zero regression).
- **`session_based_rsa.py` run unmodified** against synthetic multi-subject
  patterns using the new pair-format stimulus keys (`"word|word2"`) — RDM
  built correctly, all 10/10 stimuli matched across subjects. This is why
  that file needed no changes: its core aggregation only ever treats
  stimulus keys as opaque strings, and the one place it does read
  `--task` (`_get_non_control_stimuli`) already no-ops gracefully when
  the ds003604-only characteristics file it looks for doesn't exist.
- **The shell session-detection logic**, re-implemented portably (this
  machine's `/bin/bash` is 3.2 and lacks `mapfile`, which the real script
  requires — a pre-existing constraint, not something introduced here) and
  run against real directory listings for all four datasets: correctly
  finds `ses-5/7/9` (ds003604), `ses-T1` (ds001894), `ses-1` (ds006239), and
  falls back to `ses-all` for ds002236 (no session entity at all).
- `tests/test_multi_dataset_pipeline.py` — 26 tests, all passing, covering
  ROI-set composition, the age taxonomy, `classify_trials` against real
  events for every dataset, `FMRIPreprocessor` task/session resolution, and
  the regrouping script.

**Never run end to end on the real GPU box** — the shell orchestration
(`prepare_brain_rdms.sh`'s age-group block) has not been exercised against
real multi-GB BOLD downloads or measured for actual peak disk. Treat the
first real run of each new dataset as a validation run: watch the logs,
check `roi_mask_status.csv` (MASKING.md §5), and confirm `frac_voxels_nonzero`
the same way MASKING.md describes before trusting any resulting RDM.

## 6. Brain-side localization (2026-08-26, second pass)

`src/rsa/brain_localization.py` is now generalized for all four datasets —
but getting there surfaced a bug that predates any of this dataset work and
affected ds003604 too:

**The bug**: `prepare_brain_rdms.sh` reclaims (deletes) a task's pattern
files immediately after that task's session RDM is built — it always has,
for disk-floor reasons (`PICKUP.md`). The brain-localization call sat once,
at the very end of the whole script, by which point every task's patterns
had already been deleted. There is no trace of a real
`brain_specialization.csv` or `brain_localization_by_session.csv` anywhere
in `paper_results/` or `hf_results_staging/` — this had likely never
produced real output, for ds003604 or anyone, independent of anything
dataset-specific.

**The fix**: `scripts/run_brain_localization.py` now has `--append` (compute
+ merge one session's rows into a running table) and `--finalize-only`
(collapse the accumulated table into onsets + a figure, no pattern files
touched). `prepare_brain_rdms.sh` calls `--append` right before each reclaim
point (both the ds003604 per-session branch and the age-group branch) and
`--finalize-only` once at the very end.

**The generalization**: `build_stim_lookup_for_dataset` dispatches on
`stimuli.kind`, same pattern as `src/datasets/stim_identity.py` — ds003604's
characteristics-TSV path is untouched; the other three build their
(phenomenon, condition) lookup from events.tsv directly, via
`classify_trials`, so a stimulus's condition here can never disagree with
what it was treated as when its pattern was extracted.

**Two more real bugs found while testing this against real data** (not by
inspection — a combined multi-dataset figure that didn't match the expected
coverage table is what caught both):
- The lookup was a flat `stim_id -> single (phenomenon, condition)` dict.
  ds001894's Phon and Orth contrasts are drawn from the *same* word pairs
  (positive/negative differently for each), so building Orth's entries
  after Phon's silently overwrote every one of Phon's — ds001894 lost its
  entire Phon axis. Fixed: lookup is now `stim_id -> LIST of (phenomenon,
  condition)` pairs.
- `_list_subject_sessions`'s regex character class didn't include `+`, so
  every `ses-11+` pattern file (the oldest age-group bin) was silently
  dropped from consideration for every dataset that has one.

Verified end to end against real events.tsv-derived stimulus lookups and
synthetic patterns for all four datasets together, including the
multi-phenomenon and 11+-bin cases above.

## 7. Visualizing brain specialization by age group and domain

`scripts/plot_activation_by_age_domain.py` combines every dataset's
`brain_localization_by_session.csv` into one figure: one panel per
phenomenon, x-axis = age group, one series per dataset (only where that
dataset actually has data at that bin — matching the coverage table in §3).

```bash
python scripts/plot_activation_by_age_domain.py \
    --dataset ds003604 data/processed/fmri/ds003604/localization \
    --dataset ds001894 data/processed/fmri/ds001894/localization \
    --dataset ds006239 data/processed/fmri/ds006239/localization \
    --dataset ds002236 data/processed/fmri/ds002236/localization \
    --output-dir paper_results/activation_by_age
```

**What this shows, and does not show.** These are *not* anatomical brain
maps — no picture of a brain with colored activation on it, just the
selectivity index (or Gini/overlap/entropy, via `--metric`) per phenomenon
per age group per dataset: how concentrated and how phenomenon-specific the
condition>control response is, with no information about *where*. For an
actual picture of the brain, see §7b.

## 7b. Real anatomical brain maps (2026-08-26, third pass)

The scalar figures in §7 answer "how specialized"; this answers "where".
It reconstructs each subject's condition>control t-map back into 3D space
and warps it into MNI152 space using the SAME per-subject registration
infrastructure §6/MASKING.md built for ROI masking (`spatial_normalization.py`),
then renders it on the real MNI152 template with `nilearn.plotting`.

**Why this wasn't possible before**: `fmri_preprocessing.py` saves each
stimulus's pattern as a flat masked voxel vector; the mask's shape/affine
needed to put that vector back into 3D space wasn't saved alongside it, and
no registration to a common space existed for whole-brain (non-ROI) runs.
Both gaps are now closed.

**Three-step pipeline, all opt-in (adds registration cost, so off by
default — nothing above needs it and existing runs are unaffected):**

1. **Preprocessing** — pass `--save-native-maps` (needs `--mask-cache-dir`;
   `--roi-set` may NOT be combined with it — an ROI-intersected pattern can't
   be unmasked back against the whole-brain mask this saves, see the
   `ValueError` in `fmri_preprocessing.py`). Saves each subject-session's
   whole-brain mask and triggers registration to MNI152 while the raw BOLD
   still exists (registration needs an EPI reference volume; by localization
   time the BOLD is long gone under the reclaim discipline in §1).
   ```bash
   python src/preprocessing/batch_preprocessing.py --data-dir data/brain/ds002236 \
       --output-dir data/processed/fmri/ds002236/Sem --task Sem --dataset ds002236 \
       --mask-cache-dir data/processed/fmri/ds002236/_masks --save-native-maps
   ```
   Or in `prepare_brain_rdms.sh`: set `SAVE_NATIVE_MAPS=1` (env var, mirrors
   `ROI_SET`) — `brainprep_subject.sh` only applies it when `ROI_SET` is
   unset, per the constraint above.

2. **Export** — add `--mask-cache-dir`/`--mni-maps-dir` to
   `run_brain_localization.py` (any non-`--finalize-only` call; `prepare_brain_rdms.sh`
   wires this in automatically off the same `SAVE_NATIVE_MAPS`/`MNI_MAPS_DIR`
   env vars). Writes `<mni_maps_dir>/<dataset>/<subject>_<session>_<phenomenon>_tmap_mni.nii.gz`,
   one real NIfTI per subject-session-phenomenon, in real MNI152 space.
   For the three cross-sectional datasets `session` here is the age-group
   bin (§3), not the BIDS session the subject was actually scanned in — the
   exporter resolves the real scan session on its own (mask/registration are
   properties of a physical scan, not an age bin) and skips a subject-session
   only if that resolution is genuinely ambiguous (more than one on-disk
   session for that subject), never by guessing.

3. **Render** — `scripts/render_brain_atlas_figures.py` aggregates every
   subject's map for a given (dataset, session, phenomenon) with a one-sample
   t-test at each voxel (never pooled across datasets — different scanners,
   tasks, populations) and renders it on the real MNI152 template:
   ```bash
   python scripts/render_brain_atlas_figures.py \
       --mni-maps-dir data/processed/fmri/ds002236/_mni_maps \
       --output-dir figures/brain_atlas
   ```
   `--display-mode ortho` (default, 3-slice cuts through the template) or
   `glass` (glass-brain projection); `--threshold` is a *visualization*
   threshold only, not a corrected significance level — these are exploratory
   figures. A group with only one subject is rendered as that subject's own
   map, labeled as such rather than as a group result.

**Verified with synthetic data** (real MNI-template-derived T1/EPI, real
`ds002236` stimulus lookup): the full chain — registration →
`brain_specialization()` with export enabled → `render_brain_atlas_figures.py`
— produces `.nii.gz` files with the correct MNI152 shape/affine and PNG
figures showing real anatomy (sulcal/gyral outlines, cerebellum, ventricles)
in both display modes, for both single-subject and n=4-subject groups, and
for the age-bin-vs-real-session mismatch case specifically (confirmed the
exporter resolves the real session and, separately, correctly refuses to
guess when more than one candidate session exists). Not yet run against real
GPU/data — that's the collaborator's side, same boundary as everything else
in this doc.

## 8. What's still ds003604-only

- **`ceiling_report.py`, `run_confound_check.py`** — still read stimulus
  properties from ds003604's `Stimulus_Characteristics.tsv` norms format.
  `positive_control.py` no longer belongs on this list (2026-08-29): it now
  handles both the stim_file-keyed layout (ds003604) and the PRIME/TARGET
  word-indexed layout ds002236/ds006239 actually ship (`--dataset` selects
  which columns `WORD_PAIR_NORMS` reads) — see §9 below for what that fixed.

## 9. Running locally, without a GPU cluster

The fMRI side of this pipeline (download → GLM → RDM → positive-control gate)
is CPU-only; only the language-model embedding side needs a GPU. Verified
end-to-end on a Mac laptop, 2026-08-29 (ds002236, 5 subjects) -- real bugs
found and fixed in the process, not just environment friction:

- **macOS ships bash 3.2** (Apple hasn't shipped a GPL3 bash since 2007).
  `prepare_brain_rdms.sh`/`brainprep_subject.sh` use bash-4+ `mapfile` and
  GNU-only `df` flags throughout -- both silently produce **zero output, exit
  0** on stock macOS bash rather than an error (`mapfile: command not found`
  looks like a warning, not a fatal one, from inside a script with
  `set -uo pipefail`... but the subsequent `REAL_TASKS` array being unset
  self-corrects wrong, and downstream just finds 0 patterns). Fix:
  `brew install bash coreutils`, then run with
  `PATH="/opt/homebrew/bin:/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH"`
  prepended so both the top-level script AND every `bash ...`/`df ...` it
  shells out to internally (via `xargs`) resolve to the GNU versions too --
  invoking the top-level script with `/opt/homebrew/bin/bash` alone is not
  enough, since `brainprep_subject.sh` is invoked as bare `bash` by `xargs`
  and re-resolves via `PATH`.
- **`--aggregation hyperalignment`** (the default, and what every published
  number here uses) needs `brainiak`'s SRM, which needs a working MPI install
  (`mpi4py`) -- not present on a stock Mac. Rather than requiring
  `brew install open-mpi` just to explore, `prepare_brain_rdms.sh` now reads
  an `AGGREGATION` env var (default unchanged: `hyperalignment`). Set
  `AGGREGATION=mean` to skip the dependency entirely. **Not a drop-in
  replacement for published numbers** -- a different aggregation is a
  different RDM, not a faster way to the same one.
- **A real, environment-independent bug**: `fmri_preprocessing.py` read TR
  from `bold_img.header.get_zooms()[3]`, which nibabel returns as
  `numpy.float32`. Newer nilearn (0.10.x+) validates
  `FirstLevelModel(t_r=...)` with `isinstance(t_r, (int, float))`, which
  numpy.float32 fails -- `'t_r' must be a float or an integer`. Every subject
  on every dataset would hit this once the environment's nilearn is new
  enough; fixed by casting to `float()` at the source. Whether the GPU
  cluster's pinned nilearn is old enough to not yet show this is untested --
  worth checking before assuming it's Mac-only.
- **`brain_localization.py`'s `build_stim_lookup_for_dataset`** (what
  `conditions_from_stim_lookup`/the positive-control gate's `condition`
  control now depends on) transitively imports `torch` via
  `src.language_models.circuit_localization`, even though the function
  itself only reads `events.tsv` -- real module coupling, not a hard
  requirement of what it does. Install CPU-only torch
  (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) to
  unblock without pulling in CUDA; worth a lazy-import cleanup in
  `circuit_localization.py` at some point so this dependency isn't forced on
  every caller of an events.tsv-only utility.
- Minimal local venv for this stage: `numpy scipy pandas nibabel nilearn dipy
  pyyaml matplotlib seaborn pillow scikit-learn brainiak` (+ `torch` per
  above if you need `build_stim_lookup_for_dataset`/brain localization).
  `torch`/`transformers` are NOT needed just to build RDMs and run the gate.

Example, matching what's verified above:

```bash
PATH="/opt/homebrew/bin:/opt/homebrew/opt/coreutils/libexec/gnubin:$PATH" \
DATASET=ds002236 PHENOMENA="Sem Phon" MAX_SUBJECTS=5 JOBS=4 \
WITHIN_RUN_NORM=1 AGGREGATION=mean \
  bash prepare_brain_rdms.sh
```

## 10. The three-level ROI analysis is now the standard, not an option

**Every dataset should be analysed at three masking levels, always**:

| `ROI_SET` value | Regions | What it's for |
|---|---|---|
| `phonology` | auditory + motor cortex (`Heschl_L/R`, `Temporal_Sup_L/R`, `Precentral_L/R`) | phonological/motor-adjacent processing |
| `language` | the language network ROI set | higher-level linguistic processing |
| `all` | union of the above (language + auditory + motor) | the combined footprint |
| *(unset)* | whole-brain | the existing default -- keep running it too, as the reference condition |

This was designed and unit-tested back in MASKING.md's original work, but **had never actually been run against real data until 2026-08-29** -- and that first real run surfaced two bugs serious enough that, before they were fixed, every `ROI_SET` run on any dataset would have silently produced whole-brain-equivalent output mislabeled as ROI-restricted, with no error. Both are now fixed (see MASKING.md and the `fix-control-gate-plumbing` branch history for the full detail); this section is the operational procedure now that it actually works.

### Prerequisite: T1 anatomicals

ROI masking needs a real T1w file per subject-session for registration (MASKING.md), and nothing before this download step ever resolved one — `batch_download_bold.py` only touches `func/`, `download_stimuli.py` only touches `stimuli/`. Run this once per dataset before any `--roi-set`/`ROI_SET` run:

```bash
python scripts/download_anat.py --dataset ds002236
```

Small (a few MB per subject) and idempotent -- safe to re-run, already-resolved files are left alone.

### Running all three levels

`ROI_SET` is an env var read by `prepare_brain_rdms.sh` (and, one level down, `scripts/brainprep_subject.sh`). Each value writes to its own `roi-<value>/` subdirectory under the RDM root, so the three levels (and the whole-brain default) never collide or overwrite each other -- run them back-to-back:

```bash
for ROI in phonology language all; do
  DATASET=ds002236 PHENOMENA="Sem Phon" MAX_SUBJECTS=20 JOBS=6 \
  WITHIN_RUN_NORM=1 ROI_SET=$ROI \
    bash prepare_brain_rdms.sh
done
```

`MASK_CACHE_DIR` defaults to `<RDM_ROOT>/_masks`, shared across all three loop iterations automatically (it's keyed off `RDM_ROOT`, not `ROI_SET`) -- so **registration is computed once per subject-session and reused for all three ROI levels**, not recomputed three times. Only the ROI-specific ordering (BOLD download -> GLM -> pattern extraction) repeats per level, since the mask itself changes what gets extracted.

On a Mac, prepend the bash/coreutils PATH fix from section 9 to every invocation above.

### How to know it actually worked, not silently fell back

Check `<mask_cache_dir>/roi_mask_status.csv` after any `ROI_SET` run -- this is the audit trail, not the RDM output itself:

```bash
python -c "
import csv
from collections import Counter
rows = list(csv.DictReader(open('data/processed/fmri/ds002236/_masks/roi_mask_status.csv')))
print(Counter((r['roi_set'], r['status']) for r in rows))
"
```

Every row should say `status=ok`. A `no_anat` or `registration_failed` status means that subject-session silently fell back to the whole-brain mask for that ROI set (logged, not crashed -- see `build_subject_roi_mask`'s design in MASKING.md) -- worth knowing before treating that subject as "language-restricted" in an analysis. Verified 2026-08-29 on ds002236, 20 subjects, all three levels: **75/75 rows `ok`, zero fallbacks**, dice range 0.53-0.82, real ROI voxel counts:

| level | mean voxels/subject | range |
|---|---|---|
| phonology | 1,135 | 887-1,687 |
| language | 2,653 | 2,043-4,120 |
| all | 3,788 | 2,930-5,807 |
| whole-brain (reference) | ~15,900 | (single-subject check) |

### Reading the results: a real caveat, not a null finding

In that same verification run, noise ceiling was consistently a little *lower* for every ROI-restricted condition than for whole-brain, at every session/task cell checked. **Do not read this as "ROI masking loses signal"** without more data: at n=4-8 subjects and a ~15,900-vs-1,000-to-3,800 voxel gap, a correlation-distance RDM's split-half reliability plausibly stabilises somewhat with more voxels regardless of whether those extra voxels carry task-relevant signal -- this is a known property of high-dimensional RDM estimation, not evidence the ROI captures a worse representation. The positive-control gate's verdict (0/30 stimulus tests significant) was identical across whole-brain and all three ROI levels in this run -- expected at this n, not a level-specific finding. Revisit this comparison once cohorts are larger; the point of this run was confirming the *pipeline* works end-to-end at all three levels, not drawing a scientific conclusion from the numbers yet.
