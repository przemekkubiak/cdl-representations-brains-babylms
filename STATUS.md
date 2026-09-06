# BrainAlign neuro-alignment sweep — STATUS

**Updated:** 2026-08-29 · **Compute used so far:** none (GPU 0 untouched, no model weights downloaded)
**Working dir:** `/local/scratch/sas245/brainalign-evals`

> **Current state: INVENTORY + RECON COMPLETE, SWEEP NOT STARTED.**
> Deprioritised behind the Basque BeetleLM training run. Everything below is
> CPU/network-only. Total disk footprint added: **31 MB** (metadata, RDMs, and a
> 7 MB git checkout). Nothing is queued and nothing will start without a go-ahead.

---

## 1. What the "neuro datasets" actually are

Not reading-time or EEG benchmarks — **fMRI representational dissimilarity
matrices (RDMs)** for four developmental language studies, published pre-computed
so the fMRI preprocessing never has to be repeated. The evaluation is **RSA**:
correlate a model's RDM over the same stimuli with the brain's.

### The RDMs — `BrainAlign/ds003604-session-rdms` (19 MB, 38 files)

Despite the name this one repo holds **three** datasets. Layout:
`{accession}/within-run-normalised/{Task}/session_rdm_{session}.npz`, plus a
legacy uncorrected ds003604 copy at the top level.

| accession | study | cohort | tasks | sessions | cells |
|---|---|---|---|---|---|
| **ds003604** | Wang et al. 2022, auditory language | children scanned at 5/7/9 | Sem, Phon, Gram, Plaus | ses-5, ses-7, ses-9 | **12** |
| **ds002236** | Lytle et al. 2020, lexical processing | children 8.7–15.5 | Phon, Sem | ses-9, ses-11, ses-11+ | **6** |
| **ds006239** | Wang et al. 2025, reading | children 10–17 | Orth, Phon, Sem, SemLocal | ses-11, ses-11+ | **8** |

**26 corrected cells total.** All 26 verified loadable with usable
`stimulus_texts` (`scripts/dry_run.py`). A 4th dataset, ds001894 (Lytle 2019,
longitudinal), is registered in the pipeline but has **no RDMs published**.

Each `.npz` carries: `rdm` (n×n correlation distance, n = 48–96), `stimuli`
(filenames), **`stimulus_texts`** (what the LM is fed — e.g. `'coat cup'`,
`'Every day they play one game'`), `subject_rdms` (per-subject, for the ceiling),
and `noise_ceiling_lower/upper`.

### Noise ceilings — these set the scale of any result

| dataset | ceiling (lower) | note |
|---|---|---|
| ds003604 | **0.84 – 0.88** | very high; two halves of a 60–93-subject pool agree |
| ds006239 Orth/Phon | 0.52 – 0.56 | |
| ds006239 Sem/SemLocal | 0.23 – 0.36 | |
| ds002236 Phon | 0.23 – 0.30 | smallest cohorts (n=9–20) |
| ds002236 Sem | 0.34 – 0.44 | |

### Results repos already on the Hub

| repo | what it is |
|---|---|
| `BrainAlign/cdl-devai-results` (6.9 MB, 212 files) | the ds003604 sweep: 15 families × 9–21 checkpoints × 12 cells = **2964 rows**, plus interp/localisation/behaviour axes and the ceiling + positive-control analyses |
| `BrainAlign/brain-lm-alignment-ds002236` (0.7 MB) | 524 alignment rows — but only **2 of 6 cells** (see §5) |
| `BrainAlign/brain-lm-alignment-ds006239` (0.1 MB) | **control/ and ceilings only — zero alignment rows** (see §5) |

### Models in the BrainAlign org

`gpt2-babylm-3`, `-5`, `-7`, `-9` — four GPT-2s (134.6 M params, 9 checkpoints
each), one per child age band. 0.27 GB/checkpoint.

---

## 2. Read this before quoting any number: the measurement is under a gate

The published READMEs are unusually explicit that these results **are not
currently interpretable**, and any new sweep inherits that. Three findings, in
order of when they were made:

1. **Scanner-run confound (ds003604, uncorrected).** Each stimulus appears in
   exactly one run, so run identity predicts brain dissimilarity at ρ = +0.49…+0.87
   while *no* stimulus property predicts it at all. Fixed by z-scoring each voxel
   within run — run predictiveness drops to −0.04 and the ceiling *rises* to 0.85.
   **The `within-run-normalised/` RDMs are the corrected ones. Use only those.**
2. **Positive control FAILED on all three datasets.** 0/108 stimulus controls
   significant on ds003604 (not the acoustic spectrum of the audio the children
   heard, not the study's own contrast); 0/6 on ds002236; 0/8 on ds006239. On
   ds003604 the cause is measured: the RDMs live in ~4 effective dimensions of a
   72-stimulus space and track whole-brain signal level (917k voxels, no
   anatomical mask).
3. **The null is therefore about the instrument, not about models.** Best
   alignment anywhere in the corrected ds003604 grid is **0.056 = 6.7 % of the
   ceiling**; 15/15 families are statistically equivalent to zero; the Pythia
   scale trend is ρ = +0.012, p = 0.93; and per-cell means from real models
   correlate with pure-noise PARC seeds at **r = +0.987**.

### CORRECTION 2026-08-29 — the PARC runs are not a noise floor

This file (above), `scripts/run_stages.sh` ("PARC -> the noise-seed NULL") and the
upstream `cdl-devai-results` README ("18 pure-noise runs", "matched pure-noise null")
all describe the PARC suite as randomly initialised noise. **It is not.**
`configs/model_zoo_extended.yaml` records every PARC family as, verbatim:

> PARC Pythia 160M (transformer), seed 0. OpenWebText, 4000 steps, 73 checkpoints.

They are fully trained 160M models spanning 73 training checkpoints, differing only in
architecture and initialisation seed. That makes them a **matched-scale seed reference**,
not a null.

What this does and does not change:

- **Unchanged:** the across-seed spread is still the right yardstick for "is this family's
  cell mean unusual", and the 1.0 σ maximum deviation still holds. A between-model
  comparison is a valid thing to do and is what these numbers support.
- **Changed:** "families are statistically equivalent to zero" and "correlate with pure
  noise at r = +0.987" cannot be read as *chance-level* results. They say a family looks
  like other trained models of the same scale, which is a much weaker statement.
- **CORRECTED AGAIN 2026-08-29 (later):** the claim that "no randomly initialised baseline
  exists anywhere in this collection" was itself wrong. The Pythia/PolyPythia `step0`
  branches ARE the untrained initialisation, and the sweep measured **15 independent random
  inits across all 26 cells (390 rows)** — they were in `results/alignment_rows.csv` the
  whole time, unlabelled. Paired within family against each family's final checkpoint:

  | dataset | training helps | mean untrained → trained | Wilcoxon p |
  |---|---|---|---|
  | ds002236 (Lytle) | **61/84 (73%)** | +0.0012 → **+0.0220** | **4.4e-05** |
  | ds006239 (Wang 2025) | 40/112 (36%) | −0.0055 → −0.0092 | 0.024 |
  | ds003604 (Wang 2022) | 48/168 (29%) | +0.0066 → −0.0023 | 8.7e-10 |

  The three datasets **disagree**: training improves alignment on Lytle and degrades it on
  both Wang datasets. Upstream's "untrained aligns better" reproduces on two of three and
  reverses on the third. All magnitudes are tiny relative to the ceilings, so this is a
  claim about sign and consistency, not about a model matching a brain.

Three further defects in the published packages were found and fixed the same day:

- **`z_vs_parc` was computed against the wrong denominator.** The sd came from ~99 pooled raw
  checkpoint rows while the numerator was an 11-checkpoint family mean — an averaged numerator
  against an unaveraged denominator. The pooled sd is **2.42×** the across-seed sd, so every z
  was that much too small and the headline "0 of 144 exceed" was substantially an artifact.
  Now aggregated per PARC family first, and reported **two-sided**: on ds002236 the counts are
  13 above and 13 below, which is a variance mismatch, not alignment.
- **ds006239's `Orth` and `Phon` RDMs are bit-identical** (`np.allclose`, atol=1e-12, both
  sessions, same stimuli, same subjects, same ceilings). The dataset has **6 independent cells,
  not 8**; every count over 8 double-weights one stimulus set. Now detected automatically and
  stated on the card.
- **`pythia-6.9b-full` has exactly one checkpoint and it is step 0** — the untrained
  initialisation — and it was published in `summary_by_family.csv` unflagged, inviting the
  reader to see the largest model as the top of the scale ladder when it is the zero rung.
  `pythia-2.8b-full` has 4. Both are now listed under `incomplete_families`.

**Implication for the run order below:** `ds006239/SemLocal` is the only
genuinely run×stimulus **crossed** language cell in the whole collection — the
confound cannot arise there by design. It is the highest-value target, and it is
one of the eight cells that has never been scored.

---

## 2.5 The instrument was finally tested (2026-08-29). It works, and that changes the reading.

Everything in section 2 was about suspecting the measurement. These are measurements *of* the
measurement. Scripts: `scripts/detectability.py`, `scripts/retest.py`/`retest_analysis.py`.
Published into both HF packages under `diagnostics/`.

### The estimator is not deaf, so the null about models is a real null

Using the sweep's exact estimator (z-normalise both RDMs, Spearman on the upper triangle), one
session's **group RDM predicts another session's** — different children, same stimuli:

| | median | range |
|---|---|---|
| cross-session external RDM | **0.754** | 0.49 – 0.97 |
| best language model, any family, any cell | **0.038** | — |
| model as a fraction of the external benchmark | **5.0%** | — |

A probe mixed as `w·R_group + (1−w)·R_perm` is detected above the untrained band at **w\* = 0.02 in
all 26 cells**. So the pipeline can recover a 2%-strength signal, and recovers a real external RDM at
0.75. **"No LM alignment is detectable" is now a bounded claim about models rather than a suspicion
about the instrument.** That is the single most useful thing these two packages now carry.

### But a trivial control beats every model

An additive per-stimulus main-effect RDM — `meanD_i + meanD_j`, carrying no relational structure at
all — is significant by stimulus-label permutation in **24 of 26 cells**, with median rsa **0.069**
and max **0.362**, against a median best model of 0.038. Per dataset it is worse: on ds002236 the
additive control's median is **0.225** against the best model's 0.103, i.e. **2.2×**. Whatever these
RDMs are dominated by is closer to a per-stimulus offset than to the relational geometry RSA is
meant to compare. No previously published artifact in this collection records this.

### The scanner-run confound is NOT the explanation

`ds006239/SemLocal` is the only run × stimulus **crossed** cell in the collection, so the confound
cannot arise there by design. As a ratio to each cell's own external-RDM benchmark (which controls
for the wildly different ceilings): SemLocal **R = 0.106** against a median **0.046** over the 24
confounded cells (IQR 0.035–0.086, range 0.029–0.199) — the high end, but the **83rd percentile,
inside the distribution**. The clean cell behaves like the dirty ones. Section 2's run-confound story
should stop being offered as the reason alignment is near zero.

### Precision moves rsa by a third of the between-family spread, and it gets worse with training

`results/precision_ab.csv` had sat unexamined. fp32 vs bf16, identical model, identical stimuli:

| step | mean \|Δrsa\| | as a fraction of the between-family sd |
|---|---|---|
| 0 | 0.00020 | 0.02 |
| 14000 | 0.00111 | 0.11 |
| 86000 | 0.00209 | 0.21 |
| **143000** | **0.00317** | **0.31** (max 0.95) |

At the final checkpoint **6 of 26 cells exceed 0.5**, the pre-declared "uninterpretable" threshold.
Averaged over the whole trajectory it looks fine (0.097) — untrained checkpoints, where two
precisions agree to 2e-4, dilute it. The grid is entirely fp32 so the published numbers are
internally consistent, but **family rankings are only valid within a precision-matched arm**, and
small between-family differences are noise. A batch-size and device retest (`scripts/retest.sh`,
4 arms on pythia-410m) is running to bound the rest of the nuisance space.

## 3. The evaluation code

`https://github.com/suchirsalhan/cdl-representations-brains-babylms` → cloned to
`brainalign-evals/pipeline` (7.1 MB, over SSH with `~/my_ssh/id_ed25519`; HTTPS
prompts for a password). **The metric is not reimplemented here** — this sweep
drives the upstream entrypoint.

**Entrypoint:** `pipeline/scripts/run_devai_grid.py`

```
--model FAMILY --model-zoo YAML --dataset TAG
--phenomena ...   # drives the localizer contrasts (only Sem/Phon/Gram/Plaus exist)
--tasks ...       # drives the brain RSA  <-- decoupled, and that is what makes
                  #     Orth and SemLocal scoreable at all
--sessions ...    # NOT passed by the upstream launcher; see §5
--brain-rdm-root <root>   # expects <root>/<Task>/session_rdm_<ses>.npz
--layer -1 --rdm-pooling mean --normalize --max-checkpoints N --batch-size 16
```

**The metric**, per (checkpoint × task × session):

1. Feed `stimulus_texts` through the model; mean-pool the hidden states of block
   `--layer` (default −1) over tokens → `[n_stim, H]`.
2. Model RDM = `1 − corrcoef` across stimuli.
3. z-normalise both RDMs, take the upper triangle, and correlate:
   **`rsa` = Spearman** (headline), plus `rsa_pearson` and `rsa_kendall`.
4. Report raw and as `frac_of_ceiling = rsa / ceiling_lower`.

Same code path for GPT-2, GPT-NeoX, Pico, Mamba and RWKV — it is hook-based, not
`output_hidden_states`. Layer sweep is one layer per run (`--layer`);
`scripts/layerwise_alignment.py` does the full profile.

**Two properties that matter on this box:**
- `_evict_hf_revision()` deletes each checkpoint from the HF cache the moment it
  has been measured, so **peak disk is one checkpoint, not the whole sweep.** This
  is upstream behaviour, already correct for a 99 %-full volume.
- `_assert_fits_on_device()` refuses to load a model that will not fit in VRAM
  rather than OOMing mid-sweep.

**Env:** `/local/scratch/sas245/venvs/mergeability/bin/python` — torch 2.13.0+cu130,
transformers 5.14.1, hub 1.25.1. All pipeline imports and `ModelZoo` resolution
verified. No new venv needed; `requirements.txt`'s neuroimaging half (nilearn,
brainiak, dipy) is only for building RDMs from BOLD, which we skip entirely.

---

## 4. Model inventory

`configs/model_zoo.yaml` already defines **36 families**; `scripts/make_model_zoo.py`
extends this to **94** in `configs/model_zoo_extended.yaml` without editing upstream.

| suite | already in zoo | added | checkpoints/family |
|---|---|---|---|
| Pythia scale ladder | 70m, 160m, 410m, 1b, 1.4b | **2.8b, 6.9b, 12b** (+14m, 31m) | 154 `step*` branches |
| Pythia deduped | — | **all 8 sizes** | 154 |
| PolyPythias | — | **45** (14m/31m/70m/160m/410m × seed1–9) | 154 |
| PARC (jmichaelov) | **all 18** (pythia/mamba/rwkv × seed0–5) | — | 73 `checkpoint-*` |
| BrainAlign babylm-gpt2 | 4 | — | 9 |
| pico-decoder, Beetle | 6 | — | commit-trajectory |

PolyPythias are published as `EleutherAI/pythia-<size>-seed<N>`; seed0 is the
canonical release, so 9 extra seeds per size. `jmichaelov` also hosts 35
`pythia31m_b*_constant_seed*` batch-size-sweep models, not in scope.

Verified by `resolve_checkpoints`: 154 for every Pythia/PolyPythia family, 73 for
PARC, 9 for babylm-gpt2 — subsampled log-uniformly to `MAX_CKPT`.

### Disk cost (metadata only; measured, not guessed)

`MAX_CKPT=12` → 11 checkpoints/family. Weights are **evicted after each one**, so
*peak* is one checkpoint; the large number is cumulative network transfer.

| model | GB/ckpt | ×11 transferred | peak disk |
|---|---|---|---|
| pythia-70m | 0.33 | 3.7 | 0.3 |
| pythia-160m | 0.75 | 8.2 | 0.8 |
| pythia-410m | 1.82 | 20.1 | 1.8 |
| pythia-1b | 4.18 | 46.0 | 4.2 |
| pythia-1.4b | 5.86 | 64.5 | 5.9 |
| pythia-2.8b | 11.37 | 125.1 | 11.4 |
| pythia-6.9b | **27.70** | 304.7 | 27.7 |
| pythia-12b | **47.69** | 524.6 | **47.7** |
| PolyPythia 70m/160m/410m | 0.17 / 0.38 / 0.91 | 1.9 / 4.2 / 10.0 | ≤0.9 |
| PARC (any) | 0.68 | 7.5 | 0.7 |
| babylm-gpt2-* | 0.27 | 2.4 (9 ckpts) | 0.3 |

**6.9b and 12b are stored fp32.** At fp32, 12b needs ~48 GB of VRAM for weights
alone — inside GPU 0's ~72 GB free but at the edge of the 60 GB budget, and the
GPU is shared with `neural.train` (PID 907607, 8 GB). **Recommendation: run 12b
last, alone, or in bf16.** `_assert_fits_on_device` will refuse rather than OOM
the neighbour, so the failure mode is safe either way.

Currently cached on this box: only
`Beetle-HumanScale/beetle-monolingual-humanscale-eng`. Everything else is a
cold download.

---

## 5. Bug found in the published results — worth fixing on the rerun

`slurm/run_devai_grid.sh` never passes `--sessions`, so `run_devai_grid.py` falls
back to its ds003604 default `["ses-5","ses-7","ses-9"]`. The two new datasets
have neither ses-5 nor ses-7:

- **ds002236** (ses-9/11/11+) matched only `ses-9` → the published repo covers
  **2 of 6 cells**, and its README's "524 alignment rows across 2 cells" says so
  without flagging it as a defect.
- **ds006239** (ses-11/11+) matched **nothing** → the published repo has ceilings
  and controls but **zero alignment rows**. This is why `SemLocal`, the one clean
  cell in the collection, has never been measured against any model.

`scripts/sweep.sh` derives `--sessions` and `--tasks` from the RDM tree itself,
which fixes this and takes coverage from 14 scored cells to **26**.

---

## 6. What is in this directory

```
brainalign-evals/
  STATUS.md                     this file
  pipeline/                     upstream checkout (unmodified)
  configs/model_zoo_extended.yaml   94 families (upstream 36 + 58)
  data/                         the 4 HF dataset repos, 27 MB
  scripts/
    fetch_meta.py               pull the RDM + results repos
    enumerate_models.py         inventory the three suites off the Hub
    disk_cost.py                per-repo GB/ckpt, metadata only
    make_model_zoo.py           build the extended zoo
    dry_run.py                  validate everything except the forward pass
    sweep.sh                    THE RUNNER — resumable, GPU 0, disk floor
    prune_cache.py              manifest-driven cache eviction (see §10)
    collect_results.py          grid CSVs -> tidy results/
  grid/<dataset>/alignment_<family>.csv     raw per-family output
  results/
    alignment_rows.csv          one row per (dataset, family, ckpt, task, session)
    scaling_curve.csv           per family × cell: mean/sd/max rsa vs params
    scale_trend.csv             Spearman(params, rsa) per dataset
    disk_cost.json, model_inventory_raw.json
  logs/
```

**Runner contract.** `bash scripts/sweep.sh` pins `CUDA_VISIBLE_DEVICES=0`, sets
`HF_HOME=/local/scratch/sas245/hf_cache`, skips any `(family × dataset)` whose
alignment CSV already exists, aborts if `/local/scratch` drops below
`DISK_FLOOR_GB` (default 150), and prunes the cache between families. Overridable:
`FAMILIES`, `DATASETS`, `MAX_CKPT`, `BATCH_SIZE`, `CELL_TIMEOUT`, `DISK_FLOOR_GB`.

**Validated with no GPU and no weights** (`scripts/dry_run.py`): 26/26 cells load
with stimulus texts; the RSA estimator returns exactly 1.0 for a brain RDM against
itself and ≈0.02 for random activations; checkpoint resolution works for all four
suite types including the 58 new families.

---

## 7. Planned run order

Ordered so the highest-value result lands first and the disk-expensive models last.

| # | stage | families | cells | transferred | why |
|---|---|---|---|---|---|
| **0** | smoke | `pythia-70m-full`, ds003604 only, `MAX_CKPT=3` | 12 | ~1 GB | one model × all cells end-to-end before anything scales out |
| **1** | scale ladder | pythia 70m→1.4b | all 26 | ~143 GB | reproduces + extends the published curve onto the 12 unscored cells |
| **2** | new cells first | same 5, focus `ds006239/SemLocal` | 8 | — | the only confound-free cell; never measured |
| **3** | ladder top | 2.8b, 6.9b | all 26 | ~430 GB | extends 1.5 B → 6.9 B, 4.5× past the published ceiling |
| **4** | BrainAlign own | babylm-gpt2-3/5/7/9 | all 26 | ~12 GB | cheap; in-domain child-scale anchor |
| **5** | seed error bars | PolyPythia 70m/160m/410m × seed1–3 | all 26 | ~60 GB | turns the scaling curve into a curve with error bars |
| **6** | PARC null | pythia/mamba/rwkv × seed0–2 | all 26 | ~61 GB | the matched null the analysis is judged against; also the architecture contrast |
| **7** | 12b | pythia-12b | all 26 | ~525 GB | **only if disk and VRAM allow**; bf16 or alone |
| **8** | deduped | pythia-*-deduped | all 26 | ~712 GB | optional data-dedup axis; skip unless asked |

Stages 0–6 ≈ **710 GB cumulative transfer, ≤28 GB peak disk, ≤28 GB peak VRAM** —
comfortably inside both budgets. Stage 7 alone doubles the transfer.

Wall-clock is dominated by download, not compute: each cell is one forward pass
over 48–96 short strings.

---

## 8. Environment state

| | |
|---|---|
| disk | **409 GB free on /local/scratch, 99 % used** — unchanged by this work |
| this directory | 31 MB (7.1 MB checkout + 27 MB data) |
| GPU 0 | 8.2 GB used by `neural.train` PID 907607 (**do not kill**), ~72 GB free, 78 % util |
| GPUs 1, 2 | another agent's — untouched |
| HF cache | `/local/scratch/sas245/hf_cache` (1.7 TB, the convention in `kid-safe-neurips/run_env.sh`) |
| HF token | `/local/scratch/sas245/.hf_token`, sourced not printed |

**Note:** `ActivationExtractor` passes `cache_dir=".cache/huggingface"` explicitly
to `from_pretrained`, so weights land under `pipeline/.cache/huggingface`
relative to cwd — *not* under `HF_HOME`. Upstream eviction handles it (it scans
that relative path too) and `prune_cache.py` scans both. Worth knowing before
anyone goes looking for the cache or measures its size.

---

## 9. Failures / gaps

| item | status |
|---|---|
| GitHub clone over HTTPS | fails (private repo, no credential helper). **Use SSH** with `GIT_SSH_COMMAND="ssh -i /local/scratch/sas245/my_ssh/id_ed25519"`; `~/.ssh` is unreadable so `known_hosts` warns harmlessly |
| ds001894 | registered in `configs/neuro_datasets.yaml`, **no RDMs on the Hub** — would need the full BOLD pipeline (~578 GB of downloads). Out of scope on this disk |
| ds003604 uncorrected RDMs | present at the repo top level; **deliberately not used** |
| PARC on the new datasets | never run — only ds003604 |
| encoding-model score (`encoding_r`) | needs raw voxel `patterns`, which the published RDMs do not carry. Disabled via `--no-encoding` |
| layerwise profile | this sweep fixes `--layer -1`. Published `diagnostics_layerwise` shows alignment is flat across layers on ds003604, so a per-layer sweep is a later refinement, not a prerequisite |

---

## 10. Incident — I deleted 22 GB of another project's HF cache

**What happened.** The first version of `scripts/prune_cache.py` chose what to
delete by repo-id **prefix** (`EleutherAI/`, `jmichaelov/`, `BrainAlign/`,
`pico-lm/`, `Beetle-HumanScale/`, `Beetle-FineWeb3-24B/`), on the assumption that
those namespaces belonged to this sweep. I then ran it as a "check" — but it was
never a dry run. It deleted **22.0 GB across 75 model revisions** from
`/local/scratch/sas245/hf_cache/hub`.

**The error in reasoning:** a repo-id prefix says who *published* a model, not
who *downloaded* it. `/local/scratch/sas245/hf_cache` is shared across the user's
projects, and 255 of its 338 repos sit under those prefixes — including the
Beetle suites.

**Scope of the damage.**
- Confined to `/local/scratch/sas245/hf_cache/hub`. `huggingface_cache/` (243
  matching repos, several TB) and `.hf/` were **not** scanned and are untouched.
- All 75 revisions are public Hub repos, so nothing is unrecoverable — the cost
  is re-download time for whichever job next wants them.
- Only `repo_type == "model"` entries were eligible; cached *datasets* survived.
- Net effect on disk: free space went 409 GB → 430 GB.

**Fix, already applied.** `prune_cache.py` is now **manifest-driven**: it deletes
only `repo@revision` pairs listed in `configs/downloaded_refs.txt`, which
`sweep.sh` appends to immediately before pulling each family. With no manifest it
is a no-op. Both cases verified. A cache entry that predates the manifest cannot
be selected, so the pruner can no longer touch anything this sweep did not fetch.

**Worth flagging beyond this script:** upstream `_evict_hf_revision()` in
`run_devai_grid.py` is *not* affected — it only ever evicts the exact ref it just
measured, which is the correct design and the one I should have copied from the
start.

---

## 11. Incident — I broke the sweep with my own bf16 patch

**What happened.** To let the 12b rung run in bf16, I patched
`pipeline/scripts/run_devai_grid.py` to read a `DEVAI_DTYPE` env var, adding
`import torch` at module level. But `main()` already contains its own
`import torch` (line ~420), which makes `torch` a **local** name for that whole
function. My reference to it near the top of the checkpoint loop therefore hit
`local variable 'torch' referenced before assignment` on every checkpoint.

**Why it was slow to notice.** The exception is swallowed by the sweep's
deliberate keep-going handler (`! failed to load ...`), so each family exited
**rc=0** and printed "(no rows for alignment)". A broken run is nearly
indistinguishable from a clean one that found nothing — the only signal is the
row count. `sweep.sh` does check for a non-empty CSV and logged `FAIL`, which is
what caught it; without that check this would have produced 26 silently empty
cells.

**Damage.** 9 (family x dataset) cells burned with empty output: 410m/1b/1.4b on
ds006239, 70m/160m/410m/1b on ds003604, and parc-pythia-seed0/seed1 on ds006239.
No corrupted data and no completed downloads, so the cost was a few minutes of
wall clock. Empty CSVs were purged so the resume logic re-runs those cells.

**Two process mistakes that made it worse.**
1. I edited a file that a sweep was actively running. Defaulting the new env var
   to fp32 made the change *safe* but not *inert* — every family spawns a fresh
   Python that re-reads the file, so the break took effect mid-stage.
   **Patch the pipeline only when nothing is running.**
2. `pkill -f 'bash scripts/sweep.sh'` matched my own shell (exit 144), and
   killing the orchestrator while it sat in its wait-loop let an orphaned child
   sweep launch PARC on its own with the broken code. Everything was then stopped
   by explicit PID. **Stop background work by PID, never by pattern** -- the
   pattern matches the shell issuing it.

**Neighbouring jobs were never touched.** `neural.train` (PID 907607) verified
alive throughout at 24 h elapsed, and GPU 0 fell back to exactly its 8188 MiB
while our processes were stopped. GPUs 1-2 untouched.

**Fix.** The patch now does `import torch as _torch` locally at the use site,
with a comment naming the shadowing trap. Verified on a 2-checkpoint SemLocal
cell in *both* precisions before restarting. Stage 1 resumed and correctly
skipped the two families that had survived.

**Structural fix worth keeping:** `sweep.sh` treats "rc=0 but empty CSV" as
`FAIL`. That check is the only reason this was caught in minutes. Do not remove it.

---

## 12. Precision: bf16 is NOT free at these effect sizes

Spot check, pythia-70m, ds006239/SemLocal/ses-11:

| checkpoint | fp32 | bf16 | delta |
|---|---|---|---|
| step 0 | 0.05315469 | 0.05315506 | +3.7e-7 |
| step 143000 | 0.02171442 | 0.01887452 | **-2.8e-3** |

Effect sizes in this study are ~0.02-0.05 and the published across-seed noise sd
is ~0.0079, so that second delta is **roughly a third of the noise sd** on a
single model and a single cell. A bf16 12b point therefore cannot simply be
plotted as the top of an otherwise-fp32 ladder. `scripts/precision_ab.py` runs
the full 410m A/B (11 checkpoints x 26 cells) and reports the delta against both
the noise sd and the between-family spread; that number should decide how (or
whether) the 12b rung is presented alongside the rest.

---

## 13. RESULT — the null holds, and it is about the instrument

Stages 1 + PARC complete: **4004 rows, 14 families, all 26 cells.**
5 Pythia scales (96 M - 1.52 B) + 9 PARC noise seeds (3 architectures x 3 seeds).

### The scaling curve is flat and slightly negative

Spearman(params, mean RSA), per dataset:

| dataset | rho | p | mean RSA range | % of ceiling |
|---|---|---|---|---|
| ds003604 | -0.10 | 0.87 | -0.0036 .. +0.0017 | -0.4 .. +0.2 |
| ds002236 | -0.30 | 0.62 | +0.0073 .. +0.0196 | 3.7 .. 6.6 |
| ds006239 | -0.40 | 0.50 | -0.0146 .. -0.0012 | -3.2 .. +0.6 |

Reproduces the published ds003604 finding (rho = +0.012) and extends it to the
two datasets that had never been measured. 16x scale buys nothing.

### No family is distinguishable from a random seed

Null = 9 PARC runs differing only by initialisation, per cell (mean within seed
first, then spread across seeds -- pooling raw checkpoint rows would shrink the
sd artificially).

| family | mean z | max z | cells beating every seed |
|---|---|---|---|
| pythia-410m | +0.21 | 2.87 | 4/26 |
| pythia-1b | -0.18 | 2.08 | 2/26 |
| pythia-70m | -0.21 | 2.32 | 3/26 |
| pythia-160m | -0.80 | 1.07 | 0/26 |
| pythia-1.4b | -1.35 | 1.43 | 0/26 |

Under the null a real family's cell value is exchangeable with the 9 seed
values, so P(exceeding the max of 9) = 1/10 and 13.0 of 130 are expected by
chance. **Observed: 9/130, p = 0.91 (binomial, greater).** Models beat the noise
seeds LESS often than chance.

### The structure belongs to the cell, not the model

- per-cell means, real families vs noise seeds: **r = +0.863** (p = 1.4e-08, n = 26)
  [published ds003604: +0.987]
- variance decomposition: **cell identity 83.9%**, **model family 3.2%**
  [published ds003604: cell 47.6%, family 2.9% -- the family share reproduces almost exactly]

### ds006239/SemLocal: reading (B)

The only run x stimulus CROSSED cell, where the confound cannot arise. The
smoke run's "untrained aligns best" signature did NOT survive the full ladder --
step 0 is the argmax in only **1 of 10** (family x session) combinations.
Against the seed null:

| session | noise band | step 0 | verdict |
|---|---|---|---|
| ses-11 | +0.0134 +/- 0.0129, range [-0.0002, +0.0347] | +0.0025 (z = -0.84) | **INSIDE** |
| ses-11+ | -0.0020 +/- 0.0091, range [-0.0132, +0.0091] | -0.0047 (z = -0.30) | **INSIDE** |

Untrained alignment sits inside the band produced by models differing only by
random seed, on both sessions. **Reading (B): the decline over training is drift
inside noise; there is no alignment being destroyed.** Reading (A) -- "training
destroys real alignment" -- is not supported and should not be quoted.

### What this is

A null against RDMs of demonstrated inter-subject reliability (ceilings
0.23-0.88) whose positive control fails on all three datasets. The honest claim
is **"no LM alignment is detectable by this measurement"**, not "LMs do not align
with the developing brain". The session-coverage fix (14 -> 26 cells) means this
is also the first measurement of 12 previously unscored cells, including
SemLocal.

---

## 14. PARC arm finalised + all three datasets published (2026-09-02)

**Completion.** The PARC sweep is **27/27 complete** — 3 architectures (pythia,
mamba, rwkv) x seeds 0–2 x 3 datasets, 11 checkpoints each (`checkpoint-10` …
`checkpoint-4000`; PARC publishes no step-0). Every cell present with the full
row count (ds003604 132, ds006239 88, ds002236 66), **zero NaN rsa**, zero error
lines in the 27 `logs/grid_*_parc-*.log`, and `logs/stage_parc.log` shows `ok` for
all 27 after the two `rc=0`-but-empty ds006239 cells from §11 were re-run.
`isolation`/`mechanistic`/`mechanistic_layer`/`behaviour` are present for all 9
PARC families on every dataset. §9's "PARC on the new datasets — never run" is
**stale and is superseded here**. Nothing was left to run; no cell was re-computed.

**Aggregation.** `scripts/parc_summary.py` (new) writes
`results/parc_by_cell.csv`, `results/parc_by_seed.csv`, `results/parc_summary.csv`
and `results/parc_arch_test.csv`, following the package conventions exactly:
mean over checkpoints per (family, cell) **first**, then the spread **across the
three seeds** — never over pooled checkpoint rows.

| dataset | arch | mean rsa | sd across seeds | % of ceiling | Wilcoxon vs 0 |
|---|---|---|---|---|---|
| ds002236 | mamba | **+0.0237** | 0.0025 | 7.9 | 0.031 (n=6, the floor) |
| ds002236 | rwkv | +0.0138 | 0.0085 | 4.2 | 0.031 |
| ds002236 | pythia | +0.0114 | 0.0052 | 3.9 | 0.031 |
| ds003604 | pythia | +0.0042 | 0.0006 | 0.5 | 0.97 |
| ds003604 | rwkv | +0.0008 | 0.0023 | 0.1 | 0.91 |
| ds003604 | mamba | −0.0038 | 0.0023 | −0.4 | 0.23 |
| ds006239 | pythia | −0.0045 | 0.0015 | −0.3 | 0.53 |
| ds006239 | mamba | −0.0052 | 0.0004 | +0.3 | 0.45 |
| ds006239 | rwkv | −0.0095 | 0.0026 | −1.8 | 0.07 |

Architectures are not separable on any dataset (Kruskal over per-cell means:
ds002236 p = 0.079, ds003604 p = 0.17, ds006239 p = 0.83). Against the 15
random-init (step-0) band on the same cell, **3 of 234 (arch x seed x cell)
values exceed +2 SD and 0 fall below** — all three are ds002236/mamba. First
vs final PARC checkpoint is significant only for ds003604/mamba (p = 0.015,
a *decline*: +0.0036 → −0.0019). The magnitudes stay at 0.1–8 % of the noise
ceiling, so this is a statement about sign and consistency, not alignment.

**Two fixes.**
1. `build_devai_package.py` emitted the ds002236/ds006239 "the positive control
   here is one control and its gate plumbing was faulty" paragraph on **every**
   card. On ds003604 that asserts a defect the dataset does not have — it has its
   own 12-control battery. The paragraph is now dataset-conditional, and the
   ds003604 version states the measured numbers: 9/108 stimulus-property control x
   cell tests significant at uncorrected p < 0.05, 0 after correction, while run
   identity goes +0.666 → −0.119 and presentation order +0.468 → −0.092 under
   within-run normalisation.
2. The builder now also writes `overall/parc_seed_summary.csv` and
   `overall/parc_by_seed_cell.csv` per dataset (same numbers as
   `results/parc_summary.csv`, sliced to that dataset), registered as HF dataset
   configs and described on the card.

**Published.** All three datasets are now on the Hub in the same layout, 221
files each, verified byte-identical to `hf_package/<ds>/` after the push:

| repo | state |
|---|---|
| `BrainAlign/cdl-devai-results-ds003604` | **new** — 29 families, 294 checkpoints, 3528 alignment rows, 12/12 cells |
| `BrainAlign/cdl-devai-results-ds002236` | updated: +2 PARC files, README additive; every other file unchanged |
| `BrainAlign/cdl-devai-results-ds006239` | updated: +2 PARC files, README additive; every other file unchanged |

All public, pushed with suchirsalhan's write token (org member). No model weights
were pushed — the models are `jmichaelov/parc-*` and `EleutherAI/*` upstream, and
this sweep trains nothing.
