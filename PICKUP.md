# PICKUP — the unattended BrainAlign run

Written for someone returning cold with no memory of setting this up.
Started 2026-08-18 by an automated agent. Everything below was verified on this box.

---

## The one command

```bash
cd /root/cdl-representations-brains-babylms && bash status.sh
```

It prints liveness, the ledger, free disk, our GPUs, and the tail of every log.

**On liveness:** `status.sh` deliberately does not use `kill -0`. PID 1 does not reap
children on this box, so `kill -0` succeeds against a **defunct** process and has already
produced a false "ALIVE" reading here. Liveness requires `ps -o stat=` to return a state
that is not `Z*`. If you write your own check, do the same.

---

## GPUs — the old reservation is LIFTED (verified 2026-08-26)

**All eight cards are free.** `nvidia-smi` reports 0 MiB used and no compute
processes on any of 0–7, and the 96-hour merge sweep in `/root/mergeability` that
owned 4–7 has finished — only a metrics-publishing loop remains, which uses no
GPU. The user confirmed the other project no longer holds cards.

| GPUs | Status |
|------|--------|
| **0–7** | **all available to this project** |

The historical rule was: 0–2 ours, 3 reserved, 4–7 another project's 96-hour merge
sweep. That is why `env_brainalign.sh` still pins `CUDA_VISIBLE_DEVICES=0,1,2` and
why the launchers default to `GPUS=0,1,2`. **The pin is now a conservative default,
not a hard constraint** — widen it to `0,1,2,3,4,5,6,7` for anything that would
benefit, and check `nvidia-smi` first in case the other project has started
something new.

What has NOT changed: `slurm/run_devai_grid.sh` is an `#SBATCH` file, and under
plain `bash` those headers are inert comments, so nothing scopes the devices by
itself. Whatever you want a process to see still has to be set explicitly.

---

## Disk floor: 350 GB. Still the right number.

`/` is a **single 1.8T overlay shared with `/root/mergeability`**. That project's
sweep has finished and it ran a large disk reclaim (the box went from ~124 GB free
to ~900 GB), so the floor is no longer defending an actively-aborting neighbour —
but it is still a shared volume, this project can consume hundreds of GB in a
single stage, and the streaming design below depends on the floor being enforced.
Leave it at 350. The driver checks free space before
every stage *and every 60 s during* one, and terminates the stage if it is breached.

This matters more than it sounds. ds003604 is **3851 BOLD runs at ~154 MiB each (~578 GiB)**,
and preprocessing turns each run into **~176 MiB of voxel patterns (~660 GiB more)**. Free space is ~865 GB as of
2026-08-26, so the real budget is **~515 GB**. Downloading it all and keeping it — which is
what the original inline pipeline does — **would still breach the floor**, and the overlay
is shared, so it would take the neighbour with it.

So `prepare_brain_rdms.sh` streams: each subject's raw BOLD is deleted the moment it is
preprocessed, and each task's patterns are deleted once its session RDMs exist. Peak is
~190 GB instead of ~1.2 TB. Same inputs, same science. **If you change that script, keep
the deletions.**

---

## What is running

`run_tiers_detached.sh`, launched with `setsid` so it survives the ssh session (verify with
`ps -o ppid=` — it should report **1**). Stages in order, each with its own log, ledger
entry, and wall-clock cap:

| Stage | What | Cap |
|-------|------|-----|
| **tier 0** | brain prep: BOLD → preprocess → session RDMs, streamed | 24 h |
| **tier 1** | DevAI/workshop: alignment + isolation + mechanistic, no ablation | 12 h |
| **tier 2** | ICLR core: + causal ablation, behaviour, encoding, bootstrap CIs, held-out CV | 18 h |
| **tier 3** | ICLR strong: dense trajectory, all 126 checkpoints | 48 h |

Tier 0 must finish first — without brain RDMs the grid produces **no alignment rows at all**,
which is most of the point.

A stage that fails is recorded and the driver moves to the next one; one failure does not
cost the rest. Exceeding a cap kills the stage, not the driver.

---

## Stop it

```bash
kill -TERM $(cat logs/driver.pid)     # traps, kills the child process group, exits
```

Do **not** `kill -9` the driver first — that orphans its children on the GPUs. If you do it
anyway, clean up with `nvidia-smi` and kill the leftover pids. (Historically this said
"cards 0-2 only", because 4-7 belonged to another project's sweep. That sweep is finished —
see the GPU section — so check what is actually running rather than assuming ownership.)

## Restart it

```bash
bash launch_detached.sh
```

It resumes from `logs/tier_ledger.json`: any stage recorded `ok` is skipped, everything else
is re-attempted. It refuses to start if a live driver already exists (and is zombie-aware
about that check). Tier 0 additionally skips any task whose session RDMs are already built,
so a restart never re-downloads finished work.

---

## Things that will trip you up

**1. `pip install -r requirements.txt` gives you a torch that cannot see the GPUs.**
`torch>=2.0.0` unpinned resolves to a **cu13** wheel (`2.13.0+cu130`); the driver here is
535.309.01 / **CUDA 12.6**, and cu13 needs a newer one. `torch.cuda.is_available()` returns
**False** and the entire pipeline silently runs on CPU — `device_count()` still returns 3,
so that is not a safe check on its own. Fixed by installing
`--index-url https://download.pytorch.org/whl/cu126 "torch>=2.6,<2.9"` → `2.8.0+cu126`.
**If you rebuild the venv, redo this, and confirm `torch.cuda.is_available()` is True
before launching anything.**

**2. Every pico and Beetle checkpoint used to be skipped in silence.**
`ActivationExtractor` read `config.hidden_size`; `PicoDecoderHFConfig` only has `d_model`,
so every pico/Beetle load raised `AttributeError`, was caught by a per-checkpoint `except`,
and the grid wrote "(no rows for ...)" and **exited 0**. Since pico and Beetle are the
backbone of the suite, the sweep produced nothing while looking successful. Fixed with
`resolve_hidden_dim()`. The general lesson applies to this whole repo: **exit 0 means very
little here** — `run_devai_grid.sh` swallows per-family failures on purpose. The driver now
verifies declared outputs and non-zero GPU memory before calling a stage `ok`.

**3. The dataset checkout is not created by anything the grid calls.**
`scripts/batch_download_bold.py` resolves git-annex symlinks into OpenNeuro URLs but
*requires* the checkout to exist. On a clean clone the grid died in 4 s with
`ValueError: No subjects found in data/brain/ds003604`. `prepare_brain_rdms.sh` bootstraps it.
Note the checked-out `.nii.gz` paths are **dangling annex symlinks** until downloaded — a
390 MB `data/brain/ds003604` means metadata only, not data. `datalad` and `git-annex` are
**not installed**, so `scripts/download_bold_datalad.py` is a dead end; do not install them.

**4. pico step-0 revisions have no weights and no tokenizer.**
The first commit on each pico run branch predates the weights, so `step=0` always fails to
load (`Unrecognized configuration class ... to build an AutoTokenizer`). This is upstream,
not ours. It costs the earliest point of each trajectory; every later checkpoint is fine.

**5. `$SCRATCH` is unset.** The runbook's `$SCRATCH/hf_cache` would resolve to `/hf_cache`.
We use a dedicated `HF_HOME=/root/hf_cache_brainalign`. Do not point this at another
project's HF cache — sharing one has caused failures on this box before. The repo-relative
`.cache/huggingface` is a symlink to it, because `ActivationExtractor` passes an explicit
`cache_dir` that would otherwise put ~150 GB **inside the git repo**.

**6. The HF token** is in `/root/.ms_hf_env` (mode 600), sourced by `env_brainalign.sh`.
Never echo, log, or commit it.

**7. Models load in fp32.** That is the extractor's default and it was left alone: these are
small models (11 M–1 B params) and fp32 of the largest is a few GB against 80 GB cards, so
changing the dtype would alter the scientific numbers for no operational gain. A pre-load
VRAM check (`_assert_fits_on_device`) now refuses any load that would not fit in 75% of free
VRAM, so a marginal model is skipped and recorded rather than risking an OOM on a shared card.

---

## What was deliberately NOT run

**Tier 3, second half — the cross-dataset / second-neuro-dataset arm. Skipped.**

The runbook line was `DATASET=ds00XXXX DATA_DIR=data/brain/ds00XXXX ...`. `ds00XXXX` is a
**placeholder, not a real accession**, and it could not be resolved: `ds003604` is the only
accession named anywhere in the repo (`configs/model_zoo.yaml`, the README, the contrasts).

More decisively, **the download path is hardcoded to ds003604**.
`scripts/batch_download_bold.py` builds every URL from `OpenNeuroDatasets/ds003604` — the
annex base, the S3 path and the OpenNeuro snapshot path — and there is no argument that
changes it. `scripts/build_contrasts.py` likewise pulls ds003604 stimuli. So setting
`DATASET=` to any other tag would **re-download ds003604 into a directory named after a
different study and label those rows as a second dataset** — fabricated data, in a figure
(`fig10_cross_dataset`) whose whole claim is cross-dataset generalisation.

No accession was guessed. `run_devai_bare.sh` has no tier-3b branch, and the ledger records
`tier3b: skipped` with this reasoning. **To enable it you need both a real accession and a
dataset-aware download path.** (Unrelated but worth knowing: `make_figures.py --self-test`
synthesises a fake second dataset purely so Fig 10 renders. That flag is not used by any
tier here. Do not confuse its output with real results.)

---

## Where things are

```
env_brainalign.sh        GPU pin, HF cache, token. Source this for anything manual.
run_devai_bare.sh        --tier {0,1,2,3} | --smoke   bare-metal entry point
prepare_brain_rdms.sh    stage 0, streamed (scripts/brainprep_subject.sh = per-subject unit)
run_tiers_detached.sh    the driver
launch_detached.sh       setsid launcher
status.sh                the status command
logs/tier_ledger.json    per-stage status, timings, failure class, peak GPU memory
logs/tier{0,1,2,3}.log   per-stage logs
logs/driver.log          driver's own log
slurm/run_devai_grid.sh  UNCHANGED cluster path — still the way to sbatch this elsewhere
```

Results land in `data/processed/language_models/devai_grid/ds003604/` and are summarised
into `paper_results/`, which is committed and pushed after each **successful** stage.
Both `origin` and `przemek` were verified writable from this box (auth is via the `gh` CLI
credential helper as `suchirsalhan`). Failed stages do not publish — `przemek` is a
collaborator's repository and a failed run reached it once already, before that gate existed.
