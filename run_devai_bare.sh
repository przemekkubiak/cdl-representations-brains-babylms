#!/bin/bash
# Bare-metal entry point for the DevAI grid.
#
# WHY THIS EXISTS
#   slurm/run_devai_grid.sh carries #SBATCH headers (--gres=gpu:a100:1, --partition=gpu,
#   --time=12:00:00) and is meant to be `sbatch`ed. There is no SLURM on this box, so under
#   plain `bash` every one of those headers is an inert comment: nothing scopes the visible
#   devices, nothing caps wall-clock, nothing caps memory. In particular the process would
#   see ALL EIGHT cards, including 4-7, which are running a 96-hour merge sweep for another
#   project. This wrapper supplies the scoping the scheduler would have supplied.
#
#   slurm/run_devai_grid.sh is kept as the cluster path and is still the thing that does the
#   actual work -- this only sets the environment and the tier's variables.
#
# TIERS  (previously long ABLATE=... MAX_CKPT=... incantations in someone's shell history)
#   --tier 0   brain prep : BOLD -> preprocess -> session RDMs, streamed per task
#              (must run first; the tiers need the RDMs to produce alignment rows)
#   --tier 1   DevAI/workshop : alignment + isolation + mechanistic, no causal ablation
#   --tier 2   ICLR core      : + causal ablation, behaviour, encoding, bootstrap CIs, held-out CV
#   --tier 3   ICLR strong    : dense trajectory (all 126 checkpoints)
#   --smoke    ~2 min sanity run, no brain/RSA. HARD GATE before any tier.
#
# Usage:  bash run_devai_bare.sh --tier 1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
. "$ROOT/env_brainalign.sh"
[ -f venv/bin/activate ] && . venv/bin/activate

TIER=""; SMOKE_MODE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --smoke) SMOKE_MODE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ "$SMOKE_MODE" = "1" ]; then
  exec env SMOKE=1 BACKUP=0 bash slurm/run_devai_grid.sh pico-decoder-tiny
fi

case "$TIER" in
  0) exec bash "$ROOT/prepare_brain_rdms.sh" ;;
  1) exec env ABLATE=0 MAX_CKPT=25 bash slurm/run_devai_grid.sh \
       pico-decoder-tiny pico-decoder-small pico-decoder-medium pico-decoder-large \
       beetle-humanscale-eng beetle-fineweb3-eng \
       babylm-gpt2-3 babylm-gpt2-5 babylm-gpt2-7 babylm-gpt2 ;;
  2) exec env ABLATE=1 BOOTSTRAP=1000 MAX_CKPT=25 bash slurm/run_devai_grid.sh \
       pico-decoder-small pico-decoder-large beetle-humanscale-eng beetle-fineweb3-eng ;;
  3) exec env ABLATE=1 MAX_CKPT=0 bash slurm/run_devai_grid.sh \
       pico-decoder-small beetle-fineweb3-eng ;;
  # Tier 3, second half -- cross-dataset generalisation (Fig 10) -- is NOT defined here.
  # The runbook's `DATASET=ds00XXXX` was a placeholder, never a real accession, and the
  # download path is hardcoded to ds003604 (scripts/batch_download_bold.py builds every
  # URL from OpenNeuroDatasets/ds003604, and contrasts are built from ds003604 stimuli).
  # Pointing DATASET= at another tag would re-download ds003604 into a directory named
  # after a different study and label it as a second dataset. That is fabricated data, so
  # this stage stays unimplemented until someone supplies the real accession AND a
  # download path that honours it. See PICKUP.md.
  *) echo "usage: $0 --tier {0,1,2,3} | --smoke" >&2; exit 2 ;;
esac
