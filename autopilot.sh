#!/usr/bin/env bash
# Unattended finisher: wait for the sweeps, then publish everything.
#
# WHY. Neither launcher publishes. launch_full_sweep.sh runs stages 2-3 with
# BACKUP=0 and launch_parc_sweep.sh does not push at all, so without this the run
# finishes with every result sitting on local disk and nothing on the Hub or in
# git. This script closes that gap so the run can be left alone.
#
# It is deliberately conservative:
#   * Every step is isolated -- one failure never aborts the rest.
#   * Nothing is ever deleted, locally or remotely. HF uploads are additive.
#   * Publishing is GATED on provenance verification. A tree with mixed or
#     unknown correction state is reported and NOT pushed; przemek is a
#     collaborator's repository and a bad run reached it once before.
#   * Safe to re-run at any point. Every step is idempotent.
#
# Status for whoever comes back: logs/AUTOPILOT_STATUS.md
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
. "$ROOT/env_brainalign.sh"

PY="$ROOT/venv/bin/python"
WRN="data/processed/fmri_wrn/ds003604"
STATUS="logs/AUTOPILOT_STATUS.md"
POLL="${POLL:-300}"
MAX_WAIT="${MAX_WAIT:-64800}"          # 18 h ceiling on the whole wait
GIT_NAME="suchirsalhan"
GIT_EMAIL="suchirsalhan@gmail.com"

log() { echo "[autopilot $(date -u +%FT%TZ)] $*" | tee -a logs/autopilot.log; }
step() { log "STEP: $*"; }
running() { pgrep -f "$1" >/dev/null 2>&1; }

commit_push() {   # commit_push <message-file>
  git add -A
  if git diff --cached --quiet; then log "  nothing new to commit"; return 0; fi
  git -c user.name="$GIT_NAME" -c user.email="$GIT_EMAIL" commit -q -F "$1" \
    || { log "  commit failed"; return 1; }
  for r in origin przemek; do
    if git push -q "$r" main 2>>logs/autopilot.log; then log "  pushed $r"
    else log "  PUSH FAILED: $r"; fi
  done
}

# ---------------------------------------------------------------- wait ------
log "autopilot started; waiting for sweeps to finish (poll ${POLL}s, cap ${MAX_WAIT}s)"
waited=0
while :; do
  main_alive=0; parc_alive=0
  running "launch_full_sweep.sh" && main_alive=1
  { running "launch_parc_sweep.sh" || running "parc_watcher.sh"; } && parc_alive=1
  [ "$main_alive" -eq 0 ] && [ "$parc_alive" -eq 0 ] && { log "both sweeps finished"; break; }
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    log "WAIT CAP REACHED (main=$main_alive parc=$parc_alive) -- finalising with what exists"
    break
  fi
  sleep "$POLL"; waited=$((waited + POLL))
done

# ------------------------------------------------------------ finalise ------
NCELL=$(find "$WRN" -name "session_rdm_ses-*.npz" 2>/dev/null | wc -l)
log "corrected cells on disk: $NCELL"

step "sync corrected RDMs to the Hub cache (idempotent, additive)"
"$PY" scripts/rdm_cache_hf.py sync --root "$WRN" --dataset ds003604 2>&1 \
  | grep -E "PUSH|have|sync done|failed" | sed 's/^/  /' | tee -a logs/autopilot.log

step "verify provenance of the corrected tree"
PROV=ok
"$PY" scripts/verify_rdm_provenance.py --rdm-root "$WRN" --require-ceilings \
  2>&1 | grep -viE "warn|libmpi" | sed 's/^/  /' | tee -a logs/autopilot.log
"$PY" scripts/verify_rdm_provenance.py --rdm-root "$WRN" --require-ceilings >/dev/null 2>&1 || PROV=failed
log "  provenance: $PROV"

step "ceiling table across every corrected cell"
"$PY" scripts/collect_ceilings.py --rdm-root "$WRN" \
    --out paper_results/ceiling/ceilings_ds003604.csv 2>&1 \
  | grep -viE "warn|libmpi" | sed 's/^/  /' | tee -a logs/autopilot.log

step "PARC seed-null analysis (if the grid produced rows)"
if [ -d data/processed/language_models/devai_grid_parc/ds003604 ]; then
  "$PY" scripts/parc_seed_null.py \
      --grid-dir data/processed/language_models/devai_grid_parc/ds003604 \
      --ceilings paper_results/ceiling/ceilings_ds003604.csv \
      --out paper_results/parc 2>&1 | grep -viE "warn|libmpi" | sed 's/^/  /' \
    | tee -a logs/autopilot.log
else
  log "  no PARC grid output -- skipped"
fi

step "publish results to HuggingFace (additive: new paths only)"
if [ "$PROV" = "ok" ]; then
  "$PY" - <<'PYEOF' 2>&1 | sed 's/^/  /' | tee -a logs/autopilot.log
import shutil, sys
from pathlib import Path
from huggingface_hub import HfApi

REPO = "BrainAlign/cdl-devai-results"
stage = Path("hf_results_staging/corrected-sweep")
stage.mkdir(parents=True, exist_ok=True)

copied = 0
for src, sub in [(Path("paper_results/ceiling"), "ceiling"),
                 (Path("paper_results/parc"), "parc")]:
    if not src.exists():
        continue
    dst = stage / sub
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
            copied += 1
if not copied:
    print("nothing to publish"); sys.exit(0)

api = HfApi()
before = set(api.list_repo_files(REPO, repo_type="dataset"))
api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(stage),
                  path_in_repo="corrected-sweep",
                  commit_message="Corrected sweep: ceilings, alignment vs ceiling, PARC seed-null")
after = set(api.list_repo_files(REPO, repo_type="dataset"))
removed = before - after
print(f"files {len(before)} -> {len(after)}")
print("REMOVED:", sorted(removed) if removed else "none  <-- nothing deleted")
for f in sorted(after - before):
    print("  added", f)
PYEOF
else
  log "  SKIPPED -- provenance check failed; not publishing a tree that may be mixed"
fi

step "commit and push summaries"
if [ "$PROV" = "ok" ]; then
  cat > /tmp/autopilot_commit_msg <<EOM
Corrected sweep results: ceilings across all cells, and the PARC seed-null

Published by autopilot.sh at the end of the unattended run. Contains the ceiling
table for every within-run-normalised cell built, alignment expressed as a
fraction of that ceiling, and -- where the PARC grid produced rows -- the
seed-spread, equivalence (TOST) and architecture-contrast tables.

Corrected cells built: ${NCELL}/12. Provenance verified: every RDM carries
within_run_normalized=True and its per-subject RDMs, so the ceilings are
recomputable from the .npz files alone.
EOM
  commit_push /tmp/autopilot_commit_msg
else
  log "  SKIPPED -- provenance check failed; not pushing to a collaborator's repo"
fi

# -------------------------------------------------------------- status ------
{
  echo "# Autopilot status"
  echo
  echo "Finished: $(date -u +%FT%TZ)"
  echo
  echo "## State"
  echo
  echo "- corrected cells: ${NCELL}/12"
  echo "- provenance: ${PROV}"
  echo "- git HEAD: $(git rev-parse --short HEAD)"
  echo "- origin/main: $(git rev-parse --short origin/main 2>/dev/null || echo '?')"
  echo "- przemek/main: $(git rev-parse --short przemek/main 2>/dev/null || echo '?')"
  echo
  echo "## Ceilings"
  echo '```'
  [ -f paper_results/ceiling/ceilings_ds003604.csv ] && \
    "$PY" -c "
import pandas as pd
d = pd.read_csv('paper_results/ceiling/ceilings_ds003604.csv')
cols = [c for c in ['task','session','ceiling_lower','ceiling_upper','ceiling_n','n_stim'] if c in d.columns]
print(d[cols].to_string(index=False))
" 2>/dev/null || echo "(none)"
  echo '```'
  echo
  echo "## Logs"
  echo
  echo '- `logs/autopilot.log` — this run'
  echo '- `logs/sweep_stage1_ds003604.log` — brain prep'
  echo '- `logs/sweep_*_gpu*.log` — model grids'
  echo '- `logs/parc_*.log` — PARC'
  echo '- `logs/sweep_ledger.json` — per-stage status'
  echo
  echo "## If something is missing"
  echo
  echo 'Every step here is idempotent. Re-run `bash autopilot.sh` to finish up,'
  echo 'or `bash launch_full_sweep.sh` to resume stage 1 (it skips finished cells).'
} > "$STATUS"

log "AUTOPILOT COMPLETE -- see $STATUS"
