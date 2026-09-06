#!/usr/bin/env bash
# autonomous_supervisor.sh -- keep the neuro work running, and checkpoint it
# everywhere, every cycle, so an interruption cannot lose anything.
#
# WHY THIS EXISTS. Work has been lost in this project twice: a generator script
# deleted with no copy anywhere, and 67 GB of finished training runs sitting on
# disk referenced by no commit. Both were recoverable only by accident. This loop
# therefore treats "committed and pushed" as part of the job rather than as
# something done at the end: every cycle it commits whatever has appeared, pushes
# both git remotes, and reconciles the Hub, whether or not anything looks
# finished.
#
# It also RESTARTS a wave session that has died. The wave scripts are idempotent
# and their cohorts nest, so a restart resumes rather than redoes.
set -uo pipefail
ROOT="/local/scratch/sas245/brainalign-evals/pipeline"
REPO="/local/scratch/sas245/brainalign-evals"
cd "$ROOT"
export HOME=/local/scratch/sas245
export HF_HOME="$HOME/hf_cache"
export HF_DATASETS_CACHE="$HOME/hf_datasets_cache"
export TOKENIZERS_PARALLELISM=false
export GIT_SSH_COMMAND="ssh -i /local/scratch/sas245/sshkeys/id_ed25519 -o UserKnownHostsFile=/local/scratch/sas245/.ssh/known_hosts -o IdentitiesOnly=yes"
INTERVAL="${INTERVAL:-1800}"
L="$ROOT/logs/autonomous.log"
log() { echo "[auto $(date -u +%FT%TZ)] $*" | tee -a "$L"; }

commit_push() {   # commit_push <dir> <message>
  local d="$1" m="$2"
  git -C "$d" add -A >/dev/null 2>&1
  if ! git -C "$d" diff --cached --quiet 2>/dev/null; then
    git -C "$d" -c user.name=suchirsalhan -c user.email=suchirsalhan@gmail.com \
      commit -q -m "$m" >/dev/null 2>&1 && log "committed in $(basename "$d")"
  fi
  git -C "$d" push origin HEAD >/dev/null 2>&1 && log "pushed $(basename "$d")" \
    || log "push failed for $(basename "$d") (will retry next cycle)"
}

revive() {        # revive <session> <command>
  if ! tmux has-session -t "$1" 2>/dev/null; then
    log "session $1 is gone -- restarting (idempotent, cohorts nest)"
    tmux new-session -d -s "$1" "$2"
  fi
}

log "=== autonomous supervisor up; cycle ${INTERVAL}s"
while true; do
  # 1. keep the work alive
  revive par_ds002236 "bash $ROOT/run_brain_par.sh ds002236 0"
  revive par_ds006239 "bash $ROOT/run_brain_par.sh ds006239 1"
  revive par_ds001894 "bash $ROOT/run_brain_par.sh ds001894 2"

  # 2. refresh the derived tables so coverage reflects what is actually on disk
  bash "$REPO/refresh_results.sh" >/dev/null 2>&1

  # 3. reconcile the Hub -- publishes anything finished that the per-stage
  #    publish step dropped (it logs failure and continues without retry)
  bash "$ROOT/push_brain_to_hf.sh" >>"$ROOT/logs/auto_hf.log" 2>&1 \
    && log "hub reconciled" || log "hub reconcile had failures"

  # 4. checkpoint both repos
  TS="$(date -u +%FT%TZ)"
  commit_push "$ROOT" "Autonomous checkpoint $TS: neuro pipeline state

Scripts, logs and derived tables as of this cycle. Committed unconditionally so
that an interruption cannot lose work that exists only on disk."
  commit_push "$REPO" "Autonomous checkpoint $TS: neuro results

Coverage matrix and result tables as of this cycle."

  # 5. a one-line state summary, so the log itself is a record
  for d in ds002236 ds006239 ds001894; do
    log "  $d: $(tail -n 1 "$ROOT/logs/par_$d.log" 2>/dev/null | cut -c1-140)"
  done
  log "  alignment files: $(find "$ROOT/data/processed/language_models" -name 'alignment_*.csv' 2>/dev/null | wc -l) total"
  sleep "$INTERVAL"
done
