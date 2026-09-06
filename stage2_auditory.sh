#!/bin/bash
# Stage 2: publish masked RDMs and run the alignment sweep, ONCE PER WAVE.
#
# Two deliverables, matching the two build waves:
#   WAVE 1 (~5h): the 3 Phon cells -> sweep + package + push. This is the one
#     that has to land in the morning window, and it is a complete, publishable
#     masked-vs-unmasked comparison on the phonology task by itself.
#   WAVE 2 (~17h): Sem/Gram/Plaus land too -> re-run over all 12 cells and push
#     the full result, superseding wave 1's package.
#
# WHY A BRIDGE STEP EXISTS. prepare_brain_rdms.sh writes session RDMs under
#   pipeline/data/processed/fmri/<ds>/roi-auditory/<Task>/session_rdm_<ses>.npz
# but run_roi_sweeps.sh looks for them under
#   <root>/data/ds003604-session-rdms/<ds>/roi-auditory/<Task>/session_rdm_<ses>.npz
# (beside the unmasked within-run-normalised/ tree). Without the copy the sweep
# logs "SKIP -- no masked RDM tree found" and exits 0 -- nothing produced, but it
# reads as success. Copy, never move: the build tree is the source of truth that
# prepare_brain_rdms.sh's per-session skip reads to know what is already done.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT=/local/scratch/sas245/brainalign-evals
DS=ds003604
V=roi-auditory
SRC="data/processed/fmri/$DS/$V"
DST="$ROOT/data/$DS-session-rdms/$DS/$V"
export HF_HOME=/local/scratch/sas245/hf_cache_brainalign
export HF_TOKEN="$(cat /local/scratch/sas245/.cache/huggingface/token)"
say(){ echo "[stage2 $(date -u +%FT%TZ)] $*"; }

n_rdms(){ find "$SRC" -name 'session_rdm_*.npz' 2>/dev/null | wc -l; }
waves_running(){ pgrep -f run_masked_waves_auditory.sh >/dev/null 2>&1; }

sync_rdms(){
  local n=0 rel
  mkdir -p "$DST"
  while IFS= read -r f; do
    rel="${f#$SRC/}"
    mkdir -p "$DST/$(dirname "$rel")"
    if [ ! -f "$DST/$rel" ] || [ "$f" -nt "$DST/$rel" ]; then
      cp -p "$f" "$DST/$rel" && n=$((n+1))
    fi
  done < <(find "$SRC" -name 'session_rdm_*.npz' 2>/dev/null)
  say "synced: $(n_rdms) built, $n newly copied -> $DST"
}

run_sweep(){   # $1 = label
  sync_rdms
  if [ "$(find "$DST" -name 'session_rdm_*.npz' 2>/dev/null | wc -l)" -eq 0 ]; then
    say "$1: no masked RDMs present -- refusing to sweep (would publish an empty result)"
    return 3
  fi
  say "$1: sweep + package + push starting"
  GPU="${GPU:-1}" DATASETS="$DS" bash "$ROOT/scripts/run_roi_sweeps.sh" roi-auditory
  say "$1: sweep rc=$? ; cells now = $(find "$DST" -name 'session_rdm_*.npz' | wc -l)"
}

# ---- wave 1: fire as soon as the Phon cells are in (or the waves die) --------
say "watching $SRC (wave 1 target: 3 Phon session RDMs)"
LAST=-1
while true; do
  H=$(n_rdms)
  [ "$H" -ne "$LAST" ] && { say "session RDMs: $H"; sync_rdms; LAST=$H; }
  [ "$H" -ge 3 ] && break
  waves_running || { say "waves exited before wave 1 target (have $H)"; break; }
  sleep 120
done
run_sweep "wave1"

# ---- wave 2: everything else, then the full re-run ---------------------------
say "waiting for wave 2 (all 12 cells)"
LAST=-1
while waves_running; do
  H=$(n_rdms)
  [ "$H" -ne "$LAST" ] && { say "session RDMs: $H"; sync_rdms; LAST=$H; }
  sleep 300
done
say "waves finished; final built count = $(n_rdms)"
run_sweep "wave2"
say "STAGE 2 COMPLETE"

# Re-push the registration cache: it now also holds the roi-auditory masks and
# QC overlays. The affines themselves are unchanged (ROI-independent), so this
# only adds the new ROI-specific artefacts.
say "pushing updated registration cache (now includes roi-auditory masks)"
HF_TOKEN="$(cat /local/scratch/sas245/.cache/huggingface/token)" \
  python_bin=../venv/bin/python; ../venv/bin/python scripts/registration_cache_hf.py push --dataset ds003604
say "FINAL: all auditory artefacts pushed"
