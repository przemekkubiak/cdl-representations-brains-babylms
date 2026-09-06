#!/usr/bin/env bash
# Reconcile every completed (dataset, ROI) stage with the Hub.
#
# run_new_datasets.sh publishes at stage 6, but on failure it logs
# "PUBLISH FAILED" and carries on -- no retry, and the sweep still reports the
# stage as done. So after the sweep we check what is actually on the Hub against
# what is on disk and push whatever is missing, rather than trusting that the
# publish step ran.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT"
PY="$ROOT/venv/bin/python"
L="$ROOT/logs/brain_hf_push.log"
say() { echo "[hfpush $(date -u +%FT%TZ)] $*" | tee -a "$L"; }

say "reconciling local results against the Hub"
"$PY" - <<'PYEOF' 2>&1 | tee -a "$L"
import json, pathlib, sys
import pandas as pd
from huggingface_hub import HfApi

# Results have been published under TWO conventions over this project's life:
#   (a) one repo per dataset, each ROI level nested under roi-<set>/  -- what
#       scripts/publish_dataset_results.py does today
#   (b) a separate repo per (dataset, ROI): cdl-devai-results-<ds>-roi<set>
#       -- the older scheme, which is where ds003604's three ROI results
#       actually live (215-222 files each)
# A variant counts as published if EITHER exists. Checking only one produced a
# false "not published" for six variants that were on the Hub the whole time,
# which would have meant re-pushing hundreds of files over good data.
ROOT = pathlib.Path("/local/scratch/sas245/brainalign-evals")
cov = ROOT / "results/coverage_matrix.csv"
if not cov.exists():
    print("no coverage matrix; run coverage_matrix.py first"); sys.exit(1)
d = pd.read_csv(cov)
api = HfApi()


def listing(repo):
    try:
        return api.list_repo_files(repo, repo_type="dataset")
    except Exception:
        return None


missing = []
for _, r in d[d.cells > 0].iterrows():
    roi = "" if r.variant == "within-run-normalised" else r.variant.replace("roi-", "")
    nested = listing(f"BrainAlign/brain-lm-alignment-{r.dataset}")
    tag = f"roi-{roi}/" if roi else None
    on_nested = bool(nested) and (
        any(f.startswith(tag) for f in nested) if tag
        else bool([f for f in nested if "/" not in f and f.endswith(".csv")]))
    sep = listing(f"BrainAlign/cdl-devai-results-{r.dataset}" + (f"-roi{roi}" if roi else ""))
    on_sep = bool(sep) and any(f.endswith(".csv") for f in sep)
    if on_nested or on_sep:
        where = "nested" if on_nested else "separate repo"
        print(f"  {r.dataset:9s} {r.variant:22s} cells={r.cells:3d}  on Hub ({where})")
    else:
        print(f"  {r.dataset:9s} {r.variant:22s} cells={r.cells:3d}  NOT on Hub -> will push")
        missing.append((r.dataset, roi))
json.dump([{"dataset": a, "roi": b} for a, b in missing],
          open(ROOT / "pipeline/logs/hf_missing.json", "w"), indent=2)
print(f"\n{len(missing)} publish(es) needed")
PYEOF

n=$("$PY" -c "import json;print(len(json.load(open('logs/hf_missing.json'))))" 2>/dev/null || echo 0)
if [ "$n" -eq 0 ]; then say "nothing missing; Hub is in sync"; exit 0; fi

"$PY" -c "
import json
for m in json.load(open('logs/hf_missing.json')): print(m['dataset'], m['roi'])
" | while read -r ds roi; do
  say "publishing $ds ${roi:-wholebrain}"
  if [ -n "$roi" ]; then
    "$PY" scripts/publish_dataset_results.py --dataset "$ds" --roi-set "$roi" \
      >>"logs/hfpush_${ds}_${roi}.log" 2>&1 && say "  ok $ds/$roi" || say "  FAILED $ds/$roi (see logs/hfpush_${ds}_${roi}.log)"
  else
    "$PY" scripts/publish_dataset_results.py --dataset "$ds" \
      >>"logs/hfpush_${ds}_wb.log" 2>&1 && say "  ok $ds/wholebrain" || say "  FAILED $ds/wholebrain"
  fi
done
say "HF RECONCILE COMPLETE"
