#!/usr/bin/env bash
# Re-aggregate the per-cell tables the coverage check reads.
cd /local/scratch/sas245/brainalign-evals
for s in pipeline/scripts/corrected_sweep_summary.py pipeline/scripts/collect_ceilings.py; do
  [ -f "$s" ] && pipeline/venv/bin/python "$s" >/dev/null 2>&1
done
pipeline/venv/bin/python pipeline/scripts/coverage_matrix.py >/dev/null 2>&1
