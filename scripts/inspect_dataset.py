#!/usr/bin/env python
"""Inspect a BIDS neuro dataset before committing to it.

Reports, from the dataset's own files rather than from its paper:

  1. subjects / sessions / tasks / runs actually present
  2. whether PER-SUBJECT age is recoverable (the developmental axis needs it)
  3. the observed `trial_type` codes per task -- transcribe these into
     CONTRAST_SPECS in src/contrast_spec.py; do not guess them
  4. RUN/STIMULUS CROSSING -- the decisive one

On (4): ds003604 presents each stimulus in exactly one scanner run, so run
identity is perfectly confounded with stimulus identity, and RDMs built by
pooling runs measure scanner drift rather than language. That confound is why
the alignment results in hf_results_staging/ are uninterpretable. Any new
dataset with the same nesting inherits the same problem, so this check runs
BEFORE any preprocessing rather than after.

Only text files are needed (events.tsv, participants.tsv), so this works on a
metadata-only checkout -- no BOLD download required.

Usage:
    python scripts/inspect_dataset.py --dataset ds001894 --bootstrap
    python scripts/inspect_dataset.py --dataset ds003604            # validate
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import get_dataset, UnresolvedDatasetError

ENTITY = re.compile(r"(?:^|_)(sub|ses|task|run|acq)-([A-Za-z0-9]+)")


def entities(path: Path) -> dict[str, str]:
    return {k: v for k, v in ENTITY.findall(path.name)}


def bootstrap(spec, data_dir: Path) -> None:
    """Clone the metadata-only checkout (no annex content)."""
    if data_dir.exists() and any(data_dir.glob("sub-*")):
        print(f"  checkout already present: {data_dir}")
        return
    url = spec.git_url()
    print(f"  cloning metadata from {url}")
    data_dir.parent.mkdir(parents=True, exist_ok=True)
    env = {"GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(data_dir)],
        check=True,
        env={**dict(__import__("os").environ), **env},
    )


# Columns that identify WHICH stimulus a trial presented, in preference order.
# A trial may present a pair (ds001894 shows two words, A_stim + B_stim), in
# which case the pair jointly identifies the stimulus.
STIM_COLUMN_SETS = [
    ["stim_file"],
    ["A_stim", "B_stim"],
    ["stim_file_A", "stim_file_B"],
    ["prime_stim", "targ_stim"],
]


def load_levels(data_dir: Path, task: str) -> dict[str, str]:
    """trial_type code -> human label, from the BIDS events.json sidecar.

    Datasets commonly code trial_type numerically (ds001894 uses 1..6), which is
    unreadable without the sidecar. Reading it here is what makes the reported
    codes safe to transcribe into CONTRAST_SPECS.
    """
    for cand in (data_dir / f"task-{task}_events.json",):
        if cand.exists():
            try:
                meta = json.loads(cand.read_text())
            except Exception:
                return {}
            levels = (meta.get("trial_type") or {}).get("Levels") or {}
            return {str(k): str(v) for k, v in levels.items()}
    return {}


def stim_columns(row: dict) -> list[str] | None:
    for cols in STIM_COLUMN_SETS:
        if all(c in row for c in cols):
            return cols
    return None


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def report_participants(data_dir: Path) -> None:
    print("\n2. PER-SUBJECT AGE")
    pfile = data_dir / "participants.tsv"
    if not pfile.exists():
        print("   participants.tsv MISSING -- no per-subject metadata at all")
        return
    rows = read_tsv(pfile)
    cols = list(rows[0].keys()) if rows else []
    print(f"   participants.tsv: {len(rows)} rows, columns: {', '.join(cols)}")

    age_cols = [c for c in cols if "age" in c.lower()]
    birth_cols = [c for c in cols if "birth" in c.lower() or "dob" in c.lower()]
    # A birthdate is not a scan date; subtracting it from itself yields nothing.
    scan_date_cols = [
        c for c in cols
        if ("date" in c.lower() or "acq_time" in c.lower()) and c not in birth_cols
    ]

    if age_cols:
        print(f"   -> AGE AT SCAN available directly: {len(age_cols)} column(s)")
        print(f"      e.g. {age_cols[:4]}")
    elif birth_cols and scan_date_cols:
        print(f"   -> age DERIVABLE from {birth_cols} minus {scan_date_cols}")
        print("      (dates may be shifted for anonymity; differences are preserved)")
    elif birth_cols:
        print(f"   -> birthdate present ({birth_cols}) but NO scan-date column, so age")
        print("      at scan is NOT derivable from participants.tsv alone. Look for")
        print("      *_scans.tsv acq_time, or treat this dataset as cohort-level only.")
        scans = list(data_dir.rglob("*_scans.tsv"))
        print(f"      *_scans.tsv files present: {len(scans)}")
    else:
        print("   -> NO per-subject age found. The developmental axis needs")
        print("      per-subject age, not a cohort range -- check the paper's")
        print("      supplementary data before relying on this dataset.")


def report_structure(data_dir: Path) -> tuple[list[Path], dict]:
    print("\n1. STRUCTURE")
    events = sorted(data_dir.rglob("*_events.tsv"))
    if not events:
        print("   NO events.tsv found -- cannot derive contrasts from this checkout")
        return [], {}

    subs, sess, tasks, runs = set(), set(), set(), set()
    for f in events:
        e = entities(f)
        subs.add(e.get("sub"))
        if "ses" in e:
            sess.add(f"ses-{e['ses']}")
        tasks.add(e.get("task"))
        if "run" in e:
            runs.add(f"run-{e['run']}")

    print(f"   subjects: {len(subs)}")
    print(f"   sessions: {sorted(s for s in sess if s) or '(none -- single session)'}")
    print(f"   tasks:    {sorted(t for t in tasks if t)}")
    print(f"   runs:     {sorted(runs) or '(no run entity)'}")
    print(f"   events.tsv files: {len(events)}")
    return events, {"subjects": subs, "sessions": sess, "tasks": tasks, "runs": runs}


def report_trial_types(events: list[Path], data_dir: Path) -> None:
    print("\n3. TRIAL TYPES PER TASK  (transcribe into CONTRAST_SPECS -- do not guess)")
    by_task: dict[str, Counter] = defaultdict(Counter)
    for f in events:
        task = entities(f).get("task")
        if not task:
            continue
        try:
            for row in read_tsv(f):
                tt = (row.get("trial_type") or "").strip()
                if tt and tt != "n/a":
                    by_task[task][tt] += 1
        except Exception:
            continue

    for task in sorted(by_task):
        counts = by_task[task]
        total = sum(counts.values())
        levels = load_levels(data_dir, task)
        suffix = "" if levels else "   [no events.json Levels -- codes are opaque]"
        print(f"   {task}  ({total} trials, {len(counts)} codes){suffix}")
        for code, n in counts.most_common():
            label = levels.get(code, "")
            print(f"       {code:12s} {n:7d}  ({100 * n / total:5.1f}%)  {label}")


def report_run_crossing(events: list[Path]) -> str:
    """The decisive check: does each stimulus appear in more than one run?"""
    print("\n4. RUN / STIMULUS CROSSING  <-- decides whether RDMs are usable")

    # stimulus identity per task: which runs does each distinct stimulus occur in?
    per_task_stim_runs: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    id_col_used: dict[str, str] = {}
    unidentifiable: set[str] = set()

    for f in events:
        e = entities(f)
        task, run = e.get("task"), e.get("run")
        if not task or not run:
            continue
        try:
            rows = read_tsv(f)
        except Exception:
            continue
        if not rows:
            continue
        cols = stim_columns(rows[0])
        if cols is None:
            # No stimulus-identity column. trial_type is a CONDITION label, not a
            # stimulus, so using it here would report "crossed" for any dataset
            # whose conditions recur across runs -- i.e. all of them. Refuse.
            unidentifiable.add(task)
            continue
        id_col_used[task] = "+".join(cols)
        for row in rows:
            parts = [Path((row.get(c) or "").strip()).name for c in cols]
            if not all(parts) or any(p == "n/a" for p in parts):
                continue
            per_task_stim_runs[task]["|".join(parts)].add(f"run-{run}")

    if unidentifiable:
        print(f"   CANNOT ASSESS for: {sorted(unidentifiable)}")
        print("     no stimulus-identity column (looked for "
              f"{', '.join('+'.join(c) for c in STIM_COLUMN_SETS)}).")
        print("     trial_type is a condition label, not a stimulus -- using it would")
        print("     report CROSSED for every dataset. Resolve the stimulus identity")
        print("     from the dataset's stimuli/ directory before building RDMs.")

    if not per_task_stim_runs:
        print("   -> verdict: UNKNOWN (no assessable task)")
        return "unknown"

    verdicts = []
    for task in sorted(per_task_stim_runs):
        stim_runs = per_task_stim_runs[task]
        n_stim = len(stim_runs)
        single = sum(1 for r in stim_runs.values() if len(r) == 1)
        frac_single = single / n_stim if n_stim else 0.0
        verdict = "nested" if frac_single > 0.95 else ("crossed" if frac_single < 0.5 else "partial")
        verdicts.append(verdict)
        print(
            f"   {task:8s} id={id_col_used[task]:10s} {n_stim:5d} stimuli, "
            f"{single} ({100 * frac_single:.0f}%) appear in exactly ONE run  -> {verdict.upper()}"
        )

    overall = "nested" if all(v == "nested" for v in verdicts) else (
        "crossed" if all(v == "crossed" for v in verdicts) else "mixed"
    )
    print()
    if overall == "nested":
        print("   NESTED. Run identity is confounded with stimulus identity, exactly as in")
        print("   ds003604. Voxel patterns MUST be z-scored within run before pooling, or")
        print("   the RDMs will measure scanner drift. Record run_stimulus: nested in")
        print("   configs/neuro_datasets.yaml.")
    elif overall == "crossed":
        print("   CROSSED. Stimuli recur across runs, so run and stimulus are separable.")
        print("   This dataset does NOT inherit the ds003604 confound -- which makes it")
        print("   the more valuable one to run. Record run_stimulus: crossed.")
    else:
        print(f"   {overall.upper()} -- inspect per task before pooling.")
    return overall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="registry key or accession")
    ap.add_argument("--data-dir", default=None, help="override checkout location")
    ap.add_argument("--bootstrap", action="store_true", help="clone the metadata checkout if absent")
    args = ap.parse_args()

    try:
        spec = get_dataset(args.dataset)
    except KeyError as e:
        print(e, file=sys.stderr)
        return 2

    print("=" * 72)
    print(f"{spec.key}  --  {spec.name}")
    print(f"status in registry: {spec.status}   declared run/stimulus: {spec.run_stimulus}")
    print("=" * 72)

    try:
        data_dir = Path(args.data_dir) if args.data_dir else spec.data_dir()
    except UnresolvedDatasetError as e:
        print(f"\nCANNOT INSPECT: {e}", file=sys.stderr)
        return 2

    if args.bootstrap:
        try:
            bootstrap(spec, data_dir)
        except subprocess.CalledProcessError as e:
            print(f"\nclone failed: {e}", file=sys.stderr)
            return 1

    if not data_dir.exists():
        print(f"\nno checkout at {data_dir} -- re-run with --bootstrap", file=sys.stderr)
        return 1

    events, _ = report_structure(data_dir)
    report_participants(data_dir)
    if events:
        report_trial_types(events, data_dir)
        report_run_crossing(events)

    print("\n" + "=" * 72)
    print("Next: transcribe the trial_type codes into CONTRAST_SPECS in")
    print("src/contrast_spec.py, and record tasks/sessions/run_stimulus in")
    print("configs/neuro_datasets.yaml. Then the dataset is runnable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
