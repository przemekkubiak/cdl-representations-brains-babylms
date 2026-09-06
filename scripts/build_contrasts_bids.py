#!/usr/bin/env python
"""Build contrasts/<task>.csv for datasets whose conditions live in BIDS events.

WHY THIS EXISTS. scripts/build_contrasts.py builds the four ds003604 contrasts
from a GitHub stimulus TSV keyed by ds003604's own task names, so it cannot
produce Orth or SemLocal. Without those CSVs run_devai_grid.py raises
FileNotFoundError for every model family, prints "grid done -- 0 alignment
files" and exits 0 -- a silent total failure that has already been mistaken for
an empty result once in this project.

The datasets it covers use CONTRAST_SPECS[<dataset>][<task>] with
kind='stim_pair_filename': the condition is the events.tsv `trial_type`, and the
stimulus text is the filename stem of the two stimulus columns. Stems carry
per-dataset prefixes -- 'T3_fall.bmp' (ds006239 reading tasks), 'Sem/train.bmp'
(ds006239 LocalSem), 'fall.bmp' (ds001894) -- and perceptual/control trials use
underscore-prefixed non-words like '_F3_f4.bmp', which the spec already excludes
by listing them under `perceptual` rather than positive/negative.

Output matches the existing files exactly: two columns, positive and negative,
one space-separated word pair per row.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contrast_spec import CONTRAST_SPECS

STIM_COLS = [("prime_stim", "targ_stim"), ("A_stim", "B_stim")]


def word_of(raw: str) -> str | None:
    """'T3_fall.bmp' -> 'fall'; 'Sem/train.bmp' -> 'train'; '_F3_f4.bmp' -> None."""
    if not raw:
        return None
    stem = Path(raw.strip()).stem                 # drop directory and extension
    if stem.startswith("_"):                      # perceptual/control non-word
        return None
    stem = re.sub(r"^[A-Z]\d+_", "", stem)        # drop a 'T3_' style prefix
    stem = stem.strip().lower()
    return stem if re.fullmatch(r"[a-z][a-z'-]*", stem) else None


def pairs_for(events: list[Path], spec: dict) -> tuple[list[str], list[str]]:
    pos_codes = {str(c) for c in spec.get("positive", [])}
    neg_codes = {str(c) for c in spec.get("negative", [])}
    pos, neg, seen = [], [], set()
    for f in events:
        try:
            rows = list(csv.DictReader(f.open(), delimiter="\t"))
        except Exception:
            continue
        if not rows:
            continue
        cols = next(((a, b) for a, b in STIM_COLS if a in rows[0] and b in rows[0]), None)
        if cols is None:
            continue
        a_col, b_col = cols
        for r in rows:
            code = str(r.get("trial_type", "")).strip()
            if code not in pos_codes and code not in neg_codes:
                continue
            w1, w2 = word_of(r.get(a_col, "")), word_of(r.get(b_col, ""))
            if not w1 or not w2 or w1 == w2:
                continue
            pair = f"{w1} {w2}"
            key = (code in pos_codes, pair)
            if key in seen:
                continue
            seen.add(key)
            (pos if code in pos_codes else neg).append(pair)
    return pos, neg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--brain-dir", default=None)
    ap.add_argument("--out-dir", default="contrasts")
    a = ap.parse_args()

    brain = Path(a.brain_dir or f"data/brain/{a.dataset}")
    specs = CONTRAST_SPECS.get(a.dataset, {})
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rc = 0
    for task in a.tasks:
        spec = specs.get(task)
        if spec is None:
            print(f"{task}: no spec in CONTRAST_SPECS[{a.dataset}]"); rc = 1; continue
        # the BIDS task label can differ from the phenomenon name
        bids = spec.get("task", task)
        events = sorted(brain.rglob(f"*task-{bids}*_events.tsv"))
        if not events:
            print(f"{task}: no events.tsv for BIDS task '{bids}' under {brain}")
            rc = 1; continue
        pos, neg = pairs_for(events, spec)
        n = min(len(pos), len(neg))
        if n == 0:
            print(f"{task}: 0 usable pairs from {len(events)} events files "
                  f"(pos={len(pos)} neg={len(neg)})"); rc = 1; continue
        # equal-length columns, as in the existing files
        with (out / f"{task}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["positive", "negative"])
            for p, q in zip(pos[:n], neg[:n]):
                w.writerow([p, q])
        print(f"{task}: wrote {n} pairs from {len(events)} events files "
              f"(BIDS task '{bids}')")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
