#!/usr/bin/env python
"""Noise ceiling for every session RDM in a tree, straight from the saved files.

Needs no pattern files: RDMs built after 2026-08-25 carry their per-subject RDMs,
so the ceiling is recomputable from the .npz alone. Before that change the only
way to get a ceiling was to re-download and re-preprocess the dataset, which is
why the published results have none.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.rsa.ceiling_core import noise_ceiling_from_subject_rdms  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdm-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = []
    for f in sorted(Path(a.rdm_root).rglob("session_rdm_ses-*.npz")):
        task = f.parent.name
        m = re.search(r"session_rdm_(ses-[^.]+)\.npz", f.name)
        session = m.group(1) if m else "?"
        d = np.load(f, allow_pickle=True)

        row = {
            "task": task, "session": session, "file": str(f),
            "within_run_normalized": bool(d["within_run_normalized"])
            if "within_run_normalized" in d.files else None,
            "n_stim": int(d["rdm"].shape[0]) if "rdm" in d.files else None,
            "n_subjects_rdm": int(d["n_subjects"]) if "n_subjects" in d.files else None,
        }
        if "subject_rdms" in d.files and len(d["subject_rdms"]):
            c = noise_ceiling_from_subject_rdms(d["subject_rdms"])
            if c:
                row.update({
                    "ceiling_lower": c["lower"], "ceiling_upper": c["upper"],
                    "ceiling_lower_sem": c["lower_sem"], "ceiling_n": c["n_subjects"],
                })
        else:
            row["note"] = "no subject_rdms -- built before ceilings were retained"
        rows.append(row)

    if not rows:
        print(f"no session RDMs under {a.rdm_root}")
        return

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    have = df["ceiling_lower"].notna() if "ceiling_lower" in df else pd.Series(False, index=df.index)
    print(f"{len(df)} session RDMs, {int(have.sum())} with a ceiling -> {a.out}")
    if have.any():
        for _, r in df[have].iterrows():
            print(f"  {r['task']:8s} {r['session']:8s} "
                  f"ceiling {r['ceiling_lower']:.3f}-{r['ceiling_upper']:.3f} "
                  f"(n={int(r['ceiling_n'])}, {int(r['n_stim'])} stimuli)")


if __name__ == "__main__":
    main()
