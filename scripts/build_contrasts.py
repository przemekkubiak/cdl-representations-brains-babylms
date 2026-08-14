#!/usr/bin/env python
"""Build LM localizer contrasts from the real ds003604 stimulus characteristics.

Each phenomenon becomes a `condition > control` text contrast — the LM analogue
of the fMRI GLM localizer contrast — reconstructed from the actual auditory
stimuli in the neurodataset (word_A/word_B for the word tasks; carrier +
subject + verb(s) + number + object for the sentence tasks).

Condition mapping (from the task JSON sidecars):
  Sem   S_H high-assoc / S_L low-assoc / S_U unrelated / S_C perceptual-control
        -> positive = S_H (semantically related), negative = S_U (unrelated words)
  Phon  P_R rhyme / P_O onset / P_U unrelated / P_C perceptual-control
        -> positive = P_R + P_O (phonologically related), negative = P_U
  Gram  G_G grammatical / G_F finiteness-viol / G_P plurality-viol / G_C control
        -> positive = G_G (grammatical), negative = G_F + G_P (ungrammatical)
  Plaus SP_S strongly / SP_W weakly congruent / SP_I incongruent / SP_C control
        -> positive = SP_S + SP_W (plausible), negative = SP_I (implausible)

Source of the TSVs:
  --source github  (default) pulls from github.com/suchirsalhan/neurodataset_babylm
  --source <dir>            reads a local ds003604 Stimulus_Characteristics dir

Output: <out-dir>/{Sem,Phon,Gram,Plaus}.csv with `positive,negative` columns,
consumed by scripts/run_circuit_localization.py --contrast-dir <out-dir>.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.contrast_spec import CONTRAST_SPEC, condition_of, reconstruct_text

GITHUB_RAW = (
    "https://raw.githubusercontent.com/suchirsalhan/neurodataset_babylm/main/"
    "stimuli/Stimulus_Characteristics/task-{task}_Stimulus_Characteristics.tsv"
)


def _read_tsv(task: str, source: str) -> list[dict]:
    if source == "github":
        url = GITHUB_RAW.format(task=task)
        with urllib.request.urlopen(url) as r:  # noqa: S310 - trusted repo
            text = r.read().decode("utf-8")
    else:
        path = Path(source) / f"task-{task}_Stimulus_Characteristics.tsv"
        text = path.read_text(encoding="utf-8")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def build_task(task: str, source: str) -> tuple[list[str], list[str]]:
    spec = CONTRAST_SPEC[task]
    rows = _read_tsv(task, source)
    pos, neg = [], []
    for row in rows:
        text = reconstruct_text(row, spec["kind"])
        if not text:
            continue
        cond = condition_of(row.get("trial_type", ""), task)
        if cond == "positive":
            pos.append(text)
        elif cond == "negative":
            neg.append(text)
    return pos, neg


def write_contrast(task: str, pos: list[str], neg: list[str], out_dir: Path) -> Path:
    # balance to equal length (mirrors LangLocDataset / PhenomenonContrast)
    n = min(len(pos), len(neg))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{task}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["positive", "negative"])
        for p, q in zip(pos[:n], neg[:n]):
            w.writerow([p, q])
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="github",
                    help="'github' or a local Stimulus_Characteristics dir")
    ap.add_argument("--out-dir", default="contrasts", help="Output dir for {task}.csv")
    ap.add_argument("--tasks", nargs="+", default=list(CONTRAST_SPEC.keys()))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for task in args.tasks:
        pos, neg = build_task(task, args.source)
        path = write_contrast(task, pos, neg, out_dir)
        n = min(len(pos), len(neg))
        print(f"{task:6s} positive={len(pos):3d} negative={len(neg):3d} -> {n} pairs  {path}")


if __name__ == "__main__":
    main()
