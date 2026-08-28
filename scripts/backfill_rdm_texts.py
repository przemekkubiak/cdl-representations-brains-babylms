#!/usr/bin/env python
"""Backfill `stimulus_texts` into session RDMs that were written before it existed.

Session RDMs store `stimuli` as audio FILENAMES. scripts/run_devai_grid.py feeds that
list to the LM tokenizer, so without a parallel text array the brain-LM alignment is
computed over ".wav" strings. src/rsa/session_based_rsa.py now writes `stimulus_texts`,
but RDMs built before that (and any pulled from the Hub cache) lack it. Rather than
recompute hours of fMRI preprocessing, derive the text from the stimulus
characteristics table -- the RDM itself is unchanged, only annotated.

Idempotent: an RDM that already has a fully populated `stimulus_texts` is left alone.

    python scripts/backfill_rdm_texts.py --roots data/processed/fmri/ds003604 [--push]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.rsa.semantic_metadata import load_semantic_metadata  # noqa: E402

# ds003604's four. The other three datasets add Orth and SemLocal
# (configs/neuro_datasets.yaml `phenomena:`), and hardcoding this tuple silently
# skipped those directories -- so a dataset could come back "all ok" having left
# half its cells textless. Any directory that names a registered phenomenon is
# valid; the tuple stays as the fallback when the registry cannot be read.
TASKS = ("Sem", "Phon", "Gram", "Plaus")


def _known_phenomena() -> set:
    try:
        from src.contrast_spec import CONTRAST_SPECS
        return {p for spec in CONTRAST_SPECS.values() for p in spec}
    except Exception:
        return set(TASKS)


def backfill(path: Path, characteristics_dir: str) -> str:
    task = path.parent.name
    if task not in _known_phenomena() and task not in TASKS:
        return f"skip (unknown task dir '{task}')"
    d = np.load(path, allow_pickle=True)
    data = {k: d[k] for k in d.files}
    stimuli = [str(s) for s in data.get("stimuli", [])]
    if not stimuli:
        return "skip (no stimuli)"

    existing = data.get("stimulus_texts")
    if existing is not None and len(existing) == len(stimuli):
        if all(str(t).strip() for t in existing):
            return f"ok already ({len(stimuli)} stimuli)"

    meta = load_semantic_metadata(stimuli, task=task, characteristics_dir=characteristics_dir)
    texts = meta.get("stimulus_texts")
    if texts is None:
        return "FAILED (no characteristics table)"
    texts = [str(t).strip() for t in texts]
    missing = sum(1 for t in texts if not t)
    if missing:
        return f"FAILED ({missing}/{len(texts)} stimuli have no text)"

    data["stimulus_texts"] = np.asarray(texts, dtype=object)
    # np.savez_compressed appends ".npz" unless the name already ends with it.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(path)
    return f"backfilled {len(texts)} texts, e.g. {texts[0]!r}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["data/processed/fmri/ds003604"])
    # Age-group bins ("ses-11+") contain a regex-special character, so the glob
    # below is literal rather than patterned -- see main().
    ap.add_argument("--characteristics-dir",
                    default="data/brain/ds003604/stimuli/Stimulus_Characteristics")
    args = ap.parse_args()

    rc = 0
    for root in args.roots:
        for p in sorted(Path(root).glob("*/session_rdm_ses-*.npz")):
            msg = backfill(p, args.characteristics_dir)
            print(f"{p.parent.name:6s} {p.name:26s} {msg}")
            if msg.startswith("FAILED"):
                rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
