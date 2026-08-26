#!/usr/bin/env python
"""Does removing the global component recover stimulus signal? Test it cheaply.

THE DIAGNOSIS TO TEST. The positive control found nothing stimulus-driven in the
ds003604 RDMs, and scripts/rdm_dimensionality.py located the reason: whole-brain
patterns whose leading component tracks overall signal level (|rho| = 0.85), and
RDMs that live in 4 of a possible 72 dimensions.

If that is right, projecting the global component out of the patterns should
raise the RDM's rank and let a stimulus property correlate. If it is wrong --
if the patterns simply contain no stimulus information -- nothing will change,
and the problem is upstream in preprocessing rather than in the voxel set.

This runs on the 249 pattern files that survive on disk, so it costs minutes
instead of the ~2 h tier-0 re-download, and it decides whether that re-download
is worth starting.

Two removals are compared, because they are not the same thing:
  * MEAN     -- subtract each pattern's mean across voxels (a scalar offset)
  * PC1      -- project out the leading component across stimuli (a spatial map)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TASKS = ["Sem", "Phon", "Gram", "Plaus"]


def load_run(f: Path, stride: int) -> tuple[list[str], np.ndarray] | None:
    try:
        z = np.load(f)
        keys = list(z.keys())
        if len(keys) < 5:
            return None
        x = np.vstack([z[k] for k in keys])[:, ::stride]
    except Exception:
        return None
    return [Path(k).name for k in keys], x


def rdm(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=1, keepdims=True)
    xc /= (np.linalg.norm(xc, axis=1, keepdims=True) + 1e-12)
    return 1.0 - xc @ xc.T


def remove_mean(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=1, keepdims=True)


def remove_pc1(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    return xc - np.outer(u[:, 0] * s[0], vt[0])


def effective_rank(m: np.ndarray, thresh: float = 0.9) -> int:
    n = m.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    s = np.linalg.svd(j @ m @ j, compute_uv=False)
    return 0 if not s.sum() else int(np.searchsorted(np.cumsum(s) / s.sum(), thresh) + 1)


def perm_rsa(a: np.ndarray, b: np.ndarray, n_perm: int, seed: int = 0) -> tuple[float, float]:
    iu = np.triu_indices_from(a, k=1)
    x = stats.rankdata(a[iu]); x = (x - x.mean()) / (x.std() + 1e-12)
    yr = np.zeros_like(b); yr[iu] = stats.rankdata(b[iu]); yr = yr + yr.T
    y = yr[iu]; y0 = (y - y.mean()) / (y.std() + 1e-12)
    rho = float(np.dot(x, y0) / len(x))
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(n)
        v = yr[np.ix_(p, p)][iu]
        v = (v - v.mean()) / (v.std() + 1e-12)
        null[k] = np.dot(x, v) / len(x)
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern-root", default="data/processed/fmri_wrn/ds003604")
    ap.add_argument("--stimuli", default="data/brain/ds003604/stimuli")
    ap.add_argument("--out", default="paper_results/control")
    ap.add_argument("--max-files", type=int, default=60)
    ap.add_argument("--voxel-stride", type=int, default=7)
    ap.add_argument("--perms", type=int, default=2000)
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.pattern_root)

    # Per-stimulus duration and the acoustic spectrum are the two controls with
    # the clearest prediction in an auditory design.
    norms = {}
    for task in TASKS:
        f = Path(a.stimuli) / "Stimulus_Characteristics" / f"task-{task}_Stimulus_Characteristics.tsv"
        if not f.exists():
            continue
        d = pd.read_csv(f, sep="\t").drop_duplicates("stim_file").set_index("stim_file")
        col = "total_stim_duration" if "total_stim_duration" in d.columns else "stim_duration"
        for s, v in pd.to_numeric(d[col], errors="coerce").items():
            if np.isfinite(v):
                norms[str(s)] = float(v)

    files = sorted(root.rglob("*_patterns.npz"))[: a.max_files]
    print(f"{len(files)} pattern files, voxel stride {a.voxel_stride}")

    rows = []
    for f in files:
        got = load_run(f, a.voxel_stride)
        if got is None:
            continue
        keys, x = got
        dur = np.array([norms.get(k, np.nan) for k in keys])
        if np.isnan(dur).any() or len(np.unique(dur)) < 5:
            continue
        d_dur = np.abs(dur[:, None] - dur[None, :])
        for label, xx in (("raw", x), ("mean_removed", remove_mean(x)),
                          ("pc1_removed", remove_pc1(x))):
            r = rdm(xx)
            rho, p = perm_rsa(r, d_dur, a.perms)
            amp = x.mean(axis=1)
            d_amp = np.abs(amp[:, None] - amp[None, :])
            iu = np.triu_indices_from(r, k=1)
            rows.append({
                "file": f.name, "task": f.parent.name, "variant": label,
                "n_stim": len(keys), "effective_rank": effective_rank(r),
                "rsa_vs_duration": rho, "p_perm": p,
                "rsa_vs_amplitude": float(stats.spearmanr(r[iu], d_amp[iu]).statistic),
            })

    if not rows:
        print("no usable pattern files")
        return
    d = pd.DataFrame(rows)
    d.to_csv(out / "global_signal_probe.csv", index=False)

    summ = (d.groupby("variant")
              .agg(n_runs=("file", "nunique"),
                   effective_rank=("effective_rank", "median"),
                   rsa_vs_amplitude=("rsa_vs_amplitude", "median"),
                   mean_abs_rsa_duration=("rsa_vs_duration", lambda s: float(np.mean(np.abs(s)))),
                   max_abs_rsa_duration=("rsa_vs_duration", lambda s: float(np.max(np.abs(s)))),
                   n_p_below_05=("p_perm", lambda s: int((s < 0.05).sum())),
                   n_tests=("p_perm", "size"))
              .reset_index())
    print()
    print("  --- PROBE: does removing the global component recover stimulus signal? ---")
    print(summ.to_string(index=False))

    raw = summ[summ["variant"] == "raw"].iloc[0]
    best = summ[summ["variant"] != "raw"].sort_values("max_abs_rsa_duration").iloc[-1]
    improved = (best["max_abs_rsa_duration"] > 1.5 * raw["max_abs_rsa_duration"]
                or best["n_p_below_05"] > 2 * max(raw["n_p_below_05"], 1))
    verdict = ("removing the global component recovers stimulus structure -- the "
               "diagnosis holds and the re-run is worth starting"
               if improved else
               "removing the global component changes nothing -- the patterns carry "
               "no stimulus information at all, so the problem is upstream in "
               "preprocessing, not in the voxel set or the global component")
    print(f"\n  VERDICT: {verdict}")

    (out / "global_signal_probe.json").write_text(json.dumps({
        "verdict": verdict, "improved": bool(improved),
        "summary": summ.to_dict(orient="records"),
        "n_permutations": a.perms, "voxel_stride": a.voxel_stride,
    }, indent=2))
    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
