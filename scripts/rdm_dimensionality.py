#!/usr/bin/env python
"""Why do reliable brain RDMs carry no stimulus information? Measure their rank.

CONTEXT. The positive control (scripts/positive_control.py) returned nothing:
across 108 stimulus x cell tests on the corrected ds003604 RDMs, no stimulus
property -- not duration, not the acoustic spectrum, not word length, frequency,
phonemes, syllables or the design's own condition contrast -- correlates with the
brain RDM, even though the same permutation test recovers scanner-run identity at
rho = 0.87 on the uncorrected RDMs and the inter-subject noise ceiling is 0.85.

Reliable, and empty. That combination has one obvious explanation: the RDMs may
be reliable about something that is not stimulus-specific. This script measures
it directly, at three levels:

  1. GROUP AND SUBJECT RDM RANK -- how many dimensions does the representational
     geometry actually have? A 72-stimulus RDM that lives in 4 dimensions cannot
     express stimulus-level structure no matter what it is compared against.
  2. RAW PATTERN RANK -- the stimulus x voxel matrix each RDM is computed from.
  3. WHAT THE LEADING COMPONENT IS -- specifically, whether it is the global mean
     signal of the volume rather than any spatial pattern.

If the leading component is global amplitude, every alignment result computed on
these RDMs -- ours and the null -- is a measurement of scanner brightness, and
the pipeline needs a brain mask and per-pattern denoising before any alignment
claim can be made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

TASKS = ["Sem", "Phon", "Gram", "Plaus"]


def effective_rank(m: np.ndarray, thresh: float = 0.9) -> int:
    """Dimensions needed to reach `thresh` of the double-centred spectrum.

    Double-centring first, because an RDM's overall offset is not structure.
    """
    n = m.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    s = np.linalg.svd(j @ m @ j, compute_uv=False)
    if not s.sum():
        return 0
    return int(np.searchsorted(np.cumsum(s) / s.sum(), thresh) + 1)


def rdm_report(rdm_root: Path, sessions: list[str]) -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for session in sessions:
            f = rdm_root / task / f"session_rdm_{session}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            r = np.asarray(z["rdm"], dtype=float)
            subj = np.asarray(z["subject_rdms"], dtype=float) if "subject_rdms" in z else None
            sranks = [effective_rank(m) for m in subj[:20]] if subj is not None else []
            rows.append({
                "task": task, "session": session, "n_stim": r.shape[0],
                "group_rdm_effective_rank": effective_rank(r),
                "subject_rdm_rank_median": int(np.median(sranks)) if sranks else np.nan,
                "subject_rdm_rank_min": int(np.min(sranks)) if sranks else np.nan,
                "subject_rdm_rank_max": int(np.max(sranks)) if sranks else np.nan,
                "ceiling": float(z["noise_ceiling_lower"]) if "noise_ceiling_lower" in z else np.nan,
            })
    return pd.DataFrame(rows)


def pattern_report(rdm_root: Path, n_files: int, stride: int) -> pd.DataFrame:
    """Open raw pattern files and ask what their leading component is."""
    files = sorted(rdm_root.rglob("*_patterns.npz"))[:n_files]
    rows = []
    for f in files:
        try:
            z = np.load(f)
            keys = list(z.keys())
            if len(keys) < 5:
                continue
            x = np.vstack([z[k] for k in keys])
        except Exception:
            continue
        n_vox = x.shape[1]
        nonzero = float((np.abs(x).sum(axis=0) > 0).mean())
        sub = x[:, ::stride]
        xc = sub - sub.mean(axis=0)
        s = np.linalg.svd(xc, compute_uv=False)
        rank = int(np.searchsorted(np.cumsum(s) / s.sum(), 0.9) + 1)
        u, _, vt = np.linalg.svd(xc, full_matrices=False)

        amp = sub.mean(axis=1)                       # global signal per stimulus
        pc1_vs_amp = abs(stats.spearmanr(u[:, 0], amp).statistic)
        pc1_vs_meanmap = abs(np.corrcoef(vt[0], sub.mean(axis=0))[0, 1])

        # how much of the RDM is just "how bright was the volume"
        xn = sub - sub.mean(axis=1, keepdims=True)
        xn = xn / (np.linalg.norm(xn, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - xn @ xn.T
        d_amp = np.abs(amp[:, None] - amp[None, :])
        iu = np.triu_indices_from(d, k=1)
        rows.append({
            "file": f.name, "n_stimuli": len(keys), "n_voxels": n_vox,
            "frac_voxels_nonzero": nonzero,
            "pattern_effective_rank": rank,
            "pc1_vs_global_signal": float(pc1_vs_amp),
            "pc1_vs_mean_map": float(pc1_vs_meanmap),
            "rdm_vs_amplitude_rsa": float(stats.spearmanr(d[iu], d_amp[iu]).statistic),
            "top2_variance_share": float(s[:2].sum() / s.sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdm-root", default="data/processed/fmri_wrn/ds003604")
    ap.add_argument("--sessions", default="ses-5,ses-7,ses-9")
    ap.add_argument("--out", default="paper_results/control")
    ap.add_argument("--pattern-files", type=int, default=12,
                    help="how many raw pattern files to open (they are large)")
    ap.add_argument("--voxel-stride", type=int, default=7,
                    help="subsample voxels; the spectrum is stable under this")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    root = Path(a.rdm_root)
    sessions = [s.strip() for s in a.sessions.split(",") if s.strip()]

    r = rdm_report(root, sessions)
    if len(r):
        r.to_csv(out / "rdm_dimensionality.csv", index=False)
        print("  --- RDM DIMENSIONALITY (dims to reach 90% of the spectrum) ---")
        print(r.to_string(index=False))

    p = pattern_report(root, a.pattern_files, a.voxel_stride)
    if len(p):
        p.to_csv(out / "pattern_dimensionality.csv", index=False)
        print()
        print(f"  --- RAW PATTERNS ({len(p)} files) ---")
        print(f"  voxels per pattern      : {int(p['n_voxels'].median()):,} "
              f"({p['frac_voxels_nonzero'].median() * 100:.0f}% non-zero -- "
              f"{'NO BRAIN MASK' if p['frac_voxels_nonzero'].median() > 0.95 else 'masked'})")
        print(f"  effective rank          : median {int(p['pattern_effective_rank'].median())} "
              f"of {int(p['n_stimuli'].median())} stimuli")
        print(f"  top-2 variance share    : {p['top2_variance_share'].median():.3f}")
        print(f"  PC1 vs global signal    : |rho| = {p['pc1_vs_global_signal'].median():.3f}")
        print(f"  PC1 vs the mean map     : |r|   = {p['pc1_vs_mean_map'].median():.3f}")
        print(f"  RDM vs amplitude-only   : rho   = {p['rdm_vs_amplitude_rsa'].median():+.3f}")

    verdict = None
    if len(p):
        unmasked = p["frac_voxels_nonzero"].median() > 0.95
        global_led = (p["pc1_vs_global_signal"].median() > 0.7
                      or p["rdm_vs_amplitude_rsa"].median() > 0.3)
        degenerate = len(r) and (r["group_rdm_effective_rank"].median()
                                 <= 0.15 * r["n_stim"].median())
        if unmasked and global_led:
            verdict = ("patterns are unmasked whole-volume and their leading component "
                       "tracks the global signal; these RDMs largely measure volume "
                       "brightness, not representational geometry")
        elif degenerate:
            verdict = ("RDMs are near-degenerate relative to their stimulus count; they "
                       "cannot express stimulus-level structure")
        else:
            verdict = "patterns carry multi-dimensional structure"
        print(f"\n  VERDICT: {verdict}")

    (out / "dimensionality_summary.json").write_text(json.dumps({
        "group_rdm_effective_rank_median": None if not len(r) else int(r["group_rdm_effective_rank"].median()),
        "n_stim_median": None if not len(r) else int(r["n_stim"].median()),
        "pattern_effective_rank_median": None if not len(p) else int(p["pattern_effective_rank"].median()),
        "pattern_n_voxels_median": None if not len(p) else int(p["n_voxels"].median()),
        "frac_voxels_nonzero_median": None if not len(p) else float(p["frac_voxels_nonzero"].median()),
        "pc1_vs_global_signal_median": None if not len(p) else float(p["pc1_vs_global_signal"].median()),
        "rdm_vs_amplitude_rsa_median": None if not len(p) else float(p["rdm_vs_amplitude_rsa"].median()),
        "verdict": verdict,
    }, indent=2))
    print(f"\n  wrote -> {out}")


if __name__ == "__main__":
    main()
