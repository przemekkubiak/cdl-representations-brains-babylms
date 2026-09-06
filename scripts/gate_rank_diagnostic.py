#!/usr/bin/env python
"""Why does the positive control fail? Measure the beta rank, then try to fix it.

THE PROBLEM. The gate -- does anything stimulus-driven correlate with these
RDMs -- fails on all four datasets and every ROI level. run_new_datasets.sh's
header attributes this to per-stimulus betas being near-degenerate, "rank ~3 of
40-48 stimuli", which would mean the RDMs carry almost no stimulus structure to
align to and that every alignment null computed from them is a property of the
measurement rather than of the models.

THE HYPOTHESIS. src/preprocessing/fmri_preprocessing.py fits one regressor per
stimulus with `FirstLevelModel(...).fit(bold_img, events=...)` and passes NO
confounds. MASKING.md records that the BOLD it fits is raw and native-space --
no motion correction, no slice-timing correction, no coregistration. Fitting
40-48 single-trial regressors to unrealigned data is a plausible route to a
rank-3 beta matrix: motion dominates the design and the per-stimulus estimates
collapse onto a few directions.

WHAT THIS SCRIPT DOES. It does not assume that. For one subject-session it:

  1. fits the GLM exactly as the pipeline does, and reports the numerical rank
     and condition number of the resulting stimulus x voxel beta matrix;
  2. refits with nuisance regressors added -- aCompCor components from the
     highest-variance voxels plus spike regressors for outlier volumes, which
     capture motion-related variance without requiring realignment software the
     box does not have -- and reports the same numbers;
  3. reports the rank of the between-stimulus RDM in both cases, since that is
     what the gate actually consumes.

WHAT EACH OUTCOME MEANS.
  rank rises substantially  -> the degeneracy is nuisance variance, the fix is
                               to pass confounds, and the whole neuro track is
                               unblocked
  rank unchanged            -> the design itself is degenerate (too few volumes
                               per stimulus, or collinear onsets) and no amount
                               of denoising helps; the datasets cannot support
                               single-trial RSA as currently specified
Either way it is an answer, and it costs one subject-session rather than a sweep.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rank_report(name: str, betas: np.ndarray) -> dict:
    """Numerical rank and condition of a (stimuli x voxels) beta matrix."""
    b = betas - betas.mean(axis=0, keepdims=True)
    s = np.linalg.svd(b, compute_uv=False)
    tol = max(b.shape) * np.finfo(float).eps * s[0] if s.size else 0.0
    rank = int((s > tol).sum())
    # rank at 1% of the leading singular value: the practical rank, which is
    # what matters for an RDM built from these rows
    rank99 = int((s > 0.01 * s[0]).sum()) if s.size else 0
    rdm = np.corrcoef(b)
    rs = np.linalg.svd(rdm - rdm.mean(), compute_uv=False)
    rdm_rank = int((rs > 0.01 * rs[0]).sum()) if rs.size else 0
    out = dict(condition=name, n_stimuli=b.shape[0], n_voxels=b.shape[1],
               rank_exact=rank, rank_1pct=rank99,
               cond_number=float(s[0] / s[-1]) if s.size and s[-1] > 0 else np.inf,
               var_top3=float((s[:3] ** 2).sum() / (s ** 2).sum()) if s.size else np.nan,
               rdm_rank_1pct=rdm_rank)
    print(f"  {name:22s} stimuli={out['n_stimuli']:3d} rank(1%)={out['rank_1pct']:3d} "
          f"var in top 3 SVs={out['var_top3']:.3f}  RDM rank(1%)={out['rdm_rank_1pct']:3d}")
    return out


def nuisance(bold_data: np.ndarray, n_comp: int = 6,
             spike_z: float = 2.5) -> np.ndarray:
    """aCompCor-style components plus spike regressors, from the data itself.

    No realignment software is available here, so motion is addressed
    indirectly: the top principal components of the highest-variance voxels
    carry most structured non-stimulus variance, and volume-to-volume signal
    change (a DVARS proxy) flags the volumes where motion actually occurred.
    """
    x = bold_data.reshape(-1, bold_data.shape[-1]).T          # time x voxels
    x = x[:, np.isfinite(x).all(axis=0)]
    v = x.var(axis=0)
    keep = x[:, np.argsort(v)[-min(2000, x.shape[1]):]]
    keep = (keep - keep.mean(0)) / (keep.std(0) + 1e-8)
    comps = np.linalg.svd(keep, full_matrices=False)[0][:, :n_comp]
    dvars = np.r_[0.0, np.sqrt(((np.diff(x, axis=0)) ** 2).mean(axis=1))]
    z = (dvars - dvars.mean()) / (dvars.std() + 1e-8)
    spikes = np.zeros((len(z), int((z > spike_z).sum())))
    for k, t in enumerate(np.where(z > spike_z)[0]):
        spikes[t, k] = 1.0
    return np.c_[comps, spikes] if spikes.size else comps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ds002236")
    ap.add_argument("--task", default="Phon")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--out", default="results/gate_rank_diagnostic.csv")
    a = ap.parse_args()

    import nibabel as nib
    from nilearn.glm.first_level import FirstLevelModel
    from nilearn.maskers import NiftiMasker
    from src.preprocessing.fmri_preprocessing import FMRIPreprocessor

    root = Path(f"data/brain/{a.dataset}")
    subs = sorted(d.name for d in root.glob("sub-*") if d.is_dir())
    if not subs:
        print(f"no subjects under {root}"); return 1
    sub = a.subject or subs[0]
    pre = FMRIPreprocessor(data_dir=str(root), subject_id=sub, task=a.task,
                           dataset=a.dataset)
    print(f"subject {sub} of {len(subs)}")
    runs = pre.find_task_runs()
    if not runs:
        print(f"no runs for {a.dataset}/{a.task}"); return 1
    run = runs[0]
    print(f"{a.dataset} {a.task}: {len(runs)} runs; using {run.get('bold')}")

    bold = pre.load_bold(run["bold"])
    events = pre.load_events(run["events"])
    tr = float(bold.header.get_zooms()[3])
    bold = pre.preprocess_functional(bold, tr, verbose=False)

    masker = NiftiMasker(mask_strategy="epi", standardize=False)
    masker.fit(bold)
    ev = events.rename(columns={"trial_type": "orig_type"}).copy()
    ev["trial_type"] = [f"s{i:03d}" for i in range(len(ev))]   # one per stimulus
    ev = ev[["onset", "duration", "trial_type"]]

    rows = []
    for label, conf in (("as pipeline (no confounds)", None),
                        ("+ aCompCor & spikes", nuisance(bold.get_fdata()))):
        glm = FirstLevelModel(t_r=tr, noise_model="ar1", standardize=False,
                              hrf_model="spm", drift_model="cosine",
                              high_pass=0.01, mask_img=masker.mask_img_,
                              minimize_memory=False)
        cdf = None if conf is None else pd.DataFrame(
            conf, columns=[f"n{i}" for i in range(conf.shape[1])])
        glm = glm.fit(bold, events=ev, confounds=cdf)
        betas = np.vstack([masker.transform(
            glm.compute_contrast(t, output_type="effect_size"))
            for t in ev.trial_type])
        rows.append(rank_report(label, betas))

    d = pd.DataFrame(rows)
    d.insert(0, "dataset", a.dataset); d.insert(1, "task", a.task)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")
    if len(rows) == 2:
        g = rows[1]["rank_1pct"] - rows[0]["rank_1pct"]
        print(f"VERDICT: practical rank {rows[0]['rank_1pct']} -> {rows[1]['rank_1pct']} "
              f"({g:+d}) with confounds; RDM rank "
              f"{rows[0]['rdm_rank_1pct']} -> {rows[1]['rdm_rank_1pct']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
