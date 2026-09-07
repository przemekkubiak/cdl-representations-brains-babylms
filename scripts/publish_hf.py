#!/usr/bin/env python3
"""Publish this sweep's per-dataset alignment results to the BrainAlign org.

One HF dataset repo per fMRI dataset, named for that dataset. Non-destructive on
repos that already exist: new files go under a dated subfolder and the existing
README is preserved as README_previous.md before the front page is rewritten.
"""
import os, io as _io, json, pandas as pd
from huggingface_hub import HfApi

STAMP = "2026-08-29"
API = HfApi(); TOK = os.environ["HF_TOKEN"]
ROOT = "/local/scratch/sas245/brainalign-evals/results"
DATASETS = {
 "ds003604": dict(cohort="children ages 5/7/9, auditory", tasks="Sem, Phon, Gram, Plaus",
                  sessions="ses-5, ses-7, ses-9", cells=12),
 "ds002236": dict(cohort="children ages 8.7-15.5", tasks="Phon, Sem",
                  sessions="ses-9, ses-11, ses-11+", cells=6),
 "ds006239": dict(cohort="children ages 10-17, reading", tasks="Orth, Phon, Sem, SemLocal",
                  sessions="ses-11, ses-11+", cells=8),
}

rows = pd.read_csv(f"{ROOT}/alignment_rows.csv")
nullref = pd.read_csv(f"{ROOT}/null_referenced.csv")
curve = pd.read_csv(f"{ROOT}/scaling_curve.csv")
trend = pd.read_csv(f"{ROOT}/scale_trend.csv")
nsum = pd.read_csv(f"{ROOT}/null_summary.csv")

README = """---
license: mit
task_categories: [feature-extraction]
tags: [brain-alignment, fmri, rsa, developmental, pythia, null-result]
---

# Brain-LM alignment: {ds}

Representational-similarity alignment between language-model hidden states and
child fMRI RDMs for **{ds}** ({cohort}).

- **Tasks:** {tasks}   **Sessions:** {sessions}   **Cells:** {cells}
- **Models:** {nfam} families ({nreal} real + 9 PARC noise-seed baselines)
- **Rows:** {nrows} (family x checkpoint x task x session)
- **Generated:** {stamp}

## Headline: no model is distinguishable from a random seed

Alignment is computed as Spearman correlation over the upper triangle of the
model RDM (`1 - corrcoef` over mean-pooled final-layer states) against the brain
RDM, reported raw (`rsa`) and as a fraction of the noise ceiling.

Every value is referenced against a **PARC noise-seed null** (9 randomly
initialised seeds across 3 architectures, same cells, same pipeline) rather than
against zero -- see `null_referenced.csv` for per-cell z-scores.

Across all three datasets and 130 (family x cell) combinations:

| statistic | value |
|---|---|
| cells where a real family beats all 9 noise seeds | **9 / 130** |
| expected under the null (P = 1/10 per cell) | **13.0** |
| binomial p | **0.91** |
| per-family mean z | all within +/-1.4 sigma |
| per-cell means, real families vs noise seeds | **r = +0.863** |
| variance explained by cell identity | **83.9%** |
| variance explained by model family | **3.2%** |

Real models beat the noise seeds *less* often than chance. The only reliable
structure in these numbers is the stimulus set and the RDM, not the model.

Scale trend for {ds}: Spearman(params, mean RSA) = **{rho:.2f}** (p = {p:.2f}),
i.e. a 16x parameter increase buys nothing.

## What this does and does not license

The RDMs have demonstrated inter-subject reliability (noise ceilings 0.23-0.88),
but the upstream pipeline's **positive controls fail on all three datasets**
(0/108, 0/6 and 0/8 stimulus controls significant). The instrument has therefore
not been shown to have power against alignment that does exist.

The defensible claim is **"no LM alignment is detectable by this measurement"** --
not "language models do not align with the developing brain." This is a result
about the benchmark.

## Coverage correction

The previously published grid did not pass `--sessions`, so it fell back to
ds003604's `ses-5/7/9`. ds002236 matched only `ses-9` (2 of 6 cells) and
**ds006239 matched nothing, producing zero alignment rows**. Deriving sessions
from the RDM tree takes coverage from 14 to **26 cells**, so this release
contains the first measurement of 12 previously unscored cells -- including
`ds006239/SemLocal`, the only run x stimulus *crossed* cell, where the
scanner-run confound cannot arise.

On SemLocal, untrained (step-0) alignment falls **inside** the random-seed band on
both sessions (z = -0.84, -0.30). The "untrained models align better, training
destroys it" reading is **not supported**: the decline is drift within noise, and
there was no alignment to destroy.

## Files

| file | contents |
|---|---|
| `alignment_rows.csv` | one row per family x checkpoint x task x session |
| `null_referenced.csv` | per-cell z-score against the 9-seed PARC null |
| `scaling_curve.csv` | per-cell mean/sd/max across checkpoints, + noise ceiling |
| `null_summary.csv` | per-family mean z, max z, cells beating the null |
| `scale_trend.csv` | Spearman(params, RSA) per dataset |

Precision: fp32 throughout. A bf16 spot check shifted RSA by up to 2.8e-3 at the
final checkpoint (~1/3 of the across-seed noise sd), so precision is not mixed.
"""

for ds, meta in DATASETS.items():
    repo = f"BrainAlign/brain-lm-alignment-{ds}"
    sub = {k: v[v.dataset == ds] if "dataset" in v.columns else v
           for k, v in dict(alignment_rows=rows, null_referenced=nullref,
                            scaling_curve=curve, scale_trend=trend).items()}
    sub["null_summary"] = nsum
    real = sorted(x for x in sub["alignment_rows"].family.unique() if not x.startswith("parc-"))
    tr = trend[trend.dataset == ds]
    rho = float(tr.scale_trend_rho.iloc[0]); pv = float(tr.scale_trend_p.iloc[0])

    exists = True
    try: API.dataset_info(repo, token=TOK)
    except Exception: exists = False
    if not exists:
        API.create_repo(repo, repo_type="dataset", token=TOK, exist_ok=True)
        print(f"CREATED {repo}")
    else:
        try:  # preserve the previous front page before rewriting it
            prev = API.hf_hub_download(repo_id=repo, filename="README.md",
                                       repo_type="dataset", token=TOK)
            API.upload_file(path_or_fileobj=prev, path_in_repo="README_previous.md",
                            repo_id=repo, repo_type="dataset", token=TOK,
                            commit_message="preserve prior README before devai-sweep update")
        except Exception as e:
            print(f"  (no prior README to preserve: {type(e).__name__})")

    for name, df in sub.items():
        buf = _io.BytesIO(df.to_csv(index=False).encode())
        API.upload_file(path_or_fileobj=buf, path_in_repo=f"{name}.csv", repo_id=repo,
                        repo_type="dataset", token=TOK,
                        commit_message=f"devai sweep {STAMP}: {name}")
    txt = README.format(ds=ds, stamp=STAMP, nfam=sub["alignment_rows"].family.nunique(),
                        nreal=len(real), nrows=len(sub["alignment_rows"]),
                        rho=rho, p=pv, **meta)
    API.upload_file(path_or_fileobj=_io.BytesIO(txt.encode()), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset", token=TOK,
                    commit_message=f"devai sweep {STAMP}: null-referenced results")
    print(f"  -> {repo}: {len(sub['alignment_rows'])} rows, {sub['alignment_rows'].family.nunique()} families")
print("\nDONE")
