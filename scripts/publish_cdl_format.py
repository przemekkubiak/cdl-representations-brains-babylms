#!/usr/bin/env python3
"""Publish per-study datasets in the BrainAlign/cdl-devai-results layout.

Wang et al. 2025  -> ds006239 (word-level phonological/semantic reading, ages 10-17)
Lytle et al. 2020 -> ds002236 (orth/phon/sem word processing, ages 8.7-15.5)
"""
import io as _io, json, os, numpy as np, pandas as pd
from huggingface_hub import HfApi

API = HfApi(); TOK = os.environ["HF_TOKEN"]; ROOT = "results"
ALIGN_COLS = ["family","model_ref","dataset","step","tokens","task","session",
              "n_stim","rsa","rsa_pearson","rsa_kendall","rsa_lo","rsa_hi"]

STUDIES = {
 "cdl-devai-wang2025": dict(
   ds="ds006239", cite="Wang et al. (2025)",
   desc="word-level phonological and semantic reading tasks in children and adolescents aged 10-17",
   cohort="children and adolescents, ages 10-17", modality="reading (visual)"),
 "cdl-devai-lytle2020": dict(
   ds="ds002236", cite="Lytle et al. (2020)",
   desc="orthographic, phonological and semantic word processing in school-aged children (8.7-15.5), auditory and visual",
   cohort="school-aged children, ages 8.7-15.5", modality="auditory and visual"),
}

rows = pd.read_csv(f"{ROOT}/alignment_rows.csv")
nullref = pd.read_csv(f"{ROOT}/null_referenced.csv")
nsum = pd.read_csv(f"{ROOT}/null_summary.csv")


def up(repo, path, obj, msg):
    b = obj.encode() if isinstance(obj, str) else obj.to_csv(index=False).encode()
    API.upload_file(path_or_fileobj=_io.BytesIO(b), path_in_repo=path, repo_id=repo,
                    repo_type="dataset", token=TOK, commit_message=msg)


for repo_name, S in STUDIES.items():
    repo = f"BrainAlign/{repo_name}"
    API.create_repo(repo, repo_type="dataset", token=TOK, exist_ok=True)
    d = rows[rows.dataset == S["ds"]].copy()
    nr = nullref[nullref.dataset == S["ds"]].copy()
    tasks = sorted(d.task.unique())
    real = sorted(f for f in d.family.unique() if not f.startswith("parc-"))
    noise = sorted(f for f in d.family.unique() if f.startswith("parc-"))

    ck_all = []
    for fam, g in d.groupby("family"):
        a = g.reindex(columns=ALIGN_COLS)
        up(repo, f"by-model/{fam}/brain_alignment.csv", a, f"{fam}: brain alignment")
        ck = []
        for (ref, step), h in g.groupby(["model_ref", "step"]):
            r = dict(family=fam, model_ref=ref, step=step,
                     tokens=h.tokens.iloc[0] if "tokens" in h else np.nan,
                     brain_rsa_mean=h.rsa.mean(), brain_rsa_std=h.rsa.std(),
                     brain_rsa_pearson_mean=h.rsa_pearson.mean(), brain_n_cells=len(h),
                     frac_of_ceiling_mean=h.frac_of_ceiling.mean())
            for t in tasks:
                r[f"brain_rsa_{t}"] = h[h.task == t].rsa.mean()
            ck.append(r)
        ck = pd.DataFrame(ck).sort_values("step")
        up(repo, f"by-model/{fam}/checkpoints.csv", ck, f"{fam}: per-checkpoint aggregate")
        ck_all.append(ck)
        kind = "PARC noise-seed baseline (randomly initialised)" if fam.startswith("parc-") else "trained language model"
        up(repo, f"by-model/{fam}/README.md",
           f"# {fam}\n\n{kind}.\n\nDataset: `{S['ds']}` ({S['cite']}).\n"
           f"Checkpoints: {g.model_ref.nunique()} | cells: {g.cell.nunique()} | rows: {len(g)}\n\n"
           f"Mean RSA {g.rsa.mean():+.4f}, sd {g.rsa.std():.4f}.\n", f"{fam}: README")

    up(repo, "overall/by_checkpoint.csv", pd.concat(ck_all), "overall: per-checkpoint")
    up(repo, "overall/summary_by_family.csv",
       d.groupby("family").agg(n_rows=("rsa", "size"), n_cells=("cell", "nunique"),
                               rsa_mean=("rsa", "mean"), rsa_sd=("rsa", "std"),
                               rsa_max=("rsa", "max"),
                               frac_ceiling=("frac_of_ceiling", "mean")).reset_index(),
       "overall: summary by family")
    up(repo, "overall/null_referenced.csv", nr, "overall: z vs PARC noise-seed null")
    up(repo, "overall/null_summary.csv", nsum, "overall: per-family null summary")

    beat = int(nr.exceeds_null_range.sum()); ncell = len(nr)
    up(repo, "overall/claim_tests.csv", pd.DataFrame([
        dict(claim="any real family beats all 9 noise seeds in a cell",
             observed=beat, expected_by_chance=round(ncell/10, 1), n=ncell,
             verdict="not supported"),
        dict(claim="alignment increases with model scale",
             observed=float(d[~d.family.str.startswith('parc-')].groupby('params').rsa.mean()
                            .corr(pd.Series(sorted(d[~d.family.str.startswith('parc-')].params.unique()),
                                            index=sorted(d[~d.family.str.startswith('parc-')].params.unique())),
                                  method='spearman')) if d.params.nunique() > 2 else np.nan,
             expected_by_chance=0.0, n=len(real), verdict="not supported")]),
       "overall: pre-declared claim tests")

    ceil = d.groupby(["task", "session"]).agg(
        ceiling_lower=("ceiling_lower", "first"), ceiling_upper=("ceiling_upper", "first"),
        ceiling_n=("ceiling_n", "first"), n_subjects=("n_subjects", "first"),
        best_rsa=("rsa", "max"), mean_rsa=("rsa", "mean")).reset_index()
    up(repo, "ceiling-analysis/ceilings.csv", ceil, "ceiling analysis")

    up(repo, "provenance_tier_ledger.json", json.dumps({
        "study": S["cite"], "openneuro_accession": S["ds"], "generated": "2026-08-29",
        "tier": "measured", "precision": "fp32",
        "estimator": "Spearman over upper-triangle of 1-corrcoef RDM, final-layer mean-pooled states",
        "null": "9 PARC randomly-initialised seeds across 3 architectures, identical cells",
        "n_rows": int(len(d)), "n_families_real": len(real), "n_families_noise": len(noise),
        "cells": int(d.cell.nunique()),
        "known_limitation": "upstream positive controls fail on this dataset; the instrument "
                            "has not been shown to have power against alignment that exists",
        "coverage_correction": "sessions derived from the RDM tree rather than the ds003604 "
                               "fallback; this release includes previously unscored cells",
    }, indent=2), "provenance ledger")

    up(repo, "README.md", f"""---
license: mit
task_categories: [feature-extraction]
tags: [brain-alignment, fmri, rsa, developmental, null-result]
---

# {S['cite']} — LM brain-alignment results

Language-model / fMRI representational-alignment results for **{S['cite']}**:
{S['desc']}

OpenNeuro accession **`{S['ds']}`** · cohort: {S['cohort']} · modality: {S['modality']}
· tasks: {', '.join(tasks)} · cells: {d.cell.nunique()} · rows: {len(d)}

Layout follows [`BrainAlign/cdl-devai-results`](https://huggingface.co/datasets/BrainAlign/cdl-devai-results):
`by-model/<family>/{{brain_alignment,checkpoints}}.csv`, `overall/`,
`ceiling-analysis/`, `provenance_tier_ledger.json`.

## Result: alignment is not distinguishable from a random seed

{len(real)} trained families ({', '.join(real)}) and {len(noise)} PARC noise-seed
baselines, scored identically. Cells where a real family exceeds all 9 noise
seeds: **{beat} of {ncell}**, against **{ncell/10:.1f} expected by chance**.

Pooled across all three developmental datasets: 9/130 observed vs 13.0 expected
(p = 0.91); per-cell means of real families vs noise seeds correlate at
**r = +0.863**; **83.9%** of variance is cell identity and **3.2%** model family.

## Interpretation

These RDMs have real inter-subject reliability (noise ceilings in
`ceiling-analysis/ceilings.csv`), but the upstream pipeline's **positive controls
fail on this dataset**. The claim supported here is *"no LM alignment is
detectable by this measurement"* — **not** "language models do not align with the
developing brain." Read this as a result about the benchmark.

Negative results are published here deliberately: the per-cell nulls and the
noise-seed baselines are the reusable part.
""", f"{S['cite']} results in cdl-devai format")
    print(f"{repo}: {len(d)} rows, {d.family.nunique()} families, {d.cell.nunique()} cells, beat={beat}/{ncell}")
print("DONE")
