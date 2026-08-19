#!/usr/bin/env python
"""Rebuild BrainAlign/cdl-devai-results as a human-readable, viewer-parseable dataset.

WHY THIS EXISTS
  The first layout was a flat dump: devai_grid/ds003604/ held 58 CSVs whose names were the
  only structure, and there was no table anywhere that put a single checkpoint's brain,
  interpretability and localisation numbers side by side -- the thing the results are
  actually about. It was also unreadable to the Hub's dataset viewer: with no config
  declaration the builder globs every CSV into one table and casts them to a single schema,
  so heldout_predictor.csv (held_out_family,r2_heldout,n_test) collided with
  devai_summary_*.csv (claim,stat,value,p,n) and the viewer died with
  DatasetGenerationCastError. There are ten genuinely different schemas here; they cannot
  share one.

WHAT THIS BUILDS
  overall/     cross-model tables, including by_checkpoint.csv -- one row per
               (family, step) with brain + interp + localisation columns together.
  by-model/<family>/   one directory per model, one FILE PER AXIS so that each Hub config
               groups files that genuinely share a schema, plus a README.
  diagnostics/ the run-confound evidence, which anyone reading the brain numbers needs.
  figures/     cross-model figures.

  Every table carries `family` and `model_ref` so a reader can filter by model inside the
  viewer without downloading anything. Several source tables lacked one or both
  (ablation_alignment, mechanistic_layer, devai_summary, isolation_comparison,
  heldout_predictor); they are backfilled here rather than left for the viewer to cast.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data/processed/language_models/devai_grid/ds003604"
DEVAI = ROOT / "data/processed/language_models/devai/ds003604"
LAYERWISE = ROOT / "data/processed/language_models/layerwise/ds003604"
CONFOUND = ROOT / "data/processed/language_models/confound_check/ds003604"
FIGS = ROOT / "figures/ds003604"
OUT = ROOT / "hf_results_staging"

FAMILIES = ["pico-decoder-tiny", "pico-decoder-small", "pico-decoder-medium",
            "pico-decoder-large", "beetle-humanscale-eng", "beetle-fineweb3-eng",
            "babylm-gpt2-3", "babylm-gpt2-5", "babylm-gpt2-7", "babylm-gpt2"]

# axis -> (source glob prefix, output filename)
AXES = {
    "brain_alignment":     ("alignment", GRID),
    "localisation_isolation": ("isolation", GRID),
    "interp_mechanistic":  ("mechanistic", GRID),
    "interp_layerwise":    ("mechanistic_layer", GRID),
    "behaviour":           ("behaviour", GRID),
    "ablation_alignment":  ("ablation_alignment", GRID),
    "ablation_behaviour":  ("ablation_behaviour", GRID),
}


def read(prefix: str, fam: str, base: Path) -> pd.DataFrame | None:
    p = base / f"{prefix}_{fam}.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except Exception:
        return None
    return d if len(d) else None


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "overall").mkdir(parents=True)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "diagnostics").mkdir(parents=True)

    # ---- model_ref lookup: several tables record (family, step) but not the ref -------
    ref_map: dict[tuple[str, int], str] = {}
    for fam in FAMILIES:
        for pre in ("alignment", "mechanistic", "isolation", "behaviour"):
            d = read(pre, fam, GRID)
            if d is None or "model_ref" not in d:
                continue
            for f_, s_, r_ in zip(d["family"], d["step"], d["model_ref"]):
                ref_map.setdefault((str(f_), int(s_)), str(r_))

    def ensure_keys(d: pd.DataFrame, fam: str) -> pd.DataFrame:
        d = d.copy()
        if "family" not in d:
            d.insert(0, "family", fam)
        d["family"] = d["family"].fillna(fam)
        if "model_ref" not in d:
            d["model_ref"] = [ref_map.get((fam, int(s)), "") if pd.notna(s) else ""
                              for s in d.get("step", pd.Series([np.nan] * len(d)))]
        cols = ["family", "model_ref"] + [c for c in d.columns if c not in ("family", "model_ref")]
        return d[cols]

    per_axis: dict[str, list[pd.DataFrame]] = {k: [] for k in AXES}
    per_family_tables: dict[str, dict[str, pd.DataFrame]] = {}

    for fam in FAMILIES:
        fd = OUT / "by-model" / fam
        fd.mkdir(parents=True, exist_ok=True)
        (fd / "figures").mkdir(exist_ok=True)
        tables = {}
        for axis, (prefix, base) in AXES.items():
            d = read(prefix, fam, base)
            if d is None:
                continue
            d = ensure_keys(d, fam)
            tables[axis] = d
            per_axis[axis].append(d)
        per_family_tables[fam] = tables

    # ---- normalise columns WITHIN an axis so every config has one schema -------------
    for axis, frames in per_axis.items():
        if not frames:
            continue
        cols: list[str] = []
        for f in frames:
            for c in f.columns:
                if c not in cols:
                    cols.append(c)
        per_axis[axis] = [f.reindex(columns=cols) for f in frames]
        for fam, tabs in per_family_tables.items():
            if axis in tabs:
                tabs[axis] = tabs[axis].reindex(columns=cols)

    for fam, tabs in per_family_tables.items():
        for axis, d in tabs.items():
            d.to_csv(OUT / "by-model" / fam / f"{axis}.csv", index=False)

    # ---- the master per-checkpoint join: brain + interp + localisation --------------
    rows = []
    for fam in FAMILIES:
        t = per_family_tables.get(fam, {})
        al, me, iso, beh = (t.get("brain_alignment"), t.get("interp_mechanistic"),
                            t.get("localisation_isolation"), t.get("behaviour"))
        steps = set()
        for d in (al, me, iso, beh):
            if d is not None and "step" in d:
                steps |= {int(s) for s in d["step"].dropna()}
        for st in sorted(steps):
            r: dict = {"family": fam, "step": st,
                       "model_ref": ref_map.get((fam, st), "")}
            if al is not None:
                a = al[al.step == st]
                r["tokens"] = a["tokens"].iloc[0] if len(a) and "tokens" in a else np.nan
                r["brain_rsa_mean"] = a["rsa"].mean()
                r["brain_rsa_std"] = a["rsa"].std()
                r["brain_rsa_pearson_mean"] = a["rsa_pearson"].mean()
                r["brain_n_cells"] = int(len(a))
                for task in ("Sem", "Phon", "Gram", "Plaus"):
                    r[f"brain_rsa_{task}"] = a[a.task == task]["rsa"].mean() if len(a) else np.nan
            if me is not None:
                m = me[me.step == st]
                if len(m):
                    for c in ("norm", "gini", "hoyer", "per", "condition_number", "cka_to_prev"):
                        if c in m:
                            r[f"interp_{c}"] = m[c].iloc[0]
            if iso is not None:
                i = iso[iso.step == st]
                if len(i):
                    for c, nm in (("selectivity_index", "selectivity"),
                                  ("mean_overlap_with_others", "overlap"),
                                  ("gini", "gini"), ("entropy", "entropy"),
                                  ("layer_com", "layer_com"),
                                  ("n_active_layers", "n_active_layers")):
                        if c in i:
                            r[f"loc_{nm}"] = i[c].mean()
            if beh is not None:
                b = beh[beh.step == st]
                if len(b) and "mp_accuracy" in b:
                    r["behav_mp_accuracy"] = b["mp_accuracy"].mean()
            rows.append(r)
    by_ck = pd.DataFrame(rows).sort_values(["family", "step"])
    lead = ["family", "model_ref", "step", "tokens"]
    by_ck = by_ck[lead + [c for c in by_ck.columns if c not in lead]]
    by_ck.to_csv(OUT / "overall" / "by_checkpoint.csv", index=False)

    for fam in FAMILIES:
        sub = by_ck[by_ck.family == fam]
        if len(sub):
            sub.to_csv(OUT / "by-model" / fam / "checkpoints.csv", index=False)

    # ---- one row per model ----------------------------------------------------------
    frows = []
    for fam in FAMILIES:
        sub = by_ck[by_ck.family == fam]
        if not len(sub):
            continue
        al = per_family_tables[fam].get("brain_alignment")
        rho = p = np.nan
        if al is not None and al["step"].nunique() > 2:
            rho, p = spearmanr(al["step"], al["rsa"])
        r = {"family": fam, "n_checkpoints": int(sub["step"].nunique()),
             "first_step": int(sub["step"].min()), "last_step": int(sub["step"].max()),
             "brain_rsa_mean": sub["brain_rsa_mean"].mean(),
             "brain_rsa_min": sub["brain_rsa_mean"].min(),
             "brain_rsa_max": sub["brain_rsa_mean"].max(),
             "brain_trend_rho": rho, "brain_trend_p": p,
             "brain_trend_n": int(len(al)) if al is not None else 0}
        for c in sub.columns:
            if c.startswith(("interp_", "loc_", "behav_")):
                r[c + "_mean"] = sub[c].mean()
        abl = per_family_tables[fam].get("ablation_behaviour")
        if abl is not None and "causal_selectivity" in abl:
            r["causal_selectivity_mean"] = abl["causal_selectivity"].mean()
        frows.append(r)
    pd.DataFrame(frows).to_csv(OUT / "overall" / "summary_by_family.csv", index=False)

    # ---- claim tests + held-out predictor, with family backfilled -------------------
    cl = []
    for fam in FAMILIES:
        d = read("devai_summary", fam, DEVAI)
        if d is None:
            continue
        d = d.copy()
        d.insert(0, "family", fam)
        d.insert(1, "model_ref", "")
        cl.append(d)
    if cl:
        pd.concat(cl).to_csv(OUT / "overall" / "claim_tests.csv", index=False)

    ic = []
    for fam in FAMILIES:
        d = read("isolation_comparison", fam, DEVAI)
        if d is None:
            continue
        d = d.copy()
        d.insert(0, "family", fam)
        d.insert(1, "model_ref", "")
        ic.append(d)
        d.to_csv(OUT / "by-model" / fam / "localisation_onset.csv", index=False)
    if ic:
        pd.concat(ic).to_csv(OUT / "overall" / "localisation_onset.csv", index=False)

    hp = DEVAI / "heldout_predictor.csv"
    if hp.exists():
        d = pd.read_csv(hp)
        d.insert(0, "family", d["held_out_family"])
        d.insert(1, "model_ref", "")
        d.to_csv(OUT / "overall" / "heldout_predictor.csv", index=False)

    # ---- diagnostics ---------------------------------------------------------------
    lw = [pd.read_csv(f) for f in sorted(LAYERWISE.glob("*.csv"))]
    if lw:
        d = pd.concat(lw)
        if "model_ref" not in d:
            d["model_ref"] = ""
        d.to_csv(OUT / "diagnostics" / "layerwise_alignment.csv", index=False)
    cf = [pd.read_csv(f) for f in sorted(CONFOUND.glob("*.csv"))]
    if cf:
        d = pd.concat(cf)
        if "model_ref" not in d:
            d["model_ref"] = ""
        d.to_csv(OUT / "diagnostics" / "run_confound_check.csv", index=False)

    for f in sorted(FIGS.glob("*")):
        shutil.copy(f, OUT / "figures" / f.name)

    print("staged:", sum(1 for _ in OUT.rglob("*") if _.is_file()), "files")
    print("by_checkpoint.csv:", len(by_ck), "rows,", by_ck.family.nunique(), "families")
    print("columns:", list(by_ck.columns))


if __name__ == "__main__":
    main()
