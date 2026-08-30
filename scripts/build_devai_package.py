#!/usr/bin/env python
"""Package one dataset's sweep into the BrainAlign/cdl-devai-results layout.

`cdl-devai-results` covers ds003604 only. The other two developmental datasets in
`BrainAlign/ds003604-session-rdms` have never been published in that shape:

  ds002236  Lytle et al. 2020  -- orthographic / phonological / semantic word
                                 processing, school-aged children 8.7-15.5,
                                 auditory and visual presentation.
                                 Published repo covers 2 of 6 cells.
  ds006239  Wang et al. 2025   -- word-level phonological and semantic reading,
                                 children and adolescents 10-17.
                                 Published repo has ZERO alignment rows.

Both gaps have the same cause: `slurm/run_devai_grid.sh` never passes
`--sessions`, so the runner fell back to the ds003604 default
["ses-5","ses-7","ses-9"]. ds002236 matched only ses-9; ds006239 matched nothing.

The per-family statistics are NOT reimplemented here. This drives the same
upstream `mechanistic_brain_analysis.py` that produced the published claim tests,
then reshapes with the same logic as upstream `build_hf_results.py`, so a row in
either new repo means exactly what the same row means in cdl-devai-results.

Two things are ADDED, both because this collection cannot be read without them:
  * `frac_of_ceiling` beside every rsa, from the RDM files' own inter-subject
    noise ceilings. On ds002236 the ceiling is as low as 0.23, so a raw rsa is
    uninterpretable on its own.
  * `overall/parc_reference.csv` -- every family's alignment against the PARC
    families measured on the same cells. NOTE ON TERMINOLOGY: both this repo's
    STATUS.md and the upstream cdl-devai-results README call the PARC runs
    "pure-noise" seeds. They are not. `configs/model_zoo_extended.yaml` records
    them as "PARC Pythia 160M (transformer), seed N. OpenWebText, 4000 steps,
    73 checkpoints" -- fully TRAINED models that differ only in architecture and
    initialisation seed. They are a matched-scale seed reference, not a random-
    init noise floor, and this package labels them accordingly. No randomly
    initialised baseline exists anywhere in this collection.

Usage: build_devai_package.py --dataset ds006239 [--push] [--private]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path("/local/scratch/sas245/brainalign-evals")
PIPELINE = ROOT / "pipeline"
RDM = ROOT / "data/ds003604-session-rdms"
PY = sys.executable

AXES = {                       # output filename -> grid file prefix
    "brain_alignment": "alignment",
    "localisation_isolation": "isolation",
    "interp_mechanistic": "mechanistic",
    "interp_layerwise": "mechanistic_layer",
    "behaviour": "behaviour",
}

STUDY = {
    "ds002236": dict(
        citation="Lytle et al. 2020",
        title="Lytle et al. 2020 — orthographic, phonological and semantic word "
              "processing in school-aged children",
        cohort="children 8.7–15.5 years",
        modality="auditory and visual word presentation",
        tasks="Phon, Sem",
        sessions="ses-9, ses-11, ses-11+",
        prior_repo="BrainAlign/brain-lm-alignment-ds002236",
        prior_gap="covered 2 of 6 task × session cells",
    ),
    "ds006239": dict(
        citation="Wang et al. 2025",
        title="Wang et al. 2025 — word-level phonological and semantic reading "
              "in children and adolescents",
        cohort="children and adolescents 10–17 years",
        modality="visual (reading)",
        tasks="Orth, Phon, Sem, SemLocal",
        sessions="ses-11, ses-11+",
        prior_repo="BrainAlign/brain-lm-alignment-ds006239",
        prior_gap="had ceilings and controls but ZERO alignment rows",
    ),
    "ds003604": dict(
        citation="Wang et al. 2022",
        title="Wang et al. 2022 — auditory language comprehension in children",
        cohort="children scanned at 5, 7 and 9",
        modality="auditory",
        tasks="Sem, Phon, Gram, Plaus",
        sessions="ses-5, ses-7, ses-9",
        prior_repo="BrainAlign/cdl-devai-results",
        prior_gap="already published; rebuilt here for comparability",
    ),
}


# --------------------------------------------------------------------------- #
def ceilings(dataset: str, variant: str = "within-run-normalised") -> pd.DataFrame:
    """(task, session) -> inter-subject noise ceiling, read off the RDM files."""
    rows = []
    for f in glob.glob(str(RDM / dataset / variant / "*/session_rdm_*.npz")):
        p = Path(f)
        d = np.load(f, allow_pickle=True)
        rows.append(dict(
            task=p.parts[-2],
            session=re.sub(r"session_rdm_(.*)\.npz", r"\1", p.name),
            n_stim=int(d["rdm"].shape[0]),
            ceiling_lower=float(d["noise_ceiling_lower"]) if "noise_ceiling_lower" in d.files else np.nan,
            ceiling_upper=float(d["noise_ceiling_upper"]) if "noise_ceiling_upper" in d.files else np.nan,
            n_subjects=int(d["n_subjects"]) if "n_subjects" in d.files else -1,
        ))
    return pd.DataFrame(rows)


def read_csv_safe(f: Path) -> pd.DataFrame | None:
    """Read a CSV that a mid-sweep upstream run may have left empty.

    `mechanistic_brain_analysis.py` writes an empty frame when a family has too
    few checkpoints for any claim to be computable, which is the normal state of
    the family currently being swept. pandas raises EmptyDataError on that, and
    an unguarded read took down the whole packaging cycle for both datasets when
    pythia-6.9b-full appeared with one checkpoint.
    """
    try:
        if not f.exists() or f.stat().st_size < 3:
            return None
        d = pd.read_csv(f)
        return d if len(d) else None
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None


def duplicate_cells(dataset: str, variant: str = "within-run-normalised") -> dict[tuple[str, str], tuple[str, str]]:
    """Find (task, session) cells whose RDMs are bit-identical to another cell.

    ds006239's Orth and Phon RDMs are the same array, to atol=1e-12, in both
    sessions -- same stimuli, same subjects, same ceilings. The dataset has SIX
    independent cells, not eight, and every count that treats them as distinct
    double-weights one stimulus set.
    """
    files = sorted(glob.glob(str(RDM / dataset / variant / "*/session_rdm_*.npz")))
    loaded = {}
    for f in files:
        q = Path(f)
        key = (q.parts[-2], re.sub(r"session_rdm_(.*)\.npz", r"\1", q.name))
        loaded[key] = np.load(f, allow_pickle=True)["rdm"]
    dupes: dict[tuple[str, str], tuple[str, str]] = {}
    seen: list[tuple[tuple[str, str], "np.ndarray"]] = []
    for key, rdm in loaded.items():
        for ref_key, ref in seen:
            if ref.shape == rdm.shape and np.allclose(ref, rdm, atol=1e-12):
                dupes[key] = ref_key
                break
        else:
            seen.append((key, rdm))
    return dupes


def params_map() -> dict:
    zoo = yaml.safe_load((ROOT / "configs/model_zoo_extended.yaml").read_text())["families"]
    out = {}
    for fam, cfg in zoo.items():
        p = cfg.get("params") or cfg.get("n_params")
        if p:
            out[fam] = int(p)
            continue
        m = re.search(r"(\d+(?:\.\d+)?)([mb])", fam.lower())
        if m:
            out[fam] = int(float(m.group(1)) * (1e6 if m.group(2) == "m" else 1e9))
    return out


def run_upstream_stats(dataset: str, families: list[str], devai: Path) -> None:
    """Drive the upstream per-family analysis: claim tests + isolation onset."""
    devai.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(PIPELINE))
    for fam in families:
        if (devai / f"devai_summary_{fam}.csv").exists():
            continue
        subprocess.run(
            [PY, str(PIPELINE / "scripts/mechanistic_brain_analysis.py"),
             "--family", fam, "--grid-dir", str(ROOT / "grid" / dataset),
             "--output-dir", str(devai)],
            cwd=PIPELINE, env=env, capture_output=True, text=True)


# --------------------------------------------------------------------------- #
def build(dataset: str, out: Path, variant: str = "within-run-normalised",
          grid_dir: str | None = None) -> dict:
    grid = ROOT / (grid_dir or f"grid/{dataset}")
    families = sorted({Path(f).name[len("alignment_"):-4]
                       for f in glob.glob(str(grid / "alignment_*.csv"))})
    if not families:
        raise SystemExit(f"no alignment CSVs under {grid}")

    devai = ROOT / "devai" / dataset
    run_upstream_stats(dataset, families, devai)

    if out.exists():
        shutil.rmtree(out)
    (out / "overall").mkdir(parents=True)
    (out / "diagnostics").mkdir(parents=True)

    def read(prefix: str, fam: str):
        return read_csv_safe(grid / f"{prefix}_{fam}.csv")

    # model_ref is missing from mechanistic_layer; backfill from the tables that have it
    ref_map: dict[tuple[str, int], str] = {}
    for fam in families:
        for pre in ("alignment", "mechanistic", "isolation", "behaviour"):
            d = read(pre, fam)
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
        cols = ["family", "model_ref"] + [c for c in d.columns
                                          if c not in ("family", "model_ref")]
        return d[cols]

    ceil = ceilings(dataset, variant)
    dupes = duplicate_cells(dataset, variant)
    per_axis: dict[str, list[pd.DataFrame]] = {k: [] for k in AXES}
    tabs_by_fam: dict[str, dict[str, pd.DataFrame]] = {}

    for fam in families:
        tabs = {}
        for axis, prefix in AXES.items():
            d = read(prefix, fam)
            if d is None:
                continue
            d = ensure_keys(d, fam)
            if axis == "brain_alignment":
                # Ceiling-normalise. A raw rsa of 0.02 means something different
                # against a 0.85 ceiling than against a 0.23 one.
                d = d.merge(ceil[["task", "session", "ceiling_lower", "ceiling_upper",
                                  "n_subjects"]], on=["task", "session"], how="left")
                d["frac_of_ceiling"] = d["rsa"] / d["ceiling_lower"]
                d["rsa_lo"] = np.nan
                d["rsa_hi"] = np.nan
            tabs[axis] = d
            per_axis[axis].append(d)
        tabs_by_fam[fam] = tabs

    # one schema per axis, so each Hub config parses
    for axis, frames in per_axis.items():
        if not frames:
            continue
        cols: list[str] = []
        for f in frames:
            cols += [c for c in f.columns if c not in cols]
        per_axis[axis] = [f.reindex(columns=cols) for f in frames]
        for tabs in tabs_by_fam.values():
            if axis in tabs:
                tabs[axis] = tabs[axis].reindex(columns=cols)

    for fam, tabs in tabs_by_fam.items():
        fd = out / "by-model" / fam
        fd.mkdir(parents=True, exist_ok=True)
        for axis, d in tabs.items():
            d.to_csv(fd / f"{axis}.csv", index=False)

    tasks = sorted(ceil.task.unique())

    # ---- the per-checkpoint join: brain + interp + localisation + behaviour --
    rows = []
    for fam in families:
        t = tabs_by_fam.get(fam, {})
        al, me = t.get("brain_alignment"), t.get("interp_mechanistic")
        iso, beh = t.get("localisation_isolation"), t.get("behaviour")
        steps: set[int] = set()
        for d in (al, me, iso, beh):
            if d is not None and "step" in d:
                steps |= {int(s) for s in d["step"].dropna()}
        for st in sorted(steps):
            r: dict = {"family": fam, "step": st, "model_ref": ref_map.get((fam, st), "")}
            if al is not None:
                a = al[al.step == st]
                r["tokens"] = a["tokens"].iloc[0] if len(a) and "tokens" in a else np.nan
                r["brain_rsa_mean"] = a["rsa"].mean()
                r["brain_rsa_std"] = a["rsa"].std()
                r["brain_rsa_pearson_mean"] = a["rsa_pearson"].mean()
                r["brain_frac_of_ceiling_mean"] = a["frac_of_ceiling"].mean()
                r["brain_n_cells"] = int(len(a))
                for task in tasks:
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
    by_ck.to_csv(out / "overall" / "by_checkpoint.csv", index=False)
    for fam in families:
        sub = by_ck[by_ck.family == fam]
        if len(sub):
            sub.to_csv(out / "by-model" / fam / "checkpoints.csv", index=False)

    # ---- one row per family --------------------------------------------------
    pmap = params_map()
    frows = []
    for fam in families:
        sub = by_ck[by_ck.family == fam]
        if not len(sub):
            continue
        al = tabs_by_fam[fam].get("brain_alignment")
        rho = p = np.nan
        if al is not None and al["step"].nunique() > 2:
            rho, p = spearmanr(al["step"], al["rsa"])
        r = {"family": fam, "params": pmap.get(fam, np.nan),
             "n_checkpoints": int(sub["step"].nunique()),
             "first_step": int(sub["step"].min()), "last_step": int(sub["step"].max()),
             "brain_rsa_mean": sub["brain_rsa_mean"].mean(),
             "brain_rsa_min": sub["brain_rsa_mean"].min(),
             "brain_rsa_max": sub["brain_rsa_mean"].max(),
             "brain_frac_of_ceiling_mean": sub.get(
                 "brain_frac_of_ceiling_mean", pd.Series(dtype=float)).mean(),
             "brain_trend_rho": rho, "brain_trend_p": p,
             "brain_trend_n": int(len(al)) if al is not None else 0}
        for c in sub.columns:
            if c.startswith(("interp_", "loc_", "behav_")):
                r[c + "_mean"] = sub[c].mean()
        frows.append(r)
    fam_summary = pd.DataFrame(frows)
    fam_summary.to_csv(out / "overall" / "summary_by_family.csv", index=False)

    # ---- upstream claim tests + isolation onset ------------------------------
    cl, ic = [], []
    for fam in families:
        d = read_csv_safe(devai / f"devai_summary_{fam}.csv")
        if d is not None:
            d.insert(0, "family", fam)
            d.insert(1, "model_ref", "")
            cl.append(d)
        d = read_csv_safe(devai / f"isolation_comparison_{fam}.csv")
        if d is not None:
            d.insert(0, "family", fam)
            d.insert(1, "model_ref", "")
            ic.append(d)
            d.to_csv(out / "by-model" / fam / "localisation_onset.csv", index=False)
    if cl:
        pd.concat(cl).to_csv(out / "overall" / "claim_tests.csv", index=False)
    if ic:
        pd.concat(ic).to_csv(out / "overall" / "localisation_onset.csv", index=False)

    # ---- the null reference: every family against the matched noise seeds ----
    al_all = pd.concat(per_axis["brain_alignment"]) if per_axis["brain_alignment"] else pd.DataFrame()
    null_rows = []
    if len(al_all):
        parc = al_all[al_all.family.str.startswith("parc-")]
        real = al_all[~al_all.family.str.startswith("parc-")]
        if len(parc):
            # Aggregate to one value per PARC FAMILY first. Taking the sd over
            # pooled raw checkpoint rows compares an averaged numerator against
            # an unaveraged denominator: the pooled sd is 2.4x the across-seed
            # sd here, so every z came out 2.4x too small and "0 exceed" was
            # partly an artifact of the mismatch.
            parc_fam = parc.groupby(["task", "session", "family"]).rsa.mean().reset_index()
            floor = parc_fam.groupby(["task", "session"]).rsa.agg(
                ["mean", "std", "count"]).rename(
                columns={"mean": "parc_mean", "std": "parc_sd", "count": "parc_n"})
            per_cell = real.groupby(["family", "task", "session"]).rsa.mean().reset_index()
            per_cell = per_cell.merge(floor, on=["task", "session"], how="left")
            per_cell = per_cell.merge(
                ceil[["task", "session", "ceiling_lower"]], on=["task", "session"], how="left")
            # z against the across-seed PARC spread on the SAME cell
            per_cell["z_vs_parc"] = (per_cell.rsa - per_cell.parc_mean) / per_cell.parc_sd
            per_cell["frac_of_ceiling"] = per_cell.rsa / per_cell.ceiling_lower
            # Two-sided, always. A cell that exceeds the band as often as it
            # falls below it contributes no evidence of alignment, only of a
            # variance mismatch, and reporting only the upper tail hides that.
            per_cell["exceeds_parc_2sd"] = per_cell.z_vs_parc > 2.0
            per_cell["below_parc_2sd"] = per_cell.z_vs_parc < -2.0
            per_cell.round(6).to_csv(out / "overall" / "parc_reference.csv", index=False)
            null_rows = per_cell.to_dict("records")

    # ---- the random-init reference: step-0 checkpoints ----------------------
    # These ARE untrained networks -- Pythia/PolyPythia publish step0 as the
    # initialisation before any optimiser step, and the sweep measured 15 of
    # them across every cell. An earlier version of this package stated that no
    # random-init baseline existed anywhere in the collection. That was wrong:
    # it was already on disk, unlabelled.
    untrained_stats = {}
    if len(al_all) and "step" in al_all:
        init = al_all[al_all.step == 0]
        trained = al_all[al_all.step > 0]
        if len(init) and len(trained):
            iband = init.groupby(["task", "session", "family"]).rsa.mean().reset_index()
            band = iband.groupby(["task", "session"]).rsa.agg(
                ["mean", "std", "count"]).rename(columns={
                    "mean": "init_mean", "std": "init_sd", "count": "init_n_seeds"})
            tcell = trained.groupby(["family", "task", "session"]).rsa.mean().reset_index()
            tcell = tcell.merge(band, on=["task", "session"], how="left")
            tcell["z_vs_untrained"] = (tcell.rsa - tcell.init_mean) / tcell.init_sd
            tcell["beats_untrained_2sd"] = tcell.z_vs_untrained > 2.0
            tcell["below_untrained_2sd"] = tcell.z_vs_untrained < -2.0
            tcell.round(6).to_csv(out / "overall" / "untrained_reference.csv", index=False)

            # Paired within family: does training this model help on this cell?
            pair = []
            for (fam, task, ses), sub in al_all.groupby(["family", "task", "session"]):
                if 0 not in set(sub.step) or sub.step.max() == 0:
                    continue
                pair.append(dict(family=fam, task=task, session=ses,
                                 untrained=float(sub[sub.step == 0].rsa.mean()),
                                 trained=float(sub[sub.step == sub.step.max()].rsa.mean())))
            pdf = pd.DataFrame(pair)
            if len(pdf) >= 10:
                from scipy.stats import wilcoxon
                stat, pv = wilcoxon(pdf.untrained, pdf.trained)
                pdf["training_helps"] = pdf.trained > pdf.untrained
                pdf.round(6).to_csv(out / "overall" / "untrained_vs_trained_paired.csv",
                                    index=False)
                untrained_stats = dict(
                    n_pairs=int(len(pdf)), n_families=int(pdf.family.nunique()),
                    n_training_helps=int(pdf.training_helps.sum()),
                    frac_training_helps=round(float(pdf.training_helps.mean()), 4),
                    mean_untrained=round(float(pdf.untrained.mean()), 6),
                    mean_trained=round(float(pdf.trained.mean()), 6),
                    wilcoxon_p=float(pv),
                    verdict=("training IMPROVES alignment on this dataset"
                             if pdf.training_helps.mean() > 0.5 and pv < 0.05 else
                             "training DEGRADES alignment on this dataset"
                             if pdf.training_helps.mean() < 0.5 and pv < 0.05 else
                             "no detectable difference between trained and untrained"))

    # ---- diagnostics: the layerwise profile, if the sweep produced one -------
    lw = [pd.read_csv(f) for f in sorted(grid.glob("layerwise_*.csv"))]
    if lw:
        pd.concat(lw).to_csv(out / "diagnostics" / "layerwise_alignment.csv", index=False)
    ceil.to_csv(out / "diagnostics" / "noise_ceilings.csv", index=False)

    # ---- instrument diagnostics: is this measurement licensed at all? -------
    # These are what make a null interpretable. Copied per dataset so each
    # package is self-contained.
    instrument = {}
    for src, dst in [("detectability_wstar.csv", "detectability_wstar.csv"),
                     ("external_rdm_anchor.csv", "external_rdm_anchor.csv"),
                     ("additive_control.csv", "additive_control.csv"),
                     ("instrument_ratio.csv", "instrument_ratio.csv"),
                     ("retest_final_checkpoint.csv", "precision_retest.csv"),
                     ("retest_by_step.csv", "precision_retest_by_step.csv")]:
        f = ROOT / "results" / src
        if not f.is_file():
            continue
        d = pd.read_csv(f)
        if "dataset" in d.columns:
            d = d[d.dataset == dataset]
        if len(d):
            d.to_csv(out / "diagnostics" / dst, index=False)
    for name, f in [("detectability", ROOT / "results/detectability_verdict.json"),
                    ("precision_retest", ROOT / "results/retest_verdict.json")]:
        if f.is_file():
            instrument[name] = json.loads(f.read_text())

    anch = ROOT / "results/external_rdm_anchor.csv"
    ratio = ROOT / "results/instrument_ratio.csv"
    addf = ROOT / "results/additive_control.csv"
    wsf = ROOT / "results/detectability_wstar.csv"
    if all(f.is_file() for f in (anch, ratio, addf, wsf)):
        a_ = pd.read_csv(anch); a_ = a_[a_.dataset == dataset]
        r_ = pd.read_csv(ratio); r_ = r_[r_.dataset == dataset]
        ad = pd.read_csv(addf); ad = ad[ad.dataset == dataset]
        w_ = pd.read_csv(wsf); w_ = w_[w_.dataset == dataset]
        instrument["this_dataset"] = dict(
            cross_session_rsa_median=round(float(a_.cross_session_rsa.median()), 4),
            cross_session_rsa_min=round(float(a_.cross_session_rsa.min()), 4),
            best_model_rsa=round(float(r_.best_model_rsa.max()), 5),
            model_to_external_ratio_median=round(float(r_.ratio_model_to_external.median()), 4),
            w_star_median=(float(w_.w_star.median()) if w_.w_star.notna().any() else None),
            additive_n_significant=int(ad.significant.sum()),
            additive_n_cells=int(len(ad)),
            additive_median_rsa=round(float(ad.additive_rsa.median()), 4),
            additive_max_rsa=round(float(ad.additive_rsa.max()), 4))


    stats = dict(
        dataset=dataset, rdm_variant=variant, families=len(families), checkpoints=int(len(by_ck)),
        alignment_rows=int(len(al_all)),
        cells=int(al_all.groupby(["task", "session"]).ngroups) if len(al_all) else 0,
        tasks=tasks, sessions=sorted(ceil.session.unique()),
        n_exceeding_parc=int(sum(1 for r in null_rows if r.get("exceeds_parc_2sd"))),
        n_below_parc=int(sum(1 for r in null_rows if r.get("below_parc_2sd"))),
        untrained=untrained_stats,
        duplicate_cells={f"{k[0]}/{k[1]}": f"{v[0]}/{v[1]}" for k, v in dupes.items()},
        n_independent_cells=int(al_all.groupby(["task", "session"]).ngroups - len(dupes))
                            if len(al_all) else 0,
        incomplete_families={f: int(n) for f, n in
                             by_ck.groupby("family").step.nunique().items() if n < 5},
        n_cells_tested=len(null_rows),
        best_frac_of_ceiling=float(al_all.frac_of_ceiling.max()) if len(al_all) else float("nan"),
        best_rsa=float(al_all.rsa.max()) if len(al_all) else float("nan"),
        instrument=instrument,
    )
    (out / "overall" / "build_summary.json").write_text(json.dumps(stats, indent=2))
    return stats


# --------------------------------------------------------------------------- #
def readme(dataset: str, stats: dict) -> str:
    s = STUDY[dataset]
    exceed = stats["n_exceeding_parc"]
    below = stats.get("n_below_parc", 0)
    u = stats.get("untrained", {}) or {}
    untrained_line = (
        f"{u.get('n_training_helps','?')}/{u.get('n_pairs','?')} family x cell pairs "
        f"({u.get('frac_training_helps', float('nan')):.0%}) improve with training\n"
        f"mean rsa   untrained {u.get('mean_untrained', float('nan')):+.5f}   "
        f"trained {u.get('mean_trained', float('nan')):+.5f}\n"
        f"Wilcoxon signed-rank p = {u.get('wilcoxon_p', float('nan')):.2g}\n"
        f"VERDICT: {u.get('verdict','not computed')}"
    ) if u else "not computed -- no step-0 checkpoints in this grid"
    dup = stats.get("duplicate_cells") or {}
    dup_line = ("" if not dup else
        "\n\n**Duplicate cells.** " + ", ".join(f"`{k}` is bit-identical to `{v}`" for k, v in dup.items())
        + f". This dataset therefore has **{stats.get('n_independent_cells')} independent cells**, not "
          f"{stats['cells']}; any average over all {stats['cells']} double-weights one stimulus set. "
          "Counts elsewhere on this card that say "
          f"{stats['cells']} are counts of files, not of independent measurements.")
    ins = (stats.get("instrument") or {}).get("this_dataset", {})
    instrument_line = (
        f"external group RDM, different cohort, same stimuli : rho = "
        f"{ins.get('cross_session_rsa_median', float('nan')):.3f} (median over cells)\n"
        f"best language model anywhere in this grid            : rsa = "
        f"{ins.get('best_model_rsa', float('nan')):.4f}\n"
        f"model as a fraction of that external benchmark       : "
        f"{ins.get('model_to_external_ratio_median', float('nan')):.1%}\n"
        f"smallest detectable mixed probe (w*)                 : "
        f"{ins.get('w_star_median', float('nan'))}\n"
        f"additive per-stimulus control                        : rsa median "
        f"{ins.get('additive_median_rsa', float('nan')):.4f}, max "
        f"{ins.get('additive_max_rsa', float('nan')):.4f}, significant in "
        f"{ins.get('additive_n_significant','?')}/{ins.get('additive_n_cells','?')} cells"
    ) if ins else "instrument diagnostics not computed for this dataset"
    add_sig = ins.get("additive_n_significant", "?")
    add_n = ins.get("additive_n_cells", "?")
    inc = stats.get("incomplete_families") or {}
    inc_line = ("" if not inc else
        "\n\n**Incomplete families --- do not read these as scale-ladder points.** " + ", ".join(
            f"`{f}` has {n} checkpoint" + ("s" if n != 1 else "") for f, n in inc.items())
        + ". `pythia-6.9b-full`'s single checkpoint is **step 0**, i.e. the untrained initialisation; it "
          "is not the top of the scale ladder, it is the ladder's zero rung.")
    tested = stats["n_cells_tested"]
    return f"""---
license: cc-by-4.0
tags:
  - neuroscience
  - fmri
  - brain-alignment
  - interpretability
  - language-models
  - developmental
configs:
  - config_name: summary_by_checkpoint
    default: true
    data_files: "overall/by_checkpoint.csv"
  - config_name: summary_by_family
    data_files: "overall/summary_by_family.csv"
  - config_name: parc_reference
    data_files: "overall/parc_reference.csv"
  - config_name: untrained_reference
    data_files: "overall/untrained_reference.csv"
  - config_name: untrained_vs_trained
    data_files: "overall/untrained_vs_trained_paired.csv"
  - config_name: brain_alignment
    data_files: "by-model/*/brain_alignment.csv"
  - config_name: interp_mechanistic
    data_files: "by-model/*/interp_mechanistic.csv"
  - config_name: interp_layerwise
    data_files: "by-model/*/interp_layerwise.csv"
  - config_name: localisation_isolation
    data_files: "by-model/*/localisation_isolation.csv"
  - config_name: localisation_onset
    data_files: "by-model/*/localisation_onset.csv"
  - config_name: behaviour
    data_files: "by-model/*/behaviour.csv"
  - config_name: claim_tests
    data_files: "overall/claim_tests.csv"
  - config_name: noise_ceilings
    data_files: "diagnostics/noise_ceilings.csv"
  - config_name: detectability
    data_files: "diagnostics/detectability_wstar.csv"
  - config_name: external_rdm_anchor
    data_files: "diagnostics/external_rdm_anchor.csv"
  - config_name: additive_control
    data_files: "diagnostics/additive_control.csv"
  - config_name: precision_retest
    data_files: "diagnostics/precision_retest.csv"
---

# {dataset} ({s['citation']}) — brain × interpretability × localisation, per model per checkpoint

{s['title']}. Cohort: **{s['cohort']}**; presentation: **{s['modality']}**.
Tasks **{s['tasks']}** × sessions **{s['sessions']}** = **{stats['cells']} task × session cells**,
all of them scored here.{dup_line}{inc_line}

Same schema, same metric and same upstream analysis code as
[`BrainAlign/cdl-devai-results`](https://huggingface.co/datasets/BrainAlign/cdl-devai-results),
so rows are directly comparable across the three developmental datasets.

| axis | what it asks | source tables |
|---|---|---|
| **brain** | does the model's representational geometry match the brain's? | `brain_alignment` |
| **interp** | how is the representation organised internally? | `interp_mechanistic`, `interp_layerwise` |
| **localisation** | are linguistic phenomena isolated into dedicated units? | `localisation_isolation`, `localisation_onset` |
| **behaviour** | does it get the minimal pairs right? | `behaviour` |

**Start with `summary_by_checkpoint`** (the default): one row per model × checkpoint with
all axes side by side. **{stats['checkpoints']} rows, {stats['families']} model families,
{stats['alignment_rows']} alignment rows.**

---

## ⚠️ READ THIS BEFORE USING THE ALIGNMENT NUMBERS

**Nothing here shows that language models align with these brain data, and nothing here
shows that they fail to.** The reasons, measured rather than assumed — and note that the
first one means a masked rebuild may change these numbers substantially:

**1. These RDMs are WHOLE-BRAIN and UNMASKED, and that is a defect, not a design choice.**
Anatomical masking was not applied when these RDMs were built on the GPU cluster, so every
number here is computed over the whole acquired volume rather than over language- or
phonology-responsive cortex. On ds003604 the consequence is measured: the RDMs occupy ~4
effective dimensions of a 72-stimulus space and track whole-brain signal level. **Masked
versions are being rebuilt** (`ROI_SET=language`, `ROI_SET=phonology`, `ROI_SET=all`) and
will be published as separate `-roi*` datasets. Until then, treat every alignment number
here as a whole-brain measurement and do not read it as a claim about the language network.

**1b. The positive control on THIS dataset is one control, and its gate plumbing was
faulty.** The card previously said "the positive control failed", carried over from the
ds003604 battery. That is an over-read here twice over. First, `control/control_summary.csv`
in the predecessor repo contains exactly **one** control — `text_edit_distance` — against
the eight-control battery (duration, intensity, word length, syllables, phonemes, log
frequency, run identity, presentation order) used on ds003604; the stimulus-characteristics
tables needed for the rest were not on the machine that ran it. A single non-significant
edit-distance control does not establish that no stimulus property is recoverable. Second,
the control labels feeding those gates were empty, so the gate outcomes are not
interpretable at all and are being re-plumbed. **No claim on this card depends on the
positive control, and none should be read as supported by it.**

**2. The reference that matters is the untrained one, and it is now measured.** An earlier version of
this card said no random-initialisation baseline existed in this collection. That was wrong: the
Pythia/PolyPythia `step0` branches ARE the initialisation before any optimiser step, **15 independent
random inits were swept across every cell**, and they were sitting in the results unlabelled.
`untrained_reference` and `untrained_vs_trained_paired` report them. Paired within each family, on this
dataset:

```
{untrained_line}
```

Read `frac_of_ceiling` alongside it: the absolute magnitudes are tiny either way, so this is a statement
about sign and consistency, not about a model matching the brain.

**2b. `parc_reference` is a between-model check, not a null, and must be read two-sided.** The PARC
families (`parc-pythia`, `parc-mamba`, `parc-rwkv`, seeds 0--2) are **trained** models --- 160M,
OpenWebText, 4000 steps --- differing only in architecture and seed, so they are a matched-scale seed
reference. Both this collection's upstream notes and the
[`cdl-devai-results`](https://huggingface.co/datasets/BrainAlign/cdl-devai-results) README call them
"pure-noise runs"; that label is wrong and is corrected here. On this dataset **{exceed} of {tested}
family x cell combinations exceed the PARC band by 2 SD and {below} fall below it**. Where those two
counts are comparable, the excursions are a variance artifact rather than evidence of alignment, and the
cell contributes nothing either way.

**2c. The instrument was tested, and it works --- which is what makes the null mean something.**
A null is only interpretable with a detection floor attached, so we measured one (`diagnostics/`).

```
{instrument_line}
```

Two consequences. First, the estimator is **not deaf**: an external group RDM --- a different cohort of
children scanned on the same stimuli --- is recovered by this exact pipeline (z-normalise both, Spearman
on the upper triangle) at the rho above, and a probe mixed at w* is detected in every cell. So "no LM
alignment is detectable here" is a bounded statement about models, not a suspicion about the
measurement. Second, and less comfortably, **a trivial control outscores every language model**: an
additive per-stimulus main-effect RDM (`meanD_i + meanD_j`, carrying no relational structure at all) is
significant by stimulus-label permutation in {add_sig} of {add_n} cells. Whatever these RDMs are
dominated by, it is closer to a per-stimulus offset than to the relational geometry RSA is meant to
compare. `additive_control` reports it per cell; no previously published artifact in this collection
does.

**2c-ii. The scanner-run confound is not what suppresses alignment.** `ds006239/SemLocal` is the only
genuinely run x stimulus **crossed** cell in the whole collection --- the run confound cannot arise there
by design --- so it is the sharpest available test of the "it is the confound" explanation. Expressed as
a ratio to each cell's own external-RDM benchmark (which controls for the very different ceilings),
SemLocal scores **R = 0.106** against a median of **0.046** across the 24 confounded cells (IQR
0.035--0.086, range 0.029--0.199). It is the high end of that distribution but **inside** it, at the
83rd percentile. The clean cell behaves like the dirty ones. Whatever is holding model alignment near
zero here, the scanner-run confound is not it, and that explanation should stop being offered.

**2d. The pipeline is exactly reproducible, except for numerical precision.** Five contrasts were run
on the same family and cells, changing only things that should be no-ops. Ratios are the mean
\|delta rsa\| at the final checkpoint as a fraction of the between-family sd at that cell --- i.e. how
much of the signal a nuisance parameter can move:

| contrast | ratio | max | cells over 0.5 |
|---|---|---|---|
| different GPU (1 vs 2) | **0.000** | 0.000 | 0/26 |
| different GPU + different day (vs the main sweep) | **0.000** | 0.000 | 0/26 |
| batch size 16 vs 4 | 0.0007 | 0.003 | 0/26 |
| batch size 16 vs 32 | 0.0005 | 0.002 | 0/26 |
| **fp32 vs bf16** | **0.316** | **0.954** | **6/26** |

Device and run-to-run results are **bit-identical to 1e-12**, and batch size is negligible. Precision is
the sole exception, and it **grows monotonically with training**: at step 0 the two precisions agree to
2e-4, but by the final checkpoint the disagreement reaches 32% of the entire between-family spread and
exceeds half of it in 6 of 26 cells. This whole grid is fp32, so its numbers are internally consistent
and comparable to each other; do not compare a row here against a bf16 number computed elsewhere, and
treat small between-family differences at trained checkpoints as noise.

**3. Ceilings are low on this dataset, so use `frac_of_ceiling`, not `rsa`.** The
inter-subject noise ceilings are in `noise_ceilings` and joined onto every alignment row.
Best raw rsa anywhere in this grid is **{stats['best_rsa']:.4f}**; as a fraction of the
cell's own ceiling that is **{stats['best_frac_of_ceiling']:.3f}**.

All RDMs are the **`within-run-normalised`** ones. The uncorrected RDMs carry a scanner-run
confound in which run identity predicts brain dissimilarity at ρ = +0.49…+0.87 while no
stimulus property predicts it at all; z-scoring each voxel within run drops that to ≈−0.04.
Do not mix the two.

---

## Why this repo exists

[`{s['prior_repo']}`](https://huggingface.co/datasets/{s['prior_repo']}) {s['prior_gap']}.
The cause is a launcher bug, not a scientific choice: `slurm/run_devai_grid.sh` never
passes `--sessions`, so the runner silently fell back to the ds003604 default
`["ses-5","ses-7","ses-9"]` — sessions this dataset does not have. Sessions and tasks are
derived from the RDM tree here, which is what takes coverage to all {stats['cells']} cells.

## Method

Per (checkpoint × task × session): feed the RDM file's own `stimulus_texts` through the
model, mean-pool the final block's hidden states over tokens, build the model RDM as
`1 − corrcoef` across stimuli, then correlate the upper triangles of the z-normalised
model and brain RDMs. `rsa` is Spearman (headline); `rsa_pearson` and `rsa_kendall` are
also reported, and `frac_of_ceiling` is `rsa / ceiling_lower`.

Checkpoints are subsampled log-uniformly across each family's training trajectory, so a
family's rows trace its development rather than only its final state.

## Sessions and tasks

{s['sessions']} × {s['tasks']}.

`claim_tests` carries the upstream per-family tests (R1 alignment-rises, R2
alignment-vs-mechanistic with a step-partialled control, R5 isolation-vs-mechanistic,
R2b behaviour tests) computed by the same `mechanistic_brain_analysis.py` used for
cdl-devai-results.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(STUDY))
    ap.add_argument("--repo", default=None, help="HF dataset repo id")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rdm-variant", default="within-run-normalised",
                    help="subdirectory of the RDM tree, e.g. roi-language, roi-phonology, roi-all")
    ap.add_argument("--grid-dir", default=None,
                    help="grid directory holding this variant's sweep output")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()

    tag = "" if a.rdm_variant == "within-run-normalised" else "-" + a.rdm_variant.replace("roi-", "roi")
    out = Path(a.out) if a.out else ROOT / "hf_package" / (a.dataset + tag)
    stats = build(a.dataset, out, a.rdm_variant, a.grid_dir)
    (out / "README.md").write_text(readme(a.dataset, stats))
    print(json.dumps(stats, indent=2))
    print("staged", sum(1 for f in out.rglob("*") if f.is_file()), "files ->", out)

    if a.push:
        from huggingface_hub import HfApi
        repo = a.repo or f"BrainAlign/cdl-devai-results-{a.dataset}{tag}"
        token = os.environ.get("HF_TOKEN") or (
            Path("/local/scratch/sas245/hf_cache/token").read_text().strip())
        api = HfApi(token=token)
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=a.private)
        # delete_patterns, or a renamed table lingers on the Hub forever:
        # upload_folder only adds and overwrites. `overall/null_reference.csv`
        # survived its own rename this way, keeping a wrong framing live in the
        # repo after the corrected file was already published beside it.
        api.upload_folder(folder_path=str(out), repo_id=repo, repo_type="dataset",
                          delete_patterns=["overall/*", "by-model/*/*", "diagnostics/*"],
                          commit_message=f"{a.dataset}: {stats['alignment_rows']} alignment rows "
                                         f"across {stats['cells']} cells, "
                                         f"{stats['families']} families")
        print(f"pushed -> https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
