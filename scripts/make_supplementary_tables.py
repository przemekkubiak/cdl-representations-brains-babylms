#!/usr/bin/env python
"""Publishable LaTeX tables T1-T8, from the same data as figures S1-S9.

Each table is written twice: `<name>.tex`, a booktabs `table` environment
ready to \\input, and `<name>.csv`, the same numbers for checking. Nothing here
invents a number -- every cell is recomputed from the repo's result tables by
the loaders in scripts/figlib.py, on the same checkpoint-level CI convention
the figures use.

T1 and T2 are recomputations of the main text's Table 2 and Table 5, included
so the supplementary material states plainly where a recomputation agrees with
the submitted numbers and where it does not. The rest are new.

    python scripts/make_supplementary_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import figlib as F  # noqa: E402

OUT = REPO / "paper_results" / "tables"

#: Main-text Table 2, as submitted, so T1 can print the difference rather than
#: leaving a reader to diff two tables by eye.
PAPER_TABLE2 = {
    "Gram":  dict(pos=15, neg=0, zero=0,  first=+0.0169, last=+0.0115),
    "Plaus": dict(pos=4,  neg=9, zero=2,  first=+0.0152, last=-0.0076),
    "Sem":   dict(pos=7,  neg=3, zero=5,  first=+0.0008, last=-0.0055),
    "Phon":  dict(pos=0,  neg=4, zero=11, first=-0.0064, last=+0.0044),
}


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def write(name: str, df: pd.DataFrame, caption: str, label: str,
          column_format: str | None = None, note: str | None = None,
          escape: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False)
    body = df.to_latex(index=False, escape=escape, column_format=column_format
                       or "l" * df.shape[1], na_rep="--")
    tex = ["\\begin{table}[t]", "\\centering", "\\small", body.rstrip()]
    if note:
        tex.append("\\vspace{2pt}")
        tex.append("\\begin{minipage}{\\linewidth}\\footnotesize " + note
                   + "\\end{minipage}")
    tex += [f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(tex) + "\n")
    print(f"  {name}.tex / .csv  ({len(df)} rows)")


# ── T1: Table 2 recomputed ─────────────────────────────────────────────────

def t1_alignment_by_domain() -> pd.DataFrame:
    d = F.load_alignment_rows()
    d = d[d.task.isin(F.DOMAIN_ORDER) & d.token_bin.notna()]
    b = F.binned_ci(d, "rsa", ["task", "token_bin"])
    rows = []
    for dom in F.DOMAIN_ORDER:
        s = b[b.task == dom].sort_values("token_bin")
        pos, neg = int((s.ci_lo > 0).sum()), int((s.ci_hi < 0).sum())
        p = PAPER_TABLE2[dom]
        rows.append({
            "Domain": dom,
            "Occupied bins": len(s),
            "CI $>0$": pos,
            "CI $<0$": neg,
            "CI $\\ni 0$": len(s) - pos - neg,
            "First bin": f"{s['mean'].iloc[0]:+.4f}",
            "Last bin": f"{s['mean'].iloc[-1]:+.4f}",
            "Table 2 (submitted)": f"{p['pos']}/{p['neg']}/{p['zero']}",
            "Agrees": "yes" if (pos, neg) == (p["pos"], p["neg"]) else "no",
        })
    return pd.DataFrame(rows)


# ── T2: Table 5 recomputed ─────────────────────────────────────────────────

def t2_measure_correlations() -> pd.DataFrame:
    ck = F.load_package_checkpoints("ds003604").set_index(["family", "model_ref"])
    beh = (F.load_package_behaviour("ds003604")
           .pivot_table(index=["family", "model_ref"], columns="phenomenon",
                        values="mp_accuracy"))
    iso = F.load_package_isolation("ds003604")
    sel = iso.pivot_table(index=["family", "model_ref"], columns="phenomenon",
                          values="selectivity_index")

    pairs = [("Sparsity vs.\\ accuracy", "gini", "acc"),
             ("Sparsity vs.\\ alignment", "gini", "align"),
             ("Selectivity vs.\\ accuracy", "sel", "acc"),
             ("Selectivity vs.\\ alignment", "sel", "align"),
             ("Accuracy vs.\\ alignment", "acc", "align")]
    rows = []
    for label, a, b in pairs:
        row_r, row_p = {"Pair": label, "Stat": "$\\rho$"}, {"Pair": "", "Stat": "$p$"}
        for dom in F.DOMAIN_ORDER:
            cols = {"gini": ck["interp_gini"], "sel": sel[dom],
                    "acc": beh[dom], "align": ck[f"brain_rsa_{dom}"]}
            j = pd.DataFrame({"x": cols[a], "y": cols[b]}).dropna()
            r, p = stats.spearmanr(j.x, j.y)
            row_r[dom] = (f"\\textbf{{{r:+.3f}}}" if abs(r) > 0.5 else f"{r:+.3f}")
            row_p[dom] = f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"
        rows += [row_r, row_p]
    return pd.DataFrame(rows)


# ── T3: domain x ROI mask ──────────────────────────────────────────────────

def t3_domain_by_mask() -> pd.DataFrame:
    align, key = {}, ["family", "model_ref", "task", "session"]
    for mask, pkg in F.MASK_PACKAGES.items():
        d = F.load_package_alignment(pkg)
        align[mask] = d[d.task.isin(F.DOMAIN_ORDER)]
    common = None
    for d in align.values():
        idx = set(map(tuple, d[key].to_numpy()))
        common = idx if common is None else common & idx
    rows = []
    for dom in F.DOMAIN_ORDER:
        row = {"Domain": dom}
        for mask in F.MASK_ORDER:
            d = align[mask]
            d = d[[tuple(r) in common for r in d[key].to_numpy()]]
            per = d[d.task == dom].groupby("model_ref")["rsa"].mean()
            m, lo, hi, n = F.mean_ci(per.to_numpy())
            row[mask] = f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


# ── T4: architecture at matched seeds ──────────────────────────────────────

def t4_architecture() -> pd.DataFrame:
    d = F.load_alignment_rows()
    d = d[d.task.isin(F.DOMAIN_ORDER) & d.family.str.startswith("parc-")].copy()
    d["arch"] = d.family.map(F.architecture_of)
    b = F.binned_ci(d, "rsa", ["arch", "task"])
    rows = []
    for dom in F.DOMAIN_ORDER:
        row = {"Domain": dom}
        for arch in sorted(b.arch.unique()):
            s = b[(b.arch == arch) & (b.task == dom)]
            if s.empty:
                row[arch] = "--"
                continue
            r = s.iloc[0]
            row[arch] = f"{r['mean']:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


# ── T5: age band x dataset ─────────────────────────────────────────────────

def t5_age_by_dataset() -> pd.DataFrame:
    d = F.load_alignment_rows()
    d = d[d.task.isin(F.DOMAIN_ORDER)]
    b = F.binned_ci(d, "rsa", ["task", "dataset", "session"])
    rows = []
    for _, r in b.iterrows():
        rows.append({"Domain": r.task, "Dataset": r.dataset,
                     "Age band": F.SESSION_LABELS.get(r.session, r.session),
                     "Mean $\\rho$": f"{r['mean']:+.4f}",
                     "95\\% CI": f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]",
                     "$n$ ckpt": int(r.n)})
    order = {t: i for i, t in enumerate(F.DOMAIN_ORDER)}
    out = pd.DataFrame(rows)
    return (out.assign(_o=out.Domain.map(order))
               .sort_values(["_o", "Dataset", "Age band"]).drop(columns="_o"))


# ── T6: circuit depth ──────────────────────────────────────────────────────

def t6_circuit_depth() -> pd.DataFrame:
    ck = F.load_package_checkpoints("ds003604")
    iso = F.load_package_isolation("ds003604").merge(
        ck[["family", "model_ref", "tokens", "token_bin"]],
        on=["family", "model_ref"], how="left")
    iso = iso[iso.token_bin.notna() & iso.phenomenon.isin(F.DOMAIN_ORDER)]
    b = F.binned_ci(iso, "layer_com", ["phenomenon", "token_bin"])
    rows = []
    for dom in F.DOMAIN_ORDER:
        s = b[b.phenomenon == dom].sort_values("token_bin")
        f_, l_ = s.iloc[0], s.iloc[-1]
        rows.append({
            "Domain": dom,
            "First bin": f"{f_['mean']:.3f} [{f_.ci_lo:.3f}, {f_.ci_hi:.3f}]",
            "Last bin": f"{l_['mean']:.3f} [{l_.ci_lo:.3f}, {l_.ci_hi:.3f}]",
            "$\\Delta$": f"{l_['mean'] - f_['mean']:+.3f}",
        })
    return pd.DataFrame(rows)


# ── T7: study coverage ─────────────────────────────────────────────────────

def t7_coverage() -> pd.DataFrame:
    main = F.load_alignment_rows()
    devai = F.load_devai_wrn_alignment()
    doms = ["Sem", "Phon", "Gram", "Plaus", "Orth", "SemLocal"]
    rows = []
    for run, tbl in [("29-family grid", main), ("15-family devai grid", devai)]:
        for ds in ["ds001894", "ds002236", "ds003604", "ds006239"]:
            row = {"Run": run, "Study": ds}
            for t in doms:
                n = len(tbl[(tbl.dataset == ds) & (tbl.task == t)])
                row[t] = f"{n:,}" if n else "--"
            rows.append(row)
    return pd.DataFrame(rows)


# ── T8: noise ceilings ─────────────────────────────────────────────────────

def t8_ceilings() -> pd.DataFrame:
    d = F.load_alignment_rows()
    d = d[d.task.isin(F.DOMAIN_ORDER)]
    cells = (d.groupby(["task", "dataset", "session"])
               .agg(ceiling=("ceiling_lower", "mean"),
                    rsa=("rsa", "mean")).reset_index())
    rows = []
    for dom in F.DOMAIN_ORDER:
        s = cells[cells.task == dom]
        rows.append({
            "Domain": dom,
            "Cells": len(s),
            "Ceiling (mean)": f"{s.ceiling.mean():.3f}",
            "Ceiling (range)": f"{s.ceiling.min():.3f}--{s.ceiling.max():.3f}",
            "Mean $\\rho$": f"{s.rsa.mean():+.4f}",
            "$\\rho$ / ceiling": f"{s.rsa.mean() / s.ceiling.mean():+.4f}",
        })
    return pd.DataFrame(rows)


TABLES = [
    ("T1_alignment_by_domain", t1_alignment_by_domain,
     "Token-binned brain--model alignment by linguistic domain, recomputed from "
     "\\texttt{results/alignment\\_rows.csv} with the checkpoint as the unit of "
     "the confidence interval. The last two columns compare against Table~2 as "
     "submitted.", "tab:t1",
     "Bins are the 16 log-spaced cumulative-token bins of the main text; 15 are "
     "occupied. Plausibility reproduces exactly and grammar to within one bin, "
     "but phonology does not: the submitted row reports no bin with a positive "
     "zero-excluding interval, and this recomputation finds several. No subset "
     "of the three datasets reproduces the submitted row."),
    ("T2_measure_correlations", t2_measure_correlations,
     "Spearman correlations between checkpoint-level measures, by domain, "
     "recomputed from \\texttt{hf\\_package/ds003604}. Bold exceeds "
     "$|\\rho| = 0.5$.", "tab:t2",
     "Checkpoints within a family form a training trajectory rather than "
     "independent draws, so these are descriptive. Plausibility accuracy "
     "against plausibility alignment is the only entry of any magnitude, and it "
     "reproduces the submitted $\\rho = -0.641$."),
    ("T3_domain_by_mask", t3_domain_by_mask,
     "Alignment by domain under each anatomical mask (ds003604), restricted to "
     "the cells all four mask conditions share. Mean $\\rho$ with 95\\% CI.",
     "tab:t3",
     "Masks are AAL(SPM12) region sets: auditory = Heschl's gyrus and superior "
     "temporal gyrus; motor = precentral gyrus; phonology = their union. "
     "Grammar falls monotonically as the mask narrows; the other three domains "
     "do not move."),
    ("T4_architecture", t4_architecture,
     "Alignment by architecture at matched seeds (the \\texttt{parc-*} "
     "families, three seeds each). Mean $\\rho$ with 95\\% CI.", "tab:t4",
     "These families differ in architecture and seed only, so the grammar "
     "column is the cleanest architecture contrast available here."),
    ("T5_age_by_dataset", t5_age_by_dataset,
     "Alignment by child age band, within dataset. Age band and dataset are "
     "only partly separable, so both are shown.", "tab:t5",
     "Grammar and plausibility exist in ds003604 alone, so their age contrast "
     "is within-dataset. Age 9 is the one band two datasets share."),
    ("T6_circuit_depth", t6_circuit_depth,
     "Model circuit centre of mass in relative network depth, first versus "
     "last occupied token bin.", "tab:t6",
     "0 is the first layer and 1 the last. Three domains migrate deeper over "
     "training; phonology does not move."),
    ("T7_coverage", t7_coverage,
     "Alignment rows available per neuroimaging study and domain, in each of "
     "the two runs. No single run covers all four studies.", "tab:t7",
     "ds001894 is recorded as not run for every variant of the 29-family grid "
     "in \\texttt{results/coverage\\_matrix.csv}. The two runs share three "
     "studies and nine model families and disagree by up to 0.067 where they "
     "overlap, so they are not pooled."),
    ("T8_ceilings", t8_ceilings,
     "Noise ceilings per domain, and observed alignment as a fraction of them.",
     "tab:t8",
     "The ceiling is the lower bound of the split-half subject reliability of "
     "the brain RDMs. Grammar and plausibility are measured against ceilings "
     "around 0.85, semantics and phonology against roughly 0.55."),
]


def main() -> int:
    print(f"Writing tables to {OUT}")
    for name, fn, caption, label, note in TABLES:
        write(name, fn(), caption, label, note=note)
    print(f"\n{len(TABLES)} tables written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
