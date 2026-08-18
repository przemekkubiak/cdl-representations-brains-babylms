#!/usr/bin/env python
"""Publication figures for the DevAI paper — LINE / BAR / HEATMAP only.

Reads the per-checkpoint CSVs from run_devai_grid.py and the join summaries from
mechanistic_brain_analysis.py, and emits six two-column-ready figures plus LaTeX
tables. Every figure is one of three primitives; conditions use a fixed visual
identity and ordering everywhere (see constants below).

  Fig 1  LINE     representation emergence  (mechanistic metric vs tokens)
  Fig 2  LINE     brain alignment           (RSA vs tokens, per session, faceted)
  Fig 3  HEATMAP  layerwise representation  (layer x training stage)
  Fig 4  BAR      mechanistic predictors    (partial corr with alignment | step)
  Fig 5  HEATMAP  isolation: model vs brain (phenomenon x {LM, brain})
  Fig 6  HEATMAP  cross-model generality    (family x phenomenon, Delta vs scale)

Usage:
  python scripts/make_figures.py --families pico-decoder-small beetle-fineweb3-eng ...
  python scripts/make_figures.py --self-test        # synthesize CSVs and render all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Fixed visual identity — do not vary across figures.
# --------------------------------------------------------------------------- #
PHENOMENA = ["Sem", "Phon", "Gram", "Plaus"]           # fixed order everywhere
PHEN_COLORS = {                                         # Okabe-Ito (colourblind-safe)
    "Sem": "#0072B2", "Phon": "#E69F00", "Gram": "#009E73", "Plaus": "#CC79A7",
}
SESSIONS = ["ses-5", "ses-7", "ses-9"]                 # ordered young -> old
SESSION_COLORS = {"ses-5": "#9ecae1", "ses-7": "#4292c6", "ses-9": "#084594"}  # light->dark
METRIC_ORDER = ["per", "hoyer", "cka_to_prev", "condition_number", "norm"]
METRIC_LABELS = {"per": "eff. rank (PER)", "hoyer": "sparsity (Hoyer)",
                 "cka_to_prev": "CKA drift", "condition_number": "cond. number",
                 "norm": "activation norm"}
DIVERGING = "RdBu_r"     # zero-centred (Delta vs baseline)
SEQUENTIAL = "viridis"   # raw magnitudes

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "lines.linewidth": 1.6,
})

# axis membership: scale-axis families are dashed controls; data-axis solid.
SCALE_FAMILIES = {"pico-decoder-tiny", "pico-decoder-small",
                  "pico-decoder-medium", "pico-decoder-large",
                  "pythia-160m-full", "pythia-410m-full"}


def _style(family):
    return dict(linestyle="--", alpha=0.9) if family in SCALE_FAMILIES else dict(linestyle="-")


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def _read(grid, name, fam):
    f = Path(grid) / f"{name}_{fam}.csv"
    return pd.read_csv(f) if f.exists() else None


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig1_representation(grid, families, out):
    """LINE: mechanistic metric vs training tokens (top=PER, bottom=Hoyer)."""
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.0), sharex=True)
    for metric, ax in zip(["per", "hoyer"], axes):
        for fam in families:
            m = _read(grid, "mechanistic", fam)
            if m is None or metric not in m:
                continue
            m = m.sort_values("tokens")
            x = m["tokens"].clip(lower=1)
            ax.plot(x, m[metric], label=fam, **_style(fam))
        ax.set_xscale("log")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[-1].set_xlabel("training tokens (log)")
    axes[0].set_title("Representation emergence")
    axes[0].legend(frameon=False, ncol=1, loc="best")
    _save(fig, out, "fig1_representation_emergence")


def fig2_alignment(grid, families, out):
    """LINE: brain-LM RSA vs tokens, one line per session, small-multiple per family."""
    fams = [f for f in families if _read(grid, "alignment", f) is not None]
    if not fams:
        print("  (fig2 skipped: no alignment CSVs)")
        return
    n = len(fams)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.4), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        a = _read(grid, "alignment", fam)
        agg = a.groupby(["session", "tokens"], as_index=False)["rsa"].mean()
        for ses in SESSIONS:
            s = agg[agg["session"] == ses].sort_values("tokens")
            if len(s):
                ax.plot(s["tokens"].clip(lower=1), s["rsa"], color=SESSION_COLORS[ses], label=ses)
        ax.set_xscale("log")
        ax.set_title(fam)
        ax.set_xlabel("training tokens (log)")
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0][0].set_ylabel("brain–LM RSA")
    axes[0][-1].legend(frameon=False, title="session")
    fig.suptitle("Brain alignment over training", y=1.02)
    _save(fig, out, "fig2_brain_alignment")


def fig3_layerwise(grid, family, out, metric="per", n_bins=8):
    """HEATMAP: layer (rows) x training stage (cols); cells = mechanistic metric."""
    ml = _read(grid, "mechanistic_layer", family)
    if ml is None or metric not in ml:
        print("  (fig3 skipped: no mechanistic_layer CSV)")
        return
    steps = np.sort(ml["step"].unique())
    # bin training into stages by rank so columns are evenly filled
    bins = np.array_split(steps, min(n_bins, len(steps)))
    layers = np.sort(ml["layer"].unique())
    M = np.full((len(layers), len(bins)), np.nan)
    col_lab = []
    for j, b in enumerate(bins):
        sub = ml[ml["step"].isin(b)]
        piv = sub.groupby("layer")[metric].mean()
        for i, L in enumerate(layers):
            if L in piv.index:
                M[i, j] = piv[L]
        col_lab.append(f"{int(b[0])}–{int(b[-1])}")
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    im = ax.imshow(M, aspect="auto", cmap=SEQUENTIAL, origin="lower")
    ax.set_xticks(range(len(col_lab)))
    ax.set_xticklabels(col_lab, rotation=45, ha="right")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([int(L) for L in layers])
    ax.set_xlabel("training stage (steps)")
    ax.set_ylabel("layer")
    ax.set_title(f"Layerwise {METRIC_LABELS.get(metric, metric)} — {family}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out, "fig3_layerwise_representation")


def fig4_predictors(devai, families, out):
    """BAR (horizontal): partial corr(alignment ~ metric | step) per predictor."""
    rows = []
    for fam in families:
        f = Path(devai) / f"devai_summary_{fam}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s = s[s["claim"] == "R2_partial_control_step"]
        for _, r in s.iterrows():
            metric = r["stat"].split("(")[1].split(",")[1].split("|")[0].strip()
            rows.append({"family": fam, "metric": metric, "value": r["value"]})
    if not rows:
        print("  (fig4 skipped: no R2 partial rows)")
        return
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="metric", columns="family", values="value")
    piv = piv.reindex([m for m in METRIC_ORDER if m in piv.index])
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    y = np.arange(len(piv))
    fams = list(piv.columns)
    h = 0.8 / max(1, len(fams))
    for k, fam in enumerate(fams):
        ax.barh(y + k * h, piv[fam].values, height=h, label=fam)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_yticks(y + 0.4 - h / 2)
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in piv.index])
    ax.set_xlabel("partial corr with alignment (| step)")
    ax.set_title("Mechanistic predictors of alignment")
    ax.legend(frameon=False, fontsize=6)
    _save(fig, out, "fig4_mechanistic_predictors")


def fig5_isolation(devai, families, out):
    """HEATMAP: phenomenon (rows, ordered by brain onset) x {LM, brain} isolation."""
    frames = []
    for fam in families:
        f = Path(devai) / f"isolation_comparison_{fam}.csv"
        if f.exists():
            d = pd.read_csv(f); d["family"] = fam; frames.append(d)
    if not frames:
        print("  (fig5 skipped: no isolation_comparison CSVs)")
        return
    d = pd.concat(frames, ignore_index=True)
    agg = d.groupby("phenomenon").mean(numeric_only=True)
    cols = [c for c in ["lm_isolation", "brain_localization"] if c in agg.columns]
    if "onset_age" in agg.columns:
        agg = agg.sort_values("onset_age")
    agg = agg.reindex([p for p in agg.index if p in PHENOMENA])
    M = agg[cols].to_numpy()
    # z-score each column so LM/brain are comparable
    M = (M - np.nanmean(M, 0)) / (np.nanstd(M, 0) + 1e-9)
    fig, ax = plt.subplots(figsize=(2.6, 2.8))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, vmin=-2, vmax=2)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["LM", "brain"][:len(cols)])
    ax.set_yticks(range(len(agg.index)))
    ax.set_yticklabels(list(agg.index))
    ax.set_title("Isolation: model vs brain\n(rows ordered by brain onset)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="isolation (z)")
    _save(fig, out, "fig5_isolation_model_vs_brain")


def fig6_generality(grid, families, out):
    """HEATMAP: family (rows) x phenomenon (cols); cells = peak RSA, Delta vs scale-axis mean."""
    rows = {}
    for fam in families:
        a = _read(grid, "alignment", fam)
        if a is None:
            continue
        peak = a.groupby("task")["rsa"].max()   # peak-over-training, mean sessions already per-row
        rows[fam] = {ph: peak.get(ph, np.nan) for ph in PHENOMENA}
    if not rows:
        print("  (fig6 skipped: no alignment CSVs)")
        return
    df = pd.DataFrame(rows).T[PHENOMENA]
    scale_mask = df.index.isin(SCALE_FAMILIES)
    if scale_mask.any():
        base = df[scale_mask].mean(axis=0)
        df = df - base                          # Delta vs scale-axis control
    M = df.to_numpy()
    vmax = np.nanmax(np.abs(M)) or 1.0
    fig, ax = plt.subplots(figsize=(3.2, 0.5 * len(df) + 1.2))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(PHENOMENA))); ax.set_xticklabels(PHENOMENA)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(list(df.index))
    ax.set_title("Peak alignment (Δ vs scale control)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ RSA")
    _save(fig, out, "fig6_cross_model_generality")


def fig7_ablation(grid, families, out):
    """BAR (horizontal): causal test — brain alignment under intact vs circuit-
    ablated vs random-ablated (T2.1). Conditions in fixed scientific order."""
    frames = []
    for fam in families:
        f = Path(grid) / f"ablation_alignment_{fam}.csv"
        if f.exists():
            d = pd.read_csv(f); d["family"] = fam; frames.append(d)
    if not frames:
        print("  (fig7 skipped: no ablation_alignment CSVs)")
        return
    d = pd.concat(frames, ignore_index=True)
    conds = ["rsa_intact", "rsa_circuit_ablated", "rsa_random_ablated"]
    labels = ["intact", "circuit-ablated", "random-ablated"]  # scientific order
    means = [d[c].mean() for c in conds if c in d]
    errs = [d[c].std() / np.sqrt(max(1, d[c].notna().sum())) for c in conds if c in d]
    y = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.barh(y, means, xerr=errs, color=["#333333", "#CC79A7", "#999999"][:len(means)],
            height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels[:len(means)])
    ax.invert_yaxis()
    ax.set_xlabel("brain–LM RSA")
    ax.set_title("Causal ablation of the localized circuit")
    _save(fig, out, "fig7_causal_ablation")


def fig8_behaviour(grid, families, out):
    """LINE: minimal-pair behavioural accuracy vs tokens, one line per phenomenon,
    small-multiple per family (pairs with Fig1/Fig2, same x-axis) (T2.2)."""
    fams = [f for f in families if (Path(grid) / f"behaviour_{f}.csv").exists()]
    if not fams:
        print("  (fig8 skipped: no behaviour CSVs)")
        return
    fig, axes = plt.subplots(1, len(fams), figsize=(2.4 * len(fams), 2.4),
                             sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        b = pd.read_csv(Path(grid) / f"behaviour_{fam}.csv")
        for ph in PHENOMENA:
            s = b[b["phenomenon"] == ph].groupby("tokens", as_index=False)["mp_accuracy"].mean()
            if len(s):
                ax.plot(s["tokens"].clip(lower=1), s["mp_accuracy"],
                        color=PHEN_COLORS[ph], label=ph)
        ax.axhline(0.5, color="k", linewidth=0.6, linestyle=":")
        ax.set_xscale("log"); ax.set_title(fam); ax.set_xlabel("training tokens (log)")
        ax.grid(alpha=0.25, linewidth=0.4)
    axes[0][0].set_ylabel("minimal-pair accuracy")
    axes[0][-1].legend(frameon=False, title="phenomenon", fontsize=6)
    fig.suptitle("Linguistic behaviour over training", y=1.02)
    _save(fig, out, "fig8_behaviour")


def fig9_robustness(grid, families, out):
    """HEATMAP: alignment-metric robustness (T2.3). rows = RSA variant,
    cols = family, cells = Spearman(alignment, log-tokens) — does the rise survive
    the choice of alignment metric?"""
    variants = [("rsa", "Spearman RSA"), ("rsa_pearson", "Pearson RSA"),
                ("rsa_kendall", "Kendall RSA"), ("encoding_r", "Encoding R")]
    M = np.full((len(variants), len(families)), np.nan)
    from scipy.stats import spearmanr as _sp
    for j, fam in enumerate(families):
        f = Path(grid) / f"alignment_{fam}.csv"
        if not f.exists():
            continue
        a = pd.read_csv(f)
        for i, (col, _) in enumerate(variants):
            if col in a:
                s = a.groupby("tokens", as_index=False)[col].mean().dropna()
                if len(s) >= 3:
                    M[i, j] = _sp(np.log1p(s["tokens"]), s[col])[0]
    keep = ~np.all(np.isnan(M), axis=1)          # drop metrics with no data (e.g. no encoding)
    if not keep.any():
        print("  (fig9 skipped: no multi-metric alignment columns)")
        return
    M = M[keep]
    variants = [v for v, k in zip(variants, keep) if k]
    fig, ax = plt.subplots(figsize=(0.7 * len(families) + 1.5, 2.2))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_yticks(range(len(variants))); ax.set_yticklabels([v[1] for v in variants])
    ax.set_xticks(range(len(families)))
    ax.set_xticklabels(families, rotation=45, ha="right")
    ax.set_title("Alignment rise is metric-robust\n(corr with log-tokens)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out, "fig9_alignment_robustness")


def fig10_cross_dataset(grid_dirs, families, out):
    """HEATMAP: cross-dataset generalisation (Tier-3). rows = dataset, cols =
    phenomenon; cells = peak brain alignment (mean over families). Shows the
    effect is not an artifact of one neuro dataset."""
    frames = []
    for gd in grid_dirs:
        for fam in families:
            f = Path(gd) / f"alignment_{fam}.csv"
            if f.exists():
                frames.append(pd.read_csv(f))
    if not frames:
        print("  (fig10 skipped: no alignment CSVs)")
        return
    d = pd.concat(frames, ignore_index=True)
    if "dataset" not in d.columns:
        d["dataset"] = "ds003604"
    datasets = sorted(d["dataset"].unique())
    if len(datasets) < 2:
        print(f"  (fig10 skipped: only {len(datasets)} dataset — need >=2 for cross-dataset)")
        return
    # peak-over-training alignment per (dataset, phenomenon), mean over families
    peak = (d.groupby(["dataset", "family", "task"])["rsa"].max()
            .groupby(level=[0, 2]).mean().reset_index())
    M = np.full((len(datasets), len(PHENOMENA)), np.nan)
    for i, ds in enumerate(datasets):
        for j, ph in enumerate(PHENOMENA):
            v = peak[(peak["dataset"] == ds) & (peak["task"] == ph)]["rsa"]
            if len(v):
                M[i, j] = v.iloc[0]
    fig, ax = plt.subplots(figsize=(3.2, 0.6 * len(datasets) + 1.2))
    im = ax.imshow(M, aspect="auto", cmap=SEQUENTIAL)
    ax.set_xticks(range(len(PHENOMENA))); ax.set_xticklabels(PHENOMENA)
    ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets)
    ax.set_title("Cross-dataset generalisation (peak RSA)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="peak RSA")
    _save(fig, out, "fig10_cross_dataset")


# --------------------------------------------------------------------------- #
# Tables (LaTeX, booktabs)
# --------------------------------------------------------------------------- #
def tables(grid, devai, families, out):
    # T2: claim tests (concat summaries)
    frames = []
    for fam in families:
        f = Path(devai) / f"devai_summary_{fam}.csv"
        if f.exists():
            d = pd.read_csv(f); d.insert(0, "family", fam); frames.append(d)
    if frames:
        t2 = pd.concat(frames, ignore_index=True)
        (out / "table2_claim_tests.tex").write_text(
            t2.to_latex(index=False, float_format="%.3f", longtable=False))
        print("  wrote table2_claim_tests.tex")
    # T1: model suite (derived from CSV coverage)
    rows = []
    for fam in families:
        m = _read(grid, "mechanistic", fam)
        if m is None:
            continue
        rows.append({"family": fam, "n_checkpoints": m["step"].nunique(),
                     "min_step": int(m["step"].min()), "max_step": int(m["step"].max()),
                     "axis": "scale" if fam in SCALE_FAMILIES else "data"})
    if rows:
        t1 = pd.DataFrame(rows)
        (out / "table1_model_suite.tex").write_text(t1.to_latex(index=False))
        print("  wrote table1_model_suite.tex")


# --------------------------------------------------------------------------- #
def _synthesize(grid, devai, families, DS="ds003604"):
    """Write schema-correct synthetic CSVs so the whole script is testable offline."""
    Path(grid).mkdir(parents=True, exist_ok=True)
    Path(devai).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    steps = [0, 1000, 4000, 16000, 64000, 125000]
    for fi, fam in enumerate(families):
        scale = 0.8 if fam in SCALE_FAMILIES else 1.0
        al, iso, me, mel = [], [], [], []
        for st in steps:
            prog = np.log1p(st) / np.log1p(125000)
            for task in PHENOMENA:
                for si, ses in enumerate(SESSIONS):
                    al.append(dict(dataset=DS, family=fam, model_ref=f"r@{st}", step=st,
                                   tokens=st * 2 + 1, task=task, session=ses,
                                   rsa=0.1 + 0.4 * prog * scale - 0.03 * si + 0.01 * rng.standard_normal(),
                                   encoding_r=0.05 + 0.3 * prog * scale,
                                   n_stim=20))
                iso.append(dict(family=fam, model_ref=f"r@{st}", step=st, phenomenon=task,
                                gini=0.2 + 0.5 * prog * scale + 0.02 * PHENOMENA.index(task),
                                selectivity_index=0.3, mean_overlap_with_others=0.5 - 0.2 * prog))
            me.append(dict(family=fam, model_ref=f"r@{st}", step=st, tokens=st * 2 + 1,
                           norm=1 + st / 1e5, gini=0.3, hoyer=0.2 + 0.5 * prog,
                           per=0.2 + 0.6 * prog * scale, condition_number=50 - 20 * prog,
                           cka_to_prev=0.7 + 0.2 * prog))
            for L in range(6):
                mel.append(dict(family=fam, step=st, layer=L, per=0.2 + 0.6 * prog * (1 - L / 12),
                                gini=0.3, hoyer=0.3, norm=1.0, condition_number=40))
        # multi-metric alignment columns (T2.3) + behaviour + ablation (T2.1/2.2)
        for r in al:
            r["rsa_pearson"] = r["rsa"] * 0.95
            r["rsa_kendall"] = r["rsa"] * 0.8
        beh, aba = [], []
        for st in steps:
            prog = np.log1p(st) / np.log1p(125000)
            for task in PHENOMENA:
                beh.append(dict(family=fam, model_ref=f"r@{st}", step=st, tokens=st * 2 + 1,
                                phenomenon=task, mp_accuracy=0.5 + 0.45 * prog * scale))
            for task in PHENOMENA:
                aba.append(dict(family=fam, step=st, tokens=st * 2 + 1, task=task, session="ses-7",
                                rsa_intact=0.1 + 0.4 * prog * scale,
                                rsa_circuit_ablated=0.1 + 0.15 * prog * scale,
                                rsa_random_ablated=0.1 + 0.37 * prog * scale))
        pd.DataFrame(al).to_csv(f"{grid}/alignment_{fam}.csv", index=False)
        pd.DataFrame(iso).to_csv(f"{grid}/isolation_{fam}.csv", index=False)
        pd.DataFrame(me).to_csv(f"{grid}/mechanistic_{fam}.csv", index=False)
        pd.DataFrame(mel).to_csv(f"{grid}/mechanistic_layer_{fam}.csv", index=False)
        pd.DataFrame(beh).to_csv(f"{grid}/behaviour_{fam}.csv", index=False)
        pd.DataFrame(aba).to_csv(f"{grid}/ablation_alignment_{fam}.csv", index=False)
        # devai summary + isolation comparison
        summ = [dict(claim="R1_alignment_rises", stat="spearman(step,rsa)", value=0.9, p=0.01, n=6)]
        for mtr in METRIC_ORDER:
            summ.append(dict(claim="R2_partial_control_step",
                             stat=f"partial(rsa,{mtr}|step)",
                             value=(0.7 if mtr in ("per", "hoyer") else -0.2) + 0.05 * fi, p=0.04, n=6))
        pd.DataFrame(summ).to_csv(f"{devai}/devai_summary_{fam}.csv", index=False)
        pd.DataFrame([dict(phenomenon=p, lm_isolation=0.4 + 0.03 * i,
                           brain_localization=0.5 - 0.02 * i, lm_onset_step=1000 * (i + 1),
                           onset_age=[5, 7, 9, 9][i])
                      for i, p in enumerate(PHENOMENA)]).to_csv(
            f"{devai}/isolation_comparison_{fam}.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=[
        "pico-decoder-small", "pico-decoder-large", "beetle-humanscale-eng", "beetle-fineweb3-eng"])
    ap.add_argument("--grid-dir", default="data/processed/language_models/devai_grid")
    ap.add_argument("--grid-dirs", nargs="+", default=None,
                    help="Multiple per-dataset grid dirs for the cross-dataset figure (Fig 10)")
    ap.add_argument("--devai-dir", default="data/processed/language_models/devai")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--layer-family", default=None, help="family for the layerwise heatmap (Fig 3)")
    ap.add_argument("--self-test", action="store_true", help="synthesize CSVs and render everything")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        _synthesize(args.grid_dir, args.devai_dir, args.families, DS="ds003604")
        # a second synthetic dataset so the cross-dataset figure (Fig 10) renders
        gd2 = args.grid_dir + "_ds002_synth"
        _synthesize(gd2, args.devai_dir + "_ds002", args.families, DS="ds-second")
        args.grid_dirs = [args.grid_dir, gd2]

    fams = args.families
    layer_fam = args.layer_family or next(
        (f for f in fams if f not in SCALE_FAMILIES), fams[0])

    print("Rendering figures...")
    fig1_representation(args.grid_dir, fams, out)
    fig2_alignment(args.grid_dir, fams, out)
    fig3_layerwise(args.grid_dir, layer_fam, out)
    fig4_predictors(args.devai_dir, fams, out)
    fig5_isolation(args.devai_dir, fams, out)
    fig6_generality(args.grid_dir, fams, out)
    fig7_ablation(args.grid_dir, fams, out)
    fig8_behaviour(args.grid_dir, fams, out)
    fig9_robustness(args.grid_dir, fams, out)
    fig10_cross_dataset(args.grid_dirs or [args.grid_dir], fams, out)
    tables(args.grid_dir, args.devai_dir, fams, out)
    print(f"Done -> {out}/")


if __name__ == "__main__":
    main()
