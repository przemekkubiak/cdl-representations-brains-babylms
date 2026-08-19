#!/usr/bin/env python
"""Publication figures for the DevAI paper — LINE / BAR / HEATMAP only.

Reads the per-checkpoint CSVs from run_devai_grid.py and the join summaries from
mechanistic_brain_analysis.py, and emits the paper figures plus LaTeX tables.
Every figure is one of three primitives; conditions use a fixed visual identity
and ordering everywhere (see constants below).

  Fig 1  LINE     representation emergence  (mechanistic metric vs training)
  Fig 2  LINE     brain alignment           (RSA vs training, per session, faceted)
  Fig 3  HEATMAP  layerwise representation  (layer x training stage, per family)
  Fig 4  BAR      mechanistic predictors    (partial corr with alignment | step)
  Fig 5  HEATMAP  LM phenomenon isolation   (family x phenomenon)
  Fig 6  HEATMAP  cross-model generality    (family x phenomenon, mean RSA)
  Fig 7  BAR      causal ablation           (RSA + behavioural cost, circuit vs random)
  Fig 8  LINE     minimal-pair behaviour    (accuracy vs training, faceted)
  Fig 9  HEATMAP  alignment metric robustness (RSA variant x family)
  Fig 10 HEATMAP  cross-dataset generalisation (needs >=2 neuro datasets)

THREE THINGS THIS SCRIPT IS DELIBERATE ABOUT — do not "simplify" them away:

1. X AXIS.  The grid writer emits a `tokens` column but the checkpoint metadata
   never populated it, so it is 100% NaN for every family in this run.  A log
   axis over NaN silently renders an EMPTY panel with a 10^0-10^1 default range,
   which is what shipped before.  `_xaxis()` therefore probes the data and falls
   back to `step`, and every figure states which axis it used.  Never hard-code
   "tokens" again.

2. FAMILY COVERAGE.  Families default to whatever is actually on disk (union of
   the grid CSVs), not to a hand-typed subset.  Passing `--families` with two
   names is how the previous pass produced a 2-family paper.

3. THE RUN CONFOUND.  Brain RDMs in ds003604 correlate with scanner RUN at rho
   +0.49 to +0.87 in all 12 task x session cells, because each stimulus appears
   in exactly one run.  Brain-LM alignment is ~0 and slightly negative
   throughout.  Every figure that shows brain alignment carries CONFOUND_NOTE
   ON THE CANVAS, and no trend line is ever drawn through alignment data.
   Isolation (fig 5) and mechanistic (fig 1/3) figures are NOT affected and
   carry no such banner.

Usage:
  python scripts/make_figures.py --grid-dir data/processed/language_models/devai_grid/ds003604 \
                                 --devai-dir data/processed/language_models/devai/ds003604 \
                                 --out figures/ds003604
  python scripts/make_figures.py --self-test        # synthesize CSVs and render all
"""

from __future__ import annotations

import argparse
import re
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
METRIC_ORDER = ["per", "hoyer", "cka_to_prev", "condition_number", "norm", "gini"]
METRIC_LABELS = {"per": "eff. rank (PER)", "hoyer": "sparsity (Hoyer)",
                 "cka_to_prev": "CKA drift", "condition_number": "cond. number",
                 "norm": "activation norm", "gini": "Gini"}
DIVERGING = "RdBu_r"     # zero-centred (Delta vs baseline)
SEQUENTIAL = "viridis"   # raw magnitudes

# The whole model suite, in canonical order. Families found on disk that are not
# listed here are appended (alphabetically) rather than dropped.
SUITE_ORDER = [
    "pico-decoder-tiny", "pico-decoder-small", "pico-decoder-medium", "pico-decoder-large",
    "beetle-fineweb3-eng", "beetle-humanscale-eng",
    "babylm-gpt2", "babylm-gpt2-3", "babylm-gpt2-5", "babylm-gpt2-7",
]
FAMILY_COLORS = {
    "pico-decoder-tiny": "#c6dbef", "pico-decoder-small": "#6baed6",
    "pico-decoder-medium": "#2171b5", "pico-decoder-large": "#08306b",
    "beetle-fineweb3-eng": "#E69F00", "beetle-humanscale-eng": "#B35806",
    "babylm-gpt2": "#a1d99b", "babylm-gpt2-3": "#41ab5d",
    "babylm-gpt2-5": "#238b45", "babylm-gpt2-7": "#00441b",
}

# The ds003604 scanner-run confound. Stamped on every brain-alignment figure.
CONFOUND_NOTE = (
    "CAVEAT — ds003604 brain RDMs are confounded by scanner RUN "
    "(brain-RDM/run rho = +0.49 to +0.87 in all 12 task x session cells; each stimulus "
    "appears in exactly one run). Brain-LM alignment is ~0 and slightly negative throughout: "
    "that is a property of the RDMs, not of the models. No developmental trend is claimed "
    "and no trend line is drawn."
)
CONFOUND_COLOR = "#B00020"

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


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _style(family):
    d = dict(color=FAMILY_COLORS.get(family, "#666666"))
    d["linestyle"] = "--" if family in SCALE_FAMILIES else "-"
    return d


def _save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


def _read(grid, name, fam):
    f = Path(grid) / f"{name}_{fam}.csv"
    return pd.read_csv(f) if f.exists() else None


def discover_families(grid, devai=None):
    """Union of every family that has ANY CSV on disk, in canonical suite order.

    The previous pass rendered a 2-family paper because --families was typed by
    hand at invocation time; deriving it from disk makes that failure impossible.
    """
    found = set()
    pats = ["alignment_*.csv", "mechanistic_*.csv", "isolation_*.csv", "behaviour_*.csv"]
    for d in [p for p in (grid, devai) if p]:
        for pat in pats:
            for f in Path(d).glob(pat):
                stem = f.stem
                for pre in ("mechanistic_layer_", "ablation_alignment_", "ablation_behaviour_",
                            "isolation_comparison_", "alignment_", "mechanistic_",
                            "isolation_", "behaviour_"):
                    if stem.startswith(pre):
                        found.add(stem[len(pre):])
                        break
    ordered = [f for f in SUITE_ORDER if f in found]
    ordered += sorted(f for f in found if f not in SUITE_ORDER)
    return ordered


# Resolved once in main() and consulted by every figure.
XCOL, XLABEL = "step", "training step (log)"


def _xaxis(frames, min_frac=0.5):
    """Decide the training axis from the data. Returns (column, axis label).

    `tokens` is preferred when it is actually populated; otherwise we fall back
    to `step` and SAY SO rather than drawing a log axis over NaN/zero.
    """
    tot = ok = 0
    for d in frames:
        if d is None or "tokens" not in d:
            continue
        t = pd.to_numeric(d["tokens"], errors="coerce")
        tot += len(t)
        ok += int((t.notna() & (t > 0)).sum())
    if tot and ok / tot >= min_frac:
        return "tokens", "training tokens (log)"
    return "step", "training step (log)"


def _x(df):
    """x values for a frame, safe for a log axis (step 0 = random init -> 1)."""
    v = pd.to_numeric(df[XCOL], errors="coerce")
    return v.clip(lower=1)


def _init_note(fig, y=-0.005):
    """Log axes cannot show step 0, so it is drawn at x=1. Say so, below the axes —
    never inside the data area, where it reads as an annotation on a data point."""
    fig.text(0.5, y, "step 0 (random init) is drawn at x=1; log axes cannot show zero.",
             ha="center", va="top", fontsize=5.5, color="#777777")


def _no_data(ax, msg="no data"):
    """A panel with nothing to show says so. A blank axis that looks like a plot
    is worse than an absent figure."""
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="#B00020", style="italic")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _banner(fig, y=-0.02, size=6):
    """Put the run-confound caveat ON THE CANVAS of a brain-alignment figure."""
    fig.text(0.5, y, CONFOUND_NOTE, ha="center", va="top", fontsize=size,
             color=CONFOUND_COLOR, wrap=True,
             bbox=dict(boxstyle="round,pad=0.35", fc="#FFF3F3", ec=CONFOUND_COLOR, lw=0.6))


def _grid_axes(n, per=2.35, h=2.35, ncol_max=5, **kw):
    ncol = min(ncol_max, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(per * ncol, h * nrow), squeeze=False, **kw)
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat, nrow, ncol


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig1_representation(grid, families, out):
    """LINE: mechanistic metric vs training (top=PER, bottom=Hoyer).

    Not affected by the run confound — these are LM-internal quantities.
    """
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 4.6), sharex=True)
    drawn = {ax: 0 for ax in axes}
    for metric, ax in zip(["per", "hoyer"], axes):
        for fam in families:
            m = _read(grid, "mechanistic", fam)
            if m is None or metric not in m or m[metric].isna().all():
                continue
            m = m.sort_values(XCOL)
            ax.plot(_x(m), m[metric], label=fam, **_style(fam))
            drawn[ax] += 1
        ax.set_xscale("log")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.25, linewidth=0.4)
        if not drawn[ax]:
            _no_data(ax, f"no {metric} data")
    axes[-1].set_xlabel(XLABEL)
    axes[0].set_title(f"Representation emergence — all {len(families)} families "
                      f"(dashed = scale axis, solid = data axis)", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="center left",
               bbox_to_anchor=(1.0, 0.5), title="family", title_fontsize=7)
    fig.tight_layout()
    _init_note(fig, y=0.0)
    _save(fig, out, "fig1_representation_emergence")


def fig2_alignment(grid, families, out):
    """LINE: brain-LM RSA vs training, one line per session, small-multiple per family.

    Carries the run-confound banner. Deliberately no trend lines, and a y=0
    reference so the reader sees the effect sits on zero.
    """
    have = [(f, _read(grid, "alignment", f)) for f in families]
    have = [(f, a) for f, a in have if a is not None and len(a)]
    if not have:
        print("  (fig2 skipped: no alignment CSVs)")
        return
    fig, axes, nrow, ncol = _grid_axes(len(have), per=2.5, h=2.4, sharey=True)
    vals = pd.concat([a["rsa"] for _, a in have])
    lim = float(np.nanmax(np.abs(vals))) * 1.15 or 0.05
    for k, ((fam, a), ax) in enumerate(zip(have, axes)):
        agg = a.groupby(["session", XCOL], as_index=False)["rsa"].mean()
        n = 0
        for ses in SESSIONS:
            s = agg[agg["session"] == ses].sort_values(XCOL)
            if len(s):
                ax.plot(_x(s), s["rsa"], color=SESSION_COLORS[ses], label=ses,
                        marker="o", markersize=2.0, linewidth=1.1)
                n += 1
        if not n:
            _no_data(ax); ax.set_title(fam); continue
        ax.axhline(0.0, color="k", linewidth=0.8, linestyle="-", zorder=0)
        ax.set_xscale("log")
        ax.set_ylim(-lim, lim)
        ax.set_title(f"{fam}\n({a['step'].nunique()} ckpts)", fontsize=7.5)
        ax.grid(alpha=0.25, linewidth=0.4)
        if k // ncol == nrow - 1:
            ax.set_xlabel(XLABEL)
        if k % ncol == 0:
            ax.set_ylabel("brain–LM RSA")
    axes[0].legend(frameon=False, title="session", fontsize=6, title_fontsize=6, loc="lower left")
    fig.suptitle(f"Brain alignment over training — all {len(have)} families "
                 f"(ds003604; mean over the 4 phenomena; step 0 = init, drawn at x=1)",
                 y=1.0, fontsize=10)
    fig.tight_layout()
    _banner(fig, y=0.0)
    _save(fig, out, "fig2_brain_alignment")


def fig3_layerwise(grid, families, out, metric="per", n_bins=8):
    """HEATMAP: layer (rows) x training stage (cols) per family; cells = mechanistic
    metric. LM-internal — no confound banner."""
    have = [(f, _read(grid, "mechanistic_layer", f)) for f in families]
    have = [(f, m) for f, m in have if m is not None and metric in m and len(m)]
    if not have:
        print("  (fig3 skipped: no mechanistic_layer CSVs)")
        return
    vmin = min(float(np.nanmin(m[metric])) for _, m in have)
    vmax = max(float(np.nanmax(m[metric])) for _, m in have)
    fig, axes, nrow, ncol = _grid_axes(len(have), per=2.35, h=2.2)
    im = None
    for k, ((fam, ml), ax) in enumerate(zip(have, axes)):
        steps = np.sort(ml["step"].unique())
        bins = np.array_split(steps, min(n_bins, len(steps)))
        layers = np.sort(ml["layer"].unique())
        M = np.full((len(layers), len(bins)), np.nan)
        col_lab = []
        for j, b in enumerate(bins):
            piv = ml[ml["step"].isin(b)].groupby("layer")[metric].mean()
            for i, L in enumerate(layers):
                if L in piv.index:
                    M[i, j] = piv[L]
            col_lab.append(f"{int(b[0])}" if len(b) == 1 else f"{int(b[0])}–{int(b[-1])}")
        im = ax.imshow(M, aspect="auto", cmap=SEQUENTIAL, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(col_lab)))
        ax.set_xticklabels(col_lab, rotation=60, ha="right", fontsize=4.5)
        ax.set_yticks(range(0, len(layers), max(1, len(layers) // 6)))
        ax.set_yticklabels([int(layers[i]) for i in range(0, len(layers), max(1, len(layers) // 6))])
        ax.set_title(f"{fam}", fontsize=7.5)
        if k // ncol == nrow - 1:
            ax.set_xlabel("training stage (steps)", fontsize=6.5)
        if k % ncol == 0:
            ax.set_ylabel("layer")
    fig.suptitle(f"Layerwise {METRIC_LABELS.get(metric, metric)} — all {len(have)} families "
                 f"(shared colour scale)", y=1.0, fontsize=10)
    fig.tight_layout()
    if im is not None:
        fig.colorbar(im, ax=list(axes[:len(have)]), fraction=0.02, pad=0.02,
                     label=METRIC_LABELS.get(metric, metric))
    _save(fig, out, "fig3_layerwise_representation")


def fig4_predictors(devai, families, out):
    """BAR (horizontal): partial corr(alignment ~ metric | step) per predictor.

    Brain alignment is on the y-quantity, so the banner applies.
    """
    rows = []
    for fam in families:
        f = Path(devai) / f"devai_summary_{fam}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s = s[s["claim"] == "R2_partial_control_step"]
        for _, r in s.iterrows():
            m = re.search(r"partial\(rsa,\s*([^|)]+)\s*\|", str(r["stat"]))
            if not m:
                continue
            rows.append({"family": fam, "metric": m.group(1).strip(),
                         "value": r["value"], "p": r.get("p", np.nan)})
    if not rows:
        print("  (fig4 skipped: no R2 partial rows)")
        return
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="metric", columns="family", values="value")
    pv = df.pivot_table(index="metric", columns="family", values="p")
    piv = piv.reindex([m for m in METRIC_ORDER if m in piv.index])
    pv = pv.reindex(piv.index)
    fams = [f for f in families if f in piv.columns]
    piv, pv = piv[fams], pv[fams]
    fig, ax = plt.subplots(figsize=(6.4, 0.62 * len(piv) + 2.1))
    y = np.arange(len(piv))
    h = 0.86 / max(1, len(fams))
    n_sig = 0
    for k, fam in enumerate(fams):
        yy = y + k * h
        ax.barh(yy, piv[fam].values, height=h, color=FAMILY_COLORS.get(fam, "#666666"),
                label=fam, edgecolor="white", linewidth=0.2)
        for i, (v, p) in enumerate(zip(piv[fam].values, pv[fam].values)):
            if np.isfinite(v) and np.isfinite(p) and p < 0.05:
                ax.text(v + (0.01 if v >= 0 else -0.01), yy[i], "*", fontsize=7,
                        va="center", ha="left" if v >= 0 else "right")
                n_sig += 1
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_yticks(y + 0.43 - h / 2)
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in piv.index])
    ax.set_xlabel("partial Spearman(RSA, metric | step)")
    n_tot = int(np.isfinite(piv.to_numpy()).sum())
    ax.set_title(f"Mechanistic predictors of brain alignment — all {len(fams)} families\n"
                 f"* = p<0.05 ({n_sig}/{n_tot} tests; chance≈{0.05 * n_tot:.1f}). "
                 f"Signs disagree across families — no consistent predictor.",
                 fontsize=8.5)
    ax.legend(frameon=False, fontsize=6.5, ncol=1, loc="center left",
              bbox_to_anchor=(1.01, 0.5), title="family", title_fontsize=7)
    ax.grid(axis="x", alpha=0.25, linewidth=0.4)
    fig.tight_layout()
    _banner(fig, y=0.0)
    _save(fig, out, "fig4_mechanistic_predictors")


def fig5_isolation(grid, devai, families, out):
    """HEATMAP: family (rows) x phenomenon (cols) for the two isolation measures
    that actually discriminate.

    TWO HONESTY NOTES.

    (a) There is no brain arm. This figure used to be titled "isolation: model vs
    brain, rows ordered by brain onset" and drew a ONE-column heatmap, because
    run_brain_localization.py has never been run for this dataset
    (data/processed/fmri/localization is empty) and isolation_comparison_*.csv
    therefore carries only `lm_isolation`/`lm_onset_step` -- no `brain_localization`,
    no `onset_age`. The missing arm is now drawn as missing.

    (b) Gini -- the quantity `lm_isolation` is built from -- is FLAT across the
    suite: family x phenomenon means span 0.404-0.423, a ~1% range. Z-scoring it
    (as the old version did) turns third-decimal noise into a vivid red/blue map
    that looks like structure and is not. We plot the two measures with real
    spread instead -- selectivity index (0.47-0.58) and the depth centre-of-mass
    of the selective units (0.43-0.71) -- on raw scales, and report the Gini range
    as text rather than as a colour map.
    """
    rows_sel, rows_com, gini_vals = {}, {}, []
    for fam in families:
        d = _read(grid, "isolation", fam)
        if d is None or not len(d):
            continue
        if "selectivity_index" in d:
            g = d.groupby("phenomenon")["selectivity_index"].mean()
            rows_sel[fam] = {ph: g.get(ph, np.nan) for ph in PHENOMENA}
        if "layer_com" in d:
            g = d.groupby("phenomenon")["layer_com"].mean()
            rows_com[fam] = {ph: g.get(ph, np.nan) for ph in PHENOMENA}
        if "gini" in d:
            gini_vals.append(d.groupby("phenomenon")["gini"].mean())
    if not rows_sel and not rows_com:
        print("  (fig5 skipped: no isolation CSVs)")
        return
    gr = ""
    if gini_vals:
        gg = pd.concat(gini_vals)
        gr = (f"Gini (the `lm_isolation` quantity) is flat across the suite: "
              f"family \u00d7 phenomenon means span {gg.min():.3f}\u2013{gg.max():.3f}. "
              f"It is reported as text, not as a colour map, because z-scoring a "
              f"~1% range manufactures structure.")

    panels = []
    if rows_sel:
        panels.append(("selectivity index\n(mean over checkpoints)", rows_sel, "magma"))
    if rows_com:
        panels.append(("layer centre-of-mass\n(0 = first layer, 1 = last)", rows_com, "cividis"))

    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(3.2 * len(panels) + 2.6,
                                                          0.34 * len(families) + 2.6),
                             gridspec_kw={"width_ratios": [1] * len(panels) + [0.6]})
    axes = np.atleast_1d(axes)
    for k, (title, rows, cmap) in enumerate(panels):
        ax = axes[k]
        df = pd.DataFrame(rows).T.reindex([f for f in families if f in rows])[PHENOMENA]
        M = df.to_numpy(dtype=float)
        im = ax.imshow(M, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(PHENOMENA))); ax.set_xticklabels(PHENOMENA)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(list(df.index) if k == 0 else [], fontsize=7)
        lo, hi = np.nanmin(M), np.nanmax(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    rel = (M[i, j] - lo) / (hi - lo + 1e-12)
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                            color="w" if rel < 0.6 else "k")
        ax.set_title(f"({chr(97 + k)}) {title}\nrange {lo:.2f}-{hi:.2f}", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    axb = axes[-1]
    _no_data(axb, "brain\nlocalization\nNOT MEASURED")
    axb.set_title("(c) brain arm", fontsize=8)
    axb.text(0.5, 0.2, "run_brain_localization.py has\nnever been run for ds003604\n"
                       "(data/processed/fmri/localization\nis empty) — so there is no\n"
                       "brain_localization or onset_age\nto compare the LM against.",
             transform=axb.transAxes, ha="center", va="center", fontsize=5.5, color="#777777")
    fig.suptitle(f"Phenomenon isolation in the LM \u2014 all {len(families)} families "
                 f"(NOT affected by the scanner-run confound)", y=1.0, fontsize=10)
    fig.tight_layout()
    if gr:
        fig.text(0.5, -0.01, gr, ha="center", va="top", fontsize=6, color="#555555", wrap=True)
    _save(fig, out, "fig5_isolation_model_vs_brain")


def fig6_generality(grid, families, out):
    """HEATMAP: family (rows) x phenomenon (cols); cells = MEAN brain-LM RSA over
    all checkpoints and sessions.

    Uses the mean, not the peak-over-training the old version used: with alignment
    sitting on zero, a max over ~100 noisy checkpoints is a biased-positive order
    statistic and would manufacture a signal that is not there. The Delta-vs-scale-
    control subtraction is also dropped — 4 of the 10 families ARE the scale set,
    so it subtracted a mean from its own members.
    """
    rows, ns = {}, {}
    for fam in families:
        a = _read(grid, "alignment", fam)
        if a is None or not len(a):
            continue
        g = a.groupby("task")["rsa"].mean()
        rows[fam] = {ph: g.get(ph, np.nan) for ph in PHENOMENA}
        ns[fam] = a["step"].nunique()
    if not rows:
        print("  (fig6 skipped: no alignment CSVs)")
        return
    df = pd.DataFrame(rows).T.reindex([f for f in families if f in rows])[PHENOMENA]
    M = df.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(M))) or 1e-3
    fig, ax = plt.subplots(figsize=(4.6, 0.36 * len(df) + 2.4))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(PHENOMENA))); ax.set_xticklabels(PHENOMENA)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{f}  (n={ns[f]})" for f in df.index], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=5.5,
                        color="k" if abs(M[i, j]) < 0.6 * vmax else "w")
    ax.set_title(f"Mean brain–LM RSA by family x phenomenon\n"
                 f"all {len(df)} families; |RSA| <= {vmax:.3f} everywhere", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean RSA")
    fig.tight_layout()
    _banner(fig, y=0.0)
    _save(fig, out, "fig6_cross_model_generality")


def fig7_ablation(grid, families, out):
    """BAR: causal test (T2.1). Left = brain alignment intact / circuit-ablated /
    random-ablated (confounded, banner applies). Right = the UNCONFOUNDED arm:
    minimal-pair accuracy cost of ablating the localized circuit vs a random set
    of equal size, per family and pooled.
    """
    fa, fb = [], []
    for fam in families:
        pa = Path(grid) / f"ablation_alignment_{fam}.csv"
        pb = Path(grid) / f"ablation_behaviour_{fam}.csv"
        if pa.exists():
            d = pd.read_csv(pa); d["family"] = fam; fa.append(d)
        if pb.exists():
            d = pd.read_csv(pb); d["family"] = fam; fb.append(d)
    if not fa and not fb:
        print("  (fig7 skipped: no ablation CSVs)")
        return
    subset = sorted({*(d["family"].iloc[0] for d in fa), *(d["family"].iloc[0] for d in fb)},
                    key=lambda f: SUITE_ORDER.index(f) if f in SUITE_ORDER else 99)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    # ---- left: alignment under ablation (confounded) ----
    ax = axes[0]
    if fa:
        d = pd.concat(fa, ignore_index=True)
        conds = [("rsa_intact", "intact", "#333333"),
                 ("rsa_circuit_ablated", "circuit-ablated", "#CC79A7"),
                 ("rsa_random_ablated", "random-ablated", "#999999")]
        conds = [c for c in conds if c[0] in d]
        means = [d[c[0]].mean() for c in conds]
        errs = [d[c[0]].std() / np.sqrt(max(1, d[c[0]].notna().sum())) for c in conds]
        y = np.arange(len(conds))
        ax.barh(y, means, xerr=errs, color=[c[2] for c in conds], height=0.6)
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_yticks(y); ax.set_yticklabels([c[1] for c in conds])
        ax.invert_yaxis()
        ax.set_xlabel("brain–LM RSA")
        ax.set_title(f"(a) alignment under ablation\nn={len(d)} cells, "
                     f"{d['family'].nunique()} families", fontsize=8.5)
    else:
        _no_data(ax, "no ablation_alignment CSVs")

    # ---- right: behavioural cost (uncontaminated by the run confound) ----
    ax = axes[1]
    if fb:
        d = pd.concat(fb, ignore_index=True)
        g = d.groupby("family")[["drop_localized", "drop_random"]].mean()
        g = g.reindex([f for f in subset if f in g.index])
        labels = list(g.index) + ["POOLED"]
        loc = list(g["drop_localized"] * 100) + [d["drop_localized"].mean() * 100]
        rnd = list(g["drop_random"] * 100) + [d["drop_random"].mean() * 100]
        y = np.arange(len(labels))
        ax.barh(y - 0.19, loc, height=0.36, color="#CC79A7", label="localized circuit")
        ax.barh(y + 0.19, rnd, height=0.36, color="#999999", label="random (same size)")
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("minimal-pair accuracy cost of ablation (% points)")
        ax.legend(frameon=False, fontsize=6.5, loc="upper left")
        ax.grid(axis="x", alpha=0.25, linewidth=0.4)
        try:
            from scipy.stats import wilcoxon
            dd = d.dropna(subset=["drop_localized", "drop_random"])
            _, p = wilcoxon(dd["drop_localized"], dd["drop_random"])
            frac = float((dd["drop_localized"] > dd["drop_random"]).mean())
            verdict = ("localized > random" if loc[-1] > rnd[-1] else "random > localized")
            sub = (f"pooled {loc[-1]:.2f}% vs {rnd[-1]:.2f}%  ({verdict}), "
                   f"Wilcoxon p={p:.3f}, {frac:.0%} of {len(dd)} cells favour localized")
        except Exception:
            sub = f"pooled {loc[-1]:.2f}% vs {rnd[-1]:.2f}%"
        ax.set_title(f"(b) behavioural cost — NOT confounded by scanner run\n{sub}", fontsize=8.5)
    else:
        _no_data(ax, "no ablation_behaviour CSVs")

    fig.suptitle("Causal ablation of the localized circuit — Tier-2 subset: "
                 + ", ".join(subset) + f"  ({len(subset)} of {len(families)} families ran ablation)",
                 y=1.02, fontsize=9)
    fig.tight_layout()
    if fb:
        fig.text(0.5, 0.005,
                 "NOTE — this is the Tier-3 result and it REVERSES the Tier-2 snapshot. With the "
                 "sparse grid (316 cells) the localized circuit cost 1.13% vs 0.55% random; "
                 "densifying pico-decoder-small from 21 to 126 checkpoints (736 cells) makes the "
                 "pooled mean 1.07% vs 1.53%, i.e. random ablation now costs MORE. Two of four "
                 "families favour the localized circuit and two do not. The causal claim is not "
                 "supported by the full data.",
                 ha="center", va="top", fontsize=5.5, color="#B00020", wrap=True)
    _banner(fig, y=-0.11, size=5.5)
    _save(fig, out, "fig7_causal_ablation")


def fig8_behaviour(grid, families, out):
    """LINE: minimal-pair behavioural accuracy vs training, one line per phenomenon,
    small-multiple per family (T2.2). LM-internal behaviour — no confound banner."""
    have = [(f, _read(grid, "behaviour", f)) for f in families]
    have = [(f, b) for f, b in have if b is not None and len(b)]
    if not have:
        print("  (fig8 skipped: no behaviour CSVs)")
        return
    fig, axes, nrow, ncol = _grid_axes(len(have), per=2.5, h=2.4, sharey=True)
    for k, ((fam, b), ax) in enumerate(zip(have, axes)):
        n = 0
        for ph in PHENOMENA:
            s = b[b["phenomenon"] == ph].groupby(XCOL, as_index=False)["mp_accuracy"].mean()
            s = s.sort_values(XCOL)
            if len(s):
                ax.plot(pd.to_numeric(s[XCOL], errors="coerce").clip(lower=1),
                        s["mp_accuracy"], color=PHEN_COLORS[ph], label=ph,
                        marker="o", markersize=2.0, linewidth=1.1)
                n += 1
        if not n:
            _no_data(ax); ax.set_title(fam); continue
        ax.axhline(0.5, color="k", linewidth=0.6, linestyle=":")
        ax.set_xscale("log")
        ax.set_title(f"{fam}\n({b['step'].nunique()} ckpts)", fontsize=7.5)
        ax.grid(alpha=0.25, linewidth=0.4)
        if k // ncol == nrow - 1:
            ax.set_xlabel(XLABEL)
        if k % ncol == 0:
            ax.set_ylabel("minimal-pair accuracy")
    axes[0].legend(frameon=False, title="phenomenon", fontsize=6, title_fontsize=6)
    fig.suptitle(f"Linguistic behaviour over training — all {len(have)} families "
                 f"(dotted line = chance)", y=1.0, fontsize=10)
    fig.tight_layout()
    _save(fig, out, "fig8_behaviour")


def fig9_robustness(grid, families, out):
    """(a) HEATMAP: rows = RSA variant, cols = family, cells = Spearman(RSA, log training).
    (b) STRIP: the actual RSA band each of those correlations lives in.

    WHY THIS WAS MISSING BEFORE: it aggregated with groupby("tokens"), and tokens is
    100% NaN, so pandas dropped every row, every cell stayed NaN, and the function
    hit its own `keep.any()` early return and printed a skip message. It now
    aggregates on the resolved training axis.

    WHY PANEL (b) EXISTS: several families do show a large, metric-consistent rank
    correlation between RSA and training step. Panel (a) alone would read as a
    developmental effect. It is not one -- the quantity being correlated never
    leaves a +-0.06 band around zero in a set of RDMs that correlate with scanner
    run at rho up to +0.87. Panel (b) puts the magnitude next to the correlation so
    the two cannot be read apart.
    """
    try:
        from scipy.stats import spearmanr as _sp
    except Exception:
        print("  (fig9 skipped: scipy unavailable)")
        return
    variants = [("rsa", "Spearman RSA"), ("rsa_pearson", "Pearson RSA"),
                ("rsa_kendall", "Kendall RSA"), ("encoding_r", "Encoding R")]
    fams = [f for f in families if (Path(grid) / f"alignment_{f}.csv").exists()]
    if not fams:
        print("  (fig9 skipped: no alignment CSVs)")
        return
    M = np.full((len(variants), len(fams)), np.nan)
    P = np.full((len(variants), len(fams)), np.nan)
    band = {}
    for j, fam in enumerate(fams):
        a_ = pd.read_csv(Path(grid) / f"alignment_{fam}.csv")
        for i, (col, _) in enumerate(variants):
            if col not in a_ or a_[col].isna().all():
                continue
            s_ = a_.groupby(XCOL, as_index=False)[col].mean().dropna()
            s_ = s_[pd.to_numeric(s_[XCOL], errors="coerce").notna()]
            if len(s_) >= 3:
                r, pp = _sp(np.log1p(s_[XCOL].astype(float)), s_[col])
                M[i, j], P[i, j] = r, pp
                if col == "rsa":
                    band[fam] = (float(s_[col].min()), float(s_[col].max()))
    keep = ~np.all(np.isnan(M), axis=1)
    if not keep.any():
        print("  (fig9 skipped: no usable alignment metric columns)")
        return
    M, P = M[keep], P[keep]
    variants = [v for v, k in zip(variants, keep) if k]
    n_sig = int(np.nansum(P < 0.05))
    n_tot = int(np.isfinite(P).sum())

    fig, axes = plt.subplots(2, 1, figsize=(0.85 * len(fams) + 2.6, 0.55 * len(variants) + 4.4),
                             gridspec_kw={"height_ratios": [len(variants), 2.1]})
    ax = axes[0]
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_yticks(range(len(variants))); ax.set_yticklabels([v[1] for v in variants])
    ax.set_xticks(range(len(fams))); ax.set_xticklabels([])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                star = "*" if np.isfinite(P[i, j]) and P[i, j] < 0.05 else ""
                ax.text(j, i, f"{M[i, j]:+.2f}{star}", ha="center", va="center",
                        fontsize=5.5, color="k" if abs(M[i, j]) < 0.55 else "w")
    ax.set_title(f"(a) the RSA/training correlation is consistent across metric choices\n"
                 f"Spearman(RSA variant, log {XCOL}); * p<0.05 "
                 f"({n_sig}/{n_tot} tests). The three babylm-gpt2-N families are seed "
                 f"variants of one recipe over 9 checkpoints — effectively one test, not three.",
                 fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman rho")

    # ---- (b) magnitude the correlations actually move over ----
    axm = axes[1]
    lo = np.array([band.get(f, (np.nan, np.nan))[0] for f in fams])
    hi = np.array([band.get(f, (np.nan, np.nan))[1] for f in fams])
    x = np.arange(len(fams))
    axm.vlines(x, lo, hi, color="#4292c6", linewidth=6, alpha=0.85)
    axm.plot(x, lo, "_", color="#084594", markersize=9)
    axm.plot(x, hi, "_", color="#084594", markersize=9)
    axm.axhline(0, color="k", linewidth=0.9)
    axm.set_xticks(x); axm.set_xticklabels(fams, rotation=40, ha="right", fontsize=6.5)
    axm.set_ylabel("brain–LM RSA")
    axm.set_xlim(-0.6, len(fams) - 0.4)
    rng = np.nanmax(np.abs(np.concatenate([lo, hi])))
    axm.set_ylim(-1.25 * rng, 1.25 * rng)
    axm.grid(axis="y", alpha=0.25, linewidth=0.4)
    axm.set_title(f"(b) …but over a band of |RSA| ≤ {rng:.3f} straddling zero. "
                  f"A rank correlation on this is drift, not development.", fontsize=7.5)
    fig.tight_layout()
    _banner(fig, y=0.0)
    _save(fig, out, "fig9_alignment_robustness")


def fig10_cross_dataset(grid_dirs, families, out):
    """HEATMAP: cross-dataset generalisation (Tier-3). rows = dataset, cols =
    phenomenon. Requires >=2 neuro datasets; tier3b was deliberately skipped
    (no second real accession), so this legitimately does not render. We do NOT
    synthesise a second dataset to fill the slot."""
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
        print(f"  (fig10 SKIPPED BY DESIGN: only {len(datasets)} neuro dataset "
              f"({datasets[0]}) — tier3b had no second accession. Not faked.)")
        return
    peak = (d.groupby(["dataset", "family", "task"])["rsa"].mean()
            .groupby(level=[0, 2]).mean().reset_index())
    M = np.full((len(datasets), len(PHENOMENA)), np.nan)
    for i, ds in enumerate(datasets):
        for j, ph in enumerate(PHENOMENA):
            v = peak[(peak["dataset"] == ds) & (peak["task"] == ph)]["rsa"]
            if len(v):
                M[i, j] = v.iloc[0]
    vmax = float(np.nanmax(np.abs(M))) or 1e-3
    fig, ax = plt.subplots(figsize=(3.6, 0.6 * len(datasets) + 2.0))
    im = ax.imshow(M, aspect="auto", cmap=DIVERGING, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(PHENOMENA))); ax.set_xticklabels(PHENOMENA)
    ax.set_yticks(range(len(datasets))); ax.set_yticklabels(datasets)
    ax.set_title("Cross-dataset generalisation (mean RSA)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean RSA")
    fig.tight_layout()
    _banner(fig, y=0.0)
    _save(fig, out, "fig10_cross_dataset")


# --------------------------------------------------------------------------- #
# Tables (LaTeX, booktabs)
# --------------------------------------------------------------------------- #
def _tex(x):
    """Escape a cell/header for LaTeX text mode.

    The identifiers in these tables are full of `_` and `|` (`R2_partial_control_step`,
    `partial(rsa,cka_to_prev|step)`). Emitted raw — as they were — the .tex files do
    not compile. Everything identifier-shaped is escaped and set in \\texttt.
    """
    x = str(x)
    return (x.replace("\\", "/").replace("_", r"\_").replace("|", r"$\mid$")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
             .replace("^", r"\^{}").replace("~", r"\~{}"))


def _tt(x):
    return r"\texttt{" + _tex(x) + "}"


def tables(grid, devai, families, out):
    # ---- T1: model suite — the FULL suite, derived from CSV coverage on disk ----
    rows = []
    for fam in families:
        m = _read(grid, "mechanistic", fam)
        a = _read(grid, "alignment", fam)
        ml = _read(grid, "mechanistic_layer", fam)
        if m is None and a is None:
            continue
        src = m if m is not None else a
        rows.append({
            "family": fam,
            "axis": "scale" if fam in SCALE_FAMILIES else "data",
            "n_ckpt": int(src["step"].nunique()),
            "min_step": int(src["step"].min()),
            "max_step": int(src["step"].max()),
            "n_layers": int(ml["layer"].nunique()) if ml is not None else np.nan,
            "n_align_rows": int(len(a)) if a is not None else 0,
            "ablation": "yes" if (Path(grid) / f"ablation_behaviour_{fam}.csv").exists() else "no",
        })
    if rows:
        t1 = pd.DataFrame(rows)
        n_ck, n_al = int(t1["n_ckpt"].sum()), int(t1["n_align_rows"].sum())
        disp = t1.copy()
        disp["family"] = disp["family"].map(_tt)
        disp["n_layers"] = disp["n_layers"].map(
            lambda v: "--" if not np.isfinite(v) else str(int(v)))
        disp.columns = ["family", "axis", "\\#ckpt", "min step", "max step",
                        "\\#layers", "\\#align rows", "ablation"]
        body = disp.to_latex(index=False, escape=False, na_rep="--",
                             column_format="llrrrrrl",
                             caption=(f"Model suite: all {len(t1)} families, {n_ck} checkpoints, "
                                      f"{n_al} brain-alignment rows (ds003604). "
                                      "\\#ckpt counts distinct training steps present in the grid. "
                                      "The \\texttt{tokens} column emitted by the grid writer is "
                                      "empty for every family (checkpoint metadata carried no token "
                                      "count), so every training axis in the figures is optimiser "
                                      "\\emph{steps}, not tokens. Causal ablation (Fig.~7) ran on the "
                                      "Tier-2 subset of four families only."),
                             label="tab:model_suite")
        (out / "table1_model_suite.tex").write_text(body)
        print(f"  wrote table1_model_suite.tex ({len(t1)} families, {n_ck} ckpts, "
              f"{n_al} alignment rows)")
    else:
        print("  (table1 skipped: no mechanistic/alignment CSVs)")

    # ---- T2: claim tests (concat summaries) ----
    frames = []
    for fam in families:
        f = Path(devai) / f"devai_summary_{fam}.csv"
        if f.exists():
            d = pd.read_csv(f); d.insert(0, "family", fam); frames.append(d)
    if frames:
        t2 = pd.concat(frames, ignore_index=True)
        disp = t2.copy()
        for c in ("family", "claim", "stat"):
            disp[c] = disp[c].map(_tt)
        for c in ("value", "p"):
            disp[c] = disp[c].map(lambda v: "--" if pd.isna(v) else f"{v:.3f}")
        disp["n"] = disp["n"].map(lambda v: "--" if pd.isna(v) else str(int(v)))
        disp.columns = ["family", "claim", "statistic", "value", "$p$", "$n$"]
        (out / "table2_claim_tests.tex").write_text(
            disp.to_latex(index=False, escape=False, longtable=True,
                          column_format="lllrrr",
                          caption=(f"Claim tests R1--R6 for all {t2['family'].nunique()} families "
                                   f"({len(t2)} rows). R1, R2, R2b and the alignment half of R6 "
                                   "involve brain--LM RSA and inherit the ds003604 scanner-run "
                                   "confound (brain-RDM/run $\\rho$ = +0.49 to +0.87 in all 12 "
                                   "task $\\times$ session cells); they are reported for completeness, "
                                   "not as evidence of a developmental effect. R5 (isolation vs "
                                   "mechanistic) and the behavioural half of R6 are unaffected."),
                          label="tab:claim_tests"))
        print(f"  wrote table2_claim_tests.tex ({t2['family'].nunique()} families, {len(t2)} rows)")
    else:
        print("  (table2 skipped: no devai_summary CSVs)")


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
        for r in al:
            r["rsa_pearson"] = r["rsa"] * 0.95
            r["rsa_kendall"] = r["rsa"] * 0.8
        beh, aba, abb = [], [], []
        for st in steps:
            prog = np.log1p(st) / np.log1p(125000)
            for task in PHENOMENA:
                beh.append(dict(family=fam, model_ref=f"r@{st}", step=st, tokens=st * 2 + 1,
                                phenomenon=task, mp_accuracy=0.5 + 0.45 * prog * scale))
                aba.append(dict(family=fam, step=st, tokens=st * 2 + 1, task=task, session="ses-7",
                                rsa_intact=0.1 + 0.4 * prog * scale,
                                rsa_circuit_ablated=0.1 + 0.15 * prog * scale,
                                rsa_random_ablated=0.1 + 0.37 * prog * scale))
                abb.append(dict(family=fam, step=st, tokens=st * 2 + 1, phenomenon=task,
                                acc_none=0.6, acc_localized_ablated=0.55,
                                acc_random_ablated=0.58, drop_localized=0.05 + 0.01 * fi,
                                drop_random=0.02, causal_selectivity=0.03))
        pd.DataFrame(al).to_csv(f"{grid}/alignment_{fam}.csv", index=False)
        pd.DataFrame(iso).to_csv(f"{grid}/isolation_{fam}.csv", index=False)
        pd.DataFrame(me).to_csv(f"{grid}/mechanistic_{fam}.csv", index=False)
        pd.DataFrame(mel).to_csv(f"{grid}/mechanistic_layer_{fam}.csv", index=False)
        pd.DataFrame(beh).to_csv(f"{grid}/behaviour_{fam}.csv", index=False)
        pd.DataFrame(aba).to_csv(f"{grid}/ablation_alignment_{fam}.csv", index=False)
        pd.DataFrame(abb).to_csv(f"{grid}/ablation_behaviour_{fam}.csv", index=False)
        summ = [dict(claim="R1_alignment_rises", stat="spearman(step,rsa)", value=0.9, p=0.01, n=6)]
        for mtr in METRIC_ORDER:
            summ.append(dict(claim="R2_partial_control_step",
                             stat=f"partial(rsa,{mtr}|step)",
                             value=(0.7 if mtr in ("per", "hoyer") else -0.2) + 0.05 * fi,
                             p=0.04, n=6))
        pd.DataFrame(summ).to_csv(f"{devai}/devai_summary_{fam}.csv", index=False)
        pd.DataFrame([dict(phenomenon=p, lm_isolation=0.4 + 0.03 * i,
                           lm_onset_step=1000 * (i + 1))
                      for i, p in enumerate(PHENOMENA)]).to_csv(
            f"{devai}/isolation_comparison_{fam}.csv", index=False)


def main():
    global XCOL, XLABEL
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=None,
                    help="Restrict to these families. DEFAULT: every family found on disk. "
                         "Passing a subset here is how a 2-family paper gets made — be sure.")
    ap.add_argument("--grid-dir", default="data/processed/language_models/devai_grid/ds003604")
    ap.add_argument("--grid-dirs", nargs="+", default=None,
                    help="Multiple per-dataset grid dirs for the cross-dataset figure (Fig 10)")
    ap.add_argument("--devai-dir", default="data/processed/language_models/devai/ds003604")
    ap.add_argument("--out", default="figures/ds003604")
    ap.add_argument("--x-axis", choices=["auto", "tokens", "step"], default="auto",
                    help="Training axis. 'auto' uses tokens only if actually populated.")
    ap.add_argument("--self-test", action="store_true", help="synthesize CSVs and render everything")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        fams0 = args.families or SUITE_ORDER[:4]
        _synthesize(args.grid_dir, args.devai_dir, fams0, DS="ds003604")
        gd2 = args.grid_dir + "_ds002_synth"
        _synthesize(gd2, args.devai_dir + "_ds002", fams0, DS="ds-second")
        args.grid_dirs = [args.grid_dir, gd2]

    fams = args.families or discover_families(args.grid_dir, args.devai_dir)
    if not fams:
        print(f"No families found under {args.grid_dir}. Nothing to render.")
        return

    # Resolve the training axis ONCE, from the data, and say so.
    probe = [_read(args.grid_dir, "alignment", f) for f in fams]
    probe += [_read(args.grid_dir, "mechanistic", f) for f in fams]
    if args.x_axis == "auto":
        XCOL, XLABEL = _xaxis(probe)
    else:
        XCOL = args.x_axis
        XLABEL = "training tokens (log)" if XCOL == "tokens" else "training step (log)"
    if XCOL == "step":
        print("  NOTE: `tokens` is unpopulated in these CSVs — plotting against optimiser "
              "`step` instead. (A log axis over NaN renders an empty panel, which is the "
              "bug this replaces.)")

    print(f"Families ({len(fams)}): {', '.join(fams)}")
    print(f"Training axis: {XCOL}")
    print("Rendering figures...")
    fig1_representation(args.grid_dir, fams, out)
    fig2_alignment(args.grid_dir, fams, out)
    fig3_layerwise(args.grid_dir, fams, out)
    fig4_predictors(args.devai_dir, fams, out)
    fig5_isolation(args.grid_dir, args.devai_dir, fams, out)
    fig6_generality(args.grid_dir, fams, out)
    fig7_ablation(args.grid_dir, fams, out)
    fig8_behaviour(args.grid_dir, fams, out)
    fig9_robustness(args.grid_dir, fams, out)
    fig10_cross_dataset(args.grid_dirs or [args.grid_dir], fams, out)
    tables(args.grid_dir, args.devai_dir, fams, out)
    print(f"Done -> {out}/")


if __name__ == "__main__":
    main()
