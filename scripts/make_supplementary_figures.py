#!/usr/bin/env python
"""Supplementary figures S1-S8, in the MECO house style.

These go BEYOND the main text's Section 5. Section 5 reports alignment and
behaviour pooled across datasets, model families and child age; every figure
here opens up one axis that pooling closed, or shows a number that is
currently only a table entry:

  S1  alignment by child age band          -- the brain-side developmental axis
  S2  alignment against the noise ceiling  -- what |rho| < 0.03 is small relative to
  S3  the anatomical ROI masks themselves  -- where "auditory"/"motor"/"phonology" are
  S4  alignment under each ROI mask        -- is the domain pattern anatomically specific?
  S5  accuracy vs alignment per checkpoint -- Table 5's rho = -0.641, drawn
  S6  architecture, scale, and seed noise  -- what pooling 29 families hides
  S7  brain- vs model-side specialisation  -- the two localisation axes side by side
  S8  layer depth over training            -- where in the network each domain sits

Every figure writes <stem>.pdf, <stem>.png and <stem>.csv, the last being the
exact table the panel was drawn from.

    python scripts/make_supplementary_figures.py            # all
    python scripts/make_supplementary_figures.py S3 S4      # a subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import figlib as F                                      # noqa: E402
from viz.brainplot import plot_roi_row, roi_mask_img     # noqa: E402
from viz.plot_style import (                            # noqa: E402
    DIVERGING_CMAP, PALETTE_ACL, SEQUENTIAL_CMAP,
    add_panel_label, apply_acl_style, compact_legend, fig_double,
    fig_one_half, plot_with_ci, savefig_pub, style_axes,
)

OUT = REPO / "paper_results" / "figures" / "supplementary"


def _token_axis(ax, label: str = "Cumulative training tokens") -> None:
    """Shared x-axis treatment for every trajectory panel."""
    ax.set_xscale("log")
    ax.axvline(F.BABYLM_TOKENS, color=PALETTE_ACL["mono"], linewidth=0.6,
               linestyle=(0, (4, 2)), zorder=1)
    ax.set_xlabel(label)


def _finish(fig, stem: str, table: pd.DataFrame) -> None:
    path = OUT / stem
    F.savefig_note(path, table)
    written = savefig_pub(fig, path)
    print(f"  wrote {written['pdf'].name}, {written['png'].name}, {stem}.csv "
          f"({len(table)} rows)")


# ───────────────────────────────────────────────────────────────────────────
# S1 -- alignment by child age band
# ───────────────────────────────────────────────────────────────────────────

def fig_s1() -> None:
    """Section 5 pools over child age. Splitting by it asks a question the main
    text cannot: does a model align better with an older or a younger brain?

    Age band is only partly separable from dataset -- ds003604 supplies ages
    5/7/9, ds002236 ages 9/11/11+, ds006239 ages 11/11+ -- so this figure never
    hides which dataset a point came from. Grammar and plausibility exist in
    ds003604 alone, so their age contrast is within-dataset and clean; for
    semantics and phonology, age 9 is the one band two datasets share, and it
    is the only place the two factors can be told apart here.
    """
    d = F.load_alignment_rows()
    d = d[d.task.isin(F.DOMAIN_ORDER) & d.token_bin.notna()]

    binned = F.binned_ci(d, "rsa", ["task", "dataset", "session", "token_bin"])
    binned["tokens"] = F.BIN_CENTRES[binned.token_bin.astype(int)]

    sessions = [s for s in F.SESSION_ORDER if s in set(binned.session)]
    datasets = sorted(binned.dataset.unique())
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    ses_color = {s: cmap(i / max(len(sessions) - 1, 1) * 0.88)
                 for i, s in enumerate(sessions)}
    ds_style = dict(zip(datasets, ["-", "--", (0, (1, 1.2))]))

    fig = plt.figure(figsize=(6.9, 4.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0])

    top = [fig.add_subplot(gs[0, j]) for j in range(4)]
    for j, dom in enumerate(F.DOMAIN_ORDER):
        ax = top[j]
        sub = binned[binned.task == dom]
        for (ds, ses), g in sub.groupby(["dataset", "session"]):
            g = g.sort_values("tokens")
            plot_with_ci(ax, g.tokens, g["mean"], g.ci_lo, g.ci_hi,
                         color=ses_color[ses], marker="o", alpha_fill=0.08,
                         linestyle=ds_style[ds])
        style_axes(ax, title=F.DOMAIN_LABELS[dom], zero_line=True,
                   ylabel=r"Brain-model $\rho$" if j == 0 else None)
        _token_axis(ax, "")
        ax.tick_params(labelsize=6)
    add_panel_label(top[0], "A")

    handles = [Line2D([0], [0], color=ses_color[s], marker="o", markersize=3,
                      label=f"age {F.SESSION_LABELS[s]}") for s in sessions]
    handles += [Line2D([0], [0], color=PALETTE_ACL["mono"],
                       linestyle=ds_style[ds], label=ds) for ds in datasets]
    fig.legend(handles=handles, frameon=False, fontsize=6, ncol=8,
               loc="upper center", bbox_to_anchor=(0.5, 1.035),
               handlelength=1.4, columnspacing=1.0)

    # Row 2: the same numbers collapsed over training, with the dataset each
    # cell came from kept on the axis rather than averaged away.
    hax = fig.add_subplot(gs[1, :])
    cells = (d.groupby(["task", "dataset", "session"])["rsa"]
               .agg(["mean", "size"]).reset_index())
    cols = [(ds, s) for s in sessions for ds in datasets
            if not cells[(cells.dataset == ds) & (cells.session == s)].empty]
    grid = np.full((len(F.DOMAIN_ORDER), len(cols)), np.nan)
    for r, dom in enumerate(F.DOMAIN_ORDER):
        for c, (ds, ses) in enumerate(cols):
            hit = cells[(cells.task == dom) & (cells.dataset == ds)
                        & (cells.session == ses)]
            if not hit.empty:
                grid[r, c] = hit["mean"].iloc[0]
    vmax = np.nanmax(np.abs(grid))
    im = hax.imshow(grid, cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax,
                    aspect="auto")
    hax.set_xticks(range(len(cols)),
                   [f"{F.SESSION_LABELS[s]}\n{ds}" for ds, s in cols],
                   fontsize=6)
    hax.set_yticks(range(len(F.DOMAIN_ORDER)),
                   [F.DOMAIN_LABELS[t] for t in F.DOMAIN_ORDER], fontsize=6.5)
    for r in range(len(F.DOMAIN_ORDER)):
        for c in range(len(cols)):
            v = grid[r, c]
            txt = "--" if np.isnan(v) else f"{v:+.3f}"
            hax.text(c, r, txt, ha="center", va="center", fontsize=6,
                     color="white" if (not np.isnan(v)
                                       and abs(v) > 0.6 * vmax) else "#222222")
    # Separate the age bands visually so "age, then dataset" reads correctly.
    for c in range(1, len(cols)):
        if cols[c][1] != cols[c - 1][1]:
            hax.axvline(c - 0.5, color="white", linewidth=1.6)
    hax.set_title(r"Mean $\rho$ per age band and dataset, "
                  "pooled over training and model family", fontsize=7)
    for spine in hax.spines.values():
        spine.set_visible(False)
    hax.tick_params(length=0)
    cb = fig.colorbar(im, ax=hax, fraction=0.02, pad=0.008)
    cb.ax.tick_params(labelsize=6)
    add_panel_label(hax, "B")

    _finish(fig, "figS1_alignment_by_age_band", binned)


# ───────────────────────────────────────────────────────────────────────────
# S2 -- alignment against the noise ceiling
# ───────────────────────────────────────────────────────────────────────────

def fig_s2() -> None:
    """|rho| < 0.03 sounds like nothing. It is small against a ceiling of ~0.4
    and unremarkable against one of ~0.03; the main text never says which."""
    d = F.load_alignment_rows()
    d = d[d.task.isin(F.DOMAIN_ORDER) & d.token_bin.notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.35),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})

    # (A) the ceilings themselves, per cell, so the denominator is visible.
    ax = axes[0]
    cells = (d.groupby(["dataset", "task", "session"])
               .agg(ceiling=("ceiling_lower", "mean")).reset_index())
    for i, dom in enumerate(F.DOMAIN_ORDER):
        vals = cells[cells.task == dom].ceiling.to_numpy()
        if vals.size == 0:
            continue
        ax.scatter(np.full(vals.size, i) + np.linspace(-0.14, 0.14, vals.size),
                   vals, s=7, color=F.DOMAIN_COLORS[dom], zorder=3,
                   edgecolor="none", alpha=0.85)
        ax.hlines(vals.mean(), i - 0.26, i + 0.26,
                  color=F.DOMAIN_COLORS[dom], linewidth=1.2, zorder=4)
    ax.set_xticks(range(len(F.DOMAIN_ORDER)),
                  [F.DOMAIN_LABELS[t][:5] for t in F.DOMAIN_ORDER])
    style_axes(ax, ylabel="Noise ceiling (lower bound)",
               title="Ceiling per cell")
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "A")

    # (B) raw rho, and (C) the same trajectories as a fraction of that ceiling.
    for k, (col, lab, title) in enumerate([
        ("rsa", r"Brain-model $\rho$", "Raw alignment"),
        ("frac_of_ceiling", r"$\rho$ / noise ceiling",
         "Ceiling-normalised alignment"),
    ]):
        ax = axes[k + 1]
        b = F.binned_ci(d, col, ["task", "token_bin"])
        b["tokens"] = F.BIN_CENTRES[b.token_bin.astype(int)]
        for dom in F.DOMAIN_ORDER:
            s = b[b.task == dom].sort_values("tokens")
            plot_with_ci(ax, s.tokens, s["mean"], s.ci_lo, s.ci_hi,
                         color=F.DOMAIN_COLORS[dom], label=F.DOMAIN_LABELS[dom],
                         marker="o", alpha_fill=0.12)
        style_axes(ax, ylabel=lab, title=title, zero_line=True)
        _token_axis(ax)
        ax.tick_params(labelsize=6)
        add_panel_label(ax, "BC"[k])
    compact_legend(axes[2], ncol=2, loc="lower left")

    out = F.binned_ci(d, "frac_of_ceiling", ["task", "token_bin"])
    out["tokens"] = F.BIN_CENTRES[out.token_bin.astype(int)]
    _finish(fig, "figS2_alignment_vs_ceiling", out)


# ───────────────────────────────────────────────────────────────────────────
# S3 -- the anatomical ROI masks
# ───────────────────────────────────────────────────────────────────────────

def fig_s3() -> None:
    """Where the ROI analyses actually looked, and what was found there.

    The brain rendering lives in src/viz/brainplot.py; that module's docstring
    explains why these are mask definitions tinted by a measured scalar rather
    than per-voxel effect maps.
    """
    align = {m: F.load_package_alignment(p)
             for m, p in F.MASK_PACKAGES.items()}
    align = {m: d[d.task.isin(F.DOMAIN_ORDER)] for m, d in align.items()}

    # The ROI packages do not all cover the same model families (29 whole-brain
    # and auditory, 28 motor and phonology). Comparing marginal means across
    # them would confound the mask with which models happen to be in it, and
    # would then disagree with panel C, which IS paired. So restrict every mask
    # to the cells all four share.
    key = ["family", "model_ref", "task", "session"]
    common = None
    for d in align.values():
        idx = set(map(tuple, d[key].to_numpy()))
        common = idx if common is None else (common & idx)
    align = {m: d[[tuple(r) in common for r in d[key].to_numpy()]]
             for m, d in align.items()}
    n_common = len(common)

    summary = []
    for mask, d in align.items():
        per_ckpt = d.groupby("model_ref")["rsa"].mean()
        m, lo, hi, n = F.mean_ci(per_ckpt.to_numpy())
        summary.append({"mask": mask, "rsa_mean": m, "ci_lo": lo, "ci_hi": hi,
                        "n_checkpoints": n})
    summary = pd.DataFrame(summary).set_index("mask").loc[F.MASK_ORDER]

    rois = ["auditory", "motor", "phonology"]
    fig = plt.figure(figsize=(6.9, 3.3))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.62, 1.0])

    brain_axes = [fig.add_subplot(gs[0, j]) for j in range(len(rois))]
    plot_roi_row(rois, F.MASK_COLORS, fig=fig, axes=brain_axes)
    voxels = {roi: roi_mask_img(roi)[2] for roi in rois}
    add_panel_label(brain_axes[0], "A", dy=0.02)

    # (B) mean alignment under each mask, whole-brain included as reference.
    ax = fig.add_subplot(gs[1, 0])
    for i, mask in enumerate(F.MASK_ORDER):
        r = summary.loc[mask]
        ax.errorbar(i, r.rsa_mean, yerr=[[r.rsa_mean - r.ci_lo],
                                         [r.ci_hi - r.rsa_mean]],
                    marker="o", color=F.MASK_COLORS[mask],
                    ecolor=F.MASK_COLORS[mask], capsize=2, linestyle="none")
    ax.set_xticks(range(len(F.MASK_ORDER)),
                  [m.replace("whole-brain", "whole\nbrain") for m in F.MASK_ORDER])
    style_axes(ax, ylabel=r"Mean $\rho$", zero_line=True,
               title=f"Alignment under each mask\n({n_common:,} matched cells)")
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "B")

    # (C) the paired whole-brain -> ROI change, the actual masking test.
    ax = fig.add_subplot(gs[1, 1])
    roi_cmp = pd.read_csv(REPO / "results" / "roi_comparison.csv")
    roi_cmp["roi"] = roi_cmp.variant.str.replace("roi-", "", regex=False)
    roi_cmp = roi_cmp.set_index("roi").reindex(rois)
    y = np.arange(len(rois))
    for i, roi in enumerate(rois):
        r = roi_cmp.loc[roi]
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=F.MASK_COLORS[roi],
                linewidth=1.1, solid_capstyle="round", zorder=3)
        ax.plot(r.mean_change, i, marker="o", markersize=3.5,
                color=F.MASK_COLORS[roi], zorder=4)
    ax.axvline(0, color=PALETTE_ACL["mono"], linewidth=0.6, linestyle="--")
    ax.set_yticks(y, rois)
    ax.set_ylim(len(rois) - 0.5, -0.5)
    style_axes(ax, xlabel=r"paired $\Delta\rho$ vs whole brain", grid_axis="x",
               title="Effect of masking")
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "C")

    # (D) per-domain, so a null overall is not read as a null everywhere.
    ax = fig.add_subplot(gs[1, 2])
    per_dom = []
    for mask, d in align.items():
        for dom in F.DOMAIN_ORDER:
            per_ckpt = d[d.task == dom].groupby("model_ref")["rsa"].mean()
            m, lo, hi, n = F.mean_ci(per_ckpt.to_numpy())
            per_dom.append({"mask": mask, "task": dom, "mean": m,
                            "ci_lo": lo, "ci_hi": hi, "n_checkpoints": n})
    per_dom = pd.DataFrame(per_dom)
    width = 0.2
    for k, mask in enumerate(F.MASK_ORDER):
        s = (per_dom[per_dom["mask"] == mask]
             .set_index("task").reindex(F.DOMAIN_ORDER))
        x = np.arange(len(F.DOMAIN_ORDER)) + (k - 1.5) * width
        ax.bar(x, s["mean"], width=width * 0.9, color=F.MASK_COLORS[mask],
               label=mask, linewidth=0)
        ax.errorbar(x, s["mean"],
                    yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]],
                    linestyle="none", ecolor="#444444", elinewidth=0.5,
                    capsize=0.8)
    ax.set_xticks(range(len(F.DOMAIN_ORDER)),
                  [F.DOMAIN_LABELS[t][:5] for t in F.DOMAIN_ORDER])
    style_axes(ax, ylabel=r"Mean $\rho$", zero_line=True,
               title="By domain and mask")
    ax.tick_params(labelsize=6)
    compact_legend(ax, ncol=2, loc="upper right")
    add_panel_label(ax, "D")

    out = per_dom.copy()
    out["mask_voxels"] = out["mask"].map(voxels)
    _finish(fig, "figS3_roi_masks_anatomy", out)


# ───────────────────────────────────────────────────────────────────────────
# S4 -- alignment trajectories under each ROI mask
# ───────────────────────────────────────────────────────────────────────────

def fig_s4() -> None:
    """The main text's headline (grammar positive throughout, plausibility
    reversing) is a whole-brain result. If it is language-specific it should
    survive masking to speech cortex and weaken in motor cortex."""
    frames = []
    for mask, pkg in F.MASK_PACKAGES.items():
        d = F.load_package_alignment(pkg)
        d = d[d.task.isin(F.DOMAIN_ORDER) & d.token_bin.notna()].copy()
        d["mask"] = mask
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)

    binned = F.binned_ci(d, "rsa", ["mask", "task", "token_bin"])
    binned["tokens"] = F.BIN_CENTRES[binned.token_bin.astype(int)]

    fig, axes = fig_double(height=2.5, ncols=4, sharey=True)
    for j, dom in enumerate(F.DOMAIN_ORDER):
        ax = axes[j]
        for mask in F.MASK_ORDER:
            s = binned[(binned.task == dom) &
                       (binned["mask"] == mask)].sort_values("tokens")
            if s.empty:
                continue
            plot_with_ci(ax, s.tokens, s["mean"], s.ci_lo, s.ci_hi,
                         color=F.MASK_COLORS[mask], label=mask,
                         marker="o", alpha_fill=0.10,
                         linestyle="-" if mask == "whole-brain" else "--")
        style_axes(ax, title=F.DOMAIN_LABELS[dom], zero_line=True,
                   ylabel=r"Brain-model $\rho$" if j == 0 else None)
        _token_axis(ax)
        ax.tick_params(labelsize=6)
    compact_legend(axes[0], ncol=1, loc="lower left")
    add_panel_label(axes[0], "A")
    _finish(fig, "figS4_roi_trajectories", binned)


# ───────────────────────────────────────────────────────────────────────────
# S5 -- the accuracy/alignment dissociation
# ───────────────────────────────────────────────────────────────────────────

def fig_s5() -> None:
    """Table 5's one large correlation (Plaus, rho = -0.641) is invisible as a
    number among nineteen small ones. Drawn, it is the clearest result in the
    paper: plausibility accuracy and plausibility brain alignment move in
    opposite directions over training.

    Drawn from hf_package/ds003604 (whole-brain), which is one checkpoint set
    larger than the n = 278 the main text reports, so each panel prints its
    own n. Plausibility reproduces at rho = -0.639 against the reported
    -0.641; the three small correlations move around more, as small
    correlations do.
    """
    ck = F.load_package_checkpoints("ds003604")
    beh = F.load_package_behaviour("ds003604")

    per_dom = beh.pivot_table(index=["family", "model_ref"],
                              columns="phenomenon", values="mp_accuracy")
    ck = ck.set_index(["family", "model_ref"])

    rows = []
    for dom in F.DOMAIN_ORDER:
        j = pd.DataFrame({
            "accuracy": per_dom[dom],
            "alignment": ck[f"brain_rsa_{dom}"],
            "tokens": ck["tokens"],
        }).dropna(subset=["accuracy", "alignment"]).reset_index()
        j["task"] = dom
        rows.append(j)
    d = pd.concat(rows, ignore_index=True)

    fig, axes = fig_double(height=2.2, ncols=4, sharey=False)
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    tok = d.tokens.replace(0, np.nan)
    lo, hi = np.log10(tok.min()), np.log10(tok.max())

    for j, dom in enumerate(F.DOMAIN_ORDER):
        ax = axes[j]
        s = d[d.task == dom]
        c = np.where(s.tokens > 0,
                     (np.log10(s.tokens.replace(0, np.nan)) - lo) / (hi - lo), 0.0)
        ax.scatter(s.accuracy, s.alignment, s=6, c=cmap(np.nan_to_num(c)),
                   edgecolor="none", alpha=0.8, zorder=3)
        rho, p = stats.spearmanr(s.accuracy, s.alignment)
        # LOWESS-free trend: a straight fit is all the caption claims.
        b = np.polyfit(s.accuracy, s.alignment, 1)
        xs = np.linspace(s.accuracy.min(), s.accuracy.max(), 50)
        ax.plot(xs, np.polyval(b, xs), color=F.DOMAIN_COLORS[dom],
                linewidth=1.1, zorder=4)
        ax.axvline(0.5, color=PALETTE_ACL["mono"], linewidth=0.6,
                   linestyle=":", zorder=1)
        ax.text(0.04, 0.05, rf"$\rho$ = {rho:+.3f}" + f"\nn = {len(s)}",
                transform=ax.transAxes, fontsize=6.5, va="bottom",
                fontweight="bold" if abs(rho) > 0.5 else "normal")
        style_axes(ax, xlabel="Minimal-pair accuracy",
                   ylabel=r"Brain-model $\rho$" if j == 0 else None,
                   title=F.DOMAIN_LABELS[dom], zero_line=True)
        ax.tick_params(labelsize=6)
        rows.append(None)

    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=matplotlib.colors.Normalize(lo, hi))
    cb = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.008)
    cb.set_label("log$_{10}$ training tokens", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    add_panel_label(axes[0], "A")
    _finish(fig, "figS5_accuracy_vs_alignment", d)


# ───────────────────────────────────────────────────────────────────────────
# S6 -- architecture, scale, and seed noise
# ───────────────────────────────────────────────────────────────────────────

def fig_s6() -> None:
    """Section 5 pools 29 families and reports |rho| < 0.03. Two questions that
    pooling forecloses: does alignment depend on scale or architecture at all,
    and is any of it larger than the spread between seeds of one model?"""
    d = F.load_alignment_rows()
    d = d[d.task.isin(F.DOMAIN_ORDER)].copy()
    d["arch"] = d.family.map(F.architecture_of)

    fig, axes = fig_double(height=2.3, ncols=3)

    # (A) scale ladder: alignment vs parameter count, per domain.
    ax = axes[0]
    scale = F.binned_ci(d[d.params.notna()], "rsa", ["task", "params"])
    for dom in F.DOMAIN_ORDER:
        s = scale[scale.task == dom].sort_values("params")
        if s.empty:
            continue
        plot_with_ci(ax, s.params, s["mean"], s.ci_lo, s.ci_hi,
                     color=F.DOMAIN_COLORS[dom], label=F.DOMAIN_LABELS[dom],
                     marker="o", alpha_fill=0.12)
    ax.set_xscale("log")
    style_axes(ax, xlabel="Parameters", ylabel=r"Mean $\rho$", zero_line=True,
               title="Alignment vs model scale")
    ax.tick_params(labelsize=6)
    compact_legend(ax, ncol=2, loc="upper right")
    add_panel_label(ax, "A")

    # (B) architecture, using the seeded parc-* families where architecture is
    # the only thing that differs.
    ax = axes[1]
    arch_rows = F.binned_ci(d[d.family.str.startswith("parc-")], "rsa",
                            ["arch", "task"])
    archs = sorted(arch_rows.arch.unique())
    width = 0.8 / max(len(archs), 1)
    arch_color = {a: PALETTE_ACL[k]
                  for a, k in zip(archs, ["B1", "B2", "B3", "B5", "B6"])}
    for k, arch in enumerate(archs):
        s = (arch_rows[arch_rows.arch == arch]
             .set_index("task").reindex(F.DOMAIN_ORDER))
        x = np.arange(len(F.DOMAIN_ORDER)) + (k - (len(archs) - 1) / 2) * width
        ax.bar(x, s["mean"], width=width * 0.9, color=arch_color[arch],
               label=arch, linewidth=0)
        ax.errorbar(x, s["mean"],
                    yerr=[s["mean"] - s.ci_lo, s.ci_hi - s["mean"]],
                    linestyle="none", ecolor="#444444", elinewidth=0.5,
                    capsize=0.8)
    ax.set_xticks(range(len(F.DOMAIN_ORDER)),
                  [F.DOMAIN_LABELS[t][:5] for t in F.DOMAIN_ORDER])
    style_axes(ax, ylabel=r"Mean $\rho$", zero_line=True,
               title="Matched-seed architectures")
    ax.tick_params(labelsize=6)
    compact_legend(ax, ncol=3, loc="upper left")
    add_panel_label(ax, "B")

    # (C) the funnel: is any cell's effect bigger than its own seed spread?
    ax = axes[2]
    spread = pd.read_csv(REPO / "paper_results" / "parc" / "parc_seed_spread.csv")
    spread = spread[spread.task.isin(F.DOMAIN_ORDER)]
    for dom in F.DOMAIN_ORDER:
        s = spread[spread.task == dom]
        ax.scatter(s.rsa_sd, s.rsa_mean, s=5, color=F.DOMAIN_COLORS[dom],
                   edgecolor="none", alpha=0.7, label=F.DOMAIN_LABELS[dom],
                   zorder=3)
    lim = np.nanmax(spread.rsa_sd) * 1.05
    xs = np.linspace(0, lim, 20)
    for k in (1, 2):
        ax.plot(xs, k * xs, color=PALETTE_ACL["mono"], linewidth=0.6,
                linestyle="--" if k == 1 else ":", zorder=2)
        ax.plot(xs, -k * xs, color=PALETTE_ACL["mono"], linewidth=0.6,
                linestyle="--" if k == 1 else ":", zorder=2)
    ax.text(lim * 0.98, lim * 2, r"$\pm2$ SD", fontsize=6, ha="right",
            color=PALETTE_ACL["mono"])
    ax.set_xlim(0, lim)
    style_axes(ax, xlabel="Across-seed SD", ylabel=r"Cell mean $\rho$",
               zero_line=True, title="Effect vs seed noise")
    ax.text(0.03, 0.03, "colours as panel A", transform=ax.transAxes,
            fontsize=6, color=PALETTE_ACL["mono"])
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "C")

    _finish(fig, "figS6_scale_architecture_seeds", scale)


# ───────────────────────────────────────────────────────────────────────────
# S7 -- brain-side vs model-side specialisation
# ───────────────────────────────────────────────────────────────────────────

def fig_s7() -> None:
    """The paper compares brains and models on alignment only. Both sides also
    carry a localisation measure computed the same way, so they can be put on
    the same developmental axis: child age for the brain, training tokens for
    the model."""
    brain = F.load_brain_localization()
    brain = brain[brain.phenomenon.isin(F.DOMAIN_ORDER)]
    iso = F.load_package_isolation("ds003604")
    ck = F.load_package_checkpoints("ds003604")
    iso = iso.merge(ck[["family", "model_ref", "tokens", "token_bin"]],
                    on=["family", "model_ref"], how="left")
    iso = iso[iso.token_bin.notna()]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))

    # (A) brain-side concentration by child age.
    ax = axes[0]
    sessions = [s for s in F.SESSION_ORDER if s in set(brain.session)]
    xpos = {s: i for i, s in enumerate(sessions)}
    # Each dataset gets its own linestyle: no dataset covers every age band, so
    # a line joining points across datasets would draw a developmental change
    # that is really a change of scanner and cohort.
    ds_list = sorted(brain.dataset.unique())
    ds_style = dict(zip(ds_list, ["-", "--", (0, (1, 1.2)), (0, (3, 1, 1, 1))]))
    for dom in F.DOMAIN_ORDER:
        sub = brain[brain.phenomenon == dom].copy()
        sub["x"] = sub.session.map(xpos)
        sub = sub.dropna(subset=["x"])
        for ds, g in sub.groupby("dataset"):
            g = g.sort_values("x")
            ax.plot(g.x, g.brain_localization, marker="o", markersize=2.8,
                    color=F.DOMAIN_COLORS[dom], linewidth=1.0,
                    linestyle=ds_style[ds])
    ax.set_xticks(range(len(sessions)),
                  [F.SESSION_LABELS[s] for s in sessions])
    style_axes(ax, xlabel="Child age (years)",
               ylabel="Brain localisation (Gini)",
               title="Brain side, by age")
    ax.tick_params(labelsize=6)
    ax.legend(handles=[Line2D([0], [0], color=PALETTE_ACL["mono"],
                              linestyle=ds_style[ds], label=ds)
                       for ds in ds_list],
              frameon=False, fontsize=6, loc="lower right", handlelength=1.6,
              labelspacing=0.2)
    add_panel_label(ax, "A")

    # (B) model-side concentration by training tokens, same metric family.
    ax = axes[1]
    b = F.binned_ci(iso, "gini", ["phenomenon", "token_bin"])
    b["tokens"] = F.BIN_CENTRES[b.token_bin.astype(int)]
    for dom in F.DOMAIN_ORDER:
        s = b[b.phenomenon == dom].sort_values("tokens")
        plot_with_ci(ax, s.tokens, s["mean"], s.ci_lo, s.ci_hi,
                     color=F.DOMAIN_COLORS[dom], label=F.DOMAIN_LABELS[dom],
                     marker="o", alpha_fill=0.12)
    style_axes(ax, ylabel="Model localisation (Gini)",
               title="Model side, by tokens")
    _token_axis(ax)
    ax.tick_params(labelsize=6)
    compact_legend(ax, ncol=2, loc="lower right")
    add_panel_label(ax, "B")

    # (C) selectivity: how strongly each domain's circuit prefers its own
    # phenomenon over the other three.
    ax = axes[2]
    b2 = F.binned_ci(iso, "selectivity_index", ["phenomenon", "token_bin"])
    b2["tokens"] = F.BIN_CENTRES[b2.token_bin.astype(int)]
    for dom in F.DOMAIN_ORDER:
        s = b2[b2.phenomenon == dom].sort_values("tokens")
        plot_with_ci(ax, s.tokens, s["mean"], s.ci_lo, s.ci_hi,
                     color=F.DOMAIN_COLORS[dom], marker="o", alpha_fill=0.12)
    style_axes(ax, ylabel="Selectivity index",
               title="Model circuit selectivity")
    _token_axis(ax)
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "C")

    out = b.rename(columns={"mean": "model_gini_mean"})
    _finish(fig, "figS7_brain_vs_model_localisation", out)


# ───────────────────────────────────────────────────────────────────────────
# S8 -- depth
# ───────────────────────────────────────────────────────────────────────────

def fig_s8() -> None:
    """Alignment is computed from one pooled representation. Where in the
    network the signal sits, and whether it migrates with training, is a
    separate question the layerwise tables can answer."""
    lay = F.load_package_layerwise("ds003604")
    ck = F.load_package_checkpoints("ds003604")
    lay = lay.merge(ck[["family", "model_ref", "tokens", "token_bin"]],
                    on=["family", "model_ref"], how="left")
    lay = lay[lay.token_bin.notna()].copy()
    # Depth is comparable across families only in relative terms.
    lay["depth"] = lay.groupby("family")["layer"].transform(
        lambda s: s / max(s.max(), 1))

    iso = F.load_package_isolation("ds003604").merge(
        ck[["family", "model_ref", "tokens", "token_bin"]],
        on=["family", "model_ref"], how="left")
    iso = iso[iso.token_bin.notna() & iso.phenomenon.isin(F.DOMAIN_ORDER)]

    fig, axes = fig_one_half(height=2.3, ncols=2)
    fig.set_size_inches(6.9, 2.3)

    # (A) how training reshapes the depth profile of sparsity. This is done
    # WITHIN family (last checkpoint minus first) on purpose: plotting raw Gini
    # against absolute tokens would confound training with model identity,
    # because only the largest Pythias reach the top token bins at all, so the
    # apparent jump past 1e10 tokens would just be "these are different models".
    ax = axes[0]
    nb = 8
    lay["depth_bin"] = pd.cut(lay.depth, np.linspace(0, 1, nb + 1),
                              include_lowest=True, labels=False)
    first_last = (lay.sort_values("step")
                     .groupby(["family", "depth_bin"])["gini"]
                     .agg(first="first", last="last").reset_index())
    first_last["delta"] = first_last["last"] - first_last["first"]
    prof = F.binned_ci(first_last, "delta", ["depth_bin"],
                       unit="family").sort_values("depth_bin")
    xs = (prof.depth_bin.astype(float) + 0.5) / nb
    plot_with_ci(ax, xs, prof["mean"], prof.ci_lo, prof.ci_hi,
                 color=PALETTE_ACL["B3"], marker="o", alpha_fill=0.16)
    style_axes(ax, xlabel="Relative depth (0 = first layer, 1 = last)",
               ylabel=r"$\Delta$ Gini, last $-$ first checkpoint",
               zero_line=True, grid_axis="both",
               title=f"Training sparsifies all but\nthe earliest layers "
                     f"({first_last.family.nunique()} families, paired)")
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "A")

    # (B) where each domain's circuit sits, and whether it moves.
    ax = axes[1]
    iso = iso.copy()
    iso["depth_com"] = iso["layer_com"]
    b = F.binned_ci(iso, "depth_com", ["phenomenon", "token_bin"])
    b["tokens"] = F.BIN_CENTRES[b.token_bin.astype(int)]
    for dom in F.DOMAIN_ORDER:
        s = b[b.phenomenon == dom].sort_values("tokens")
        plot_with_ci(ax, s.tokens, s["mean"], s.ci_lo, s.ci_hi,
                     color=F.DOMAIN_COLORS[dom], label=F.DOMAIN_LABELS[dom],
                     marker="o", alpha_fill=0.12)
    style_axes(ax, ylabel="Circuit centre of mass (relative depth)",
               title="Where each domain sits")
    _token_axis(ax)
    ax.tick_params(labelsize=6)
    compact_legend(ax, ncol=2, loc="best")
    add_panel_label(ax, "B")

    _finish(fig, "figS8_depth", b)


# ───────────────────────────────────────────────────────────────────────────
# S9 -- every neuroimaging study, including the one the main grid never ran
# ───────────────────────────────────────────────────────────────────────────

def fig_s9() -> None:
    """Four studies were collected; the main results use three.

    ds001894 (Lytle et al. 2019, longitudinal phonology, children scanned at
    ~10 and ~12) never ran in the 29-family grid behind
    results/alignment_rows.csv -- results/coverage_matrix.csv records it as
    "not run" for every variant. It DID run in the earlier 15-family devai
    grid, which is where this figure gets it, together with ds002236 and
    ds006239 so all three sit on identical footing.

    The two runs are not poolable. Where they overlap (ds002236 and ds006239,
    nine shared families) their rho values differ by up to 0.067, which is
    wider than the entire effect range the paper reports, so panels A and B
    are drawn wholly inside the devai run and never mixed with Figures S1-S8.
    Panel C is the coverage map that makes the whole situation visible: no
    single alignment run covers all four studies, because ds003604 was never
    run in the devai grid either.
    """
    d = F.load_devai_wrn_alignment()
    d = d[d.token_bin.notna()]

    fig = plt.figure(figsize=(6.9, 5.0))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 0.72, 0.72])

    ds_list = F.DEVAI_WRN_DATASETS
    ds_color = dict(zip(ds_list, [PALETTE_ACL["B1"], PALETTE_ACL["B2"],
                                  PALETTE_ACL["B3"]]))

    # (A) phonology -- the one domain all three of these studies measure, and
    # the only domain ds001894 has at all.
    ax = fig.add_subplot(gs[0, 0])
    phon = F.binned_ci(d[d.task == "Phon"], "rsa", ["dataset", "token_bin"])
    phon["tokens"] = F.BIN_CENTRES[phon.token_bin.astype(int)]
    for ds in ds_list:
        s = phon[phon.dataset == ds].sort_values("tokens")
        plot_with_ci(ax, s.tokens, s["mean"], s.ci_lo, s.ci_hi,
                     color=ds_color[ds], label=ds, marker="o", alpha_fill=0.12)
    style_axes(ax, ylabel=r"Brain-model $\rho$", zero_line=True,
               title="Phonology, by study")
    _token_axis(ax)
    ax.tick_params(labelsize=6)
    compact_legend(ax, ncol=1, loc="best")
    add_panel_label(ax, "A")

    # (B) ds001894's own age structure -- it is longitudinal, and it is the
    # only study here that reaches down to age 7.
    ax = fig.add_subplot(gs[0, 1])
    lyt = d[(d.dataset == "ds001894") & (d.task == "Phon")]
    sess = [s for s in F.SESSION_ORDER if s in set(lyt.session)]
    cmap = plt.get_cmap(SEQUENTIAL_CMAP)
    b = F.binned_ci(lyt, "rsa", ["session", "token_bin"])
    b["tokens"] = F.BIN_CENTRES[b.token_bin.astype(int)]
    for i, ses in enumerate(sess):
        s = b[b.session == ses].sort_values("tokens")
        plot_with_ci(ax, s.tokens, s["mean"], s.ci_lo, s.ci_hi,
                     color=cmap(i / max(len(sess) - 1, 1) * 0.88),
                     label=f"age {F.SESSION_LABELS[ses]}", marker="o",
                     alpha_fill=0.10)
    style_axes(ax, ylabel=r"Brain-model $\rho$", zero_line=True,
               title="ds001894 by age band")
    _token_axis(ax)
    ax.tick_params(labelsize=6)
    compact_legend(ax, ncol=2, loc="best")
    add_panel_label(ax, "B")

    # (C) every domain each study measures, pooled over training.
    ax = fig.add_subplot(gs[0, 2])
    doms = ["Sem", "Phon", "Orth", "SemLocal"]
    marg = F.binned_ci(d, "rsa", ["dataset", "task"])
    width = 0.26
    for k, ds in enumerate(ds_list):
        s = (marg[marg.dataset == ds].set_index("task").reindex(doms))
        x = np.arange(len(doms)) + (k - 1) * width
        ax.bar(x, s["mean"].fillna(0), width=width * 0.9, color=ds_color[ds],
               label=ds, linewidth=0)
        ax.errorbar(x, s["mean"], yerr=[s["mean"] - s.ci_lo,
                                        s.ci_hi - s["mean"]],
                    linestyle="none", ecolor="#444444", elinewidth=0.5,
                    capsize=0.8)
    ax.set_xticks(range(len(doms)), ["Sem", "Phon", "Orth", "SemLoc"])
    style_axes(ax, ylabel=r"Mean $\rho$", zero_line=True,
               title="All domains, by study")
    ax.tick_params(labelsize=6)
    add_panel_label(ax, "C")

    # (D, E) the coverage map. One axes per run, so no study's row has to
    # carry a run label crammed against its own name, and so the two runs are
    # visibly separate objects rather than one table with a line through it.
    main = F.load_alignment_rows()
    all_ds = ["ds001894", "ds002236", "ds003604", "ds006239"]
    all_dom = ["Sem", "Phon", "Gram", "Plaus", "Orth", "SemLocal"]
    runs = [("29-family grid (main results: Figures 1-2, S1-S8)", main),
            ("15-family devai grid (the only run covering ds001894)", d)]

    rows, labels = [], []
    for r, (run_name, tbl) in enumerate(runs):
        ax = fig.add_subplot(gs[1 + r, :])
        counts = np.array([[len(tbl[(tbl.dataset == ds) & (tbl.task == t)])
                            for t in all_dom] for ds in all_ds], dtype=float)
        rows.extend(counts.tolist())
        labels.extend((run_name, ds) for ds in all_ds)
        # Availability is binary here on purpose: shading a cell by its row
        # count would imply that more rows is a better result.
        ax.imshow((counts > 0).astype(float), aspect="auto", vmin=0, vmax=1.6,
                  cmap=matplotlib.colors.ListedColormap(
                      ["#f2f2f2", PALETTE_ACL["B1"]]))
        for i in range(counts.shape[0]):
            for j in range(counts.shape[1]):
                v = counts[i, j]
                ax.text(j, i, "not run" if v == 0 else f"{int(v):,} rows",
                        ha="center", va="center", fontsize=6,
                        color=PALETTE_ACL["mono"] if v == 0 else "white")
        ax.set_yticks(range(len(all_ds)), all_ds, fontsize=6.5)
        ax.set_xticks(range(len(all_dom)),
                      [F.DOMAIN_LABELS[t] for t in all_dom] if r == 1 else [],
                      fontsize=6.5)
        style_axes(ax, grid=False, title=run_name)
        ax.title.set_fontsize(7)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        add_panel_label(ax, "DE"[r])
    grid_counts = np.array(rows)

    out = pd.DataFrame(grid_counts, columns=all_dom)
    out.insert(0, "dataset", [ds for _, ds in labels])
    out.insert(0, "run", [r.replace("\n", " ") for r, _ in labels])
    _finish(fig, "figS9_all_studies_coverage", out)


FIGURES = {
    "S1": fig_s1, "S2": fig_s2, "S3": fig_s3, "S4": fig_s4,
    "S5": fig_s5, "S6": fig_s6, "S7": fig_s7, "S8": fig_s8,
    "S9": fig_s9,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="*", default=[],
                    help=f"figures to build (default: all of {sorted(FIGURES)})")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    global OUT
    if args.out_dir:
        OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    apply_acl_style()
    wanted = [w.upper() for w in args.which] or sorted(FIGURES)
    unknown = [w for w in wanted if w not in FIGURES]
    if unknown:
        ap.error(f"unknown figure(s) {unknown}; known: {sorted(FIGURES)}")

    failed = []
    for name in wanted:
        print(f"[{name}] {FIGURES[name].__doc__.splitlines()[0]}")
        try:
            FIGURES[name]()
        except Exception as exc:                      # keep going, report at end
            failed.append((name, exc))
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n{len(failed)} figure(s) failed: {[n for n, _ in failed]}")
        return 1
    print(f"\nAll {len(wanted)} figure(s) written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
