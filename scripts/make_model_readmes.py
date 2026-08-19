#!/usr/bin/env python
"""Write a genuinely readable README for each model directory.

Each page leads with the run-confound warning -- a clean per-model layout makes the
confounded brain numbers MORE likely to be taken at face value, so the warning has to
travel with the data rather than living only at the repo root.
"""
from pathlib import Path

import numpy as np
import pandas as pd

STAGE = Path(__file__).resolve().parents[1] / "hf_results_staging"

INFO = {
 "pico-decoder-tiny":     ("pico-lm/pico-decoder-tiny", "pico decoder, smallest of the scale ladder", "dense commit-history checkpoints"),
 "pico-decoder-small":    ("pico-lm/pico-decoder-small", "pico decoder ~65M (12 layers, width 384)", "dense checkpoints -- the full 126-checkpoint trajectory, the densest in this release"),
 "pico-decoder-medium":   ("pico-lm/pico-decoder-medium", "pico decoder, medium of the scale ladder", "dense commit-history checkpoints"),
 "pico-decoder-large":    ("pico-lm/pico-decoder-large", "pico decoder ~500M, largest of the scale ladder", "dense commit-history checkpoints"),
 "beetle-humanscale-eng": ("Beetle-HumanScale/beetle-monolingual-humanscale-eng", "English at a developmentally plausible ('human-scale') data budget", "35 step branches"),
 "beetle-fineweb3-eng":   ("Beetle-FineWeb3-24B/beetle-monolingual-fineweb3-eng", "English at a web-scale budget (FineWeb, 24B tokens)", "19 step branches"),
 "babylm-gpt2-3":         ("BrainAlign/gpt2-babylm-3", "GPT-2 on BabyLM, smallest data budget", "9 epoch checkpoints"),
 "babylm-gpt2-5":         ("BrainAlign/gpt2-babylm-5", "GPT-2 on BabyLM, mid data budget", "9 epoch checkpoints"),
 "babylm-gpt2-7":         ("BrainAlign/gpt2-babylm-7", "GPT-2 on BabyLM, larger data budget", "9 epoch checkpoints"),
 "babylm-gpt2":           ("BrainAlign/gpt2-babylm-9", "GPT-2 on BabyLM, child-scale data (developmentally honest)", "9 epoch checkpoints"),
}

WARNING = """> ## ⚠️ The brain alignment numbers on this page are confounded
>
> `brain_alignment.csv`, every `brain_*` column in `checkpoints.csv`, and
> `ablation_alignment.csv` are **not interpretable as a result about this model.**
>
> In ds003604 each stimulus appears in exactly one scanner run, so run membership is
> perfectly confounded with stimulus identity. "Different run" predicts brain
> dissimilarity at Spearman **+0.49 to +0.87** across all 12 task × session cells, while
> no stimulus property predicts it at all (trial type ≈ −0.02, length ≈ 0, lexical
> overlap ≈ 0). A language model cannot represent a scanner run, so its RSA against that
> structure is ≈0 by construction.
>
> Ruled out as explanations: **cohort size** (40 vs 98 subjects agree at rho 0.928) and
> **layer choice** (flat across all 12–14 layers). Within-run normalisation removes the
> confound (+0.562 → −0.041) and leaves alignment at **+0.022, not significant**.
>
> **`interp_*`, `localisation_*`, `behaviour` and `ablation_behaviour` on this page are
> unaffected** — they are computed from the model alone and never touch the fMRI data.
> See the [repo README](../../README.md) for the full account.
"""


def fmt(v, n=4):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


def main() -> None:
    fam_sum = pd.read_csv(STAGE / "overall" / "summary_by_family.csv").set_index("family")
    for fd in sorted((STAGE / "by-model").iterdir()):
        fam = fd.name
        ck = pd.read_csv(fd / "checkpoints.csv")
        s = fam_sum.loc[fam] if fam in fam_sum.index else None
        repo, desc, ckdesc = INFO.get(fam, ("?", "?", "?"))
        files = sorted(p.name for p in fd.glob("*.csv"))

        L = [f"# {fam}", "", f"**{desc}** — [`{repo}`](https://huggingface.co/{repo}) · {ckdesc}.",
             "", f"{len(ck)} checkpoints, steps {int(ck.step.min())}–{int(ck.step.max())}.", "",
             WARNING, "", "## Headline numbers", "",
             "| metric | value | safe to use? |", "|---|---|---|"]
        L.append(f"| brain RSA (mean over checkpoints × 12 cells) | {fmt(s['brain_rsa_mean'])} | ❌ confounded |")
        L.append(f"| brain RSA vs training step | rho {fmt(s['brain_trend_rho'],3)}, p {fmt(s['brain_trend_p'],3)} (n={int(s['brain_trend_n'])}) | ❌ confounded |")
        for col, lab in (("interp_per_mean", "interp: participation ratio (effective dim / hidden)"),
                         ("interp_gini_mean", "interp: gini (activation sparsity)"),
                         ("interp_hoyer_mean", "interp: hoyer sparsity"),
                         ("interp_cka_to_prev_mean", "interp: CKA to previous checkpoint")):
            if s is not None and col in s and not pd.isna(s[col]):
                L.append(f"| {lab} | {fmt(s[col])} | ✅ |")
        for col, lab in (("loc_selectivity_mean", "localisation: selectivity index"),
                         ("loc_overlap_mean", "localisation: mean overlap with other phenomena"),
                         ("loc_n_active_layers_mean", "localisation: active layers"),
                         ("loc_layer_com_mean", "localisation: layer centre of mass")):
            if s is not None and col in s and not pd.isna(s[col]):
                L.append(f"| {lab} | {fmt(s[col])} | ✅ |")
        if s is not None and "behav_mp_accuracy_mean" in s and not pd.isna(s["behav_mp_accuracy_mean"]):
            L.append(f"| behaviour: minimal-pair accuracy (chance 0.5) | {fmt(s['behav_mp_accuracy_mean'])} | ✅ |")
        if s is not None and "causal_selectivity_mean" in s and not pd.isna(s["causal_selectivity_mean"]):
            L.append(f"| causal selectivity (localized vs random ablation) | {fmt(s['causal_selectivity_mean'])} | ✅ |")

        # honest per-model reading
        L += ["", "## What this model shows", ""]
        acc = s["behav_mp_accuracy_mean"] if s is not None else np.nan
        per = s["interp_per_mean"] if s is not None else np.nan
        sel = s["loc_selectivity_mean"] if s is not None else np.nan
        notes = []
        if not pd.isna(acc):
            if acc < 0.55:
                notes.append(f"- **Behaviour is close to chance** ({acc:.3f} vs 0.5). Read every other number here with that in mind: a model that barely discriminates minimal pairs is not a strong test of anything.")
            elif acc > 0.65:
                notes.append(f"- Minimal-pair accuracy **{acc:.3f}**, clearly above chance — the model has learned the contrasts being probed.")
            else:
                notes.append(f"- Minimal-pair accuracy **{acc:.3f}**, modestly above chance.")
        if not pd.isna(per):
            notes.append(f"- Participation ratio **{per:.3f}**: the representation uses ~{per*100:.0f}% of its available dimensions. "
                         + ("Very compressed relative to the other families here." if per < 0.12
                            else "High-dimensional relative to the other families here." if per > 0.25
                            else "Mid-range for this release."))
        if not pd.isna(sel):
            notes.append(f"- Selectivity **{sel:.3f}** — like every model in this release, phenomena are only moderately isolated. Nothing here shows sharp, dedicated linguistic circuitry.")
        if "cka_to_prev" in ck.columns or "interp_cka_to_prev" in ck.columns:
            c = ck.get("interp_cka_to_prev")
            if c is not None and c.notna().sum() > 1 and c.dropna().iloc[-1] > 0.99:
                notes.append(f"- CKA to the previous checkpoint reaches {c.dropna().iloc[-1]:.4f} by the end: **the representation has stopped changing**, so late checkpoints are near-duplicates.")
        notes.append("- Brain alignment: **no claim is made**, see the warning above.")
        L += notes

        L += ["", "## Files", "",
              "| file | contents | confounded? |", "|---|---|---|"]
        DESC = {
          "checkpoints.csv": ("this model's rows of the main per-checkpoint table (all three axes)", "brain_* columns only"),
          "brain_alignment.csv": ("RSA per task × session × checkpoint", "**yes — whole file**"),
          "interp_mechanistic.csv": ("per-checkpoint representation metrics", "no"),
          "interp_layerwise.csv": ("the same metrics per layer per checkpoint", "no"),
          "localisation_isolation.csv": ("per-phenomenon circuit localisation", "no"),
          "localisation_onset.csv": ("localisation onset step per phenomenon", "no"),
          "behaviour.csv": ("minimal-pair accuracy per phenomenon per checkpoint", "no"),
          "ablation_alignment.csv": ("brain RSA with the localized circuit ablated", "**yes — whole file**"),
          "ablation_behaviour.csv": ("accuracy under localized vs random ablation", "no"),
        }
        for f in files:
            d, c = DESC.get(f, ("", ""))
            L.append(f"| `{f}` | {d} | {c} |")
        L += ["", f"![overview](figures/{fam}_overview.png)", ""]
        (fd / "README.md").write_text("\n".join(L))
        print("wrote", fd.name)


if __name__ == "__main__":
    main()
