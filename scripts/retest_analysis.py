#!/usr/bin/env python
"""Is `rsa` reproducible enough to license a family ranking?

Every conclusion in this collection is a BETWEEN-FAMILY comparison, and the
between-family sd of family-mean rsa is only 0.005-0.016 per cell. This compares
that signal against the jitter produced by changes that are mathematically
no-ops: batch size, and which GPU the forward pass ran on. It also folds in the
already-completed fp32-vs-bf16 A/B, which was never analysed.

Pre-declared rule, per cell, with s_ff = between-family sd of family-mean rsa and
j = mean |delta rsa| over matched (cell x checkpoint) pairs:

  j <  0.2 * s_ff   instrument stable; cross-family comparison licensed
  j <  0.5 * s_ff   rankings reportable only within a matched arm
  j >= 0.5 * s_ff   cross-family comparison UNINTERPRETABLE; report per-cell
                    distributions only, never a family ranking
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/local/scratch/sas245/brainalign-evals")
FAMILY = "pythia-410m-full"
RES = ROOT / "results"
KEY = ["dataset", "task", "session", "step"]


def load_arm(griddir: Path) -> pd.DataFrame | None:
    fs = glob.glob(str(griddir / "*" / "alignment_*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d[KEY + ["rsa"]] if "rsa" in d else None


def between_family_sd() -> pd.DataFrame:
    """s_ff: sd across FAMILY MEANS at each cell -- the signal being compared."""
    a = pd.read_csv(RES / "alignment_rows.csv")
    fam = a.groupby(["dataset", "task", "session", "family"]).rsa.mean().reset_index()
    return (fam.groupby(["dataset", "task", "session"]).rsa.std()
            .rename("s_ff").reset_index())


def main() -> int:
    sff = between_family_sd()
    arms = {p.name.replace("grid_retest_", ""): load_arm(p)
            for p in sorted(ROOT.glob("grid_retest_*"))}
    arms = {k: v for k, v in arms.items() if v is not None}

    # The main sweep itself is a fifth arm: same batch size and precision, but a
    # different GPU (0 vs 1) and a different day. It is the cleanest possible
    # no-op, so it bounds pure run-to-run and device nondeterminism.
    main = None
    fs = glob.glob(str(ROOT / "grid" / "*" / f"alignment_{FAMILY}.csv"))
    if fs:
        d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        if "rsa" in d:
            main = d[KEY + ["rsa"]]
            arms["main_sweep_gpu0"] = main

    pairs = []
    ref = arms.get("a_batch16")
    if ref is not None:
        for name, d in arms.items():
            if name == "a_batch16":
                continue
            m = ref.merge(d, on=KEY, suffixes=("_ref", "_alt"))
            m["abs_delta"] = (m.rsa_ref - m.rsa_alt).abs()
            m["contrast"] = f"batch16 vs {name}"
            pairs.append(m[KEY + ["abs_delta", "contrast"]])

    # the bf16/fp32 A/B that already exists but was never analysed
    ab = RES / "precision_ab.csv"
    if ab.is_file():
        d = pd.read_csv(ab)
        d = d.rename(columns={"step_fp32": "step"})
        d["contrast"] = "fp32 vs bf16"
        pairs.append(d[KEY + ["abs_delta", "contrast"]])

    if not pairs:
        print("no retest arms and no precision A/B -- nothing to analyse")
        return 1

    allp = pd.concat(pairs, ignore_index=True).merge(
        sff, on=["dataset", "task", "session"], how="left")

    # Headline on the FINAL checkpoint only. Averaging over the whole trajectory
    # dilutes the effect badly, because at step 0 two precisions agree to 2e-4
    # while at step 143000 they differ by 3e-3 -- the disagreement grows
    # monotonically with training, and it is trained checkpoints that anyone
    # actually compares across families.
    fin = allp[allp.step == allp.groupby("contrast").step.transform("max")]
    fin_cell = (fin.groupby(["contrast", "dataset", "task", "session"])
                .agg(j=("abs_delta", "mean"), s_ff=("s_ff", "first")).reset_index())
    fin_cell["ratio"] = fin_cell.j / fin_cell.s_ff
    fin_cell["verdict"] = np.select(
        [fin_cell.ratio < 0.2, fin_cell.ratio < 0.5],
        ["stable", "matched-arm only"], "UNINTERPRETABLE")
    fin_cell.round(6).to_csv(RES / "retest_final_checkpoint.csv", index=False)

    by_step = (allp.groupby(["contrast", "step"])
               .agg(mean_abs_delta=("abs_delta", "mean"), s_ff=("s_ff", "mean"),
                    n=("abs_delta", "size")).reset_index())
    by_step["ratio"] = by_step.mean_abs_delta / by_step.s_ff
    by_step.round(6).to_csv(RES / "retest_by_step.csv", index=False)

    per_cell = (allp.groupby(["contrast", "dataset", "task", "session"])
                .agg(j=("abs_delta", "mean"), j_max=("abs_delta", "max"),
                     n=("abs_delta", "size"), s_ff=("s_ff", "first"))
                .reset_index())
    per_cell["ratio"] = per_cell.j / per_cell.s_ff
    per_cell["verdict"] = np.select(
        [per_cell.ratio < 0.2, per_cell.ratio < 0.5],
        ["stable", "matched-arm only"], "UNINTERPRETABLE")
    per_cell.round(6).to_csv(RES / "retest_per_cell.csv", index=False)

    summ = (per_cell.groupby("contrast")
            .agg(mean_j=("j", "mean"), mean_s_ff=("s_ff", "mean"),
                 mean_ratio=("ratio", "mean"), max_ratio=("ratio", "max"),
                 n_cells=("ratio", "size"),
                 n_uninterpretable=("verdict", lambda v: int((v == "UNINTERPRETABLE").sum())),
                 n_matched_only=("verdict", lambda v: int((v == "matched-arm only").sum())))
            .reset_index())
    summ.round(6).to_csv(RES / "retest_summary.csv", index=False)

    fin_summ = (fin_cell.groupby("contrast")
                .agg(mean_ratio=("ratio", "mean"), max_ratio=("ratio", "max"),
                     n_cells=("ratio", "size"),
                     n_uninterpretable=("verdict", lambda v: int((v == "UNINTERPRETABLE").sum())))
                .reset_index())
    worst = float(fin_summ.mean_ratio.max())
    # Name the contrasts that actually breach, rather than asserting that every
    # nuisance parameter matters. Batch size and device turn out not to.
    breach = fin_summ[fin_summ.mean_ratio >= 0.2].contrast.tolist()
    clean = fin_summ[fin_summ.mean_ratio < 0.2].contrast.tolist()
    if worst >= 0.5:
        verdict = ("cross-family comparison UNINTERPRETABLE at trained checkpoints; "
                   f"driven by: {', '.join(breach)}")
    elif worst >= 0.2:
        verdict = (f"family rankings valid only within an arm matched on: "
                   f"{', '.join(breach)}. No effect from: {', '.join(clean) or 'none tested'}")
    else:
        verdict = ("instrument stable on every nuisance parameter tested "
                   f"({', '.join(clean)}); cross-family comparison licensed")
    out = dict(all_checkpoints=summ.to_dict("records"),
               final_checkpoint=fin_summ.to_dict("records"),
               worst_mean_ratio_final=round(worst, 4), verdict=verdict,
               note=("the ratio grows monotonically with training step; the "
                     "all-checkpoint average is diluted by untrained checkpoints "
                     "where two precisions agree almost exactly"))
    (RES / "retest_verdict.json").write_text(json.dumps(out, indent=2, default=float))
    print("ALL CHECKPOINTS:"); print(summ.to_string(index=False)); print()
    print("FINAL CHECKPOINT:"); print(fin_summ.to_string(index=False)); print()
    print("VERDICT:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
