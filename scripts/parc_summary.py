#!/usr/bin/env python3
"""Aggregate the PARC arm of the sweep: 3 architectures x 3 seeds x 3 datasets.

Conventions copied from scripts/build_devai_package.py so the numbers here are
comparable with what is already published:

  * one value per (family, cell) FIRST -- mean over the 11 checkpoints -- before
    any spread is taken, so an averaged numerator is never compared against an
    unaveraged denominator;
  * `frac_of_ceiling` = rsa / ceiling_lower, read off the RDM npz files;
  * the reference band is the 15 random-init (step-0) Pythia/PolyPythia
    checkpoints measured on the SAME cell (`untrained_reference` convention),
    two-sided;
  * error bars across the 3 seeds within an architecture, never across pooled
    checkpoint rows.

Writes results/parc_by_cell.csv, results/parc_by_seed.csv, results/parc_summary.csv.
"""
import glob, re, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, wilcoxon, kruskal

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RDM = ROOT / "data/ds003604-session-rdms"
DATASETS = ["ds003604", "ds002236", "ds006239"]
ARCHES = ["pythia", "mamba", "rwkv"]
SEEDS = [0, 1, 2]


def ceilings(ds, variant="within-run-normalised"):
    rows = []
    for f in glob.glob(str(RDM / ds / variant / "*/session_rdm_*.npz")):
        p = Path(f); d = np.load(f, allow_pickle=True)
        rows.append(dict(dataset=ds, task=p.parts[-2],
                         session=re.sub(r"session_rdm_(.*)\.npz", r"\1", p.name),
                         ceiling_lower=float(d["noise_ceiling_lower"]),
                         ceiling_upper=float(d["noise_ceiling_upper"])))
    return pd.DataFrame(rows)


def main():
    align = pd.read_csv(ROOT / "results/alignment_rows.csv")
    ceil = pd.concat([ceilings(d) for d in DATASETS], ignore_index=True)

    parc = align[align.family.str.startswith("parc-")].copy()
    parc["arch"] = parc.family.str.extract(r"parc-([a-z]+)-seed")
    parc["seed"] = parc.family.str.extract(r"seed(\d)").astype(int)

    # ---- per (family, cell): mean over checkpoints, plus the training trend ----
    recs = []
    for (ds, fam, task, ses), sub in parc.groupby(["dataset", "family", "task", "session"]):
        sub = sub.sort_values("step")
        rho, p = spearmanr(sub.step, sub.rsa)
        recs.append(dict(dataset=ds, family=fam,
                         arch=sub.arch.iloc[0], seed=int(sub.seed.iloc[0]),
                         task=task, session=ses, n_checkpoints=len(sub),
                         rsa_mean=sub.rsa.mean(), rsa_sd_over_ckpt=sub.rsa.std(),
                         rsa_first=sub.rsa.iloc[0], rsa_final=sub.rsa.iloc[-1],
                         first_step=int(sub.step.iloc[0]), last_step=int(sub.step.iloc[-1]),
                         trend_rho=rho, trend_p=p))
    cell = pd.DataFrame(recs).merge(ceil, on=["dataset", "task", "session"], how="left")
    cell["frac_of_ceiling"] = cell.rsa_mean / cell.ceiling_lower

    # ---- random-init reference band, same cell, 15 step-0 inits ----
    init = align[align.step == 0]
    band = (init.groupby(["dataset", "task", "session", "family"]).rsa.mean()
                .reset_index()
                .groupby(["dataset", "task", "session"]).rsa
                .agg(["mean", "std", "count"])
                .rename(columns={"mean": "init_mean", "std": "init_sd",
                                 "count": "init_n_seeds"}).reset_index())
    cell = cell.merge(band, on=["dataset", "task", "session"], how="left")
    cell["z_vs_untrained"] = (cell.rsa_mean - cell.init_mean) / cell.init_sd
    cell["beats_untrained_2sd"] = cell.z_vs_untrained > 2.0
    cell["below_untrained_2sd"] = cell.z_vs_untrained < -2.0
    cell.round(6).to_csv(ROOT / "results/parc_by_cell.csv", index=False)

    # ---- per (dataset, arch, seed): mean over that dataset's cells ----
    seed_lvl = (cell.groupby(["dataset", "arch", "seed"])
                    .agg(n_cells=("rsa_mean", "size"),
                         rsa_mean=("rsa_mean", "mean"),
                         rsa_sd_across_cells=("rsa_mean", "std"),
                         rsa_final_mean=("rsa_final", "mean"),
                         frac_of_ceiling_mean=("frac_of_ceiling", "mean"),
                         mean_z_vs_untrained=("z_vs_untrained", "mean"))
                    .reset_index())
    seed_lvl.round(6).to_csv(ROOT / "results/parc_by_seed.csv", index=False)

    # ---- per (dataset, arch): error bars ACROSS SEEDS ----
    out = []
    for (ds, arch), sub in seed_lvl.groupby(["dataset", "arch"]):
        c = cell[(cell.dataset == ds) & (cell.arch == arch)]
        # per-cell value averaged over the 3 seeds, then tested against 0
        percell = c.groupby(["task", "session"]).rsa_mean.mean()
        try:
            w_stat, w_p = wilcoxon(percell.values)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        # training effect within PARC: first vs final checkpoint, paired over
        # (seed x cell); PARC publishes no step-0, so 'first' is checkpoint-10.
        try:
            t_stat, t_p = wilcoxon(c.rsa_first.values, c.rsa_final.values)
        except ValueError:
            t_stat, t_p = np.nan, np.nan
        out.append(dict(
            dataset=ds, arch=arch, n_seeds=len(sub), n_cells=int(sub.n_cells.iloc[0]),
            rsa_mean=sub.rsa_mean.mean(),
            rsa_sd_across_seeds=sub.rsa_mean.std(ddof=1),
            rsa_sem_across_seeds=sub.rsa_mean.std(ddof=1) / np.sqrt(len(sub)),
            rsa_ci95_lo=sub.rsa_mean.mean() - 4.303 * sub.rsa_mean.std(ddof=1) / np.sqrt(len(sub)),
            rsa_ci95_hi=sub.rsa_mean.mean() + 4.303 * sub.rsa_mean.std(ddof=1) / np.sqrt(len(sub)),
            frac_of_ceiling_mean=sub.frac_of_ceiling_mean.mean(),
            frac_of_ceiling_sd_across_seeds=sub.frac_of_ceiling_mean.std(ddof=1),
            mean_z_vs_untrained=sub.mean_z_vs_untrained.mean(),
            cells_beating_untrained_2sd=int(c.beats_untrained_2sd.sum()),
            cells_below_untrained_2sd=int(c.below_untrained_2sd.sum()),
            n_family_cells=int(len(c)),
            wilcoxon_rsa_vs_zero_p=w_p,
            wilcoxon_first_vs_final_p=t_p,
            mean_rsa_first=c.rsa_first.mean(), mean_rsa_final=c.rsa_final.mean(),
            median_trend_rho=c.trend_rho.median(),
        ))
    summ = pd.DataFrame(out)
    summ.round(6).to_csv(ROOT / "results/parc_summary.csv", index=False)

    # ---- do the three architectures differ at all, per dataset? ----
    arch_test = []
    for ds, sub in cell.groupby("dataset"):
        groups = [sub[sub.arch == a].groupby(["task", "session"]).rsa_mean.mean().values
                  for a in ARCHES]
        h, p = kruskal(*groups)
        arch_test.append(dict(dataset=ds, test="kruskal_arch_percell", H=h, p=p,
                              n_cells=len(groups[0])))
    pd.DataFrame(arch_test).round(6).to_csv(ROOT / "results/parc_arch_test.csv", index=False)

    print(summ.round(5).to_string(index=False))
    print()
    print(pd.DataFrame(arch_test).round(5).to_string(index=False))
    print()
    print(seed_lvl.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
