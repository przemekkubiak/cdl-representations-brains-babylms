#!/usr/bin/env python
"""What is the smallest true alignment this instrument could have found?

"No LM alignment is detectable" is only meaningful with a detection floor
attached. This measures one, using the pipeline's exact estimator, and cross-
checks it against an empirical anchor: one session's GROUP RDM predicting
another session's, different children, same stimuli. If the instrument recovers
that at high rho, it is not deaf, and any w* that would have missed it indicts
the simulation rather than the instrument.

Method. Build probes of KNOWN true alignment by mixing the group RDM with a
stimulus-label permutation of itself, which preserves the marginal distribution
exactly:  R(w) = w*R_group + (1-w)*R_perm.  Score R(w) against the brain RDM with
the sweep's own estimator (z-normalise both, Spearman on the upper triangle), and
find the smallest w whose score clears the untrained-model band at 80% power.

Also reported, because both bear directly on how the published tables read:
  * cross-session external-RDM agreement per cell -- the empirical anchor
  * the additive per-stimulus main-effect control (meanD_i + meanD_j), which no
    published artifact records and which outscores every model
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RDM = ROOT / "data/ds003604-session-rdms"
RES = ROOT / "results"
RNG = np.random.default_rng(0)
WS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
N_SIM = 200


def ut(m):
    return m[np.triu_indices_from(m, k=1)]


def zscore(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def est(a, b):
    """The sweep's estimator: z-normalise both, Spearman on the upper triangle."""
    return float(stats.spearmanr(zscore(ut(a)), zscore(ut(b))).statistic)


def cells(dataset):
    out = {}
    for f in sorted(glob.glob(str(RDM / dataset / "within-run-normalised/*/session_rdm_*.npz"))):
        q = Path(f)
        d = np.load(f, allow_pickle=True)
        out[(q.parts[-2], re.sub(r"session_rdm_(.*)\.npz", r"\1", q.name))] = d
    return out


def permuted(rdm, rng):
    """Stimulus-label permutation: preserves the marginal distribution exactly."""
    p = rng.permutation(rdm.shape[0])
    return rdm[np.ix_(p, p)]


def main() -> int:
    align = pd.read_csv(RES / "alignment_rows.csv")
    init = align[align.step == 0]
    band = (init.groupby(["dataset", "task", "session", "family"]).rsa.mean()
            .reset_index().groupby(["dataset", "task", "session"]).rsa
            .agg(["mean", "std"]).rename(columns={"mean": "init_mean", "std": "init_sd"}))

    anchor_rows, wstar_rows, additive_rows = [], [], []
    for ds in ["ds003604", "ds002236", "ds006239"]:
        cs = cells(ds)
        for (task, ses), d in cs.items():
            R = np.asarray(d["rdm"], float)
            n = R.shape[0]

            # --- empirical anchor: another session of the same task ----------
            best = np.nan
            for (t2, s2), d2 in cs.items():
                if t2 != task or s2 == ses:
                    continue
                R2 = np.asarray(d2["rdm"], float)
                if R2.shape == R.shape:
                    v = est(R, R2)
                    best = v if best != best or v > best else best
            ceil_lo = float(d["noise_ceiling_lower"]) if "noise_ceiling_lower" in d.files else np.nan
            anchor_rows.append(dict(dataset=ds, task=task, session=ses, n_stim=n,
                                    ceiling_lower=round(ceil_lo, 4),
                                    cross_session_rsa=round(best, 4) if best == best else None))

            # --- additive per-stimulus main effect ---------------------------
            md = R.mean(axis=1)
            A = md[:, None] + md[None, :]
            r_add = est(R, A)
            null = [est(R, permuted(A, RNG)) for _ in range(200)]
            p_add = float((np.abs(null) >= abs(r_add)).mean())
            additive_rows.append(dict(dataset=ds, task=task, session=ses,
                                      additive_rsa=round(r_add, 4), perm_p=p_add,
                                      significant=bool(p_add < 0.05)))

            # --- detection floor w* ------------------------------------------
            try:
                b = band.loc[(ds, task, ses)]
                thr = float(b.init_mean + 2 * b.init_sd)
            except KeyError:
                continue
            row = dict(dataset=ds, task=task, session=ses, threshold=round(thr, 5),
                       ceiling_lower=round(ceil_lo, 4))
            wstar = None
            for w in WS:
                hits = 0
                for _ in range(N_SIM):
                    probe = w * R + (1 - w) * permuted(R, RNG)
                    if est(R, probe) > thr:
                        hits += 1
                power = hits / N_SIM
                row[f"power_w{w}"] = round(power, 3)
                if wstar is None and power >= 0.8:
                    wstar = w
            row["w_star"] = wstar
            row["w_star_frac_of_ceiling"] = (round(wstar / ceil_lo, 3)
                                             if wstar is not None and ceil_lo == ceil_lo
                                             and ceil_lo > 0 else None)
            wstar_rows.append(row)

    anchor = pd.DataFrame(anchor_rows)
    add = pd.DataFrame(additive_rows)
    ws = pd.DataFrame(wstar_rows)
    anchor.to_csv(RES / "external_rdm_anchor.csv", index=False)
    add.to_csv(RES / "additive_control.csv", index=False)
    ws.to_csv(RES / "detectability_wstar.csv", index=False)

    best_model = align.groupby(["dataset", "task", "session"]).rsa.max().rename("best_model_rsa")
    j = anchor.merge(best_model, on=["dataset", "task", "session"], how="left")
    j["ratio_model_to_external"] = (j.best_model_rsa / j.cross_session_rsa).round(4)
    j.to_csv(RES / "instrument_ratio.csv", index=False)

    summary = dict(
        n_cells=int(len(anchor)),
        cross_session_rsa=dict(min=round(float(anchor.cross_session_rsa.min()), 4),
                               median=round(float(anchor.cross_session_rsa.median()), 4),
                               max=round(float(anchor.cross_session_rsa.max()), 4)),
        best_model_rsa_median=round(float(j.best_model_rsa.median()), 5),
        model_to_external_ratio_median=round(float(j.ratio_model_to_external.median()), 4),
        additive_control=dict(
            n_significant=int(add.significant.sum()), n_cells=int(len(add)),
            max_rsa=round(float(add.additive_rsa.max()), 4),
            median_rsa=round(float(add.additive_rsa.median()), 4)),
        w_star=dict(median=(float(ws.w_star.median()) if ws.w_star.notna().any() else None),
                    n_cells_with_wstar=int(ws.w_star.notna().sum()),
                    n_cells=int(len(ws))),
        verdict=None)
    med_w = summary["w_star"]["median"]
    med_ceil = float(anchor.ceiling_lower.median())
    if med_w is not None:
        frac = med_w / med_ceil if med_ceil else float("nan")
        summary["verdict"] = (
            "instrument has real power; the null about models STANDS" if frac <= 0.15 else
            "null is VACUOUS: the instrument could not have detected a plausible effect"
            if frac > 0.5 else
            f"null is bounded: no alignment above w*={med_w} ({frac:.2f} of ceiling) is detectable")
    (RES / "detectability_verdict.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
