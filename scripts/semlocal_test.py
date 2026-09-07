#!/usr/bin/env python
"""The ds006239/SemLocal test that decides between two very different readings.

SemLocal is the only run x stimulus CROSSED cell in the collection -- the
scanner-run confound that invalidated ds003604 cannot arise there by design. In
the smoke run, pythia-70m's alignment on SemLocal/ses-11 was HIGHEST at step 0
(untrained, 0.053) and fell monotonically to 0.022 by step 143000.

Two readings are consistent with that, and they are different papers:

  (A) "training destroys alignment" -- the untrained model has some real
      surface-form geometry that matches the brain RDM, and training moves the
      representation away from it. Requires step-0 alignment to be ABOVE the
      noise-seed band.

  (B) "the metric never measured anything" -- step 0 is inside the band that
      models differing only by random seed produce, so the decline is drift
      within noise and there is no alignment to destroy.

This script reports BOTH explicitly rather than choosing. It needs the PARC
families to have run.
"""
from pathlib import Path
import pandas as pd
import numpy as np

RES = Path("/local/scratch/sas245/brainalign-evals/results")
d = pd.read_csv(RES / "alignment_rows.csv")

CELL = d[(d.dataset == "ds006239") & (d.task == "SemLocal")]
if CELL.empty:
    print("[semlocal] no SemLocal rows yet"); raise SystemExit

null = CELL[CELL.family.str.startswith("parc-")]
real = CELL[~CELL.family.str.startswith("parc-")]
if null.empty:
    print("[semlocal] no PARC noise seeds yet -- cannot separate reading (A) "
          "from reading (B). Run the PARC families first.")
    raise SystemExit

for ses, g in CELL.groupby("session"):
    n = g[g.family.str.startswith("parc-")]
    r = g[~g.family.str.startswith("parc-")]
    # Null band = across-seed spread of per-seed means at THIS cell.
    per_seed = n.groupby("family")["rsa"].mean()
    lo, hi = per_seed.min(), per_seed.max()
    mu, sd = per_seed.mean(), per_seed.std()
    print(f"\n=== ds006239/SemLocal/{ses} ===")
    print(f"  noise seeds (n={len(per_seed)}): mean {mu:+.4f}  sd {sd:.4f}  "
          f"range [{lo:+.4f}, {hi:+.4f}]")

    step0 = r[r.step == 0]["rsa"]
    final = r[r.step == r.step.max()]["rsa"]
    if len(step0):
        z0 = (step0.mean() - mu) / sd if sd else float("nan")
        inside = lo <= step0.mean() <= hi
        print(f"  UNTRAINED (step 0)  : {step0.mean():+.4f}  z={z0:+.2f}  "
              f"{'INSIDE' if inside else 'OUTSIDE'} the noise band")
        print(f"  trained (final step): {final.mean():+.4f}  "
              f"z={(final.mean()-mu)/sd if sd else float('nan'):+.2f}")
        print()
        if inside:
            print("  => READING (B) SUPPORTED: step-0 alignment is within the")
            print("     band produced by models differing only by random seed.")
            print("     The decline over training is drift inside noise; there")
            print("     is no alignment being destroyed. This is a result about")
            print("     the benchmark, not about training.")
        else:
            print("  => READING (A) SUPPORTED: step-0 alignment exceeds every")
            print("     noise seed, so the untrained model carries geometry the")
            print("     brain RDM shares and training moves away from it. This")
            print("     is a claim about training dynamics and needs the")
            print("     positive-control caveat attached.")
        print("  Both readings are reported; do not quote one alone.")
