#!/usr/bin/env python
"""Precision A/B: does bf16 change the RSA values relative to fp32?

The 12b rung must run in bf16 (fp32 weights alone would be ~48 GB against a
60 GB VRAM budget on a shared GPU). That means the ladder would mix precisions,
so this measures the size of that inconsistency on a model we can afford to run
both ways -- pythia-410m, same 11 checkpoints, same 26 cells.

If the deltas are far below the between-family spread, mixing is a footnote. If
they are not, the 12b point has to be reported as a different measurement rather
than as the top of the same curve.
"""
from pathlib import Path
import glob
import pandas as pd

ROOT = Path("/local/scratch/sas245/brainalign-evals")
RES = ROOT / "results"
KEY = ["dataset", "task", "session", "model_ref"]


def load(griddir: Path, fam: str) -> pd.DataFrame:
    fs = glob.glob(str(griddir / f"*/alignment_{fam}.csv"))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)


a = load(ROOT / "grid", "pythia-410m-full")
b = load(ROOT / "grid_bf16", "pythia-410m-full")
if a.empty or b.empty:
    print("[precision] need both fp32 and bf16 runs of pythia-410m-full; skipping")
    raise SystemExit

m = a.merge(b, on=KEY, suffixes=("_fp32", "_bf16"))
m["delta"] = m["rsa_bf16"] - m["rsa_fp32"]
m["abs_delta"] = m["delta"].abs()
m.to_csv(RES / "precision_ab.csv", index=False)

print(f"[precision] pythia-410m fp32 vs bf16, {len(m)} matched (cell x checkpoint) rows")
print(f"  mean |delta rsa| : {m['abs_delta'].mean():.5f}")
print(f"  max  |delta rsa| : {m['abs_delta'].max():.5f}")
print(f"  corr(fp32, bf16) : {m['rsa_fp32'].corr(m['rsa_bf16']):.6f}")

# Scale reference: how big is the delta next to the spread the study cares about?
rows = RES / "alignment_rows.csv"
if rows.exists():
    d = pd.read_csv(rows)
    spread = d.groupby("family")["rsa"].mean().std()
    print(f"  between-family sd of mean rsa : {spread:.5f}")
    if spread and spread == spread:
        print(f"  => bf16 error is {m['abs_delta'].mean() / spread:.1%} of the "
              f"between-family spread")
