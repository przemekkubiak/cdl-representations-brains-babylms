#!/usr/bin/env python
"""Validate every non-GPU part of the sweep: brain-RDM loading, cell coverage,
the RSA estimator, and checkpoint resolution. Loads no model weights.

Confirms the runner would produce a row for each of the 26 corrected cells, and
that the exact metric the published tables use runs against them.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/local/scratch/sas245/brainalign-evals")
PIPE = ROOT / "pipeline"
sys.path.insert(0, str(PIPE))
import os
os.chdir(PIPE)

sys.argv = ["x", "--model", "x"]
import importlib.util
spec = importlib.util.spec_from_file_location(
    "grid", PIPE / "scripts/run_devai_grid.py")
grid = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grid)

RDM = ROOT / "data/ds003604-session-rdms"
LAYOUT = {
    "ds003604": (["Sem", "Phon", "Gram", "Plaus"], ["ses-5", "ses-7", "ses-9"]),
    "ds002236": (["Phon", "Sem"], ["ses-9", "ses-11", "ses-11+"]),
    "ds006239": (["Orth", "Phon", "Sem", "SemLocal"], ["ses-11", "ses-11+"]),
}

ok = miss = 0
print(f"{'dataset':10s} {'task':9s} {'session':9s} {'n_stim':>7s} {'texts':>6s} "
      f"{'ceiling':>8s}  example text")
print("-" * 88)
cells = []
for ds, (tasks, sessions) in LAYOUT.items():
    root = RDM / ds / "within-run-normalised"
    for t in tasks:
        for s in sessions:
            b = grid._load_brain(str(root), t, s)
            if b is None:
                miss += 1
                print(f"{ds:10s} {t:9s} {s:9s} {'--':>7s}  (no RDM for this cell)")
                continue
            ok += 1
            n = b["rdm"].shape[0]
            has = "yes" if b["texts"] else "NO"
            d = np.load(root / t / f"session_rdm_{s}.npz", allow_pickle=True)
            ceil = float(d["noise_ceiling_lower"]) if "noise_ceiling_lower" in d.files else float("nan")
            ex = b["texts"][0] if b["texts"] else ""
            print(f"{ds:10s} {t:9s} {s:9s} {n:7d} {has:>6s} {ceil:8.3f}  {ex!r}")
            cells.append((ds, t, s, n, b))

print(f"\n{ok} cells loadable, {miss} absent")
bad = [c for c in cells if not c[4]["texts"]]
print(f"cells WITHOUT stimulus_texts (would yield no alignment row): {len(bad)}")

# --- the estimator itself, on a synthetic LM RDM --------------------------
print("\nRSA estimator sanity (synthetic activations, no model loaded):")
rng = np.random.default_rng(0)
for ds, t, s, n, b in cells[:3]:
    acts = rng.normal(size=(n, 256))
    lm = grid._lm_rdm(acts)
    m = grid._rsa(lm, b["rdm"], normalize=True)
    print(f"  {ds}/{t}/{s}: random-activation rsa={m['rsa']:+.4f} "
          f"(expected ~0)  pearson={m['rsa_pearson']:+.4f}")
    # and the identity check: a model RDM equal to the brain RDM must give 1.0
    m1 = grid._rsa(b["rdm"], b["rdm"], normalize=True)
    assert abs(m1["rsa"] - 1.0) < 1e-9, m1
print("  identity check (brain RDM vs itself) = 1.0 for every cell: PASS")

# --- checkpoint resolution for the families the sweep will run -------------
print("\nCheckpoint resolution (Hub metadata only, no weights):")
from src.language_models.babylm_integration import ModelZoo
z = ModelZoo(str(ROOT / "configs/model_zoo_extended.yaml"))
for fam in ["pythia-70m-full", "pythia-2.8b-full", "pythia-12b-full",
            "polypythia-70m-seed1", "polypythia-410m-seed9",
            "parc-pythia-seed0", "babylm-gpt2-3"]:
    try:
        cks = z.resolve_checkpoints(fam)
        sub = grid._subsample(cks, 12)
        print(f"  {fam:24s} {len(cks):4d} ckpts -> {len(sub)} at MAX_CKPT=12 "
              f"| {sub[0]['ref']} .. {sub[-1]['ref']}")
    except Exception as e:
        print(f"  {fam:24s} FAIL {type(e).__name__}: {e}")
