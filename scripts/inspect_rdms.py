import glob, os
import numpy as np

BASE = "/local/scratch/sas245/brainalign-evals/data/ds003604-session-rdms"
files = sorted(glob.glob(os.path.join(BASE, "**", "*.npz"), recursive=True))
print(f"{len(files)} npz files\n")
for f in files:
    rel = os.path.relpath(f, BASE)
    d = np.load(f, allow_pickle=True)
    keys = list(d.keys())
    rdm = d["rdm"] if "rdm" in keys else None
    n = rdm.shape[0] if rdm is not None else -1
    print(f"{rel:70s} n={n:4d} keys={keys}")

# deep dive on one corrected file per dataset
for f in [
    "ds003604/within-run-normalised/Phon/session_rdm_ses-5.npz",
    "ds002236/within-run-normalised/Sem/session_rdm_ses-11+.npz",
    "ds006239/within-run-normalised/SemLocal/session_rdm_ses-11+.npz",
]:
    p = os.path.join(BASE, f)
    if not os.path.exists(p):
        print("MISSING", f); continue
    d = np.load(p, allow_pickle=True)
    print("\n" + "=" * 78); print(f)
    for k in d.keys():
        v = d[k]
        if v.ndim == 0:
            print(f"  {k}: scalar = {v.item()!r}")
        else:
            print(f"  {k}: shape={v.shape} dtype={v.dtype} head={list(v.ravel()[:4])}")
    if "stimulus_texts" in d:
        print("  --- first 8 stimulus_texts:")
        for t in d["stimulus_texts"][:8]:
            print("     ", repr(t))
