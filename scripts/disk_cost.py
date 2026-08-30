"""Metadata-only disk-cost estimate per model repo. Downloads no weights."""
import json, os, glob
from huggingface_hub import HfApi

api = HfApi()

REPOS = {
    # Pythia scaling suite (final + intermediate revisions all same size per ckpt)
    "pythia": [f"EleutherAI/pythia-{s}{d}" for s in
               ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
               for d in ["", "-deduped"]],
    # PolyPythias
    "polypythia": [f"EleutherAI/pythia-{s}-seed{n}" for s in
                   ["14m", "31m", "70m", "160m", "410m"] for n in range(1, 10)],
    # PARC
    "parc": [f"jmichaelov/parc-{a}-seed{n}" for a in ["pythia", "mamba", "rwkv"]
             for n in range(6)],
    # BrainAlign + other zoo families
    "zoo_other": ["BrainAlign/gpt2-babylm-3", "BrainAlign/gpt2-babylm-5",
                  "BrainAlign/gpt2-babylm-7", "BrainAlign/gpt2-babylm-9",
                  "pico-lm/pico-decoder-tiny", "pico-lm/pico-decoder-small",
                  "pico-lm/pico-decoder-medium", "pico-lm/pico-decoder-large",
                  "Beetle-HumanScale/beetle-monolingual-humanscale-eng",
                  "Beetle-FineWeb3-24B/beetle-monolingual-fineweb3-eng"],
}

# what is already on this box
cached = set()
for root in ["/local/scratch/sas245/hf_cache", "/local/scratch/sas245/.hf",
             "/local/scratch/sas245/huggingface_cache"]:
    for sub in ["hub", ""]:
        for p in glob.glob(os.path.join(root, sub, "models--*")):
            cached.add(os.path.basename(p)[len("models--"):].replace("--", "/"))

rows = []
for group, ids in REPOS.items():
    for rid in ids:
        try:
            info = api.repo_info(rid, files_metadata=True)
        except Exception as e:
            rows.append(dict(group=group, repo=rid, gb_per_ckpt=None,
                             n_step_branches=None, cached=rid in cached,
                             error=f"{type(e).__name__}"))
            continue
        gb = sum((s.size or 0) for s in info.siblings
                 if s.rfilename.endswith((".bin", ".safetensors"))) / 1e9
        try:
            brs = [b.name for b in api.list_repo_refs(rid).branches]
        except Exception:
            brs = []
        nstep = sum(1 for b in brs if b.startswith(("step", "checkpoint")))
        rows.append(dict(group=group, repo=rid, gb_per_ckpt=round(gb, 3),
                         n_step_branches=nstep, cached=rid in cached, error=None))
        print(f"{group:12s} {rid:52s} {gb:7.2f} GB/ckpt  branches={nstep:4d}  "
              f"cached={rid in cached}", flush=True)

with open("/local/scratch/sas245/brainalign-evals/results/disk_cost.json", "w") as f:
    json.dump(rows, f, indent=1)

print("\n--- totals (one final checkpoint each) ---")
for group in REPOS:
    g = [r for r in rows if r["group"] == group and r["gb_per_ckpt"]]
    print(f"{group:12s} n={len(g):3d}  sum={sum(r['gb_per_ckpt'] for r in g):8.1f} GB")
print(f"\nlocally cached repos matching the list: "
      f"{sorted(r['repo'] for r in rows if r['cached'])}")
