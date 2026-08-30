"""CPU/network-only inventory of the model suites. Downloads nothing but metadata."""
import json, re
from huggingface_hub import HfApi

api = HfApi()
out = {}


def repo_size(rid):
    """Sum of weight-file sizes, in GB, from metadata only."""
    try:
        info = api.repo_info(rid, files_metadata=True)
    except Exception as e:
        return None, None, str(e)
    tot = 0
    for s in info.siblings:
        if s.rfilename.endswith((".bin", ".safetensors", ".pt", ".h5", ".msgpack")):
            tot += (s.size or 0)
    nrev = None
    return tot / 1e9, nrev, None


def n_revisions(rid):
    try:
        refs = api.list_repo_refs(rid)
        return len(refs.branches or []) + len(refs.converts or [])
    except Exception:
        return None


print("=" * 78)
print("PYTHIA SUITE (EleutherAI/pythia-*)")
print("=" * 78)
pythia = sorted(m.id for m in api.list_models(author="EleutherAI", search="pythia"))
for m in pythia:
    print(" ", m)
out["pythia_all"] = pythia

print()
print("=" * 78)
print("POLYPYTHIAS (EleutherAI, seed variants)")
print("=" * 78)
poly = sorted(m.id for m in api.list_models(author="EleutherAI", search="seed"))
for m in poly:
    print(" ", m)
out["polypythia_all"] = poly

print()
print("=" * 78)
print("jmichaelov PARC models")
print("=" * 78)
parc = sorted(m.id for m in api.list_models(author="jmichaelov"))
for m in parc:
    print(" ", m)
out["jmichaelov_all"] = parc

with open("/local/scratch/sas245/brainalign-evals/results/model_inventory_raw.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote model_inventory_raw.json")
