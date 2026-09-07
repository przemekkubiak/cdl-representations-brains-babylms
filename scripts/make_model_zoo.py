#!/usr/bin/env python
"""Build an EXTENDED model_zoo.yaml = pipeline's 36 families + the ones this
sweep adds (Pythia 2.8b/6.9b/12b, the deduped ladder, and all 45 PolyPythias).

Written to brainalign-evals/configs/model_zoo_extended.yaml and passed to
scripts/run_devai_grid.py via --model-zoo. The upstream config is never edited.
"""
import copy
from pathlib import Path
import yaml

PIPE = Path("/local/scratch/sas245/brainalign-evals/pipeline")
OUT = Path("/local/scratch/sas245/brainalign-evals/configs/model_zoo_extended.yaml")
OUT.parent.mkdir(parents=True, exist_ok=True)

zoo = yaml.safe_load((PIPE / "configs/model_zoo.yaml").read_text())
fams = zoo["families"]
TPS = 2097152  # Pythia tokens/step (batch 1024 x seq 2048)

added = []

# --- rungs of the scale ladder the pipeline is missing --------------------
for size in ["2.8b", "6.9b", "12b"]:
    key = f"pythia-{size}-full"
    if key not in fams:
        fams[key] = {"hf_repo": f"EleutherAI/pythia-{size}", "arch": "gptneox",
                     "all_revisions": True, "tokens_per_step": TPS,
                     "notes": f"{size} rung -- extends the ladder past 1.4b."}
        added.append(key)

# --- the deduped ladder (data-dedup axis at matched scale) -----------------
for size in ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]:
    key = f"pythia-{size}-deduped-full"
    if key not in fams:
        fams[key] = {"hf_repo": f"EleutherAI/pythia-{size}-deduped",
                     "arch": "gptneox", "all_revisions": True,
                     "tokens_per_step": TPS,
                     "notes": "Deduped-Pile counterpart at matched scale."}
        added.append(key)

# --- PolyPythias: seed variance at 5 sizes x 9 extra seeds ----------------
# seed0 is the canonical pythia-<size> release, already in the zoo.
for size in ["14m", "31m", "70m", "160m", "410m"]:
    for seed in range(1, 10):
        key = f"polypythia-{size}-seed{seed}"
        if key not in fams:
            fams[key] = {"hf_repo": f"EleutherAI/pythia-{size}-seed{seed}",
                         "arch": "gptneox", "all_revisions": True,
                         "tokens_per_step": TPS,
                         "polypythia": {"size": size, "seed": seed},
                         "notes": "PolyPythias seed replicate."}
            added.append(key)

# 14m/31m have no canonical seed0-equivalent in the zoo; add the base repos too
for size in ["14m", "31m"]:
    key = f"pythia-{size}-full"
    if key not in fams:
        fams[key] = {"hf_repo": f"EleutherAI/pythia-{size}", "arch": "gptneox",
                     "all_revisions": True, "tokens_per_step": TPS,
                     "notes": "Below the published ladder's floor."}
        added.append(key)

OUT.write_text(yaml.safe_dump(zoo, sort_keys=False, width=200))
print(f"wrote {OUT}")
print(f"families: {len(fams)} (+{len(added)} added)")
for a in added:
    print("  +", a)
