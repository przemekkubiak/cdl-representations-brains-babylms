"""Download the small BrainAlign metadata/RDM dataset repos (~27MB total)."""
import os
from huggingface_hub import snapshot_download

REPOS = [
    "BrainAlign/ds003604-session-rdms",
    "BrainAlign/brain-lm-alignment-ds006239",
    "BrainAlign/brain-lm-alignment-ds002236",
    "BrainAlign/cdl-devai-results",
]
BASE = "/local/scratch/sas245/brainalign-evals/data"

for r in REPOS:
    p = snapshot_download(r, repo_type="dataset",
                          local_dir=os.path.join(BASE, r.split("/")[1]))
    print(r, "->", p, flush=True)
