"""CPU-only import check: can this venv run the LM half of the pipeline?"""
import sys
from pathlib import Path
ROOT = Path("/local/scratch/sas245/brainalign-evals/pipeline")
sys.path.insert(0, str(ROOT))
import os
os.chdir(ROOT)

mods = [
    "numpy", "scipy", "pandas", "yaml", "torch", "transformers", "sklearn",
    "huggingface_hub", "matplotlib",
]
for m in mods:
    try:
        mod = __import__(m)
        print(f"  OK   {m:18s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  MISS {m:18s} {type(e).__name__}: {e}")

print("\npipeline modules needed by scripts/run_devai_grid.py:")
for m in [
    "src.language_models.babylm_integration",
    "src.language_models.circuit_localization",
    "src.language_models.mechanistic_metrics",
    "src.rsa.encoding_model",
    "src.rsa",
]:
    try:
        __import__(m)
        print(f"  OK   {m}")
    except Exception as e:
        print(f"  FAIL {m}: {type(e).__name__}: {e}")

print("\nModelZoo:")
try:
    from src.language_models.babylm_integration import ModelZoo
    z = ModelZoo("configs/model_zoo.yaml")
    print("  families:", len(z.list_families()))
    for fam in ["pythia-70m-full", "parc-pythia-seed0", "babylm-gpt2-3"]:
        try:
            cks = z.resolve_checkpoints(fam)
            print(f"  {fam}: {len(cks)} ckpts, first={cks[0]['ref']}, last={cks[-1]['ref']}")
        except Exception as e:
            print(f"  {fam}: FAIL {type(e).__name__}: {e}")
except Exception as e:
    print("  FAIL", e)
