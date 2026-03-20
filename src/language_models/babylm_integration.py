"""
Integration with BabyLM language models.
Loads trained language models from configs/ directory for RDM computation.
"""

import json
from pathlib import Path
from typing import List, Optional
import logging

import yaml

logger = logging.getLogger(__name__)


class BabyLMModelRegistry:
    """
    Registry for BabyLM models defined in configs/.
    Handles model loading and RDM computation for trained language models.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize registry from config directory.
        
        Args:
            config_dir: Directory containing model configs
        """
        self.config_dir = Path(config_dir)
        self.models = self._load_configs()
    
    def _load_configs(self) -> dict:
        """Load all BabyLM model configs."""
        models = {}
        
        for config_file in self.config_dir.glob("babylm_*.yaml"):
            with open(config_file) as f:
                config = yaml.safe_load(f)
                model_name = config_file.stem
                models[model_name] = config
                logger.info(f"Loaded config: {model_name}")
        
        return models
    
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Get the checkpoint path for a BabyLM model.
        
        Args:
            model_name: Model name (e.g., "babylm_60M")
            
        Returns:
            Path to model checkpoint, or None if not found
        """
        if model_name not in self.models:
            logger.warning(f"Model not found: {model_name}")
            return None
        
        config = self.models[model_name]
        
        # Model path is typically in config or can be inferred
        if "model_path" in config:
            return Path(config["model_path"])
        
        # Default: assume model is at configs/<model_name>_checkpoint
        default_path = self.config_dir.parent / "checkpoints" / f"{model_name}_checkpoint"
        
        if default_path.exists():
            return default_path
        
        logger.warning(f"Model path not found for {model_name}")
        return None
    
    def list_models(self) -> List[str]:
        """Get list of available BabyLM models."""
        return list(self.models.keys())
    
    def get_config(self, model_name: str) -> Optional[dict]:
        """Get config for a BabyLM model."""
        return self.models.get(model_name)


def compute_babylm_rdm(
    model_name: str,
    task: str = "Sem",
    output_dir: str = "data/processed/language_models"
) -> Optional[Path]:
    """
    Compute RDM for a BabyLM language model.
    
    Args:
        model_name: BabyLM model name (babylm_60M, babylm_100M, etc.)
        task: Task name (Sem, Gram, Phon, Plaus)
        output_dir: Output directory for RDM
        
    Returns:
        Path to saved RDM file
    """
    from language_model_rdm import LanguageModelEmbeddingExtractor, LanguageModelRDMComputer
    
    # Get model registry
    registry = BabyLMModelRegistry()
    
    # Get model path
    model_path = registry.get_model_path(model_name)
    if model_path is None:
        logger.error(f"Could not find model path for {model_name}")
        return None
    
    logger.info(f"Loading model from: {model_path}")
    
    try:
        # Initialize extractor with model path
        extractor = LanguageModelEmbeddingExtractor(
            model_name=str(model_path),
            layer=-1,
            pooling="mean"
        )
        
        # Compute RDM
        rdm_computer = LanguageModelRDMComputer(output_dir=output_dir)
        characteristics, words, word_to_idx = (
            rdm_computer.load_stimulus_characteristics(task=task)
        )
        
        embeddings = extractor.extract_batch_embeddings(words)
        rdm = rdm_computer.compute_stimulus_rdm(
            characteristics, embeddings, words, word_to_idx
        )
        
        # Save RDM
        output_file = f"lm_rdm_{model_name}_{task}.npz"
        return rdm_computer.save_rdm(rdm, output_file)
        
    except Exception as e:
        logger.error(f"Failed to compute RDM for {model_name}: {e}")
        return None


def main():
    """Example: compute RDMs for all available BabyLM models."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute RDMs for BabyLM models")
    parser.add_argument("--task", default="Sem", help="Task name")
    parser.add_argument("--output-dir", default="data/processed/language_models")
    parser.add_argument("--models", nargs="+", help="Specific models to compute (default: all)")
    
    args = parser.parse_args()
    
    # Get registry
    registry = BabyLMModelRegistry()
    available_models = registry.list_models()
    
    if not available_models:
        print("No BabyLM models found in configs/")
        return
    
    print(f"Available models: {available_models}")
    
    # Select models to compute
    models_to_compute = args.models if args.models else available_models
    
    # Compute RDMs
    results = {}
    for model in models_to_compute:
        if model not in available_models:
            print(f"Warning: {model} not found")
            continue
        
        print(f"\nComputing RDM for {model}...")
        output_file = compute_babylm_rdm(
            model_name=model,
            task=args.task,
            output_dir=args.output_dir
        )
        
        if output_file:
            results[model] = str(output_file)
            print(f"✓ Saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary: Computed {len(results)} RDMs")
    for model, path in results.items():
        print(f"  {model}: {path}")


if __name__ == "__main__":
    main()
