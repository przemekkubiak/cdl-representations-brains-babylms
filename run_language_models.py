#!/usr/bin/env python
"""
Runner script for computing language model RDMs.
Can be used on cloud or locally. Orchestrates:
1. Speech recognition (optional, if transcriptions needed)
2. Language model RDM computation for multiple models
3. Comparison with brain RDMs
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from language_models.language_model_rdm import (
    LanguageModelEmbeddingExtractor,
    LanguageModelRDMComputer
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LanguageModelPipeline:
    """
    End-to-end pipeline for computing and comparing language model RDMs.
    """
    
    def __init__(
        self,
        output_dir: str = "data/processed/language_models",
        brain_rdm_dir: str = "data/processed/fmri",
        characteristics_dir: str = "data/brain/ds003604/stimuli/Stimulus_Characteristics",
    ):
        """
        Initialize pipeline.
        
        Args:
            output_dir: Directory for language model outputs
            brain_rdm_dir: Directory containing brain RDMs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.brain_rdm_dir = Path(brain_rdm_dir)
        self.characteristics_dir = Path(characteristics_dir)
        self.rdm_computer = LanguageModelRDMComputer(output_dir=str(self.output_dir))
        
        self.results = {
            "language_models": {},
            "comparisons": {}
        }

    def _non_control_indices(self, task: str) -> np.ndarray:
        """Return row indices of non-control trials based on stimulus characteristics."""
        char_file = self.characteristics_dir / f"task-{task}_Stimulus_Characteristics.tsv"
        characteristics = pd.read_csv(
            char_file,
            sep="\t",
            keep_default_na=False,
            na_values=[""],
        )
        if "trial_type" not in characteristics.columns:
            return np.arange(len(characteristics))
        mask = characteristics["trial_type"] != "S_C"
        return np.where(mask.to_numpy())[0]

    def export_brain_rdms_for_comparison(
        self,
        sessions: List[str],
        task: str = "Sem",
        exclude_controls: bool = True,
    ):
        """Save brain RDM matrices used for comparison (subsetted if needed)."""
        keep_idx = self._non_control_indices(task) if exclude_controls else None

        for session in sessions:
            brain_rdm_file = self.brain_rdm_dir / f"session_rdm_{session}.npz"
            if not brain_rdm_file.exists():
                logger.warning(f"Brain RDM not found for export: {brain_rdm_file}")
                continue

            brain_data = np.load(brain_rdm_file)
            brain_rdm = brain_data["rdm"]

            if exclude_controls and keep_idx is not None:
                if brain_rdm.shape[0] >= keep_idx.max() + 1:
                    brain_rdm = brain_rdm[np.ix_(keep_idx, keep_idx)]
                    out_file = self.output_dir / f"brain_rdm_{session}_{task}_no_controls.npz"
                    np.savez_compressed(out_file, rdm=brain_rdm, keep_indices=keep_idx)
                    logger.info(f"Saved subset brain RDM: {out_file} (shape={brain_rdm.shape})")
                else:
                    logger.warning(
                        f"Skipping subset export for {session}: index exceeds matrix size {brain_rdm.shape}"
                    )
            else:
                out_file = self.output_dir / f"brain_rdm_{session}_{task}_full.npz"
                np.savez_compressed(out_file, rdm=brain_rdm)
                logger.info(f"Saved full brain RDM: {out_file} (shape={brain_rdm.shape})")
    
    def compute_lm_rdm(
        self,
        model_name: str,
        task: str = "Sem",
        layer: int = -1,
        pooling: str = "mean",
        save: bool = True,
        exclude_controls: bool = True,
    ) -> Dict:
        """
        Compute RDM for a language model.
        
        Args:
            model_name: Model name (HuggingFace format)
            task: Task name
            layer: Which layer to extract
            pooling: Pooling strategy
            save: Whether to save RDM
            
        Returns:
            Dictionary with RDM info
        """
        logger.info(f"Computing RDM for model: {model_name}")
        
        try:
            # Optional format: <hf_repo>@<revision>
            revision = None
            model_id = model_name
            if "@" in model_name and not model_name.startswith("/"):
                parts = model_name.split("@", 1)
                if len(parts) == 2 and parts[0] and parts[1]:
                    model_id, revision = parts

            # Initialize extractor
            extractor = LanguageModelEmbeddingExtractor(
                model_name=model_id,
                layer=layer,
                pooling=pooling,
                revision=revision,
            )
            
            # Load characteristics
            characteristics, words, word_to_idx = (
                self.rdm_computer.load_stimulus_characteristics(
                    task=task,
                    exclude_controls=exclude_controls,
                )
            )
            
            # Extract embeddings
            logger.info(f"Extracting {len(words)} word embeddings")
            embeddings = extractor.extract_batch_embeddings(words)
            
            # Compute RDM
            logger.info("Computing RDM from stimulus characteristics")
            rdm = self.rdm_computer.compute_stimulus_rdm(
                characteristics, embeddings, words, word_to_idx
            )
            
            # Get statistics
            rdm_mean = rdm[np.triu_indices_from(rdm, k=1)].mean()
            rdm_std = rdm[np.triu_indices_from(rdm, k=1)].std()
            
            result = {
                "model": model_name,
                "model_id": model_id,
                "revision": revision,
                "task": task,
                "layer": layer,
                "pooling": pooling,
                "shape": rdm.shape,
                "mean_dissimilarity": float(rdm_mean),
                "std_dissimilarity": float(rdm_std),
                "rdm": rdm
            }
            
            logger.info(f"RDM shape: {rdm.shape}, mean dissim: {rdm_mean:.4f}")
            
            # Save if requested
            if save:
                model_safe_name = model_name.replace("/", "_")
                model_safe_name = model_safe_name.replace("@", "_at_")
                filename = f"lm_rdm_{model_safe_name}_{task}_layer{layer}.npz"
                self.rdm_computer.save_rdm(rdm, filename)
                result["filename"] = filename
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to compute RDM for {model_name}: {e}")
            return None
    
    def compare_rdms(
        self,
        lm_rdm: np.ndarray,
        brain_session: str = "ses-7",
        distance_metric: str = "spearman",
        task: str = "Sem",
        exclude_controls: bool = True,
    ) -> Dict:
        """
        Compare language model RDM with brain RDM.
        
        Args:
            lm_rdm: Language model RDM matrix
            brain_session: Brain session to compare with (ses-5, ses-7, ses-9)
            distance_metric: Distance metric for comparison (spearman, pearson, euclidean)
            
        Returns:
            Dictionary with correlation results
        """
        # Load brain RDM
        brain_rdm_file = self.brain_rdm_dir / f"session_rdm_{brain_session}.npz"
        if not brain_rdm_file.exists():
            logger.warning(f"Brain RDM not found: {brain_rdm_file}")
            return None
        
        brain_data = np.load(brain_rdm_file)
        brain_rdm = brain_data["rdm"]

        if exclude_controls:
            keep_idx = self._non_control_indices(task)
            if brain_rdm.shape[0] >= keep_idx.max() + 1:
                brain_rdm = brain_rdm[np.ix_(keep_idx, keep_idx)]
                logger.info(f"Using non-control brain RDM subset: {brain_rdm.shape}")
            else:
                logger.warning(
                    "Could not subset brain RDM to non-control trials; index exceeds matrix size"
                )
        
        logger.info(f"Comparing with brain RDM {brain_session}: {brain_rdm.shape}")
        
        # Ensure same size
        if lm_rdm.shape != brain_rdm.shape:
            logger.warning(
                f"Shape mismatch: LM {lm_rdm.shape} vs Brain {brain_rdm.shape}"
            )
            return None
        
        # Flatten upper triangles (avoid diagonal and redundancy)
        triu_indices = np.triu_indices_from(lm_rdm, k=1)
        lm_flat = lm_rdm[triu_indices]
        brain_flat = brain_rdm[triu_indices]
        
        # Compute correlation
        if distance_metric == "spearman":
            corr, pval = spearmanr(lm_flat, brain_flat)
        elif distance_metric == "pearson":
            corr = np.corrcoef(lm_flat, brain_flat)[0, 1]
            pval = 1 - corr  # Placeholder
        else:
            raise ValueError(f"Unknown metric: {distance_metric}")
        
        result = {
            "brain_session": brain_session,
            "correlation": float(corr),
            "p_value": float(pval),
            "metric": distance_metric,
            "n_comparisons": len(lm_flat)
        }
        
        logger.info(f"Correlation: {corr:.4f}, p-value: {pval:.4e}")
        
        return result
    
    def compute_all_models(
        self,
        model_names: List[str],
        task: str = "Sem",
        compare_sessions: List[str] = None,
        save: bool = True,
        exclude_controls: bool = True,
    ) -> Dict:
        """
        Compute RDMs for all models and compare with brain.
        
        Args:
            model_names: List of model names to compute
            task: Task name
            compare_sessions: Brain sessions to compare with
            save: Whether to save RDMs
            
        Returns:
            Results dictionary
        """
        if compare_sessions is None:
            compare_sessions = ["ses-7"]
        
        all_results = {
            "models": [],
            "comparisons": []
        }

        # Export brain RDM matrices that are actually used in LM comparison.
        self.export_brain_rdms_for_comparison(
            sessions=compare_sessions,
            task=task,
            exclude_controls=exclude_controls,
        )
        
        for model_name in model_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing model: {model_name}")
            logger.info(f"{'='*60}")
            
            # Compute RDM
            lm_result = self.compute_lm_rdm(
                model_name,
                task=task,
                save=save,
                exclude_controls=exclude_controls,
            )
            
            if lm_result is None:
                logger.error(f"Failed to compute RDM for {model_name}")
                continue
            
            # Remove RDM from result (too large for JSON)
            rdm = lm_result.pop("rdm")
            all_results["models"].append(lm_result)
            
            # Compare with brain RDMs
            for session in compare_sessions:
                logger.info(f"Comparing with {session}")
                comp_result = self.compare_rdms(
                    rdm,
                    brain_session=session,
                    task=task,
                    exclude_controls=exclude_controls,
                )
                
                if comp_result:
                    comp_result["model"] = model_name
                    all_results["comparisons"].append(comp_result)
        
        # Save results
        self.save_results(all_results)
        
        return all_results
    
    def save_results(self, results: Dict):
        """Save results to JSON."""
        output_file = self.output_dir / "language_model_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {output_file}")
    
    def plot_comparison(
        self,
        lm_rdm: np.ndarray,
        brain_session: str = "ses-7",
        model_name: str = "unknown",
        save: bool = True,
        task: str = "Sem",
        exclude_controls: bool = True,
    ):
        """
        Plot RDM heatmaps and comparison.
        
        Args:
            lm_rdm: Language model RDM
            brain_session: Brain session to compare
            model_name: Model name for title
            save: Whether to save plot
        """
        # Load brain RDM
        brain_rdm_file = self.brain_rdm_dir / f"session_rdm_{brain_session}.npz"
        if not brain_rdm_file.exists():
            logger.warning(f"Brain RDM not found: {brain_rdm_file}")
            return
        
        brain_data = np.load(brain_rdm_file)
        brain_rdm = brain_data["rdm"]

        if exclude_controls:
            keep_idx = self._non_control_indices(task)
            if brain_rdm.shape[0] >= keep_idx.max() + 1:
                brain_rdm = brain_rdm[np.ix_(keep_idx, keep_idx)]
        
        if lm_rdm.shape != brain_rdm.shape:
            logger.warning("Shape mismatch, skipping plot")
            return
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot RDMs
        sns.heatmap(lm_rdm, ax=axes[0], cmap="viridis", cbar=True)
        axes[0].set_title(f"Language Model RDM\n{model_name}")
        
        sns.heatmap(brain_rdm, ax=axes[1], cmap="viridis", cbar=True)
        axes[1].set_title(f"Brain RDM\n{brain_session}")
        
        plt.tight_layout()
        
        # Save
        if save:
            model_safe = model_name.replace("/", "_")
            output_file = self.output_dir / f"rdm_comparison_{model_safe}_{brain_session}.png"
            plt.savefig(output_file, dpi=150, bbox_inches="tight")
            logger.info(f"Saved plot to {output_file}")
        
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compute and compare language model RDMs with brain RDMs"
    )
    
    # BrainAlign BabyLM models from HuggingFace
    default_models = [
        "BrainAlign/gpt2-babylm-5",
        "BrainAlign/gpt2-babylm-7",
        "BrainAlign/gpt2-babylm-9"
    ]
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=default_models,
        help="Model names to compute RDMs for (HuggingFace format)"
    )
    parser.add_argument(
        "--task",
        default="Sem",
        help="Task name (Sem, Gram, Phon, Plaus)"
    )
    parser.add_argument(
        "--compare-sessions",
        nargs="+",
        default=["ses-5", "ses-7", "ses-9"],
        help="Brain sessions to compare with"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/language_models",
        help="Output directory"
    )
    parser.add_argument(
        "--brain-rdm-dir",
        default="data/processed/fmri",
        help="Directory with brain RDMs"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save RDMs to file"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots"
    )
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Include control trials (S_C). Default excludes them."
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = LanguageModelPipeline(
        output_dir=args.output_dir,
        brain_rdm_dir=args.brain_rdm_dir
    )
    
    # Compute RDMs for all models
    results = pipeline.compute_all_models(
        model_names=args.models,
        task=args.task,
        compare_sessions=args.compare_sessions,
        save=not args.no_save,
        exclude_controls=not args.include_controls,
    )
    
    # Generate plots if requested
    if args.plot:
        for model in args.models:
            try:
                extractor = LanguageModelEmbeddingExtractor(model_name=model)
                rdm_computer = LanguageModelRDMComputer(output_dir=args.output_dir)
                characteristics, words, word_to_idx = (
                    rdm_computer.load_stimulus_characteristics(
                        task=args.task,
                        exclude_controls=not args.include_controls,
                    )
                )
                embeddings = extractor.extract_batch_embeddings(words)
                rdm = rdm_computer.compute_stimulus_rdm(
                    characteristics, embeddings, words, word_to_idx
                )
                
                for session in args.compare_sessions:
                    pipeline.plot_comparison(
                        rdm,
                        brain_session=session,
                        model_name=model,
                        save=True,
                        task=args.task,
                        exclude_controls=not args.include_controls,
                    )
            except Exception as e:
                logger.error(f"Failed to plot for {model}: {e}")
    
    logger.info("\nPipeline complete!")
    print(f"\nResults saved to {pipeline.output_dir}")


if __name__ == "__main__":
    main()
