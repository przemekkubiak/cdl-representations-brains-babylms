"""
Neural RSA Analysis

Compute and analyze Representational Dissimilarity Matrices (RDMs) from fMRI data.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import pandas as pd
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rsa import compute_rdm, compare_rdms


class NeuralRSA:
    """
    Compute and analyze neural RDMs from fMRI patterns.
    """
    
    def __init__(self, pattern_dir: str = "data/processed/fmri"):
        """
        Initialize neural RSA analyzer.
        
        Parameters
        ----------
        pattern_dir : str
            Directory containing pattern .npz files
        """
        self.pattern_dir = Path(pattern_dir)
        self.patterns_data = {}
        self.rdms = {}
        
    def load_patterns(self, subject_id: str = "sub-5007") -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load all pattern files for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
            
        Returns
        -------
        dict
            Nested dictionary: {session: {run: {stim_file: pattern}}}
        """
        pattern_files = sorted(self.pattern_dir.glob(f"{subject_id}_*.npz"))
        
        if not pattern_files:
            raise ValueError(f"No pattern files found for {subject_id}")
        
        print(f"Loading patterns for {subject_id}")
        print("=" * 70)
        
        organized_data = {}
        
        for pf in pattern_files:
            # Parse filename: sub-5007_ses-7_run-01_patterns.npz
            parts = pf.stem.split("_")
            session = parts[1]  # ses-7
            run = parts[2]  # run-01
            
            # Load patterns
            data = np.load(str(pf))
            patterns = {key: data[key] for key in data.keys()}
            
            if session not in organized_data:
                organized_data[session] = {}
            organized_data[session][run] = patterns
            
            print(f"  {pf.name}: {len(patterns)} stimuli")
        
        self.patterns_data = organized_data
        print("=" * 70)
        
        return organized_data
    
    def get_common_stimuli(self, patterns_dict: Dict[str, np.ndarray]) -> List[str]:
        """
        Get list of stimulus files, sorted consistently.
        
        Parameters
        ----------
        patterns_dict : dict
            Dictionary mapping stimulus files to patterns
            
        Returns
        -------
        list
            Sorted list of stimulus files
        """
        return sorted(patterns_dict.keys())
    
    def patterns_to_matrix(
        self,
        patterns_dict: Dict[str, np.ndarray],
        stimuli: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Convert patterns dictionary to matrix.
        
        Parameters
        ----------
        patterns_dict : dict
            Dictionary mapping stimulus files to patterns
        stimuli : list, optional
            List of stimuli to include (in order)
            
        Returns
        -------
        np.ndarray
            Matrix of shape (n_stimuli, n_voxels)
        """
        if stimuli is None:
            stimuli = self.get_common_stimuli(patterns_dict)
        
        return np.array([patterns_dict[stim] for stim in stimuli])
    
    def compute_neural_rdm(
        self,
        session: str,
        run: str,
        metric: str = "correlation"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute neural RDM for a specific session and run.
        
        Parameters
        ----------
        session : str
            Session identifier (e.g., 'ses-7')
        run : str
            Run identifier (e.g., 'run-01')
        metric : str
            Distance metric ('correlation', 'euclidean', 'cosine')
            
        Returns
        -------
        rdm : np.ndarray
            Representational Dissimilarity Matrix
        stimuli : list
            List of stimulus names (in RDM order)
        """
        if not self.patterns_data:
            raise ValueError("No patterns loaded. Call load_patterns() first.")
        
        if session not in self.patterns_data:
            raise ValueError(f"Session {session} not found")
        
        if run not in self.patterns_data[session]:
            raise ValueError(f"Run {run} not found in {session}")
        
        patterns_dict = self.patterns_data[session][run]
        stimuli = self.get_common_stimuli(patterns_dict)
        
        # Convert to matrix
        patterns_matrix = self.patterns_to_matrix(patterns_dict, stimuli)
        
        # Compute RDM
        rdm = compute_rdm(patterns_matrix, metric=metric)
        
        # Store
        key = f"{session}_{run}"
        self.rdms[key] = {"rdm": rdm, "stimuli": stimuli, "metric": metric}
        
        return rdm, stimuli
    
    def compute_all_rdms(self, metric: str = "correlation") -> Dict[str, Dict]:
        """
        Compute RDMs for all sessions and runs.
        
        Parameters
        ----------
        metric : str
            Distance metric
            
        Returns
        -------
        dict
            Dictionary of RDM results
        """
        print(f"\nComputing neural RDMs (metric: {metric})")
        print("=" * 70)
        
        for session in self.patterns_data:
            for run in self.patterns_data[session]:
                rdm, stimuli = self.compute_neural_rdm(session, run, metric=metric)
                print(f"  {session} {run}: RDM shape {rdm.shape}, {len(stimuli)} stimuli")
        
        print("=" * 70)
        
        return self.rdms
    
    def average_rdms(
        self,
        sessions: Optional[List[str]] = None,
        runs: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Average RDMs across sessions and/or runs.
        
        Parameters
        ----------
        sessions : list, optional
            Sessions to include (default: all)
        runs : list, optional
            Runs to include (default: all)
            
        Returns
        -------
        rdm_avg : np.ndarray
            Averaged RDM
        stimuli : list
            List of stimulus names
        """
        if not self.rdms:
            raise ValueError("No RDMs computed. Call compute_all_rdms() first.")
        
        # Filter RDMs
        selected_rdms = []
        stimuli = None
        
        for key, data in self.rdms.items():
            session, run = key.split("_")
            
            if sessions is not None and session not in sessions:
                continue
            if runs is not None and run not in runs:
                continue
            
            selected_rdms.append(data["rdm"])
            
            if stimuli is None:
                stimuli = data["stimuli"]
        
        if not selected_rdms:
            raise ValueError("No RDMs match the selection criteria")
        
        # Average
        rdm_avg = np.mean(selected_rdms, axis=0)
        
        print(f"\nAveraged {len(selected_rdms)} RDMs")
        print(f"  Shape: {rdm_avg.shape}")
        print(f"  Mean dissimilarity: {rdm_avg.mean():.4f}")
        print(f"  Std dissimilarity: {rdm_avg.std():.4f}")
        
        return rdm_avg, stimuli
    
    def visualize_rdm(
        self,
        rdm: np.ndarray,
        stimuli: List[str],
        title: str = "Neural RDM",
        output_path: Optional[str] = None,
        show_labels: bool = False,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Visualize an RDM as a heatmap.
        
        Parameters
        ----------
        rdm : np.ndarray
            RDM to visualize
        stimuli : list
            Stimulus labels
        title : str
            Plot title
        output_path : str, optional
            Path to save figure
        show_labels : bool
            Whether to show stimulus labels
        figsize : tuple
            Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(rdm, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Dissimilarity', rotation=270, labelpad=20)
        
        # Labels
        if show_labels and len(stimuli) <= 50:
            # Simplify stimulus names for display
            labels = [Path(s).stem for s in stimuli]
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_xlabel(f'Stimuli (n={len(stimuli)})')
            ax.set_ylabel(f'Stimuli (n={len(stimuli)})')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")
        
        plt.close(fig)  # Close figure instead of showing it
    
    def compare_within_subject(
        self,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Compare RDMs within subject (across sessions/runs).
        
        Parameters
        ----------
        method : str
            Correlation method ('spearman', 'pearson')
            
        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        if not self.rdms:
            raise ValueError("No RDMs computed. Call compute_all_rdms() first.")
        
        print(f"\nComparing RDMs within subject (method: {method})")
        print("=" * 70)
        
        keys = list(self.rdms.keys())
        n = len(keys)
        
        results = []
        
        for i in range(n):
            for j in range(i + 1, n):
                key1, key2 = keys[i], keys[j]
                rdm1 = self.rdms[key1]["rdm"]
                rdm2 = self.rdms[key2]["rdm"]
                
                corr, pval = compare_rdms(rdm1, rdm2, method=method)
                
                results.append({
                    "RDM1": key1,
                    "RDM2": key2,
                    "correlation": corr,
                    "p_value": pval
                })
                
                print(f"  {key1} vs {key2}: r = {corr:.4f}, p = {pval:.4e}")
        
        print("=" * 70)
        
        return pd.DataFrame(results)
    
    def save_rdm(
        self,
        rdm: np.ndarray,
        stimuli: List[str],
        output_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Save RDM to disk.
        
        Parameters
        ----------
        rdm : np.ndarray
            RDM to save
        stimuli : list
            Stimulus names
        output_path : str
            Output file path
        metadata : dict, optional
            Additional metadata
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "rdm": rdm,
            "stimuli": np.array(stimuli)
        }
        
        if metadata:
            save_dict.update(metadata)
        
        np.savez_compressed(str(output_path), **save_dict)
        print(f"\nSaved RDM to: {output_path}")


def main():
    """Run neural RSA analysis."""
    # Initialize
    rsa = NeuralRSA(pattern_dir="data/processed/fmri")
    
    # Load patterns
    patterns = rsa.load_patterns(subject_id="sub-5007")
    
    # Compute RDMs for all sessions/runs
    rdms = rsa.compute_all_rdms(metric="correlation")
    
    # Compare RDMs within subject
    comparison_df = rsa.compare_within_subject(method="spearman")
    
    # Average across all runs
    print("\n" + "=" * 70)
    print("Computing averaged RDM across all runs")
    print("=" * 70)
    rdm_avg, stimuli = rsa.average_rdms()
    
    # Visualize averaged RDM
    print("\nVisualizing averaged RDM...")
    rsa.visualize_rdm(
        rdm=rdm_avg,
        stimuli=stimuli,
        title="Neural RDM (averaged across all runs)",
        output_path="data/processed/fmri/neural_rdm_averaged.png",
        figsize=(12, 10)
    )
    
    # Visualize individual RDMs
    print("\nVisualizing individual RDMs...")
    for key, data in rdms.items():
        rsa.visualize_rdm(
            rdm=data["rdm"],
            stimuli=data["stimuli"],
            title=f"Neural RDM ({key})",
            output_path=f"data/processed/fmri/neural_rdm_{key}.png",
            figsize=(10, 8)
        )
    
    # Save averaged RDM
    rsa.save_rdm(
        rdm=rdm_avg,
        stimuli=stimuli,
        output_path="data/processed/fmri/neural_rdm_averaged.npz",
        metadata={
            "subject": "sub-5007",
            "n_runs": len(rdms),
            "metric": "correlation"
        }
    )
    
    # Save comparison results
    comparison_df.to_csv(
        "data/processed/fmri/rdm_comparison.csv",
        index=False
    )
    print("\nSaved comparison results to: data/processed/fmri/rdm_comparison.csv")
    
    print("\n" + "=" * 70)
    print("Neural RSA analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
