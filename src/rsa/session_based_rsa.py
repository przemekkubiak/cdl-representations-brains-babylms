"""
Session-based RSA Analysis

Aggregate RDMs across subjects within each session (ses-5, ses-7, ses-9).
This creates 3 RDMs representing neural representations at each timepoint.

Uses hyperalignment (Shared Response Model) to align subjects to a common
representational space before computing RDMs.
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
import json
import warnings

HYPERALIGNMENT_IMPORT_ERROR = None
try:
    from brainiak.funcalign.srm import SRM
    HYPERALIGNMENT_AVAILABLE = True
except Exception as e:
    HYPERALIGNMENT_AVAILABLE = False
    HYPERALIGNMENT_IMPORT_ERROR = str(e)
    warnings.warn(
        "Hyperalignment unavailable (BrainIAK/SRM import failed). "
        "Install MPI + mpi4py/BrainIAK or use --aggregation mean|median. "
        f"Details: {HYPERALIGNMENT_IMPORT_ERROR}"
    )

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rsa import compute_rdm, compare_rdms


class SessionBasedRSA:
    """
    Compute session-level RDMs aggregated across subjects.
    """
    
    def __init__(self, pattern_dir: str = "data/processed/fmri"):
        """
        Initialize session-based RSA analyzer.
        
        Parameters
        ----------
        pattern_dir : str
            Directory containing pattern .npz files from all subjects
        """
        self.pattern_dir = Path(pattern_dir)
        self.patterns_by_subject = {}
        self.session_rdms = {}
        
    def find_all_subjects(self) -> List[str]:
        """
        Find all subjects with processed patterns.
        
        Returns
        -------
        list
            List of subject IDs
        """
        pattern_files = list(self.pattern_dir.glob("sub-*_patterns.npz"))
        subjects = sorted(set([f.name.split("_")[0] for f in pattern_files]))
        return subjects

    @staticmethod
    def _stack_with_min_features(vectors: List[np.ndarray]) -> np.ndarray:
        """Stack 1D vectors after truncating all to the minimum length."""
        if not vectors:
            return np.array([])
        min_features = min(v.shape[0] for v in vectors)
        return np.vstack([v[:min_features] for v in vectors])
    
    def load_subject_patterns(
        self,
        subject_id: str,
        sessions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Load all pattern files for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject identifier
        sessions : list, optional
            Sessions to load (default: all)
            
        Returns
        -------
        dict
            Nested dictionary: {session: {run: {stim_file: pattern}}}
        """
        pattern_files = sorted(self.pattern_dir.glob(f"{subject_id}_*.npz"))
        
        if not pattern_files:
            return {}
        
        organized_data = {}
        
        for pf in pattern_files:
            # Parse filename: sub-5007_ses-7_run-01_patterns.npz
            parts = pf.stem.split("_")
            session = parts[1]  # ses-7
            run = parts[2]  # run-01
            
            # Filter sessions if specified
            if sessions and session not in sessions:
                continue
            
            # Load patterns
            data = np.load(str(pf))
            patterns = {}
            for key in data.keys():
                # Normalize to filename so the same stimulus can match across runs,
                # e.g., Sem/Sem_run-01/foo.wav and Sem/Sem_run-02/foo.wav -> foo.wav
                norm_key = Path(str(key)).name
                if norm_key in patterns:
                    warnings.warn(
                        f"Duplicate normalized stimulus key '{norm_key}' in {pf.name}; "
                        "keeping the first occurrence."
                    )
                    continue
                patterns[norm_key] = data[key]
            
            if session not in organized_data:
                organized_data[session] = {}
            organized_data[session][run] = patterns
        
        return organized_data
    
    def load_all_patterns(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
        """
        Load patterns for all subjects.
        
        Parameters
        ----------
        subjects : list, optional
            Subject IDs to load (default: all)
        sessions : list, optional
            Sessions to load (default: all)
            
        Returns
        -------
        dict
            Nested dictionary: {subject: {session: {run: {stim_file: pattern}}}}
        """
        all_subjects = self.find_all_subjects()
        
        if not all_subjects:
            raise ValueError(f"No pattern files found in {self.pattern_dir}")
        
        if subjects:
            all_subjects = [s for s in all_subjects if s in subjects]
        
        print(f"Loading patterns for {len(all_subjects)} subjects")
        print("=" * 70)
        
        patterns_by_subject = {}
        
        for subject_id in all_subjects:
            patterns = self.load_subject_patterns(subject_id, sessions=sessions)
            if patterns:
                patterns_by_subject[subject_id] = patterns
                print(f"  {subject_id}: {list(patterns.keys())}")
        
        print("=" * 70)
        
        self.patterns_by_subject = patterns_by_subject
        
        return patterns_by_subject
    
    def get_common_stimuli(
        self,
        subject_patterns: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> List[str]:
        """
        Find stimuli common to all subjects.
        
        Parameters
        ----------
        subject_patterns : dict
            Dictionary of subject patterns
            
        Returns
        -------
        list
            Sorted list of common stimulus files
        """
        # Collect all stimulus sets
        all_stim_sets = []
        
        for subject_data in subject_patterns.values():
            for session_data in subject_data.values():
                for run_data in session_data.values():
                    all_stim_sets.append(set(run_data.keys()))
        
        if not all_stim_sets:
            return []
        
        # Find intersection
        common_stim = sorted(set.intersection(*all_stim_sets))
        
        return common_stim
    
    def compute_subject_rdm(
        self,
        subject_id: str,
        session: str,
        common_stimuli: List[str],
        metric: str = "correlation"
    ) -> Optional[np.ndarray]:
        """
        Compute RDM for a single subject at a single session.
        Averages across runs within the session.
        
        Parameters
        ----------
        subject_id : str
            Subject ID
        session : str
            Session identifier (e.g., 'ses-7')
        common_stimuli : list
            List of stimuli to include (in order)
        metric : str
            Distance metric
            
        Returns
        -------
        np.ndarray or None
            RDM matrix, or None if subject doesn't have this session
        """
        if subject_id not in self.patterns_by_subject:
            return None
        
        subject_data = self.patterns_by_subject[subject_id]
        
        if session not in subject_data:
            return None
        
        # Average each stimulus across all runs where it is present.
        avg_patterns = []
        for stim in common_stimuli:
            stim_patterns = []
            for _, patterns in subject_data[session].items():
                if stim in patterns:
                    stim_patterns.append(patterns[stim])

            if not stim_patterns:
                return None

            stim_matrix = self._stack_with_min_features(stim_patterns)
            avg_patterns.append(np.mean(stim_matrix, axis=0))

        if not avg_patterns:
            return None

        avg_patterns = self._stack_with_min_features(avg_patterns)
        
        # Compute RDM
        rdm = compute_rdm(avg_patterns, metric=metric)
        
        return rdm
    
    def hyperalign_subjects(
        self,
        subject_patterns: List[np.ndarray],
        n_iter: int = 10,
        features: int = None
    ) -> Tuple[np.ndarray, List]:
        """
        Apply hyperalignment (SRM) to align subjects to common space.
        
        Parameters
        ----------
        subject_patterns : list of np.ndarray
            List of pattern matrices, one per subject (stimuli x voxels)
        n_iter : int
            Number of SRM iterations
        features : int, optional
            Number of shared features (default: min of subject dimensions)
            
        Returns
        -------
        aligned_patterns : np.ndarray
            Averaged patterns in shared space (stimuli x features)
        transformations : list
            Subject-specific transformation matrices
        """
        if not HYPERALIGNMENT_AVAILABLE:
            raise ImportError("BrainIAK not installed. Install with: pip install brainiak")
        
        # Prepare data: SRM expects list of (voxels x stimuli) arrays
        subject_data = [p.T for p in subject_patterns]  # Transpose to voxels x stimuli
        
        # Determine number of shared features
        if features is None:
            features = min(p.shape[0] for p in subject_data)  # Min across voxel dimensions
            features = min(features, subject_patterns[0].shape[0])  # Don't exceed n_stimuli
        
        print(f"    Hyperalignment: {len(subject_data)} subjects, {features} shared features")
        
        # Apply SRM
        srm = SRM(n_iter=n_iter, features=features)
        srm.fit(subject_data)
        
        # Transform subjects to shared space
        aligned_data = srm.transform(subject_data)
        
        # Average in shared space and transpose back to (stimuli x features)
        aligned_patterns = np.mean(aligned_data, axis=0).T
        
        return aligned_patterns, srm.w_
    
    def compute_session_rdm(
        self,
        session: str,
        metric: str = "correlation",
        aggregation: str = "hyperalignment",
        n_iter: int = 10,
        features: int = None
    ) -> Tuple[np.ndarray, List[str], int]:
        """
        Compute session-level RDM aggregated across subjects.
        
        Parameters
        ----------
        session : str
            Session identifier (e.g., 'ses-7')
        metric : str
            Distance metric for computing RDMs
        aggregation : str
            How to aggregate across subjects:
            - 'hyperalignment': Use SRM to align subjects to common space (default)
            - 'mean': Simple mean of RDMs
            - 'median': Median of RDMs
        n_iter : int
            Number of SRM iterations (only for hyperalignment)
        features : int, optional
            Number of shared features for SRM (default: auto)
            
        Returns
        -------
        rdm : np.ndarray
            Aggregated RDM for this session
        stimuli : list
            List of stimulus names
        n_subjects : int
            Number of subjects included
        """
        # Find common stimuli for this session only.
        # For each subject, take union across runs in this session,
        # then intersect across subjects.
        subject_session_stimuli = []
        for subject_data in self.patterns_by_subject.values():
            if session not in subject_data:
                continue

            run_sets = [
                set(run_data.keys())
                for run_data in subject_data[session].values()
                if run_data
            ]
            if not run_sets:
                continue

            subject_union = set.union(*run_sets)
            if subject_union:
                subject_session_stimuli.append(subject_union)

        if not subject_session_stimuli:
            raise ValueError(f"No stimuli found for session {session}")

        common_stimuli = sorted(set.intersection(*subject_session_stimuli))

        if not common_stimuli:
            raise ValueError(f"No common stimuli found across subjects for session {session}")
        
        print(f"\nComputing session RDM for {session}")
        print(f"  Common stimuli: {len(common_stimuli)}")
        print(f"  Aggregation: {aggregation}")
        
        if aggregation == "hyperalignment" and not HYPERALIGNMENT_AVAILABLE:
            warnings.warn(
                "Requested aggregation='hyperalignment' but SRM is unavailable. "
                "Falling back to aggregation='mean'."
            )
            aggregation = "mean"

        if aggregation == "hyperalignment":
            # Collect patterns from all subjects (before computing RDMs)
            subject_patterns = []
            subject_ids = []
            
            for subject_id in sorted(self.patterns_by_subject.keys()):
                subject_data = self.patterns_by_subject[subject_id]
                
                if session not in subject_data:
                    print(f"    {subject_id}: ✗ (no data)")
                    continue
                
                # Average each stimulus across all runs where it is present.
                avg_patterns = []
                for stim in common_stimuli:
                    stim_patterns = []
                    for _, patterns in subject_data[session].items():
                        if stim in patterns:
                            stim_patterns.append(patterns[stim])

                    if not stim_patterns:
                        avg_patterns = []
                        break

                    stim_matrix = self._stack_with_min_features(stim_patterns)
                    avg_patterns.append(np.mean(stim_matrix, axis=0))

                if avg_patterns:
                    avg_patterns = self._stack_with_min_features(avg_patterns)
                    subject_patterns.append(avg_patterns)
                    subject_ids.append(subject_id)
                    print(f"    {subject_id}: ✓ ({len(subject_data[session])} runs)")
                else:
                    print(f"    {subject_id}: ✗ (no valid runs)")
            
            if len(subject_patterns) < 2:
                raise ValueError(f"Hyperalignment requires at least 2 subjects, found {len(subject_patterns)}")
            
            # Apply hyperalignment
            aligned_patterns, _ = self.hyperalign_subjects(
                subject_patterns,
                n_iter=n_iter,
                features=features
            )
            
            # Compute RDM from aligned patterns
            session_rdm = compute_rdm(aligned_patterns, metric=metric)
            
            print(f"  Hyperaligned {len(subject_patterns)} subjects")
            print(f"  Aligned pattern shape: {aligned_patterns.shape}")
            print(f"  RDM shape: {session_rdm.shape}")
            print(f"  Mean dissimilarity: {session_rdm.mean():.4f}")
            
        else:
            # Original approach: compute RDMs then aggregate
            subject_rdms = []
            subject_ids = []
            
            for subject_id in sorted(self.patterns_by_subject.keys()):
                rdm = self.compute_subject_rdm(
                    subject_id=subject_id,
                    session=session,
                    common_stimuli=common_stimuli,
                    metric=metric
                )
                
                if rdm is not None:
                    subject_rdms.append(rdm)
                    subject_ids.append(subject_id)
                    print(f"    {subject_id}: ✓")
                else:
                    print(f"    {subject_id}: ✗ (no data)")
            
            if not subject_rdms:
                raise ValueError(f"No subjects have data for {session}")
            
            # Aggregate across subjects
            subject_rdms = np.array(subject_rdms)
            
            if aggregation == "mean":
                session_rdm = np.mean(subject_rdms, axis=0)
            elif aggregation == "median":
                session_rdm = np.median(subject_rdms, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            print(f"  Aggregated {len(subject_rdms)} subjects ({aggregation})")
            print(f"  RDM shape: {session_rdm.shape}")
            print(f"  Mean dissimilarity: {session_rdm.mean():.4f}")
        
        # Store
        self.session_rdms[session] = {
            "rdm": session_rdm,
            "stimuli": common_stimuli,
            "n_subjects": len(subject_ids),
            "subject_ids": subject_ids,
            "metric": metric,
            "aggregation": aggregation
        }
        
        return session_rdm, common_stimuli, len(subject_ids)
    
    def compute_all_sessions(
        self,
        sessions: Optional[List[str]] = None,
        metric: str = "correlation",
        aggregation: str = "hyperalignment",
        n_iter: int = 10,
        features: int = None
    ) -> Dict[str, Dict]:
        """
        Compute RDMs for all sessions.
        
        Parameters
        ----------
        sessions : list, optional
            Sessions to compute (default: all available)
        metric : str
            Distance metric
        aggregation : str
            Aggregation method ('hyperalignment', 'mean', or 'median')
        n_iter : int
            Number of SRM iterations (only for hyperalignment)
        features : int, optional
            Number of shared features for SRM
            
        Returns
        -------
        dict
            Dictionary of session RDM results
        """
        # Determine available sessions
        available_sessions = set()
        for subject_data in self.patterns_by_subject.values():
            available_sessions.update(subject_data.keys())
        
        if sessions:
            available_sessions = available_sessions.intersection(sessions)
        
        available_sessions = sorted(available_sessions)
        
        if not available_sessions:
            raise ValueError("No sessions available")
        
        print(f"\nComputing RDMs for sessions: {available_sessions}")
        print("=" * 70)
        
        for session in available_sessions:
            self.compute_session_rdm(
                session=session,
                metric=metric,
                aggregation=aggregation,
                n_iter=n_iter,
                features=features
            )
        
        print("=" * 70)
        
        return self.session_rdms
    
    def compare_sessions(
        self,
        method: str = "spearman"
    ) -> pd.DataFrame:
        """
        Compare RDMs across sessions.
        
        Parameters
        ----------
        method : str
            Correlation method ('spearman', 'pearson')
            
        Returns
        -------
        pd.DataFrame
            Comparison results
        """
        if not self.session_rdms:
            raise ValueError("No session RDMs computed")
        
        print(f"\nComparing session RDMs (method: {method})")
        print("=" * 70)
        
        sessions = sorted(self.session_rdms.keys())
        n = len(sessions)
        
        results = []
        
        for i in range(n):
            for j in range(i + 1, n):
                session1, session2 = sessions[i], sessions[j]
                rdm1 = self.session_rdms[session1]["rdm"]
                rdm2 = self.session_rdms[session2]["rdm"]
                
                corr, pval = compare_rdms(rdm1, rdm2, method=method)
                
                results.append({
                    "session1": session1,
                    "session2": session2,
                    "correlation": corr,
                    "p_value": pval
                })
                
                print(f"  {session1} vs {session2}: r = {corr:.4f}, p = {pval:.4e}")
        
        print("=" * 70)
        
        return pd.DataFrame(results)
    
    def visualize_rdm(
        self,
        session: str,
        output_path: Optional[str] = None,
        show_labels: bool = False,
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Visualize a session RDM.
        
        Parameters
        ----------
        session : str
            Session to visualize
        output_path : str, optional
            Path to save figure
        show_labels : bool
            Whether to show stimulus labels
        figsize : tuple
            Figure size
        """
        if session not in self.session_rdms:
            raise ValueError(f"No RDM computed for {session}")
        
        data = self.session_rdms[session]
        rdm = data["rdm"]
        stimuli = data["stimuli"]
        n_subjects = data["n_subjects"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(rdm, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Dissimilarity', rotation=270, labelpad=20)
        
        # Labels
        if show_labels and len(stimuli) <= 50:
            labels = [Path(s).stem for s in stimuli]
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_xlabel(f'Stimuli (n={len(stimuli)})')
            ax.set_ylabel(f'Stimuli (n={len(stimuli)})')
        
        title = f"Neural RDM - {session} (n={n_subjects} subjects)"
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_path}")
        
        plt.close(fig)
    
    def save_session_rdm(
        self,
        session: str,
        output_path: str
    ):
        """
        Save session RDM to disk.
        
        Parameters
        ----------
        session : str
            Session identifier
        output_path : str
            Output file path
        """
        if session not in self.session_rdms:
            raise ValueError(f"No RDM computed for {session}")
        
        data = self.session_rdms[session]
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "rdm": data["rdm"],
            "stimuli": np.array(data["stimuli"]),
            "n_subjects": data["n_subjects"],
            "subject_ids": np.array(data["subject_ids"]),
            "metric": data["metric"],
            "aggregation": data["aggregation"]
        }
        
        np.savez_compressed(str(output_path), **save_dict)
        print(f"Saved {session} RDM to: {output_path}")
    
    def save_all_results(self, output_dir: Optional[str] = None):
        """
        Save all session RDMs and visualizations.
        
        Parameters
        ----------
        output_dir : str, optional
            Output directory (default: pattern_dir)
        """
        if output_dir is None:
            output_dir = self.pattern_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\nSaving session RDMs...")
        print("=" * 70)
        
        for session in sorted(self.session_rdms.keys()):
            # Save RDM data
            rdm_path = output_dir / f"session_rdm_{session}.npz"
            self.save_session_rdm(session, str(rdm_path))
            
            # Save visualization
            viz_path = output_dir / f"session_rdm_{session}.png"
            self.visualize_rdm(session, output_path=str(viz_path))
        
        print("=" * 70)


def main():
    """Run session-based RSA analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Session-based RSA analysis")
    parser.add_argument(
        "--pattern-dir",
        type=str,
        default="data/processed/fmri",
        help="Directory containing pattern files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: same as pattern-dir)"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to include (default: all)"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=["ses-5", "ses-7", "ses-9"],
        help="Sessions to analyze (default: all)"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="correlation",
        choices=["correlation", "euclidean", "cosine"],
        help="Distance metric (default: correlation)"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="hyperalignment",
        choices=["hyperalignment", "mean", "median"],
        help="Aggregation method (default: hyperalignment)"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of SRM iterations for hyperalignment (default: 10)"
    )
    parser.add_argument(
        "--features",
        type=int,
        help="Number of shared features for hyperalignment (default: auto)"
    )
    
    args = parser.parse_args()
    
    # Initialize
    rsa = SessionBasedRSA(pattern_dir=args.pattern_dir)
    
    # Load patterns
    rsa.load_all_patterns(subjects=args.subjects, sessions=args.sessions)
    
    # Compute session RDMs
    session_rdms = rsa.compute_all_sessions(
        sessions=args.sessions,
        metric=args.metric,
        aggregation=args.aggregation,
        n_iter=args.n_iter,
        features=args.features
    )
    
    # Compare sessions
    if len(session_rdms) > 1:
        comparison_df = rsa.compare_sessions(method="spearman")
        
        # Save comparison
        output_dir = Path(args.output_dir) if args.output_dir else rsa.pattern_dir
        comparison_path = output_dir / "session_rdm_comparison.csv"
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nSaved comparison to: {comparison_path}")
    
    # Save all results
    rsa.save_all_results(output_dir=args.output_dir)
    
    print("\n" + "=" * 70)
    print("Session-based RSA analysis complete!")
    print(f"Created {len(session_rdms)} session-level RDMs")
    print("=" * 70)


if __name__ == "__main__":
    main()
