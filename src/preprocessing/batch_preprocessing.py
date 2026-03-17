"""
Batch preprocessing for multiple subjects.

This script processes all subjects in the dataset, handling variable
sessions (ses-5, ses-7, ses-9) and runs (run-01, run-02, etc.).
"""

import os
import sys
from pathlib import Path
import argparse
import json
from typing import List, Dict, Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.preprocessing.fmri_preprocessing import FMRIPreprocessor


class BatchPreprocessor:
    """Process multiple subjects with error handling."""
    
    def __init__(
        self,
        data_dir: str = "data/brain/ds003604",
        output_dir: str = "data/processed/fmri",
        smoothing_fwhm: float = 6.0,
        high_pass: float = 0.01,
        use_glm: bool = True,
        mask_path: Optional[str] = None
    ):
        """
        Initialize batch preprocessor.
        
        Parameters
        ----------
        data_dir : str
            Path to BIDS dataset directory
        output_dir : str
            Path to save processed patterns
        smoothing_fwhm : float
            Spatial smoothing FWHM in mm
        high_pass : float
            High-pass filter cutoff in Hz
        use_glm : bool
            Use GLM with HRF modeling
        mask_path : str, optional
            Path to NIfTI mask used to restrict voxel selection.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.smoothing_fwhm = smoothing_fwhm
        self.high_pass = high_pass
        self.use_glm = use_glm
        self.mask_path = mask_path
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_subjects(self) -> List[str]:
        """
        Find all subject directories in the dataset.
        
        Returns
        -------
        list
            List of subject IDs (e.g., ['sub-5007', 'sub-5008', ...])
        """
        subject_dirs = sorted(self.data_dir.glob("sub-*"))
        subjects = [d.name for d in subject_dirs if d.is_dir() and not d.name.startswith('.')]
        return subjects
    
    def check_subject_sessions(self, subject_id: str) -> Dict[str, List[str]]:
        """
        Check what sessions and runs exist for a subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID
            
        Returns
        -------
        dict
            Dictionary mapping sessions to lists of runs
        """
        subject_dir = self.data_dir / subject_id
        sessions_runs = {}
        
        for session_dir in sorted(subject_dir.glob("ses-*")):
            func_dir = session_dir / "func"
            if not func_dir.exists():
                continue
            
            # Find semantic task BOLD files
            bold_files = sorted(func_dir.glob("*task-Sem*_bold.nii.gz"))
            
            if bold_files:
                runs = []
                for bold_file in bold_files:
                    # Extract run number
                    for part in bold_file.stem.split("_"):
                        if part.startswith("run-"):
                            runs.append(part)
                            break
                
                sessions_runs[session_dir.name] = runs
        
        return sessions_runs
    
    def process_subject(
        self,
        subject_id: str,
        sessions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process a single subject.
        
        Parameters
        ----------
        subject_id : str
            Subject ID
        sessions : list, optional
            List of sessions to process (default: all available)
            
        Returns
        -------
        dict
            Nested dictionary: {session: {run: {stim_file: pattern}}}
        """
        print(f"\n{'='*70}")
        print(f"Processing {subject_id}")
        print(f"{'='*70}")
        
        # Check what data exists
        sessions_runs = self.check_subject_sessions(subject_id)
        
        if not sessions_runs:
            print(f"  No semantic task data found for {subject_id}")
            return {}
        
        print(f"  Found sessions: {list(sessions_runs.keys())}")
        for session, runs in sessions_runs.items():
            print(f"    {session}: {runs}")
        
        # Filter sessions if specified
        if sessions:
            sessions_runs = {s: r for s, r in sessions_runs.items() if s in sessions}
            if not sessions_runs:
                print(f"  No matching sessions for {subject_id}")
                return {}
        
        # Initialize preprocessor
        try:
            preprocessor = FMRIPreprocessor(
                data_dir=str(self.data_dir),
                subject_id=subject_id,
                smoothing_fwhm=self.smoothing_fwhm,
                high_pass=self.high_pass,
                use_glm=self.use_glm,
                mask_path=self.mask_path
            )
            
            # Process all runs
            all_patterns = preprocessor.process_all_runs(
                output_dir=str(self.output_dir),
                save_results=True
            )
            
            return all_patterns
            
        except Exception as e:
            print(f"  ERROR processing {subject_id}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def process_all_subjects(
        self,
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Process all subjects in the dataset.
        
        Parameters
        ----------
        subjects : list, optional
            List of subject IDs to process (default: all)
        sessions : list, optional
            List of sessions to process (default: all)
            
        Returns
        -------
        dict
            Nested dictionary: {subject: {session: {run: {stim_file: pattern}}}}
        """
        # Find subjects
        all_subjects = self.find_subjects()
        
        if not all_subjects:
            raise ValueError(f"No subjects found in {self.data_dir}")
        
        print(f"Found {len(all_subjects)} subjects: {all_subjects}")
        
        # Filter if specified
        if subjects:
            all_subjects = [s for s in all_subjects if s in subjects]
            print(f"Processing {len(all_subjects)} subjects: {all_subjects}")
        
        # Process each subject
        results = {}
        successful = 0
        failed = 0
        
        for subject_id in all_subjects:
            patterns = self.process_subject(subject_id, sessions=sessions)
            
            if patterns:
                results[subject_id] = patterns
                successful += 1
            else:
                failed += 1
        
        # Summary
        print(f"\n{'='*70}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*70}")
        print(f"Total subjects: {len(all_subjects)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"{'='*70}")
        
        # Save summary
        self._save_summary(results)
        
        return results
    
    def _save_summary(self, results: Dict):
        """Save processing summary to JSON."""
        summary = {
            "n_subjects": len(results),
            "subjects": {}
        }
        
        for subject_id, patterns in results.items():
            summary["subjects"][subject_id] = {
                "sessions": list(patterns.keys()),
                "n_sessions": len(patterns),
                "runs_per_session": {
                    session: list(runs.keys())
                    for session, runs in patterns.items()
                }
            }
        
        summary_path = self.output_dir / "preprocessing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSaved summary to: {summary_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch fMRI preprocessing")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/brain/ds003604",
        help="Path to BIDS dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/fmri",
        help="Path to save processed patterns"
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject IDs to process (default: all)"
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        choices=["ses-5", "ses-7", "ses-9"],
        help="Sessions to process (default: all)"
    )
    parser.add_argument(
        "--smoothing-fwhm",
        type=float,
        default=6.0,
        help="Spatial smoothing FWHM in mm (default: 6.0)"
    )
    parser.add_argument(
        "--high-pass",
        type=float,
        default=0.01,
        help="High-pass filter cutoff in Hz (default: 0.01)"
    )
    parser.add_argument(
        "--no-glm",
        action="store_true",
        help="Disable GLM modeling (use simple averaging)"
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        help="Path to NIfTI mask for language-responsive voxels"
    )
    
    args = parser.parse_args()
    
    # Initialize batch preprocessor
    batch = BatchPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        smoothing_fwhm=args.smoothing_fwhm,
        high_pass=args.high_pass,
        use_glm=not args.no_glm,
        mask_path=args.mask_path
    )
    
    # Process subjects
    batch.process_all_subjects(
        subjects=args.subjects,
        sessions=args.sessions
    )


if __name__ == "__main__":
    main()
