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
        task: str = "Sem",
        dataset: str = "ds003604",
        smoothing_fwhm: float = 6.0,
        high_pass: float = 0.01,
        use_glm: bool = True,
        mask_path: Optional[str] = None,
        roi_set: Optional[str] = None,
        mask_cache_dir: Optional[str] = None,
        mni_template_resolution: int = 2,
        save_native_maps: bool = False,
    ):
        """
        Initialize batch preprocessor.

        Parameters
        ----------
        data_dir : str
            Path to BIDS dataset directory
        output_dir : str
            Path to save processed patterns
        task : str
            A PHENOMENON key (e.g. "Sem", "Phon") -- see
            FMRIPreprocessor.__init__ for how this resolves to real BIDS task
            name(s) via the registry.
        dataset : str
            Registry key from configs/neuro_datasets.yaml (default: "ds003604").
        smoothing_fwhm : float
            Spatial smoothing FWHM in mm
        high_pass : float
            High-pass filter cutoff in Hz
        use_glm : bool
            Use GLM with HRF modeling
        mask_path : str, optional
            Path to a static NIfTI mask, already in BOLD space. See
            FMRIPreprocessor's docstring -- mutually exclusive with roi_set.
        roi_set : str, optional
            Comma-separated named ROI sets (see
            src.preprocessing.roi_atlas.ROI_SETS), e.g. "auditory,motor".
            Triggers real per-subject registration -- see
            FMRIPreprocessor.__init__ and MASKING.md.
        mask_cache_dir : str, optional
            Shared cache dir for per-subject-session registrations and ROI
            masks. Required if roi_set is set; pass the SAME directory across
            every task for a dataset (e.g. one level above the per-task
            output dirs) so registration is computed once per subject-session,
            not once per task.
        mni_template_resolution : int
            MNI152 template resolution (mm) for T1->MNI registration.
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.task = task
        self.dataset = dataset
        self.smoothing_fwhm = smoothing_fwhm
        self.high_pass = high_pass
        self.use_glm = use_glm
        self.mask_path = mask_path
        self.roi_set = roi_set
        self.mask_cache_dir = mask_cache_dir
        self.mni_template_resolution = mni_template_resolution
        self.save_native_maps = save_native_maps

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
        from src.datasets import get_dataset
        from src.preprocessing.fmri_preprocessing import SESSIONLESS_LABEL

        subject_dir = self.data_dir / subject_id
        sessions_runs = {}

        # self.task is a PHENOMENON key; resolve to real BIDS task name(s) the
        # same way FMRIPreprocessor.find_task_runs does, so this reporting
        # function actually finds what process_subject is about to process
        # instead of gating it on the old ds003604-only assumption that
        # phenomenon == BIDS task name.
        real_tasks = get_dataset(self.dataset).phenomena.get(self.task) or [self.task]

        session_dirs = sorted(subject_dir.glob("ses-*"))
        session_targets = (
            [(d.name, d / "func") for d in session_dirs]
            if session_dirs else [(SESSIONLESS_LABEL, subject_dir / "func")]
        )

        for session_label, func_dir in session_targets:
            if not func_dir.exists():
                continue

            bold_files = []
            for real_task in real_tasks:
                bold_files.extend(sorted(func_dir.glob(f"*task-{real_task}*_bold.nii.gz")))

            if bold_files:
                runs = []
                for bold_file in bold_files:
                    # Extract run number
                    for part in bold_file.stem.split("_"):
                        if part.startswith("run-"):
                            runs.append(part)
                            break

                sessions_runs[session_label] = runs

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
            print(f"  No {self.task} task data found for {subject_id}")
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
                task=self.task,
                dataset=self.dataset,
                smoothing_fwhm=self.smoothing_fwhm,
                high_pass=self.high_pass,
                use_glm=self.use_glm,
                mask_path=self.mask_path,
                roi_set=self.roi_set,
                mask_cache_dir=self.mask_cache_dir,
                mni_template_resolution=self.mni_template_resolution,
                save_native_maps=self.save_native_maps,
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
        default=None,
        help="Path to BIDS dataset directory (default: data/brain/<--dataset>)"
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
        help="Sessions to process (default: all). Dataset-specific labels "
             "(e.g. ses-5/ses-7/ses-9 for ds003604, ses-T1/ses-T2 for "
             "ds001894); no longer restricted to ds003604's own set."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Sem",
        help="Phenomenon to process (default: Sem). Validated against "
             "--dataset's registered phenomena below, not a fixed list -- "
             "each dataset declares its own in configs/neuro_datasets.yaml."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ds003604",
        help="Registry key from configs/neuro_datasets.yaml (default: ds003604)."
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
        help="Path to a static NIfTI mask, already in BOLD space. Mutually "
             "exclusive with --roi-set -- see FMRIPreprocessor's docstring."
    )
    parser.add_argument(
        "--roi-set",
        type=str,
        default=None,
        help="Comma-separated named ROI sets to restrict analysis to, e.g. "
             "'auditory,motor'. Triggers per-subject registration to MNI152 "
             "-- see src/preprocessing/roi_atlas.py and MASKING.md. Omit for "
             "the default whole-brain mask (still gets the mask_strategy="
             "'epi' fix regardless of this flag)."
    )
    parser.add_argument(
        "--mask-cache-dir",
        type=str,
        default=None,
        help="Shared cache dir for per-subject-session registrations/ROI "
             "masks. Required with --roi-set; pass the SAME directory across "
             "every task for a dataset."
    )
    parser.add_argument(
        "--mni-template-resolution",
        type=int,
        default=2,
        choices=[1, 2],
        help="MNI152 template resolution in mm for T1->MNI registration (default: 2)"
    )
    parser.add_argument(
        "--save-native-maps",
        action="store_true",
        help="Save each subject-session's whole-brain mask (native EPI space) "
             "to --mask-cache-dir, once per session. Needed for real spatial "
             "brain figures (scripts/render_brain_atlas_figures.py); off by "
             "default since nothing else needs it. Requires --mask-cache-dir "
             "(does not require --roi-set)."
    )

    args = parser.parse_args()

    from src.datasets import get_dataset
    spec = get_dataset(args.dataset)
    if args.task not in spec.phenomena:
        parser.error(
            f"--task '{args.task}' is not a registered phenomenon for "
            f"--dataset '{args.dataset}'. Known: {sorted(spec.phenomena)} "
            f"(see configs/neuro_datasets.yaml)."
        )
    data_dir = args.data_dir if args.data_dir is not None else str(spec.data_dir())

    # Initialize batch preprocessor
    batch = BatchPreprocessor(
        data_dir=data_dir,
        output_dir=args.output_dir,
        task=args.task,
        dataset=args.dataset,
        smoothing_fwhm=args.smoothing_fwhm,
        high_pass=args.high_pass,
        use_glm=not args.no_glm,
        mask_path=args.mask_path,
        roi_set=args.roi_set,
        mask_cache_dir=args.mask_cache_dir,
        mni_template_resolution=args.mni_template_resolution,
        save_native_maps=args.save_native_maps,
    )

    # Process subjects
    batch.process_all_subjects(
        subjects=args.subjects,
        sessions=args.sessions
    )


if __name__ == "__main__":
    main()
