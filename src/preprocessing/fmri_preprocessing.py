"""
fMRI preprocessing module for extracting stimulus-specific brain activity.

This module handles:
1. Loading BOLD fMRI data (.nii.gz files)
2. Loading event timing information (.tsv files)
3. Spatial smoothing
4. High-pass filtering
5. GLM modeling with hemodynamic response
6. Extracting stimulus-specific brain activity patterns
7. Preparing data for RSA analysis
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, List, Tuple, Optional
from nilearn import image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.maskers import NiftiMasker
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class FMRIPreprocessor:
    """
    Preprocessor for fMRI data to extract stimulus-specific brain activity.
    Includes spatial smoothing, high-pass filtering, and GLM modeling.
    """
    
    def __init__(
        self, 
        data_dir: str, 
        subject_id: str = "sub-5007",
        smoothing_fwhm: float = 6.0,
        high_pass: float = 0.01,
        use_glm: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Parameters
        ----------
        data_dir : str
            Path to the brain data directory
        subject_id : str
            Subject identifier (default: "sub-5007")
        smoothing_fwhm : float
            Full-width at half-maximum for Gaussian smoothing in mm (default: 6.0)
        high_pass : float
            High-pass filter cutoff in Hz (default: 0.01, i.e., 1/100s)
            Note: When use_glm=True, filtering is done by GLM's drift model
        use_glm : bool
            Whether to use GLM modeling (default: True)
        """
        self.data_dir = Path(data_dir)
        self.subject_id = subject_id
        self.subject_dir = self.data_dir / subject_id
        self.smoothing_fwhm = smoothing_fwhm
        self.high_pass = high_pass
        self.use_glm = use_glm
        
        if not self.subject_dir.exists():
            raise ValueError(f"Subject directory not found: {self.subject_dir}")
    
    def find_semantic_runs(self) -> List[Dict[str, Path]]:
        """
        Find all semantic task runs across sessions.
        
        Returns
        -------
        list of dict
            List of dictionaries containing 'bold', 'events', 'session', 'run' info
        """
        runs = []
        
        # Iterate through sessions (ses-5, ses-7, ses-9)
        for session_dir in sorted(self.subject_dir.glob("ses-*")):
            func_dir = session_dir / "func"
            if not func_dir.exists():
                continue
            
            # Find all semantic task BOLD files
            for bold_file in sorted(func_dir.glob("*task-Sem*_bold.nii.gz")):
                # Construct corresponding events file
                events_file = bold_file.parent / bold_file.name.replace("_bold.nii.gz", "_events.tsv")
                
                if events_file.exists():
                    run_info = {
                        "bold": bold_file,
                        "events": events_file,
                        "session": session_dir.name,
                        "run": self._extract_run_number(bold_file.name)
                    }
                    runs.append(run_info)
        
        return runs
    
    @staticmethod
    def _extract_run_number(filename: str) -> str:
        """Extract run number from filename."""
        for part in filename.split("_"):
            if part.startswith("run-"):
                return part
        return "unknown"
    
    def load_events(self, events_file: Path) -> pd.DataFrame:
        """
        Load events file.
        
        Parameters
        ----------
        events_file : Path
            Path to events.tsv file
        
        Returns
        -------
        pd.DataFrame
            Events dataframe with timing and stimulus information
        """
        df = pd.read_csv(events_file, sep="\t")
        
        # Clean up stimulus file names
        df['stim_file'] = df['stim_file'].str.strip()
        
        return df
    
    def load_bold(self, bold_file: Path) -> nib.Nifti1Image:
        """
        Load BOLD fMRI data.
        
        Parameters
        ----------
        bold_file : Path
            Path to BOLD .nii.gz file
        
        Returns
        -------
        img : nibabel.Nifti1Image
            Nibabel image object
        """
        img = nib.load(str(bold_file))
        return img
    
    def preprocess_functional(
        self,
        bold_img: nib.Nifti1Image,
        tr: float,
        verbose: bool = True
    ) -> nib.Nifti1Image:
        """
        Apply spatial smoothing to BOLD data.
        Note: High-pass filtering is handled by GLM's drift model for efficiency.
        
        Parameters
        ----------
        bold_img : nibabel.Nifti1Image
            BOLD image
        tr : float
            Repetition time in seconds
        verbose : bool
            Print processing steps
        
        Returns
        -------
        preprocessed_img : nibabel.Nifti1Image
            Preprocessed BOLD image
        """
        if verbose:
            print(f"  Preprocessing:")
        
        # Spatial smoothing
        if self.smoothing_fwhm > 0:
            if verbose:
                print(f"    - Spatial smoothing (FWHM={self.smoothing_fwhm}mm)")
            bold_img = image.smooth_img(bold_img, fwhm=self.smoothing_fwhm)
        
        # Note: High-pass filtering done by GLM for efficiency
        if verbose and self.use_glm:
            print(f"    - High-pass filtering will be handled by GLM (cutoff={self.high_pass}Hz)")
        
        return bold_img
    
    def extract_stimulus_activity_glm(
        self,
        bold_img: nib.Nifti1Image,
        events_df: pd.DataFrame,
        tr: float,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract stimulus-specific activity using GLM with proper HRF modeling.
        
        Parameters
        ----------
        bold_img : nibabel.Nifti1Image
            Preprocessed BOLD image
        events_df : pd.DataFrame
            Events dataframe with onset, duration, stim_file columns
        tr : float
            Repetition time in seconds
        verbose : bool
            Print processing steps
        
        Returns
        -------
        dict
            Dictionary mapping stimulus files to brain activity patterns (beta maps)
        """
        if verbose:
            print(f"  GLM modeling:")
        
        # Prepare events for GLM
        # Each unique stimulus is a separate condition
        events_for_glm = events_df.copy()
        events_for_glm['trial_type'] = events_for_glm['stim_file']
        events_for_glm = events_for_glm[['onset', 'duration', 'trial_type']]
        
        # Create brain mask
        if verbose:
            print(f"    - Creating brain mask")
        masker = NiftiMasker(
            standardize=False,
            detrend=False,  # Already detrended in preprocessing
            memory='nilearn_cache',
            memory_level=1
        )
        masker.fit(bold_img)
        
        # Fit GLM
        if verbose:
            print(f"    - Fitting GLM with canonical HRF")
        
        fmri_glm = FirstLevelModel(
            t_r=tr,
            noise_model='ar1',
            standardize=False,
            hrf_model='spm',  # SPM canonical HRF
            drift_model='cosine',  # Cosine drift model for high-pass filtering
            high_pass=self.high_pass,  # High-pass filter cutoff
            mask_img=masker.mask_img_,
            minimize_memory=False
        )
        
        fmri_glm = fmri_glm.fit(bold_img, events=events_for_glm)
        
        # Extract beta maps for each stimulus
        if verbose:
            print(f"    - Extracting beta maps for {len(events_df)} stimuli")
        
        stimulus_patterns = {}
        unique_stimuli = events_df['stim_file'].unique()
        
        for stim_file in unique_stimuli:
            try:
                # Compute contrast for this stimulus
                beta_map = fmri_glm.compute_contrast(stim_file, output_type='effect_size')
                
                # Convert to 1D array
                pattern = masker.transform(beta_map).ravel()
                
                stimulus_patterns[stim_file] = pattern
                
            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not extract pattern for {stim_file}: {e}")
                continue
        
        return stimulus_patterns
    
    def extract_stimulus_activity_simple(
        self,
        bold_img: nib.Nifti1Image,
        events_df: pd.DataFrame,
        tr: float,
        baseline_trs: int = 1,
        response_trs: int = 3,
        verbose: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Extract brain activity for each stimulus using simple averaging (no GLM).
        
        Parameters
        ----------
        bold_img : nibabel.Nifti1Image
            Preprocessed BOLD image
        events_df : pd.DataFrame
            Events dataframe with onset, duration, stim_file columns
        tr : float
            Repetition time in seconds
        baseline_trs : int
            Number of TRs before stimulus onset for baseline
        response_trs : int
            Number of TRs after stimulus onset to extract
        verbose : bool
            Print processing steps
        
        Returns
        -------
        dict
            Dictionary mapping stimulus files to brain activity patterns
        """
        bold_data = bold_img.get_fdata()
        stimulus_patterns = {}
        
        # Create brain mask
        mean_bold = np.mean(bold_data, axis=3)
        mask = mean_bold > np.percentile(mean_bold, 10)
        
        # Extract voxel coordinates
        voxel_coords = np.where(mask)
        n_voxels = len(voxel_coords[0])
        
        if verbose:
            print(f"  Using {n_voxels} voxels for extraction")
        
        # Process each stimulus
        for idx, row in events_df.iterrows():
            onset = row['onset']
            stim_file = row['stim_file']
            
            # Convert onset time to TR index
            onset_tr = int(np.round(onset / tr))
            
            # Extract time window
            start_tr = max(0, onset_tr - baseline_trs)
            end_tr = min(bold_data.shape[3], onset_tr + response_trs)
            
            # Extract activity
            stim_activity = bold_data[voxel_coords[0], voxel_coords[1], voxel_coords[2], start_tr:end_tr]
            stim_activity = stim_activity.T
            
            # Baseline correction
            if baseline_trs > 0:
                baseline = stim_activity[:baseline_trs, :].mean(axis=0)
                stim_activity = stim_activity - baseline
            
            # Average across time
            mean_pattern = stim_activity.mean(axis=0)
            
            stimulus_patterns[stim_file] = mean_pattern
        
        return stimulus_patterns
    
    def process_all_runs(
        self,
        output_dir: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process all semantic task runs for the subject.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save processed data
        save_results : bool
            Whether to save results to disk
        
        Returns
        -------
        dict
            Nested dictionary: {session: {run: {stim_file: pattern}}}
        """
        runs = self.find_semantic_runs()
        
        if not runs:
            raise ValueError(f"No semantic task runs found for {self.subject_id}")
        
        print(f"Found {len(runs)} semantic task runs")
        print("=" * 60)
        
        all_patterns = {}
        
        for run_info in runs:
            session = run_info['session']
            run = run_info['run']
            
            print(f"\nProcessing {session} {run}")
            print(f"  BOLD: {run_info['bold'].name}")
            print(f"  Events: {run_info['events'].name}")
            
            # Load data
            events_df = self.load_events(run_info['events'])
            print(f"  Loaded {len(events_df)} trials")
            
            bold_img = self.load_bold(run_info['bold'])
            bold_data = bold_img.get_fdata()
            print(f"  BOLD shape: {bold_data.shape}")
            
            # Get TR from header
            tr = bold_img.header.get_zooms()[3] if len(bold_img.header.get_zooms()) > 3 else 2.0
            print(f"  TR: {tr}s")
            
            # Preprocess functional data
            bold_img_preprocessed = self.preprocess_functional(bold_img, tr=tr, verbose=True)
            
            # Extract patterns
            if self.use_glm:
                patterns = self.extract_stimulus_activity_glm(
                    bold_img=bold_img_preprocessed,
                    events_df=events_df,
                    tr=tr,
                    verbose=True
                )
            else:
                patterns = self.extract_stimulus_activity_simple(
                    bold_img=bold_img_preprocessed,
                    events_df=events_df,
                    tr=tr,
                    verbose=True
                )
            
            print(f"  Extracted {len(patterns)} stimulus patterns")
            
            # Store results
            if session not in all_patterns:
                all_patterns[session] = {}
            all_patterns[session][run] = patterns
            
            # Save if requested
            if save_results and output_dir:
                self._save_patterns(patterns, output_dir, session, run)
        
        print("\n" + "=" * 60)
        print("Processing complete!")
        
        return all_patterns
    
    def _save_patterns(
        self,
        patterns: Dict[str, np.ndarray],
        output_dir: str,
        session: str,
        run: str
    ):
        """Save extracted patterns to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{self.subject_id}_{session}_{run}_patterns.npz"
        filepath = output_path / filename
        
        # Save as compressed numpy archive
        np.savez_compressed(str(filepath), **patterns)
        print(f"  Saved patterns to: {filepath}")


def main():
    """Example usage."""
    # Initialize preprocessor with enhanced preprocessing
    preprocessor = FMRIPreprocessor(
        data_dir="data/brain/ds003604",
        subject_id="sub-5007",
        smoothing_fwhm=6.0,  # 6mm spatial smoothing
        high_pass=0.01,      # 0.01 Hz high-pass filter (1/100s)
        use_glm=True         # Use GLM with HRF modeling
    )
    
    # Process all runs
    all_patterns = preprocessor.process_all_runs(
        output_dir="data/processed/fmri",
        save_results=True
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    for session, runs in all_patterns.items():
        for run, patterns in runs.items():
            print(f"  {session} {run}: {len(patterns)} stimuli")
            # Show example pattern shape
            example_stim = list(patterns.keys())[0]
            print(f"    Pattern shape: {patterns[example_stim].shape}")


if __name__ == "__main__":
    main()
