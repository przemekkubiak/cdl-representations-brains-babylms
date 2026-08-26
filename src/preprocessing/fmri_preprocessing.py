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

MASKING -- READ BEFORE CHANGING. Until 2026-08-26 every run of this pipeline
left `mask_path` unset, so `NiftiMasker` fell back to nilearn's default
`mask_strategy='background'`. That strategy assumes voxels outside the brain
are at or near zero; ds003604's raw BOLD does not have that property, so it
measurably failed -- every extracted pattern kept ~100% of the raw acquisition
volume (917,504 voxels, air included; see paper_results/control/README.md).
The fix below is `mask_strategy='epi'`, nilearn's strategy for exactly this
case (a functional image without a clean zero background), used whenever no
explicit mask is supplied. See MASKING.md for the full diagnosis and for why
ROI restriction (roi_set below) needed real per-subject registration rather
than resampling an MNI atlas directly onto native BOLD.
"""

import logging
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
from typing import Dict, List, Tuple, Optional
from nilearn import image
from nilearn.image import resample_to_img
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.maskers import NiftiMasker
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing import spatial_normalization as spatial_norm
from src.preprocessing.roi_atlas import available_roi_sets, parse_roi_sets
from src.datasets import get_dataset
from src.datasets.stim_identity import classify_trials

logger = logging.getLogger(__name__)

# Session label used for datasets with no ses-* BIDS entity at all (e.g.
# ds002236 -- single cross-sectional visit, files sit directly under
# sub-X/func/). Deliberately NOT an age-group label: which developmental bin
# a subject belongs to is computed from their real per-subject age at the
# RDM-aggregation stage (src/datasets/age_groups.py), not baked into where
# their raw files happen to live on disk. Keeping the two separate means a
# preprocessing re-run never has to change just because a bin boundary moves.
SESSIONLESS_LABEL = "ses-all"


class FMRIPreprocessor:
    """
    Preprocessor for fMRI data to extract stimulus-specific brain activity.
    Includes spatial smoothing, high-pass filtering, and GLM modeling.
    """
    
    def __init__(
        self,
        data_dir: str,
        subject_id: str = "sub-5007",
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
        Initialize the preprocessor.

        Parameters
        ----------
        data_dir : str
            Path to the brain data directory
        subject_id : str
            Subject identifier (default: "sub-5007")
        task : str
            A PHENOMENON key (e.g. "Sem", "Phon"), not necessarily a literal
            BIDS task label -- resolved to the dataset's real task name(s) via
            `configs/neuro_datasets.yaml`'s `phenomena` mapping (see
            `_resolve_real_tasks`). For ds003604 this resolves to itself
            (phenomenon and BIDS task happen to share a name there), so
            nothing changes for existing callers.
        dataset : str
            Registry key from configs/neuro_datasets.yaml (default: "ds003604",
            preserving old behaviour for existing callers that don't pass it).
            Determines: which BIDS task(s) `task` resolves to, how stimulus
            identity/condition is derived from events.tsv (see
            src/datasets/stim_identity.py), and whether sessions are expected
            on disk at all.
        smoothing_fwhm : float
            Full-width at half-maximum for Gaussian smoothing in mm (default: 6.0)
        high_pass : float
            High-pass filter cutoff in Hz (default: 0.01, i.e., 1/100s)
            Note: When use_glm=True, filtering is done by GLM's drift model
        use_glm : bool
            Whether to use GLM modeling (default: True)
        mask_path : str, optional
            Path to a STATIC binary/probabilistic mask NIfTI, already in the
            same space as the BOLD (e.g. a dataset that ships pre-registered
            data). Mutually exclusive with `roi_set` -- ds003604 is NOT
            pre-registered, so a static mask resampled onto it would silently
            land in the wrong place per subject; use `roi_set` instead. If
            neither is given, every run still gets a real per-subject
            whole-brain mask via `mask_strategy='epi'` (see module docstring).
        roi_set : str, optional
            Comma-separated names from `src.preprocessing.roi_atlas.ROI_SETS`
            (currently: language, auditory, motor), e.g. "auditory,motor".
            When set, each subject-session is registered to MNI152 (rigid
            EPI->T1, affine T1->MNI; see spatial_normalization.py) and the
            named ROI(s) are warped into that subject's native space and
            intersected with their whole-brain mask. Registration is cached
            per (subject, session) under `mask_cache_dir` and reused across
            tasks. If registration fails or no T1w is found for a given
            subject-session, that subject-session falls back to the
            whole-brain mask alone -- logged, and recorded in
            `<mask_cache_dir>/roi_mask_status.csv` -- rather than raising, so
            one bad subject cannot take down an unattended batch run.
        mask_cache_dir : str, optional
            Where per-subject-session registrations and warped ROI masks are
            cached. REQUIRED if `roi_set` is set -- pass the same directory
            across every task for a dataset so registration is computed once
            per subject-session rather than once per task.
        mni_template_resolution : int
            MNI152 template resolution in mm for the T1->MNI registration
            target (default 2mm; nilearn ships this without a download).
        save_native_maps : bool
            Save this subject-session's whole-brain mask (native EPI space,
            correct affine/shape) to `<mask_cache_dir>/<subject>/<subject>_
            <session>_wholebrain_mask.nii.gz`, once per session. Off by
            default because it's an extra file per subject-session that
            nothing needs unless real spatial figures are wanted -- patterns
            themselves are flat masked vectors with no spatial information
            retained, by design (they're what RSA/RDM building needs). This
            is what lets that spatial information be reconstructed
            afterwards: src/rsa/brain_localization.py loads this mask to
            un-flatten a condition>control t-map back into a real 3D image,
            then warps it to MNI152 via spatial_normalization.py's
            registration (see scripts/render_brain_atlas_figures.py). REQUIRES
            `mask_cache_dir` (does not require `roi_set`).
        """
        self.data_dir = Path(data_dir)
        self.subject_id = subject_id
        self.task = task
        self.dataset = dataset
        self.subject_dir = self.data_dir / subject_id
        self.smoothing_fwhm = smoothing_fwhm
        self.high_pass = high_pass
        self.use_glm = use_glm
        self.mask_path = Path(mask_path) if mask_path else None
        # Checked BEFORE nib.load: loading a nonexistent path raises
        # nibabel's own (less clear) error first otherwise -- the check two
        # lines below this used to never fire for that reason.
        if self.mask_path and not self.mask_path.exists():
            raise ValueError(f"Mask file not found: {self.mask_path}")
        self.mask_img = nib.load(str(self.mask_path)) if self.mask_path else None

        if mask_path and roi_set:
            raise ValueError(
                "mask_path and roi_set are mutually exclusive -- mask_path is "
                "a static pre-registered mask, roi_set triggers per-subject "
                "registration. Combining them is not implemented because it "
                "is not a well-defined operation (which space is the static "
                "mask in?). Pick one."
            )
        self.roi_sets = parse_roi_sets(roi_set) if roi_set else None
        if self.roi_sets and not mask_cache_dir:
            raise ValueError(
                "roi_set requires mask_cache_dir (registration is expensive; "
                "it must be cached and shared across tasks, not recomputed "
                f"per task). Known roi sets: {available_roi_sets()}"
            )
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else None
        self.mni_template_resolution = mni_template_resolution
        self.save_native_maps = save_native_maps
        if self.save_native_maps and not self.mask_cache_dir:
            raise ValueError("save_native_maps requires mask_cache_dir (same directory roi_set uses)")
        if self.save_native_maps and self.roi_sets:
            raise ValueError(
                "save_native_maps and roi_set together is not implemented -- the "
                "mask this saves is the WHOLE-BRAIN mask, but roi_set makes the "
                "actual extracted pattern use a SMALLER, ROI-intersected mask, so "
                "reconstructing a spatial map from the pattern against the saved "
                "whole-brain mask would be a shape mismatch (or worse, silently "
                "misaligned if the counts happened to coincide). Run a whole-brain "
                "pass (no roi_set) for spatial map export -- which is also the "
                "more useful map for a general 'show me brain activation' figure."
            )
        # Populated lazily, once per session, the first time a run from that
        # session is processed -- see `_get_run_mask`.
        self._session_roi_native: Dict[str, Optional[np.ndarray]] = {}
        self._session_registered: set = set()  # sessions where save_native_maps has already tried registration

        if not self.subject_dir.exists():
            raise ValueError(f"Subject directory not found: {self.subject_dir}")

    def _get_run_mask(self, bold_img: nib.Nifti1Image, bold_path: Path,
                       session: str) -> nib.Nifti1Image:
        """The mask used for this run's `NiftiMasker`.

        Three cases, in priority order:
          1. `self.mask_img` set (legacy static mask, pre-registered dataset)
             -- resampled onto this run's grid, unchanged from before.
          2. `self.roi_sets` set -- whole-brain EPI mask AND the subject's
             registered ROI mask (cached per session; built from this run
             only if this is the first run seen for the session).
          3. Neither -- whole-brain EPI mask alone. This is the bug fix:
             previously nilearn's own default here (`mask_strategy=
             'background'`) silently kept ~100% of the volume.
        """
        if self.mask_img is not None:
            return resample_to_img(self.mask_img, bold_img, interpolation='nearest')

        whole_brain = NiftiMasker(
            mask_strategy='epi', standardize=False, detrend=False,
            memory=None, memory_level=0,
        )
        whole_brain.fit(bold_img)
        whole_brain_mask = whole_brain.mask_img_

        if self.save_native_maps:
            self._save_wholebrain_mask_once(whole_brain_mask, session)
            # Registration must run NOW, while raw BOLD (and so a valid EPI
            # reference volume) still exists -- brainprep_subject.sh drops
            # BOLD immediately after this subject is preprocessed, well
            # before brain_localization.py's t-map export step ever runs.
            # get_or_register is idempotent/cached, so this costs nothing
            # extra when roi_set already triggered the same registration.
            if session not in self._session_registered:
                epi_ref = spatial_norm.epi_reference_volume(bold_path)
                reg = spatial_norm.get_or_register(
                    subject=self.subject_id, session=session, data_dir=self.data_dir,
                    cache_dir=self.mask_cache_dir, epi_ref_img=epi_ref,
                    template_resolution=self.mni_template_resolution,
                    status_csv=self.mask_cache_dir / "roi_mask_status.csv",
                    status_extra={"roi_set": "tmap_export"},
                )
                self._session_registered.add(session)
                if reg is None:
                    logger.warning(
                        "  %s/%s: registration for spatial map export failed -- "
                        "brain_localization.py will not be able to place this "
                        "subject-session's t-map in MNI space", self.subject_id, session)

        if not self.roi_sets:
            return whole_brain_mask

        if session not in self._session_roi_native:
            logger.info("  %s/%s: building registered ROI mask (%s)",
                        self.subject_id, session, "+".join(self.roi_sets))
            epi_ref = spatial_norm.epi_reference_volume(bold_path)
            self._session_roi_native[session] = spatial_norm.build_subject_roi_mask(
                subject=self.subject_id, session=session, data_dir=self.data_dir,
                cache_dir=self.mask_cache_dir, roi_sets=self.roi_sets,
                epi_ref_img=epi_ref, template_resolution=self.mni_template_resolution,
            )

        roi_native = self._session_roi_native[session]
        if roi_native is None:
            # Already logged (and recorded in roi_mask_status.csv) by
            # build_subject_roi_mask -- fall back quietly here so a missing
            # T1 or a failed registration doesn't spam every run's log.
            return whole_brain_mask

        whole_brain_data = np.asarray(whole_brain_mask.get_fdata()) > 0
        if roi_native.shape != whole_brain_data.shape:
            # Same subject-session, so this should be rare (a protocol change
            # mid-session); resample rather than silently mismatching shapes.
            roi_img_tmp = nib.Nifti1Image(roi_native.astype(np.uint8), whole_brain_mask.affine)
            roi_native = resample_to_img(
                roi_img_tmp, bold_img, interpolation='nearest'
            ).get_fdata() > 0
        combined = np.logical_and(whole_brain_data, roi_native)
        if combined.sum() == 0:
            logger.warning(
                "  %s/%s: ROI mask has no overlap with this run's whole-brain "
                "mask -- using whole-brain mask alone for this run", self.subject_id, session)
            return whole_brain_mask
        return nib.Nifti1Image(combined.astype(np.uint8), whole_brain_mask.affine)

    def _save_wholebrain_mask_once(self, mask_img: nib.Nifti1Image, session: str) -> None:
        """Save this subject-session's native-space whole-brain mask,
        skipping if already present. Same acquisition protocol within a
        session means every run's mask should already agree; not enforced
        here (that would need loading + comparing every run's mask, which
        costs more than it's worth) -- the first run's mask wins, same
        assumption the ROI-mask cache above already makes.
        """
        base = self.mask_cache_dir / self.subject_id
        base.mkdir(parents=True, exist_ok=True)
        out = base / f"{self.subject_id}_{session}_wholebrain_mask.nii.gz"
        if out.exists():
            return
        tmp = out.parent / (out.name.replace(".nii.gz", ".tmp.nii.gz"))
        nib.save(mask_img, str(tmp))
        tmp.rename(out)

    def _resolve_real_tasks(self) -> List[str]:
        """`self.task` is a PHENOMENON key, not necessarily a literal BIDS
        task label. Resolve via the registry's phenomena mapping; falls back
        to treating self.task as a literal BIDS task name if the dataset
        declares no mapping for it (keeps direct/manual use working). For
        ds003604 every phenomenon maps to a single-element list containing
        its own name (phenomena: {Sem: [Sem], ...}), so this resolves to
        exactly what the old hardcoded glob searched for -- zero behaviour
        change for the existing pipeline.
        """
        spec = get_dataset(self.dataset)
        return spec.phenomena.get(self.task) or [self.task]

    def find_task_runs(self) -> List[Dict[str, Path]]:
        """
        Find all task runs across sessions.

        Returns
        -------
        list of dict
            List of dictionaries containing 'bold', 'events', 'session', 'run',
            'real_task' info
        """
        runs = []
        real_tasks = self._resolve_real_tasks()

        session_dirs = sorted(self.subject_dir.glob("ses-*"))
        # Some datasets have no ses-* entity at all (single cross-sectional
        # visit -- e.g. ds002236, files sit directly under sub-X/func/).
        # SESSIONLESS_LABEL stands in so every downstream consumer of
        # run_info['session'] still gets a well-formed, non-empty label; it
        # is NOT a developmental/age-group label -- see the module-level
        # comment on SESSIONLESS_LABEL for why that's a deliberate split.
        session_targets = (
            [(d.name, d / "func") for d in session_dirs]
            if session_dirs else [(SESSIONLESS_LABEL, self.subject_dir / "func")]
        )

        for session_label, func_dir in session_targets:
            if not func_dir.exists():
                continue
            for real_task in real_tasks:
                for bold_file in sorted(func_dir.glob(f"*task-{real_task}*_bold.nii.gz")):
                    events_file = bold_file.parent / bold_file.name.replace("_bold.nii.gz", "_events.tsv")
                    if events_file.exists():
                        runs.append({
                            "bold": bold_file,
                            "events": events_file,
                            "session": session_label,
                            "run": self._extract_run_number(bold_file.name),
                            "real_task": real_task,
                        })

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

        # Clean up stimulus file names -- only when this column exists.
        # ds003604 (github_tsv) has it; the stim_pair_filename datasets
        # (ds001894/ds006239/ds002236) use different column names entirely
        # (see src/datasets/stim_identity.py), so unconditionally requiring
        # 'stim_file' here used to KeyError on all three of them.
        if 'stim_file' in df.columns:
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
        bold_path: Path,
        session: str,
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
        bold_path : Path
            Path to the raw BOLD file for this run -- used as the EPI
            reference volume for registration if `roi_set` is active and this
            is the first run seen for `session` (see `_get_run_mask`).
        session : str
            Session label (e.g. "ses-5") -- registration and the resulting
            ROI mask are cached per session, not per run.
        verbose : bool
            Print processing steps

        Returns
        -------
        dict
            Dictionary mapping stimulus files to brain activity patterns (beta maps)
        """
        if verbose:
            print(f"  GLM modeling:")

        # Which rows become GLM regressors, and under what stimulus identity,
        # is dataset-aware -- see src/datasets/stim_identity.py. For ds003604
        # (kind: github_tsv) this reproduces the exact old behaviour: every
        # row, trial_type set to stim_file, no condition filtering (controls
        # are excluded downstream in session_based_rsa.py, same as always).
        # For the stim_pair_filename datasets, only positive/negative trials
        # (per src/contrast_spec.py) become regressors at all -- perceptual/
        # null trials never get fit, rather than being fit and discarded
        # later, since there is no "later" step for those datasets that would
        # discard them.
        trials = classify_trials(events_df.to_dict("records"), self.dataset, self.task)
        if not trials:
            if verbose:
                print(f"    - No classified trials for phenomenon '{self.task}' in this run")
            return {}
        # ALL trials go into the design matrix (nilearn groups rows sharing a
        # trial_type into one regressor internally) -- do not dedupe here, or
        # a stimulus presented more than once in a run loses everything but
        # its first onset. Dedup only for the compute_contrast loop below,
        # where one contrast per unique condition is what's wanted.
        events_for_glm = pd.DataFrame([
            {"onset": float(t.row["onset"]), "duration": float(t.row["duration"]), "trial_type": t.stim_id}
            for t in trials
        ])
        unique_stim_ids = list(dict.fromkeys(t.stim_id for t in trials))

        # Brain mask: static mask_path, or a real per-subject whole-brain (+
        # optionally ROI) mask -- see `_get_run_mask` for the full logic and
        # why `mask_strategy='epi'` replaces nilearn's failing default.
        if verbose:
            print(f"    - Building brain mask")
        mask_img = self._get_run_mask(bold_img, bold_path, session)
        masker = NiftiMasker(
            mask_img=mask_img,
            standardize=False,
            detrend=False,
            memory=None,   # disabled: joblib cache is write-only here (every
            memory_level=0 # subject is a unique input) and reached 361GB on 2026-08-18
        )
        masker.fit(bold_img)
        if verbose:
            n_vox = int((np.asarray(mask_img.get_fdata()) > 0).sum())
            print(f"    - Mask: {n_vox:,} voxels")

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
            print(f"    - Extracting beta maps for {len(unique_stim_ids)} stimuli "
                  f"({len(trials)} trials, {len(events_df)} raw rows)")

        stimulus_patterns = {}

        for stim_id in unique_stim_ids:
            try:
                # Compute contrast for this stimulus
                beta_map = fmri_glm.compute_contrast(stim_id, output_type='effect_size')

                # Convert to 1D array
                pattern = masker.transform(beta_map).ravel()

                stimulus_patterns[stim_id] = pattern

            except Exception as e:
                if verbose:
                    print(f"      Warning: Could not extract pattern for {stim_id}: {e}")
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
        if self.dataset != "ds003604":
            raise NotImplementedError(
                "extract_stimulus_activity_simple (--no-glm) has not been "
                f"generalized to dataset '{self.dataset}' -- it still reads "
                "'stim_file' directly and has no path through "
                "src/datasets/stim_identity.py. use_glm=True (the default "
                "everywhere in this pipeline) is unaffected. Raising here "
                "rather than silently extracting patterns keyed by the wrong "
                "column, or crashing deep inside a KeyError."
            )
        bold_data = bold_img.get_fdata()
        stimulus_patterns = {}

        # Create brain mask
        if self.mask_img is not None:
            mask_img = resample_to_img(self.mask_img, bold_img, interpolation='nearest')
            mask = mask_img.get_fdata() > 0
        else:
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
        Process all task runs for the subject.
        
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
        runs = self.find_task_runs()
        
        if not runs:
            raise ValueError(f"No {self.task} task runs found for {self.subject_id}")
        
        print(f"Found {len(runs)} {self.task} task runs")
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
                    bold_path=run_info['bold'],
                    session=session,
                    verbose=True
                )
            else:
                if self.roi_sets:
                    raise ValueError(
                        "roi_set is only implemented for the GLM extraction path "
                        "(use_glm=True). extract_stimulus_activity_simple's own "
                        "mask (mean_bold > 10th percentile) is not touched by "
                        "this change; combining it with roi_set would silently "
                        "ignore the ROI restriction, so this raises instead."
                    )
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
        use_glm=True,        # Use GLM with HRF modeling
        mask_path=None       # Set to NIfTI mask path to restrict voxels
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
