"""
Per-subject spatial normalization: native EPI -> native T1 -> MNI152.

WHY THIS EXISTS. ds003604 -- and every dataset this repo currently
supports -- is never spatially normalized upstream of this repo (see
MASKING.md). There is no coregistration, no template-warping, nothing: every
subject's functional data sits in its own native scanner space, which is
exactly why `src/rsa/session_based_rsa.py` aggregates subjects with a Shared
Response Model rather than by voxel position. An anatomical ROI defined in
MNI152 space (auditory cortex, motor cortex, ...) therefore cannot be
resampled onto a subject's BOLD and mean anything on its own --
`nilearn.image.resample_to_img` only reconciles voxel GRIDS via each image's
affine, it does not perform registration between two genuinely different
spaces. This module adds the missing step: real per-subject registration, so
a named ROI (src/preprocessing/roi_atlas.py) lands in the right place in each
subject's own functional space.

DESIGN CHOICES, AND WHY -- READ BEFORE CHANGING ANY OF THESE.

  * RIGID (6 DOF) for EPI -> T1, not affine. Same subject, same session: brain
    shape and size do not differ between the two scans, only position and
    orientation (and, via mutual information rather than correlation,
    contrast). Rigid is the textbook choice here and has far fewer failure
    modes than a higher-DOF fit would.

  * AFFINE (12 DOF), NOT diffeomorphic, for T1 -> MNI. A full nonlinear warp
    (e.g. SyN, what fMRIPrep uses) places an ROI boundary more precisely, but
    is slow, has more failure modes, and needs more supervision than an
    unattended cluster run with no interactive access can give it. Auditory
    and motor cortex, as used here, are centimeter-scale regions; an affine
    fit is adequate to place a mask of that size correctly, and an affine
    failure is much easier to catch automatically (see `_sanity_check`) than
    a diffeomorphic one, which can look locally plausible and still be wrong.
    If per-voxel precision is ever needed, add diffeomorphic registration as
    a deliberate follow-up on top of a verified affine baseline -- not a
    silent upgrade here.

  * MUTUAL INFORMATION metric throughout. EPI (T2*-weighted), T1
    (T1-weighted) and the MNI template (a population average) all have
    different intensity relationships; mutual information is the standard
    metric for exactly this cross-contrast registration problem. A
    correlation-based metric would be a silent, hard-to-notice bug here.

  * EVERY registration is wrapped so it can fail LOUDLY without failing the
    whole run. `register_subject_to_mni` raises on a bad fit rather than
    returning something plausible-but-wrong; `build_subject_roi_mask` (the
    entry point everything else calls) catches that, logs it, records it in
    a CSV ledger, and returns None so the caller falls back to the
    whole-brain mask for that one subject -- never crashes the batch, and
    never silently mislabels a bad registration as a good one.

  * CACHED PER (subject, session), on disk, not per (subject, session, task).
    Registration does not depend on task or run, so it is computed once and
    reused for Sem/Phon/Gram/Plaus alike. The cache is keyed by roi-set too,
    so asking for a second ROI set on an already-registered subject reuses
    the registration and only re-warps the new mask (cheap -- no
    optimization involved, just two `transform_inverse` calls).
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import mean_img, resample_to_img

from dipy.align.imaffine import AffineMap, AffineRegistration, MutualInformationMetric
from dipy.align.transforms import (
    AffineTransform3D,
    RigidTransform3D,
    TranslationTransform3D,
)

from src.preprocessing.roi_atlas import build_roi_mask_mni

logger = logging.getLogger(__name__)

# Coarse-to-fine registration schedule. Deliberately modest -- this runs once
# per subject-session, potentially hundreds of times, sharing a GPU box with
# other work (see PICKUP.md). These are the values used in dipy's own affine
# registration tutorial and are adequate for a rigid/affine fit at the
# ROI granularity used here (see module docstring); they are not tuned
# per-dataset because there was no GPU access available to tune them against.
_LEVEL_ITERS = [1000, 100, 10]
_SIGMAS = [3.0, 1.0, 0.0]
_FACTORS = [4, 2, 1]
_NBINS = 32

# Below this Dice overlap between the registered T1 and the template's own
# brain mask, treat the registration as failed rather than merely imprecise.
# A correct affine fit of a real brain to MNI152 typically clears 0.7-0.8; 0.5
# is a generous floor that still catches a registration that converged to
# roughly the wrong place (e.g. local optimum from a bad starting point).
_MIN_DICE = 0.5
# Translation sanity bounds in mm. A rigid EPI->T1 shift or an affine
# T1->MNI shift beyond these is not a real anatomical offset for a human
# head -- it means the optimizer diverged.
_MAX_EPI_TO_T1_MM = 100.0
_MAX_T1_TO_MNI_MM = 150.0


@dataclass
class RegistrationResult:
    epi_to_t1: AffineMap
    t1_to_mni: AffineMap
    t1_shape: tuple
    quality: dict
    t1_path: Optional[Path] = None


def _affreg() -> AffineRegistration:
    metric = MutualInformationMetric(_NBINS, None)
    return AffineRegistration(
        metric=metric, level_iters=_LEVEL_ITERS, sigmas=_SIGMAS, factors=_FACTORS
    )


def _optimize_rigid(static_img: nib.Nifti1Image, moving_img: nib.Nifti1Image) -> AffineMap:
    affreg = _affreg()
    static = np.asarray(static_img.get_fdata(), dtype=np.float64)
    moving = np.asarray(moving_img.get_fdata(), dtype=np.float64)
    translation = affreg.optimize(
        static, moving, TranslationTransform3D(), None,
        static_img.affine, moving_img.affine, starting_affine=np.eye(4),
    )
    return affreg.optimize(
        static, moving, RigidTransform3D(), None,
        static_img.affine, moving_img.affine, starting_affine=translation.affine,
    )


def _optimize_affine(static_img: nib.Nifti1Image, moving_img: nib.Nifti1Image) -> AffineMap:
    affreg = _affreg()
    static = np.asarray(static_img.get_fdata(), dtype=np.float64)
    moving = np.asarray(moving_img.get_fdata(), dtype=np.float64)
    translation = affreg.optimize(
        static, moving, TranslationTransform3D(), None,
        static_img.affine, moving_img.affine, starting_affine=np.eye(4),
    )
    return affreg.optimize(
        static, moving, AffineTransform3D(), None,
        static_img.affine, moving_img.affine, starting_affine=translation.affine,
    )


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    denom = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / denom) if denom else 0.0


def _sanity_check(
    epi_to_t1: AffineMap,
    t1_to_mni: AffineMap,
    t1_img: nib.Nifti1Image,
    mni_template: nib.Nifti1Image,
    template_resolution: int,
) -> dict:
    """Fast, automatic checks that catch a badly-failed registration without
    a human looking at an image -- the audit trail a run with no interactive
    GPU access needs. These do not PROVE the registration is good, only that
    it is not obviously broken; the QC PNG written by
    `build_subject_roi_mask` is the thing to actually look at, later."""
    epi_to_t1_mm = float(np.linalg.norm(epi_to_t1.affine[:3, 3]))
    t1_to_mni_mm = float(np.linalg.norm(t1_to_mni.affine[:3, 3]))

    template_brain = datasets.load_mni152_brain_mask(resolution=template_resolution)
    template_brain = resample_to_img(template_brain, mni_template, interpolation="nearest")
    t1_ones = np.ones(t1_img.shape, dtype=np.float64)
    t1_in_mni = t1_to_mni.transform(t1_ones, interpolation="nearest")
    dice = _dice(t1_in_mni > 0, np.asarray(template_brain.get_fdata()) > 0)

    ok = (
        dice > _MIN_DICE
        and epi_to_t1_mm < _MAX_EPI_TO_T1_MM
        and t1_to_mni_mm < _MAX_T1_TO_MNI_MM
    )
    return {
        "epi_to_t1_translation_mm": epi_to_t1_mm,
        "t1_to_mni_translation_mm": t1_to_mni_mm,
        "t1_in_mni_vs_template_brain_dice": dice,
        "ok": bool(ok),
    }


def find_t1w(data_dir: Path, subject: str, session: str) -> Optional[Path]:
    """Locate a subject-session T1w anatomical.

    ds003604 ships TWO T1w acquisitions per subject-session
    (acq-D1S2, acq-D1S7 as of the 2026-08-26 checkout) -- either is
    anatomically fine for registration, so this deterministically picks the
    lexicographically first and logs when there was a choice to make, so
    which file was used is auditable without re-running anything.
    """
    data_dir = Path(data_dir)
    hits = sorted((data_dir / subject / session / "anat").glob(f"{subject}_{session}_*T1w.nii.gz"))
    if not hits:
        # Fall back to a subject-level (non-session-nested) anat layout, in
        # case a future dataset in the registry uses one.
        hits = sorted((data_dir / subject / "anat").glob(f"{subject}_*T1w.nii.gz"))
    if not hits:
        return None
    if len(hits) > 1:
        logger.info(
            "  %s/%s: %d T1w acquisitions found, using %s (deterministic: first sorted)",
            subject, session, len(hits), hits[0].name,
        )
    return hits[0]


def register_subject_to_mni(
    epi_ref_img: nib.Nifti1Image,
    t1_path: Path,
    template_resolution: int = 2,
) -> RegistrationResult:
    """Rigid EPI->T1 then affine T1->MNI. Raises RuntimeError if the sanity
    check fails -- callers (`build_subject_roi_mask`) decide the fallback;
    this function never returns a result it has reason to believe is wrong.
    """
    if len(epi_ref_img.shape) != 3:
        raise ValueError(
            f"epi_ref_img must be a 3D reference volume, got shape "
            f"{epi_ref_img.shape}. A raw 4D BOLD run needs to go through "
            "epi_reference_volume() (or nilearn.image.mean_img) first -- "
            "passing a 4D run directly fails deep inside dipy's registration "
            "internals with a much less readable error."
        )
    t1_img = nib.load(str(t1_path))
    if len(t1_img.shape) != 3:
        raise ValueError(f"T1 at {t1_path} is not 3D (shape {t1_img.shape})")
    mni_template = datasets.load_mni152_template(resolution=template_resolution)

    logger.info("    registering EPI -> T1 (rigid, mutual information)")
    epi_to_t1 = _optimize_rigid(t1_img, epi_ref_img)

    logger.info("    registering T1 -> MNI152 (affine, mutual information)")
    t1_to_mni = _optimize_affine(mni_template, t1_img)

    quality = _sanity_check(epi_to_t1, t1_to_mni, t1_img, mni_template, template_resolution)
    if not quality["ok"]:
        raise RuntimeError(f"registration sanity check failed: {quality}")

    return RegistrationResult(
        epi_to_t1=epi_to_t1, t1_to_mni=t1_to_mni, t1_shape=tuple(t1_img.shape),
        quality=quality, t1_path=Path(t1_path),
    )


def warp_roi_to_native(reg: RegistrationResult, roi_mask_mni: nib.Nifti1Image) -> np.ndarray:
    """MNI-space binary ROI -> the subject's native EPI grid.

    Nearest-neighbor at every step, so the result stays a clean 0/1 mask
    rather than a blurred one -- this is a label map, not an intensity image.
    """
    roi_mni = np.asarray(roi_mask_mni.get_fdata(), dtype=np.float64)
    if roi_mni.shape != reg.t1_to_mni.domain_shape:
        # domain_shape is the "static" (MNI-template) grid this AffineMap was
        # fit against; a mismatch here means the ROI mask and the template
        # used for registration were built at different resolutions, which
        # would silently corrupt the warp. Fail loudly instead.
        raise ValueError(
            f"ROI mask shape {roi_mni.shape} does not match the MNI template "
            f"grid this subject was registered against {reg.t1_to_mni.domain_shape}; "
            "build the ROI mask with the same template_resolution."
        )
    roi_in_t1 = reg.t1_to_mni.transform_inverse(roi_mni, interpolation="nearest")
    roi_in_epi = reg.epi_to_t1.transform_inverse(roi_in_t1, interpolation="nearest")
    return roi_in_epi > 0


def warp_native_to_mni(
    reg: RegistrationResult, native_img: nib.Nifti1Image, interpolation: str = "linear",
) -> nib.Nifti1Image:
    """Native EPI-space statistical image (e.g. a condition>control t-map) ->
    MNI152 space, on the exact template grid this subject was registered
    against. The reverse direction of warp_roi_to_native -- forward
    `.transform()` instead of `.transform_inverse()` -- and linear
    interpolation by default, deliberately unlike warp_roi_to_native's
    `nearest`: that function warps a binary label map (a 0/1 ROI, where
    blending values would be meaningless), this one warps a continuous
    statistical intensity image, where nearest-neighbor would introduce
    blocky artifacts a real analysis shouldn't have.
    """
    native = np.asarray(native_img.get_fdata(), dtype=np.float64)
    if native.shape != reg.epi_to_t1.codomain_shape:
        # codomain_shape is the EPI ("moving") grid this AffineMap was fit
        # against -- see the direction-semantics note in warp_roi_to_native
        # and the module docstring. A mismatch means native_img isn't on the
        # same grid the registration actually used.
        raise ValueError(
            f"native image shape {native.shape} does not match the EPI grid "
            f"this subject was registered against {reg.epi_to_t1.codomain_shape}"
        )
    in_t1 = reg.epi_to_t1.transform(native, interpolation=interpolation)
    in_mni = reg.t1_to_mni.transform(in_t1, interpolation=interpolation)
    return nib.Nifti1Image(in_mni.astype(np.float32), reg.t1_to_mni.domain_grid2world)


def _cache_paths(cache_dir: Path, subject: str, session: str, roi_key: str):
    base = Path(cache_dir) / subject
    base.mkdir(parents=True, exist_ok=True)
    stem = f"{subject}_{session}"
    return {
        "registration_json": base / f"{stem}_registration.json",
        "registration_npz": base / f"{stem}_registration.npz",
        "roi_mask": base / f"{stem}_roi-{roi_key}_mask.nii.gz",
        "qc_png": base / f"{stem}_roi-{roi_key}_qc.png",
        "status_csv": Path(cache_dir) / "roi_mask_status.csv",
    }


def _log_status(status_csv: Path, row: dict) -> None:
    """Append one row to the shared status ledger. This is the file to check
    after an unattended run to see, without re-running anything, which
    subjects got a real ROI mask and which fell back to whole-brain-only."""
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject", "session", "roi_set", "status", "detail",
        "epi_to_t1_translation_mm", "t1_to_mni_translation_mm", "dice", "n_roi_voxels",
    ]
    write_header = not status_csv.exists()
    with open(status_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fieldnames})


def _save_registration(paths: dict, reg: RegistrationResult, t1_path: Path) -> None:
    np.savez(
        paths["registration_npz"],
        epi_to_t1_affine=reg.epi_to_t1.affine,
        t1_to_mni_affine=reg.t1_to_mni.affine,
        epi_to_t1_domain_shape=np.array(reg.epi_to_t1.domain_shape),
        epi_to_t1_domain_grid2world=reg.epi_to_t1.domain_grid2world,
        epi_to_t1_codomain_shape=np.array(reg.epi_to_t1.codomain_shape),
        epi_to_t1_codomain_grid2world=reg.epi_to_t1.codomain_grid2world,
        t1_to_mni_domain_shape=np.array(reg.t1_to_mni.domain_shape),
        t1_to_mni_domain_grid2world=reg.t1_to_mni.domain_grid2world,
        t1_to_mni_codomain_shape=np.array(reg.t1_to_mni.codomain_shape),
        t1_to_mni_codomain_grid2world=reg.t1_to_mni.codomain_grid2world,
    )
    paths["registration_json"].write_text(json.dumps({
        "t1_path": str(t1_path), "quality": reg.quality,
    }, indent=2))


def _load_registration(paths: dict) -> Optional[RegistrationResult]:
    if not (paths["registration_npz"].exists() and paths["registration_json"].exists()):
        return None
    z = np.load(paths["registration_npz"])
    meta = json.loads(paths["registration_json"].read_text())
    if not meta["quality"]["ok"]:
        return None
    epi_to_t1 = AffineMap(
        z["epi_to_t1_affine"],
        domain_grid_shape=tuple(z["epi_to_t1_domain_shape"]),
        domain_grid2world=z["epi_to_t1_domain_grid2world"],
        codomain_grid_shape=tuple(z["epi_to_t1_codomain_shape"]),
        codomain_grid2world=z["epi_to_t1_codomain_grid2world"],
    )
    t1_to_mni = AffineMap(
        z["t1_to_mni_affine"],
        domain_grid_shape=tuple(z["t1_to_mni_domain_shape"]),
        domain_grid2world=z["t1_to_mni_domain_grid2world"],
        codomain_grid_shape=tuple(z["t1_to_mni_codomain_shape"]),
        codomain_grid2world=z["t1_to_mni_codomain_grid2world"],
    )
    return RegistrationResult(
        epi_to_t1=epi_to_t1, t1_to_mni=t1_to_mni,
        t1_shape=tuple(z["t1_to_mni_domain_shape"]), quality=meta["quality"],
        t1_path=Path(meta["t1_path"]) if meta.get("t1_path") else None,
    )


def _write_qc_png(png_path: Path, t1_path: Path, reg: RegistrationResult,
                   roi_mask_mni: nib.Nifti1Image, subject: str, session: str) -> None:
    """One glance-able image per subject-session: the warped ROI overlaid on
    the subject's own T1, in T1 space (so it's checkable without also
    trusting the EPI registration). This -- not re-running the pipeline -- is
    how a registration gets verified when nobody has interactive GPU access.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from nilearn import plotting

        roi_mni = np.asarray(roi_mask_mni.get_fdata(), dtype=np.float64)
        roi_in_t1 = reg.t1_to_mni.transform_inverse(roi_mni, interpolation="nearest")
        t1_img = nib.load(str(t1_path))
        roi_img = nib.Nifti1Image((roi_in_t1 > 0).astype(np.uint8), t1_img.affine)
        display = plotting.plot_roi(
            roi_img, bg_img=t1_img, title=f"{subject}/{session}: ROI warped into native T1",
            display_mode="ortho",
        )
        display.savefig(str(png_path))
        display.close()
    except Exception as e:  # QC image is a nice-to-have, never fatal
        logger.warning("  %s/%s: could not write QC image (%s)", subject, session, e)


def get_or_register(
    *,
    subject: str,
    session: str,
    data_dir: Path,
    cache_dir: Path,
    epi_ref_img: Optional[nib.Nifti1Image] = None,
    template_resolution: int = 2,
    status_csv: Optional[Path] = None,
    status_extra: Optional[dict] = None,
) -> Optional[RegistrationResult]:
    """Load the cached registration for (subject, session), or run and cache
    a fresh one. Shared by `build_subject_roi_mask` (ROI masking) and
    `spatial_normalization`'s MNI-space export path (brain_localization.py) --
    both need the exact same rigid-EPI-to-T1, affine-T1-to-MNI chain, and
    registration is by far the expensive/risky part, so it must only ever be
    computed once per subject-session regardless of which caller needs it
    first. Returns None (never raises) on any failure -- missing T1, missing
    EPI reference with nothing cached yet, or a registration that fails its
    own sanity check -- logging why via `logger` and, if `status_csv` is
    given, appending a row there (merged with `status_extra`, e.g. {"purpose":
    "roi:language"} vs {"purpose": "tmap_export"} so a shared status file
    stays legible about WHICH caller's attempt each row belongs to).
    """
    cache_dir = Path(cache_dir)
    # roi_key "_registration" is deliberately not a real roi_key: this cache
    # entry is registration-only (see _cache_paths), and any roi_key routes
    # to the exact same registration_json/registration_npz paths -- the
    # value passed here never reaches the roi_mask/qc_png paths this function
    # doesn't use.
    paths = _cache_paths(cache_dir, subject, session, "_registration")
    reg = _load_registration(paths)
    if reg is not None:
        return reg

    row = {"subject": subject, "session": session, **(status_extra or {})}

    t1_path = find_t1w(Path(data_dir), subject, session)
    if t1_path is None:
        logger.warning("  %s/%s: no T1w anatomical found", subject, session)
        if status_csv:
            _log_status(status_csv, {**row, "status": "no_anat", "detail": ""})
        return None

    if epi_ref_img is None:
        logger.warning("  %s/%s: no cached registration and no EPI reference volume supplied", subject, session)
        if status_csv:
            _log_status(status_csv, {**row, "status": "no_epi_ref", "detail": ""})
        return None

    try:
        reg = register_subject_to_mni(epi_ref_img, t1_path, template_resolution)
        _save_registration(paths, reg, t1_path)
        return reg
    except Exception as e:
        logger.warning("  %s/%s: registration failed (%s)", subject, session, e)
        if status_csv:
            _log_status(status_csv, {**row, "status": "registration_failed", "detail": str(e)})
        return None


def build_subject_roi_mask(
    *,
    subject: str,
    session: str,
    data_dir: Path,
    cache_dir: Path,
    roi_sets: list,
    epi_ref_img: Optional[nib.Nifti1Image] = None,
    template_resolution: int = 2,
    write_qc: bool = True,
) -> Optional[np.ndarray]:
    """The single entry point `fmri_preprocessing.py` calls.

    Returns a boolean array on `epi_ref_img`'s grid marking the union of the
    requested ROI sets in this subject's native space, or None if it could
    not be built (missing T1, registration failed) -- in which case the
    caller falls back to the whole-brain EPI mask and this function has
    already logged why, both to the log stream and to
    `<cache_dir>/roi_mask_status.csv`.

    Idempotent and cached on disk per (subject, session): a second call for
    the same subject-session (e.g. a different task, or a re-run after an
    interruption) reuses the registration; a second call with a DIFFERENT
    roi_sets reuses the registration and only re-warps the new mask.
    """
    roi_key = "+".join(sorted(roi_sets))
    paths = _cache_paths(Path(cache_dir), subject, session, roi_key)

    if paths["roi_mask"].exists():
        img = nib.load(str(paths["roi_mask"]))
        return np.asarray(img.get_fdata()) > 0

    status_row = {"subject": subject, "session": session, "roi_set": roi_key}

    reg = get_or_register(
        subject=subject, session=session, data_dir=data_dir, cache_dir=cache_dir,
        epi_ref_img=epi_ref_img, template_resolution=template_resolution,
        status_csv=paths["status_csv"], status_extra={"roi_set": roi_key},
    )
    if reg is None:
        logger.warning("  %s/%s: falling back to whole-brain mask (see warning above)", subject, session)
        return None

    try:
        roi_mask_mni, matched_labels = build_roi_mask_mni(roi_sets, template_resolution=template_resolution)
        roi_native = warp_roi_to_native(reg, roi_mask_mni)
    except Exception as e:
        logger.warning("  %s/%s: ROI warp failed (%s) -- falling back to whole-brain mask", subject, session, e)
        _log_status(paths["status_csv"], {**status_row, "status": "warp_failed", "detail": str(e)})
        return None

    if roi_native.sum() == 0:
        logger.warning(
            "  %s/%s: warped ROI mask is empty (registration likely converged to the "
            "wrong place despite passing the sanity check) -- falling back to whole-brain mask",
            subject, session,
        )
        _log_status(paths["status_csv"], {**status_row, "status": "empty_after_warp", "detail": ""})
        return None

    # The native-grid affine comes from the registration itself
    # (epi_to_t1's codomain -- the EPI grid it was fit against), not from
    # epi_ref_img directly: on a cache-reuse call (a second roi_set for an
    # already-registered subject-session) epi_ref_img is None, and the
    # output must still land on the correct grid.
    out_img = nib.Nifti1Image(roi_native.astype(np.uint8), reg.epi_to_t1.codomain_grid2world)
    # The tmp file needs ITS OWN valid ".nii.gz" trailing extension -- nibabel
    # infers the format from the filename and refuses to save to a name it
    # doesn't recognise (e.g. anything ending ".tmp"), so ".tmp" is inserted
    # BEFORE ".nii.gz" rather than appended after it. (Path.with_suffix()
    # would also be wrong here for a different reason: it only replaces the
    # last suffix, so on a ".nii.gz" name it mangles the ".nii" part too --
    # both failure modes were caught by tests/test_masking_pipeline.py.)
    tmp = paths["roi_mask"].parent / paths["roi_mask"].name.replace(".nii.gz", ".tmp.nii.gz")
    nib.save(out_img, str(tmp))
    tmp.rename(paths["roi_mask"])  # atomic within the cache dir

    if write_qc and reg.t1_path is not None:
        _write_qc_png(paths["qc_png"], reg.t1_path, reg, roi_mask_mni, subject, session)

    _log_status(paths["status_csv"], {
        **status_row, "status": "ok", "detail": "+".join(matched_labels),
        "epi_to_t1_translation_mm": reg.quality["epi_to_t1_translation_mm"],
        "t1_to_mni_translation_mm": reg.quality["t1_to_mni_translation_mm"],
        "dice": reg.quality["t1_in_mni_vs_template_brain_dice"],
        "n_roi_voxels": int(roi_native.sum()),
    })
    logger.info("  %s/%s: ROI mask ok, %d voxels (%s)", subject, session, int(roi_native.sum()), roi_key)
    return roi_native


def epi_reference_volume(bold_path: Path) -> nib.Nifti1Image:
    """Temporal-mean 3D volume from a 4D BOLD run, for use as the EPI side of
    registration. Any run for the subject-session works equally well since
    registration only needs typical brain contrast, not this specific run's
    task content -- see the caching note in the module docstring for which
    run ends up being used in practice."""
    return mean_img(str(bold_path))
