"""
Self-test for the masking/registration pipeline (MASKING.md).

RUN THIS FIRST, BEFORE SPENDING ANY GPU TIME. It needs no BOLD data, no GPU
and no OpenNeuro download -- only network access once, to fetch nilearn's AAL
atlas and (if not already cached) its bundled MNI152 template. On a machine
that already has these cached (`~/nilearn_data`), it runs with no network at
all. Wall-clock budget: well under two minutes; the registration tests use a
deliberately coarse synthetic grid to stay fast while still exercising dipy's
real optimizer against real brain-shaped data (a downsampled copy of the
actual bundled MNI152 template, not a toy blob) rather than mocking it.

    pytest tests/test_masking_pipeline.py -v

If this fails, nothing about the real pipeline has been touched yet -- fix it
here first. If it passes, the failure modes it checks for (silently-empty
mask, wrong-direction registration, missing-anat crash, unmatched ROI name
producing an empty mask) are the ones that would otherwise only show up after
a subject-batch has already spent GPU hours on the real cluster.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import roi_atlas
from src.preprocessing import spatial_normalization as spatial_norm


# --------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def mni_coarse():
    """A real (not synthetic-blob) brain volume, downsampled hard so the
    registration tests run in seconds instead of minutes. Downsampling is
    done by simple striding on the bundled MNI152 template + brain mask, so
    the result is still a real, brain-shaped, anatomically structured image
    -- exercising the actual optimizer, not a mocked one."""
    from nilearn import datasets

    template = datasets.load_mni152_template(resolution=2)
    brain_mask = datasets.load_mni152_brain_mask(resolution=2)
    data = np.asarray(template.get_fdata())
    mask = np.asarray(brain_mask.get_fdata()) > 0
    data = data * mask  # zero outside the brain, like a defaced/skull-stripped T1
    # Crop to the brain's own bounding box (+ small margin) rather than
    # striding: striding throws away fine structure, which starved mutual
    # information of anything to lock onto and produced a registration that
    # merely looked imprecise (Dice ~0.35) rather than clearly broken. A
    # tight crop keeps full local detail while still shrinking the array
    # enough to keep this test fast.
    xs, ys, zs = np.where(mask)
    pad = 3
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, data.shape[0])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, data.shape[1])
    z0, z1 = max(zs.min() - pad, 0), min(zs.max() + pad + 1, data.shape[2])
    cropped = data[x0:x1, y0:y1, z0:z1]
    affine = template.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(template.affine, [x0, y0, z0])
    img = nib.Nifti1Image(cropped.astype(np.float32), affine)
    return img


def _shift_affine(affine: np.ndarray, mm: tuple) -> np.ndarray:
    out = affine.copy()
    out[:3, 3] += np.array(mm)
    return out


@pytest.fixture(scope="module")
def synthetic_subject(mni_coarse, tmp_path_factory):
    """A synthetic BIDS-ish tree for one subject-session: a 'T1w' that is the
    coarse MNI brain with a small known rigid offset (standing in for "a real
    T1, roughly but not exactly in MNI space already"), and an 'EPI' that is
    a further-downsampled, further-shifted copy of it (standing in for a
    lower-resolution functional reference volume). Registration should be
    able to recover both known offsets to within the pipeline's own sanity
    thresholds.
    """
    root = tmp_path_factory.mktemp("bids")
    subject, session = "sub-9001", "ses-5"

    t1_data = np.asarray(mni_coarse.get_fdata())
    t1_affine = _shift_affine(mni_coarse.affine, (4.0, -3.0, 2.0))
    t1_img = nib.Nifti1Image(t1_data, t1_affine)

    anat_dir = root / subject / session / "anat"
    anat_dir.mkdir(parents=True)
    t1_path = anat_dir / f"{subject}_{session}_acq-D1S2_T1w.nii.gz"
    nib.save(t1_img, str(t1_path))
    # A second acquisition, to exercise the "multiple T1s, pick
    # deterministically" path -- deliberately a worse (noisier) copy so a
    # test could tell if the wrong one were ever picked.
    t1_path2 = anat_dir / f"{subject}_{session}_acq-D1S7_T1w.nii.gz"
    rng = np.random.default_rng(0)
    nib.save(nib.Nifti1Image(t1_data + rng.normal(0, 5, t1_data.shape).astype(np.float32), t1_affine),
              str(t1_path2))

    epi_data = t1_data[::2, ::2, ::2]
    epi_affine = _shift_affine(t1_affine, (-2.0, 1.0, -1.5))
    epi_affine[:3, :3] *= 2
    # A 3D reference volume, deliberately -- this is what
    # spatial_norm.epi_reference_volume() (mean_img over a 4D run) produces
    # for the real pipeline. register_subject_to_mni takes a 3D volume, not a
    # 4D run; that mismatch is exactly what test_register_rejects_4d_epi_ref
    # below checks for.
    epi_img = nib.Nifti1Image(epi_data.astype(np.float32), epi_affine)
    epi_4d_img = nib.Nifti1Image(
        np.stack([epi_data, epi_data * 0.98, epi_data * 1.02], axis=-1).astype(np.float32),
        epi_affine,
    )

    return {
        "root": root, "subject": subject, "session": session,
        "t1_path": t1_path, "t1_img": t1_img, "epi_img": epi_img,
        "epi_4d_img": epi_4d_img,
    }


# --------------------------------------------------------------- roi_atlas --
def test_parse_roi_sets_accepts_known_names():
    assert roi_atlas.parse_roi_sets("auditory,motor") == ["auditory", "motor"]


def test_parse_roi_sets_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown ROI set"):
        roi_atlas.parse_roi_sets("auditory,visual_cortex")


@pytest.mark.parametrize("roi_set", ["auditory", "motor", "language"])
def test_build_roi_mask_mni_nonempty(roi_set):
    mask_img, matched = roi_atlas.build_roi_mask_mni([roi_set])
    data = np.asarray(mask_img.get_fdata())
    assert data.sum() > 0
    assert len(matched) >= 1


def test_build_roi_mask_mni_auditory_and_motor_are_disjoint():
    """A basic correctness check independent of exact voxel counts: primary
    auditory cortex and primary motor cortex should not overlap."""
    aud, _ = roi_atlas.build_roi_mask_mni(["auditory"])
    mot, _ = roi_atlas.build_roi_mask_mni(["motor"])
    overlap = np.logical_and(
        np.asarray(aud.get_fdata()) > 0, np.asarray(mot.get_fdata()) > 0
    ).sum()
    assert overlap == 0


def test_build_roi_mask_mni_union_is_superset():
    aud, _ = roi_atlas.build_roi_mask_mni(["auditory"])
    mot, _ = roi_atlas.build_roi_mask_mni(["motor"])
    both, _ = roi_atlas.build_roi_mask_mni(["auditory", "motor"])
    aud_n = (np.asarray(aud.get_fdata()) > 0).sum()
    mot_n = (np.asarray(mot.get_fdata()) > 0).sum()
    both_n = (np.asarray(both.get_fdata()) > 0).sum()
    assert both_n == aud_n + mot_n  # disjoint, so union is the exact sum


def test_unmatched_substring_raises_not_silently_empty(monkeypatch):
    """The whole point of name-based lookup: a typo/renamed region must
    raise, never silently produce an empty or partial mask."""
    bad_sets = dict(roi_atlas.ROI_SETS)
    bad_sets["broken"] = ["Nonexistent_Region_Name_Xyz"]
    monkeypatch.setattr(roi_atlas, "ROI_SETS", bad_sets)
    with pytest.raises(ValueError, match="not found in the AAL label list"):
        roi_atlas.build_roi_mask_mni(["broken"])


# --------------------------------------------------------- spatial_norm ---
def test_find_t1w_picks_deterministically(synthetic_subject):
    found = spatial_norm.find_t1w(
        synthetic_subject["root"], synthetic_subject["subject"], synthetic_subject["session"]
    )
    assert found is not None
    assert found.name.endswith("acq-D1S2_T1w.nii.gz")  # lexicographically first


def test_find_t1w_missing_returns_none(synthetic_subject):
    found = spatial_norm.find_t1w(synthetic_subject["root"], "sub-doesnotexist", "ses-5")
    assert found is None


def test_register_rejects_4d_epi_ref_with_a_clear_error(synthetic_subject):
    """A 4D BOLD run passed where a 3D reference volume is expected must fail
    with a readable message, not a bare dipy IndexError three frames down in
    scalespace.py -- that's the difference between a log line a collaborator
    can act on and a traceback they have to send back to me to interpret."""
    with pytest.raises(ValueError, match="3D"):
        spatial_norm.register_subject_to_mni(
            synthetic_subject["epi_4d_img"], synthetic_subject["t1_path"],
        )


def test_register_subject_to_mni_passes_sanity_check(synthetic_subject):
    reg = spatial_norm.register_subject_to_mni(
        synthetic_subject["epi_img"], synthetic_subject["t1_path"], template_resolution=2,
    )
    assert reg.quality["ok"] is True
    assert reg.quality["t1_in_mni_vs_template_brain_dice"] > 0.5


def test_register_subject_to_mni_raises_on_garbage_t1(tmp_path, synthetic_subject):
    """A T1 that isn't a brain at all (pure noise) should fail the sanity
    check and raise -- not return a plausible-looking wrong registration."""
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, (30, 30, 30)).astype(np.float32)
    bad_t1_path = tmp_path / "noise_T1w.nii.gz"
    nib.save(nib.Nifti1Image(noise, np.eye(4)), str(bad_t1_path))
    with pytest.raises(RuntimeError, match="sanity check failed"):
        spatial_norm.register_subject_to_mni(synthetic_subject["epi_img"], bad_t1_path)


def test_warp_roi_to_native_shape_mismatch_raises(synthetic_subject):
    reg = spatial_norm.register_subject_to_mni(
        synthetic_subject["epi_img"], synthetic_subject["t1_path"], template_resolution=2,
    )
    wrong_shape_roi = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4))
    with pytest.raises(ValueError, match="does not match the MNI template grid"):
        spatial_norm.warp_roi_to_native(reg, wrong_shape_roi)


def test_warp_roi_to_native_recovers_a_planted_region(synthetic_subject, mni_coarse):
    """End-to-end direction check: plant a blob in MNI space, warp it down to
    native EPI space through both registration steps, and confirm the result
    is non-empty and roughly the expected size -- this is what would be
    silently backwards if the domain/codomain convention in
    spatial_normalization.py were ever inverted by a future edit."""
    reg = spatial_norm.register_subject_to_mni(
        synthetic_subject["epi_img"], synthetic_subject["t1_path"], template_resolution=2,
    )
    from nilearn import datasets

    template = datasets.load_mni152_template(resolution=2)
    blob = np.zeros(template.shape, dtype=np.uint8)
    cx, cy, cz = (np.array(template.shape) // 2)
    blob[cx - 4:cx + 4, cy - 4:cy + 4, cz - 4:cz + 4] = 1
    blob_img = nib.Nifti1Image(blob, template.affine)

    native = spatial_norm.warp_roi_to_native(reg, blob_img)
    assert native.shape == np.asarray(synthetic_subject["epi_img"].get_fdata()).shape[:3]
    assert native.sum() > 0
    # It should land inside the brain, not off in empty space -- the coarse
    # EPI's own nonzero footprint is the brain here (see mni_coarse fixture).
    epi_brain = np.asarray(synthetic_subject["epi_img"].get_fdata()) > 0
    frac_inside_brain = np.logical_and(native, epi_brain).sum() / max(native.sum(), 1)
    assert frac_inside_brain > 0.5


def test_warp_native_to_mni_roundtrips_a_planted_blob(synthetic_subject):
    """The reverse direction (native EPI statistical map -> MNI152), used to
    export real spatial figures rather than just ROI masks. Plant a blob in
    native EPI space, warp it to MNI, and check it lands inside the brain and
    that warping it straight back down (via warp_roi_to_native on a
    binarized copy) recovers most of the original voxels -- this is the
    check that would catch the forward/inverse direction being swapped."""
    reg = spatial_norm.register_subject_to_mni(
        synthetic_subject["epi_img"], synthetic_subject["t1_path"], template_resolution=2,
    )
    epi_data = np.asarray(synthetic_subject["epi_img"].get_fdata())
    native_map = np.zeros(epi_data.shape, dtype=np.float32)
    cx, cy, cz = np.array(epi_data.shape) // 2
    native_map[cx - 2:cx + 2, cy - 2:cy + 2, cz - 2:cz + 2] = 5.0
    native_img = nib.Nifti1Image(native_map, synthetic_subject["epi_img"].affine)

    mni_img = spatial_norm.warp_native_to_mni(reg, native_img)
    mni_data = np.asarray(mni_img.get_fdata())

    from nilearn import datasets
    template_shape = datasets.load_mni152_template(resolution=2).shape
    assert mni_data.shape == template_shape

    template_brain = np.asarray(datasets.load_mni152_brain_mask(resolution=2).get_fdata()) > 0
    strong = mni_data > 1.0
    assert strong.sum() > 0
    assert (strong & template_brain).sum() / strong.sum() > 0.5

    roi_from_map = nib.Nifti1Image(strong.astype(np.uint8), mni_img.affine)
    back_in_native = spatial_norm.warp_roi_to_native(reg, roi_from_map)
    original = native_map > 1.0
    assert np.logical_and(back_in_native, original).sum() / original.sum() > 0.3


def test_warp_native_to_mni_shape_mismatch_raises(synthetic_subject):
    reg = spatial_norm.register_subject_to_mni(
        synthetic_subject["epi_img"], synthetic_subject["t1_path"], template_resolution=2,
    )
    wrong_shape = nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4))
    with pytest.raises(ValueError, match="does not match the EPI grid"):
        spatial_norm.warp_native_to_mni(reg, wrong_shape)


# ---------------------------------------------------- build_subject_roi_mask
def test_build_subject_roi_mask_end_to_end(synthetic_subject, tmp_path):
    cache_dir = tmp_path / "masks"
    result = spatial_norm.build_subject_roi_mask(
        subject=synthetic_subject["subject"], session=synthetic_subject["session"],
        data_dir=synthetic_subject["root"], cache_dir=cache_dir,
        roi_sets=["motor"], epi_ref_img=synthetic_subject["epi_img"],
    )
    assert result is not None
    assert result.sum() > 0
    assert (cache_dir / "roi_mask_status.csv").exists()
    status = (cache_dir / "roi_mask_status.csv").read_text()
    assert "ok" in status
    # A QC image should have been written -- the thing to actually look at
    # without re-running anything.
    qc_files = list(cache_dir.rglob("*_qc.png"))
    assert len(qc_files) == 1


def test_build_subject_roi_mask_is_cached_on_second_call(synthetic_subject, tmp_path):
    cache_dir = tmp_path / "masks"
    first = spatial_norm.build_subject_roi_mask(
        subject=synthetic_subject["subject"], session=synthetic_subject["session"],
        data_dir=synthetic_subject["root"], cache_dir=cache_dir,
        roi_sets=["motor"], epi_ref_img=synthetic_subject["epi_img"],
    )
    # Second call passes epi_ref_img=None -- if this succeeds, it MUST have
    # come from the cache (registration is impossible without a reference).
    second = spatial_norm.build_subject_roi_mask(
        subject=synthetic_subject["subject"], session=synthetic_subject["session"],
        data_dir=synthetic_subject["root"], cache_dir=cache_dir,
        roi_sets=["motor"], epi_ref_img=None,
    )
    assert second is not None
    assert np.array_equal(first, second)


def test_build_subject_roi_mask_reuses_registration_for_a_new_roi_set(synthetic_subject, tmp_path):
    """Registration is the expensive/risky part; a second, DIFFERENT roi_set
    for an already-registered subject-session should not need epi_ref_img
    either -- only the (cheap) warp step should run again."""
    cache_dir = tmp_path / "masks"
    spatial_norm.build_subject_roi_mask(
        subject=synthetic_subject["subject"], session=synthetic_subject["session"],
        data_dir=synthetic_subject["root"], cache_dir=cache_dir,
        roi_sets=["motor"], epi_ref_img=synthetic_subject["epi_img"],
    )
    auditory = spatial_norm.build_subject_roi_mask(
        subject=synthetic_subject["subject"], session=synthetic_subject["session"],
        data_dir=synthetic_subject["root"], cache_dir=cache_dir,
        roi_sets=["auditory"], epi_ref_img=None,
    )
    assert auditory is not None
    assert auditory.sum() > 0


def test_build_subject_roi_mask_missing_anat_falls_back_gracefully(tmp_path, synthetic_subject):
    cache_dir = tmp_path / "masks"
    empty_root = tmp_path / "empty_bids"
    (empty_root / "sub-9002" / "ses-5" / "anat").mkdir(parents=True)
    result = spatial_norm.build_subject_roi_mask(
        subject="sub-9002", session="ses-5", data_dir=empty_root, cache_dir=cache_dir,
        roi_sets=["motor"], epi_ref_img=synthetic_subject["epi_img"],
    )
    assert result is None
    status = (cache_dir / "roi_mask_status.csv").read_text()
    assert "no_anat" in status


def test_build_subject_roi_mask_bad_registration_falls_back_gracefully(tmp_path):
    """A subject whose T1 is unusable should degrade to 'no ROI mask' for
    that one subject, never raise out of the batch."""
    cache_dir = tmp_path / "masks"
    root = tmp_path / "bad_bids"
    subject, session = "sub-9003", "ses-5"
    anat_dir = root / subject / session / "anat"
    anat_dir.mkdir(parents=True)
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, (30, 30, 30)).astype(np.float32)
    nib.save(nib.Nifti1Image(noise, np.eye(4)), str(anat_dir / f"{subject}_{session}_acq-X_T1w.nii.gz"))

    dummy_epi = nib.Nifti1Image(rng.normal(0, 1, (20, 20, 20, 2)).astype(np.float32), np.eye(4))
    result = spatial_norm.build_subject_roi_mask(
        subject=subject, session=session, data_dir=root, cache_dir=cache_dir,
        roi_sets=["motor"], epi_ref_img=dummy_epi,
    )
    assert result is None
    status = (cache_dir / "roi_mask_status.csv").read_text()
    assert "registration_failed" in status


# --------------------------------------------------- fmri_preprocessing ---
def test_no_clean_background_no_longer_masks_everything():
    """Direct reproduction of the original bug (paper_results/control/
    README.md: 917,504 voxels, 100% non-zero, 'NO BRAIN MASK'): a synthetic
    volume with NO true zero background -- every voxel has some signal, like
    the real ds003604 BOLD apparently does -- must NOT come back ~100%
    unmasked once `mask_strategy='epi'` is used instead of nilearn's default.
    """
    from nilearn.maskers import NiftiMasker

    rng = np.random.default_rng(3)
    shape = (40, 40, 30)
    brain = np.zeros(shape, dtype=np.float32)
    brain[10:30, 10:30, 8:22] = 500.0
    # No true background: everywhere gets some nonzero signal, reproducing
    # ds003604's measured failure mode for nilearn's 'background' strategy.
    everywhere_nonzero = brain + rng.uniform(1, 20, shape).astype(np.float32)
    img = nib.Nifti1Image(everywhere_nonzero, np.eye(4))

    old_default = NiftiMasker(standardize=False, detrend=False, memory=None, memory_level=0)
    old_default.fit(img)
    old_frac = (np.asarray(old_default.mask_img_.get_fdata()) > 0).mean()

    fixed = NiftiMasker(mask_strategy='epi', standardize=False, detrend=False,
                         memory=None, memory_level=0)
    fixed.fit(img)
    fixed_frac = (np.asarray(fixed.mask_img_.get_fdata()) > 0).mean()

    assert old_frac > 0.95, (
        "test fixture no longer reproduces the original bug -- the 'no clean "
        "background' synthetic volume should still defeat nilearn's default "
        "mask_strategy='background'; if this assertion fails, nilearn's "
        "default behaviour has changed and the whole premise of this fix "
        "needs re-checking, not just this test"
    )
    assert fixed_frac < 0.95, (
        "mask_strategy='epi' should exclude a meaningful fraction of the "
        "volume on data with no clean background -- it did not"
    )


def test_roi_set_and_mask_path_are_mutually_exclusive(tmp_path):
    from src.preprocessing.fmri_preprocessing import FMRIPreprocessor

    real_mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), str(real_mask_path))
    with pytest.raises(ValueError, match="mutually exclusive"):
        FMRIPreprocessor(
            data_dir=str(tmp_path), subject_id="sub-1", mask_path=str(real_mask_path),
            roi_set="motor",
        )


def test_roi_set_requires_mask_cache_dir(tmp_path):
    from src.preprocessing.fmri_preprocessing import FMRIPreprocessor

    with pytest.raises(ValueError, match="mask_cache_dir"):
        FMRIPreprocessor(data_dir=str(tmp_path), subject_id="sub-1", roi_set="motor")
