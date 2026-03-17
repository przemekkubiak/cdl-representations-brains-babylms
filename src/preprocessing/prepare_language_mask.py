"""
Prepare a language-responsive mask for fMRI preprocessing.

This script can:
1) Load one or more NIfTI masks
2) Threshold probabilistic maps
3) Combine masks (union or intersection)
4) Resample to a reference BOLD image
5) Save a binary mask NIfTI
"""

from pathlib import Path
import argparse
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import datasets


def load_and_binarize_mask(mask_path: Path, threshold: float = 0.0) -> nib.Nifti1Image:
    """Load a NIfTI mask and binarize it using threshold."""
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    binary = (data > threshold).astype(np.uint8)
    return nib.Nifti1Image(binary, img.affine, img.header)


def combine_masks(mask_imgs, mode: str = "union") -> nib.Nifti1Image:
    """Combine multiple masks in the same space."""
    if len(mask_imgs) == 1:
        return mask_imgs[0]

    ref = mask_imgs[0]
    combined = ref.get_fdata().astype(bool)

    for m in mask_imgs[1:]:
        data = m.get_fdata().astype(bool)
        if mode == "union":
            combined = np.logical_or(combined, data)
        elif mode == "intersection":
            combined = np.logical_and(combined, data)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    out = combined.astype(np.uint8)
    return nib.Nifti1Image(out, ref.affine, ref.header)


def build_mask_from_aal_roi_ids(roi_ids, atlas_version: str = "SPM12") -> nib.Nifti1Image:
    """Build binary mask from AAL atlas ROI numbers."""
    atlas = datasets.fetch_atlas_aal(version=atlas_version)
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()

    # AAL indices can be strings, map them to ints where possible.
    atlas_indices = []
    for idx in atlas.indices:
        try:
            atlas_indices.append(int(idx))
        except ValueError:
            # Keep non-int labels unreachable by numeric ROI input.
            continue

    requested = [int(r) for r in roi_ids]
    selected_codes = []
    missing = []

    for roi in requested:
        # Primary interpretation: ROI number is the AAL atlas code.
        if roi in atlas_indices:
            selected_codes.append(roi)
            continue

        # Fallback: treat ROI number as 1-based position in AAL label list.
        if 1 <= roi <= len(atlas.indices):
            try:
                fallback_code = int(atlas.indices[roi - 1])
                selected_codes.append(fallback_code)
                continue
            except ValueError:
                pass

        missing.append(roi)

    if not selected_codes:
        raise ValueError("None of the requested ROI numbers were found in the AAL atlas")

    mask_data = np.isin(atlas_data.astype(int), selected_codes).astype(np.uint8)
    out = nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)

    print(f"Using AAL atlas version: {atlas_version}")
    print(f"Requested ROI numbers: {requested}")
    print(f"Matched atlas codes: {sorted(set(selected_codes))}")
    if missing:
        print(f"ROI numbers not matched: {missing}")

    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare language-responsive NIfTI mask")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--mask-files",
        nargs="+",
        help="One or more input NIfTI mask files"
    )
    source_group.add_argument(
        "--aal-rois",
        nargs="+",
        type=int,
        help="AAL ROI numbers to include (e.g., 7 8 9 10 11 12 67 68 69 70 85 86)"
    )
    parser.add_argument(
        "--aal-version",
        type=str,
        default="SPM12",
        help="AAL atlas version for nilearn fetch_atlas_aal (default: SPM12)"
    )
    parser.add_argument(
        "--output-mask",
        type=str,
        default="data/processed/fmri/language_mask.nii.gz",
        help="Output binary mask path"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Threshold for probabilistic masks (default: 0.0)"
    )
    parser.add_argument(
        "--combine",
        type=str,
        choices=["union", "intersection"],
        default="union",
        help="How to combine multiple masks (default: union)"
    )
    parser.add_argument(
        "--reference-bold",
        type=str,
        help="Optional reference BOLD NIfTI to resample mask into functional space"
    )

    args = parser.parse_args()

    if args.aal_rois:
        print("Building mask from AAL ROI numbers...")
        combined = build_mask_from_aal_roi_ids(args.aal_rois, atlas_version=args.aal_version)
    else:
        mask_paths = [Path(p) for p in args.mask_files]
        for p in mask_paths:
            if not p.exists():
                raise FileNotFoundError(f"Mask not found: {p}")

        print("Loading masks...")
        mask_imgs = [load_and_binarize_mask(p, threshold=args.threshold) for p in mask_paths]

        print(f"Combining masks with mode: {args.combine}")
        combined = combine_masks(mask_imgs, mode=args.combine)

    if args.reference_bold:
        ref_path = Path(args.reference_bold)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference BOLD not found: {ref_path}")
        print(f"Resampling mask to reference BOLD space: {ref_path}")
        ref_img = nib.load(str(ref_path))
        combined = resample_to_img(combined, ref_img, interpolation="nearest")
        data = (combined.get_fdata() > 0).astype(np.uint8)
        combined = nib.Nifti1Image(data, combined.affine, combined.header)

    output_path = Path(args.output_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(combined, str(output_path))

    n_voxels = int(np.sum(combined.get_fdata() > 0))
    print(f"Saved mask to: {output_path}")
    print(f"Mask voxels: {n_voxels}")


if __name__ == "__main__":
    main()
