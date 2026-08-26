"""
Named anatomical ROI sets, defined by AAL region NAMES rather than numeric codes.

WHY NAMES, NOT NUMBERS. `prepare_language_mask.py --aal-rois` takes numeric AAL
codes and, when a number is not a real atlas code, silently reinterprets it as a
1-based POSITION in the label list instead (`build_mask_from_aal_roi_ids`'s
fallback). That fallback is not hypothetical: `run_analysis.py`'s
DEFAULT_LANGUAGE_ROIS = [7, 8, 9, 10, 11, 12, 67, 68, 69, 70, 85, 86] does not
match any real code in the AAL(SPM12) atlas nilearn fetches today -- its codes
look like 2001, 8101, not small integers -- so every one of those twelve numbers
silently falls through to "position N" and lands on
Frontal_Sup_Orb_R / Frontal_Mid_L / Frontal_Mid_R / Frontal_Mid_Orb_L /
Frontal_Mid_Orb_R / Frontal_Inf_Oper_L / Angular_R / Precuneus_L / Precuneus_R /
Paracentral_Lobule_L / Temporal_Pole_Sup_R / Temporal_Mid_L -- verified against
a live fetch of `nilearn.datasets.fetch_atlas_aal(version="SPM12")` on
2026-08-26. Precuneus and the paracentral lobule are not a standard language
network; whoever chose those twelve numbers most likely intended real AAL codes
under a different numbering (an older AAL version numbers regions 1-116
directly), and the silent fallback has been serving a different ROI set ever
since. See MASKING.md for the full writeup. This module does not fix that
mismatch -- changing what "language" means is a separate decision -- it exists
so AUDITORY and MOTOR cannot land in the same trap: every set here is matched by
substring against the atlas's own label strings, and an unmatched substring
raises immediately instead of silently selecting nothing or the wrong region.

The existing LANGUAGE set is reproduced here by the exact names the numeric
fallback resolves to today, so callers that want "language" get the same
regions as before -- traceable and by name, not a new judgement call.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np
from nilearn import datasets
from nilearn.image import resample_to_img

logger = logging.getLogger(__name__)

# Bilateral AAL(SPM12) region-name substrings per named set. A region is
# included if its label CONTAINS the substring (case-sensitive, matching AAL's
# own naming convention), so "Heschl" matches both Heschl_L and Heschl_R.
ROI_SETS: Dict[str, List[str]] = {
    # Primary auditory cortex (Heschl's gyrus) + secondary/associative
    # auditory cortex (superior temporal gyrus, immediately adjacent). This is
    # the standard "auditory cortex" ROI in the speech/auditory literature --
    # the right target for the acoustic-spectrum positive control (TODO.md).
    "auditory": ["Heschl_L", "Heschl_R", "Temporal_Sup_L", "Temporal_Sup_R"],
    # Primary motor cortex (precentral gyrus). Non-linguistic, movement-control
    # region -- a useful comparison/control area (TODO.md SS3: is any residual
    # alignment specific to language regions, or generic to any cortex?).
    "motor": ["Precentral_L", "Precentral_R"],
    # Reproduces run_analysis.py's DEFAULT_LANGUAGE_ROIS = [7, 8, 9, 10, 11, 12,
    # 67, 68, 69, 70, 85, 86] as the AAL(SPM12) region NAMES those numbers
    # actually resolve to today (via the position fallback described above),
    # so callers asking for "language" get an unchanged region set, addressed
    # by name so it cannot silently drift if the atlas's internal ordering
    # ever changes.
    "language": [
        "Frontal_Sup_Orb_R", "Frontal_Mid_L", "Frontal_Mid_R",
        "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R", "Frontal_Inf_Oper_L",
        "Angular_R", "Precuneus_L", "Precuneus_R", "Paracentral_Lobule_L",
        "Temporal_Pole_Sup_R", "Temporal_Mid_L",
    ],
}
# "phonology" and "all" are convenience UNIONS of the sets above, not
# independently-chosen regions -- defined by composition so they can never
# drift out of sync with edits to "auditory"/"motor"/"language" above.
ROI_SETS["phonology"] = sorted(set(ROI_SETS["auditory"]) | set(ROI_SETS["motor"]))
ROI_SETS["all"] = sorted(set(ROI_SETS["language"]) | set(ROI_SETS["phonology"]))


def available_roi_sets() -> List[str]:
    return sorted(ROI_SETS)


def parse_roi_sets(spec: str) -> List[str]:
    """'auditory,motor' -> ['auditory', 'motor']. Raises on an unknown name
    rather than silently ignoring it."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in ROI_SETS]
    if unknown:
        raise ValueError(
            f"Unknown ROI set(s) {unknown}; known: {available_roi_sets()}"
        )
    return names


def _match_labels(labels: List[str], substrings: List[str]) -> List[str]:
    """Every substring must match at least one label, or this raises. A
    partial match is exactly the failure mode this module exists to prevent:
    better to crash loudly during mask-building (cheap, no GPU involved) than
    to silently build a mask missing half its intended regions."""
    matched = [lab for lab in labels if any(s in lab for s in substrings)]
    missing = [s for s in substrings if not any(s in lab for lab in labels)]
    if missing:
        raise ValueError(
            f"ROI substrings not found in the AAL label list, refusing to "
            f"build a partial mask silently: {missing}. First 10 labels in "
            f"this atlas: {labels[:10]}"
        )
    return matched


def build_roi_mask_mni(
    roi_sets: List[str],
    template_resolution: int = 2,
    aal_version: str = "SPM12",
) -> Tuple[nib.Nifti1Image, List[str]]:
    """Union of one or more named ROI sets, as a binary mask on the MNI152
    template grid at `template_resolution` mm.

    Returns (mask_image, matched_region_names) -- the names are returned so
    callers can log/record exactly which AAL regions went into the mask,
    rather than trusting the set name alone.
    """
    unknown = [r for r in roi_sets if r not in ROI_SETS]
    if unknown:
        raise ValueError(f"Unknown ROI set(s) {unknown}; known: {available_roi_sets()}")
    if not roi_sets:
        raise ValueError("roi_sets is empty")

    atlas = datasets.fetch_atlas_aal(version=aal_version)
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    labels = list(atlas.labels)
    indices = [int(i) for i in atlas.indices]

    substrings: List[str] = []
    for r in roi_sets:
        substrings.extend(ROI_SETS[r])
    matched_labels = _match_labels(labels, substrings)
    matched_codes = [indices[labels.index(lab)] for lab in matched_labels]

    mask_data = np.isin(atlas_data.astype(int), matched_codes).astype(np.uint8)
    mask_img = nib.Nifti1Image(mask_data, atlas_img.affine, atlas_img.header)

    # Re-grid (not register -- both are already true MNI world coordinates)
    # onto the exact template grid registration will target, so the two line
    # up voxel-for-voxel with no further resampling needed downstream.
    template = datasets.load_mni152_template(resolution=template_resolution)
    mask_on_template = resample_to_img(mask_img, template, interpolation="nearest")
    data = (np.asarray(mask_on_template.get_fdata()) > 0).astype(np.uint8)
    mask_on_template = nib.Nifti1Image(data, mask_on_template.affine, mask_on_template.header)

    n_vox = int(data.sum())
    logger.info(
        "  ROI set %s -> %d AAL regions (%s), %d voxels at %dmm MNI",
        "+".join(roi_sets), len(matched_labels), ", ".join(matched_labels),
        n_vox, template_resolution,
    )
    if n_vox == 0:
        raise RuntimeError(
            f"ROI mask for {roi_sets} is empty after re-gridding -- this "
            "should never happen and means the atlas or template changed "
            "shape unexpectedly. Refusing to return an empty mask."
        )
    return mask_on_template, matched_labels
