"""Anatomical brain-region plotting for brainalign-evals.

This is the library behind figure S3. It exists so that "draw the ROIs we
masked to, coloured by what we measured in them" is one call rather than
thirty lines of nilearn buried inside a figure function.

WHAT THIS CAN AND CANNOT DRAW -- read before using it
-----------------------------------------------------
It draws **mask definitions** in MNI152 space: the set of AAL(SPM12) parcels
named by ``src/preprocessing/roi_atlas.ROI_SETS``, rendered as a glass brain or
as slice overlays, optionally tinted by a scalar you measured under that mask.

It does **not** draw per-voxel effect maps, because this pipeline cannot
produce them after the fact. ``src/preprocessing/fmri_preprocessing.py`` saves
each stimulus's pattern as a flat masked voxel vector
(``masker.transform(...).ravel()``) and does not save the mask affine/shape
alongside it, so there is no way to put those values back into 3D space from
what is on disk. See the module docstring of
``scripts/plot_activation_by_age_domain.py`` for the same warning. Producing
real per-voxel maps is a preprocessing change (store the affine), not a
plotting one.

So the honest figure is: the region we looked in, coloured by the single
number we got out of it. ``roi_mask_img`` gives you the first half and
``plot_roi_row`` / ``plot_roi_value_row`` the second.

OFFLINE USE
-----------
Everything here works without network access provided the AAL atlas has been
fetched once. On this machine it is already cached at
``<repo>/../nilearn_data/aal_SPM12``, which is the default ``data_dir``. The
MNI152 template and the glass-brain outlines ship inside nilearn itself.

EXAMPLES
--------
Draw the three masks used in the paper, each tinted by its own mean rho::

    from viz.brainplot import plot_roi_value_row
    fig = plot_roi_value_row(
        {"auditory": 0.0135, "motor": 0.0091, "phonology": 0.0041},
        cmap="RdBu_r", vlim=0.02, label=r"grammar $\\rho$")

Get a mask as a Nifti image and do your own thing with it::

    from viz.brainplot import roi_mask_img
    img, labels, n_voxels = roi_mask_img("language")
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

#: Where the AAL atlas is cached. nilearn re-downloads if this is missing, so
#: on an offline box point it at an existing copy rather than letting it try.
DEFAULT_ATLAS_DIR = _REPO.parent / "nilearn_data"

#: Views for a glass brain. "lzr" is left sagittal, axial, right sagittal --
#: enough to place a temporal or precentral ROI without a full mosaic.
DEFAULT_DISPLAY_MODE = "lzr"

__all__ = [
    "DEFAULT_ATLAS_DIR", "DEFAULT_DISPLAY_MODE",
    "available_roi_sets", "roi_mask_img", "roi_voxel_counts",
    "plot_roi_glass", "plot_roi_row", "plot_roi_value_row", "plot_roi_slices",
]


def _atlas(data_dir: str | Path | None = None):
    from nilearn import datasets
    return datasets.fetch_atlas_aal(
        version="SPM12", data_dir=str(data_dir or DEFAULT_ATLAS_DIR))


def available_roi_sets() -> list[str]:
    """Names accepted by every function here (from roi_atlas.ROI_SETS)."""
    from preprocessing.roi_atlas import ROI_SETS
    return sorted(ROI_SETS)


def roi_mask_img(roi_set: str, data_dir: str | Path | None = None):
    """Binary MNI-space mask for a named ROI set.

    Parameters
    ----------
    roi_set
        A key of ``roi_atlas.ROI_SETS``: "auditory", "motor", "language",
        "phonology", "all", or "language_aal_legacy".
    data_dir
        Where the AAL atlas is cached. Defaults to ``DEFAULT_ATLAS_DIR``.

    Returns
    -------
    (img, labels, n_voxels)
        A Nifti1Image of 0/1 in the atlas's own space, the AAL label strings
        that matched, and the voxel count.

    Notes
    -----
    Matching is by region-name substring, exactly as ``roi_atlas`` defines it,
    and an unmatched substring raises there rather than silently producing a
    partial mask. That is the whole reason the ROI sets are named rather than
    numbered -- see the roi_atlas module docstring.
    """
    import nibabel as nib
    from nilearn import image
    from preprocessing.roi_atlas import ROI_SETS

    if roi_set not in ROI_SETS:
        raise ValueError(f"unknown ROI set {roi_set!r}; "
                         f"known: {available_roi_sets()}")
    atlas = _atlas(data_dir)
    img = nib.load(atlas.maps)
    data = np.asarray(img.dataobj)
    labels = list(atlas.labels)
    wanted = [lab for lab in labels
              if any(s in lab for s in ROI_SETS[roi_set])]
    if not wanted:
        raise ValueError(f"ROI set {roi_set!r} matched no AAL label")
    codes = [int(atlas.indices[labels.index(lab)]) for lab in wanted]
    mask = np.isin(data, codes).astype(np.int16)
    return image.new_img_like(img, mask), wanted, int(mask.sum())


def roi_voxel_counts(roi_sets: Iterable[str],
                     data_dir: str | Path | None = None) -> dict[str, int]:
    """Voxel count per ROI set, for captions and sanity checks."""
    return {r: roi_mask_img(r, data_dir)[2] for r in roi_sets}


def plot_roi_glass(roi_set: str, ax=None, color: str = "#1b9e77",
                   display_mode: str = DEFAULT_DISPLAY_MODE,
                   alpha: float | None = None,
                   data_dir: str | Path | None = None):
    """Draw one ROI set as a filled contour on a glass brain.

    Returns the nilearn display object, so you can keep adding to it.

    ``alpha`` is passed through only if given: nilearn has moved this kwarg
    between versions (it now prefers ``transparency``), and a half-transparent
    mask tends to read as a weaker effect anyway, so the default is solid.
    """
    from nilearn import plotting

    img, _, _ = roi_mask_img(roi_set, data_dir)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.0, 1.2))
    disp = plotting.plot_glass_brain(
        None, axes=ax, display_mode=display_mode, plot_abs=False,
        annotate=False)
    kw = {} if alpha is None else {"alpha": alpha}
    disp.add_contours(img, levels=[0.5], colors=[color], filled=True, **kw)
    return disp


def plot_roi_row(roi_sets: Sequence[str], colors: Mapping[str, str],
                 fig=None, axes=None, titles: Mapping[str, str] | None = None,
                 display_mode: str = DEFAULT_DISPLAY_MODE,
                 show_voxels: bool = True,
                 data_dir: str | Path | None = None):
    """A row of glass brains, one per ROI set, each in its own colour.

    Pass ``axes`` to draw into an existing figure (this is what figure S3
    does, so the brains share a gridspec with the result panels below them).
    """
    if axes is None:
        fig, axes = plt.subplots(1, len(roi_sets),
                                 figsize=(2.3 * len(roi_sets), 1.3))
        axes = np.atleast_1d(axes)
    for ax, roi in zip(axes, roi_sets):
        _, _, nvox = roi_mask_img(roi, data_dir)
        plot_roi_glass(roi, ax=ax, color=colors[roi],
                       display_mode=display_mode, data_dir=data_dir)
        title = (titles or {}).get(roi, roi)
        if show_voxels:
            title = f"{title}  ({nvox:,} voxels)"
        ax.set_title(title, fontsize=7, pad=2)
    return fig, axes


def plot_roi_value_row(values: Mapping[str, float], cmap: str = "RdBu_r",
                       vlim: float | None = None, label: str | None = None,
                       fig=None, axes=None,
                       display_mode: str = DEFAULT_DISPLAY_MODE,
                       data_dir: str | Path | None = None):
    """A row of glass brains tinted by a measured scalar, with a colourbar.

    ``values`` maps ROI-set name to the number you measured under that mask
    (a mean rho, a paired change, a t statistic). ``vlim`` sets a symmetric
    colour range; it defaults to the largest absolute value present, which
    makes two figures with different value ranges non-comparable -- pass it
    explicitly whenever you intend a comparison.
    """
    import matplotlib as mpl

    rois = list(values)
    if vlim is None:
        vlim = max(abs(v) for v in values.values()) or 1.0
    norm = mpl.colors.Normalize(-vlim, vlim)
    cm = plt.get_cmap(cmap)
    colors = {r: mpl.colors.to_hex(cm(norm(values[r]))) for r in rois}
    titles = {r: f"{r}  ({values[r]:+.4f})" for r in rois}

    fig, axes = plot_roi_row(rois, colors, fig=fig, axes=axes, titles=titles,
                             display_mode=display_mode, show_voxels=False,
                             data_dir=data_dir)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    cb = (fig or plt.gcf()).colorbar(sm, ax=list(np.atleast_1d(axes)),
                                     fraction=0.02, pad=0.01)
    if label:
        cb.set_label(label, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    return fig, axes


def plot_roi_slices(roi_set: str, ax=None, color: str = "#1b9e77",
                    cut_coords: Sequence[float] | int = 5,
                    display_mode: str = "z",
                    data_dir: str | Path | None = None):
    """The same mask on anatomical slices over the MNI152 template.

    Use this instead of the glass brain when a reader needs to see depth --
    a glass brain projects through the volume and makes a deep ROI look
    larger than it is.
    """
    from nilearn import datasets, plotting

    img, _, _ = roi_mask_img(roi_set, data_dir)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 1.4))
    disp = plotting.plot_anat(
        datasets.load_mni152_template(), axes=ax, display_mode=display_mode,
        cut_coords=cut_coords, annotate=False, draw_cross=False)
    disp.add_contours(img, levels=[0.5], colors=[color], filled=True)
    return disp
