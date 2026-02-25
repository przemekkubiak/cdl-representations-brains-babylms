"""
Utilities for loading and preprocessing neuroimaging data.
"""

import numpy as np
from typing import Optional, Tuple, Union
import nibabel as nib


def load_fmri_data(
    filepath: str,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Load fMRI data from NIfTI file.
    
    Parameters
    ----------
    filepath : str
        Path to NIfTI file
    mask : np.ndarray, optional
        Brain mask to apply
        
    Returns
    -------
    np.ndarray
        Loaded fMRI data
    """
    img = nib.load(filepath)
    data = img.get_fdata()
    
    if mask is not None:
        data = data[mask]
    
    return data


def extract_roi_timeseries(
    data: np.ndarray,
    roi_mask: np.ndarray,
    aggregation: str = "mean"
) -> np.ndarray:
    """
    Extract timeseries from a region of interest.
    
    Parameters
    ----------
    data : np.ndarray
        4D fMRI data (x, y, z, time)
    roi_mask : np.ndarray
        3D binary mask defining ROI
    aggregation : str
        How to aggregate voxels ('mean', 'median', 'pca')
        
    Returns
    -------
    np.ndarray
        ROI timeseries
    """
    roi_data = data[roi_mask]
    
    if aggregation == "mean":
        return np.mean(roi_data, axis=0)
    elif aggregation == "median":
        return np.median(roi_data, axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
