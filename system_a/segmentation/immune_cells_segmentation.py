"""
Invading Cells Segmentation
============================
Bootstrap segmentation for invading / migrating cells (e.g. microglia)
using percentile-based hysteresis thresholding.

Pipeline overview
-----------------
The full invading-cell segmentation pipeline used an iterative
bootstrap-and-refine strategy:

1. **Percentile-based hysteresis** (this module) — generate initial pseudo-
   labels by applying 3D hysteresis thresholding with percentile-derived
   thresholds.  Thresholds were tuned per image stack because intensity
   distributions varied across samples.
2. **nnU-Net training** — pseudo-labels were used to train a nnU-Net model
   (Dataset ID 305, 2D configuration, fold 0).  Training used the standard
   ``nnUNetv2_train`` command; no custom training code was added.
3. **Inference** — the trained model was applied to unlabelled samples via
   ``nnUNetv2_predict``.

The nnU-Net model weights are not included in this repository.  The
percentile-based bootstrap function below is the only code-level
contribution from this stage; the remainder of the pipeline relies on
nnU-Net's standard CLI and manual annotation.
"""

import numpy as np
from typing import Tuple

from src.segmentation.hystresis_3D import *
from src.core.save_and_load_images import save_3d_stack_as_tiff


def segment_imune_cells_bootstrap(
    image_stack: np.ndarray,
    low_percentile: float = 85.0,
    high_percentile: float = 95.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Generate an initial segmentation mask for invading cells using
    percentile-based hysteresis thresholding.

    This function was used to **bootstrap pseudo-labels** for subsequent
    nnU-Net training.  Because intensity distributions differed across
    samples, the percentile values were adjusted per image stack during
    the labelling phase.

    Parameters
    ----------
    image_stack : np.ndarray
        3D image ``(Z, Y, X)``, typically a single fluorescence channel.
    low_percentile : float
        Percentile used to set the low hysteresis threshold.  Voxels
        below this are classified as background.
    high_percentile : float
        Percentile used to set the high hysteresis threshold.  Voxels
        above this are classified as foreground.

    Returns
    -------
    mask : np.ndarray
        Binary mask (0/1, int), same shape as *image_stack*.
    low_thresh : float
        The computed low threshold value.
    high_thresh : float
        The computed high threshold value.

    Notes
    -----
    The resulting mask is intended as a *starting point* for nnU-Net
    training, not as a final segmentation.

    Examples
    --------
    >>> mask, lo, hi = segment_invading_cells_bootstrap(
    ...     stack, low_percentile=80, high_percentile=92
    ... )
    """
    low_thresh = float(np.percentile(image_stack, low_percentile))
    high_thresh = float(np.percentile(image_stack, high_percentile))

    if high_thresh <= low_thresh:
        high_thresh = low_thresh + 1e-3

    mask = hysteresis_threshold_3d(image_stack, low_thresh, high_thresh)

    return mask, low_thresh, high_thresh


def bootstrap_and_save(
    image_stack: np.ndarray,
    output_folder: str,
    filename: str = "bootstrap_mask.tiff",
    low_percentile: float = 85.0,
    high_percentile: float = 95.0,
) -> Tuple[np.ndarray, float, float]:
    """
    Convenience wrapper: segment and save the bootstrap mask to disk.

    Parameters
    ----------
    image_stack : np.ndarray
        3D image ``(Z, Y, X)``.
    output_folder : str
        Directory to save the mask.
    filename : str
        Output filename.
    low_percentile, high_percentile : float
        See :func:`segment_invading_cells_bootstrap`.

    Returns
    -------
    mask : np.ndarray
        Binary bootstrap mask.
    low_thresh, high_thresh : float
        Computed threshold values.
    """
    mask, low_thresh, high_thresh = segment_imune_cells_bootstrap(
        image_stack,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )

    save_3d_stack_as_tiff(mask.astype(np.uint8), output_folder, filename)
    print(f"Bootstrap mask saved: {filename}")
    print(f"  Thresholds: low={low_thresh:.4f}, high={high_thresh:.4f}")

    return mask, low_thresh, high_thresh