"""
Tumor Segmentation Pipeline
============================
End-to-end 3D tumour segmentation using bilateral filtering, adaptive
hysteresis thresholding, and optional marching-cubes shape analysis.

Pipeline steps:
    1. Bilateral filtering (slice-wise)
    2. Global thresholding → largest connected component (approximate tumour)
    3. Local intensity statistics inside the tumour bounding box
    4. Hysteresis thresholding with data-driven thresholds
    5. (Optional) Shape feature extraction via marching cubes
"""

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.io_utils import load_image_stack, save_3d_stack_as_tiff
from src.core.preprocessing import bilateral_filter_3d
from src.morphology.operations import find_connected_components_3d
from src.segmentation.thresholding import hysteresis_threshold_3d
from src.analysis.shape_analysis import shape_analysis


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def segment_tumor_hysteresis(
    image_stack: np.ndarray,
    global_threshold: float,
    folder_path: str,
    mask_name: str,
    *,
    save_mask: bool = True,
    low_thresh_coeff: float = 2.0,
    high_thresh_coeff: float = 1.0,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 30.0,
    bilateral_sigma_space: float = 5.0,
    return_intermediate: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    3D hysteresis-based tumour segmentation with adaptive thresholds.

    The low and high hysteresis thresholds are derived from the mean and
    standard deviation of voxel intensities inside the bounding box of the
    largest object found by global thresholding:

        ``low  = max(mu - low_thresh_coeff  * sigma, 0.01)``
        ``high = max(mu - high_thresh_coeff * sigma, low + 0.01)``

    Parameters
    ----------
    image_stack : np.ndarray
        3D image ``(Z, Y, X)``, unfiltered.
    global_threshold : float
        Threshold applied to the normalised filtered stack to obtain
        an initial foreground mask.
    folder_path : str
        Directory for saved outputs.
    mask_name : str
        Filename for the final mask (e.g. ``"tumor_mask.tiff"``).
    save_mask : bool
        Write the final mask to disk.
    low_thresh_coeff : float
        Number of standard deviations below the mean for the low threshold.
    high_thresh_coeff : float
        Number of standard deviations below the mean for the high threshold.
    bilateral_d : int
        Bilateral filter neighbourhood diameter.
    bilateral_sigma_color : float
        Bilateral filter colour-space sigma.
    bilateral_sigma_space : float
        Bilateral filter spatial sigma.
    return_intermediate : bool
        If True, also return a dictionary of intermediate results and save
        them to disk.

    Returns
    -------
    mask : np.ndarray
        Binary segmentation mask (0/1, int).
    intermediates : dict   *(only when* ``return_intermediate=True`` *)*
        Keys: ``filtered_stack``, ``initial_mask``, ``main_tumor``,
        ``low_thresh``, ``high_thresh``.
    """
    # 1. Bilateral filtering
    filtered = bilateral_filter_3d(
        image_stack,
        d=bilateral_d,
        sigma_color=bilateral_sigma_color,
        sigma_space=bilateral_sigma_space,
    )
    filtered_norm = filtered / filtered.max()

    # 2. Global threshold → initial mask
    initial_mask = filtered_norm > global_threshold

    # 3. Largest connected component (approximate tumour location)
    labeled, sizes = find_connected_components_3d(initial_mask)
    main_tumor = labeled == (np.argmax(sizes) + 1)

    # 4. Local statistics inside tumour bounding box
    coords = np.where(main_tumor)
    min_d, min_h, min_w = np.min(coords, axis=1)
    max_d, max_h, max_w = np.max(coords, axis=1)

    image_norm = image_stack / image_stack.max()
    tumor_box = image_norm[min_d:max_d, min_h:max_h, min_w:max_w]
    foreground_vals = tumor_box[tumor_box > global_threshold]

    mu = foreground_vals.mean()
    sigma = foreground_vals.std()

    low_thresh = max(mu - low_thresh_coeff * sigma, 0.01)
    high_thresh = max(mu - high_thresh_coeff * sigma, low_thresh + 0.01)

    # 5. Hysteresis thresholding
    mask = hysteresis_threshold_3d(filtered_norm, low_thresh, high_thresh)

    # 6. Save
    if save_mask:
        save_3d_stack_as_tiff(mask, folder_path, mask_name)

    if return_intermediate:
        base = os.path.splitext(mask_name)[0]
        save_3d_stack_as_tiff(filtered, folder_path, f"{base}_filtered.tiff")
        save_3d_stack_as_tiff(
            initial_mask.astype(np.uint8), folder_path, f"{base}_initial_mask.tiff",
        )
        save_3d_stack_as_tiff(
            main_tumor.astype(np.uint8), folder_path, f"{base}_main_tumor.tiff",
        )
        intermediates = {
            "filtered_stack": filtered,
            "initial_mask": initial_mask,
            "main_tumor": main_tumor,
            "low_thresh": low_thresh,
            "high_thresh": high_thresh,
        }
        return mask, intermediates

    return mask


# ---------------------------------------------------------------------------
# Single-image processing (segmentation + optional shape analysis)
# ---------------------------------------------------------------------------

def process_image(
    image_stack: np.ndarray,
    output_folder: str,
    mask_name: str = "tumor_mask.tiff",
    global_threshold: float = 0.05,
    x_y_resolution: float = 0.59,
    z_resolution: float = 4.0,
    run_shape_analysis: bool = True,
    save_mask: bool = True,
    return_intermediate: bool = False,
    low_thresh_coeff: float = 2.0,
    high_thresh_coeff: float = 1.0,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 30.0,
    bilateral_sigma_space: float = 5.0,
) -> Union[np.ndarray, pd.DataFrame, Tuple]:
    """
    Segment a 3D image and optionally extract shape features.

    Parameters
    ----------
    image_stack : np.ndarray
        3D image ``(Z, Y, X)``.
    output_folder : str
        Directory for all outputs.
    mask_name : str
        Filename for the saved mask.
    global_threshold : float
        Initial global threshold (applied to normalised intensities).
    x_y_resolution, z_resolution : float
        Voxel dimensions in microns.
    run_shape_analysis : bool
        Compute volume / surface area / sphericity features.
    save_mask : bool
        Write the mask to disk.
    return_intermediate : bool
        Return filtering / thresholding intermediates.
    low_thresh_coeff, high_thresh_coeff : float
        Hysteresis threshold coefficients (see :func:`segment_tumor_hysteresis`).
    bilateral_d, bilateral_sigma_color, bilateral_sigma_space
        Bilateral filter parameters.

    Returns
    -------
    pd.DataFrame or np.ndarray
        Shape-feature table (if ``run_shape_analysis``) or binary mask.
        When ``return_intermediate`` is True the return is a tuple
        ``(result, intermediates_dict)``.
    """
    seg_result = segment_tumor_hysteresis(
        image_stack=image_stack,
        global_threshold=global_threshold,
        folder_path=output_folder,
        mask_name=mask_name,
        save_mask=save_mask,
        low_thresh_coeff=low_thresh_coeff,
        high_thresh_coeff=high_thresh_coeff,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
        return_intermediate=return_intermediate,
    )

    if return_intermediate:
        mask_3d, debug_info = seg_result
    else:
        mask_3d = seg_result
        debug_info = None

    if run_shape_analysis:
        num_objs, vols, vols_mc, areas_mc, sphs_mc = shape_analysis(
            mask_3D=mask_3d,
            x_y_resolution=x_y_resolution,
            z_resolution=z_resolution,
        )
        df = pd.DataFrame({
            "Volume": vols,
            "Volume_MC": vols_mc,
            "Surface_Area_MC": areas_mc,
            "Sphericity_MC": sphs_mc,
        })
        return (df, debug_info) if return_intermediate else df

    return (mask_3d, debug_info) if return_intermediate else mask_3d


# ---------------------------------------------------------------------------
# Batch-compatible wrapper
# ---------------------------------------------------------------------------

def tumor_segmentation_pipeline_wrapper(
    file_path: str,
    fish_index: int,
    channel_num: int = 0,
    output_subfolder: str = "tumor_segmentation",
    mask_name: str = "tumor_mask.tiff",
    global_threshold: float = 0.05,
    x_y_resolution: float = 0.59,
    z_resolution: float = 4.0,
    run_shape_analysis: bool = True,
    save_mask: bool = True,
    return_intermediate: bool = False,
    low_thresh_coeff: float = 2.0,
    high_thresh_coeff: float = 1.0,
    bilateral_d: int = 7,
    bilateral_sigma_color: float = 30.0,
    bilateral_sigma_space: float = 5.0,
):
    """
    Load an image file and run :func:`process_image`.

    Designed to be called from a batch-processing loop.  The output folder
    is derived from the input path as
    ``<parent>/<fish{fish_index}>/<output_subfolder>/``.

    Parameters
    ----------
    file_path : str
        Path to the multi-channel image file.
    fish_index : int
        Sample index (used for folder and mask naming).
    channel_num : int
        Channel to extract.
    output_subfolder : str
        Sub-directory name under the per-fish folder.

    All remaining parameters are forwarded to :func:`process_image`.
    """
    image_stack = load_image_stack(file_path, channel_num=channel_num)

    parent_folder = os.path.dirname(file_path)
    output_folder = os.path.join(parent_folder, f"fish{fish_index}", output_subfolder)
    os.makedirs(output_folder, exist_ok=True)

    if mask_name == "tumor_mask.tiff":
        mask_name = f"fish{fish_index}_mask.tiff"

    result = process_image(
        image_stack=image_stack,
        output_folder=output_folder,
        mask_name=mask_name,
        global_threshold=global_threshold,
        x_y_resolution=x_y_resolution,
        z_resolution=z_resolution,
        run_shape_analysis=run_shape_analysis,
        save_mask=save_mask,
        return_intermediate=return_intermediate,
        low_thresh_coeff=low_thresh_coeff,
        high_thresh_coeff=high_thresh_coeff,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
    )

    if run_shape_analysis:
        df = result[0] if return_intermediate else result
        excel_path = os.path.join(output_folder, "shape_stats.xlsx")
        df.to_excel(excel_path, index=False)
        print(f"Shape analysis saved to: {excel_path}")

    return result