from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from ..data.models import AnalysisLimits, GrainResult


def detect_grains(
    image_bgr: np.ndarray,
    px_per_mm: float,
    limits: AnalysisLimits,
) -> Tuple[List[GrainResult], np.ndarray]:
    """Run the full watershed pipeline on *image_bgr*.

    Returns
    -------
    grains      : list of GrainResult (all, including excluded)
    label_mask  : integer-labeled ndarray  (0 = background)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE improves contrast uniformity across the field
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu threshold → binary
    _, binary = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If grains are darker than background, invert
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)

    # Morphological cleanup — structuring element ~40 % of min grain diameter
    se_px = max(2, int(0.4 * limits.min_mm * px_per_mm))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_px, se_px))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    binary_bool = binary.astype(bool)

    # Distance transform
    dist = distance_transform_edt(binary_bool)

    # Peak detection — minimum separation = 60 % of smallest expected grain radius
    min_grain_r_px = (limits.min_mm / 2.0) * px_per_mm
    min_dist = max(2, int(min_grain_r_px * 0.6))

    coords = peak_local_max(dist, min_distance=min_dist, labels=binary_bool)

    marker_mask = np.zeros(dist.shape, dtype=bool)
    if len(coords):
        marker_mask[coords[:, 0], coords[:, 1]] = True
    markers = label(marker_mask)

    # Watershed
    labeled = watershed(-dist, markers, mask=binary_bool)

    # Extract per-region properties
    props = regionprops(labeled)

    grains: List[GrainResult] = []
    grain_id = 1
    for prop in props:
        major_px = prop.axis_major_length
        minor_px = prop.axis_minor_length

        if major_px < 1 or minor_px < 1:
            continue

        major_mm = major_px / px_per_mm
        minor_mm = minor_px / px_per_mm

        # Size limits
        excluded = False
        reason = ""
        if major_mm > limits.max_mm:
            excluded = True
            reason = "exceeds max size"
        elif major_mm < limits.min_mm:
            excluded = True
            reason = "below min size"

        # Convert regionprops orientation → Qt rotation degrees
        #   regionprops.orientation : angle from row-axis (y-down) to major axis, CCW, radians
        #   Qt setRotation          : CW degrees from x-axis
        #   Conversion              : Qt_deg = 90 - degrees(orientation)
        orientation_deg = 90.0 - math.degrees(prop.orientation)

        cy, cx = prop.centroid  # row=y, col=x

        grains.append(
            GrainResult(
                id=grain_id,
                major_px=major_px,
                minor_px=minor_px,
                major_mm=major_mm,
                minor_mm=minor_mm,
                sphericalness=minor_mm / major_mm if major_mm > 0 else 1.0,
                orientation_deg=orientation_deg,
                centroid_x_px=float(cx),
                centroid_y_px=float(cy),
                excluded=excluded,
                exclusion_reason=reason,
            )
        )
        grain_id += 1

    return grains, labeled
