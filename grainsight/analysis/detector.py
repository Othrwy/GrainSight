from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import watershed

from ..data.models import AnalysisLimits, CupMask, GrainResult


def detect_grains(
    image_bgr: np.ndarray,
    px_per_mm: float,
    limits: AnalysisLimits,
    cup_mask: Optional[CupMask] = None,
) -> Tuple[List[GrainResult], np.ndarray]:
    """Run the full watershed pipeline on *image_bgr*.

    Parameters
    ----------
    image_bgr   : BGR image
    px_per_mm   : calibration scale
    limits      : size filter limits
    cup_mask    : optional circular region-of-interest; grains outside or
                  touching the boundary are excluded

    Returns
    -------
    grains      : list of GrainResult (all, including excluded)
    label_mask  : integer-labeled ndarray  (0 = background)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Build circular ROI mask if a cup_mask is supplied
    h_img, w_img = gray.shape
    if cup_mask is not None:
        roi_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cx_i = int(round(cup_mask.cx_px))
        cy_i = int(round(cup_mask.cy_px))
        r_i  = int(round(cup_mask.radius_px))
        cv2.circle(roi_mask, (cx_i, cy_i), r_i, 255, thickness=-1)
    else:
        roi_mask = None

    # CLAHE improves contrast uniformity across the field
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu threshold — compute from inside-cup pixels only to avoid bias
    # from the bright image background outside the receptacle.
    if roi_mask is not None:
        roi_pixels = enhanced[roi_mask > 0].reshape(1, -1).astype(np.uint8)
        thresh_val, _ = cv2.threshold(
            roi_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = (enhanced >= int(thresh_val)).view(np.uint8) * 255
    else:
        _, binary = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # Inversion check — evaluate mean only within the cup to avoid bright
    # background pixels flipping the polarity the wrong way.
    if roi_mask is not None:
        check_mean = float(cv2.mean(binary, mask=roi_mask)[0])
    else:
        check_mean = float(binary.mean())
    if check_mean > 127:
        binary = cv2.bitwise_not(binary)

    # Morphological cleanup — structuring element ~40 % of min grain diameter
    se_px = max(2, int(0.4 * limits.min_mm * px_per_mm))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_px, se_px))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Apply circular ROI mask — restrict analysis to the inside of the cup
    if roi_mask is not None:
        binary = cv2.bitwise_and(binary, roi_mask)

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

        # Boundary exclusion: reject grains whose ellipse overlaps the cup rim
        # or the image edge (catches false positives at the detection boundary)
        cy_f, cx_f = prop.centroid
        semi_major_px = major_px / 2.0

        # Circularity filter — rejects texture artefacts from cup material
        # which tend to be elongated or irregular rather than round.
        perim = prop.perimeter
        if perim > 0:
            circ = 4.0 * math.pi * prop.area / (perim * perim)
        else:
            circ = 0.0
        if not excluded and limits.min_circularity > 0 and circ < limits.min_circularity:
            excluded = True
            reason = "not circular"

        if not excluded:
            if cup_mask is not None:
                # Distance from centroid to cup rim
                dist_from_centre = math.sqrt(
                    (cx_f - cup_mask.cx_px) ** 2 + (cy_f - cup_mask.cy_px) ** 2
                )
                if dist_from_centre + semi_major_px > cup_mask.radius_px:
                    excluded = True
                    reason = "on cup boundary"
            else:
                # No cup mask — exclude grains touching the image edge
                margin = semi_major_px
                if (
                    cx_f - margin < 0
                    or cy_f - margin < 0
                    or cx_f + margin >= w_img
                    or cy_f + margin >= h_img
                ):
                    excluded = True
                    reason = "on image boundary"

        # Convert regionprops orientation → Qt rotation degrees
        orientation_deg = 90.0 - math.degrees(prop.orientation)

        grains.append(
            GrainResult(
                id=grain_id,
                major_px=major_px,
                minor_px=minor_px,
                major_mm=major_mm,
                minor_mm=minor_mm,
                sphericalness=minor_mm / major_mm if major_mm > 0 else 1.0,
                orientation_deg=orientation_deg,
                centroid_x_px=float(cx_f),
                centroid_y_px=float(cy_f),
                excluded=excluded,
                exclusion_reason=reason,
                circularity=round(circ, 3),
            )
        )
        grain_id += 1

    return grains, labeled
