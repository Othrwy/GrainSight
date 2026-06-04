from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks


def detect_ruler_edges(
    image_gray: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
) -> Tuple[float, np.ndarray, str, float, float]:
    """Detect two dominant parallel edges inside a user-drawn ROI.

    Parameters
    ----------
    image_gray : H×W uint8 greyscale array
    roi_rect   : (x, y, w, h) in image pixel coordinates

    Returns
    -------
    pixel_distance : perpendicular distance between detected edges (px)
    debug_bgr      : BGR image of ROI with detected edges drawn
    axis           : 'row' (horizontal edges) or 'col' (vertical edges)
    pos1, pos2     : edge positions within the ROI (px from top/left edge)

    Raises
    ------
    ValueError  if fewer than two edges can be resolved
    """
    x, y, w, h = (int(v) for v in roi_rect)

    # Clamp to image bounds
    img_h, img_w = image_gray.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))

    roi = image_gray[y : y + h, x : x + w].copy()

    # Enhance contrast inside ROI
    roi_eq = cv2.equalizeHist(roi)
    blurred = cv2.GaussianBlur(roi_eq, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    # Project edges onto the shorter axis:
    #   landscape ROI → sum rows → detect horizontal (row) edges
    #   portrait ROI  → sum cols → detect vertical (col) edges
    if w >= h:
        profile = edges.sum(axis=1).astype(float)
        axis = "row"
    else:
        profile = edges.sum(axis=0).astype(float)
        axis = "col"

    n = len(profile)
    if n < 8:
        raise ValueError("ROI is too small to detect edges reliably.")

    # Smooth the profile
    k = max(3, n // 20) | 1  # ensure odd
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")

    peaks, _ = find_peaks(
        smooth,
        height=smooth.max() * 0.15,
        distance=n // 8,
    )

    if len(peaks) < 2:
        raise ValueError(
            f"Only {len(peaks)} edge peak(s) found. "
            "Try drawing a tighter box around both edges of the reference."
        )

    # From the top-N strongest peaks, pick the pair with the greatest separation
    heights = smooth[peaks]
    top_n = peaks[np.argsort(-heights)[: min(8, len(peaks))]]
    best_pair: Tuple[int, int] = (int(top_n[0]), int(top_n[1]))
    best_sep = 0
    for i in range(len(top_n)):
        for j in range(i + 1, len(top_n)):
            sep = abs(int(top_n[i]) - int(top_n[j]))
            if sep > best_sep:
                best_sep = sep
                best_pair = (int(top_n[i]), int(top_n[j]))

    pos1, pos2 = sorted(best_pair)
    pixel_distance = float(pos2 - pos1)

    # Build a debug image with the detected edges drawn on
    debug = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    line_colour = (0xD0, 0xE0, 0x50)  # faded yellow-green
    if axis == "row":
        cv2.line(debug, (0, pos1), (w - 1, pos1), line_colour, 1)
        cv2.line(debug, (0, pos2), (w - 1, pos2), line_colour, 1)
    else:
        cv2.line(debug, (pos1, 0), (pos1, h - 1), line_colour, 1)
        cv2.line(debug, (pos2, 0), (pos2, h - 1), line_colour, 1)

    return pixel_distance, debug, axis, float(pos1), float(pos2)


def compute_scale_from_edge_detect(
    pixel_distance: float, physical_mm: float
) -> float:
    """Return px_per_mm."""
    if physical_mm <= 0:
        raise ValueError("Physical distance must be > 0 mm.")
    if pixel_distance <= 0:
        raise ValueError("Pixel distance must be > 0.")
    return pixel_distance / physical_mm


def compute_scale_two_point(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    physical_mm: float,
) -> float:
    """Return px_per_mm from two clicked points."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    pixel_distance = float(np.sqrt(dx * dx + dy * dy))
    return compute_scale_from_edge_detect(pixel_distance, physical_mm)
