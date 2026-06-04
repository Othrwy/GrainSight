"""Cup / receptacle detection helpers for GrainSight.

Provides three detection utilities:
- detect_blue_sticker  : find the blue Ø10.05 mm calibration sticker by colour
- detect_cup_rim       : find the circular receptacle rim via Hough circles
- circle_from_three_points : compute circumcircle from three user-clicked points
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

# Physical diameter of the blue reference sticker (mm)
STICKER_DIAMETER_MM: float = 10.05


def _detect_sticker_by_hsv(
    image_bgr: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Shared HSV blob detector used by all sticker colour variants.

    Returns (cx_px, cy_px, radius_px) using an area-based radius estimate
    (more accurate than minEnclosingCircle on noisy contours), or None.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best: Optional[Tuple[float, float, float]] = None
    best_score = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        perim = cv2.arcLength(c, True)
        if perim < 1:
            continue
        circularity = 4.0 * math.pi * area / (perim * perim)
        score = area * circularity * circularity
        if score > best_score:
            best_score = score
            r_equiv = math.sqrt(area / math.pi)
            best = (float(cx), float(cy), r_equiv)
    return best


def detect_blue_sticker(
    image_bgr: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Detect the cobalt-blue reference sticker (HSV hue ≈ 95–135).

    Returns (cx_px, cy_px, radius_px) or None.
    px_per_mm = (2 * radius_px) / STICKER_DIAMETER_MM
    """
    lower = np.array([95, 70, 40], dtype=np.uint8)
    upper = np.array([135, 255, 255], dtype=np.uint8)
    return _detect_sticker_by_hsv(image_bgr, lower, upper)


def detect_orange_sticker(
    image_bgr: np.ndarray,
) -> Optional[Tuple[float, float, float]]:
    """Detect a fluorescent-orange reference sticker (HSV hue ≈ 5–25).

    Orange is the recommended sticker colour: single continuous hue band,
    no HSV wrap-around, maximally distinct from grey cup material.

    Returns (cx_px, cy_px, radius_px) or None.
    px_per_mm = (2 * radius_px) / STICKER_DIAMETER_MM
    """
    lower = np.array([5, 140, 80], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    return _detect_sticker_by_hsv(image_bgr, lower, upper)


def detect_cup_rim(
    image_bgr: np.ndarray,
    min_diameter_mm: float = 24.0,
    px_per_mm: float = 0.0,
) -> Optional[Tuple[float, float, float]]:
    """Detect the circular receptacle rim using Hough circle transform.

    The cup occupies roughly 30–90\u202f% of the shorter image dimension.
    The most central candidate circle is returned.

    Parameters
    ----------
    image_bgr       : BGR source image
    min_diameter_mm : minimum accepted rim diameter in mm (default 24 mm).
                      Only applied when px_per_mm > 0.
    px_per_mm       : calibration scale; used to enforce min_diameter_mm.

    Returns
    -------
    (cx_px, cy_px, radius_px) or None if no circle found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    short = min(h, w)

    min_r_frac = int(short * 0.30)
    max_r = int(short * 0.90)

    # Enforce physical minimum diameter if calibration is available
    if px_per_mm > 0:
        min_r_mm = int(min_diameter_mm / 2.0 * px_per_mm)
        min_r = max(min_r_frac, min_r_mm)
    else:
        min_r = min_r_frac

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=short * 0.25,
        param1=60,
        param2=28,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    candidates = np.round(circles[0]).astype(float)
    cx_img, cy_img = w / 2.0, h / 2.0
    best = min(
        candidates,
        key=lambda c: (c[0] - cx_img) ** 2 + (c[1] - cy_img) ** 2,
    )
    return float(best[0]), float(best[1]), float(best[2])


def circle_from_three_points(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
) -> Optional[Tuple[float, float, float]]:
    """Compute the circumcircle of three points.

    Returns
    -------
    (cx, cy, radius) or None if the points are collinear.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    ax, ay = x2 - x1, y2 - y1
    bx, by = x3 - x1, y3 - y1
    D = 2.0 * (ax * by - ay * bx)
    if abs(D) < 1e-8:
        return None  # collinear

    ax2ay2 = ax * ax + ay * ay
    bx2by2 = bx * bx + by * by
    ux = (by * ax2ay2 - ay * bx2by2) / D
    uy = (ax * bx2by2 - bx * ax2ay2) / D

    cx = x1 + ux
    cy = y1 + uy
    r = math.sqrt(ux * ux + uy * uy)
    return (cx, cy, r)
