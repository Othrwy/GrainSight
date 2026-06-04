from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GrainResult:
    id: int
    major_px: float
    minor_px: float
    major_mm: float
    minor_mm: float
    sphericalness: float        # minor / major  (1.0 = perfect sphere)
    orientation_deg: float      # Qt rotation degrees (CW from x-axis)
    centroid_x_px: float
    centroid_y_px: float
    excluded: bool = False
    exclusion_reason: str = ""

    @property
    def avg_diameter_mm(self) -> float:
        return (self.major_mm + self.minor_mm) / 2.0


@dataclass
class CalibrationData:
    mode: str                          # 'two_point' | 'edge_detect'
    px_per_mm: float
    reference_mm: float
    pixel_distance: float = 0.0
    # two_point fields
    p1: Optional[Tuple[float, float]] = None
    p2: Optional[Tuple[float, float]] = None
    # edge_detect fields
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    edge_axis: Optional[str] = None                  # 'row' | 'col'
    edge_pos1: Optional[float] = None
    edge_pos2: Optional[float] = None


@dataclass
class AnalysisLimits:
    min_mm: float = 0.2
    max_mm: float = 1.0


@dataclass
class ReportMetadata:
    title: str = "Grain Size Analysis Report"
    summary: str = ""
    date: str = field(default_factory=lambda: datetime.date.today().isoformat())
    user: str = ""
    sample_id: str = ""
    test_details: str = ""


@dataclass
class ImageSession:
    image_path: str
    calibration: Optional[CalibrationData] = None
    limits: AnalysisLimits = field(default_factory=AnalysisLimits)
    grains: List[GrainResult] = field(default_factory=list)
    analysed: bool = False

    @property
    def included_grains(self) -> List[GrainResult]:
        return [g for g in self.grains if not g.excluded]

    @property
    def excluded_grains(self) -> List[GrainResult]:
        return [g for g in self.grains if g.excluded]

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None and self.calibration.px_per_mm > 0

    def refilter(self) -> None:
        """Re-apply size limits to existing grains without re-running detection."""
        for g in self.grains:
            if g.major_mm > self.limits.max_mm or g.major_mm < self.limits.min_mm:
                g.excluded = True
                g.exclusion_reason = "outside size limits"
            elif g.excluded and g.exclusion_reason == "outside size limits":
                # Previously excluded by limits only — re-include
                g.excluded = False
                g.exclusion_reason = ""

    def auto_summary(self) -> str:
        n = len(self.included_grains)
        if n == 0:
            return "No grains detected."
        avg = sum(g.avg_diameter_mm for g in self.included_grains) / n
        sph = sum(g.sphericalness for g in self.included_grains) / n
        import os
        fname = os.path.basename(self.image_path)
        return (
            f"Analysis of image '{fname}'. "
            f"{n} grains detected within size limits "
            f"({self.limits.min_mm}–{self.limits.max_mm} mm). "
            f"Mean average diameter: {avg:.3f} mm. "
            f"Mean sphericalness: {sph:.2f}."
        )
