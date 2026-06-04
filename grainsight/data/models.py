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
    circularity: float = 0.0   # 4π·area / perimeter² (1.0 = perfect circle)

    @property
    def avg_diameter_mm(self) -> float:
        return (self.major_mm + self.minor_mm) / 2.0

    @property
    def volume_mm3(self) -> float:
        # Oblate spheroid assumption: depth axis = minor diameter.
        # V = (pi/6) * D_major * D_minor^2
        return (math.pi / 6.0) * self.major_mm * self.minor_mm ** 2

    def mass_mg(self, density_g_cm3: float) -> float:
        return self.volume_mm3 * density_g_cm3


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
class CupMask:
    cx_px: float
    cy_px: float
    radius_px: float
    source: str = "auto"  # 'auto' | 'manual_3pt'


@dataclass
class AnalysisLimits:
    min_mm: float = 0.5
    max_mm: float = 1.1
    min_circularity: float = 0.65   # 0 = disabled; 1 = perfect circle only


@dataclass
class ReportMetadata:
    title: str = "Grain Size Analysis Report"
    summary: str = ""
    date: str = field(default_factory=lambda: datetime.date.today().isoformat())
    user: str = ""
    sample_id: str = ""
    test_details: str = ""


@dataclass
class TestConfig:
    disc: str = ""   # "1", "2", "3"
    hopper: str = "" # "A", "B", "C"


@dataclass
class CycleResult:
    cycle: int
    image_path: str
    grain_count: int
    px_per_mm: float
    sticker_detected: bool
    cup_detected: bool


@dataclass
class BatchTestResult:
    test_config: TestConfig
    cycles: List[CycleResult]
    limits: AnalysisLimits
    created: str = field(default_factory=lambda: datetime.date.today().isoformat())


@dataclass
class ImageSession:
    image_path: str
    calibration: Optional[CalibrationData] = None
    limits: AnalysisLimits = field(default_factory=AnalysisLimits)
    grains: List[GrainResult] = field(default_factory=list)
    analysed: bool = False
    cup_mask: Optional[CupMask] = None
    grain_density_g_cm3: float = 1.5

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
        """Re-apply size and circularity limits to existing grains without re-running detection."""
        for g in self.grains:
            if g.major_mm > self.limits.max_mm or g.major_mm < self.limits.min_mm:
                g.excluded = True
                g.exclusion_reason = "outside size limits"
            elif (
                self.limits.min_circularity > 0
                and g.circularity > 0
                and g.circularity < self.limits.min_circularity
            ):
                g.excluded = True
                g.exclusion_reason = "not circular"
            elif g.excluded and g.exclusion_reason in ("outside size limits", "not circular"):
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
        total_mass = sum(g.volume_mm3 * self.grain_density_g_cm3 for g in self.included_grains)
        return (
            f"Analysis of image '{fname}'. "
            f"{n} grains detected within size limits "
            f"({self.limits.min_mm}–{self.limits.max_mm} mm). "
            f"Mean average diameter: {avg:.3f} mm. "
            f"Mean sphericalness: {sph:.2f}. "
            f"Estimated total mass: {total_mass:.4f} mg."
        )
