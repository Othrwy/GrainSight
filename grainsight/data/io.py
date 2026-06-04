from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .models import (
    AnalysisLimits,
    BatchTestResult,
    CalibrationData,
    CupMask,
    CycleResult,
    GrainResult,
    ImageSession,
    TestConfig,
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _session_to_dict(session: ImageSession) -> dict:
    d = {
        "image_path": session.image_path,
        "calibration": asdict(session.calibration) if session.calibration else None,
        "limits": asdict(session.limits),
        "grains": [asdict(g) for g in session.grains],
        "analysed": session.analysed,
        "cup_mask": asdict(session.cup_mask) if session.cup_mask else None,
        "grain_density_g_cm3": session.grain_density_g_cm3,
    }
    return d


def _dict_to_session(d: dict) -> ImageSession:
    cal = None
    if d.get("calibration"):
        c = d["calibration"]
        # Tuple fields stored as lists in JSON — convert back
        for key in ("p1", "p2"):
            if c.get(key) is not None:
                c[key] = tuple(c[key])
        if c.get("roi") is not None:
            c["roi"] = tuple(c["roi"])
        cal = CalibrationData(**c)
    limits = AnalysisLimits(**d["limits"])
    grains = [GrainResult(**g) for g in d.get("grains", [])]
    cup_mask = CupMask(**d["cup_mask"]) if d.get("cup_mask") else None
    return ImageSession(
        image_path=d["image_path"],
        calibration=cal,
        limits=limits,
        grains=grains,
        analysed=d.get("analysed", False),
        cup_mask=cup_mask,
        grain_density_g_cm3=d.get("grain_density_g_cm3", 1.5),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_session(session: ImageSession, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_session_to_dict(session), f, indent=2)


def load_session(path: str | Path) -> ImageSession:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        # Multi-session file — return first entry
        return _dict_to_session(d[0])
    return _dict_to_session(d)


def save_multi_session(sessions: List[ImageSession], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([_session_to_dict(s) for s in sessions], f, indent=2)


def load_multi_session(path: str | Path) -> List[ImageSession]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, list):
        return [_dict_to_session(item) for item in d]
    return [_dict_to_session(d)]


def export_csv(session: ImageSession, path: str | Path) -> None:
    if not session.grains:
        return
    fieldnames = [
        "id",
        "major_mm",
        "minor_mm",
        "avg_diameter_mm",
        "volume_mm3",
        "mass_mg",
        "sphericalness",
        "orientation_deg",
        "centroid_x_px",
        "centroid_y_px",
        "excluded",
        "exclusion_reason",
    ]
    density = session.grain_density_g_cm3
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for g in session.grains:
            writer.writerow(
                {
                    "id": g.id,
                    "major_mm": round(g.major_mm, 4),
                    "minor_mm": round(g.minor_mm, 4),
                    "avg_diameter_mm": round(g.avg_diameter_mm, 4),
                    "volume_mm3": round(g.volume_mm3, 6),
                    "mass_mg": round(g.mass_mg(density), 6),
                    "sphericalness": round(g.sphericalness, 4),
                    "orientation_deg": round(g.orientation_deg, 2),
                    "centroid_x_px": round(g.centroid_x_px, 1),
                    "centroid_y_px": round(g.centroid_y_px, 1),
                    "excluded": g.excluded,
                    "exclusion_reason": g.exclusion_reason,
                }
            )


# ---------------------------------------------------------------------------
# Batch test result persistence
# ---------------------------------------------------------------------------

def save_batch_result(result: BatchTestResult, path: str | Path) -> None:
    """Save a BatchTestResult to a JSON file."""
    d = {
        "test_config": asdict(result.test_config),
        "cycles": [asdict(c) for c in result.cycles],
        "limits": asdict(result.limits),
        "created": result.created,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)


def load_batch_result(path: str | Path) -> BatchTestResult:
    """Load a BatchTestResult from a JSON file."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    cfg = TestConfig(**d["test_config"])
    cycles = [CycleResult(**c) for c in d.get("cycles", [])]
    limits = AnalysisLimits(**d["limits"])
    return BatchTestResult(
        test_config=cfg,
        cycles=cycles,
        limits=limits,
        created=d.get("created", ""),
    )
