from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import List

from .models import (
    AnalysisLimits,
    CalibrationData,
    GrainResult,
    ImageSession,
)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _session_to_dict(session: ImageSession) -> dict:
    return {
        "image_path": session.image_path,
        "calibration": asdict(session.calibration) if session.calibration else None,
        "limits": asdict(session.limits),
        "grains": [asdict(g) for g in session.grains],
        "analysed": session.analysed,
    }


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
    return ImageSession(
        image_path=d["image_path"],
        calibration=cal,
        limits=limits,
        grains=grains,
        analysed=d.get("analysed", False),
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
        "sphericalness",
        "orientation_deg",
        "centroid_x_px",
        "centroid_y_px",
        "excluded",
        "exclusion_reason",
    ]
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
                    "sphericalness": round(g.sphericalness, 4),
                    "orientation_deg": round(g.orientation_deg, 2),
                    "centroid_x_px": round(g.centroid_x_px, 1),
                    "centroid_y_px": round(g.centroid_y_px, 1),
                    "excluded": g.excluded,
                    "exclusion_reason": g.exclusion_reason,
                }
            )
