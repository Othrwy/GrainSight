from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..data.models import GrainResult

# Metric key → human label
METRIC_LABELS: Dict[str, str] = {
    "avg_size": "Average Diameter (mm)",
    "d_max": "Major Diameter D_max (mm)",
    "d_min": "Minor Diameter D_min (mm)",
    "sphericalness": "Sphericalness (D_min / D_max)",
}


def get_values(grains: List[GrainResult], metric: str) -> List[float]:
    """Extract metric values from the *included* grains only."""
    included = [g for g in grains if not g.excluded]
    extractors = {
        "avg_size": lambda g: g.avg_diameter_mm,
        "d_max": lambda g: g.major_mm,
        "d_min": lambda g: g.minor_mm,
        "sphericalness": lambda g: g.sphericalness,
    }
    if metric not in extractors:
        raise ValueError(f"Unknown metric '{metric}'. "
                         f"Valid options: {list(extractors)}")
    return [extractors[metric](g) for g in included]


def fit_normal(values: List[float]):
    """Return (mean, std) using sample statistics."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def sd_percentages(values: List[float]) -> Dict:
    """Return dict with mean, std, and fraction within ±1/2/3 σ."""
    mean, std = fit_normal(values)
    result: Dict = {"mean": mean, "std": std, "n": len(values)}
    if std == 0 or len(values) < 2:
        result.update(pct_1sd=100.0, pct_2sd=100.0, pct_3sd=100.0)
        return result
    arr = np.asarray(values, dtype=float)
    for n in (1, 2, 3):
        within = float(np.sum(np.abs(arr - mean) <= n * std) / len(arr) * 100)
        result[f"pct_{n}sd"] = within
    return result
