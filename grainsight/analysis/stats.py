from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy import stats as _scipy_stats

from ..data.models import GrainResult

# Candidate distributions for automatic fitting
_CANDIDATE_DISTRIBUTIONS = [
    ("normal",    "Normal",     _scipy_stats.norm),
    ("lognormal", "Log-normal", _scipy_stats.lognorm),
    ("gamma",     "Gamma",      _scipy_stats.gamma),
    ("weibull",   "Weibull",    _scipy_stats.weibull_min),
]

# Metric key → human label
METRIC_LABELS: Dict[str, str] = {
    "avg_size": "Average Diameter (mm)",
    "d_max": "Major Diameter D_max (mm)",
    "d_min": "Minor Diameter D_min (mm)",
    "sphericalness": "Sphericalness (D_min / D_max)",
    "volume_mm3": "Volume (mm³)",
    "mass_mg": "Mass (mg)",
}


def get_values(grains: List[GrainResult], metric: str, density: float = 1.5) -> List[float]:
    """Extract metric values from the *included* grains only."""
    included = [g for g in grains if not g.excluded]
    extractors = {
        "avg_size": lambda g: g.avg_diameter_mm,
        "d_max": lambda g: g.major_mm,
        "d_min": lambda g: g.minor_mm,
        "sphericalness": lambda g: g.sphericalness,
        "volume_mm3": lambda g: g.volume_mm3,
        "mass_mg": lambda g: g.mass_mg(density),
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


def fit_lognormal(values: List[float]):
    """Closed-form MLE log-normal fit. Returns (mu_ln, sigma_ln) in log-space.
    Returns (0.0, 0.0) on empty or non-positive input."""
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0):
        return 0.0, 0.0
    log_arr = np.log(arr)
    mu_ln = float(log_arr.mean())
    sigma_ln = float(log_arr.std(ddof=1)) if len(arr) >= 2 else 0.0
    return mu_ln, sigma_ln


def fit_distributions(values: List[float]) -> List[Dict]:
    """Fit candidate distributions via MLE and rank by KS test p-value (best first).

    Returns a list of dicts with keys: key, label, params, ks_stat, p_value.
    Returns [] if fewer than 5 values (not enough data for meaningful fitting).
    """
    if len(values) < 5:
        return []
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0):
        # Filter to positive values only (needed for log-normal/gamma/weibull)
        arr = arr[arr > 0]
        if len(arr) < 5:
            return []
    results = []
    for key, label, dist in _CANDIDATE_DISTRIBUTIONS:
        try:
            # Fix location=0 for distributions defined on (0,∞) to avoid spurious shifts
            params = dist.fit(arr, floc=0) if key != "normal" else dist.fit(arr)
            ks_stat, p_value = _scipy_stats.kstest(arr, dist.cdf, args=params)
            results.append({
                "key": key,
                "label": label,
                "params": params,
                "ks_stat": float(ks_stat),
                "p_value": float(p_value),
            })
        except Exception:
            pass
    results.sort(key=lambda r: r["p_value"], reverse=True)
    return results


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
