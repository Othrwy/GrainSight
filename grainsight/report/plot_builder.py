from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import norm

from ..analysis.stats import METRIC_LABELS, fit_lognormal, fit_normal, get_values, sd_percentages  # noqa: F401
from ..data.models import GrainResult, ImageSession

# ---------------------------------------------------------------------------
# Shared matplotlib style (dark retro theme)
# ---------------------------------------------------------------------------
MPL_RC: Dict = {
    "figure.facecolor": "#1a1a1a",
    "axes.facecolor": "#1e1e1e",
    "axes.edgecolor": "#3a3a3a",
    "axes.labelcolor": "#e8e5d8",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#9e9b8e",
    "ytick.color": "#9e9b8e",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": "#e8e5d8",
    "grid.color": "#333333",
    "grid.alpha": 0.5,
    "lines.linewidth": 1.5,
    "legend.facecolor": "#252525",
    "legend.edgecolor": "#3a3a3a",
    "legend.fontsize": 8,
}

# Accent colours
_GOLD = "#c9a84c"
_RED = "#b84c3c"
_BLUE = "#4a7a9c"
_DKGOLD = "#8a6e2a"
_SD_COLOURS = ["#b84c3c", "#4a7a9c", "#5a9c6a"]  # 1σ, 2σ, 3σ


def _apply_rc(fig: Figure) -> None:
    """Apply retro RC params to an existing figure."""
    with plt.rc_context(MPL_RC):
        pass  # RC applied when creating the figure


def make_histogram(
    sessions: List[ImageSession],
    metric: str,
    bins: int = 30,
    fit_distribution: str = "none",
    show_mean: bool = False,
    show_sd: Tuple[bool, bool, bool] = (False, False, False),
    per_image: bool = False,
    # Legacy keyword args kept for backward compatibility
    show_normal_fit: bool = False,
    show_lognormal_fit: bool = False,
) -> Figure:
    """Return a matplotlib Figure containing a histogram."""
    # Resolve legacy boolean params when fit_distribution not explicitly set
    if fit_distribution == "none":
        if show_normal_fit:
            fit_distribution = "normal"
        elif show_lognormal_fit:
            fit_distribution = "lognormal"

    with plt.rc_context(MPL_RC):
        fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.set_facecolor(MPL_RC["axes.facecolor"])
    ax.set_xlabel(METRIC_LABELS.get(metric, metric))
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)

    if per_image:
        colours = [_GOLD, _BLUE, _RED, "#5a9c6a", "#9a6a9c"]
        for idx, session in enumerate(sessions):
            density = session.grain_density_g_cm3
            vals = get_values(session.grains, metric, density=density)
            if not vals:
                continue
            colour = colours[idx % len(colours)]
            label = _short_name(session.image_path)
            ax.hist(vals, bins=bins, color=colour, alpha=0.65, label=label,
                    edgecolor="#1a1a1a", linewidth=0.4)
        if show_mean:
            all_grains_c: List[GrainResult] = []
            for s in sessions:
                all_grains_c.extend(s.grains)
            density_c = sessions[0].grain_density_g_cm3 if sessions else 1.5
            all_vals_c = get_values(all_grains_c, metric, density=density_c)
            if all_vals_c:
                mean_c = float(np.mean(all_vals_c))
                ax.axvline(mean_c, color=_GOLD, linewidth=1.4, linestyle="--",
                           label=f"Mean = {mean_c:.4f}")
        ax.legend()
    else:
        all_grains: List[GrainResult] = []
        for s in sessions:
            all_grains.extend(s.grains)
        density = sessions[0].grain_density_g_cm3 if sessions else 1.5
        vals = get_values(all_grains, metric, density=density)
        if not vals:
            ax.set_title("No data")
            return fig
        ax.hist(vals, bins=bins, color=_GOLD, alpha=0.72,
                edgecolor="#1a1a1a", linewidth=0.4)

        if show_mean:
            mean_val = float(np.mean(vals))
            ax.axvline(mean_val, color=_GOLD, linewidth=1.4, linestyle="--",
                       label=f"Mean = {mean_val:.4f}")

        # SD bands and fit curve are now independent — both can appear together
        if any(show_sd):
            _overlay_sd_bands(ax, vals, show_sd)

        if fit_distribution == "normal":
            _overlay_normal(ax, vals)
        elif fit_distribution == "lognormal":
            _overlay_lognormal(ax, vals)
        elif fit_distribution == "gamma":
            _overlay_gamma(ax, vals)
        elif fit_distribution == "weibull":
            _overlay_weibull(ax, vals)

        if show_mean or fit_distribution != "none" or any(show_sd):
            ax.legend(fontsize=8)

    ax.set_title(METRIC_LABELS.get(metric, metric), color="#c9a84c", fontsize=11)
    fig.tight_layout()
    return fig


def make_boxplot(
    sessions: List[ImageSession],
    metric: str,
    per_image: bool = True,
) -> Figure:
    """Return a matplotlib Figure containing a box-and-whisker plot."""
    with plt.rc_context(MPL_RC):
        fig, ax = plt.subplots(figsize=(max(4, len(sessions) * 1.4 + 1.5), 4.5))

    ax.set_facecolor(MPL_RC["axes.facecolor"])
    ax.set_ylabel(METRIC_LABELS.get(metric, metric))
    ax.grid(True, axis="y", alpha=0.3)

    data = []
    labels = []
    for session in sessions:
        density = session.grain_density_g_cm3
        vals = get_values(session.grains, metric, density=density)
        if vals:
            data.append(vals)
            labels.append(_short_name(session.image_path))

    if not data:
        ax.set_title("No data")
        return fig

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color=_GOLD, linewidth=2),
        whiskerprops=dict(color="#9e9b8e"),
        capprops=dict(color="#9e9b8e"),
        flierprops=dict(marker="o", color=_RED, markersize=3, alpha=0.6),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#2a3a4a")
        patch.set_edgecolor(_BLUE)

    ax.set_title(
        f"Box & Whisker — {METRIC_LABELS.get(metric, metric)}",
        color=_GOLD, fontsize=11,
    )
    ax.tick_params(axis="x", labelrotation=15)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _overlay_normal(ax, vals: List[float]) -> None:
    stats_d = sd_percentages(vals)
    mu, std = stats_d["mean"], stats_d["std"]
    if std == 0:
        return
    x = np.linspace(min(vals) - 2 * std, max(vals) + 2 * std, 300)
    counts, edges = np.histogram(vals, bins="auto")
    bin_w = edges[1] - edges[0]
    scale = len(vals) * bin_w
    ax.plot(x, norm.pdf(x, mu, std) * scale, color=_BLUE, linewidth=1.6,
            label=f"Normal fit  μ={mu:.3f}  σ={std:.3f}")


def _overlay_sd_bands(ax, vals: List[float], show_sd: Tuple[bool, bool, bool]) -> None:
    stats_d = sd_percentages(vals)
    mu, std = stats_d["mean"], stats_d["std"]
    if std == 0:
        return
    for n, visible in enumerate(show_sd, start=1):
        if visible:
            col = _SD_COLOURS[n - 1]
            ax.axvline(mu + n * std, color=col, linewidth=0.9, linestyle="--",
                       label=f"+{n}σ  ({stats_d.get(f'pct_{n}sd', 0):.1f}% within)")
            ax.axvline(mu - n * std, color=col, linewidth=0.9, linestyle="--")


def _overlay_lognormal(ax, vals: List[float]) -> None:
    from scipy.stats import lognorm
    mu_ln, sigma_ln = fit_lognormal(vals)
    if sigma_ln == 0:
        return
    counts, edges = np.histogram(vals, bins="auto")
    bin_w = edges[1] - edges[0]
    scale = len(vals) * bin_w
    x = np.linspace(min(vals) * 0.5, max(vals) * 1.5, 400)
    y = lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu_ln)) * scale
    ax.plot(x, y, color=_RED, linewidth=1.6,
            label=f"Log-normal fit  μₗₙ={mu_ln:.3f}  σₗₙ={sigma_ln:.3f}")
    ax.legend(fontsize=8)


def _overlay_gamma(ax, vals: List[float]) -> None:
    from scipy.stats import gamma
    try:
        a, loc, scale = gamma.fit(vals, floc=0)
    except Exception:
        return
    if scale == 0:
        return
    counts, edges = np.histogram(vals, bins="auto")
    bin_w = edges[1] - edges[0]
    scale_factor = len(vals) * bin_w
    x = np.linspace(min(vals) * 0.5, max(vals) * 1.5, 400)
    y = gamma.pdf(x, a, loc=loc, scale=scale) * scale_factor
    ax.plot(x, y, color=_RED, linewidth=1.6,
            label=f"Gamma fit  α={a:.3f}  scale={scale:.4f}")
    ax.legend(fontsize=8)


def _overlay_weibull(ax, vals: List[float]) -> None:
    from scipy.stats import weibull_min
    try:
        c, loc, scale = weibull_min.fit(vals, floc=0)
    except Exception:
        return
    if scale == 0:
        return
    counts, edges = np.histogram(vals, bins="auto")
    bin_w = edges[1] - edges[0]
    scale_factor = len(vals) * bin_w
    x = np.linspace(min(vals) * 0.5, max(vals) * 1.5, 400)
    y = weibull_min.pdf(x, c, loc=loc, scale=scale) * scale_factor
    ax.plot(x, y, color=_RED, linewidth=1.6,
            label=f"Weibull fit  k={c:.3f}  λ={scale:.4f}")
    ax.legend(fontsize=8)


def _short_name(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0]
