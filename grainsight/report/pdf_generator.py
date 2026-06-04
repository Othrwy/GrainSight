from __future__ import annotations

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from ..analysis.stats import METRIC_LABELS, get_values, sd_percentages
from ..data.models import ImageSession, ReportMetadata
from .plot_builder import MPL_RC, make_boxplot, make_histogram

_GOLD = "#c9a84c"
_BLUE = "#4a7a9c"
_RED = "#b84c3c"
_BG = "#1a1a1a"
_TEXT = "#e8e5d8"
_MUTED = "#9e9b8e"


def _title_page(meta: ReportMetadata, sessions: List[ImageSession]) -> Figure:
    """Render a title / metadata page as a matplotlib figure."""
    fig = Figure(figsize=(8.27, 11.69), facecolor=_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(_BG)
    ax.axis("off")

    def t(x, y, text, **kw):
        ax.text(x, y, text, transform=ax.transAxes, **kw)

    # Decorative header bar
    ax.axhspan(0.90, 1.0, xmin=0, xmax=1, color=_GOLD, alpha=0.12)
    ax.axhline(0.90, color=_GOLD, linewidth=1.0, alpha=0.5)

    t(0.08, 0.94, "GrainSight", color=_GOLD, fontsize=11, fontweight="bold",
      va="center", alpha=0.7)
    t(0.08, 0.86, meta.title, color=_TEXT, fontsize=18, fontweight="bold", va="top")
    ax.axhline(0.83, color=_GOLD, linewidth=0.6, alpha=0.35, xmin=0.08, xmax=0.92)

    # Metadata block
    fields = [
        ("Date", meta.date),
        ("User", meta.user or "—"),
        ("Sample ID", meta.sample_id or "—"),
        ("Test Details", meta.test_details or "—"),
    ]
    y0 = 0.78
    for label, value in fields:
        t(0.08, y0, f"{label}:", color=_MUTED, fontsize=9, va="top")
        t(0.25, y0, value, color=_TEXT, fontsize=9, va="top")
        y0 -= 0.040

    # Summary
    t(0.08, y0 - 0.01, "Summary", color=_GOLD, fontsize=10, va="top")
    y0 -= 0.045
    summary = meta.summary or "\n".join(s.auto_summary() for s in sessions)
    # word-wrap at ~90 chars
    words = summary.split()
    lines, line = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > 90:
            lines.append(" ".join(line[:-1]))
            line = [w]
    if line:
        lines.append(" ".join(line))
    for ln in lines:
        t(0.08, y0, ln, color=_TEXT, fontsize=9, va="top")
        y0 -= 0.030

    # Per-session summary table
    y0 -= 0.02
    ax.axhline(y0 + 0.015, color=_MUTED, linewidth=0.4, alpha=0.4, xmin=0.08, xmax=0.92)
    t(0.08, y0, "Image", color=_GOLD, fontsize=8.5, va="top")
    t(0.45, y0, "Grains (incl / excl)", color=_GOLD, fontsize=8.5, va="top")
    t(0.70, y0, "Mean Avg Diam (mm)", color=_GOLD, fontsize=8.5, va="top")
    y0 -= 0.032
    for s in sessions:
        inc = len(s.included_grains)
        exc = len(s.excluded_grains)
        vals = get_values(s.grains, "avg_size")
        mean_str = f"{np.mean(vals):.3f}" if vals else "—"
        name = os.path.splitext(os.path.basename(s.image_path))[0]
        t(0.08, y0, name[:45], color=_TEXT, fontsize=8, va="top")
        t(0.45, y0, f"{inc} / {exc}", color=_TEXT, fontsize=8, va="top")
        t(0.70, y0, mean_str, color=_TEXT, fontsize=8, va="top")
        y0 -= 0.028

    # Footer
    ax.axhline(0.04, color=_GOLD, linewidth=0.4, alpha=0.4)
    t(0.08, 0.025, "GrainSight v0.1.0", color=_MUTED, fontsize=7)
    t(0.92, 0.025, meta.date, color=_MUTED, fontsize=7, ha="right")

    return fig


def generate_pdf(
    sessions: List[ImageSession],
    plot_configs: List[Dict[str, Any]],
    meta: ReportMetadata,
    output_path: str,
) -> None:
    """Write a multi-page PDF report.

    Parameters
    ----------
    plot_configs : list of dicts, each with keys:
        type         – 'histogram' | 'boxplot'
        metric       – one of stats.METRIC_LABELS keys
        scope        – 'combined' | 'per_image'
        bins         – int (histograms only)
        normal_fit   – bool
        show_1sd     – bool
        show_2sd     – bool
        show_3sd     – bool
    """
    with PdfPages(output_path) as pdf:
        # Title page
        with plt.rc_context(MPL_RC):
            title_fig = _title_page(meta, sessions)
        pdf.savefig(title_fig, bbox_inches="tight", facecolor=_BG)
        plt.close(title_fig)

        # Plot pages
        for cfg in plot_configs:
            ptype = cfg.get("type", "histogram")
            metric = cfg.get("metric", "avg_size")
            scope = cfg.get("scope", "combined")
            per_image = scope == "per_image"
            bins = int(cfg.get("bins", 30))
            show_fit = bool(cfg.get("normal_fit", False))
            show_sd = (
                bool(cfg.get("show_1sd", False)),
                bool(cfg.get("show_2sd", False)),
                bool(cfg.get("show_3sd", False)),
            )

            if ptype == "histogram":
                fig = make_histogram(
                    sessions, metric, bins=bins,
                    show_normal_fit=show_fit, show_sd=show_sd,
                    per_image=per_image,
                )
            elif ptype == "boxplot":
                fig = make_boxplot(sessions, metric, per_image=per_image)
            else:
                continue

            pdf.savefig(fig, bbox_inches="tight", facecolor=_BG)
            plt.close(fig)
