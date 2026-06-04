from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from ..report.plot_builder import make_boxplot, make_histogram
from .theme import ACCENT_GOLD, TEXT_SECONDARY


class _NoWheelComboBox(QComboBox):
    """ComboBox that ignores scroll-wheel events to prevent accidental value changes."""
    def wheelEvent(self, event):
        event.ignore()


# Metric choices shown in the type combo
_HIST_TYPES = {
    "Histogram — Average Diameter":  ("histogram", "avg_size"),
    "Histogram — D_max":             ("histogram", "d_max"),
    "Histogram — D_min":             ("histogram", "d_min"),
    "Histogram — Sphericalness":     ("histogram", "sphericalness"),
    "Histogram — Volume":            ("histogram", "volume_mm3"),
    "Histogram — Mass":              ("histogram", "mass_mg"),
    "Box & Whisker — Average Diameter": ("boxplot", "avg_size"),
    "Box & Whisker — D_max":            ("boxplot", "d_max"),
    "Box & Whisker — D_min":            ("boxplot", "d_min"),
    "Box & Whisker — Sphericalness":    ("boxplot", "sphericalness"),
    "Box & Whisker — Volume":           ("boxplot", "volume_mm3"),
    "Box & Whisker — Mass":             ("boxplot", "mass_mg"),
}

# Static fallback fit items when no data is available yet
_STATIC_FIT_ITEMS = [
    ("none",      "No fit"),
    ("normal",    "Normal"),
    ("lognormal", "Log-normal"),
    ("gamma",     "Gamma"),
    ("weibull",   "Weibull"),
]


class PlotRowWidget(QFrame):
    """Single configurable plot row in the report builder."""

    remove_requested = pyqtSignal(object)  # emits self

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("plot_row_frame")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        self._sessions: List = []  # cached sessions for GoF re-calculation

        layout = QGridLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)

        # Row 0: remove btn + type combo + scope combo
        self._remove_btn = QPushButton("✕")
        self._remove_btn.setObjectName("remove_btn")
        self._remove_btn.setFixedWidth(26)
        self._remove_btn.setToolTip("Remove this plot")
        self._remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        layout.addWidget(self._remove_btn, 0, 0)

        self._type_combo = _NoWheelComboBox()
        self._type_combo.addItems(list(_HIST_TYPES.keys()))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self._type_combo, 0, 1)

        self._scope_combo = _NoWheelComboBox()
        self._scope_combo.addItems(["Combined (all images)", "Per image"])
        layout.addWidget(self._scope_combo, 0, 2)

        # Row 1: histogram options
        opts_widget = QWidget()
        opts_layout = QHBoxLayout(opts_widget)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(10)

        fit_lbl = QLabel("Fit:")
        fit_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")
        self._fit_combo = _NoWheelComboBox()
        self._fit_combo.setMinimumWidth(160)
        self._fit_combo.setToolTip(
            "Distribution fit overlay.\n"
            "p-value shown when data is available (higher p = better fit).\n"
            "★ marks the best-fitting distribution."
        )
        self._fit_combo.currentIndexChanged.connect(self._on_fit_changed)
        opts_layout.addWidget(fit_lbl)
        opts_layout.addWidget(self._fit_combo)

        self._mean_cb = QCheckBox("Mean")
        self._mean_cb.setToolTip("Show a vertical line at the mean value")
        opts_layout.addWidget(self._mean_cb)

        self._sd1_cb = QCheckBox("±1σ")
        self._sd2_cb = QCheckBox("±2σ")
        self._sd3_cb = QCheckBox("±3σ")
        for cb in (self._sd1_cb, self._sd2_cb, self._sd3_cb):
            opts_layout.addWidget(cb)

        bins_lbl = QLabel("Bins:")
        bins_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(5, 200)
        self._bins_spin.setValue(30)
        self._bins_spin.setFixedWidth(60)
        opts_layout.addWidget(bins_lbl)
        opts_layout.addWidget(self._bins_spin)
        opts_layout.addStretch()

        layout.addWidget(opts_widget, 1, 1, 1, 2)

        # Populate fit combo with static items initially
        self._populate_fit_static()
        self._on_type_changed()  # set initial enabled state

    # ------------------------------------------------------------------
    # Fit combo population
    # ------------------------------------------------------------------

    def _populate_fit_static(self) -> None:
        """Fill fit combo with plain names (no GoF data available yet)."""
        self._fit_combo.blockSignals(True)
        self._fit_combo.clear()
        for key, label in _STATIC_FIT_ITEMS:
            self._fit_combo.addItem(label, userData=key)
        self._fit_combo.blockSignals(False)

    def update_fit_options(self, sessions) -> None:
        """Recompute GoF for the current metric and rebuild the fit combo.

        Called by the report panel whenever sessions change or the panel expands.
        """
        self._sessions = sessions or []
        ptype, metric = _HIST_TYPES[self._type_combo.currentText()]
        if ptype != "histogram" or not self._sessions:
            return

        from ..analysis.stats import fit_distributions, get_values

        all_grains = [g for s in self._sessions for g in s.included_grains]
        density = self._sessions[0].grain_density_g_cm3
        try:
            vals = get_values(all_grains, metric, density=density)
            results = fit_distributions(vals)
        except Exception:
            results = []

        current_key = self._fit_combo.currentData() or "none"

        self._fit_combo.blockSignals(True)
        self._fit_combo.clear()
        self._fit_combo.addItem("No fit", userData="none")

        if results:
            best_key = results[0]["key"]
            for r in results:
                star = " ★" if r["key"] == best_key else ""
                label = f"{r['label']}   p={r['p_value']:.2f}{star}"
                self._fit_combo.addItem(label, userData=r["key"])
        else:
            # Too few data points — fall back to plain labels
            for key, label in _STATIC_FIT_ITEMS[1:]:
                self._fit_combo.addItem(label, userData=key)

        # Restore previous selection if still present, otherwise keep "No fit"
        for i in range(self._fit_combo.count()):
            if self._fit_combo.itemData(i) == current_key:
                self._fit_combo.setCurrentIndex(i)
                break

        self._fit_combo.blockSignals(False)
        self._on_fit_changed()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_type_changed(self) -> None:
        ptype, _ = _HIST_TYPES[self._type_combo.currentText()]
        is_hist = ptype == "histogram"
        for w in (self._fit_combo, self._mean_cb, self._sd1_cb, self._sd2_cb,
                  self._sd3_cb, self._bins_spin):
            w.setEnabled(is_hist)
            if not is_hist and isinstance(w, QCheckBox):
                w.setChecked(False)
        # Re-run GoF for the new metric if we already have sessions
        if self._sessions:
            self.update_fit_options(self._sessions)

    def _on_fit_changed(self) -> None:
        # SD band checkboxes are available for any histogram type regardless of
        # the chosen distribution — they show mean ± n*std of the raw data.
        # No action needed here; enabled state is managed by _on_type_changed.
        pass

    # ------------------------------------------------------------------
    # Config and preview
    # ------------------------------------------------------------------

    def config(self) -> Dict[str, Any]:
        """Return the current plot configuration dict."""
        ptype, metric = _HIST_TYPES[self._type_combo.currentText()]
        scope = (
            "per_image"
            if self._scope_combo.currentIndex() == 1
            else "combined"
        )
        fit_key = self._fit_combo.currentData() or "none"
        return {
            "type": ptype,
            "metric": metric,
            "scope": scope,
            "bins": self._bins_spin.value(),
            "fit_distribution": fit_key,
            "show_mean": self._mean_cb.isChecked(),
            "show_1sd": self._sd1_cb.isChecked(),
            "show_2sd": self._sd2_cb.isChecked(),
            "show_3sd": self._sd3_cb.isChecked(),
        }

    def preview(self, sessions) -> Optional[FigureCanvas]:
        """Build a live preview canvas for this plot."""
        if not sessions:
            return None
        cfg = self.config()
        ptype = cfg["type"]
        try:
            if ptype == "histogram":
                fig = make_histogram(
                    sessions, cfg["metric"],
                    bins=cfg["bins"],
                    fit_distribution=cfg["fit_distribution"],
                    show_mean=cfg["show_mean"],
                    show_sd=(cfg["show_1sd"], cfg["show_2sd"], cfg["show_3sd"]),
                    per_image=cfg["scope"] == "per_image",
                )
            else:
                fig = make_boxplot(
                    sessions, cfg["metric"],
                    per_image=cfg["scope"] == "per_image",
                )
        except Exception:
            return None
        canvas = FigureCanvas(fig)
        return canvas
