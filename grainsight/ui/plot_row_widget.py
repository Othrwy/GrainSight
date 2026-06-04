from __future__ import annotations

from typing import Any, Dict, Optional

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

# Metric choices shown in the type combo
_HIST_TYPES = {
    "Histogram — Average Diameter": ("histogram", "avg_size"),
    "Histogram — D_max":            ("histogram", "d_max"),
    "Histogram — D_min":            ("histogram", "d_min"),
    "Histogram — Sphericalness":    ("histogram", "sphericalness"),
    "Box & Whisker — Average Diameter": ("boxplot", "avg_size"),
    "Box & Whisker — D_max":            ("boxplot", "d_max"),
    "Box & Whisker — D_min":            ("boxplot", "d_min"),
    "Box & Whisker — Sphericalness":    ("boxplot", "sphericalness"),
}


class PlotRowWidget(QFrame):
    """Single configurable plot row in the report builder."""

    remove_requested = pyqtSignal(object)  # emits self

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("plot_row_frame")
        self.setFrameShape(QFrame.Shape.StyledPanel)

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

        self._type_combo = QComboBox()
        self._type_combo.addItems(list(_HIST_TYPES.keys()))
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)
        layout.addWidget(self._type_combo, 0, 1)

        self._scope_combo = QComboBox()
        self._scope_combo.addItems(["Combined (all images)", "Per image"])
        layout.addWidget(self._scope_combo, 0, 2)

        # Row 1: histogram options
        opts_widget = QWidget()
        opts_layout = QHBoxLayout(opts_widget)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(10)

        self._fit_cb = QCheckBox("Normal fit")
        opts_layout.addWidget(self._fit_cb)

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

        self._on_type_changed()  # set initial state

    # ------------------------------------------------------------------
    def _on_type_changed(self) -> None:
        ptype, _ = _HIST_TYPES[self._type_combo.currentText()]
        is_hist = ptype == "histogram"
        for w in (self._fit_cb, self._sd1_cb, self._sd2_cb, self._sd3_cb,
                  self._bins_spin):
            w.setEnabled(is_hist)

    def config(self) -> Dict[str, Any]:
        """Return the current plot configuration dict."""
        ptype, metric = _HIST_TYPES[self._type_combo.currentText()]
        scope = (
            "per_image"
            if self._scope_combo.currentIndex() == 1
            else "combined"
        )
        return {
            "type": ptype,
            "metric": metric,
            "scope": scope,
            "bins": self._bins_spin.value(),
            "normal_fit": self._fit_cb.isChecked(),
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
                    show_normal_fit=cfg["normal_fit"],
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
