"""Batch test processing tab for GrainSight.

Handles a single test (disc × hopper combination) with up to 10 image cycles.
For each image the pipeline:
  1. Auto-detects the blue Ø10.05 mm sticker → px/mm calibration
  2. Auto-detects the cup rim → circular ROI mask
  3. Runs grain detection (watershed) inside the cup
  4. Records the grain count for that cycle

Results are summarised and can be exported to JSON.
"""
from __future__ import annotations

import os
import statistics
from typing import List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import (
    QObject,
    QThread,
    Qt,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QBrush, QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.cup_detector import (
    STICKER_DIAMETER_MM,
    detect_blue_sticker,
    detect_cup_rim,
)
from ..analysis.detector import detect_grains
from ..data.io import save_batch_result
from ..data.models import (
    AnalysisLimits,
    BatchTestResult,
    CalibrationData,
    CupMask,
    CycleResult,
    GrainResult,
    ImageSession,
    TestConfig,
)
from .theme import (
    ACCENT_BLUE,
    ACCENT_GOLD,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_DARK,
    BG_PANEL,
    OVERLAY_EXCLUDED,
    OVERLAY_VALID,
    TEXT_SECONDARY,
)

MAX_CYCLES = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_qpixmap(image_bgr: np.ndarray, max_w: int = 400, max_h: int = 400) -> QPixmap:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    pm = QPixmap.fromImage(qi.copy())
    return pm.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                     Qt.TransformationMode.SmoothTransformation)


def _draw_results(image_bgr: np.ndarray, grains: List[GrainResult],
                  cup_mask: Optional[CupMask]) -> np.ndarray:
    """Return a BGR preview with grain outlines and cup circle drawn."""
    out = image_bgr.copy()
    # Draw cup ring
    if cup_mask is not None:
        cv2.circle(
            out,
            (int(cup_mask.cx_px), int(cup_mask.cy_px)),
            int(cup_mask.radius_px),
            (0xA0, 0x70, 0x20),  # blue-ish in BGR
            2,
        )
    for g in grains:
        colour = (0x20, 0xD0, 0x70) if not g.excluded else (0x40, 0x60, 0xE0)
        cx, cy = int(g.centroid_x_px), int(g.centroid_y_px)
        r = max(1, int(g.major_px / 2))
        cv2.circle(out, (cx, cy), r, colour, 1)
    return out


# ---------------------------------------------------------------------------
# Background worker for a single cycle
# ---------------------------------------------------------------------------

class _CycleWorker(QObject):
    """Process one image: sticker calibration → cup detection → grain count."""
    finished = pyqtSignal(object)   # CycleResult
    preview  = pyqtSignal(object)   # (BGR ndarray with overlays, cup_mask)
    error    = pyqtSignal(str)

    def __init__(
        self,
        cycle: int,
        image_path: str,
        image_bgr: np.ndarray,
        limits: AnalysisLimits,
        fallback_px_per_mm: float = 0.0,
        shared_cup_mask: Optional[CupMask] = None,
    ) -> None:
        super().__init__()
        self._cycle = cycle
        self._image_path = image_path
        self._img = image_bgr
        self._limits = limits
        self._fallback_ppm = fallback_px_per_mm
        self._shared_cup = shared_cup_mask

    @pyqtSlot()
    def run(self) -> None:
        try:
            # 1. Calibration via blue sticker
            sticker = detect_blue_sticker(self._img)
            sticker_ok = sticker is not None
            if sticker_ok:
                _, _, r = sticker
                px_per_mm = (2.0 * r) / STICKER_DIAMETER_MM
            elif self._fallback_ppm > 0:
                px_per_mm = self._fallback_ppm
            else:
                self.error.emit(
                    f"Cycle {self._cycle}: blue sticker not found and no fallback "
                    "calibration available. Skipping."
                )
                return

            # 2. Cup mask — use shared if provided, else auto-detect
            if self._shared_cup is not None:
                cup_mask = self._shared_cup
                cup_ok = True
            else:
                rim = detect_cup_rim(
                    self._img,
                    min_diameter_mm=24.0,
                    px_per_mm=px_per_mm,
                )
                cup_ok = rim is not None
                if cup_ok:
                    cx, cy, r = rim
                    cup_mask: Optional[CupMask] = CupMask(
                        cx_px=cx, cy_px=cy, radius_px=r, source="auto"
                    )
                else:
                    cup_mask = None

            # 3. Grain detection
            grains, _ = detect_grains(self._img, px_per_mm, self._limits, cup_mask)
            included = [g for g in grains if not g.excluded]

            result = CycleResult(
                cycle=self._cycle,
                image_path=self._image_path,
                grain_count=len(included),
                px_per_mm=round(px_per_mm, 3),
                sticker_detected=sticker_ok,
                cup_detected=cup_ok,
            )

            preview_bgr = _draw_results(self._img, grains, cup_mask)
            self.preview.emit((preview_bgr, cup_mask))
            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(f"Cycle {self._cycle}: {exc}")


# ---------------------------------------------------------------------------
# Batch tab
# ---------------------------------------------------------------------------

class BatchTab(QWidget):
    """Full batch-test processing widget."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._cycles: List[Optional[CycleResult]] = [None] * MAX_CYCLES
        self._image_paths: List[Optional[str]] = [None] * MAX_CYCLES
        self._limits = AnalysisLimits()
        self._shared_cup_mask: Optional[CupMask] = None
        self._last_px_per_mm: float = 0.0  # fallback when sticker hidden
        self._active_threads: List[QThread] = []
        self._pending: int = 0   # cycles not yet finished in a batch run
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Config row ────────────────────────────────────────────────
        cfg_row = QWidget()
        cfg_row.setStyleSheet("background-color: #181818; border-bottom: 1px solid #363630;")
        cfg_hl = QHBoxLayout(cfg_row)
        cfg_hl.setContentsMargins(8, 4, 8, 4)
        cfg_hl.setSpacing(10)

        cfg_hl.addWidget(QLabel("Disc:"))
        self._disc_combo = QComboBox()
        self._disc_combo.addItems(["1", "2", "3"])
        self._disc_combo.setFixedWidth(55)
        cfg_hl.addWidget(self._disc_combo)

        cfg_hl.addWidget(QLabel("Hopper:"))
        self._hopper_combo = QComboBox()
        self._hopper_combo.addItems(["A", "B", "C"])
        self._hopper_combo.setFixedWidth(55)
        cfg_hl.addWidget(self._hopper_combo)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #363630;")
        cfg_hl.addWidget(sep)

        cfg_hl.addWidget(QLabel("Min:"))
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(0.01, 10.0)
        self._min_spin.setDecimals(2)
        self._min_spin.setValue(self._limits.min_mm)
        self._min_spin.setSuffix(" mm")
        self._min_spin.setFixedWidth(88)
        self._min_spin.valueChanged.connect(self._on_limits_changed)
        cfg_hl.addWidget(self._min_spin)

        cfg_hl.addWidget(QLabel("Max:"))
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setRange(0.01, 20.0)
        self._max_spin.setDecimals(2)
        self._max_spin.setValue(self._limits.max_mm)
        self._max_spin.setSuffix(" mm")
        self._max_spin.setFixedWidth(88)
        self._max_spin.valueChanged.connect(self._on_limits_changed)
        cfg_hl.addWidget(self._max_spin)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #363630;")
        cfg_hl.addWidget(sep2)

        cfg_hl.addWidget(QLabel("Cup mask:"))
        self._cup_status_lbl = QLabel("Not set — auto-detect per image")
        self._cup_status_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")
        cfg_hl.addWidget(self._cup_status_lbl)

        cfg_hl.addStretch()
        root.addWidget(cfg_row)

        # ── Main splitter: table | preview ───────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: table + action buttons
        left = QWidget()
        left_vl = QVBoxLayout(left)
        left_vl.setContentsMargins(0, 0, 0, 0)
        left_vl.setSpacing(4)

        self._table = QTableWidget(MAX_CYCLES, 5)
        self._table.setHorizontalHeaderLabels(
            ["Cycle", "Image", "Grains", "px/mm", "Status"]
        )
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        for col, w in enumerate((50, 200, 60, 65)):
            self._table.setColumnWidth(col, w)
        for row in range(MAX_CYCLES):
            self._table.setItem(row, 0, _cell(str(row + 1)))
            self._table.setItem(row, 1, _cell("—"))
            self._table.setItem(row, 2, _cell("—"))
            self._table.setItem(row, 3, _cell("—"))
            self._table.setItem(row, 4, _cell("No image"))
        self._table.currentCellChanged.connect(self._on_table_row_changed)
        left_vl.addWidget(self._table, 1)

        # Summary
        self._summary_lbl = QLabel("No results yet.")
        self._summary_lbl.setWordWrap(True)
        self._summary_lbl.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9pt; padding: 2px 4px;"
        )
        left_vl.addWidget(self._summary_lbl)

        # Button row
        btn_row = QWidget()
        btn_hl = QHBoxLayout(btn_row)
        btn_hl.setContentsMargins(0, 0, 0, 0)
        btn_hl.setSpacing(6)

        add_btn = QPushButton("+ Add Images")
        add_btn.setToolTip("Add up to 10 images for this test")
        add_btn.clicked.connect(self._browse_images)
        btn_hl.addWidget(add_btn)

        self._run_btn = QPushButton("▶  Run All")
        self._run_btn.setObjectName("accent_btn")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._run_all)
        btn_hl.addWidget(self._run_btn)

        self._export_btn = QPushButton("Export JSON")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_json)
        btn_hl.addWidget(self._export_btn)

        btn_hl.addStretch()
        left_vl.addWidget(btn_row)

        splitter.addWidget(left)

        # Right: preview
        right = QWidget()
        right.setMinimumWidth(300)
        right_vl = QVBoxLayout(right)
        right_vl.setContentsMargins(4, 4, 4, 4)
        right_vl.setSpacing(4)

        self._preview_lbl = QLabel("Select a cycle to preview.")
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._preview_lbl.setMinimumHeight(240)
        self._preview_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        right_vl.addWidget(self._preview_lbl, 1)

        self._detail_lbl = QLabel("")
        self._detail_lbl.setWordWrap(True)
        self._detail_lbl.setStyleSheet(
            f"color: {TEXT_SECONDARY}; font-size: 9pt; padding: 2px;"
        )
        right_vl.addWidget(self._detail_lbl)

        splitter.addWidget(right)
        splitter.setSizes([500, 340])

        root.addWidget(splitter, 1)

        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(6)
        self._progress.hide()
        root.addWidget(self._progress)

        # Store preview images keyed by cycle index
        self._preview_cache: dict[int, QPixmap] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _on_limits_changed(self) -> None:
        self._limits.min_mm = self._min_spin.value()
        self._limits.max_mm = self._max_spin.value()

    def _test_config(self) -> TestConfig:
        return TestConfig(
            disc=self._disc_combo.currentText(),
            hopper=self._hopper_combo.currentText(),
        )

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def _browse_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Images for Batch Test",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not paths:
            return
        self._add_images(paths)

    def _add_images(self, paths: List[str]) -> None:
        slot = 0
        for path in paths:
            # Find first empty slot
            while slot < MAX_CYCLES and self._image_paths[slot] is not None:
                slot += 1
            if slot >= MAX_CYCLES:
                QMessageBox.information(
                    self, "Limit Reached",
                    f"A test supports at most {MAX_CYCLES} images. "
                    f"{len(paths) - slot} image(s) were not added."
                )
                break
            self._image_paths[slot] = path
            name = os.path.basename(path)
            self._table.item(slot, 1).setText(name)
            self._table.item(slot, 1).setToolTip(path)
            self._table.item(slot, 4).setText("Ready")
            self._table.item(slot, 4).setForeground(QColor(ACCENT_GOLD))
            slot += 1

        has_images = any(p is not None for p in self._image_paths)
        self._run_btn.setEnabled(has_images)
        self._table.resizeRowsToContents()

    # ------------------------------------------------------------------
    # Run analysis
    # ------------------------------------------------------------------

    def _run_all(self) -> None:
        indices = [i for i, p in enumerate(self._image_paths) if p is not None]
        if not indices:
            return

        self._run_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._pending = len(indices)
        self._progress.setRange(0, self._pending)
        self._progress.setValue(0)
        self._progress.show()

        for idx in indices:
            path = self._image_paths[idx]
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                self._pending -= 1
                self._table.item(idx, 4).setText("Load error")
                self._table.item(idx, 4).setForeground(QColor(ACCENT_RED))
                continue

            self._table.item(idx, 4).setText("Running…")
            self._table.item(idx, 4).setForeground(QColor(ACCENT_GOLD))

            worker = _CycleWorker(
                cycle=idx + 1,
                image_path=path,
                image_bgr=image_bgr,
                limits=self._limits,
                fallback_px_per_mm=self._last_px_per_mm,
                shared_cup_mask=self._shared_cup_mask,
            )
            thread = QThread()
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            worker.finished.connect(lambda r, i=idx: self._on_cycle_done(i, r))
            worker.preview.connect(lambda data, i=idx: self._on_cycle_preview(i, data))
            worker.error.connect(lambda msg, i=idx: self._on_cycle_error(i, msg))
            worker.finished.connect(thread.quit)
            worker.error.connect(thread.quit)
            self._active_threads.append(thread)
            thread.start()

    @pyqtSlot(int, object)
    def _on_cycle_done(self, idx: int, result: CycleResult) -> None:
        self._cycles[idx] = result
        if result.px_per_mm > 0:
            self._last_px_per_mm = result.px_per_mm

        self._table.item(idx, 2).setText(str(result.grain_count))
        self._table.item(idx, 3).setText(f"{result.px_per_mm:.1f}")
        status_parts = []
        if not result.sticker_detected:
            status_parts.append("no sticker")
        if not result.cup_detected:
            status_parts.append("no cup")
        status = "✔ Done" if not status_parts else "⚠ " + ", ".join(status_parts)
        self._table.item(idx, 4).setText(status)
        colour = ACCENT_GREEN if not status_parts else ACCENT_GOLD
        self._table.item(idx, 4).setForeground(QColor(colour))

        self._pending -= 1
        self._progress.setValue(self._progress.value() + 1)
        if self._pending == 0:
            self._progress.hide()
            self._run_btn.setEnabled(True)
            self._export_btn.setEnabled(True)
            self._update_summary()

    @pyqtSlot(int, object)
    def _on_cycle_preview(self, idx: int, data) -> None:
        preview_bgr, cup_mask = data
        pm = _bgr_to_qpixmap(preview_bgr, 380, 380)
        self._preview_cache[idx] = pm
        # If this row is currently selected, update the display
        if self._table.currentRow() == idx:
            self._show_preview(idx)

    @pyqtSlot(int, str)
    def _on_cycle_error(self, idx: int, msg: str) -> None:
        self._table.item(idx, 4).setText("Error")
        self._table.item(idx, 4).setForeground(QColor(ACCENT_RED))
        self._table.item(idx, 4).setToolTip(msg)

        self._pending -= 1
        self._progress.setValue(self._progress.value() + 1)
        if self._pending == 0:
            self._progress.hide()
            self._run_btn.setEnabled(True)
            self._export_btn.setEnabled(
                any(c is not None for c in self._cycles)
            )
            self._update_summary()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _on_table_row_changed(self, row: int, *_) -> None:
        if row < 0:
            return
        self._show_preview(row)

    def _show_preview(self, idx: int) -> None:
        pm = self._preview_cache.get(idx)
        if pm:
            self._preview_lbl.setPixmap(pm)
        else:
            path = self._image_paths[idx]
            if path:
                img = cv2.imread(path)
                if img is not None:
                    self._preview_lbl.setPixmap(
                        _bgr_to_qpixmap(img, 380, 380)
                    )
                else:
                    self._preview_lbl.setText("Cannot load image.")
            else:
                self._preview_lbl.setText("No image assigned.")

        result = self._cycles[idx]
        if result:
            sticker_s = "✔" if result.sticker_detected else "✘"
            cup_s = "✔" if result.cup_detected else "✘"
            self._detail_lbl.setText(
                f"Cycle {result.cycle}  |  Grains: {result.grain_count}  |  "
                f"Scale: {result.px_per_mm:.2f} px/mm  |  "
                f"Sticker: {sticker_s}  Cup: {cup_s}"
            )
        else:
            self._detail_lbl.setText("")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _update_summary(self) -> None:
        counts = [c.grain_count for c in self._cycles if c is not None]
        if not counts:
            self._summary_lbl.setText("No results yet.")
            return
        mean = statistics.mean(counts)
        stdev = statistics.stdev(counts) if len(counts) > 1 else 0.0
        cfg = self._test_config()
        self._summary_lbl.setText(
            f"Disc {cfg.disc} / Hopper {cfg.hopper}  —  "
            f"{len(counts)} cycles completed  |  "
            f"Mean: {mean:.1f}  StdDev: {stdev:.1f}  "
            f"Min: {min(counts)}  Max: {max(counts)}"
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _export_json(self) -> None:
        cfg = self._test_config()
        cycles = [c for c in self._cycles if c is not None]
        if not cycles:
            QMessageBox.information(self, "Nothing to Export", "Run analysis first.")
            return

        default_name = f"batch_disc{cfg.disc}_hopper{cfg.hopper}.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Batch Results", default_name, "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"

        result = BatchTestResult(
            test_config=cfg,
            cycles=cycles,
            limits=self._limits,
        )
        try:
            save_batch_result(result, path)
            QMessageBox.information(self, "Exported", f"Batch results saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))


# ---------------------------------------------------------------------------
# Private helper — table cell
# ---------------------------------------------------------------------------

def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item
