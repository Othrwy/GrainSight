from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QThread,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItemGroup,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRubberBand,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..analysis.calibration import (
    compute_scale_from_edge_detect,
    compute_scale_two_point,
    detect_ruler_edges,
)
from ..analysis.detector import detect_grains
from ..data.io import export_csv
from ..data.models import AnalysisLimits, CalibrationData, GrainResult, ImageSession
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_qpixmap(image_bgr: np.ndarray) -> QPixmap:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    qi = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qi.copy())  # copy detaches from numpy buffer


# ---------------------------------------------------------------------------
# Background worker for grain detection
# ---------------------------------------------------------------------------

class _AnalysisWorker(QObject):
    finished = pyqtSignal(object)  # (grains, labeled_mask)
    error = pyqtSignal(str)

    def __init__(self, image_bgr: np.ndarray, px_per_mm: float,
                 limits: AnalysisLimits) -> None:
        super().__init__()
        self._img = image_bgr
        self._ppm = px_per_mm
        self._lim = limits

    @pyqtSlot()
    def run(self) -> None:
        try:
            grains, labeled = detect_grains(self._img, self._ppm, self._lim)
            self.finished.emit((grains, labeled))
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Calibration dialog (both modes)
# ---------------------------------------------------------------------------

class _CalibrationDialog(QDialog):
    """Ask the user to confirm the physical distance for calibration."""

    def __init__(
        self,
        mode: str,
        pixel_distance: float,
        debug_image: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set Calibration")
        self.setMinimumWidth(380)

        vl = QVBoxLayout(self)

        if mode == "edge_detect" and debug_image is not None:
            lbl = QLabel()
            max_w, max_h = 420, 220
            pm = _bgr_to_qpixmap(debug_image).scaled(
                max_w, max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            lbl.setPixmap(pm)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vl.addWidget(lbl)
            info = QLabel(
                f"Detected edge separation: <b>{pixel_distance:.1f} px</b><br>"
                "Yellow lines show the detected edges. "
                "Enter the physical distance between them below."
            )
        else:
            info = QLabel(
                f"Two-point distance: <b>{pixel_distance:.1f} px</b><br>"
                "Enter the physical distance between the two points."
            )

        info.setWordWrap(True)
        info.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt; padding: 4px 0;")
        vl.addWidget(info)

        fl = QFormLayout()
        self._mm_spin = QDoubleSpinBox()
        self._mm_spin.setRange(0.01, 1000.0)
        self._mm_spin.setDecimals(3)
        self._mm_spin.setValue(20.0)
        self._mm_spin.setSuffix(" mm")
        fl.addRow("Physical distance:", self._mm_spin)
        vl.addLayout(fl)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vl.addWidget(buttons)

    @property
    def mm_value(self) -> float:
        return self._mm_spin.value()


# ---------------------------------------------------------------------------
# Image viewer  (QGraphicsView with calibration + grain overlay)
# ---------------------------------------------------------------------------

class ImageViewer(QGraphicsView):
    calib_roi_selected = pyqtSignal(QRectF)          # scene/image coords
    calib_points_selected = pyqtSignal(QPointF, QPointF)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item = None
        self._calib_mode: Optional[str] = None   # 'two_point' | 'edge_detect'
        self._calib_active = False

        # Two-point state
        self._tp_p1: Optional[QPointF] = None
        self._tp_p1_item = None
        self._tp_line_item = None

        # Rubber-band (edge detect)
        self._rb_origin = QPoint()
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self.viewport())

        # Overlay groups (grain ellipses)
        self._valid_items: List = []     # list of (ellipse, label) tuples
        self._excluded_items: List = []  # list of (ellipse, label) tuples
        self._calib_items: List = []     # calibration marker items

        # Rendering hints
        from PyQt6.QtGui import QPainter
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor(BG_DARK)))

    # ------------------------------------------------------------------
    # Image loading
    # ------------------------------------------------------------------

    def load_image(self, image_bgr: np.ndarray) -> None:
        self._clear_grain_overlay()
        self._clear_calib_overlay()
        if self._pixmap_item:
            self._scene.removeItem(self._pixmap_item)
        pixmap = _bgr_to_qpixmap(image_bgr)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(-1)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap_item:
            self.fitInView(self._scene.sceneRect(),
                           Qt.AspectRatioMode.KeepAspectRatio)

    # ------------------------------------------------------------------
    # Calibration interaction
    # ------------------------------------------------------------------

    def start_calibration(self, mode: str) -> None:
        """Activate calibration input mode. mode = 'two_point' | 'edge_detect'"""
        self._calib_mode = mode
        self._calib_active = True
        self._tp_p1 = None
        self._clear_calib_overlay()
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _end_calibration_mode(self) -> None:
        self._calib_active = False
        self._calib_mode = None
        self._tp_p1 = None
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def _clear_calib_overlay(self) -> None:
        for item in self._calib_items:
            self._scene.removeItem(item)
        self._calib_items.clear()
        self._tp_p1_item = None
        self._tp_line_item = None

    # ------------------------------------------------------------------
    # Grain overlay
    # ------------------------------------------------------------------

    def draw_grain_overlay(self, grains: List[GrainResult]) -> None:
        self._clear_grain_overlay()
        valid_pen = QPen(QColor(OVERLAY_VALID), 1.2)
        valid_pen.setCosmetic(True)
        excl_pen = QPen(QColor(OVERLAY_EXCLUDED), 1.2)
        excl_pen.setCosmetic(True)
        no_brush = QBrush(Qt.BrushStyle.NoBrush)

        font = QFont("Segoe UI", 7)

        for g in grains:
            pen = valid_pen if not g.excluded else excl_pen
            col = QColor(OVERLAY_VALID) if not g.excluded else QColor(OVERLAY_EXCLUDED)
            semi_a = g.major_px / 2.0
            semi_b = g.minor_px / 2.0

            ellipse = QGraphicsEllipseItem(
                QRectF(-semi_a, -semi_b, g.major_px, g.minor_px)
            )
            ellipse.setPen(pen)
            ellipse.setBrush(no_brush)
            ellipse.setPos(g.centroid_x_px, g.centroid_y_px)
            ellipse.setRotation(g.orientation_deg)
            ellipse.setZValue(1)
            self._scene.addItem(ellipse)

            text = QGraphicsSimpleTextItem(str(g.id))
            text.setFont(font)
            text.setBrush(QBrush(col))
            text.setPos(g.centroid_x_px + semi_a + 1, g.centroid_y_px - 6)
            text.setZValue(2)
            self._scene.addItem(text)

            pair = (ellipse, text)
            if g.excluded:
                self._excluded_items.append(pair)
            else:
                self._valid_items.append(pair)

    def set_valid_visible(self, visible: bool) -> None:
        for e, t in self._valid_items:
            e.setVisible(visible)
            t.setVisible(visible)

    def set_excluded_visible(self, visible: bool) -> None:
        for e, t in self._excluded_items:
            e.setVisible(visible)
            t.setVisible(visible)

    def _clear_grain_overlay(self) -> None:
        for e, t in self._valid_items + self._excluded_items:
            self._scene.removeItem(e)
            self._scene.removeItem(t)
        self._valid_items.clear()
        self._excluded_items.clear()

    # Show calibration overlay after edge detection
    def show_edge_overlay(
        self,
        roi_rect: Tuple[int, int, int, int],
        axis: str,
        pos1: float,
        pos2: float,
    ) -> None:
        self._clear_calib_overlay()
        rx, ry, rw, rh = roi_rect
        pen = QPen(QColor(ACCENT_GOLD), 1)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)

        # ROI rectangle
        rect_item = self._scene.addRect(
            QRectF(rx, ry, rw, rh),
            QPen(QColor(ACCENT_GOLD), 1),
            QBrush(Qt.BrushStyle.NoBrush),
        )
        rect_item.setZValue(3)
        self._calib_items.append(rect_item)

        # Edge lines (in scene/image coordinates, offset into ROI)
        if axis == "row":
            l1 = self._scene.addLine(rx, ry + pos1, rx + rw, ry + pos1,
                                     QPen(QColor(OVERLAY_VALID), 1))
            l2 = self._scene.addLine(rx, ry + pos2, rx + rw, ry + pos2,
                                     QPen(QColor(OVERLAY_VALID), 1))
        else:
            l1 = self._scene.addLine(rx + pos1, ry, rx + pos1, ry + rh,
                                     QPen(QColor(OVERLAY_VALID), 1))
            l2 = self._scene.addLine(rx + pos2, ry, rx + pos2, ry + rh,
                                     QPen(QColor(OVERLAY_VALID), 1))
        for item in (l1, l2):
            item.setZValue(3)
            self._calib_items.append(item)

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if not self._calib_active:
            super().mousePressEvent(event)
            return

        scene_pos = self.mapToScene(event.pos())

        if self._calib_mode == "two_point":
            if self._tp_p1 is None:
                self._tp_p1 = scene_pos
                # draw marker dot
                dot = self._scene.addEllipse(
                    QRectF(scene_pos.x() - 4, scene_pos.y() - 4, 8, 8),
                    QPen(QColor(ACCENT_GOLD), 1),
                    QBrush(QColor(ACCENT_GOLD)),
                )
                dot.setZValue(3)
                self._calib_items.append(dot)
                self._tp_p1_item = dot
            else:
                p2 = scene_pos
                p1 = self._tp_p1
                self._end_calibration_mode()
                self.calib_points_selected.emit(p1, p2)

        elif self._calib_mode == "edge_detect":
            self._rb_origin = event.pos()
            self._rubber_band.setGeometry(
                QRect(self._rb_origin, QSize())
            )
            self._rubber_band.show()

    def mouseMoveEvent(self, event) -> None:
        if self._calib_active and self._calib_mode == "edge_detect":
            self._rubber_band.setGeometry(
                QRect(self._rb_origin, event.pos()).normalized()
            )
        elif (
            self._calib_active
            and self._calib_mode == "two_point"
            and self._tp_p1 is not None
        ):
            # Draw preview line
            scene_pos = self.mapToScene(event.pos())
            if self._tp_line_item:
                self._scene.removeItem(self._tp_line_item)
            self._tp_line_item = self._scene.addLine(
                self._tp_p1.x(), self._tp_p1.y(),
                scene_pos.x(), scene_pos.y(),
                QPen(QColor(ACCENT_GOLD), 1, Qt.PenStyle.DashLine),
            )
            self._tp_line_item.setZValue(3)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._calib_active and self._calib_mode == "edge_detect":
            self._rubber_band.hide()
            rb_rect = QRect(self._rb_origin, event.pos()).normalized()
            if rb_rect.width() > 10 and rb_rect.height() > 10:
                scene_rect = self.mapToScene(rb_rect).boundingRect()
                self._end_calibration_mode()
                self.calib_roi_selected.emit(scene_rect)
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)


# ---------------------------------------------------------------------------
# Image Tab
# ---------------------------------------------------------------------------

class ImageTab(QWidget):
    """Full analysis tab for a single image."""

    session_changed = pyqtSignal()  # fires whenever analysis/calibration updates
    propagate_to_all = pyqtSignal(object, object, object, object)  # (calibration, cup_mask, limits, density)

    def __init__(
        self,
        session: ImageSession,
        image_bgr: np.ndarray,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._session = session
        self._image_bgr = image_bgr
        self._thread: Optional[QThread] = None
        self._worker: Optional[_AnalysisWorker] = None
        self._calib_mode_choice = "two_point"  # default

        self._build_ui()
        self._viewer.load_image(image_bgr)

    @property
    def session(self) -> ImageSession:
        return self._session

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ─────────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setStyleSheet(f"background-color: #181818; border-bottom: 1px solid #363630;")
        toolbar.setFixedHeight(44)
        tbl = QHBoxLayout(toolbar)
        tbl.setContentsMargins(8, 4, 8, 4)
        tbl.setSpacing(8)

        # Calibration mode toggle
        self._calib_mode_combo = QComboBox()
        self._calib_mode_combo.addItems(["Edge Detection", "Two-Point Click"])
        self._calib_mode_combo.setCurrentIndex(1)
        self._calib_mode_combo.setToolTip(
            "Edge Detection: draw a box around reference edges\n"
            "Two-Point Click: click two known points on the image"
        )
        self._calib_mode_combo.currentIndexChanged.connect(self._on_calib_mode_changed)
        tbl.addWidget(QLabel("Calib:"))
        tbl.addWidget(self._calib_mode_combo)

        self._calib_btn = QPushButton("Set Calibration")
        self._calib_btn.clicked.connect(self._start_calibration)
        tbl.addWidget(self._calib_btn)

        self._calib_status = QLabel("⚠  Not calibrated")
        self._calib_status.setObjectName("status_warn")
        tbl.addWidget(self._calib_status)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: #363630;")
        tbl.addWidget(sep)

        # Size limits
        tbl.addWidget(QLabel("Min:"))
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(0.01, 10.0)
        self._min_spin.setDecimals(2)
        self._min_spin.setValue(self._session.limits.min_mm)
        self._min_spin.setSuffix(" mm")
        self._min_spin.setFixedWidth(90)
        self._min_spin.valueChanged.connect(self._on_limits_changed)
        tbl.addWidget(self._min_spin)

        tbl.addWidget(QLabel("Max:"))
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setRange(0.01, 20.0)
        self._max_spin.setDecimals(2)
        self._max_spin.setValue(self._session.limits.max_mm)
        self._max_spin.setSuffix(" mm")
        self._max_spin.setFixedWidth(90)
        self._max_spin.valueChanged.connect(self._on_limits_changed)
        tbl.addWidget(self._max_spin)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet(f"color: #363630;")
        tbl.addWidget(sep2)

        self._find_btn = QPushButton("🔍  Look for Grains")
        self._find_btn.setObjectName("accent_btn")
        self._find_btn.setEnabled(False)
        self._find_btn.clicked.connect(self._run_analysis)
        tbl.addWidget(self._find_btn)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.VLine)
        sep3.setStyleSheet(f"color: #363630;")
        tbl.addWidget(sep3)

        # Overlay toggles
        self._show_valid_cb = QCheckBox("Valid")
        self._show_valid_cb.setChecked(True)
        self._show_valid_cb.toggled.connect(
            lambda v: self._viewer.set_valid_visible(v)
        )
        tbl.addWidget(self._show_valid_cb)

        self._show_excl_cb = QCheckBox("Excluded")
        self._show_excl_cb.setChecked(True)
        self._show_excl_cb.toggled.connect(
            lambda v: self._viewer.set_excluded_visible(v)
        )
        tbl.addWidget(self._show_excl_cb)

        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.VLine)
        sep4.setStyleSheet(f"color: #363630;")
        tbl.addWidget(sep4)

        self._export_btn = QPushButton("Export CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export_csv)
        tbl.addWidget(self._export_btn)

        sep5 = QFrame()
        sep5.setFrameShape(QFrame.Shape.VLine)
        sep5.setStyleSheet("color: #363630;")
        tbl.addWidget(sep5)

        tbl.addWidget(QLabel("Density:"))
        self._density_spin = QDoubleSpinBox()
        self._density_spin.setRange(0.1, 20.0)
        self._density_spin.setDecimals(3)
        self._density_spin.setValue(self._session.grain_density_g_cm3)
        self._density_spin.setSuffix(" g/cm³")
        self._density_spin.setFixedWidth(110)
        self._density_spin.setToolTip(
            "Grain material density for mass estimation.\n"
            "Volume assumes depth = minor diameter (oblate spheroid)."
        )
        self._density_spin.valueChanged.connect(self._on_density_changed)
        tbl.addWidget(self._density_spin)

        sep6 = QFrame()
        sep6.setFrameShape(QFrame.Shape.VLine)
        sep6.setStyleSheet("color: #363630;")
        tbl.addWidget(sep6)

        self._propagate_btn = QPushButton("Propagate to All")
        self._propagate_btn.setToolTip("Copy calibration, cup mask, limits, and density to all other image tabs")
        self._propagate_btn.setEnabled(False)
        self._propagate_btn.clicked.connect(self._on_propagate_btn_clicked)
        tbl.addWidget(self._propagate_btn)

        tbl.addStretch()
        root.addWidget(toolbar)

        # ── Progress bar (hidden until analysis) ────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setFixedHeight(4)
        self._progress.hide()
        root.addWidget(self._progress)

        # ── Main splitter ────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._viewer = ImageViewer()
        self._viewer.calib_roi_selected.connect(self._on_roi_selected)
        self._viewer.calib_points_selected.connect(self._on_points_selected)
        splitter.addWidget(self._viewer)

        # Side panel
        side = QWidget()
        side.setMaximumWidth(320)
        side_vl = QVBoxLayout(side)
        side_vl.setContentsMargins(4, 4, 4, 4)
        side_vl.setSpacing(4)

        self._count_bar = QLabel("No analysis run yet.")
        self._count_bar.setObjectName("count_bar")
        self._count_bar.setWordWrap(True)
        side_vl.addWidget(self._count_bar)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["#", "D_max mm", "D_min mm", "Spherical.", "Status"]
        )
        self._table.setAlternatingRowColors(True)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        for col, w in enumerate((30, 68, 68, 68)):
            self._table.setColumnWidth(col, w)
        side_vl.addWidget(self._table)

        splitter.addWidget(side)
        splitter.setSizes([700, 260])
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def _on_calib_mode_changed(self, idx: int) -> None:
        self._calib_mode_choice = "edge_detect" if idx == 0 else "two_point"

    def _start_calibration(self) -> None:
        self._viewer.start_calibration(self._calib_mode_choice)

    def _on_roi_selected(self, scene_rect: QRectF) -> None:
        """Called after user draws rubber-band (edge detect mode)."""
        x = int(scene_rect.x())
        y = int(scene_rect.y())
        w = int(scene_rect.width())
        h = int(scene_rect.height())
        roi_rect = (x, y, w, h)

        gray = cv2.cvtColor(self._image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            px_dist, debug_img, axis, pos1, pos2 = detect_ruler_edges(gray, roi_rect)
        except ValueError as exc:
            QMessageBox.warning(self, "Edge Detection Failed", str(exc))
            return

        # Show detected edges on viewer
        self._viewer.show_edge_overlay(roi_rect, axis, pos1, pos2)

        dlg = _CalibrationDialog("edge_detect", px_dist, debug_img, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        mm = dlg.mm_value
        try:
            ppm = compute_scale_from_edge_detect(px_dist, mm)
        except ValueError as exc:
            QMessageBox.warning(self, "Calibration Error", str(exc))
            return

        self._session.calibration = CalibrationData(
            mode="edge_detect",
            px_per_mm=ppm,
            reference_mm=mm,
            pixel_distance=px_dist,
            roi=roi_rect,
            edge_axis=axis,
            edge_pos1=pos1,
            edge_pos2=pos2,
        )
        self._update_calib_status()

    def _on_points_selected(self, p1: QPointF, p2: QPointF) -> None:
        """Called after user clicks two points (two-point mode)."""
        tp1 = (p1.x(), p1.y())
        tp2 = (p2.x(), p2.y())
        dx, dy = tp2[0] - tp1[0], tp2[1] - tp1[1]
        px_dist = math.sqrt(dx * dx + dy * dy)

        dlg = _CalibrationDialog("two_point", px_dist, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        mm = dlg.mm_value
        try:
            ppm = compute_scale_two_point(tp1, tp2, mm)
        except ValueError as exc:
            QMessageBox.warning(self, "Calibration Error", str(exc))
            return

        self._session.calibration = CalibrationData(
            mode="two_point",
            px_per_mm=ppm,
            reference_mm=mm,
            pixel_distance=px_dist,
            p1=tp1,
            p2=tp2,
        )
        self._update_calib_status()

    def _update_calib_status(self) -> None:
        cal = self._session.calibration
        if cal and cal.px_per_mm > 0:
            micron = 1000.0 / cal.px_per_mm
            self._calib_status.setText(
                f"✔  {cal.px_per_mm:.2f} px/mm  ({micron:.2f} µm/px)"
            )
            self._calib_status.setObjectName("status_ok")
            self._calib_status.style().unpolish(self._calib_status)
            self._calib_status.style().polish(self._calib_status)
            self._find_btn.setEnabled(True)
            self._propagate_btn.setEnabled(True)
        else:
            self._calib_status.setText("⚠  Not calibrated")
            self._calib_status.setObjectName("status_warn")
            self._find_btn.setEnabled(False)
            self._propagate_btn.setEnabled(False)
        self.session_changed.emit()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _on_density_changed(self, value: float) -> None:
        self._session.grain_density_g_cm3 = value
        if self._session.analysed:
            n_total = len(self._session.grains)
            n_inc = len(self._session.included_grains)
            n_exc = len(self._session.excluded_grains)
            total_mass = sum(g.volume_mm3 * value for g in self._session.included_grains)
            self._count_bar.setText(
                f"Detected: {n_total}  |  Included: {n_inc}  |"
                f"  Excluded: {n_exc}  |  Mass: {total_mass:.4f} mg"
            )
        self.session_changed.emit()

    def _on_limits_changed(self) -> None:
        self._session.limits.min_mm = self._min_spin.value()
        self._session.limits.max_mm = self._max_spin.value()
        if self._session.analysed:
            self._session.refilter()
            self._refresh_overlay_and_table()
            self.session_changed.emit()

    def _run_analysis(self) -> None:
        if not self._session.is_calibrated:
            QMessageBox.warning(self, "Not Calibrated",
                                "Please calibrate before running analysis.")
            return
        if self._thread and self._thread.isRunning():
            return  # already running

        self._find_btn.setEnabled(False)
        self._progress.show()

        self._session.limits.min_mm = self._min_spin.value()
        self._session.limits.max_mm = self._max_spin.value()

        self._worker = _AnalysisWorker(
            self._image_bgr,
            self._session.calibration.px_per_mm,
            self._session.limits,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_analysis_done)
        self._worker.error.connect(self._on_analysis_error)
        self._worker.finished.connect(self._thread.quit)
        self._thread.start()

    @pyqtSlot(object)
    def _on_analysis_done(self, result) -> None:
        grains, _ = result
        self._progress.hide()
        self._session.grains = grains
        self._session.analysed = True
        self._find_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._refresh_overlay_and_table()
        self.session_changed.emit()

    @pyqtSlot(str)
    def _on_analysis_error(self, msg: str) -> None:
        self._progress.hide()
        self._find_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis Error", msg)

    def _refresh_overlay_and_table(self) -> None:
        grains = self._session.grains
        self._viewer.draw_grain_overlay(grains)
        self._viewer.set_valid_visible(self._show_valid_cb.isChecked())
        self._viewer.set_excluded_visible(self._show_excl_cb.isChecked())
        self._refresh_table(grains)
        n_total = len(grains)
        n_inc = len(self._session.included_grains)
        n_exc = len(self._session.excluded_grains)
        density = self._session.grain_density_g_cm3
        total_mass = sum(g.volume_mm3 * density for g in self._session.included_grains)
        self._count_bar.setText(
            f"Detected: {n_total}  |  Included: {n_inc}  |"
            f"  Excluded: {n_exc}  |  Mass: {total_mass:.4f} mg"
        )

    def _refresh_table(self, grains: List[GrainResult]) -> None:
        self._table.setRowCount(len(grains))
        for row, g in enumerate(grains):
            def cell(val, align=Qt.AlignmentFlag.AlignCenter):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(align)
                return item

            self._table.setItem(row, 0, cell(g.id))
            self._table.setItem(row, 1, cell(f"{g.major_mm:.3f}"))
            self._table.setItem(row, 2, cell(f"{g.minor_mm:.3f}"))
            self._table.setItem(row, 3, cell(f"{g.sphericalness:.2f}"))
            status = "Included" if not g.excluded else f"Excl: {g.exclusion_reason}"
            s_item = cell(status, Qt.AlignmentFlag.AlignLeft)
            if g.excluded:
                s_item.setForeground(QColor(OVERLAY_EXCLUDED))
            else:
                s_item.setForeground(QColor(OVERLAY_VALID))
            self._table.setItem(row, 4, s_item)
        self._table.resizeRowsToContents()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def _export_csv(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        base = os.path.splitext(os.path.basename(self._session.image_path))[0]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", f"{base}_grains.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        try:
            export_csv(self._session, path)
            QMessageBox.information(self, "Exported", f"CSV saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Propagate settings to other tabs
    # ------------------------------------------------------------------

    def _on_propagate_btn_clicked(self) -> None:
        self.propagate_to_all.emit(
            self._session.calibration,
            self._session.cup_mask,
            self._session.limits,
            self._session.grain_density_g_cm3,
        )

    def apply_shared_settings(
        self,
        calibration: object,
        cup_mask: object,
        limits: object,
        density: float = 1.5,
    ) -> None:
        """Apply calibration, cup mask, limits, and density from another tab."""
        self._session.calibration = calibration
        self._session.cup_mask = cup_mask
        self._session.limits = limits
        self._session.grain_density_g_cm3 = density

        # Sync UI controls to the new values
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._density_spin.blockSignals(True)
        self._min_spin.setValue(limits.min_mm)
        self._max_spin.setValue(limits.max_mm)
        self._density_spin.setValue(density)
        self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(False)
        self._density_spin.blockSignals(False)

        self._update_calib_status()
