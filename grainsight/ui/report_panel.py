from __future__ import annotations

import datetime
import os
from typing import Callable, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..data.models import ImageSession, ReportMetadata
from ..report.pdf_generator import generate_pdf
from .plot_row_widget import PlotRowWidget
from .theme import ACCENT_GOLD, BG_DARK, BG_PANEL, TEXT_SECONDARY


class ReportPanel(QFrame):
    """Collapsible bottom panel that builds and exports a PDF report."""

    def __init__(
        self,
        get_sessions: Callable[[], List[ImageSession]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("report_frame")
        self._get_sessions = get_sessions
        self._plot_rows: List[PlotRowWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header / toggle bar ──────────────────────────────────────────
        header = QWidget()
        header.setFixedHeight(28)
        header.setStyleSheet(
            f"background-color: {BG_PANEL}; border-top: 2px solid {ACCENT_GOLD};"
        )
        hh = QHBoxLayout(header)
        hh.setContentsMargins(10, 0, 10, 0)

        self._toggle_btn = QPushButton("▲  Report Builder")
        self._toggle_btn.setFlat(True)
        self._toggle_btn.setStyleSheet(
            f"color: {ACCENT_GOLD}; font-weight: bold; font-size: 9pt; "
            "text-align: left; border: none; padding: 0;"
        )
        self._toggle_btn.clicked.connect(self._toggle)
        hh.addWidget(self._toggle_btn)
        hh.addStretch()

        self._gen_btn = QPushButton("Generate PDF Report")
        self._gen_btn.setObjectName("accent_btn")
        self._gen_btn.clicked.connect(self._generate)
        hh.addWidget(self._gen_btn)

        root.addWidget(header)

        # ── Content (collapsible) ────────────────────────────────────────
        self._content = QWidget()
        content_layout = QHBoxLayout(self._content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left — metadata
        meta_widget = self._build_meta_panel()
        splitter.addWidget(meta_widget)

        # Right — plot list
        plots_widget = self._build_plots_panel()
        splitter.addWidget(plots_widget)
        splitter.setSizes([340, 560])
        content_layout.addWidget(splitter)

        root.addWidget(self._content)
        self._content.hide()
        self._expanded = False

    # ------------------------------------------------------------------
    # Build sub-panels
    # ------------------------------------------------------------------

    def _build_meta_panel(self) -> QWidget:
        w = QWidget()
        fl = QFormLayout(w)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.setSpacing(6)
        fl.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        def lbl(text):
            l = QLabel(text)
            l.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 9pt;")
            return l

        self._title_edit = QLineEdit("Grain Size Analysis Report")
        self._user_edit = QLineEdit()
        self._sample_edit = QLineEdit()
        self._date_edit = QLineEdit(datetime.date.today().isoformat())
        self._details_edit = QTextEdit()
        self._details_edit.setFixedHeight(60)
        self._summary_edit = QTextEdit()
        self._summary_edit.setFixedHeight(80)
        self._summary_edit.setPlaceholderText(
            "Auto-generated from session data (edit here to customise)…"
        )

        auto_btn = QPushButton("Auto-generate summary")
        auto_btn.clicked.connect(self._auto_summary)
        auto_btn.setStyleSheet("font-size: 8.5pt; padding: 2px 8px;")

        fl.addRow(lbl("Title:"), self._title_edit)
        fl.addRow(lbl("Date:"), self._date_edit)
        fl.addRow(lbl("User:"), self._user_edit)
        fl.addRow(lbl("Sample ID:"), self._sample_edit)
        fl.addRow(lbl("Test Details:"), self._details_edit)
        fl.addRow(lbl("Summary:"), self._summary_edit)
        fl.addRow("", auto_btn)
        return w

    def _build_plots_panel(self) -> QWidget:
        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(6)

        plots_lbl = QLabel("PLOTS")
        plots_lbl.setObjectName("section_label")
        vl.addWidget(plots_lbl)

        # Scroll area for plot rows
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._plots_container = QWidget()
        self._plots_vl = QVBoxLayout(self._plots_container)
        self._plots_vl.setContentsMargins(0, 0, 0, 0)
        self._plots_vl.setSpacing(6)
        self._plots_vl.addStretch()

        scroll.setWidget(self._plots_container)
        vl.addWidget(scroll, 1)

        add_btn = QPushButton("＋  Add Plot")
        add_btn.setObjectName("add_btn")
        add_btn.clicked.connect(self._add_plot_row)
        vl.addWidget(add_btn)

        return w

    # ------------------------------------------------------------------
    # Plot row management
    # ------------------------------------------------------------------

    def _add_plot_row(self) -> None:
        row = PlotRowWidget()
        row.remove_requested.connect(self._remove_plot_row)
        self._plot_rows.append(row)
        # Insert before the stretch
        idx = self._plots_vl.count() - 1
        self._plots_vl.insertWidget(idx, row)
        # Populate GoF values immediately if sessions are available
        sessions = [s for s in self._get_sessions() if s.analysed]
        if sessions:
            row.update_fit_options(sessions)

    def _remove_plot_row(self, row: PlotRowWidget) -> None:
        self._plot_rows.remove(row)
        self._plots_vl.removeWidget(row)
        row.deleteLater()

    # ------------------------------------------------------------------
    # Collapse / expand
    # ------------------------------------------------------------------

    def _toggle(self) -> None:
        if self._expanded:
            self._content.hide()
            self._toggle_btn.setText("▲  Report Builder")
        else:
            self._content.show()
            self._toggle_btn.setText("▼  Report Builder")
            self.refresh_fit_options()
        self._expanded = not self._expanded

    def refresh_fit_options(self) -> None:
        """Recompute GoF for all plot rows using the current analysed sessions."""
        sessions = [s for s in self._get_sessions() if s.analysed]
        for row in self._plot_rows:
            row.update_fit_options(sessions)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _auto_summary(self) -> None:
        sessions = self._get_sessions()
        if not sessions:
            self._summary_edit.setPlainText("No sessions loaded.")
            return
        parts = [s.auto_summary() for s in sessions]
        self._summary_edit.setPlainText("  ".join(parts))

    def _gather_metadata(self) -> ReportMetadata:
        return ReportMetadata(
            title=self._title_edit.text().strip() or "Grain Analysis Report",
            summary=self._summary_edit.toPlainText().strip(),
            date=self._date_edit.text().strip(),
            user=self._user_edit.text().strip(),
            sample_id=self._sample_edit.text().strip(),
            test_details=self._details_edit.toPlainText().strip(),
        )

    def _generate(self) -> None:
        sessions = [s for s in self._get_sessions() if s.analysed]
        if not sessions:
            QMessageBox.warning(
                self, "No Data",
                "No analysed images found. Run grain analysis first.",
            )
            return

        plot_cfgs = [row.config() for row in self._plot_rows]
        if not plot_cfgs:
            QMessageBox.warning(
                self, "No Plots",
                "Add at least one plot to the report before generating.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report", "", "PDF Files (*.pdf)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"

        meta = self._gather_metadata()
        if not meta.summary:
            meta.summary = "  ".join(s.auto_summary() for s in sessions)

        try:
            generate_pdf(sessions, plot_cfgs, meta, path)
        except Exception as exc:
            QMessageBox.critical(self, "PDF Error", str(exc))
            return

        QMessageBox.information(
            self, "Report Saved",
            f"Report saved to:\n{path}",
        )
        # Try to open with the system PDF viewer
        try:
            os.startfile(path)
        except Exception:
            pass
