from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..data.io import export_csv, load_multi_session, save_multi_session
from ..data.models import ImageSession
from .batch_tab import BatchTab
from .image_tab import ImageTab
from .report_panel import ReportPanel
from .theme import (
    ACCENT_GOLD,
    ACCENT_GREEN,
    ACCENT_RED,
    BG_PANEL,
    QSS,
    TEXT_SECONDARY,
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GrainSight")
        self.resize(1280, 820)
        self.setStyleSheet(QSS)

        self._sessions: Dict[int, ImageSession] = {}  # tab_index → session
        self._tabs_to_sessions: Dict[QWidget, ImageSession] = {}

        self._build_ui()
        self._build_menus()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_vl = QVBoxLayout(central)
        root_vl.setContentsMargins(0, 0, 0, 0)
        root_vl.setSpacing(0)

        # Main vertical splitter: [top_content, report_panel]
        self._main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top horizontal splitter: [left_panel, tab_widget]
        self._top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: image list ───────────────────────────────────────
        left = QWidget()
        left.setMinimumWidth(160)
        left.setMaximumWidth(240)
        left_vl = QVBoxLayout(left)
        left_vl.setContentsMargins(6, 6, 6, 6)
        left_vl.setSpacing(4)

        hdr = QLabel("IMAGES")
        hdr.setObjectName("section_label")
        left_vl.addWidget(hdr)

        self._image_list = QListWidget()
        self._image_list.currentRowChanged.connect(self._on_list_row_changed)
        left_vl.addWidget(self._image_list, 1)

        btn_row = QWidget()
        btn_rl = QHBoxLayout(btn_row)
        btn_rl.setContentsMargins(0, 0, 0, 0)
        btn_rl.setSpacing(4)

        add_btn = QPushButton("+ Add")
        add_btn.clicked.connect(self._browse_images)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_current_image)

        btn_rl.addWidget(add_btn)
        btn_rl.addWidget(remove_btn)
        left_vl.addWidget(btn_row)

        analyse_all_btn = QPushButton("Analyse All")
        analyse_all_btn.setObjectName("accent_btn")
        analyse_all_btn.clicked.connect(self._analyse_all)
        left_vl.addWidget(analyse_all_btn)

        batch_btn = QPushButton("+ Batch Test")
        batch_btn.setToolTip("Open a new batch test tab (disc × hopper, 10 cycles)")
        batch_btn.clicked.connect(self._new_batch_tab)
        left_vl.addWidget(batch_btn)

        self._top_splitter.addWidget(left)

        # ── Tab widget ───────────────────────────────────────────────────
        self._tab_widget = QTabWidget()
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.tabCloseRequested.connect(self._close_tab)
        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        placeholder = QLabel("Add images to begin.\nDrag & drop or use '+ Add'.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13pt;")
        self._tab_widget.addTab(placeholder, "Welcome")
        self._tab_widget.setTabsClosable(False)
        self._welcome_tab = placeholder

        self._top_splitter.addWidget(self._tab_widget)
        self._top_splitter.setSizes([190, 1050])

        self._main_splitter.addWidget(self._top_splitter)

        # ── Report panel ─────────────────────────────────────────────────
        self._report_panel = ReportPanel(get_sessions=self._get_analysed_sessions)
        self._main_splitter.addWidget(self._report_panel)
        self._main_splitter.setSizes([620, 30])  # report collapsed by default

        root_vl.addWidget(self._main_splitter, 1)

        # Accept drag-and-drop of image files
        self.setAcceptDrops(True)

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File menu
        file_menu = mb.addMenu("&File")

        act = QAction("Add Images…", self)
        act.setShortcut("Ctrl+O")
        act.triggered.connect(self._browse_images)
        file_menu.addAction(act)

        act = QAction("Open Session (JSON)…", self)
        act.setShortcut("Ctrl+Shift+O")
        act.triggered.connect(self._load_session)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("Save Session (JSON)…", self)
        act.setShortcut("Ctrl+S")
        act.triggered.connect(self._save_session)
        file_menu.addAction(act)

        file_menu.addSeparator()

        act = QAction("Exit", self)
        act.setShortcut("Ctrl+Q")
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        # Analysis menu
        anal_menu = mb.addMenu("&Analysis")
        act = QAction("Analyse All Images", self)
        act.triggered.connect(self._analyse_all)
        anal_menu.addAction(act)

        act = QAction("New Batch Test…", self)
        act.setShortcut("Ctrl+B")
        act.triggered.connect(self._new_batch_tab)
        anal_menu.addAction(act)

        # Help menu
        help_menu = mb.addMenu("&Help")
        act = QAction("About GrainSight", self)
        act.triggered.connect(self._about)
        help_menu.addAction(act)

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def _browse_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Open Image(s)",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if paths:
            self._add_images(paths)

    def _add_images(self, paths: List[str]) -> None:
        for path in paths:
            if any(s.image_path == path for s in self._tabs_to_sessions.values()):
                continue  # already loaded

            image_bgr = cv2.imread(path)
            if image_bgr is None:
                QMessageBox.warning(
                    self, "Load Error",
                    f"Could not read image:\n{path}"
                )
                continue

            session = ImageSession(image_path=path)
            self._open_tab(session, image_bgr)

    def _open_tab(
        self,
        session: ImageSession,
        image_bgr: Optional = None,
    ) -> None:
        """Create and open an ImageTab for *session*."""
        # Remove welcome placeholder if present
        if self._welcome_tab is not None:
            idx = self._tab_widget.indexOf(self._welcome_tab)
            if idx >= 0:
                self._tab_widget.removeTab(idx)
            self._welcome_tab = None
            self._tab_widget.setTabsClosable(True)

        tab = ImageTab(session, image_bgr)
        tab.session_changed.connect(self._refresh_list_item)
        tab.session_changed.connect(self._report_panel.refresh_fit_options)
        tab.propagate_to_all.connect(self._on_propagate_to_all)

        name = os.path.splitext(os.path.basename(session.image_path))[0]
        idx = self._tab_widget.addTab(tab, name)
        self._tabs_to_sessions[tab] = session
        self._tab_widget.setCurrentIndex(idx)
        self._add_list_item(session, name)

    def _add_list_item(self, session: ImageSession, name: str) -> None:
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, session)
        item.setToolTip(session.image_path)
        self._update_list_item_style(item, session)
        self._image_list.addItem(item)

    def _refresh_list_item(self) -> None:
        """Update all list items to reflect current analysis status."""
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            session: ImageSession = item.data(Qt.ItemDataRole.UserRole)
            if session:
                self._update_list_item_style(item, session)

    def _update_list_item_style(
        self, item: QListWidgetItem, session: ImageSession
    ) -> None:
        if session.analysed:
            n = len(session.included_grains)
            item.setText(f"✔ {os.path.splitext(os.path.basename(session.image_path))[0]}\n   {n} grains")
            item.setForeground(__import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(ACCENT_GREEN))
        elif session.is_calibrated:
            item.setText(f"◎ {os.path.splitext(os.path.basename(session.image_path))[0]}\n   Calibrated")
            item.setForeground(__import__("PyQt6.QtGui", fromlist=["QColor"]).QColor(ACCENT_GOLD))
        else:
            item.setText(os.path.splitext(os.path.basename(session.image_path))[0])
            from PyQt6.QtGui import QColor
            item.setForeground(QColor(TEXT_SECONDARY))

    def _remove_current_image(self) -> None:
        row = self._image_list.currentRow()
        if row < 0:
            return
        item = self._image_list.takeItem(row)
        if item is None:
            return
        session = item.data(Qt.ItemDataRole.UserRole)
        # Find matching tab and close it
        for i in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            if isinstance(widget, ImageTab) and widget.session is session:
                self._tab_widget.removeTab(i)
                del self._tabs_to_sessions[widget]
                break

    def _close_tab(self, idx: int) -> None:
        widget = self._tab_widget.widget(idx)
        self._tab_widget.removeTab(idx)
        if isinstance(widget, ImageTab):
            session = widget.session
            self._tabs_to_sessions.pop(widget, None)
            # Remove from list
            for i in range(self._image_list.count()):
                item = self._image_list.item(i)
                if item and item.data(Qt.ItemDataRole.UserRole) is session:
                    self._image_list.takeItem(i)
                    break
        if self._tab_widget.count() == 0:
            self._show_welcome()

    def _show_welcome(self) -> None:
        placeholder = QLabel("Add images to begin.\nDrag & drop or use '+ Add'.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 13pt;")
        self._tab_widget.addTab(placeholder, "Welcome")
        self._tab_widget.setTabsClosable(False)
        self._welcome_tab = placeholder

    def _on_list_row_changed(self, row: int) -> None:
        """Sync list selection → tab selection."""
        if row < 0:
            return
        item = self._image_list.item(row)
        if item is None:
            return
        session = item.data(Qt.ItemDataRole.UserRole)
        for i in range(self._tab_widget.count()):
            w = self._tab_widget.widget(i)
            if isinstance(w, ImageTab) and w.session is session:
                self._tab_widget.setCurrentIndex(i)
                break

    def _on_tab_changed(self, idx: int) -> None:
        """Sync tab selection → list selection."""
        w = self._tab_widget.widget(idx)
        if not isinstance(w, ImageTab):
            return
        session = w.session
        for i in range(self._image_list.count()):
            item = self._image_list.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) is session:
                self._image_list.setCurrentRow(i)
                break

    def _analyse_all(self) -> None:
        """Trigger analysis on every calibrated but not-yet-analysed tab."""
        for i in range(self._tab_widget.count()):
            w = self._tab_widget.widget(i)
            if isinstance(w, ImageTab) and w.session.is_calibrated and not w.session.analysed:
                w._run_analysis()

    def _on_propagate_to_all(
        self,
        calibration: object,
        cup_mask: object,
        limits: object,
        density: float = 1.5,
    ) -> None:
        """Apply one tab's calibration/cup/limits/density to all other image tabs."""
        source_tab = self.sender()
        count = 0
        for i in range(self._tab_widget.count()):
            w = self._tab_widget.widget(i)
            if isinstance(w, ImageTab) and w is not source_tab:
                w.apply_shared_settings(calibration, cup_mask, limits, density)
                count += 1
        if count:
            QMessageBox.information(
                self,
                "Settings Applied",
                f"Calibration, cup mask, and limits copied to {count} other tab(s).\n"
                "Click \u2018Look for Grains\u2019 on each tab to run detection."
            )

    def _new_batch_tab(self) -> None:
        """Open a new batch-test tab."""
        # Remove welcome placeholder if present
        if self._welcome_tab is not None:
            idx = self._tab_widget.indexOf(self._welcome_tab)
            if idx >= 0:
                self._tab_widget.removeTab(idx)
            self._welcome_tab = None
            self._tab_widget.setTabsClosable(True)

        tab = BatchTab()
        batch_count = sum(
            1 for i in range(self._tab_widget.count())
            if isinstance(self._tab_widget.widget(i), BatchTab)
        ) + 1
        label = f"Batch {batch_count}"
        idx = self._tab_widget.addTab(tab, label)
        self._tab_widget.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _get_analysed_sessions(self) -> List[ImageSession]:
        return [
            w.session
            for i in range(self._tab_widget.count())
            for w in [self._tab_widget.widget(i)]
            if isinstance(w, ImageTab)
        ]

    def _save_session(self) -> None:
        sessions = self._get_analysed_sessions()
        if not sessions:
            QMessageBox.information(self, "Nothing to save", "No images loaded.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "", "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            save_multi_session(sessions, path)
            QMessageBox.information(self, "Saved", f"Session saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Error", str(exc))

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "", "JSON Files (*.json)"
        )
        if not path:
            return
        try:
            sessions = load_multi_session(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", str(exc))
            return

        for session in sessions:
            image_bgr = None
            if os.path.isfile(session.image_path):
                image_bgr = cv2.imread(session.image_path)
            if image_bgr is None:
                # Prompt user to locate the image
                img_path, _ = QFileDialog.getOpenFileName(
                    self,
                    f"Locate image for session: "
                    f"{os.path.basename(session.image_path)}",
                    "",
                    "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
                )
                if img_path:
                    session.image_path = img_path
                    image_bgr = cv2.imread(img_path)
                else:
                    continue

            self._open_tab(session, image_bgr)

    # ------------------------------------------------------------------
    # Drag & drop
    # ------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p.lower().endswith(
                (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
            ):
                paths.append(p)
        if paths:
            self._add_images(paths)

    # ------------------------------------------------------------------
    # About
    # ------------------------------------------------------------------

    def _about(self) -> None:
        QMessageBox.information(
            self,
            "About GrainSight",
            "<b>GrainSight v0.1.0</b><br><br>"
            "Grain dimension analyser for macro photographs "
            "of small spheroidal particles.<br><br>"
            "Workflow: Load → Calibrate → Analyse → Export CSV → Generate PDF Report.",
        )
