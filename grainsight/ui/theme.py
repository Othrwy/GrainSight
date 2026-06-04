"""Colour palette and QSS stylesheet for GrainSight — dark 80s retro theme."""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Palette constants
# ---------------------------------------------------------------------------
BG_DARK     = "#1a1a1a"    # main window background
BG_PANEL    = "#212121"    # panel / sidebar background
BG_WIDGET   = "#2a2a2a"    # widget surface (inputs, tables)
BG_HEADER   = "#181818"    # toolbar / header strip
BG_ALT      = "#242424"    # alternating table row

TEXT_PRIMARY   = "#e8e5d8"  # off-white body text
TEXT_SECONDARY = "#9e9b8e"  # muted / label text
TEXT_DISABLED  = "#52504a"  # disabled

ACCENT_GOLD  = "#c9a84c"    # primary accent — faded amber/gold
ACCENT_RED   = "#b84c3c"    # secondary accent — faded red
ACCENT_BLUE  = "#4a7a9c"    # tertiary accent — faded blue
ACCENT_GREEN = "#5a9c6a"    # status-ok green

OVERLAY_VALID    = "#f5d76e"  # yellow grain outline
OVERLAY_EXCLUDED = "#e05a42"  # red excluded grain outline

BORDER       = "#363630"    # subtle border
BORDER_FOCUS = "#c9a84c"    # focused input border (gold)

# ---------------------------------------------------------------------------
# QSS stylesheet
# ---------------------------------------------------------------------------
QSS = f"""
/* ===== Global ===== */
QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
    font-family: "Segoe UI", sans-serif;
    font-size: 10pt;
    selection-background-color: {ACCENT_GOLD};
    selection-color: {BG_DARK};
}}

/* ===== Main window & menus ===== */
QMainWindow {{
    background-color: {BG_DARK};
}}
QMenuBar {{
    background-color: {BG_HEADER};
    color: {TEXT_PRIMARY};
    border-bottom: 1px solid {BORDER};
    padding: 2px;
}}
QMenuBar::item:selected {{
    background-color: {BG_WIDGET};
}}
QMenu {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    padding: 4px;
}}
QMenu::item:selected {{
    background-color: {BG_WIDGET};
    color: {ACCENT_GOLD};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 4px 8px;
}}

/* ===== Toolbar ===== */
QToolBar {{
    background-color: {BG_HEADER};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 3px 6px;
}}
QToolBar QLabel {{
    color: {TEXT_SECONDARY};
    font-size: 9pt;
}}

/* ===== Buttons ===== */
QPushButton {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 12px;
    min-width: 60px;
}}
QPushButton:hover {{
    background-color: #333330;
    border-color: {ACCENT_GOLD};
    color: {ACCENT_GOLD};
}}
QPushButton:pressed {{
    background-color: #1e1e1a;
    border-color: {ACCENT_GOLD};
}}
QPushButton:disabled {{
    color: {TEXT_DISABLED};
    border-color: #2e2e2e;
}}
QPushButton#accent_btn {{
    background-color: #2a2010;
    border: 1px solid {ACCENT_GOLD};
    color: {ACCENT_GOLD};
    font-weight: bold;
}}
QPushButton#accent_btn:hover {{
    background-color: #3a2e14;
}}
QPushButton#remove_btn {{
    background-color: transparent;
    color: {ACCENT_RED};
    border: none;
    padding: 2px 6px;
    font-weight: bold;
    min-width: 20px;
}}
QPushButton#remove_btn:hover {{
    color: #e87060;
}}
QPushButton#add_btn {{
    background-color: transparent;
    border: 1px dashed {BORDER};
    color: {TEXT_SECONDARY};
    border-radius: 3px;
    padding: 4px;
}}
QPushButton#add_btn:hover {{
    border-color: {ACCENT_GOLD};
    color: {ACCENT_GOLD};
}}

/* ===== Inputs ===== */
QLineEdit, QTextEdit, QPlainTextEdit {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 2px;
    padding: 3px 6px;
    selection-background-color: {ACCENT_GOLD};
    selection-color: {BG_DARK};
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {BORDER_FOCUS};
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 2px;
    padding: 2px 4px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {BORDER_FOCUS};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    background-color: {BG_PANEL};
    border: none;
    width: 14px;
}}

/* ===== ComboBox ===== */
QComboBox {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 2px;
    padding: 3px 8px;
    min-width: 80px;
}}
QComboBox:focus {{
    border-color: {BORDER_FOCUS};
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_PANEL};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    selection-background-color: {BG_WIDGET};
    selection-color: {ACCENT_GOLD};
}}

/* ===== CheckBox ===== */
QCheckBox {{
    spacing: 6px;
    color: {TEXT_PRIMARY};
}}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {BORDER};
    background: {BG_WIDGET};
    border-radius: 2px;
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_GOLD};
    border-color: {ACCENT_GOLD};
}}
QCheckBox::indicator:hover {{
    border-color: {ACCENT_GOLD};
}}

/* ===== Labels ===== */
QLabel#section_label {{
    color: {ACCENT_GOLD};
    font-size: 9pt;
    font-weight: bold;
    letter-spacing: 1px;
}}
QLabel#status_ok {{
    color: {ACCENT_GREEN};
    font-size: 8.5pt;
}}
QLabel#status_warn {{
    color: {ACCENT_RED};
    font-size: 8.5pt;
}}
QLabel#count_bar {{
    color: {TEXT_SECONDARY};
    font-size: 9pt;
    padding: 3px 6px;
    border-top: 1px solid {BORDER};
}}

/* ===== Tab widget ===== */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG_DARK};
}}
QTabBar::tab {{
    background: {BG_PANEL};
    color: {TEXT_SECONDARY};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 5px 14px;
    margin-right: 2px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}}
QTabBar::tab:selected {{
    background: {BG_DARK};
    color: {ACCENT_GOLD};
    border-bottom: 1px solid {BG_DARK};
}}
QTabBar::tab:hover:!selected {{
    background: {BG_WIDGET};
    color: {TEXT_PRIMARY};
}}

/* ===== List widget (left image panel) ===== */
QListWidget {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 2px;
    outline: none;
}}
QListWidget::item {{
    padding: 5px 8px;
    border-bottom: 1px solid {BORDER};
    color: {TEXT_SECONDARY};
}}
QListWidget::item:selected {{
    background: {BG_WIDGET};
    color: {ACCENT_GOLD};
}}
QListWidget::item:hover:!selected {{
    background: #252520;
}}

/* ===== Table widget ===== */
QTableWidget {{
    background-color: {BG_PANEL};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    alternate-background-color: {BG_ALT};
    outline: none;
}}
QTableWidget::item {{
    padding: 3px 6px;
    color: {TEXT_PRIMARY};
    border: none;
}}
QTableWidget::item:selected {{
    background: {BG_WIDGET};
    color: {ACCENT_GOLD};
}}
QHeaderView::section {{
    background-color: {BG_HEADER};
    color: {ACCENT_GOLD};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 4px 8px;
    font-size: 8.5pt;
    font-weight: bold;
}}

/* ===== Scrollbars ===== */
QScrollBar:vertical {{
    background: {BG_PANEL};
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #444440;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background: #666660;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{
    background: {BG_PANEL};
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: #444440;
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::handle:horizontal:hover {{
    background: #666660;
}}

/* ===== Splitter ===== */
QSplitter::handle {{
    background: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 3px;
}}
QSplitter::handle:vertical {{
    height: 3px;
}}

/* ===== GroupBox ===== */
QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    margin-top: 18px;
    padding-top: 6px;
    color: {TEXT_SECONDARY};
    font-size: 9pt;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    color: {TEXT_SECONDARY};
    left: 8px;
}}

/* ===== Dialog ===== */
QDialog {{
    background-color: {BG_DARK};
}}

/* ===== Progress bar ===== */
QProgressBar {{
    background-color: {BG_WIDGET};
    border: 1px solid {BORDER};
    border-radius: 2px;
    text-align: center;
    color: {TEXT_PRIMARY};
    font-size: 8pt;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_GOLD};
    border-radius: 2px;
}}

/* ===== Report panel frame ===== */
QFrame#report_frame {{
    background-color: {BG_PANEL};
    border-top: 2px solid {ACCENT_GOLD};
}}
QFrame#plot_row_frame {{
    background-color: {BG_WIDGET};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
"""
