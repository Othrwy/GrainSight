import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication

from grainsight.ui.main_window import MainWindow

_ICON_PATH = Path(__file__).resolve().parent / "grainsight" / "assets" / "icons" / "app_icon.svg"


def _build_icon(svg_path: Path) -> QIcon:
    """Render the SVG at every size Windows needs for title bar, taskbar, and Alt+Tab."""
    try:
        from PyQt6.QtSvg import QSvgRenderer
        renderer = QSvgRenderer(str(svg_path))
        icon = QIcon()
        for size in (16, 24, 32, 48, 64, 128, 256):
            px = QPixmap(size, size)
            px.fill(Qt.GlobalColor.transparent)
            painter = QPainter(px)
            renderer.render(painter)
            painter.end()
            icon.addPixmap(px)
        return icon
    except Exception:
        return QIcon(str(svg_path))


def main() -> None:
    # Tell Windows to treat this as its own app (not python.exe) so the
    # taskbar button and pinned shortcut show the correct icon.
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "OWDesign.GrainSight.0.1"
        )
    except Exception:
        pass

    # Enable high-DPI fractional scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("GrainSight")
    app.setApplicationVersion("0.1.0")

    if _ICON_PATH.exists():
        app.setWindowIcon(_build_icon(_ICON_PATH))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
