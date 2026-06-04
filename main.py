import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from grainsight.ui.main_window import MainWindow


def main() -> None:
    # Enable high-DPI fractional scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("GrainSight")
    app.setApplicationVersion("0.1.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
