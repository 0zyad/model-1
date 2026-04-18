"""
app.py — PyQt5 main application for the Proppant QC System.

Kiosk-ready desktop app with wizard flow:
  Dashboard -> Place Tray (camera) -> Running -> Results
  Dashboard -> Upload (file dialog) -> Running -> Results

Usage:
    python app.py
    python app.py --windowed     # force windowed mode (no fullscreen)
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# IMPORTANT: Import torch BEFORE PyQt5 to avoid DLL conflict on Windows
import torch  # noqa: F401

from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from config import APP_TITLE, FULLSCREEN, CELLPOSE_MODEL_PATH
from widgets.common import get_theme_qss, HeaderBar
from widgets.dashboard import DashboardScreen
from widgets.place_tray import PlaceTrayScreen
from widgets.running import RunningScreen
from widgets.results import ResultsScreen
from inference_stardist import ProppantAnalyzer
from logger import ResultLogger


# Screen indices
SCREEN_DASHBOARD = 0
SCREEN_PLACE_TRAY = 1
SCREEN_RUNNING = 2
SCREEN_RESULTS = 3


class ProppantQCApp(QMainWindow):
    """Main application window with stacked screens."""

    def __init__(self, windowed: bool = False):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(800, 480)

        self.analyzer = None
        self.logger = ResultLogger()
        self.run_counter = 0
        self._current_batch_id = ""
        self._current_image_path = ""
        self._windowed = windowed

        self._is_dark = True
        self.setStyleSheet(get_theme_qss(True))

        self._build_ui()
        self._connect_signals()
        self._load_model()

        # Fullscreen or windowed
        if not windowed and FULLSCREEN:
            self.showFullScreen()
        else:
            self.resize(800, 480)
            self.show()

    def _build_ui(self):
        # Central widget
        central = QStackedWidget()
        self.setCentralWidget(central)

        # Header bar (shared — drawn as part of each screen's layout via main window)
        self.header = HeaderBar()

        # Screens
        self.dashboard = DashboardScreen()
        self.place_tray = PlaceTrayScreen()
        self.running = RunningScreen()
        self.results = ResultsScreen()

        # Wrap each screen with header
        from PyQt5.QtWidgets import QWidget, QVBoxLayout

        for screen in [self.dashboard, self.place_tray, self.running, self.results]:
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            # Each page gets its own copy of the header reference
            header = HeaderBar()
            self._headers = getattr(self, "_headers", [])
            self._headers.append(header)
            layout.addWidget(header)
            layout.addWidget(screen, stretch=1)

            central.addWidget(page)

        self.stack = central

    def _connect_signals(self):
        # Theme toggle — wire all headers
        for h in getattr(self, "_headers", []):
            h.theme_toggled.connect(self._on_theme_toggled)

        # Dashboard
        self.dashboard.start_test_clicked.connect(self._on_start_test)
        self.dashboard.upload_clicked.connect(self._on_upload)
        self.dashboard.history_clicked.connect(self._on_history)

        # Place Tray
        self.place_tray.captured.connect(self._on_captured)
        self.place_tray.back_clicked.connect(self._go_dashboard)

        # Running
        self.running.analysis_complete.connect(self._on_analysis_done)

        # Results
        self.results.new_test_clicked.connect(self._go_dashboard)
        self.results.home_clicked.connect(self._go_dashboard)

    def _load_model(self):
        """Load the CellPose model at startup. Falls back to pretrained cyto3 if not trained yet."""
        try:
            model_path = CELLPOSE_MODEL_PATH if (CELLPOSE_MODEL_PATH and CELLPOSE_MODEL_PATH.exists()) else None
            self.analyzer = ProppantAnalyzer(model_path)
            device = "GPU" if self._check_cuda() else "CPU"
            from config import CELLPOSE_PRETRAINED
            label  = "fine-tuned" if model_path else f"pretrained {CELLPOSE_PRETRAINED}"
            self._update_all_headers(f"CellPose ({label}) | {device}", True)
            self.dashboard.set_model_status(f"CellPose {label}", device)
        except Exception as e:
            self._update_all_headers(f"Model error: {e}", False)
            self.dashboard.set_model_error(str(e))

        self.dashboard.refresh_last_test()

    def _check_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _update_all_headers(self, text: str, ok: bool):
        for h in getattr(self, "_headers", []):
            h.set_status(text, ok)

    def _on_theme_toggled(self, is_dark: bool):
        self._is_dark = is_dark
        self.setStyleSheet(get_theme_qss(is_dark))
        for h in getattr(self, "_headers", []):
            h.set_theme(is_dark)
        self.dashboard.apply_theme(is_dark)
        self.results.apply_theme(is_dark)

    # ── Navigation ──────────────────────────────────────────────────

    def _go_dashboard(self):
        self.place_tray.stop_camera()
        self.stack.setCurrentIndex(SCREEN_DASHBOARD)
        self.dashboard.refresh_last_test()

    def _go_screen(self, index: int):
        self.stack.setCurrentIndex(index)

    # ── Actions ─────────────────────────────────────────────────────

    def _on_start_test(self):
        """Camera flow: go to Place Tray screen."""
        if self.analyzer is None:
            QMessageBox.warning(self, "No Model", "Load a model first (train or select one).")
            return
        self._current_batch_id = self.dashboard.get_batch_id()
        self._go_screen(SCREEN_PLACE_TRAY)
        self.place_tray.start_camera()

    def _on_upload(self, file_path: str):
        """Upload flow: skip camera, go straight to Running."""
        if self.analyzer is None:
            QMessageBox.warning(self, "No Model", "Load a model first (train or select one).")
            return
        self._current_batch_id = self.dashboard.get_batch_id()
        self._current_image_path = file_path
        self._go_screen(SCREEN_RUNNING)
        self.running.start_analysis(self.analyzer, file_path)

    def _on_captured(self, image_path: str):
        """Camera captured a frame — go to Running."""
        self._current_image_path = image_path
        self._go_screen(SCREEN_RUNNING)
        self.running.start_analysis(self.analyzer, image_path)

    def _on_analysis_done(self, result: dict):
        """Analysis complete — log and show Results."""
        self.run_counter += 1
        today = datetime.now().strftime("%Y-%m-%d")
        run_id = f"{today}_{self.run_counter:03d}"
        if self._current_batch_id:
            run_id = f"{self._current_batch_id} / {run_id}"

        # Log to disk
        log_paths = self.logger.log(result)

        # Show results
        self._go_screen(SCREEN_RESULTS)
        self.results.show_result(result, log_paths, run_id)

    def _on_history(self):
        """Show a simple message — history is a stretch goal."""
        csv_path = Path("logs/results.csv")
        if csv_path.exists():
            QMessageBox.information(
                self, "Test History",
                f"History log is saved at:\n{csv_path.resolve()}\n\n"
                "Open it in Excel or any CSV viewer to see all past results."
            )
        else:
            QMessageBox.information(self, "Test History", "No test history yet.")

    # ── Keyboard shortcuts ──────────────────────────────────────────

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        elif key == Qt.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
        else:
            super().keyPressEvent(event)


# ── Entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Proppant QC System — PyQt5 App")
    parser.add_argument("--windowed", action="store_true",
                        help="Force windowed mode (no fullscreen)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 12))
    app.setStyle("Fusion")

    window = ProppantQCApp(windowed=args.windowed)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
