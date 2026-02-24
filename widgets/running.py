"""
widgets/running.py — Analysis in progress screen with threaded inference.
"""
import time

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from config import FONT_SIZE_LARGE, FONT_SIZE_NORMAL, MUTED_COLOR
from widgets.common import Card, ProgressCard


class AnalysisWorker(QThread):
    """Runs ProppantAnalyzer.analyze() in a background thread."""

    progress = pyqtSignal(int, str)      # (percentage, status_text)
    finished = pyqtSignal(dict)          # result dict
    error = pyqtSignal(str)              # error message

    def __init__(self, analyzer, image_path: str):
        super().__init__()
        self.analyzer = analyzer
        self.image_path = image_path

    def run(self):
        try:
            self.progress.emit(10, "Loading image...")
            time.sleep(0.15)

            self.progress.emit(30, "Running YOLO segmentation...")
            result = self.analyzer.analyze(self.image_path)

            self.progress.emit(70, "Computing composition...")
            time.sleep(0.15)

            self.progress.emit(90, "Generating overlay...")
            time.sleep(0.15)

            self.progress.emit(100, "Done!")
            time.sleep(0.2)

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class RunningScreen(QWidget):
    """Shows progress while analysis runs in a background thread."""

    analysis_complete = pyqtSignal(dict)  # emits result dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._thumb_data = None  # prevent GC of image data
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addStretch()

        # Title
        self.title_label = QLabel("Analyzing Image...")
        self.title_label.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("color: white;")
        layout.addWidget(self.title_label)

        # Image thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setFixedHeight(110)
        self.thumb_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.thumb_label)

        # Progress card
        self.progress_card = ProgressCard()
        self.progress_card.setMaximumWidth(480)
        self.progress_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Center the card
        h_center = QHBoxLayout()
        h_center.addStretch()
        h_center.addWidget(self.progress_card)
        h_center.addStretch()
        layout.addLayout(h_center)

        # Error label (hidden by default)
        self.error_label = QLabel()
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #f44747; font-size: 14px;")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        layout.addStretch()

    def start_analysis(self, analyzer, image_path: str):
        """Begin analysis in a background thread."""
        self.error_label.hide()
        self.title_label.setText("Analyzing Image...")
        self.progress_card.set_progress(0, "Starting...")

        # Show thumbnail — keep a reference to the numpy array
        try:
            img = cv2.imread(image_path)
            if img is not None:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._thumb_data = rgb.copy()  # prevent GC
                h, w, ch = self._thumb_data.shape
                bytes_per_line = ch * w
                qimg = QImage(self._thumb_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(
                    300, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.thumb_label.setPixmap(pixmap)
            else:
                self.thumb_label.setText("Could not load preview")
        except Exception:
            pass

        # Start worker
        self._worker = AnalysisWorker(analyzer, image_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, value: int, text: str):
        self.progress_card.set_progress(value, text)

    def _on_finished(self, result: dict):
        self.analysis_complete.emit(result)

    def _on_error(self, msg: str):
        self.error_label.setText(f"Error: {msg}")
        self.error_label.show()
        self.title_label.setText("Analysis Failed")
        self.progress_card.set_progress(0, "Failed")
