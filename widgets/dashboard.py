"""
widgets/dashboard.py — Home screen: batch ID, start test, upload, history.
"""
import csv
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QFileDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont

from config import (
    LOG_DIR, FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_NORMAL,
    CARD_COLOR, ACCENT_BLUE, TEXT_COLOR, MUTED_COLOR, ACCENT_GREEN,
)
from widgets.common import BigButton, Card


class DashboardScreen(QWidget):
    """Main dashboard — entry point for tests."""

    start_test_clicked = pyqtSignal()
    upload_clicked = pyqtSignal(str)        # emits file path
    history_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(6)

        # Title
        title = QLabel("Proppant QC System v2.0")
        title.setFont(QFont("Segoe UI", FONT_SIZE_TITLE, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {ACCENT_BLUE}; margin-top: 4px;")
        outer.addWidget(title)

        subtitle = QLabel("Automated proppant quality control with AI segmentation")
        subtitle.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet(f"color: {MUTED_COLOR};")
        outer.addWidget(subtitle)

        # Center content area
        center = QHBoxLayout()
        center.setSpacing(10)

        # ── Left column: batch ID + action buttons ──
        left_card = Card()
        left_layout = QVBoxLayout(left_card)
        left_layout.setSpacing(8)

        batch_label = QLabel("Batch ID")
        batch_label.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        left_layout.addWidget(batch_label)

        self.batch_input = QLineEdit()
        self.batch_input.setPlaceholderText("Enter batch ID (e.g., BATCH-2026-001)")
        self.batch_input.setMinimumHeight(34)
        left_layout.addWidget(self.batch_input)

        self.btn_start = BigButton("Start New Test", "primary")
        self.btn_start.clicked.connect(self.start_test_clicked.emit)
        left_layout.addWidget(self.btn_start)

        self.btn_upload = BigButton("Upload Image", "outlined")
        self.btn_upload.clicked.connect(self._on_upload)
        left_layout.addWidget(self.btn_upload)

        self.btn_history = BigButton("Test History", "small")
        self.btn_history.clicked.connect(self.history_clicked.emit)
        left_layout.addWidget(self.btn_history)

        left_layout.addStretch()
        center.addWidget(left_card, stretch=1)

        # ── Right column: status + last test summary ──
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        # Model status card
        status_card = Card()
        status_layout = QVBoxLayout(status_card)
        status_layout.setSpacing(8)

        status_title = QLabel("System Status")
        status_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        status_layout.addWidget(status_title)

        self.model_status_label = QLabel("Model: Loading...")
        self.model_status_label.setStyleSheet(f"color: {MUTED_COLOR};")
        status_layout.addWidget(self.model_status_label)

        self.device_label = QLabel("Device: detecting...")
        self.device_label.setStyleSheet(f"color: {MUTED_COLOR};")
        status_layout.addWidget(self.device_label)

        right_col.addWidget(status_card)

        # Last test summary card
        summary_card = Card()
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setSpacing(8)

        summary_title = QLabel("Last Test")
        summary_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        summary_layout.addWidget(summary_title)

        self.last_test_label = QLabel("No tests yet")
        self.last_test_label.setStyleSheet(f"color: {MUTED_COLOR};")
        self.last_test_label.setWordWrap(True)
        summary_layout.addWidget(self.last_test_label)

        right_col.addWidget(summary_card)

        right_col.addStretch()

        center.addLayout(right_col, stretch=1)
        outer.addLayout(center, stretch=1)

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image for Analysis",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.upload_clicked.emit(path)

    def get_batch_id(self) -> str:
        return self.batch_input.text().strip()

    def set_model_status(self, model_name: str, device: str):
        self.model_status_label.setText(f"Model: {model_name}")
        self.model_status_label.setStyleSheet(f"color: {ACCENT_GREEN};")
        self.device_label.setText(f"Device: {device}")
        self.device_label.setStyleSheet(f"color: {ACCENT_GREEN};")

    def set_model_error(self, msg: str):
        self.model_status_label.setText(f"Model: {msg}")
        self.model_status_label.setStyleSheet("color: #f44747;")

    def refresh_last_test(self):
        """Read last row from results.csv and display summary."""
        csv_path = LOG_DIR / "results.csv"
        if not csv_path.exists():
            self.last_test_label.setText("No tests yet")
            return
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                self.last_test_label.setText("No tests yet")
                return
            header = rows[0]
            last = rows[-1]
            data = dict(zip(header, last))
            text = (
                f"Image: {data.get('image_name', '?')}\n"
                f"Verdict: {data.get('verdict', '?')}\n"
                f"Particles: {data.get('total_particles', '?')}\n"
                f"40/70: {data.get('pct_proppant_40_70', '?')}%  "
                f"20/40: {data.get('pct_proppant_20_40', '?')}%\n"
                f"Time: {data.get('timestamp', '?')}"
            )
            self.last_test_label.setText(text)
            self.last_test_label.setStyleSheet(f"color: {TEXT_COLOR};")
        except Exception:
            self.last_test_label.setText("Error reading history")
