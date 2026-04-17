"""
widgets/calibration.py — Camera calibration dialog.

User measures a reference object in the app and enters its known mm size.
Saves PIXELS_PER_MM to calibration.json in the project root.
"""
import json
from pathlib import Path

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFrame,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from config import PROJECT_ROOT, FONT_SIZE_NORMAL, FONT_SIZE_LARGE, FONT_SIZE_TITLE
from config import BG_COLOR, CARD_COLOR, ACCENT_BLUE, ACCENT_GREEN, TEXT_COLOR, MUTED_COLOR, ACCENT_RED

CALIBRATION_FILE = PROJECT_ROOT / "calibration.json"


def load_calibration() -> float | None:
    """Load saved PIXELS_PER_MM from calibration.json. Returns None if not set."""
    if CALIBRATION_FILE.exists():
        try:
            data = json.loads(CALIBRATION_FILE.read_text())
            val = float(data.get("pixels_per_mm", 0))
            return val if val > 0 else None
        except Exception:
            return None
    return None


def save_calibration(pixels_per_mm: float):
    """Save PIXELS_PER_MM to calibration.json."""
    CALIBRATION_FILE.write_text(json.dumps({"pixels_per_mm": round(pixels_per_mm, 4)}, indent=2))


class CalibrationDialog(QDialog):
    """
    Simple calibration dialog.

    Workflow:
      1. Place a reference object (ruler, coin, or known bead) under the camera.
      2. Capture an image and note its diameter in pixels (shown in the overlay).
      3. Enter the pixel diameter and actual mm size here.
      4. Hit Save — PIXELS_PER_MM is calculated and saved permanently.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera Calibration")
        self.setMinimumWidth(420)
        self.setStyleSheet(f"""
            QDialog {{ background: {BG_COLOR}; color: {TEXT_COLOR}; }}
            QLabel  {{ color: {TEXT_COLOR}; }}
            QLineEdit {{
                background: {CARD_COLOR}; color: {TEXT_COLOR};
                border: 1px solid #444; border-radius: 4px;
                padding: 6px; font-size: {FONT_SIZE_NORMAL}pt;
            }}
            QPushButton {{
                border-radius: 4px; padding: 8px 16px;
                font-size: {FONT_SIZE_NORMAL}pt; font-weight: bold;
            }}
        """)
        self._build_ui()
        self._load_existing()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(20, 20, 20, 20)

        title = QLabel("Camera Calibration")
        title.setFont(QFont("Segoe UI", FONT_SIZE_TITLE, QFont.Bold))
        title.setStyleSheet(f"color: {ACCENT_BLUE};")
        layout.addWidget(title)

        instructions = QLabel(
            "Place a reference object under the camera and capture an image.\n"
            "Use the segmentation overlay to read the particle diameter in pixels,\n"
            "or measure a known object (ruler, coin) directly in the image."
        )
        instructions.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL))
        instructions.setStyleSheet(f"color: {MUTED_COLOR};")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)

        # Pixel input
        px_label = QLabel("Reference object diameter  (pixels)")
        px_label.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        layout.addWidget(px_label)

        self.px_input = QLineEdit()
        self.px_input.setPlaceholderText("e.g.  95  (read from segmentation overlay)")
        layout.addWidget(self.px_input)

        # MM input
        mm_label = QLabel("Reference object diameter  (mm)")
        mm_label.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        layout.addWidget(mm_label)

        self.mm_input = QLineEdit()
        self.mm_input.setPlaceholderText("e.g.  0.35  (actual physical size)")
        layout.addWidget(self.mm_input)

        # Result preview
        self.result_label = QLabel("")
        self.result_label.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.result_label)

        self.px_input.textChanged.connect(self._update_preview)
        self.mm_input.textChanged.connect(self._update_preview)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_save = QPushButton("Save Calibration")
        self.btn_save.setStyleSheet(f"background: {ACCENT_BLUE}; color: white;")
        self.btn_save.clicked.connect(self._on_save)
        btn_row.addWidget(self.btn_save)

        btn_clear = QPushButton("Clear")
        btn_clear.setStyleSheet(f"background: #444; color: {TEXT_COLOR};")
        btn_clear.clicked.connect(self._on_clear)
        btn_row.addWidget(btn_clear)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet(f"background: #333; color: {MUTED_COLOR};")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        layout.addLayout(btn_row)

    def _load_existing(self):
        val = load_calibration()
        if val:
            self.result_label.setText(f"Current: {val:.4f} px/mm")
            self.result_label.setStyleSheet(f"color: {ACCENT_GREEN};")

    def _update_preview(self):
        try:
            px = float(self.px_input.text())
            mm = float(self.mm_input.text())
            if px > 0 and mm > 0:
                ppm = px / mm
                self.result_label.setText(f"→  {ppm:.4f} px/mm")
                self.result_label.setStyleSheet(f"color: {ACCENT_GREEN};")
                return
        except ValueError:
            pass
        self.result_label.setText("")

    def _on_save(self):
        try:
            px = float(self.px_input.text())
            mm = float(self.mm_input.text())
            if px <= 0 or mm <= 0:
                raise ValueError
        except ValueError:
            self.result_label.setText("Enter valid positive numbers.")
            self.result_label.setStyleSheet(f"color: {ACCENT_RED};")
            return

        ppm = px / mm
        save_calibration(ppm)
        self.result_label.setText(f"Saved: {ppm:.4f} px/mm")
        self.result_label.setStyleSheet(f"color: {ACCENT_GREEN};")
        self.accept()

    def _on_clear(self):
        if CALIBRATION_FILE.exists():
            CALIBRATION_FILE.unlink()
        self.px_input.clear()
        self.mm_input.clear()
        self.result_label.setText("Calibration cleared.")
        self.result_label.setStyleSheet(f"color: {MUTED_COLOR};")
