"""
widgets/place_tray.py — Live camera preview screen with capture button.
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSizePolicy,
    QFrame,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

from config import (
    CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    FONT_SIZE_LARGE, FONT_SIZE_NORMAL, MUTED_COLOR,
    VIBRATOR_GPIO_PIN,
)
from camera import CameraCapture
from vibrator import Vibrator
from widgets.common import BigButton, Card


class PlaceTrayScreen(QWidget):
    """Live camera feed with capture functionality."""

    captured = pyqtSignal(str)   # emits temp file path of captured image
    back_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = CameraCapture()
        self.vibrator = Vibrator(VIBRATOR_GPIO_PIN)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        self._last_frame = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Top bar: title + camera selector
        top = QHBoxLayout()

        title = QLabel("Position Tray Under Camera")
        title.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        title.setStyleSheet(f"color: #e2e8f0;")
        top.addWidget(title)

        top.addStretch()

        cam_label = QLabel("Camera:")
        cam_label.setStyleSheet(f"color: {MUTED_COLOR};")
        top.addWidget(cam_label)

        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(150)
        self.cam_combo.currentIndexChanged.connect(self._on_camera_changed)
        top.addWidget(self.cam_combo)

        layout.addLayout(top)

        # Camera feed
        self.feed_label = QLabel("Starting camera...")
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setStyleSheet(
            "background-color: #242428; border: 1px solid #404040; border-radius: 4px; color: #a0a0a0; font-size: 16px;"
        )
        self.feed_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.feed_label.setMinimumHeight(280)
        layout.addWidget(self.feed_label, stretch=1)

        # Status text
        self.status_label = QLabel("Adjust the tray position, then press Capture")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL))
        self.status_label.setStyleSheet(f"color: {MUTED_COLOR};")
        layout.addWidget(self.status_label)

        # Vibrator panel
        vib_panel = QFrame()
        vib_panel.setStyleSheet(
            "QFrame { background: #242428; border: 1px solid #404040; border-radius: 6px; }"
        )
        vib_row = QHBoxLayout(vib_panel)
        vib_row.setContentsMargins(12, 6, 12, 6)
        vib_row.setSpacing(10)

        vib_icon = QLabel("⚡")
        vib_icon.setFont(QFont("Segoe UI", FONT_SIZE_LARGE))
        vib_row.addWidget(vib_icon)

        vib_title = QLabel("Tray Vibrator")
        vib_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        vib_title.setStyleSheet("color: #e2e8f0; border: none; background: transparent;")
        vib_row.addWidget(vib_title)

        self.vib_status = QLabel("STOPPED")
        self.vib_status.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        self.vib_status.setAlignment(Qt.AlignCenter)
        self.vib_status.setFixedWidth(90)
        self._set_vib_status_style(False)
        vib_row.addWidget(self.vib_status)

        vib_row.addStretch()

        self.btn_vib_start = BigButton("Start", "primary")
        self.btn_vib_start.setFixedHeight(38)
        self.btn_vib_start.clicked.connect(self._on_vib_start)
        vib_row.addWidget(self.btn_vib_start)

        self.btn_vib_restart = BigButton("Restart", "outlined")
        self.btn_vib_restart.setFixedHeight(38)
        self.btn_vib_restart.setEnabled(False)
        self.btn_vib_restart.clicked.connect(self._on_vib_restart)
        vib_row.addWidget(self.btn_vib_restart)

        self.btn_vib_stop = BigButton("Stop", "danger")
        self.btn_vib_stop.setFixedHeight(38)
        self.btn_vib_stop.setEnabled(False)
        self.btn_vib_stop.clicked.connect(self._on_vib_stop)
        vib_row.addWidget(self.btn_vib_stop)

        layout.addWidget(vib_panel)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.btn_back = BigButton("Back", "small")
        self.btn_back.clicked.connect(self.back_clicked.emit)
        btn_row.addWidget(self.btn_back)

        btn_row.addStretch()

        self.btn_capture = BigButton("Capture", "success")
        self.btn_capture.clicked.connect(self._on_capture)
        btn_row.addWidget(self.btn_capture)

        layout.addLayout(btn_row)

    def start_camera(self):
        """Detect cameras, open default, start preview timer."""
        self.btn_capture.setEnabled(False)
        self.feed_label.setText("Detecting cameras...")

        # Detect available cameras
        available = CameraCapture.list_available(max_check=5)
        self.cam_combo.blockSignals(True)
        self.cam_combo.clear()
        if not available:
            self.cam_combo.addItem("No camera found")
            self.feed_label.setText(
                "No camera detected.\n\n"
                "Connect a USB camera and try again,\n"
                "or go back and use 'Upload Image' instead."
            )
            self.status_label.setText("No camera available")
            self.cam_combo.blockSignals(False)
            return

        for idx in available:
            self.cam_combo.addItem(f"Camera {idx}", idx)
        self.cam_combo.blockSignals(False)

        # Open first camera
        cam_idx = available[0]
        if self.camera.open(cam_idx, CAMERA_WIDTH, CAMERA_HEIGHT):
            self.btn_capture.setEnabled(True)
            self.status_label.setText("Adjust the tray position, then press Capture")
            self.timer.start(33)  # ~30 fps
        else:
            self.feed_label.setText(f"Failed to open Camera {cam_idx}")

    def stop_camera(self):
        """Stop preview, release camera, and ensure vibrator is off."""
        self.timer.stop()
        self.camera.release()
        self._last_frame = None
        self._on_vib_stop()

    # ── Vibrator controls ──────────────────────────────────────────

    def _set_vib_status_style(self, running: bool):
        if running:
            self.vib_status.setText("RUNNING")
            self.vib_status.setStyleSheet(
                "color: #3ddc84; border: 1px solid #1a6040; background: #0d1f15;"
                "border-radius: 4px; padding: 2px 6px;"
            )
        else:
            self.vib_status.setText("STOPPED")
            self.vib_status.setStyleSheet(
                "color: #a0a0a0; border: 1px solid #404040; background: #1a1a1e;"
                "border-radius: 4px; padding: 2px 6px;"
            )

    def _on_vib_start(self):
        self.vibrator.start()
        self._set_vib_status_style(True)
        self.btn_vib_start.setEnabled(False)
        self.btn_vib_restart.setEnabled(True)
        self.btn_vib_stop.setEnabled(True)
        self.status_label.setText("Vibrator running — check sample distribution, then Capture or Restart")

    def _on_vib_restart(self):
        self.vibrator.restart()
        self._set_vib_status_style(True)
        self.status_label.setText("Vibrator restarted — check sample distribution again")

    def _on_vib_stop(self):
        self.vibrator.stop()
        self._set_vib_status_style(False)
        self.btn_vib_start.setEnabled(True)
        self.btn_vib_restart.setEnabled(False)
        self.btn_vib_stop.setEnabled(False)
        if self.timer.isActive():
            self.status_label.setText("Adjust the tray position, then press Capture")

    def _on_camera_changed(self, index):
        cam_idx = self.cam_combo.currentData()
        if cam_idx is None:
            return
        self.timer.stop()
        self.camera.release()
        if self.camera.open(cam_idx, CAMERA_WIDTH, CAMERA_HEIGHT):
            self.btn_capture.setEnabled(True)
            self.timer.start(33)
        else:
            self.feed_label.setText(f"Failed to open Camera {cam_idx}")
            self.btn_capture.setEnabled(False)

    def _update_frame(self):
        frame = self.camera.read_frame()
        if frame is None:
            return
        self._last_frame = frame

        # Convert BGR -> RGB -> QImage -> QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit label
        label_size = self.feed_label.size()
        pixmap = QPixmap.fromImage(qimg).scaled(
            label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.feed_label.setPixmap(pixmap)

    def _on_capture(self):
        if self._last_frame is None:
            return

        # Save frame to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, self._last_frame)
        tmp.close()

        self.stop_camera()
        self.status_label.setText("Image captured!")
        self.captured.emit(tmp.name)
