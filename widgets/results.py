"""
widgets/results.py — Full results screen with overlay, composition, SWE checks, exports.
"""
import io
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFileDialog, QSizePolicy, QFrame, QGridLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor

from config import (
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_NORMAL,
    ACCENT_GREEN, ACCENT_RED, ACCENT_BLUE, TEXT_COLOR, MUTED_COLOR,
    CARD_COLOR, BG_COLOR,
)
from widgets.common import BigButton, Card, StatusPill

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class CompositionBar(QWidget):
    """Single horizontal bar showing a percentage with label."""

    def __init__(self, name: str, pct: float, count: int, color: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.pct = pct
        self.count = count
        self.bar_color = QColor(color)
        self.setFixedHeight(22)
        self.setMinimumWidth(120)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        # Background
        p.setBrush(QColor("#333333"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, w, h, 6, 6)

        # Fill bar
        fill_w = max(2, int(w * self.pct / 100.0))
        p.setBrush(self.bar_color)
        p.drawRoundedRect(0, 0, fill_w, h, 6, 6)

        # Text
        p.setPen(QColor("white"))
        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        label = f"  {self.name}: {self.pct:.1f}% ({self.count})"
        p.drawText(0, 0, w - 8, h, Qt.AlignVCenter | Qt.AlignLeft, label)
        p.end()


class ResultsScreen(QWidget):
    """Full results display with overlay, composition, verdict, SWE checks, exports."""

    new_test_clicked = pyqtSignal()
    home_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result = None
        self._log_paths = None
        self._build_ui()

    def _build_ui(self):
        # Scroll area wrapper
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(8, 4, 8, 6)
        self.main_layout.setSpacing(6)

        # Top bar
        top = QHBoxLayout()
        self.btn_back = BigButton("< Back", "small")
        self.btn_back.clicked.connect(self.home_clicked.emit)
        top.addWidget(self.btn_back)

        self.run_title = QLabel("Results")
        self.run_title.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        self.run_title.setStyleSheet("color: white;")
        top.addWidget(self.run_title)

        top.addStretch()

        self.btn_home = BigButton("Home", "small")
        self.btn_home.clicked.connect(self.home_clicked.emit)
        top.addWidget(self.btn_home)

        self.btn_new = BigButton("New Test", "primary")
        self.btn_new.clicked.connect(self.new_test_clicked.emit)
        top.addWidget(self.btn_new)

        self.main_layout.addLayout(top)

        # Verdict pill (large, centered)
        verdict_row = QHBoxLayout()
        verdict_row.addStretch()
        self.verdict_pill = StatusPill("—", "muted")
        self.verdict_pill.setMinimumWidth(200)
        verdict_row.addWidget(self.verdict_pill)

        self.confidence_label = QLabel("")
        self.confidence_label.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL))
        self.confidence_label.setStyleSheet(f"color: {TEXT_COLOR};")
        verdict_row.addWidget(self.confidence_label)
        verdict_row.addStretch()
        self.main_layout.addLayout(verdict_row)

        # Two-column area: overlay+chart on left, data on right
        columns = QHBoxLayout()
        columns.setSpacing(8)

        # ── Left column: overlay image + pie chart ──
        left = QVBoxLayout()
        left.setSpacing(12)

        overlay_card = Card()
        overlay_layout = QVBoxLayout(overlay_card)
        overlay_title = QLabel("Segmentation Overlay")
        overlay_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        overlay_layout.addWidget(overlay_title)

        self.overlay_label = QLabel("No overlay")
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setMinimumHeight(140)
        self.overlay_label.setStyleSheet("background: #111; border-radius: 6px;")
        overlay_layout.addWidget(self.overlay_label)
        left.addWidget(overlay_card)

        # Pie chart
        chart_card = Card()
        chart_layout = QVBoxLayout(chart_card)
        chart_title = QLabel("Composition Chart")
        chart_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        chart_layout.addWidget(chart_title)

        self.chart_label = QLabel("")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setMinimumHeight(100)
        chart_layout.addWidget(self.chart_label)
        left.addWidget(chart_card)

        columns.addLayout(left, stretch=1)

        # ── Right column: composition bars + SWE + metrics ──
        right = QVBoxLayout()
        right.setSpacing(12)

        # Composition bars card
        comp_card = Card()
        self.comp_layout = QVBoxLayout(comp_card)
        comp_title = QLabel("Composition")
        comp_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        self.comp_layout.addWidget(comp_title)
        self.comp_bars_container = QVBoxLayout()
        self.comp_layout.addLayout(self.comp_bars_container)
        self.particles_label = QLabel("")
        self.particles_label.setStyleSheet(f"color: {MUTED_COLOR};")
        self.comp_layout.addWidget(self.particles_label)
        right.addWidget(comp_card)

        # SWE checks card
        swe_card = Card()
        self.swe_layout = QVBoxLayout(swe_card)
        swe_title = QLabel("SWE Spec Compliance")
        swe_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        self.swe_layout.addWidget(swe_title)
        self.swe_checks_container = QVBoxLayout()
        self.swe_layout.addLayout(self.swe_checks_container)
        right.addWidget(swe_card)

        # Metrics card
        metrics_card = Card()
        metrics_layout = QHBoxLayout(metrics_card)

        self.blur_label = QLabel("Blur: —")
        self.blur_label.setStyleSheet(f"color: {TEXT_COLOR};")
        metrics_layout.addWidget(self.blur_label)

        self.time_label = QLabel("Time: —")
        self.time_label.setStyleSheet(f"color: {TEXT_COLOR};")
        metrics_layout.addWidget(self.time_label)

        self.reason_label = QLabel("")
        self.reason_label.setStyleSheet(f"color: {MUTED_COLOR}; font-size: 12px;")
        self.reason_label.setWordWrap(True)
        metrics_layout.addWidget(self.reason_label, stretch=1)

        right.addWidget(metrics_card)
        right.addStretch()

        columns.addLayout(right, stretch=1)
        self.main_layout.addLayout(columns, stretch=1)

        # Export buttons row
        export_row = QHBoxLayout()
        export_row.setSpacing(6)

        self.btn_json = BigButton("Export JSON", "outlined")
        self.btn_json.clicked.connect(self._export_json)
        export_row.addWidget(self.btn_json)

        self.btn_csv = BigButton("Export CSV", "outlined")
        self.btn_csv.clicked.connect(self._export_csv)
        export_row.addWidget(self.btn_csv)

        self.btn_overlay = BigButton("Save Overlay", "outlined")
        self.btn_overlay.clicked.connect(self._export_overlay)
        export_row.addWidget(self.btn_overlay)

        self.main_layout.addLayout(export_row)

        scroll.setWidget(container)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def show_result(self, result: dict, log_paths: dict, run_id: str = ""):
        """Populate all fields with analysis result data."""
        self._result = result
        self._log_paths = log_paths

        self.run_title.setText(f"Results — {run_id}" if run_id else "Results")

        # Verdict
        verdict = result.get("verdict", "?")
        self.verdict_pill.set_pass_fail(verdict)
        conf = result.get("avg_confidence", 0)
        self.confidence_label.setText(f"Confidence: {conf:.1f}%")

        # Overlay image
        overlay = result.get("overlay")
        if overlay is not None:
            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                320, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.overlay_label.setPixmap(pixmap)
        else:
            self.overlay_label.setText("No overlay available")

        # Composition bars
        self._clear_layout(self.comp_bars_container)
        comp = result.get("composition", {})
        color_map = {
            "proppant_40_70": "#00cc00",
            "proppant_20_40": "#ff8800",
            "sand": "#ff3333",
        }
        for name in ["proppant_40_70", "proppant_20_40", "sand"]:
            data = comp.get(name, {})
            pct = data.get("percentage", 0)
            cnt = data.get("count", 0)
            bar = CompositionBar(name, pct, cnt, color_map.get(name, "#888"))
            self.comp_bars_container.addWidget(bar)

        total = result.get("total_particles", 0)
        self.particles_label.setText(f"Total: {total} particles")

        # Pie chart
        self._render_pie_chart(comp)

        # SWE checks
        self._clear_layout(self.swe_checks_container)
        swe = result.get("swe_checks", {})

        checks = [
            (
                f">=90% classified ({swe.get('classified_rate_pct', 0):.0f}%)",
                swe.get("classified_pass", False),
            ),
            (
                f"±10 wt% sieve acc. ({swe.get('mean_size_error_pct', 0):.1f}%)",
                swe.get("size_error_pass", False),
            ),
            (
                f"<=20 sec ({swe.get('processing_time_sec', 0):.1f}s)",
                swe.get("processing_time_pass", False),
            ),
        ]
        for text, passed in checks:
            icon = "+" if passed else "X"
            color = ACCENT_GREEN if passed else ACCENT_RED
            lbl = QLabel(f"  {icon}  {text}")
            lbl.setFont(QFont("Consolas", FONT_SIZE_NORMAL))
            lbl.setStyleSheet(f"color: {color};")
            self.swe_checks_container.addWidget(lbl)

        # Metrics
        self.blur_label.setText(f"Blur: {result.get('blur_score', 0):.0f}")
        proc = result.get("processing_time_sec", 0)
        self.time_label.setText(f"Time: {proc:.1f}s")
        self.reason_label.setText(result.get("reason", ""))

    def _render_pie_chart(self, composition: dict):
        if not HAS_MATPLOTLIB or not composition:
            self.chart_label.setText("matplotlib not available" if not HAS_MATPLOTLIB else "")
            return

        labels, sizes, colors = [], [], []
        color_map = {
            "proppant_40_70": "#00cc00",
            "proppant_20_40": "#ff8800",
            "sand": "#ff3333",
        }
        for name in ["proppant_40_70", "proppant_20_40", "sand"]:
            pct = composition.get(name, {}).get("percentage", 0)
            if pct > 0:
                labels.append(f"{name}\n{pct:.1f}%")
                sizes.append(pct)
                colors.append(color_map.get(name, "#888"))

        if not sizes:
            return

        fig, ax = plt.subplots(figsize=(2.4, 1.6), facecolor=BG_COLOR)
        ax.pie(
            sizes, labels=labels, colors=colors, startangle=90,
            textprops={"color": TEXT_COLOR, "fontsize": 8},
            wedgeprops={"edgecolor": BG_COLOR, "linewidth": 1.5},
        )
        ax.set_title("Particle Composition", color="#cccccc", fontsize=10, pad=8)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                    facecolor=BG_COLOR, edgecolor="none")
        plt.close(fig)
        buf.seek(0)

        data = buf.getvalue()
        qimg = QImage()
        qimg.loadFromData(data)
        self.chart_label.setPixmap(QPixmap.fromImage(qimg))

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    # ── Export methods ──

    def _export_json(self):
        if not self._log_paths:
            return
        src = self._log_paths.get("json_path", "")
        if not src or not Path(src).exists():
            return
        dst, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON (*.json)")
        if dst:
            shutil.copy2(src, dst)

    def _export_csv(self):
        if not self._log_paths:
            return
        src = self._log_paths.get("csv_path", "")
        if not src or not Path(src).exists():
            return
        dst, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
        if dst:
            shutil.copy2(src, dst)

    def _export_overlay(self):
        if not self._log_paths:
            return
        src = self._log_paths.get("overlay_path", "")
        if not src or not Path(src).exists():
            return
        dst, _ = QFileDialog.getSaveFileName(self, "Save Overlay", "", "JPEG (*.jpg)")
        if dst:
            shutil.copy2(src, dst)
