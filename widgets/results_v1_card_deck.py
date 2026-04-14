"""
widgets/results.py — Card-deck results viewer (touch-optimized for Jetson).

5 full-screen cards navigated with left/right arrows — no scrolling.
  Card 1: Segmentation Overlay
  Card 2: Verdict & Composition
  Card 3: SWE Spec Compliance
  Card 4: Sieve Distribution Chart
  Card 5: Summary & Actions (New Test / Home / Export)
"""
import io
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QSizePolicy, QStackedWidget, QPushButton,
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
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Dark palette constants (match config) ────────────────────────────────────
_DARK_BG   = "#0f1117"
_DARK_CARD = "#1e2130"
_DARK_GRID = "#2d3148"
_DARK_TEXT = "#e2e8f0"


# ── Composition bar widget ────────────────────────────────────────────────────

class CompositionBar(QWidget):
    """Horizontal bar showing a percentage with label — touch-sized."""

    def __init__(self, name: str, pct: float, count: int, color: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.pct = pct
        self.count = count
        self.bar_color = QColor(color)
        self.setFixedHeight(30)
        self.setMinimumWidth(120)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setBrush(QColor("#252840"))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, 0, w, h, 8, 8)

        fill_w = max(4, int(w * self.pct / 100.0))
        p.setBrush(self.bar_color)
        p.drawRoundedRect(0, 0, fill_w, h, 8, 8)

        p.setPen(QColor("#e2e8f0"))
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        label = f"  {self.name}:  {self.pct:.1f}%  ({self.count})"
        p.drawText(0, 0, w - 8, h, Qt.AlignVCenter | Qt.AlignLeft, label)
        p.end()


# ── Arrow button ──────────────────────────────────────────────────────────────

class ArrowButton(QPushButton):
    """Large touch-friendly arrow for card navigation."""

    def __init__(self, direction: str, parent=None):
        super().__init__("‹" if direction == "left" else "›", parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(52)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: #1e2130;
                color: #94a3b8;
                border: 1px solid #2d3148;
                border-radius: 8px;
                font-size: 32px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #252840;
                color: {ACCENT_BLUE};
                border-color: {ACCENT_BLUE};
            }}
            QPushButton:pressed {{
                background-color: #2d3148;
            }}
            QPushButton:disabled {{
                color: #2d3148;
                border-color: #1e2130;
                background-color: #0f1117;
            }}
        """)


# ── Individual cards ──────────────────────────────────────────────────────────

class OverlayCard(QWidget):
    """Card 1 — Segmentation overlay image."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        self.img_label = QLabel("No image")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet(
            "background: #252840; border-radius: 10px; color: #94a3b8;"
        )
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.img_label, stretch=1)

        self.info_label = QLabel("")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(f"color: {MUTED_COLOR}; font-size: 12px;")
        layout.addWidget(self.info_label)

    def load(self, overlay: np.ndarray, total: int, image_name: str):
        if overlay is not None:
            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
            self._scale_image()
        self.info_label.setText(f"{image_name}  |  {total} particles detected")

    def _scale_image(self):
        if hasattr(self, "_pixmap"):
            scaled = self._pixmap.scaled(
                self.img_label.width() - 8,
                self.img_label.height() - 8,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_image()


class VerdictCard(QWidget):
    """Card 2 — Verdict pill + composition bars."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setSpacing(14)

        layout.addStretch()

        # Verdict pill
        pill_row = QHBoxLayout()
        pill_row.addStretch()
        self.verdict_pill = StatusPill("—", "muted")
        self.verdict_pill.setMinimumWidth(240)
        self.verdict_pill.setMinimumHeight(54)
        self.verdict_pill.setFont(QFont("Segoe UI", FONT_SIZE_TITLE + 2, QFont.Bold))
        pill_row.addWidget(self.verdict_pill)
        pill_row.addStretch()
        layout.addLayout(pill_row)

        # Confidence
        self.conf_label = QLabel("")
        self.conf_label.setAlignment(Qt.AlignCenter)
        self.conf_label.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL))
        self.conf_label.setStyleSheet(f"color: {MUTED_COLOR};")
        layout.addWidget(self.conf_label)

        # Divider
        div = QLabel()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #2d3148;")
        layout.addWidget(div)

        # Composition bars
        comp_title = QLabel("Particle Composition")
        comp_title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        comp_title.setStyleSheet(f"color: {TEXT_COLOR};")
        layout.addWidget(comp_title)

        self.bar_4070 = CompositionBar("40/70", 0, 0, "#00cc44")
        self.bar_2040 = CompositionBar("20/40", 0, 0, "#ff8800")
        layout.addWidget(self.bar_4070)
        layout.addWidget(self.bar_2040)

        self.total_label = QLabel("")
        self.total_label.setStyleSheet(f"color: {MUTED_COLOR}; font-size: 12px;")
        layout.addWidget(self.total_label)

        # Reason
        self.reason_label = QLabel("")
        self.reason_label.setWordWrap(True)
        self.reason_label.setStyleSheet(f"color: {MUTED_COLOR}; font-size: 12px;")
        layout.addWidget(self.reason_label)

        layout.addStretch()

    def load(self, verdict: str, composition: dict, confidence: float,
             total: int, reason: str):
        self.verdict_pill.set_pass_fail(verdict)
        self.conf_label.setText(f"Confidence: {confidence:.1f}%")

        pct_4070 = composition.get("proppant_40_70", {}).get("percentage", 0)
        cnt_4070 = composition.get("proppant_40_70", {}).get("count", 0)
        pct_2040 = composition.get("proppant_20_40", {}).get("percentage", 0)
        cnt_2040 = composition.get("proppant_20_40", {}).get("count", 0)

        self.bar_4070.pct   = pct_4070
        self.bar_4070.count = cnt_4070
        self.bar_2040.pct   = pct_2040
        self.bar_2040.count = cnt_2040
        self.bar_4070.update()
        self.bar_2040.update()

        self.total_label.setText(f"Total: {total} particles")
        self.reason_label.setText(reason)


class SWECard(QWidget):
    """Card 3 — SWE Spec Compliance checks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(10)

        layout.addStretch()

        title = QLabel("Spec Compliance")
        title.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_COLOR};")
        layout.addWidget(title)

        div = QLabel()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #2d3148; margin: 4px 0;")
        layout.addWidget(div)

        self.check_labels = []
        for _ in range(3):
            row = QWidget()
            row.setStyleSheet("background: #1e2130; border-radius: 10px;")
            row.setFixedHeight(58)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(16, 0, 16, 0)

            icon = QLabel("—")
            icon.setFixedWidth(36)
            icon.setFont(QFont("Segoe UI", 22, QFont.Bold))
            icon.setAlignment(Qt.AlignCenter)

            text = QLabel("—")
            text.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL))
            text.setStyleSheet(f"color: {TEXT_COLOR};")

            row_layout.addWidget(icon)
            row_layout.addWidget(text, stretch=1)
            layout.addWidget(row)
            self.check_labels.append((row, icon, text))

        layout.addStretch()

    def load(self, swe: dict):
        checks = [
            (
                f">= 90% classified  ({swe.get('classified_rate_pct', 0):.0f}%)",
                swe.get("classified_pass", False),
            ),
            (
                f"+/- 10 wt% size accuracy  ({swe.get('mean_size_error_pct', 0):.1f}%)",
                swe.get("size_error_pass", False),
            ),
            (
                f"<= 20 sec processing  ({swe.get('processing_time_sec', 0):.1f}s)",
                swe.get("processing_time_pass", True),
            ),
        ]
        for (row, icon, text), (label, passed) in zip(self.check_labels, checks):
            if passed:
                icon.setText("✓")
                icon.setStyleSheet(f"color: {ACCENT_GREEN};")
                row.setStyleSheet(
                    f"background: #0f2a1a; border: 1px solid {ACCENT_GREEN}; border-radius: 10px;"
                )
            else:
                icon.setText("✗")
                icon.setStyleSheet(f"color: {ACCENT_RED};")
                row.setStyleSheet(
                    f"background: #2a0f0f; border: 1px solid {ACCENT_RED}; border-radius: 10px;"
                )
            text.setText(label)


class SieveCard(QWidget):
    """Card 4 — Sieve distribution chart: Model vs Lab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title = QLabel("Sieve Distribution — Model vs Lab")
        title.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_COLOR};")
        layout.addWidget(title)

        self.chart_label = QLabel("")
        self.chart_label.setAlignment(Qt.AlignCenter)
        self.chart_label.setStyleSheet("background: #1e2130; border-radius: 8px;")
        self.chart_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.chart_label, stretch=1)

        self.note_label = QLabel("")
        self.note_label.setAlignment(Qt.AlignCenter)
        self.note_label.setStyleSheet(f"color: {MUTED_COLOR}; font-size: 11px;")
        layout.addWidget(self.note_label)

    def load(self, particles: list, verdict: str):
        self._render(particles, verdict)

    def _render(self, particles: list, verdict: str):
        if not HAS_MATPLOTLIB or not particles:
            self.chart_label.setText("No data" if particles else "matplotlib not available")
            return

        is_2040 = "20_40" in verdict

        if is_2040:
            class_id    = 1
            mesh_sizes  = [16, 20, 25, 30, 35, 40, 50]
            openings_mm = [1.18, 0.85, 0.71, 0.60, 0.50, 0.425, 0.30]
            d_min_px, d_max_px = 100.0, 200.0
            d_min_mm, d_max_mm = 0.42, 0.84
            lab_ref_key = "proppant_20_40"
        else:
            class_id    = 0
            mesh_sizes  = [30, 40, 50, 60, 70, 100]
            openings_mm = [0.60, 0.425, 0.30, 0.25, 0.212, 0.150]
            d_min_px, d_max_px = 70.0, 130.0
            d_min_mm, d_max_mm = 0.21, 0.42
            lab_ref_key = "proppant_40_70"

        # ── Model curve ──────────────────────────────────────────────────
        from inference_stardist import ProppantAnalyzer
        sieve_refs_full = ProppantAnalyzer._load_sieve_references_full()

        relevant = particles if "FAIL" in verdict else [
            p for p in particles if p["class_id"] == class_id
        ]
        if not relevant:
            self.chart_label.setText("No classified particles")
            return

        def px_to_mm(d_px):
            t = max(0.0, min(1.0, (d_px - d_min_px) / max(d_max_px - d_min_px, 1.0)))
            return d_min_mm + t * (d_max_mm - d_min_mm)

        diams_mm = [px_to_mm(p["diameter_px"]) for p in relevant]
        masses   = [d ** 3 for d in diams_mm]
        total_m  = sum(masses) or 1.0

        retained_pct = []
        for i, opening in enumerate(openings_mm):
            next_opening = openings_mm[i - 1] if i > 0 else 999.0
            retained = sum(
                m for d, m in zip(diams_mm, masses)
                if opening <= d < next_opening
            )
            retained_pct.append(retained / total_m * 100.0)

        cum_model, cumulative = [], 100.0
        for r in retained_pct:
            cumulative -= r
            cum_model.append(max(0.0, cumulative))

        # ── Lab curve ────────────────────────────────────────────────────
        lab_per_mesh = sieve_refs_full.get(lab_ref_key, {})
        has_lab = bool(lab_per_mesh)
        if has_lab:
            cum_lab, cumulative_lab = [], 100.0
            for mesh in mesh_sizes:
                cumulative_lab -= lab_per_mesh.get(mesh, 0.0)
                cum_lab.append(max(0.0, cumulative_lab))

        # ── Plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(4.8, 2.8), facecolor=_DARK_CARD)
        ax.set_facecolor(_DARK_BG)

        ax.plot(mesh_sizes, cum_model, color="#3b82f6", linewidth=2.5,
                marker="o", markersize=5, label="Model")
        if has_lab:
            ax.plot(mesh_sizes, cum_lab, color="#f97316", linewidth=2.5,
                    linestyle="--", marker="s", markersize=5, label="Lab sieve")

        ax.set_xlabel("Mesh size", color=_DARK_TEXT, fontsize=9)
        ax.set_ylabel("Cumulative passing %", color=_DARK_TEXT, fontsize=9)
        title_str = f"{'20/40' if is_2040 else '40/70'} Sieve Distribution"
        ax.set_title(title_str, color=_DARK_TEXT, fontsize=11, pad=8, fontweight="bold")
        ax.tick_params(colors=_DARK_TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(_DARK_GRID)
        ax.grid(True, color=_DARK_GRID, linewidth=0.7, linestyle="--")
        ax.set_ylim(0, 105)
        ax.legend(fontsize=9, facecolor=_DARK_CARD, edgecolor=_DARK_GRID,
                  labelcolor=_DARK_TEXT, loc="upper right")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=_DARK_CARD, edgecolor="none")
        plt.close(fig)
        buf.seek(0)

        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        self.chart_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.chart_label.width() - 4,
                self.chart_label.height() - 4,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        if not has_lab:
            self.note_label.setText("Lab line unavailable — check SIEVE_EXCEL_PATH in config.py")
        else:
            self.note_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-render on resize so chart fills available space
        if hasattr(self, "_last_particles"):
            self._render(self._last_particles, self._last_verdict)

    def load(self, particles: list, verdict: str):
        self._last_particles = particles
        self._last_verdict   = verdict
        self._render(particles, verdict)


class SummaryCard(QWidget):
    """Card 5 — Run summary + action buttons."""

    new_test_clicked = pyqtSignal()
    home_clicked     = pyqtSignal()
    export_clicked   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)

        layout.addStretch()

        title = QLabel("Test Summary")
        title.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"color: {TEXT_COLOR};")
        layout.addWidget(title)

        div = QLabel()
        div.setFixedHeight(1)
        div.setStyleSheet("background: #2d3148;")
        layout.addWidget(div)

        self.run_label    = QLabel("Run: —")
        self.image_label  = QLabel("Image: —")
        self.blur_label   = QLabel("Blur score: —")
        self.time_label   = QLabel("Processing time: —")

        for lbl in [self.run_label, self.image_label,
                    self.blur_label, self.time_label]:
            lbl.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL))
            lbl.setStyleSheet(f"color: {TEXT_COLOR};")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

        layout.addStretch()

        div2 = QLabel()
        div2.setFixedHeight(1)
        div2.setStyleSheet("background: #2d3148;")
        layout.addWidget(div2)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_home = BigButton("Home", "small")
        self.btn_home.clicked.connect(self.home_clicked.emit)
        btn_row.addWidget(self.btn_home)

        self.btn_export = BigButton("Export", "outlined")
        self.btn_export.clicked.connect(self.export_clicked.emit)
        btn_row.addWidget(self.btn_export)

        self.btn_new = BigButton("New Test", "primary")
        self.btn_new.clicked.connect(self.new_test_clicked.emit)
        btn_row.addWidget(self.btn_new)

        layout.addLayout(btn_row)

    def load(self, run_id: str, image_name: str, blur: float, proc_time: float):
        self.run_label.setText(f"Run:  {run_id}")
        self.image_label.setText(f"Image:  {image_name}")
        self.blur_label.setText(f"Blur score:  {blur:.0f}")
        self.time_label.setText(f"Processing time:  {proc_time:.1f} s")


# ── Dot indicator ─────────────────────────────────────────────────────────────

class DotIndicator(QWidget):
    """Row of N dots showing current page."""

    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.n = n
        self.current = 0
        self.setFixedHeight(20)

    def set_page(self, idx: int):
        self.current = idx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = 6
        gap = 16
        total_w = self.n * (r * 2) + (self.n - 1) * gap
        x = (self.width() - total_w) // 2
        y = self.height() // 2

        for i in range(self.n):
            cx = x + i * (r * 2 + gap) + r
            if i == self.current:
                p.setBrush(QColor(ACCENT_BLUE))
            else:
                p.setBrush(QColor("#2d3148"))
            p.setPen(Qt.NoPen)
            p.drawEllipse(cx - r, y - r, r * 2, r * 2)
        p.end()


# ── Main ResultsScreen ────────────────────────────────────────────────────────

CARD_NAMES = [
    "Segmentation Overlay",
    "Verdict & Composition",
    "Spec Compliance",
    "Sieve Distribution",
    "Summary",
]

NUM_CARDS = len(CARD_NAMES)


class ResultsScreen(QWidget):
    """Full results display as a swipeable card deck."""

    new_test_clicked = pyqtSignal()
    home_clicked     = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result    = None
        self._log_paths = None
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Top bar: Home button | card name | page counter ───────────
        top_bar = QWidget()
        top_bar.setFixedHeight(38)
        top_bar.setStyleSheet("background: #1e2130; border-bottom: 1px solid #2d3148;")
        top_row = QHBoxLayout(top_bar)
        top_row.setContentsMargins(8, 0, 8, 0)

        self.btn_home_top = QPushButton("Home")
        self.btn_home_top.setCursor(Qt.PointingHandCursor)
        self.btn_home_top.setFixedHeight(28)
        self.btn_home_top.clicked.connect(self.home_clicked.emit)
        self.btn_home_top.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {MUTED_COLOR};
                border: 1px solid #2d3148;
                border-radius: 6px;
                padding: 0 12px;
                font-size: 12px;
            }}
            QPushButton:hover {{ color: {ACCENT_BLUE}; border-color: {ACCENT_BLUE}; }}
        """)
        top_row.addWidget(self.btn_home_top)

        top_row.addStretch()

        self.card_name_label = QLabel(CARD_NAMES[0])
        self.card_name_label.setFont(QFont("Segoe UI", FONT_SIZE_NORMAL, QFont.Bold))
        self.card_name_label.setStyleSheet(f"color: {TEXT_COLOR};")
        self.card_name_label.setAlignment(Qt.AlignCenter)
        top_row.addWidget(self.card_name_label)

        top_row.addStretch()

        self.page_counter = QLabel(f"1 / {NUM_CARDS}")
        self.page_counter.setStyleSheet(f"color: {MUTED_COLOR}; font-size: 12px;")
        top_row.addWidget(self.page_counter)

        outer.addWidget(top_bar)

        # ── Middle: arrow + stacked cards + arrow ─────────────────────
        mid = QHBoxLayout()
        mid.setContentsMargins(4, 4, 4, 4)
        mid.setSpacing(4)

        self.btn_left = ArrowButton("left")
        self.btn_left.clicked.connect(self._prev_card)
        mid.addWidget(self.btn_left)

        # Card stack
        self.stack = QStackedWidget()
        self.card_overlay  = OverlayCard()
        self.card_verdict  = VerdictCard()
        self.card_swe      = SWECard()
        self.card_sieve    = SieveCard()
        self.card_summary  = SummaryCard()

        for card in [self.card_overlay, self.card_verdict, self.card_swe,
                     self.card_sieve, self.card_summary]:
            self.stack.addWidget(card)

        mid.addWidget(self.stack, stretch=1)

        self.btn_right = ArrowButton("right")
        self.btn_right.clicked.connect(self._next_card)
        mid.addWidget(self.btn_right)

        outer.addLayout(mid, stretch=1)

        # ── Bottom: dot indicator ──────────────────────────────────────
        self.dots = DotIndicator(NUM_CARDS)
        self.dots.setStyleSheet("background: #1e2130; border-top: 1px solid #2d3148;")
        self.dots.setFixedHeight(24)
        outer.addWidget(self.dots)

        # Signals from summary card
        self.card_summary.new_test_clicked.connect(self.new_test_clicked.emit)
        self.card_summary.home_clicked.connect(self.home_clicked.emit)
        self.card_summary.export_clicked.connect(self._export_dialog)

        self._update_nav()

    # ── Navigation ────────────────────────────────────────────────────

    def _go_to(self, idx: int):
        idx = max(0, min(NUM_CARDS - 1, idx))
        self.stack.setCurrentIndex(idx)
        self.card_name_label.setText(CARD_NAMES[idx])
        self.page_counter.setText(f"{idx + 1} / {NUM_CARDS}")
        self.dots.set_page(idx)
        self._update_nav()

    def _next_card(self):
        self._go_to(self.stack.currentIndex() + 1)

    def _prev_card(self):
        self._go_to(self.stack.currentIndex() - 1)

    def _update_nav(self):
        idx = self.stack.currentIndex()
        self.btn_left.setEnabled(idx > 0)
        self.btn_right.setEnabled(idx < NUM_CARDS - 1)

    # ── Load result ───────────────────────────────────────────────────

    def show_result(self, result: dict, log_paths: dict, run_id: str = ""):
        self._result    = result
        self._log_paths = log_paths

        comp      = result.get("composition", {})
        verdict   = result.get("verdict", "?")
        particles = result.get("particles", [])

        self.card_overlay.load(
            result.get("overlay"),
            result.get("total_particles", 0),
            result.get("image_name", ""),
        )
        self.card_verdict.load(
            verdict,
            comp,
            result.get("avg_confidence", 0),
            result.get("total_particles", 0),
            result.get("reason", ""),
        )
        self.card_swe.load(result.get("swe_checks", {}))
        self.card_sieve.load(particles, verdict)
        self.card_summary.load(
            run_id or "—",
            result.get("image_name", "—"),
            result.get("blur_score", 0),
            result.get("processing_time_sec", 0),
        )

        # Always start at card 1
        self._go_to(0)

    # ── Export ────────────────────────────────────────────────────────

    def _export_dialog(self):
        """Show a simple export options dialog."""
        if not self._log_paths:
            return
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background: #1e2130;
                color: {TEXT_COLOR};
                border: 1px solid #2d3148;
                border-radius: 6px;
                padding: 4px;
            }}
            QMenu::item {{ padding: 8px 20px; border-radius: 4px; }}
            QMenu::item:selected {{ background: {ACCENT_BLUE}; color: white; }}
        """)
        menu.addAction("Export JSON",    self._export_json)
        menu.addAction("Export CSV",     self._export_csv)
        menu.addAction("Save Overlay",   self._export_overlay)
        menu.exec_(self.card_summary.btn_export.mapToGlobal(
            self.card_summary.btn_export.rect().topLeft()
        ))

    def _export_json(self):
        src = (self._log_paths or {}).get("json_path", "")
        if src and Path(src).exists():
            dst, _ = QFileDialog.getSaveFileName(self, "Save JSON", "", "JSON (*.json)")
            if dst:
                shutil.copy2(src, dst)

    def _export_csv(self):
        src = (self._log_paths or {}).get("csv_path", "")
        if src and Path(src).exists():
            dst, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
            if dst:
                shutil.copy2(src, dst)

    def _export_overlay(self):
        src = (self._log_paths or {}).get("overlay_path", "")
        if src and Path(src).exists():
            dst, _ = QFileDialog.getSaveFileName(self, "Save Overlay", "", "JPEG (*.jpg)")
            if dst:
                shutil.copy2(src, dst)
