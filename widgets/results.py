"""
widgets/results.py — Industrial card-deck results viewer (ISA-101 compliant).

5 full-screen cards, left/right arrow navigation, no scrolling.
  Card 1: Segmentation Overlay
  Card 2: Verdict & Composition
  Card 3: Spec Compliance
  Card 4: Sieve Distribution
  Card 5: Summary & Actions
"""
import io
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFileDialog, QSizePolicy, QStackedWidget, QPushButton, QFrame,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen

from config import (
    FONT_SIZE_TITLE, FONT_SIZE_LARGE, FONT_SIZE_NORMAL,
    ACCENT_GREEN, ACCENT_RED, ACCENT_BLUE, TEXT_COLOR, MUTED_COLOR,
    CARD_COLOR, BG_COLOR,
)
from widgets.common import BigButton, Card, BORDER_COLOR, PANEL_COLOR

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── ISA-101 palette ───────────────────────────────────────────────────────────
_BG      = "#1a1a1e"
_PANEL   = "#242428"
_CARD    = "#2d2d30"
_BORDER  = "#404040"
_TEXT    = "#ffffff"
_MUTED   = "#a0a0a0"
_GREEN   = "#3ddc84"    # softer green — readable without neon glare
_RED     = "#f05050"    # softer red
_AMBER   = "#ffb81c"
_BLUE    = "#4a9eff"    # slightly lighter blue for dark bg readability

# State tints — subtle, not overwhelming
_PASS_BG     = "#0d1f15"
_PASS_BORDER = "#3ddc84"
_FAIL_BG     = "#1f0d0d"
_FAIL_BORDER = "#f05050"


# ─────────────────────────────────────────────────────────────────────────────
# Shared sub-components
# ─────────────────────────────────────────────────────────────────────────────

def _divider():
    d = QLabel()
    d.setFixedHeight(1)
    d.setStyleSheet(f"background: {_BORDER};")
    return d


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
    lbl.setStyleSheet(f"color: {_MUTED}; letter-spacing: 2px;")
    return lbl


class MeasurementBar(QWidget):
    """Industrial horizontal bar — thick, labeled, with value on the right."""

    def __init__(self, label: str, pct: float, count: int,
                 color: str, parent=None):
        super().__init__(parent)
        self.label    = label
        self.pct      = pct
        self.count    = count
        self.bar_color = QColor(color)
        self.setFixedHeight(38)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Layout: label zone (left 40%), bar zone (middle), value zone (right 120px)
        val_w   = 120
        lbl_w   = int(w * 0.36)
        bar_x   = lbl_w + 8
        bar_w   = w - lbl_w - val_w - 16
        bar_y   = h // 2 - 7
        bar_h   = 14

        # Bar track
        p.setBrush(QColor(_PANEL))
        p.setPen(QPen(QColor(_BORDER), 1))
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 3, 3)

        # Bar fill
        fill_w = max(0, int(bar_w * self.pct / 100.0))
        if fill_w > 0:
            p.setBrush(self.bar_color)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 3, 3)

        # Left label — always white
        p.setPen(QColor(_TEXT))
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.drawText(6, 0, lbl_w, h, Qt.AlignVCenter | Qt.AlignLeft, self.label)

        # Right value — colored text on dark background (separate zone, never overlaps bar)
        val_x = w - val_w
        p.setPen(self.bar_color)
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        val = f"{self.pct:.1f}%  ({self.count})"
        p.drawText(val_x, 0, val_w - 4, h, Qt.AlignVCenter | Qt.AlignRight, val)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Card 1 — Segmentation Overlay
# ─────────────────────────────────────────────────────────────────────────────

class OverlayCard(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(5)

        self.img_label = QLabel("No image loaded")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet(
            f"background: {_PANEL}; border: 1px solid {_BORDER};"
            f" border-radius: 4px; color: {_MUTED};"
        )
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.img_label, stretch=1)

        self.info_bar = QWidget()
        self.info_bar.setFixedHeight(28)
        self.info_bar.setStyleSheet(
            f"background: {_PANEL}; border: 1px solid {_BORDER}; border-radius: 4px;"
        )
        info_row = QHBoxLayout(self.info_bar)
        info_row.setContentsMargins(10, 0, 10, 0)

        self.name_lbl  = QLabel("—")
        self.count_lbl = QLabel("—")
        for lbl in (self.name_lbl, self.count_lbl):
            lbl.setFont(QFont("Segoe UI", 11))
            lbl.setStyleSheet(f"color: {_MUTED};")
        self.count_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        info_row.addWidget(self.name_lbl, stretch=1)
        info_row.addWidget(self.count_lbl, stretch=1)

        layout.addWidget(self.info_bar)

    def load(self, overlay, total: int, image_name: str):
        self._pixmap = None
        if overlay is not None:
            self._rgb_data = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)  # keep ref
            h, w, ch = self._rgb_data.shape
            qimg = QImage(self._rgb_data.data, w, h, ch * w, QImage.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
            self._rescale()
        self.name_lbl.setText(image_name)
        self.count_lbl.setText(f"{total}  particles detected")

    def _rescale(self):
        if self._pixmap:
            self.img_label.setPixmap(
                self._pixmap.scaled(
                    self.img_label.width() - 4,
                    self.img_label.height() - 4,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


# ─────────────────────────────────────────────────────────────────────────────
# Card 2 — Verdict & Composition
# ─────────────────────────────────────────────────────────────────────────────

class VerdictCard(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)

        # Top verdict zone — full-width, subtle colored background
        self.verdict_zone = QWidget()
        self.verdict_zone.setMinimumHeight(150)
        vz_layout = QHBoxLayout(self.verdict_zone)
        vz_layout.setContentsMargins(28, 16, 28, 20)
        vz_layout.setSpacing(20)

        # Big icon on left
        self.icon_lbl = QLabel("")
        self.icon_lbl.setFont(QFont("Segoe UI", 56, QFont.Bold))
        self.icon_lbl.setAlignment(Qt.AlignCenter)
        self.icon_lbl.setFixedWidth(80)
        vz_layout.addWidget(self.icon_lbl)

        # Text block
        text_block = QVBoxLayout()
        text_block.setSpacing(4)
        text_block.addStretch()

        self.verdict_lbl = QLabel("")
        self.verdict_lbl.setFont(QFont("Segoe UI", 28, QFont.Bold))
        self.verdict_lbl.setStyleSheet("letter-spacing: 3px;")
        text_block.addWidget(self.verdict_lbl)

        self.conf_lbl = QLabel("")
        self.conf_lbl.setFont(QFont("Segoe UI", 12))
        self.conf_lbl.setStyleSheet(f"color: {_MUTED};")
        text_block.addWidget(self.conf_lbl)

        text_block.addStretch()
        vz_layout.addLayout(text_block, stretch=1)

        self._layout.addWidget(self.verdict_zone)

        # Bottom composition zone
        comp_zone = QWidget()
        comp_zone.setStyleSheet(f"background: {_BG};")
        cz_layout = QVBoxLayout(comp_zone)
        cz_layout.setContentsMargins(20, 14, 20, 14)
        cz_layout.setSpacing(10)

        cz_layout.addWidget(_section_label("Particle Composition"))
        cz_layout.addWidget(_divider())

        self.bar_4070 = MeasurementBar("40/70  Mesh", 0, 0, _GREEN)
        self.bar_2040 = MeasurementBar("20/40  Mesh", 0, 0, _AMBER)
        cz_layout.addWidget(self.bar_4070)
        cz_layout.addWidget(self.bar_2040)

        row = QHBoxLayout()
        self.total_lbl  = QLabel("Total: —")
        self.reason_lbl = QLabel("")
        for lbl in (self.total_lbl, self.reason_lbl):
            lbl.setFont(QFont("Segoe UI", 11))
            lbl.setStyleSheet(f"color: {_MUTED};")
        self.reason_lbl.setWordWrap(True)
        self.reason_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        row.addWidget(self.total_lbl, stretch=1)
        row.addWidget(self.reason_lbl, stretch=2)
        cz_layout.addLayout(row)

        self._layout.addWidget(comp_zone, stretch=1)

    def load(self, verdict: str, composition: dict,
             confidence: float, total: int, reason: str):
        is_pass = verdict.startswith("PASS")
        is_fail = verdict in ("FAIL", "ERROR")

        if is_pass:
            bg, border, fg = _PASS_BG, _PASS_BORDER, _GREEN
            icon = "✓"
        elif is_fail:
            bg, border, fg = _FAIL_BG, _FAIL_BORDER, _RED
            icon = "✗"
        else:
            bg, border, fg = _PANEL, _BORDER, _MUTED
            icon = "?"

        self.verdict_zone.setStyleSheet(
            f"background: {bg}; border-bottom: 3px solid {border};"
        )
        self.icon_lbl.setText(icon)
        self.icon_lbl.setStyleSheet(f"color: {fg}; font-size: 56px; font-weight: bold;")
        self.verdict_lbl.setText(verdict)
        self.verdict_lbl.setStyleSheet(
            f"color: {fg}; font-size: 28px; font-weight: bold; letter-spacing: 3px;"
        )
        self.conf_lbl.setText(f"Confidence  {confidence:.1f}%")

        p40 = composition.get("proppant_40_70", {})
        p20 = composition.get("proppant_20_40", {})
        self.bar_4070.pct   = p40.get("percentage", 0)
        self.bar_4070.count = p40.get("count", 0)
        self.bar_2040.pct   = p20.get("percentage", 0)
        self.bar_2040.count = p20.get("count", 0)
        self.bar_4070.update()
        self.bar_2040.update()
        self.total_lbl.setText(f"Total  {total}  particles")
        self.reason_lbl.setText(reason)


# ─────────────────────────────────────────────────────────────────────────────
# Card 3 — Spec Compliance
# ─────────────────────────────────────────────────────────────────────────────

class SpecCard(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        layout.addStretch()
        layout.addWidget(_section_label("Spec Compliance"))
        layout.addWidget(_divider())

        self._rows = []
        for _ in range(3):
            # Outer wrapper holds the left-color accent bar
            outer = QWidget()
            outer.setFixedHeight(72)
            outer_layout = QHBoxLayout(outer)
            outer_layout.setContentsMargins(0, 0, 0, 0)
            outer_layout.setSpacing(0)

            # Left accent bar (4px wide, colored)
            accent = QLabel()
            accent.setFixedWidth(5)
            accent.setStyleSheet(f"background: {_BORDER}; border-radius: 2px;")

            # Row content
            row_widget = QWidget()
            row_widget.setStyleSheet(
                f"background: {_PANEL}; border: 1px solid {_BORDER};"
                f" border-left: none; border-radius: 0 4px 4px 0;"
            )
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(16, 0, 16, 0)
            row_layout.setSpacing(14)

            icon = QLabel("")
            icon.setFixedWidth(36)
            icon.setFont(QFont("Segoe UI", 22, QFont.Bold))
            icon.setAlignment(Qt.AlignCenter)

            text = QLabel("")
            text.setFont(QFont("Segoe UI", 12))
            text.setStyleSheet(f"color: {_TEXT};")

            val = QLabel("")
            val.setFont(QFont("Segoe UI", 18, QFont.Bold))
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setMinimumWidth(140)

            row_layout.addWidget(icon)
            row_layout.addWidget(text, stretch=1)
            row_layout.addWidget(val)

            outer_layout.addWidget(accent)
            outer_layout.addWidget(row_widget, stretch=1)

            layout.addWidget(outer)
            self._rows.append((outer, accent, row_widget, icon, text, val))

        layout.addStretch()

    def load(self, swe: dict):
        checks = [
            (
                "Classified rate",
                f"{swe.get('classified_rate_pct', 0):.0f}%",
                swe.get("classified_pass", False),
                ">= 90%  required",
            ),
            (
                "Size accuracy",
                f"{swe.get('mean_size_error_pct', 0):.1f}%  error",
                swe.get("size_error_pass", False),
                "+/- 10 wt%  limit",
            ),
            (
                "Processing time",
                f"{swe.get('processing_time_sec', 0):.1f} s",
                swe.get("processing_time_pass", True),
                "<= 20 s  limit",
            ),
        ]
        for (outer, accent, row_w, icon, text, val), \
                (label, value, passed, sublabel) in zip(self._rows, checks):
            if passed:
                fg = _GREEN
                accent.setStyleSheet(f"background: {_GREEN}; border-radius: 2px;")
                row_w.setStyleSheet(
                    f"background: {_PANEL}; border: 1px solid {_BORDER};"
                    f" border-left: none;"
                )
                icon.setText("✓")
            else:
                fg = _RED
                accent.setStyleSheet(f"background: {_RED}; border-radius: 2px;")
                row_w.setStyleSheet(
                    f"background: {_FAIL_BG}; border: 1px solid {_FAIL_BORDER};"
                    f" border-left: none;"
                )
                icon.setText("✗")

            icon.setStyleSheet(f"color: {fg}; font-size: 22px; font-weight: bold;")
            text.setText(f"{label}")
            text.setStyleSheet(
                f"color: {_TEXT}; font-size: 13px; font-weight: bold;"
            )
            val.setText(value)
            val.setStyleSheet(f"color: {fg}; font-size: 18px; font-weight: bold;")


# ─────────────────────────────────────────────────────────────────────────────
# Card 4 — Sieve Distribution Chart
# ─────────────────────────────────────────────────────────────────────────────

class SieveCard(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)

        hdr = QHBoxLayout()
        hdr.addWidget(_section_label("Sieve Distribution"))
        self.legend_lbl = QLabel("")
        self.legend_lbl.setFont(QFont("Segoe UI", 11))
        self.legend_lbl.setStyleSheet(f"color: {_MUTED};")
        self.legend_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        hdr.addWidget(self.legend_lbl, stretch=1)
        layout.addLayout(hdr)
        layout.addWidget(_divider())

        self.chart_lbl = QLabel("")
        self.chart_lbl.setAlignment(Qt.AlignCenter)
        self.chart_lbl.setStyleSheet(
            f"background: {_PANEL}; border: 1px solid {_BORDER}; border-radius: 4px;"
        )
        self.chart_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.chart_lbl, stretch=1)

        self.note_lbl = QLabel("")
        self.note_lbl.setAlignment(Qt.AlignCenter)
        self.note_lbl.setFont(QFont("Segoe UI", 10))
        self.note_lbl.setStyleSheet(f"color: {_MUTED};")
        layout.addWidget(self.note_lbl)

        self._last_particles = []
        self._last_verdict   = ""

    def load(self, particles: list, verdict: str):
        self._last_particles = particles
        self._last_verdict   = verdict
        self._render()

    def _render(self):
        particles = self._last_particles
        verdict   = self._last_verdict

        if not HAS_MATPLOTLIB or not particles:
            self.chart_lbl.setText("No data available")
            return

        is_fail  = "FAIL" in verdict
        is_2040  = "20_40" in verdict

        # For FAIL (mixed), determine dominant class and show that class only
        if is_fail and not is_2040:
            n_4070 = sum(1 for p in particles if p["class_id"] == 0)
            n_2040 = sum(1 for p in particles if p["class_id"] == 1)
            is_2040 = n_2040 > n_4070

        if is_2040:
            cls_id      = 1
            mesh_sizes  = [16, 20, 25, 30, 35, 40, 50]
            openings_mm = [1.18, 0.85, 0.71, 0.60, 0.50, 0.425, 0.30]
            d_min_mm, d_max_mm = 0.42, 0.84
            lab_key     = "proppant_20_40"
            title_str   = ("Mixed Sample — 20/40 Dominant  |  Sieve Distribution"
                           if is_fail else "20/40 Mesh  —  Sieve Distribution")
        else:
            cls_id      = 0
            mesh_sizes  = [30, 40, 50, 60, 70, 100]
            openings_mm = [0.60, 0.425, 0.30, 0.25, 0.212, 0.150]
            d_min_mm, d_max_mm = 0.21, 0.42
            lab_key     = "proppant_40_70"
            title_str   = ("Mixed Sample — 40/70 Dominant  |  Sieve Distribution"
                           if is_fail else "40/70 Mesh  —  Sieve Distribution")

        from inference_stardist import ProppantAnalyzer
        lab_full = ProppantAnalyzer._load_sieve_references_full()

        relevant = [p for p in particles if p["class_id"] == cls_id]
        if not relevant:
            self.chart_lbl.setText("No classified particles")
            return

        # If calibrated (diameter_mm set), use it directly.
        # Otherwise auto-estimate px/mm from the observed median particle size:
        # median particle is assumed to be at the physical midpoint of its mesh class.
        _has_mm = any(p.get("diameter_mm") is not None for p in relevant)
        if _has_mm:
            diams = [
                float(p["diameter_mm"]) if p.get("diameter_mm") is not None
                else (d_min_mm + d_max_mm) / 2.0
                for p in relevant
            ]
        else:
            _mid_mm = (d_min_mm + d_max_mm) / 2.0
            _px_vals = np.array([p["diameter_px"] for p in relevant])
            _p50_px  = float(np.median(_px_vals))
            _est_ppm = _p50_px / _mid_mm if _mid_mm > 0 else 1.0
            _est_min_px = _est_ppm * d_min_mm
            _est_max_px = _est_ppm * d_max_mm

            def px_to_mm(d):
                t = max(0.0, min(1.0, (d - _est_min_px) / max(_est_max_px - _est_min_px, 1.0)))
                return d_min_mm + t * (d_max_mm - d_min_mm)

            diams = [px_to_mm(p["diameter_px"]) for p in relevant]
        masses = [d ** 3 for d in diams]
        total_m = sum(masses) or 1.0

        retained, cum_model = [], []
        for i, opening in enumerate(openings_mm):
            nxt = openings_mm[i - 1] if i > 0 else 999.0
            retained.append(
                sum(m for d, m in zip(diams, masses) if opening <= d < nxt) / total_m * 100
            )
        cumulative = 100.0
        for r in retained:
            cumulative -= r
            cum_model.append(max(0.0, cumulative))

        lab_data = lab_full.get(lab_key, {})
        has_lab  = bool(lab_data)
        if has_lab:
            cum_lab, c = [], 100.0
            for m in mesh_sizes:
                c -= lab_data.get(m, 0.0)
                cum_lab.append(max(0.0, c))

        # ── Plot ─────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(5.2, 3.0), facecolor=_PANEL)
        ax.set_facecolor(_BG)

        ax.plot(mesh_sizes, cum_model,
                color=_BLUE, linewidth=2.5, marker="o", markersize=6,
                label="Model", zorder=3)
        if has_lab:
            ax.plot(mesh_sizes, cum_lab,
                    color=_AMBER, linewidth=2.5, linestyle="--",
                    marker="s", markersize=6, label="Lab sieve", zorder=3)

        ax.set_xlabel("Mesh size", color=_MUTED, fontsize=10, labelpad=6)
        ax.set_ylabel("Cumulative passing  %", color=_MUTED, fontsize=10, labelpad=6)
        ax.set_title(title_str, color=_TEXT, fontsize=11, fontweight="bold", pad=10)
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.set_ylim(-2, 107)
        ax.grid(True, color="#2a2a2e", linewidth=0.8, linestyle="-")

        for spine in ax.spines.values():
            spine.set_color(_BORDER)

        if has_lab:
            ax.legend(
                fontsize=10, facecolor=_PANEL, edgecolor=_BORDER,
                labelcolor=_TEXT, loc="upper right",
                framealpha=0.95,
            )

        fig.tight_layout(pad=1.2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130,
                    facecolor=_PANEL, edgecolor="none")
        plt.close(fig)
        buf.seek(0)

        qimg = QImage()
        qimg.loadFromData(buf.getvalue())
        self.chart_lbl.setPixmap(
            QPixmap.fromImage(qimg).scaled(
                self.chart_lbl.width() - 4,
                self.chart_lbl.height() - 4,
                Qt.KeepAspectRatio, Qt.SmoothTransformation,
            )
        )
        self.legend_lbl.setText(
            "Blue = Model   |   Amber = Lab sieve" if has_lab else "Blue = Model  (Lab data unavailable)"
        )
        if not has_lab:
            self.note_lbl.setText("Check SIEVE_EXCEL_PATH in config.py")
        else:
            self.note_lbl.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_particles:
            self._render()


# ─────────────────────────────────────────────────────────────────────────────
# Card 5 — Summary & Actions
# ─────────────────────────────────────────────────────────────────────────────

class SummaryCard(QWidget):

    new_test_clicked = pyqtSignal()
    home_clicked     = pyqtSignal()
    export_clicked   = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 20)
        layout.setSpacing(14)

        layout.addStretch()
        layout.addWidget(_section_label("Test Summary"))
        layout.addWidget(_divider())

        # Data rows
        self._fields = {}
        for key in ("Run ID", "Image", "Blur Score", "Processing Time"):
            row = QHBoxLayout()
            key_lbl = QLabel(key)
            key_lbl.setFont(QFont("Segoe UI", 12))
            key_lbl.setStyleSheet(f"color: {_MUTED};")
            key_lbl.setFixedWidth(160)

            val_lbl = QLabel("—")
            val_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
            val_lbl.setStyleSheet(f"color: {_TEXT};")

            row.addWidget(key_lbl)
            row.addWidget(val_lbl, stretch=1)
            layout.addLayout(row)
            self._fields[key] = val_lbl

        layout.addStretch()
        layout.addWidget(_divider())

        # Action buttons — large, touch-friendly (80px height)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.btn_home   = QPushButton("HOME")
        self.btn_export = QPushButton("EXPORT")
        self.btn_new    = QPushButton("NEW TEST")

        self.btn_home.setFixedHeight(70)
        self.btn_export.setFixedHeight(70)
        self.btn_new.setFixedHeight(70)

        self.btn_home.setStyleSheet(f"""
            QPushButton {{
                background: {_PANEL};
                color: {_MUTED};
                border: 1px solid {_BORDER};
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 2px;
            }}
            QPushButton:hover  {{ color: {_TEXT}; border-color: #606060; }}
            QPushButton:pressed {{ background: #3a3a3e; }}
        """)
        self.btn_export.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {_BLUE};
                border: 2px solid {_BLUE};
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 2px;
            }}
            QPushButton:hover  {{ background: rgba(13,110,253,0.15); }}
            QPushButton:pressed {{ background: rgba(13,110,253,0.30); }}
        """)
        self.btn_new.setStyleSheet(f"""
            QPushButton {{
                background: {_BLUE};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 2px;
            }}
            QPushButton:hover  {{ background: #1a7aff; }}
            QPushButton:pressed {{ background: #0a55d4; }}
        """)

        self.btn_home.setCursor(Qt.PointingHandCursor)
        self.btn_export.setCursor(Qt.PointingHandCursor)
        self.btn_new.setCursor(Qt.PointingHandCursor)

        self.btn_home.clicked.connect(self.home_clicked.emit)
        self.btn_export.clicked.connect(self.export_clicked.emit)
        self.btn_new.clicked.connect(self.new_test_clicked.emit)

        btn_row.addWidget(self.btn_home, stretch=1)
        btn_row.addWidget(self.btn_export, stretch=1)
        btn_row.addWidget(self.btn_new, stretch=2)
        layout.addLayout(btn_row)

    def load(self, run_id: str, image_name: str, blur: float, proc_time: float):
        self._fields["Run ID"].setText(run_id or "—")
        self._fields["Image"].setText(image_name or "—")
        self._fields["Blur Score"].setText(f"{blur:.0f}")
        self._fields["Processing Time"].setText(f"{proc_time:.1f} s")


# ─────────────────────────────────────────────────────────────────────────────
# Dot indicator
# ─────────────────────────────────────────────────────────────────────────────

class DotIndicator(QWidget):

    def __init__(self, n: int, parent=None):
        super().__init__(parent)
        self.n = n
        self.current = 0
        self.setFixedHeight(22)

    def set_page(self, idx: int):
        self.current = idx
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r, gap = 5, 14
        total_w = self.n * r * 2 + (self.n - 1) * gap
        x = (self.width() - total_w) // 2
        y = self.height() // 2
        for i in range(self.n):
            cx = x + i * (r * 2 + gap) + r
            if i == self.current:
                p.setBrush(QColor(_BLUE))
                p.setPen(Qt.NoPen)
                p.drawEllipse(cx - r, y - r, r * 2, r * 2)
            else:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(_BORDER), 1.5))
                p.drawEllipse(cx - r, y - r, r * 2, r * 2)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Arrow button
# ─────────────────────────────────────────────────────────────────────────────

class ArrowButton(QPushButton):

    def __init__(self, direction: str, parent=None):
        super().__init__("‹" if direction == "left" else "›", parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedWidth(80)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setStyleSheet(f"""
            QPushButton {{
                background: {_PANEL};
                color: {_MUTED};
                border: 2px solid {_BORDER};
                border-radius: 8px;
                font-size: 48px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background: #2d2d30;
                color: {_BLUE};
                border-color: {_BLUE};
            }}
            QPushButton:pressed {{ background: #3a3a3e; }}
            QPushButton:disabled {{
                color: #2a2a2e;
                border-color: #2a2a2e;
                background: {_BG};
            }}
        """)


# ─────────────────────────────────────────────────────────────────────────────
# Main ResultsScreen
# ─────────────────────────────────────────────────────────────────────────────

CARD_NAMES = [
    "Segmentation Overlay",
    "Verdict & Composition",
    "Spec Compliance",
    "Sieve Distribution",
    "Summary",
]
NUM_CARDS = len(CARD_NAMES)


class ResultsScreen(QWidget):

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

        # ── Top nav bar ──────────────────────────────────────────────────
        nav = QWidget()
        nav.setFixedHeight(36)
        nav.setStyleSheet(
            f"background: {_PANEL}; border-bottom: 1px solid {_BORDER};"
        )
        nav_row = QHBoxLayout(nav)
        nav_row.setContentsMargins(10, 0, 10, 0)
        nav_row.setSpacing(10)

        self.btn_home_nav = QPushButton("HOME")
        self.btn_home_nav.setFixedHeight(26)
        self.btn_home_nav.setCursor(Qt.PointingHandCursor)
        self.btn_home_nav.clicked.connect(self.home_clicked.emit)
        self.btn_home_nav.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {_MUTED};
                border: 1px solid {_BORDER};
                border-radius: 3px;
                font-size: 11px;
                font-weight: bold;
                padding: 0 12px;
                letter-spacing: 1px;
            }}
            QPushButton:hover {{ color: {_TEXT}; border-color: #606060; }}
        """)
        nav_row.addWidget(self.btn_home_nav)

        nav_row.addStretch()

        self.card_name_lbl = QLabel(CARD_NAMES[0])
        self.card_name_lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.card_name_lbl.setStyleSheet(
            f"color: {_TEXT}; letter-spacing: 1px;"
        )
        self.card_name_lbl.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.card_name_lbl)

        nav_row.addStretch()

        self.run_id_lbl = QLabel("—")
        self.run_id_lbl.setFont(QFont("Segoe UI", 11))
        self.run_id_lbl.setStyleSheet(f"color: {_MUTED};")
        nav_row.addWidget(self.run_id_lbl)

        outer.addWidget(nav)

        # ── Cards + arrows ────────────────────────────────────────────────
        mid = QHBoxLayout()
        mid.setContentsMargins(4, 4, 4, 4)
        mid.setSpacing(4)

        self.btn_left  = ArrowButton("left")
        self.btn_right = ArrowButton("right")
        self.btn_left.clicked.connect(self._prev)
        self.btn_right.clicked.connect(self._next)

        self.stack = QStackedWidget()
        self.c_overlay = OverlayCard()
        self.c_verdict = VerdictCard()
        self.c_spec    = SpecCard()
        self.c_sieve   = SieveCard()
        self.c_summary = SummaryCard()

        for c in [self.c_overlay, self.c_verdict, self.c_spec,
                  self.c_sieve, self.c_summary]:
            self.stack.addWidget(c)

        mid.addWidget(self.btn_left)
        mid.addWidget(self.stack, stretch=1)
        mid.addWidget(self.btn_right)
        outer.addLayout(mid, stretch=1)

        # ── Dot indicator ─────────────────────────────────────────────────
        self.dots = DotIndicator(NUM_CARDS)
        self.dots.setStyleSheet(
            f"background: {_PANEL}; border-top: 1px solid {_BORDER};"
        )
        outer.addWidget(self.dots)

        # Signals
        self.c_summary.new_test_clicked.connect(self.new_test_clicked.emit)
        self.c_summary.home_clicked.connect(self.home_clicked.emit)
        self.c_summary.export_clicked.connect(self._export_dialog)

        self._sync_nav()

    # ── Navigation ────────────────────────────────────────────────────────

    def _go(self, idx: int):
        idx = max(0, min(NUM_CARDS - 1, idx))
        self.stack.setCurrentIndex(idx)
        self.card_name_lbl.setText(CARD_NAMES[idx])
        self.dots.set_page(idx)
        self._sync_nav()

    def _next(self): self._go(self.stack.currentIndex() + 1)
    def _prev(self): self._go(self.stack.currentIndex() - 1)

    def _sync_nav(self):
        idx = self.stack.currentIndex()
        self.btn_left.setEnabled(idx > 0)
        self.btn_right.setEnabled(idx < NUM_CARDS - 1)

    # ── Data ──────────────────────────────────────────────────────────────

    def show_result(self, result: dict, log_paths: dict, run_id: str = ""):
        self._result    = result
        self._log_paths = log_paths

        comp      = result.get("composition", {})
        verdict   = result.get("verdict", "?")
        particles = result.get("particles", [])

        self.run_id_lbl.setText(run_id or "—")

        self.c_overlay.load(
            result.get("overlay"),
            result.get("total_particles", 0),
            result.get("image_name", ""),
        )
        self.c_verdict.load(
            verdict, comp,
            result.get("avg_confidence", 0),
            result.get("total_particles", 0),
            result.get("reason", ""),
        )
        self.c_spec.load(result.get("swe_checks", {}))
        self.c_sieve.load(particles, verdict)
        self.c_summary.load(
            run_id or "—",
            result.get("image_name", "—"),
            result.get("blur_score", 0),
            result.get("processing_time_sec", 0),
        )
        self._go(0)

    # ── Export ────────────────────────────────────────────────────────────

    def _export_dialog(self):
        if not self._log_paths:
            return
        from PyQt5.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Export JSON",  self._export_json)
        menu.addAction("Export CSV",   self._export_csv)
        menu.addAction("Save Overlay", self._export_overlay)
        menu.exec_(
            self.c_summary.btn_export.mapToGlobal(
                self.c_summary.btn_export.rect().topLeft()
            )
        )

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
