"""
widgets/common.py — Shared UI components and light industrial theme stylesheet.
"""
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFrame, QHBoxLayout, QVBoxLayout,
    QProgressBar, QSizePolicy,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from config import (
    BG_COLOR, CARD_COLOR, ACCENT_BLUE, ACCENT_GREEN, ACCENT_RED,
    TEXT_COLOR, MUTED_COLOR, TOUCH_BUTTON_HEIGHT,
    FONT_SIZE_NORMAL, FONT_SIZE_LARGE, FONT_SIZE_TITLE,
)

# ── Header constants ─────────────────────────────────────────────────
HEADER_BG   = "#1a3a5c"   # Navy — stays dark for contrast
BORDER_COLOR = "#dde3ec"  # Subtle card border

# ── Global light theme QSS ────────────────────────────────────────────

DARK_THEME_QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: "Segoe UI", "Noto Sans", sans-serif;
    font-size: {FONT_SIZE_NORMAL}px;
}}

QLabel {{
    color: {TEXT_COLOR};
    background: transparent;
}}

QLineEdit {{
    background-color: {CARD_COLOR};
    color: {TEXT_COLOR};
    border: 2px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 12px 16px;
    font-size: {FONT_SIZE_LARGE}px;
}}
QLineEdit:focus {{
    border-color: {ACCENT_BLUE};
}}

QComboBox {{
    background-color: {CARD_COLOR};
    color: {TEXT_COLOR};
    border: 2px solid {BORDER_COLOR};
    border-radius: 8px;
    padding: 10px 14px;
    font-size: {FONT_SIZE_NORMAL}px;
}}
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD_COLOR};
    color: {TEXT_COLOR};
    selection-background-color: {ACCENT_BLUE};
    selection-color: white;
    border: 1px solid {BORDER_COLOR};
}}

QProgressBar {{
    background-color: #dde3ec;
    border: none;
    border-radius: 8px;
    height: 22px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: {FONT_SIZE_NORMAL}px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_BLUE};
    border-radius: 8px;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QScrollBar:vertical {{
    background: {BG_COLOR};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: #b0bec5;
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QTableWidget {{
    background-color: {CARD_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 8px;
    gridline-color: {BORDER_COLOR};
    font-size: {FONT_SIZE_NORMAL}px;
}}
QTableWidget::item {{
    padding: 10px;
}}
QHeaderView::section {{
    background-color: #e8eef5;
    color: {TEXT_COLOR};
    border: none;
    border-bottom: 2px solid {BORDER_COLOR};
    padding: 10px;
    font-weight: bold;
}}
"""


# ── Reusable widgets ──────────────────────────────────────────────────

class BigButton(QPushButton):
    """Large, touch-friendly button with color variants."""

    STYLES = {
        "primary": f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 10px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 10px 24px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ background-color: #1976d2; }}
            QPushButton:pressed {{ background-color: #0d47a1; }}
            QPushButton:disabled {{ background-color: #b0bec5; color: #eceff1; }}
        """,
        "success": f"""
            QPushButton {{
                background-color: {ACCENT_GREEN};
                color: white;
                border: none;
                border-radius: 10px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 10px 24px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ background-color: #388e3c; }}
            QPushButton:pressed {{ background-color: #1b5e20; }}
            QPushButton:disabled {{ background-color: #b0bec5; color: #eceff1; }}
        """,
        "danger": f"""
            QPushButton {{
                background-color: {ACCENT_RED};
                color: white;
                border: none;
                border-radius: 10px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 10px 24px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ background-color: #d32f2f; }}
            QPushButton:pressed {{ background-color: #b71c1c; }}
        """,
        "outlined": f"""
            QPushButton {{
                background-color: white;
                color: {ACCENT_BLUE};
                border: 2px solid {ACCENT_BLUE};
                border-radius: 10px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 10px 24px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ background-color: #e3f0fc; }}
            QPushButton:pressed {{ background-color: #bbdefb; }}
        """,
        "small": f"""
            QPushButton {{
                background-color: #e8eef5;
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                font-size: {FONT_SIZE_NORMAL}px;
                font-weight: bold;
                padding: 6px 16px;
                min-height: 36px;
            }}
            QPushButton:hover {{ background-color: #d0dce8; }}
            QPushButton:pressed {{ background-color: #b0c4d8; }}
        """,
    }

    def __init__(self, text: str, variant: str = "primary", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self.STYLES.get(variant, self.STYLES["primary"]))


class Card(QFrame):
    """White rounded card with subtle border shadow."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 10px;
                padding: 8px;
            }}
        """)
        self.setFrameShape(QFrame.StyledPanel)


class StatusPill(QLabel):
    """Colored pill label for PASS/FAIL/INFO status."""

    def __init__(self, text: str = "", variant: str = "info", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_variant(variant)

    def set_variant(self, variant: str):
        colors = {
            "pass":  (ACCENT_GREEN, "white"),
            "fail":  (ACCENT_RED,   "white"),
            "info":  (ACCENT_BLUE,  "white"),
            "muted": ("#b0bec5",    TEXT_COLOR),
        }
        bg, fg = colors.get(variant, colors["info"])
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border-radius: 16px;
                padding: 8px 28px;
                font-size: {FONT_SIZE_TITLE}px;
                font-weight: bold;
                letter-spacing: 1px;
            }}
        """)

    def set_pass_fail(self, verdict: str):
        self.setText(verdict)
        if verdict.startswith("PASS"):
            self.set_variant("pass")
        elif verdict in ("FAIL", "ERROR"):
            self.set_variant("fail")
        else:
            self.set_variant("info")


class HeaderBar(QWidget):
    """Navy top bar with title and model/GPU status badge."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {HEADER_BG};
                border-bottom: 2px solid #0d2840;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)

        self.title_label = QLabel("Proppant QC System v2.0")
        self.title_label.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        self.title_label.setStyleSheet("color: white; background: transparent;")
        layout.addWidget(self.title_label)

        layout.addStretch()

        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255,255,255,0.12);
                color: #90caf9;
                border-radius: 10px;
                padding: 4px 16px;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)

    def set_status(self, text: str, ok: bool = True):
        color = "#a5d6a7" if ok else "#ef9a9a"
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: rgba(255,255,255,0.12);
                color: {color};
                border-radius: 10px;
                padding: 4px 16px;
                font-size: 12px;
                font-weight: bold;
            }}
        """)
        self.status_label.setText(text)


class ProgressCard(Card):
    """Card with a progress bar and status text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", FONT_SIZE_LARGE))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"color: {TEXT_COLOR};")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(26)
        layout.addWidget(self.progress_bar)

    def set_progress(self, value: int, text: str = ""):
        self.progress_bar.setValue(value)
        if text:
            self.status_label.setText(text)
