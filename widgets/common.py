"""
widgets/common.py — Shared UI components and dark theme stylesheet.
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


# ── Global dark theme QSS ────────────────────────────────────────────

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
    border: 2px solid #3c3c3c;
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
    border: 2px solid #3c3c3c;
    border-radius: 6px;
    padding: 8px 12px;
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
}}

QProgressBar {{
    background-color: #333333;
    border: none;
    border-radius: 8px;
    height: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_BLUE};
    border-radius: 8px;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QTableWidget {{
    background-color: {CARD_COLOR};
    color: {TEXT_COLOR};
    border: none;
    gridline-color: #3c3c3c;
    font-size: {FONT_SIZE_NORMAL}px;
}}
QTableWidget::item {{
    padding: 8px;
}}
QHeaderView::section {{
    background-color: #333333;
    color: {TEXT_COLOR};
    border: none;
    padding: 8px;
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
                border-radius: 8px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 8px 20px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ background-color: #1177bb; }}
            QPushButton:pressed {{ background-color: #0a4f7a; }}
            QPushButton:disabled {{ background-color: #3c3c3c; color: #808080; }}
        """,
        "success": f"""
            QPushButton {{
                background-color: {ACCENT_GREEN};
                color: #1e1e1e;
                border: none;
                border-radius: 8px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 8px 20px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ background-color: #5ee0c0; }}
            QPushButton:pressed {{ background-color: #3aaa8a; }}
            QPushButton:disabled {{ background-color: #3c3c3c; color: #808080; }}
        """,
        "danger": f"""
            QPushButton {{
                background-color: {ACCENT_RED};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 8px 20px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ background-color: #ff6666; }}
            QPushButton:pressed {{ background-color: #cc3333; }}
        """,
        "outlined": f"""
            QPushButton {{
                background-color: transparent;
                color: {TEXT_COLOR};
                border: 2px solid #555555;
                border-radius: 8px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 8px 20px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover {{ border-color: {ACCENT_BLUE}; color: white; }}
            QPushButton:pressed {{ background-color: #333333; }}
        """,
        "small": f"""
            QPushButton {{
                background-color: #3c3c3c;
                color: {TEXT_COLOR};
                border: none;
                border-radius: 6px;
                font-size: {FONT_SIZE_NORMAL}px;
                padding: 5px 12px;
                min-height: 28px;
            }}
            QPushButton:hover {{ background-color: #505050; }}
            QPushButton:pressed {{ background-color: #2a2a2a; }}
        """,
    }

    def __init__(self, text: str, variant: str = "primary", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self.STYLES.get(variant, self.STYLES["primary"]))


class Card(QFrame):
    """Rounded container with dark background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_COLOR};
                border-radius: 8px;
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
            "pass": (ACCENT_GREEN, "#1e1e1e"),
            "fail": (ACCENT_RED, "white"),
            "info": (ACCENT_BLUE, "white"),
            "muted": ("#3c3c3c", MUTED_COLOR),
        }
        bg, fg = colors.get(variant, colors["info"])
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border-radius: 14px;
                padding: 6px 20px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
            }}
        """)

    def set_pass_fail(self, verdict: str):
        self.setText(verdict)
        if verdict.startswith("PASS"):
            self.set_variant("pass")
        elif verdict == "FAIL" or verdict == "ERROR":
            self.set_variant("fail")
        else:
            self.set_variant("info")


class HeaderBar(QWidget):
    """Top bar with title, status, and navigation buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: #2d2d2d;
                border-bottom: 1px solid #3c3c3c;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 10, 0)

        self.title_label = QLabel("Proppant QC System v2.0")
        self.title_label.setFont(QFont("Segoe UI", FONT_SIZE_LARGE, QFont.Bold))
        self.title_label.setStyleSheet(f"color: white; background: transparent;")
        layout.addWidget(self.title_label)

        layout.addStretch()

        self.status_label = QLabel("Loading...")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: #3c3c3c;
                color: {MUTED_COLOR};
                border-radius: 10px;
                padding: 4px 14px;
                font-size: 12px;
            }}
        """)
        layout.addWidget(self.status_label)

    def set_status(self, text: str, ok: bool = True):
        color = ACCENT_GREEN if ok else ACCENT_RED
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: #3c3c3c;
                color: {color};
                border-radius: 10px;
                padding: 4px 14px;
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
        layout.setSpacing(12)

        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Segoe UI", FONT_SIZE_LARGE))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(24)
        layout.addWidget(self.progress_bar)

    def set_progress(self, value: int, text: str = ""):
        self.progress_bar.setValue(value)
        if text:
            self.status_label.setText(text)
