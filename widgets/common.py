"""
widgets/common.py — Shared UI components, ISA-101 industrial dark theme.
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

# ── ISA-101 structural colors ────────────────────────────────────────────────
HEADER_BG    = "#111114"   # Slightly darker than BG — clear visual separation
BORDER_COLOR = "#404040"   # Neutral gray border — no blue tint
PANEL_COLOR  = "#242428"   # Between BG and CARD — for nested panels

# ── Global ISA-101 dark theme QSS ─────────────────────────────────────────────

DARK_THEME_QSS = f"""
QMainWindow, QWidget {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: {FONT_SIZE_NORMAL}px;
}}

QLabel {{
    color: {TEXT_COLOR};
    background: transparent;
}}

QLineEdit {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 10px 14px;
    font-size: {FONT_SIZE_LARGE}px;
}}
QLineEdit:focus {{
    border: 2px solid {ACCENT_BLUE};
}}

QComboBox {{
    background-color: {PANEL_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 8px 12px;
    font-size: {FONT_SIZE_NORMAL}px;
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD_COLOR};
    color: {TEXT_COLOR};
    selection-background-color: {ACCENT_BLUE};
    selection-color: white;
    border: 1px solid {BORDER_COLOR};
}}

QProgressBar {{
    background-color: #3a3a3e;
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    height: 24px;
    text-align: center;
    color: white;
    font-weight: bold;
    font-size: {FONT_SIZE_NORMAL}px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT_BLUE};
    border-radius: 3px;
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
    background: #505050;
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
    border-radius: 4px;
    gridline-color: {BORDER_COLOR};
    font-size: {FONT_SIZE_NORMAL}px;
}}
QTableWidget::item {{
    padding: 10px;
}}
QHeaderView::section {{
    background-color: {PANEL_COLOR};
    color: {MUTED_COLOR};
    border: none;
    border-bottom: 1px solid {BORDER_COLOR};
    padding: 10px;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QMenu {{
    background-color: {CARD_COLOR};
    color: {TEXT_COLOR};
    border: 1px solid {BORDER_COLOR};
    border-radius: 4px;
    padding: 4px;
}}
QMenu::item {{
    padding: 8px 20px;
    border-radius: 3px;
}}
QMenu::item:selected {{
    background-color: {ACCENT_BLUE};
    color: white;
}}
"""


# ── Reusable widgets ──────────────────────────────────────────────────────────

class BigButton(QPushButton):
    """Industrial touch-friendly button — flat, high-contrast, ISA-101 compliant."""

    STYLES = {
        "primary": f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 4px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 12px 28px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover  {{ background-color: #1a7aff; }}
            QPushButton:pressed {{ background-color: #0a55d4; }}
            QPushButton:disabled {{ background-color: #404040; color: #606060; }}
        """,
        "success": f"""
            QPushButton {{
                background-color: #007700;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 12px 28px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover  {{ background-color: #009900; }}
            QPushButton:pressed {{ background-color: #005500; }}
            QPushButton:disabled {{ background-color: #404040; color: #606060; }}
        """,
        "danger": f"""
            QPushButton {{
                background-color: #cc0000;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 12px 28px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover  {{ background-color: #ee1111; }}
            QPushButton:pressed {{ background-color: #aa0000; }}
        """,
        "outlined": f"""
            QPushButton {{
                background-color: transparent;
                color: {ACCENT_BLUE};
                border: 2px solid {ACCENT_BLUE};
                border-radius: 4px;
                font-size: {FONT_SIZE_LARGE}px;
                font-weight: bold;
                padding: 10px 28px;
                min-height: {TOUCH_BUTTON_HEIGHT}px;
            }}
            QPushButton:hover  {{ background-color: rgba(13,110,253,0.15); }}
            QPushButton:pressed {{ background-color: rgba(13,110,253,0.30); }}
        """,
        "small": f"""
            QPushButton {{
                background-color: {PANEL_COLOR};
                color: {MUTED_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                font-size: {FONT_SIZE_NORMAL}px;
                font-weight: bold;
                padding: 6px 16px;
                min-height: 36px;
            }}
            QPushButton:hover  {{ background-color: #3a3a3e; color: {TEXT_COLOR}; }}
            QPushButton:pressed {{ background-color: #404044; }}
        """,
    }

    def __init__(self, text: str, variant: str = "primary", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(self.STYLES.get(variant, self.STYLES["primary"]))


class Card(QFrame):
    """Industrial panel — flat, gray, minimal border."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {CARD_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
            }}
        """)
        self.setFrameShape(QFrame.StyledPanel)


class StatusPill(QLabel):
    """Status label — used inside verdict card (plain text, no pill shape at top level)."""

    def __init__(self, text: str = "", variant: str = "info", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.set_variant(variant)

    def set_variant(self, variant: str):
        configs = {
            "pass":  ("#00dd00", "#001a00"),
            "fail":  ("#ff3333", "#1a0000"),
            "info":  (ACCENT_BLUE, "#001033"),
            "muted": ("#606060", BG_COLOR),
        }
        fg, bg = configs.get(variant, configs["info"])
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border: 2px solid {fg};
                border-radius: 4px;
                padding: 10px 32px;
                font-size: {FONT_SIZE_TITLE}px;
                font-weight: bold;
                letter-spacing: 2px;
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
    """Top header bar — dark, industrial, shows title and model status."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(44)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {HEADER_BG};
                border-bottom: 1px solid {BORDER_COLOR};
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)

        # Left: app name
        self.title_label = QLabel("PROPPANT QC  v2.0")
        self.title_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        self.title_label.setStyleSheet("color: white; background: transparent; letter-spacing: 2px;")
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Right: status badge
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: #2d2d30;
                color: {MUTED_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 3px;
                padding: 3px 14px;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }}
        """)
        layout.addWidget(self.status_label)

    def set_status(self, text: str, ok: bool = True):
        color  = "#3ddc84" if ok else "#f05050"
        border = "#1a6040" if ok else "#602020"
        bg     = "#0d1f15" if ok else "#1f0d0d"
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {color};
                border: 1px solid {border};
                border-radius: 3px;
                padding: 3px 14px;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 0.5px;
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
        self.progress_bar.setFixedHeight(24)
        layout.addWidget(self.progress_bar)

    def set_progress(self, value: int, text: str = ""):
        self.progress_bar.setValue(value)
        if text:
            self.status_label.setText(text)
