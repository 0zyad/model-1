"""
widgets/common.py — Shared UI components, ISA-101 industrial dark theme.
"""
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QFrame, QHBoxLayout, QVBoxLayout,
    QProgressBar, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
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


def get_theme_qss(is_dark: bool) -> str:
    """Return the global QSS for dark or light mode."""
    if is_dark:
        bg     = "#1a1a1e"
        card   = "#2d2d30"
        panel  = "#242428"
        border = "#404040"
        text   = "#ffffff"
        muted  = "#a0a0a0"
        pb_bg  = "#3a3a3e"
        pb_txt = "white"
        scroll_bg     = "#1a1a1e"
        scroll_handle = "#505050"
    else:
        bg     = "#f6f8fa"
        card   = "#ffffff"
        panel  = "#eef1f5"
        border = "#d0d7de"
        text   = "#1f2328"
        muted  = "#636e7b"
        pb_bg  = "#e1e4e8"
        pb_txt = "#1f2328"
        scroll_bg     = "#f0f2f5"
        scroll_handle = "#b8bfc8"

    return f"""
QMainWindow, QWidget {{
    background-color: {bg};
    color: {text};
    font-family: "Segoe UI", "Arial", sans-serif;
    font-size: {FONT_SIZE_NORMAL}px;
}}

QLabel {{
    color: {text};
    background: transparent;
}}

QLineEdit {{
    background-color: {panel};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 10px 14px;
    font-size: {FONT_SIZE_LARGE}px;
}}
QLineEdit:focus {{
    border: 2px solid {ACCENT_BLUE};
}}

QComboBox {{
    background-color: {panel};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    padding: 8px 12px;
    font-size: {FONT_SIZE_NORMAL}px;
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background-color: {card};
    color: {text};
    selection-background-color: {ACCENT_BLUE};
    selection-color: white;
    border: 1px solid {border};
}}

QProgressBar {{
    background-color: {pb_bg};
    border: 1px solid {border};
    border-radius: 4px;
    height: 24px;
    text-align: center;
    color: {pb_txt};
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
    background: {scroll_bg};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {scroll_handle};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

QTableWidget {{
    background-color: {card};
    color: {text};
    border: 1px solid {border};
    border-radius: 4px;
    gridline-color: {border};
    font-size: {FONT_SIZE_NORMAL}px;
}}
QTableWidget::item {{
    padding: 10px;
}}
QHeaderView::section {{
    background-color: {panel};
    color: {muted};
    border: none;
    border-bottom: 1px solid {border};
    padding: 10px;
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QMenu {{
    background-color: {card};
    color: {text};
    border: 1px solid {border};
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


# Keep DARK_THEME_QSS for backward compatibility
DARK_THEME_QSS = get_theme_qss(True)


# ── Reusable widgets ──────────────────────────────────────────────────────────

class BigButton(QPushButton):
    """Industrial touch-friendly button — flat, high-contrast."""

    @staticmethod
    def _make_styles(is_dark: bool) -> dict:
        if is_dark:
            small_bg      = "#2a2a2e"
            small_fg      = "#a0a0a0"
            small_border  = "#404040"
            small_hover   = "#3a3a3e"
            small_hover_fg = "#ffffff"
            small_pressed = "#404044"
            dis_bg        = "#404040"
            dis_fg        = "#606060"
        else:
            small_bg      = "#ffffff"
            small_fg      = "#444d56"
            small_border  = "#d0d7de"
            small_hover   = "#f3f4f6"
            small_hover_fg = "#1f2328"
            small_pressed = "#e8ecf0"
            dis_bg        = "#e1e4e8"
            dis_fg        = "#959da5"
        return {
            "primary": f"""
                QPushButton {{
                    background-color: {ACCENT_BLUE};
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: {FONT_SIZE_LARGE}px;
                    font-weight: bold;
                    padding: 12px 28px;
                    min-height: {TOUCH_BUTTON_HEIGHT}px;
                    letter-spacing: 0.5px;
                }}
                QPushButton:hover  {{ background-color: #1a7aff; }}
                QPushButton:pressed {{ background-color: #0a55d4; }}
                QPushButton:disabled {{ background-color: {dis_bg}; color: {dis_fg}; }}
            """,
            "success": f"""
                QPushButton {{
                    background-color: #1a7f37;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: {FONT_SIZE_LARGE}px;
                    font-weight: bold;
                    padding: 12px 28px;
                    min-height: {TOUCH_BUTTON_HEIGHT}px;
                }}
                QPushButton:hover  {{ background-color: #1f9d44; }}
                QPushButton:pressed {{ background-color: #155d27; }}
                QPushButton:disabled {{ background-color: {dis_bg}; color: {dis_fg}; }}
            """,
            "danger": f"""
                QPushButton {{
                    background-color: #cf222e;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    font-size: {FONT_SIZE_LARGE}px;
                    font-weight: bold;
                    padding: 12px 28px;
                    min-height: {TOUCH_BUTTON_HEIGHT}px;
                }}
                QPushButton:hover  {{ background-color: #a40e26; }}
                QPushButton:pressed {{ background-color: #82071e; }}
            """,
            "outlined": f"""
                QPushButton {{
                    background-color: transparent;
                    color: {ACCENT_BLUE};
                    border: 2px solid {ACCENT_BLUE};
                    border-radius: 6px;
                    font-size: {FONT_SIZE_LARGE}px;
                    font-weight: bold;
                    padding: 10px 28px;
                    min-height: {TOUCH_BUTTON_HEIGHT}px;
                }}
                QPushButton:hover  {{ background-color: rgba(9,105,218,0.10); }}
                QPushButton:pressed {{ background-color: rgba(9,105,218,0.20); }}
            """,
            "small": f"""
                QPushButton {{
                    background-color: {small_bg};
                    color: {small_fg};
                    border: 1px solid {small_border};
                    border-radius: 6px;
                    font-size: {FONT_SIZE_NORMAL}px;
                    font-weight: bold;
                    padding: 6px 16px;
                    min-height: 36px;
                }}
                QPushButton:hover  {{ background-color: {small_hover}; color: {small_hover_fg}; }}
                QPushButton:pressed {{ background-color: {small_pressed}; }}
            """,
        }

    def __init__(self, text: str, variant: str = "primary", parent=None):
        super().__init__(text, parent)
        self._variant = variant
        self.setCursor(Qt.PointingHandCursor)
        self._apply(True)

    def _apply(self, is_dark: bool):
        styles = self._make_styles(is_dark)
        self.setStyleSheet(styles.get(self._variant, styles["primary"]))

    def apply_theme(self, is_dark: bool):
        self._apply(is_dark)


class Card(QFrame):
    """Panel card — adapts to current theme."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self._apply(True)

    def _apply(self, is_dark: bool):
        if is_dark:
            bg, border, radius = "#2d2d30", "#404040", "6px"
        else:
            bg, border, radius = "#ffffff", "#d0d7de", "8px"
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg};
                border: 1px solid {border};
                border-radius: {radius};
            }}
        """)

    def apply_theme(self, is_dark: bool):
        self._apply(is_dark)


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

    theme_toggled = pyqtSignal(bool)   # True = dark mode

    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_dark = True
        self._status_ok = None
        self.setFixedHeight(44)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(8)

        # Left: app name
        self.title_label = QLabel("PROPPANT QC  v2.0")
        self.title_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        layout.addWidget(self.title_label)

        layout.addStretch()

        # Right: status badge
        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Segoe UI", 11))
        layout.addWidget(self.status_label)

        # Theme toggle button
        self.theme_btn = QPushButton("Light")
        self.theme_btn.setFixedSize(64, 26)
        self.theme_btn.setCursor(Qt.PointingHandCursor)
        self.theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self.theme_btn)

        # Apply initial styles after all widgets exist
        self._apply_bar_style()
        self._apply_toggle_style()

    def _apply_bar_style(self):
        if self._is_dark:
            bg, border, title_color = "#111114", "#404040", "#ffffff"
        else:
            bg, border, title_color = "#ffffff", "#d0d7de", "#1f2328"
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg};
                border-bottom: 1px solid {border};
            }}
        """)
        self.title_label.setStyleSheet(
            f"color: {title_color}; background: transparent; letter-spacing: 2px;"
        )
        # Re-apply status badge neutral style for theme
        if not hasattr(self, '_status_ok') or self._status_ok is None:
            if self._is_dark:
                self.status_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: #2d2d30;
                        color: #a0a0a0;
                        border: 1px solid #404040;
                        border-radius: 4px;
                        padding: 3px 14px;
                        font-size: 11px;
                        font-weight: bold;
                        letter-spacing: 0.5px;
                    }}
                """)
            else:
                self.status_label.setStyleSheet(f"""
                    QLabel {{
                        background-color: #f6f8fa;
                        color: #636e7b;
                        border: 1px solid #d0d7de;
                        border-radius: 4px;
                        padding: 3px 14px;
                        font-size: 11px;
                        font-weight: bold;
                        letter-spacing: 0.5px;
                    }}
                """)

    def _apply_toggle_style(self):
        if self._is_dark:
            self.theme_btn.setStyleSheet("""
                QPushButton {
                    background-color: #e8e8ec;
                    color: #1a1a1a;
                    border: 1px solid #c8c8cc;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #ffffff; }
                QPushButton:pressed { background-color: #d0d0d8; }
            """)
        else:
            self.theme_btn.setStyleSheet("""
                QPushButton {
                    background-color: #242428;
                    color: #ffffff;
                    border: 1px solid #404040;
                    border-radius: 3px;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover { background-color: #2d2d30; }
                QPushButton:pressed { background-color: #1a1a1e; }
            """)

    def _toggle_theme(self):
        self._is_dark = not self._is_dark
        self._sync_button_label()
        self.theme_toggled.emit(self._is_dark)

    def _sync_button_label(self):
        self.theme_btn.setText("Light" if self._is_dark else "Dark")
        self._apply_toggle_style()
        self._apply_bar_style()

    def set_theme(self, is_dark: bool):
        """Called externally to sync button state without emitting signal."""
        self._is_dark = is_dark
        self._sync_button_label()
        if self._status_ok is not None:
            self.set_status(self.status_label.text(), self._status_ok)

    def set_status(self, text: str, ok: bool = True):
        self._status_ok = ok
        if ok:
            color  = "#1a7f37" if not self._is_dark else "#3ddc84"
            border = "#1a7f37" if not self._is_dark else "#1a6040"
            bg     = "#dafbe1" if not self._is_dark else "#0d1f15"
        else:
            color  = "#cf222e" if not self._is_dark else "#f05050"
            border = "#cf222e" if not self._is_dark else "#602020"
            bg     = "#ffebe9" if not self._is_dark else "#1f0d0d"
        self.status_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {color};
                border: 1px solid {border};
                border-radius: 4px;
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
