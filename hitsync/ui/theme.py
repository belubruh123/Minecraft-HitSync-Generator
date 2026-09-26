"""Dark theme: colours, stylesheet, and small shared helpers."""
from __future__ import annotations

import sys

from PySide6.QtGui import QColor, QPalette

BG = "#101218"
PANEL = "#171a22"
CARD = "#1e222c"
CARD_HI = "#262b37"
LINE = "#2c3140"
TEXT = "#e8ebf2"
MUTED = "#8a93a6"
ACCENT = "#7c5cff"        # violet: primary actions
ACCENT_HI = "#9479ff"
GOOD = "#3ddc97"
WARN = "#ffb020"
BAD = "#ff5d6c"
COMBO_COLORS = ["#ffb020", "#3ddc97", "#ff6b9a", "#b18cff", "#5ad1ff", "#f5e663",
                "#ff8a4c", "#9be15d"]

MOD = "⌘" if sys.platform == "darwin" else "Ctrl+"


def apply(app):
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in ((QPalette.Window, BG), (QPalette.WindowText, TEXT),
                      (QPalette.Base, CARD), (QPalette.AlternateBase, PANEL),
                      (QPalette.Text, TEXT), (QPalette.Button, CARD), (QPalette.ButtonText, TEXT),
                      (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
                      (QPalette.ToolTipBase, CARD_HI), (QPalette.ToolTipText, TEXT),
                      (QPalette.PlaceholderText, MUTED)):
        pal.setColor(role, QColor(col))
    pal.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(MUTED))
    pal.setColor(QPalette.Disabled, QPalette.WindowText, QColor(MUTED))
    pal.setColor(QPalette.Disabled, QPalette.Text, QColor(MUTED))
    app.setPalette(pal)
    app.setStyleSheet(STYLESHEET)


STYLESHEET = f"""
QWidget {{ font-size: 13px; }}
QMainWindow, QDialog {{ background: {BG}; }}
QToolTip {{ background: {CARD_HI}; color: {TEXT}; border: 1px solid {LINE}; padding: 4px; }}
QPushButton {{
    background: {CARD}; color: {TEXT}; border: 1px solid {LINE}; border-radius: 8px;
    padding: 6px 12px;
}}
QPushButton:hover {{ background: {CARD_HI}; }}
QPushButton:pressed {{ background: {LINE}; }}
QPushButton:disabled {{ color: {MUTED}; background: {PANEL}; }}
QPushButton[primary="true"] {{
    background: {ACCENT}; border: none; color: white; font-weight: 600; padding: 8px 18px;
}}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HI}; }}
QPushButton[primary="true"]:disabled {{ background: {LINE}; color: {MUTED}; }}
QPushButton[chip="true"] {{
    border-radius: 13px; padding: 3px 10px; background: {PANEL}; color: {TEXT};
}}
QPushButton[chip="true"]:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton[flat="true"] {{ background: transparent; border: none; color: {MUTED}; }}
QPushButton[flat="true"]:hover {{ color: {TEXT}; }}
QPushButton[seg="true"] {{ border-radius: 0px; padding: 5px 12px; margin: 0px; }}
QPushButton[seg="true"]:checked {{ background: {ACCENT}; border-color: {ACCENT}; color: white; }}
QPushButton[segfirst="true"] {{ border-top-left-radius: 8px; border-bottom-left-radius: 8px; }}
QPushButton[seglast="true"] {{ border-top-right-radius: 8px; border-bottom-right-radius: 8px; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background: {PANEL}; border: 1px solid {LINE}; border-radius: 6px; padding: 4px 6px;
    color: {TEXT}; selection-background-color: {ACCENT};
}}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {LINE};
    selection-background-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: {LINE}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: white; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px;
}}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{
    background: transparent; color: {MUTED}; padding: 7px 14px; border: none;
    border-bottom: 2px solid transparent; font-weight: 600;
}}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QListWidget {{ background: transparent; border: none; outline: none; }}
QListWidget::item {{ border: none; }}
QListWidget::item:selected {{ background: transparent; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {LINE}; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {LINE}; border-radius: 5px; min-width: 30px; }}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border-radius: 4px;
    border: 1px solid {LINE}; background: {PANEL}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT};
    image: none; }}
QProgressBar {{ background: {PANEL}; border: none; border-radius: 3px; height: 6px;
    text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QFrame[card="true"] {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px; }}
QFrame[panel="true"] {{ background: {PANEL}; border-radius: 12px; }}
QLabel[muted="true"] {{ color: {MUTED}; }}
QLabel[title="true"] {{ font-size: 15px; font-weight: 700; }}
QLabel[h2="true"] {{ font-size: 12px; font-weight: 700; color: {MUTED}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QMenuBar {{ background: {PANEL}; }}
QMenuBar::item:selected {{ background: {CARD_HI}; }}
QMenu {{ background: {CARD}; border: 1px solid {LINE}; }}
QMenu::item:selected {{ background: {ACCENT}; }}
"""


def fmt_time(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{t % 60:05.2f}"


def fmt_short(t: float) -> str:
    t = max(0.0, float(t))
    return f"{int(t // 60)}:{int(t % 60):02d}"
