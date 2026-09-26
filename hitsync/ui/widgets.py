"""Small reusable widgets: segmented buttons, value sliders, drop cards."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (QButtonGroup, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSlider, QVBoxLayout, QWidget, QSizePolicy)

from . import theme

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v", ".wmv"}
AUDIO_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma"}


def kind_of(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    return "video" if ext in VIDEO_EXT else "audio" if ext in AUDIO_EXT else None


def button(text, tip="", primary=False, chip=False, flat=False, checkable=False, slot=None):
    b = QPushButton(text)
    if tip:
        b.setToolTip(tip)
    for prop, on in (("primary", primary), ("chip", chip), ("flat", flat)):
        if on:
            b.setProperty(prop, True)
    b.setCheckable(checkable or chip and checkable)
    b.setCursor(Qt.PointingHandCursor)
    if slot:
        b.clicked.connect(slot)
    return b


def label(text="", muted=False, title=False, h2=False, wrap=False):
    lb = QLabel(text)
    if muted:
        lb.setProperty("muted", True)
    if title:
        lb.setProperty("title", True)
    if h2:
        lb.setProperty("h2", True)
    lb.setWordWrap(wrap)
    return lb


class Segmented(QWidget):
    """A row of mutually exclusive buttons; emits the chosen value."""

    changed = Signal(object)

    def __init__(self, options, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.values = []
        for i, (value, text) in enumerate(options):
            b = QPushButton(text)
            b.setCheckable(True)
            b.setProperty("seg", True)
            if i == 0:
                b.setProperty("segfirst", True)
            if i == len(options) - 1:
                b.setProperty("seglast", True)
            b.setCursor(Qt.PointingHandCursor)
            self.group.addButton(b, i)
            lay.addWidget(b)
            self.values.append(value)
        self.group.idClicked.connect(lambda i: self.changed.emit(self.values[i]))

    def set(self, value):
        for i, v in enumerate(self.values):
            if str(v) == str(value):
                self.group.button(i).setChecked(True)
                return
        checked = self.group.checkedButton()
        if checked:                              # custom value: none selected
            self.group.setExclusive(False)
            checked.setChecked(False)
            self.group.setExclusive(True)


class ValueSlider(QWidget):
    """Label + slider + a box you can type an exact value into.

    Enter / Tab / clicking away applies a typed value, Esc undoes. Typed
    values can go past the slider range and aren't rounded to its steps.
    Percent fields accept "25" or "25%"; a comma works as decimal point.
    """

    changed = Signal(float)

    def __init__(self, text, lo, hi, value=0.0, fmt="{:.2f}", steps=200, integer=False,
                 percent=False, tip="", label_width=150, parent=None):
        super().__init__(parent)
        self.lo, self.hi, self.steps = float(lo), float(hi), int(steps)
        self.fmt, self.integer, self.percent = fmt, integer, percent
        self._value = float(value)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        self.name = QLabel(text)
        self.name.setMinimumWidth(label_width)
        if tip:
            self.name.setToolTip(tip)
            self.setToolTip(tip)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.steps)
        self.box = QLineEdit()
        self.box.setFixedWidth(64)
        self.box.setAlignment(Qt.AlignRight)
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.box)
        self.slider.valueChanged.connect(self._slid)
        self.box.editingFinished.connect(self._typed)
        self.set(value)

    def _to_pos(self, v):
        if self.hi <= self.lo:
            return 0
        return int(round((min(max(v, self.lo), self.hi) - self.lo) / (self.hi - self.lo)
                         * self.steps))

    def _show(self):
        v = self._value
        if self.percent:
            text = f"{v * 100:.0f}%"
        else:
            text = self.fmt.format(v)
            try:
                if abs(float(text) - v) > 1e-9:
                    text = f"{v:.4f}".rstrip("0").rstrip(".")
            except ValueError:
                pass
        self.box.setText(text)

    def set(self, v, emit=False):
        self._value = int(round(v)) if self.integer else float(v)
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(self._value))
        self.slider.blockSignals(False)
        self._show()
        if emit:
            self.changed.emit(float(self._value))

    def value(self) -> float:
        return float(self._value)

    def _slid(self, pos):
        v = self.lo + (self.hi - self.lo) * pos / max(1, self.steps)
        self._value = int(round(v)) if self.integer else v
        self._show()
        self.changed.emit(float(self._value))

    def _typed(self):
        text = self.box.text().strip().replace(",", ".")
        try:
            is_pct = text.endswith("%")
            v = float(text.rstrip("%").strip())
            if self.percent and (is_pct or abs(v) > 1.0):
                v /= 100.0
            if self.lo >= 0:
                v = max(v, 0.0)
            v = int(round(v)) if self.integer else v
        except ValueError:
            self.box.setStyleSheet(f"border-color: {theme.BAD};")
            self._show()
            return
        self.box.setStyleSheet("")
        if abs(v - self._value) > 1e-12:
            self.set(v, emit=True)
        else:
            self._show()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self._show()
            self.box.clearFocus()
            return
        super().keyPressEvent(e)


class DropCard(QFrame):
    """A big 'drop a file here / click to choose' card with a status line."""

    clicked = Signal()
    dropped = Signal(list)

    def __init__(self, icon, title, hint, parent=None):
        super().__init__(parent)
        self.setProperty("card", True)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(58)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 8)
        self.icon = QLabel(icon)
        self.icon.setStyleSheet("font-size: 24px;")
        lay.addWidget(self.icon)
        col = QVBoxLayout()
        col.setSpacing(1)
        self.title = label(title, title=True)
        self.status = label(hint, muted=True)
        col.addWidget(self.title)
        col.addWidget(self.status)
        lay.addLayout(col, 1)
        self.extra = QHBoxLayout()
        lay.addLayout(self.extra)
        self._hover = False
        self.progress = 0.0

    def set_state(self, title=None, status=None, progress=None):
        if title is not None:
            self.title.setText(title)
        if status is not None:
            self.status.setText(status)
        if progress is not None:
            self.progress = progress
            self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._hover = True
            self.setStyleSheet(f"QFrame {{ border: 2px dashed {theme.ACCENT}; }}")

    def dragLeaveEvent(self, e):
        self._hover = False
        self.setStyleSheet("")

    def dropEvent(self, e):
        self._hover = False
        self.setStyleSheet("")
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.dropped.emit(paths)
            e.acceptProposedAction()

    def paintEvent(self, e):
        super().paintEvent(e)
        if 0 < self.progress < 1:
            p = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(theme.ACCENT), 3)
            p.setPen(pen)
            w = self.width() - 24
            p.drawLine(12, self.height() - 4, 12 + int(w * self.progress), self.height() - 4)
            p.end()
