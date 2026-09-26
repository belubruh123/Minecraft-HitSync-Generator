"""Small reusable widgets: segmented buttons, value sliders, drop cards, a
wrapping flow layout, an eliding label and the wheel guard (the mouse wheel
scrolls panels instead of changing the slider under the pointer)."""
from __future__ import annotations

import os

from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QPainter, QPen, QColor
from PySide6.QtWidgets import (QAbstractSpinBox, QButtonGroup, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QLayout, QLineEdit, QPushButton, QSlider, QVBoxLayout,
                               QWidget, QSizePolicy)

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
        self.name = ElidedLabel(text)
        self.name.setFixedWidth(label_width)          # aligned; long names end in "…"
        if tip:
            self.name.setToolTip(tip)
            self.setToolTip(tip)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.steps)
        self.slider.setMinimumWidth(40)
        self.box = QLineEdit()
        self.box.setFixedWidth(58)
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
    """A big 'drop a file here / click to choose' card with a status line,
    and a ✕ (plus a right-click menu) to remove what it holds."""

    clicked = Signal()
    dropped = Signal(list)
    remove = Signal(object, bool)    # where (global QPoint), from the ✕ (not right-click)

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
        self.title = ElidedLabel(title, title=True)
        self.status = ElidedLabel(hint, muted=True)
        col.addWidget(self.title)
        col.addWidget(self.status)
        lay.addLayout(col, 1)
        self.extra = QHBoxLayout()
        lay.addLayout(self.extra)
        self.remove_btn = button("✕", "Remove", flat=True,
                                 slot=lambda: self.remove.emit(self.remove_btn.mapToGlobal(
                                     self.remove_btn.rect().bottomLeft()), True))
        self.remove_btn.setFixedSize(30, 30)
        self.remove_btn.setVisible(False)
        self.extra.addWidget(self.remove_btn)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(
            lambda pos: self.remove_btn.isVisible() and self.remove.emit(self.mapToGlobal(pos),
                                                                          False))
        self._defaults = (title, hint)
        self._hover = False
        self.progress = 0.0

    def reset(self):
        """Back to the empty 'drop a file here' state."""
        self.title.setText(self._defaults[0])
        self.status.setText(self._defaults[1])
        self.progress = 0.0
        self.set_removable(False)
        self.update()

    def set_removable(self, on: bool, tip: str = "Remove"):
        self.remove_btn.setToolTip(tip)
        self.remove_btn.setVisible(on)

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


class ElidedLabel(QLabel):
    """A one-line label that shrinks with "…" instead of forcing the window
    wider (the full text is in the tooltip)."""

    def __init__(self, text="", muted=False, title=False, parent=None):
        super().__init__(text, parent)
        if muted:
            self.setProperty("muted", True)
        if title:
            self.setProperty("title", True)
        # full width when there is room, down to a few letters when not
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self._full = text

    def setText(self, text):
        self._full = text
        self.setToolTip(text if text else "")
        super().setText(text)
        self.update()

    def minimumSizeHint(self):
        return QSize(24, super().minimumSizeHint().height())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setPen(self.palette().color(self.foregroundRole()))
        p.setFont(self.font())
        r = self.contentsRect()
        text = self.fontMetrics().elidedText(self._full, Qt.ElideRight, r.width())
        p.drawText(r, int(self.alignment() | Qt.AlignVCenter), text)
        p.end()


class FlowLayout(QLayout):
    """Lays widgets out left to right and wraps onto the next line when the
    row is full (so a row of buttons never forces the window wider)."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do(rect, apply=True)

    def sizeHint(self):
        w = h = 0
        for it in self._items:
            if it.widget() is not None and it.widget().isHidden():
                continue
            sz = it.sizeHint()
            w += sz.width() + (self._spacing if w else 0)
            h = max(h, sz.height())
        m = self.contentsMargins()
        return QSize(w + m.left() + m.right(), h + m.top() + m.bottom())

    def minimumSize(self):
        size = QSize()
        for it in self._items:
            size = size.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do(self, rect, apply):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for it in self._items:
            wdg = it.widget()
            if wdg is not None and wdg.isHidden():
                continue
            sz = it.sizeHint()
            if x > r.x() and x + sz.width() > r.right() + 1:
                x, y = r.x(), y + line_h + self._spacing
                line_h = 0
            if apply:
                it.setGeometry(QRect(QPoint(x, y), sz))
            x += sz.width() + self._spacing
            line_h = max(line_h, sz.height())
        return y + line_h - rect.y() + m.bottom()


class WheelGuard(QObject):
    """App-wide: the mouse wheel over a slider, dropdown or spin box that you
    haven't clicked scrolls the panel behind it instead of changing its
    value. Click it first (focus) and the wheel adjusts it as usual."""

    TYPES = (QSlider, QComboBox, QAbstractSpinBox)

    def eventFilter(self, obj, e):
        t = e.type()
        if t == QEvent.Wheel and isinstance(obj, self.TYPES) and not obj.hasFocus():
            e.ignore()
            return True                    # ignored: Qt hands it to the parent (scroll area)
        if t == QEvent.Polish and isinstance(obj, self.TYPES) and \
                obj.focusPolicy() == Qt.WheelFocus:
            obj.setFocusPolicy(Qt.StrongFocus)  # the wheel alone never focuses it
        return False


def install_wheel_guard(app) -> WheelGuard:
    guard = getattr(app, "_hitsync_wheel_guard", None)
    if guard is None:
        guard = WheelGuard(app)
        app.installEventFilter(guard)
        app._hitsync_wheel_guard = guard
    return guard


def shrinkable(combo: QComboBox, chars: int = 8) -> QComboBox:
    """A dropdown that can get narrower than its longest entry."""
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(chars)
    return combo
