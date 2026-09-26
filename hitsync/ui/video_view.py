"""Preview surfaces: the video view (also used fullscreen) and the song view."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QSlider, QVBoxLayout, QWidget

from . import theme
from .widgets import button, label


def scaled_font(base: QFont, factor: float, bold: bool = False) -> QFont:
    """A copy of ``base`` scaled by ``factor`` (works for point or pixel sizes)."""
    f = QFont(base)
    if f.pointSizeF() > 0:
        f.setPointSizeF(max(6.0, f.pointSizeF() * factor))
    elif f.pixelSize() > 0:
        f.setPixelSize(max(8, int(round(f.pixelSize() * factor))))
    f.setBold(bold)
    return f


def to_qimage(frame: np.ndarray) -> QImage:
    h, w = frame.shape[:2]
    frame = np.ascontiguousarray(frame)
    img = QImage(frame.data, w, h, frame.strides[0], QImage.Format_BGR888)
    img._keep = frame                      # the buffer must outlive the QImage
    return img


class VideoView(QWidget):
    """Draws the latest frame, aspect-fit on black; a friendly placeholder
    when there is nothing to show. Double-click = fullscreen."""

    double_clicked = Signal()
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 180)
        self.img: QImage | None = None
        self.placeholder = ("Drop your gameplay video and a song",
                            "That's it: beats, hits and combos are found automatically.")
        self.badge = ""
        self.smooth = True
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def set_frame(self, frame: np.ndarray | None):
        self.img = None if frame is None else to_qimage(frame)
        self.update()

    def set_placeholder(self, title, sub=""):
        self.img = None
        self.placeholder = (title, sub)
        self.update()

    def frame_rect(self) -> QRectF:
        if self.img is None:
            return QRectF(self.rect())
        iw, ih = self.img.width(), self.img.height()
        s = min(self.width() / iw, self.height() / ih)
        w, h = iw * s, ih * s
        return QRectF((self.width() - w) / 2, (self.height() - h) / 2, w, h)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#000000"))
        if self.img is not None:
            if self.smooth:
                p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(self.frame_rect(), self.img)
        else:
            title, sub = self.placeholder
            p.setPen(QColor(theme.TEXT))
            p.setFont(scaled_font(self.font(), 1.6, bold=True))
            r = self.rect().adjusted(20, 0, -20, -20)
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, title)
            if sub:
                p.setPen(QColor(theme.MUTED))
                p.setFont(self.font())
                p.drawText(self.rect().adjusted(20, 60, -20, 0), Qt.AlignCenter | Qt.TextWordWrap,
                           sub)
        if self.badge:
            p.setPen(QColor("#ffffff"))
            f = QFont(self.font())
            f.setBold(True)
            p.setFont(f)
            fm = p.fontMetrics()
            w = fm.horizontalAdvance(self.badge) + 16
            rr = QRectF(10, 10, w, fm.height() + 8)
            path = QPainterPath()
            path.addRoundedRect(rr, 6, 6)
            p.fillPath(path, QColor(0, 0, 0, 150))
            p.drawText(rr, Qt.AlignCenter, self.badge)
        p.end()

    def mouseDoubleClickEvent(self, _):
        self.double_clicked.emit()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit()


class FullscreenPlayer(QWidget):
    """Borderless fullscreen preview with controls that hide when idle."""

    toggle_play = Signal()
    seek = Signal(float)
    closed = Signal()

    def __init__(self):
        super().__init__(None, Qt.Window | Qt.FramelessWindowHint)
        self.setWindowTitle("Hit-Sync preview")
        self.setStyleSheet("background: black;")
        self.setMouseTracking(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = VideoView()
        self.view.setMouseTracking(True)
        self.view.double_clicked.connect(self.close)
        self.view.clicked.connect(self.toggle_play.emit)
        lay.addWidget(self.view, 1)
        self.bar = QWidget(self)
        self.bar.setStyleSheet("background: rgba(10,10,14,200); border-radius: 10px;")
        b = QHBoxLayout(self.bar)
        b.setContentsMargins(12, 8, 12, 8)
        self.play_btn = button("▶", "Play / pause (Space)", flat=True,
                               slot=lambda: self.toggle_play.emit())
        self.play_btn.setStyleSheet("font-size: 20px; color: white;")
        self.time = label("0:00.00")
        self.time.setStyleSheet("color: white; font-family: monospace;")
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.sliderMoved.connect(lambda v: self.seek.emit(v / 1000.0))
        exit_btn = button("✕  Exit fullscreen", "Esc", flat=True, slot=self.close)
        exit_btn.setStyleSheet("color: white;")
        b.addWidget(self.play_btn)
        b.addWidget(self.time)
        b.addWidget(self.scrub, 1)
        b.addWidget(exit_btn)
        self._hide = QTimer(self, singleShot=True, interval=2200, timeout=self._idle)

    def resizeEvent(self, e):
        w = min(self.width() - 60, 1100)
        self.bar.setGeometry((self.width() - w) // 2, self.height() - 80, w, 56)
        super().resizeEvent(e)

    def show_controls(self):
        self.bar.show()
        self.setCursor(Qt.ArrowCursor)
        self._hide.start()

    def _idle(self):
        self.bar.hide()
        self.setCursor(Qt.BlankCursor)

    def mouseMoveEvent(self, e):
        self.show_controls()
        super().mouseMoveEvent(e)

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Escape, Qt.Key_F):
            self.close()
        elif e.key() == Qt.Key_Space:
            self.toggle_play.emit()
            self.show_controls()
        else:
            super().keyPressEvent(e)

    def set_state(self, playing, t_rel, dur):
        self.play_btn.setText("⏸" if playing else "▶")
        self.time.setText(f"{theme.fmt_time(t_rel)} / {theme.fmt_time(dur)}")
        if not self.scrub.isSliderDown() and dur > 0:
            self.scrub.setValue(int(1000 * t_rel / dur))

    def closeEvent(self, e):
        self.closed.emit()
        super().closeEvent(e)


class SongView(QWidget):
    """A song's loudness with beats, bar lines, start candidates, where it
    ends / switches to the next song, and the playhead. Click to jump; the
    candidates are drawn as flags."""

    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.curve_t = np.zeros(0)
        self.level = np.zeros(0)
        self.beats = np.zeros(0)
        self.downbeats = np.zeros(0)
        self.duration = 1.0
        self.playhead = 0.0
        self.start = None               # chosen music start
        self.candidates: list = []      # (time, label)
        self.taps: list = []
        self.title = ""
        self.end = None                 # where it switches to the next song / ends
        self.end_label = ""
        self.end_auto = True

    def set_song(self, curve_t, level, beats, downbeats, duration, start, candidates, title,
                 end=None, end_label="", end_auto=True):
        self.curve_t, self.level = np.asarray(curve_t), np.asarray(level)
        self.beats, self.downbeats = np.asarray(beats), np.asarray(downbeats)
        self.duration = max(1e-3, duration)
        self.start, self.candidates, self.title = start, candidates, title
        self.end, self.end_label, self.end_auto = end, end_label, end_auto
        self.update()

    def x_of(self, t):
        return 12 + (self.width() - 24) * t / self.duration

    def t_of(self, x):
        return float(np.clip((x - 12) / max(1, self.width() - 24) * self.duration, 0,
                             self.duration))

    def mousePressEvent(self, e):
        self.seek.emit(self.t_of(e.position().x()))

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self.seek.emit(self.t_of(e.position().x()))

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))
        h = self.height()
        top, base = 34, h - 22
        if len(self.level):
            n = max(2, (self.width() - 24) // 2)
            ts = np.linspace(0, self.duration, n)
            idx = np.clip(np.searchsorted(self.curve_t, ts), 0, len(self.level) - 1)
            vals = self.level[idx]
            path = QPainterPath(QPointF(self.x_of(0), base))
            for t, v in zip(ts, vals):
                path.lineTo(self.x_of(t), base - v * (base - top))
            path.lineTo(self.x_of(self.duration), base)
            p.fillPath(path, QColor("#34405a"))
        # beats and bar lines
        px_per_beat = (self.width() - 24) / self.duration * (
            float(np.median(np.diff(self.beats))) if len(self.beats) > 1 else 1.0)
        if px_per_beat > 5:
            p.setPen(QPen(QColor("#4aa3ff"), 1))
            for b in self.beats:
                x = self.x_of(b)
                p.drawLine(QPointF(x, base - 8), QPointF(x, base))
        p.setPen(QPen(QColor("#79b8ff"), 1))
        for b in self.downbeats:
            x = self.x_of(b)
            p.drawLine(QPointF(x, base - 16), QPointF(x, base))
        # candidates
        f = scaled_font(self.font(), 0.85)
        p.setFont(f)
        for t, text in self.candidates:
            x = self.x_of(t)
            p.setPen(QPen(QColor(theme.MUTED), 1, Qt.DashLine))
            p.drawLine(QPointF(x, top - 4), QPointF(x, base))
            p.setPen(QColor(theme.MUTED))
            p.drawText(QPointF(x + 4, top + 8), f"{text} {theme.fmt_short(t)}")
        if self.start is not None:
            x = self.x_of(self.start)
            p.setPen(QPen(QColor(theme.BAD), 2))
            p.drawLine(QPointF(x, 18), QPointF(x, base))
            p.setPen(QColor(theme.BAD))
            f.setBold(True)
            p.setFont(f)
            p.drawText(QPointF(x + 4, 28), "MUSIC STARTS")
        if self.end is not None and self.end < self.duration - 1e-3:
            x = self.x_of(self.end)
            p.fillRect(QRectF(x, 18, self.x_of(self.duration) - x, base - 18),
                       QColor(0, 0, 0, 110))                # not played
            color = QColor(theme.ACCENT)
            p.setPen(QPen(color, 2, Qt.DashLine if self.end_auto else Qt.SolidLine))
            p.drawLine(QPointF(x, 18), QPointF(x, base))
            p.setPen(color)
            f.setBold(True)
            p.setFont(f)
            text = f"✂ {self.end_label}" + (" (auto)" if self.end_auto else "")
            w = p.fontMetrics().horizontalAdvance(text)
            p.drawText(QPointF(x - w - 4 if x + w + 8 > self.width() else x + 4, 28), text)
        for t in self.taps[-12:]:
            x = self.x_of(t)
            p.setPen(QPen(QColor(theme.WARN), 2))
            p.drawLine(QPointF(x, base - 24), QPointF(x, base))
        # playhead
        x = self.x_of(self.playhead)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(QPointF(x, 6), QPointF(x, h - 6))
        p.setPen(QColor(theme.TEXT))
        f.setBold(True)
        p.setFont(f)
        p.drawText(QPointF(12, 16), self.title)
        p.setPen(QColor(theme.MUTED))
        p.drawText(QPointF(12, h - 6), theme.fmt_time(self.playhead))
        p.drawText(QPointF(self.width() - 60, h - 6), theme.fmt_short(self.duration))
        p.end()
