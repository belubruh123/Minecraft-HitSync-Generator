"""Interactive timeline: music beats / the edit / video hits.

Mouse:
  click marker ............ select (click empty space + drag = pan)
  double-click ............ add beat (MUSIC) or hit (VIDEO)
  right-click marker ...... delete it
  drag a caption/effect ... move its edges (snap to beats) on the EDIT track
  wheel / scrollbar ....... scroll; Ctrl/Cmd+wheel = zoom around the cursor
Keys: Delete/Backspace = delete, S = split combo, M = merge with previous,
  F = zoom to fit, E = hit effects on/off for the selected hit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QScrollBar, QWidget

from . import theme

TRACK_BG = "#191c24"
GRID = "#252a35"
BEAT = "#4aa3ff"
DROP = "#ff4d6d"
INTRO = "#3b2d5c"
REMOVED = QColor(0, 0, 0, 130)
SEG_COLORS = {"intro": "#7a5cc7", "leadin": "#9b6cff", "lead": "#3b6fb6", "bridge": "#3b6fb6",
              "combo": "#2e9d6a", "trim": "#d6b43a", "tail": "#4a5060", "outro": "#4a5060"}
RULER_H = 20
TRACKS = [("audio", "MUSIC", 70), ("edit", "EDIT", 64), ("video", "VIDEO", 86)]
PICK_PX = 7


class Timeline(QWidget):
    changed = Signal(str)            # markers edited
    selected_changed = Signal(object)
    clicked = Signal(str, float)     # track, time
    overlay_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.project = None
        self.pps = 40.0
        self.offset = 0.0
        self.selected: Optional[tuple] = None
        self.playhead: Optional[float] = None
        self.playhead_src: Optional[float] = None
        self.score_t = np.zeros(0)
        self.score = np.zeros(0)
        self._pan = None
        self._drag_edge = None
        self._track_y = {}
        y = RULER_H
        for key, _, h in TRACKS:
            self._track_y[key] = (y, y + h)
            y += h + 2
        self.setFixedHeight(y + 2)
        self.setFocusPolicy(Qt.ClickFocus)
        self.setMouseTracking(True)
        # a horizontal scrollbar the window puts under the timeline (pixels)
        self.hbar = QScrollBar(Qt.Horizontal)
        self.hbar.valueChanged.connect(self._scrolled)

    # ---------------------------------------------------------- coordinates
    def x_of(self, t):
        return (t - self.offset) * self.pps

    def t_of(self, x):
        return self.offset + x / self.pps

    def track_at(self, y):
        for key, (y0, y1) in self._track_y.items():
            if y0 <= y <= y1:
                return key
        return None

    def set_score(self, times, score):
        self.score_t, self.score = np.asarray(times, float), np.asarray(score, float)

    def zoom_fit(self):
        p = self.project
        if p is None:
            return
        dur = max(p.music_duration, p.video_duration, 10.0)
        self.offset = 0.0
        self.pps = max(1.0, (self.width() - 10) / dur)
        self.update()

    def _zoom(self, x, factor):
        t = self.t_of(x)
        self.pps = float(np.clip(self.pps * factor, 1.0, 2000.0))
        self.offset = max(0.0, t - x / self.pps)
        self.update()

    def wheelEvent(self, e):
        d = e.angleDelta().y() or e.angleDelta().x()
        if not d:
            return
        if e.modifiers() & Qt.ControlModifier:          # Ctrl (Cmd on a Mac) + wheel
            self._zoom(e.position().x(), 1.2 if d > 0 else 1 / 1.2)
        else:                                           # wheel / trackpad: scroll
            self.offset = min(self._max_offset(), max(0.0, self.offset - d / 120 * 80 / self.pps))
            self.update()

    # ---------------------------------------------------------- scrollbar
    def _span(self) -> float:
        p = self.project
        if p is None:
            return 10.0
        return max(p.music_duration, p.video_duration, 10.0) + 2.0

    def _max_offset(self) -> float:
        return max(0.0, self._span() - self.width() / self.pps)

    def _sync_bar(self):
        bar = self.hbar
        bar.blockSignals(True)
        bar.setRange(0, int(round(self._max_offset() * self.pps)))
        bar.setPageStep(max(1, self.width()))
        bar.setSingleStep(max(1, self.width() // 20))
        bar.setValue(int(round(self.offset * self.pps)))
        bar.blockSignals(False)

    def _scrolled(self, v):
        self.offset = v / self.pps
        self.update()

    def update(self, *args):
        # every view change (scroll, zoom, new project) goes through here, so
        # the scrollbar follows even while the timeline is scrolled out of view
        self._sync_bar()
        super().update(*args)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._sync_bar()

    # -------------------------------------------------------------- picking
    def _nearest(self, track, x):
        m = self.project.markers
        if track == "audio" and m.beats:
            xs = self.x_of(np.asarray(m.beats))
            i = int(np.argmin(np.abs(xs - x)))
            return ("beat", i) if abs(xs[i] - x) <= PICK_PX else None
        if track == "video" and m.hits:
            xs = self.x_of(np.array([h.t for h in m.hits]))
            i = int(np.argmin(np.abs(xs - x)))
            return ("hit", i) if abs(xs[i] - x) <= PICK_PX else None
        return None

    def _overlay_items(self):
        p = self.project
        return [("text", i, it) for i, it in enumerate(p.texts)] + \
               [("effect", i, it) for i, it in enumerate(p.effects)]

    def _overlay_rows(self):
        y0, y1 = self._track_y["edit"]
        return y1 - 20, y1 - 3

    def _edge_at(self, x, y):
        ya, yb = self._overlay_rows()
        if not (ya - 2 <= y <= yb + 2):
            return None
        for kind, i, it in self._overlay_items():
            for edge in ("start", "end"):
                if abs(self.x_of(getattr(it, edge)) - x) <= 5:
                    return (kind, i, edge)
            if self.x_of(it.start) < x < self.x_of(it.end):
                return (kind, i, "body")
        return None

    def mousePressEvent(self, e):
        if self.project is None:
            return
        x, y = e.position().x(), e.position().y()
        if e.button() == Qt.RightButton or (e.button() == Qt.LeftButton
                                            and e.modifiers() & Qt.ControlModifier
                                            and self._mac()):
            pick = self._nearest(self.track_at(y), x)
            if pick:
                self.selected = pick
                self.delete_selected()
            return
        track = self.track_at(y)
        edge = self._edge_at(x, y) if track == "edit" else None
        if edge:
            self._drag_edge = edge
            self.selected = (edge[0], edge[1])
            self.selected_changed.emit(self.selected)
            self.update()
            return
        pick = self._nearest(track, x)
        if pick:
            self.selected = pick
            self.selected_changed.emit(pick)
            self.update()
        else:
            self._pan = (x, self.offset, False)
        if track:
            self.clicked.emit(track, self.t_of(x))

    @staticmethod
    def _mac():
        import sys

        return sys.platform == "darwin"

    def mouseMoveEvent(self, e):
        x = e.position().x()
        if self._drag_edge:
            kind, i, edge = self._drag_edge
            items = self.project.texts if kind == "text" else self.project.effects
            if i < len(items):
                it = items[i]
                t = self._snap(self.t_of(x))
                if edge == "start" and t < it.end - 0.05:
                    it.start = t
                elif edge == "end" and t > it.start + 0.05:
                    it.end = t
                self.update()
            return
        if self._pan:
            x0, off0, _ = self._pan
            self.offset = max(0.0, off0 - (x - x0) / self.pps)
            self._pan = (x0, off0, True)
            self.update()
        elif self.project is not None:
            edge = self._edge_at(x, e.position().y())
            self.setCursor(Qt.SizeHorCursor if edge and edge[2] != "body" else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        if self._drag_edge:
            self._drag_edge = None
            self.overlay_changed.emit()
        self._pan = None

    def _snap(self, t):
        b = self.project.markers.beats
        if not b:
            return t
        arr = np.asarray(b)
        j = int(np.argmin(np.abs(arr - t)))
        return float(arr[j]) if abs(arr[j] - t) * self.pps < 12 else t

    def mouseDoubleClickEvent(self, e):
        if self.project is None:
            return
        track = self.track_at(e.position().y())
        t = self.t_of(e.position().x())
        m = self.project.markers
        if track == "audio":
            self.selected = ("beat", m.add_beat(t))
            self.changed.emit("Added beat")
        elif track == "video":
            from ..video_analysis import snap_to_peak

            t = snap_to_peak(self.project.video, self.project.detect, t)
            self.selected = ("hit", m.add_hit(t, self.project.sync.combo_gap))
            self.changed.emit("Added hit")
        else:
            return
        self.selected_changed.emit(self.selected)
        self.update()

    def keyPressEvent(self, e):
        k = e.key()
        if e.modifiers() & (Qt.ControlModifier | Qt.MetaModifier):
            return super().keyPressEvent(e)
        if k in (Qt.Key_Delete, Qt.Key_Backspace):
            self.delete_selected()
        elif k == Qt.Key_S:
            self.split_selected()
        elif k == Qt.Key_M:
            self.merge_selected()
        elif k == Qt.Key_F:
            self.zoom_fit()
        elif k == Qt.Key_E:
            self.toggle_fx_selected()
        else:
            super().keyPressEvent(e)

    # --------------------------------------------------------------- edits
    def delete_selected(self):
        if not self.selected or self.project is None:
            return
        kind, i = self.selected
        p = self.project
        if kind == "beat":
            p.markers.delete_beat(i)
        elif kind == "hit":
            p.markers.delete_hit(i)
        elif kind == "text" and i < len(p.texts):
            del p.texts[i]
        elif kind == "effect" and i < len(p.effects):
            del p.effects[i]
        self.selected = None
        self.selected_changed.emit(None)
        self.changed.emit(f"Deleted {kind}")
        self.update()

    def split_selected(self):
        if self.selected and self.selected[0] == "hit":
            self.project.markers.split_combo_at(self.selected[1])
            self.changed.emit("Split combo")

    def merge_selected(self):
        if self.selected and self.selected[0] == "hit":
            self.project.markers.merge_with_previous(self.selected[1])
            self.changed.emit("Merged combo")

    def toggle_fx_selected(self):
        if self.selected and self.selected[0] == "hit":
            h = self.project.markers.hits[self.selected[1]]
            default = self.project.render_params.hit_fx_default
            on = default if h.fx is None else h.fx
            h.fx = not on
            self.changed.emit("Hit effects " + ("on" if h.fx else "off"))

    # ------------------------------------------------------------ playhead
    def set_playhead(self, t, src_t=None, follow=False):
        self.playhead, self.playhead_src = t, src_t
        w = max(10, self.width())
        if follow and t is not None and not (self.t_of(0) <= t <= self.t_of(w * 0.9)):
            self.offset = max(0.0, t - (w * 0.1) / self.pps)
        self.update()

    # ------------------------------------------------------------- drawing
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(theme.BG))
        if self.project is None:
            p.end()
            return
        w = self.width()
        t0, t1 = self.t_of(0), self.t_of(w)
        from .video_view import scaled_font

        small = scaled_font(self.font(), 0.8)
        p.setFont(small)
        self._ruler(p, w, t0, t1)
        for key, name, _ in TRACKS:
            y0, y1 = self._track_y[key]
            p.fillRect(QRectF(0, y0, w, y1 - y0), QColor(TRACK_BG))
        self._audio(p, w, t0, t1)
        self._edit(p, w, t0, t1)
        self._video(p, w, t0, t1)
        bold = QFont(small)
        bold.setBold(True)
        p.setFont(bold)
        p.setPen(QColor(theme.MUTED))
        for key, name, _ in TRACKS:
            y0, _ = self._track_y[key]
            p.drawText(QPointF(6, y0 + 12), name)
        self._draw_playhead(p)
        p.end()

    def _ruler(self, p, w, t0, t1):
        span = t1 - t0
        step = next((s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120)
                     if span / s <= w / 70), 300)
        t = np.floor(t0 / step) * step
        p.setPen(QColor(theme.MUTED))
        while t <= t1:
            x = self.x_of(t)
            p.setPen(QColor(GRID))
            p.drawLine(QPointF(x, RULER_H), QPointF(x, self.height()))
            p.setPen(QColor(theme.MUTED))
            text = f"{int(t // 60)}:{t % 60:05.2f}" if step < 1 else f"{int(t // 60)}:{int(t % 60):02d}"
            p.drawText(QPointF(x + 3, 13), text)
            t += step

    def _envelope(self, p, times, values, w, base_y, height, color):
        times = np.asarray(times, float)
        values = np.asarray(values, float)
        if not len(times):
            return
        cols = np.arange(0, w, 2)
        ts = self.t_of(cols)
        inside = (ts >= times[0]) & (ts <= times[-1])
        if not inside.any():
            return
        idx = np.clip(np.searchsorted(times, ts), 0, len(values) - 1)
        vals = values[idx] * inside
        poly = QPolygonF([QPointF(0, base_y)] + [QPointF(float(x), float(base_y - v * height))
                                                 for x, v in zip(cols, vals)]
                         + [QPointF(float(cols[-1]), base_y)])
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(color))
        p.drawPolygon(poly)

    def _audio(self, p, w, t0, t1):
        pr = self.project
        y0, y1 = self._track_y["audio"]
        a = pr.audio
        if a is not None and len(a.level):
            tl = pr.timeline()
            if tl is None:
                self._envelope(p, a.curve_t, a.level, w, y1 - 2, (y1 - y0) - 22, "#2a3550")
            else:                              # each song's curve on the music clock
                for piece in tl.pieces:
                    song = pr._song(piece.song)
                    if song is None:
                        continue
                    ct = song.curve_t + piece.offset
                    sel = (ct >= piece.out_a) & (ct <= piece.out_b)
                    self._envelope(p, ct[sel], song.level[sel], w, y1 - 2, (y1 - y0) - 22,
                                   "#2a3550" if piece.song % 2 == 0 else "#352a50")
                p.setPen(QPen(QColor(theme.WARN), 2))
                for T in tl.handovers:
                    x = self.x_of(T)
                    p.drawLine(QPointF(x, y0), QPointF(x, y1))
                    p.drawText(QPointF(x + 3, y0 + 24), "NEXT SONG")
        if pr.sync.drop_time > 0:
            x = self.x_of(pr.sync.drop_time)
            p.setPen(QPen(QColor(DROP), 2))
            p.drawLine(QPointF(x, y0), QPointF(x, y1))
            p.drawText(QPointF(x + 3, y1 - 4), "MUSIC STARTS")
        beats = np.asarray(pr.markers.beats, float)
        sel = self.selected[1] if self.selected and self.selected[0] == "beat" else -1
        stride = 1 if self.pps > 12 else int(np.ceil(12 / self.pps))
        vis = np.nonzero((beats >= t0 - 1) & (beats <= t1 + 1))[0]
        downs = set(np.round(np.asarray(pr.downbeats, float), 4))
        for i in vis:
            if i % stride and i != sel:
                continue
            x = self.x_of(beats[i])
            is_db = round(beats[i], 4) in downs
            p.setPen(QPen(QColor("#ffffff" if i == sel else ("#79b8ff" if is_db else BEAT)),
                          3 if i == sel else 1))
            p.drawLine(QPointF(x, y0 + (22 if is_db else 30)), QPointF(x, y1 - 2))
        sched = pr.schedule
        if sched is not None:
            placed = sched.placed_hits()
            ty = y0 + 16
            p.setPen(QPen(QColor(theme.GOOD), 2))
            for prev, pl in zip(placed, placed[1:]):
                if not (prev.locked and pl.locked) or pl.out_t < t0 - 1 or prev.out_t > t1 + 1:
                    continue
                k = np.searchsorted(beats, [prev.out_t - 1e-3, pl.out_t - 1e-3])
                if 1 <= k[1] - k[0] <= 2:
                    p.drawLine(QPointF(self.x_of(prev.out_t), ty + 1),
                               QPointF(self.x_of(pl.out_t), ty + 1))
            p.setPen(Qt.NoPen)
            for pl in placed:
                if not (t0 - 1 <= pl.out_t <= t1 + 1):
                    continue
                x = self.x_of(pl.out_t)
                p.setBrush(QColor(theme.GOOD if pl.locked else theme.WARN))
                p.drawPolygon(QPolygonF([QPointF(x - 4, ty), QPointF(x + 4, ty),
                                         QPointF(x, y0 + 23)]))

    def _edit(self, p, w, t0, t1):
        pr = self.project
        sched = pr.schedule
        y0, y1 = self._track_y["edit"]
        if sched is None:
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(0, y0, w, y1 - y0), Qt.AlignCenter, "The edit appears here")
            return
        ty, by = y0 + 16, y1 - 24
        for seg in sched.segments:
            if seg.out_end < t0 or seg.out_start > t1:
                continue
            xa, xb = self.x_of(seg.out_start), self.x_of(seg.out_end)
            p.setPen(QPen(QColor("#111111")))
            p.setBrush(QColor(SEG_COLORS.get(seg.kind, "#555555")))
            p.drawRect(QRectF(xa, ty, max(1.0, xb - xa), by - ty))
            if xb - xa > 34:
                p.setPen(QColor("#e8ecf2"))
                p.drawText(QRectF(xa, ty, xb - xa, by - ty), Qt.AlignCenter, f"{seg.speed:.2f}x")
        p.setPen(QPen(QColor(DROP), 2))
        for c in sched.cuts:
            x = self.x_of(c)
            p.drawLine(QPointF(x, y0 + 12), QPointF(x, by))
        # letterbox envelope along the top
        if sched.letterbox:
            cols = np.arange(0, w, 3)
            amts = [sched.letterbox_amount(self.t_of(x)) for x in cols]
            if max(amts, default=0) > 0:
                poly = QPolygonF([QPointF(0, ty)] + [QPointF(float(x), ty - 9 * a)
                                                      for x, a in zip(cols, amts)]
                                 + [QPointF(float(cols[-1]), ty)])
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#c9ced8"))
                p.drawPolygon(poly)
        # overlays: captions and effect ranges
        ya, yb = self._overlay_rows()
        for kind, i, it in self._overlay_items():
            if it.end < t0 or it.start > t1:
                continue
            xa, xb = self.x_of(it.start), self.x_of(it.end)
            col = QColor("#ffcc33") if kind == "text" else QColor("#ff5d9e")
            is_sel = self.selected == (kind, i)
            path = QPainterPath()
            path.addRoundedRect(QRectF(xa, ya, max(3.0, xb - xa), yb - ya), 4, 4)
            p.fillPath(path, col.darker(160) if not is_sel else col)
            p.setPen(QPen(col, 1))
            p.drawPath(path)
            p.setPen(QColor("#ffffff"))
            name = f'“{it.text}”' if kind == "text" else it.kind.upper()
            p.drawText(QRectF(xa + 4, ya, max(0.0, xb - xa - 8), yb - ya),
                       Qt.AlignVCenter | Qt.AlignLeft, name)
        if sched.duration:
            p.setPen(QPen(QColor("#ffffff"), 1, Qt.DashLine))
            for t in (sched.start, sched.end):
                x = self.x_of(t)
                p.drawLine(QPointF(x, y0), QPointF(x, y1))
        # the part of the song that isn't in the montage
        ya0, _ = self._track_y["audio"]
        p.setPen(Qt.NoPen)
        p.setBrush(REMOVED)
        for a, b in ((0.0, sched.start), (sched.end, 1e6)):
            if b - a > 1e-3 and b > t0 and a < t1:
                p.drawRect(QRectF(self.x_of(max(a, t0 - 1)), ya0,
                                  self.x_of(min(b, t1 + 1)) - self.x_of(max(a, t0 - 1)), y1 - ya0))

    def _video(self, p, w, t0, t1):
        pr = self.project
        y0, y1 = self._track_y["video"]
        s = pr.sync
        if pr.schedule is not None and pr.schedule.segments and \
                pr.schedule.segments[0].kind == "intro":
            seg = pr.schedule.segments[0]
            xa, xb = self.x_of(seg.src_start), self.x_of(seg.src_end)
            p.fillRect(QRectF(xa, y0 + 14, xb - xa, y1 - y0 - 14), QColor(INTRO))
            p.setPen(QColor("#c7b5ff"))
            p.drawText(QPointF(xa + 3, y0 + 26), "SLOW-MO INTRO")
        if pr.schedule is not None:
            p.setPen(Qt.NoPen)
            p.setBrush(REMOVED)
            for a, b in pr.schedule.removed_source_ranges():
                if b < t0 or a > t1:
                    continue
                p.drawRect(QRectF(self.x_of(a), y0 + 14, self.x_of(b) - self.x_of(a), y1 - y0 - 14))
        if len(self.score):
            self._envelope(p, self.score_t, np.clip(self.score, 0, 1.2) / 1.2, w, y1 - 2,
                           (y1 - y0) - 40, "#352f42")
        if pr.video is not None:
            x = self.x_of(pr.video_duration)
            p.setPen(QPen(QColor("#666666"), 1, Qt.DashLine))
            p.drawLine(QPointF(x, y0), QPointF(x, y1))
        hits = pr.markers.hits
        ids = pr.markers.combo_ids()
        sel = self.selected[1] if self.selected and self.selected[0] == "hit" else -1
        for grp in pr.markers.combos():
            a, b = hits[grp[0]].t, hits[grp[-1]].t
            if b < t0 - 1 or a > t1 + 1:
                continue
            col = QColor(theme.COMBO_COLORS[ids[grp[0]] % len(theme.COMBO_COLORS)])
            p.fillRect(QRectF(self.x_of(a) - 2, y0 + 30, self.x_of(b) - self.x_of(a) + 4, 5), col)
            if len(grp) > 1 and self.x_of(b) - self.x_of(a) > 30:
                p.setPen(col)
                p.drawText(QPointF(self.x_of(a), y0 + 48), f"x{len(grp)}")
        status = {pl.hit_index: pl for pl in pr.schedule.placements} if pr.schedule else {}
        default_fx = pr.render_params.hit_fx_default
        for i, h in enumerate(hits):
            if not (t0 - 1 <= h.t <= t1 + 1):
                continue
            x = self.x_of(h.t)
            col = QColor(theme.COMBO_COLORS[ids[i] % len(theme.COMBO_COLORS)])
            is_sel = i == sel
            p.setPen(QPen(QColor("#ffffff") if is_sel else col, 3 if is_sel else 2))
            p.drawLine(QPointF(x, y0 + 40), QPointF(x, y1 - 2))
            pl = status.get(i)
            head = QColor(theme.GOOD) if pl and pl.locked else (
                QColor("#666666") if pl and pl.out_t is None else col)
            r = 5 if is_sel else 4
            p.setPen(QPen(QColor("#ffffff")) if is_sel else Qt.NoPen)
            p.setBrush(head)
            p.drawEllipse(QPointF(x, y0 + 40), r, r)
            fx = default_fx if h.fx is None else h.fx
            if fx and pl and pl.out_t is not None:
                p.setPen(QColor(theme.WARN))
                p.drawText(QPointF(x - 3, y0 + 62), "✦")

    def _draw_playhead(self, p):
        if self.playhead is None:
            return
        x = self.x_of(self.playhead)
        _, y_edit_end = self._track_y["edit"]
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(QPointF(x, 0), QPointF(x, y_edit_end))
        if self.playhead_src is not None:
            xs = self.x_of(self.playhead_src)
            y0, y1 = self._track_y["video"]
            p.setPen(QPen(QColor("#ffffff"), 2, Qt.DashLine))
            p.drawLine(QPointF(xs, y0), QPointF(xs, y1))
