"""Interactive dual-track timeline: music beats / output edit / video hits.

Mouse:
  click marker ............ select (click empty space + drag = pan)
  double-click ............ add beat (MUSIC track) or hit (VIDEO track)
  right-click marker ...... delete it
  wheel ................... zoom around cursor; Shift+wheel = pan
Keys (canvas focused):  Delete/Backspace = delete selected, S = split combo,
  M = merge with previous combo, F = zoom to fit.
"""
from __future__ import annotations

import tkinter as tk
from typing import Callable, Optional

import numpy as np

BG = "#15171c"
TRACK_BG = "#1d2027"
GRID = "#2a2e37"
TEXT = "#9aa3b2"
BEAT = "#4aa3ff"
DROP = "#ff4d6d"
INTRO = "#3b2d5c"
REMOVED = "#0b0c0f"
SELECT = "#ffffff"
COMBO_COLORS = ["#ffb020", "#3ddc97", "#ff6b9a", "#b18cff", "#5ad1ff", "#f5e663",
                "#ff8a4c", "#9be15d"]
SEG_COLORS = {"intro": "#7a5cc7", "lead": "#3b6fb6", "bridge": "#3b6fb6", "combo": "#2e9d6a",
              "trim": "#d6b43a", "tail": "#4a5060", "outro": "#4a5060"}

RULER_H = 22
TRACKS = [("audio", "MUSIC · beats (music time)", 82),
          ("edit", "EDIT · output schedule (music time)", 52),
          ("video", "VIDEO · hits & combos (source time)", 100)]
PICK_PX = 7


class Timeline(tk.Canvas):
    def __init__(self, master, project, on_change: Callable[[str], None],
                 on_select: Callable[[Optional[tuple]], None],
                 on_click: Callable[[str, float], None],
                 snap_hit: Callable[[float], float], scale: float = 1.0, **kw):
        # tk.Canvas works in physical pixels; customtkinter widgets are DPI
        # scaled, so every canvas dimension goes through self.u().
        self.scale = max(0.5, float(scale))
        self.ruler_h = self.u(RULER_H)
        height = self.ruler_h + sum(self.u(h) + self.u(2) for _, _, h in TRACKS) + self.u(4)
        super().__init__(master, bg=BG, height=height, highlightthickness=0, **kw)
        self.project = project
        self.on_change, self.on_select, self.on_click = on_change, on_select, on_click
        self.snap_hit = snap_hit
        self.pps = 40.0          # pixels per second
        self.offset = 0.0        # seconds at left edge
        self.selected: Optional[tuple] = None   # ("beat"|"hit", index)
        self.playhead: Optional[float] = None    # output (music) time
        self.playhead_src: Optional[float] = None  # matching source time
        self.score_times = np.zeros(0)
        self.score = np.zeros(0)
        self._pan = None
        self._track_y = {}
        y = self.ruler_h
        for key, _, h in TRACKS:
            self._track_y[key] = (y, y + self.u(h))
            y += self.u(h) + self.u(2)
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<Button-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_pan", None))
        self.bind("<Double-Button-1>", self._double)
        self.bind("<Button-3>", self._right)
        self.bind("<MouseWheel>", self._wheel)
        self.bind("<Shift-MouseWheel>", self._shift_wheel)
        self.bind("<Button-4>", lambda e: self._zoom(e.x, 1.2))   # X11
        self.bind("<Button-5>", lambda e: self._zoom(e.x, 1 / 1.2))
        for key in ("<Delete>", "<BackSpace>"):
            self.bind(key, lambda e: self.delete_selected())
        # plain keys only: Ctrl+S etc. belong to the app's menu shortcuts
        plain = lambda fn: (lambda e: None if e.state & 0x4 else fn())  # noqa: E731
        self.bind("<KeyPress-s>", plain(self.split_selected))
        self.bind("<KeyPress-m>", plain(self.merge_selected))
        self.bind("<KeyPress-f>", plain(self.zoom_fit))

    # ---------------------------------------------------------- coordinates
    def u(self, v: float) -> int:
        return int(round(v * self.scale))

    def x_of(self, t):
        return (t - self.offset) * self.pps

    def t_of(self, x):
        return self.offset + x / self.pps

    def track_at(self, y) -> Optional[str]:
        for key, (y0, y1) in self._track_y.items():
            if y0 <= y <= y1:
                return key
        return None

    def set_score(self, times, score):
        self.score_times = np.asarray(times, float)
        self.score = np.asarray(score, float)

    def zoom_fit(self):
        dur = max(self.project.music_duration, self.project.video_duration, 10.0)
        self.offset = 0.0
        self.pps = max(1.0, (self.winfo_width() - self.u(10)) / dur)
        self.redraw()

    def _zoom(self, x, factor):
        t = self.t_of(x)
        self.pps = float(np.clip(self.pps * factor, 1.0, 2000.0))
        self.offset = max(0.0, t - x / self.pps)
        self.redraw()

    def _wheel(self, e):
        self._zoom(e.x, 1.2 if e.delta > 0 else 1 / 1.2)

    def _shift_wheel(self, e):
        self.offset = max(0.0, self.offset - (e.delta / 120) * self.u(80) / self.pps)
        self.redraw()

    # -------------------------------------------------------------- picking
    def _nearest(self, track, x):
        if track == "audio":
            beats = self.project.markers.beats
            if not beats:
                return None
            xs = np.array([self.x_of(b) for b in beats])
            i = int(np.argmin(np.abs(xs - x)))
            return ("beat", i) if abs(xs[i] - x) <= self.u(PICK_PX) else None
        if track == "video":
            hits = self.project.markers.hits
            if not hits:
                return None
            xs = np.array([self.x_of(h.t) for h in hits])
            i = int(np.argmin(np.abs(xs - x)))
            return ("hit", i) if abs(xs[i] - x) <= self.u(PICK_PX) else None
        return None

    def _press(self, e):
        self.focus_set()
        track = self.track_at(e.y)
        pick = self._nearest(track, e.x)
        if pick:
            self.selected = pick
            self.on_select(pick)
            self.redraw()
        else:
            self._pan = (e.x, self.offset, False)
        if track:
            self.on_click(track, self.t_of(e.x))

    def _drag(self, e):
        if self._pan:
            x0, off0, _ = self._pan
            self.offset = max(0.0, off0 - (e.x - x0) / self.pps)
            self._pan = (x0, off0, True)
            self.redraw()

    def _double(self, e):
        track = self.track_at(e.y)
        t = self.t_of(e.x)
        m = self.project.markers
        if track == "audio":
            self.selected = ("beat", m.add_beat(t))
            self.on_change("Added beat")
        elif track == "video":
            self.selected = ("hit", m.add_hit(self.snap_hit(t), self.project.sync.combo_gap))
            self.on_change("Added hit")
        else:
            return
        self.on_select(self.selected)
        self.redraw()

    def _right(self, e):
        pick = self._nearest(self.track_at(e.y), e.x)
        if pick:
            self.selected = pick
            self.delete_selected()

    # --------------------------------------------------------------- edits
    def delete_selected(self):
        if not self.selected:
            return
        kind, i = self.selected
        if kind == "beat":
            self.project.markers.delete_beat(i)
        else:
            self.project.markers.delete_hit(i)
        self.selected = None
        self.on_select(None)
        self.on_change(f"Deleted {kind}")
        self.redraw()

    def split_selected(self):
        if self.selected and self.selected[0] == "hit":
            self.project.markers.split_combo_at(self.selected[1])
            self.on_change("Split combo")
            self.redraw()

    def merge_selected(self):
        if self.selected and self.selected[0] == "hit":
            self.project.markers.merge_with_previous(self.selected[1])
            self.on_change("Merged combo")
            self.redraw()

    # ------------------------------------------------------------- playhead
    def set_playhead(self, t: Optional[float], src_t: Optional[float] = None,
                     follow: bool = False):
        self.playhead, self.playhead_src = t, src_t
        w = max(10, self.winfo_width())
        if follow and t is not None and not (self.t_of(0) <= t <= self.t_of(w * 0.9)):
            self.offset = max(0.0, t - (w * 0.1) / self.pps)
            self.redraw()
        else:
            self._draw_playhead()

    def _draw_playhead(self):
        self.delete("playhead")
        if self.playhead is None:
            return
        x = self.x_of(self.playhead)
        _, y_edit_end = self._track_y["edit"]
        lw = max(1, self.u(2))
        self.create_line(x, 0, x, y_edit_end, fill="#ffffff", width=lw, tags="playhead")
        a = self.u(6)
        self.create_polygon(x - a, 0, x + a, 0, x, a * 1.5, fill="#ffffff", tags="playhead")
        if self.playhead_src is not None:
            xs = self.x_of(self.playhead_src)
            y0, y1 = self._track_y["video"]
            self.create_line(xs, y0, xs, y1, fill="#ffffff", width=lw, dash=(4, 3),
                             tags="playhead")

    # -------------------------------------------------------------- drawing
    def redraw(self):
        self.delete("all")
        w = max(10, self.winfo_width())
        t0, t1 = self.t_of(0), self.t_of(w)
        self._draw_ruler(w, t0, t1)
        for key, label, _ in TRACKS:
            y0, y1 = self._track_y[key]
            self.create_rectangle(0, y0, w, y1, fill=TRACK_BG, outline="")
        self._draw_audio(w, t0, t1)
        self._draw_edit(w, t0, t1)
        self._draw_video(w, t0, t1)
        for key, label, _ in TRACKS:
            y0, _ = self._track_y[key]
            self.create_text(self.u(6), y0 + self.u(3), text=label, anchor="nw", fill=TEXT,
                             font=("Segoe UI", 8, "bold"))
        self._draw_playhead()

    def _draw_ruler(self, w, t0, t1):
        span = t1 - t0
        step = next((s for s in (0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120)
                     if span / s <= w / self.u(70)), 300)
        t = np.floor(t0 / step) * step
        while t <= t1:
            x = self.x_of(t)
            self.create_line(x, self.ruler_h - self.u(6), x, self.ruler_h, fill=TEXT)
            label = f"{int(t // 60)}:{t % 60:05.2f}" if step < 1 else f"{int(t // 60)}:{int(t % 60):02d}"
            self.create_text(x + self.u(3), self.u(3), text=label, anchor="nw", fill=TEXT, font=("Segoe UI", 7))
            self.create_line(x, self.ruler_h, x, self.winfo_height(), fill=GRID)
            t += step

    def _visible(self, arr, t0, t1):
        arr = np.asarray(arr, float)
        return (arr >= t0 - 1) & (arr <= t1 + 1)

    def _draw_audio(self, w, t0, t1):
        p = self.project
        y0, y1 = self._track_y["audio"]
        mid = y1 - self.u(4)
        if p.audio is not None and len(p.audio.rms):
            self._envelope(p.audio.rms_times, p.audio.rms, t0, t1, w, mid, (y1 - y0) - self.u(22), "#2f3b52")
        # drop marker
        if p.sync.intro_enabled and p.sync.drop_time > 0:
            x = self.x_of(p.sync.drop_time)
            self.create_line(x, y0, x, y1, fill=DROP, width=self.u(2))
            self.create_text(x + self.u(3), y1 - self.u(4), text="DROP", anchor="sw", fill=DROP,
                             font=("Segoe UI", 7, "bold"))
        beats = np.asarray(p.markers.beats, float)
        sel = self.selected[1] if self.selected and self.selected[0] == "beat" else -1
        stride = 1 if self.pps > self.u(12) else int(np.ceil(self.u(12) / self.pps))
        for i in np.nonzero(self._visible(beats, t0, t1))[0]:
            if i % stride and i != sel:
                continue
            x = self.x_of(beats[i])
            self.create_line(x, y0 + self.u(26), x, y1 - self.u(2), fill=SELECT if i == sel else BEAT,
                             width=self.u(3) if i == sel else max(1, self.u(1)))
        # where each hit landed in the output (green = locked, orange = unsynced);
        # hits on neighbouring beats are joined, so an unbroken chain means
        # every beat has its hit
        sched = p.schedule
        if sched is not None:
            placed = sorted((pl for pl in sched.placements if pl.out_t is not None),
                            key=lambda q: q.out_t)
            a, ty = self.u(4), y0 + self.u(18)
            for prev, pl in zip(placed, placed[1:]):
                if not (prev.locked and pl.locked) or pl.out_t < t0 - 1 or prev.out_t > t1 + 1:
                    continue
                k = np.searchsorted(beats, [prev.out_t - 1e-3, pl.out_t - 1e-3])
                if k[1] - k[0] == 1:          # consecutive beats
                    self.create_line(self.x_of(prev.out_t), ty + self.u(1), self.x_of(pl.out_t),
                                     ty + self.u(1), fill="#3ddc97", width=max(1, self.u(2)))
            for pl in placed:
                if not (t0 - 1 <= pl.out_t <= t1 + 1):
                    continue
                x = self.x_of(pl.out_t)
                col = "#3ddc97" if pl.locked else "#ff9f43"
                self.create_polygon(x - a, ty, x + a, ty, x, y0 + self.u(25), fill=col, outline="")

    def _envelope(self, times, values, t0, t1, w, base_y, height, color):
        times = np.asarray(times, float)
        values = np.asarray(values, float)
        cols = np.arange(0, w, max(1, self.u(2)))
        ts = self.t_of(cols)
        inside = (ts >= times[0]) & (ts <= times[-1])
        if not inside.any():
            return
        idx = np.clip(np.searchsorted(times, ts), 0, len(values) - 1)
        vals = values[idx] * inside
        pts = [0, base_y]
        for x, v in zip(cols, vals):
            pts += [float(x), float(base_y - v * height)]
        pts += [float(cols[-1]), base_y]
        self.create_polygon(*pts, fill=color, outline="")

    def _draw_edit(self, w, t0, t1):
        sched = self.project.schedule
        y0, y1 = self._track_y["edit"]
        if sched is None:
            self.create_text(w / 2, (y0 + y1) / 2, text="Press “Recalculate Alignment”",
                             fill=TEXT, font=("Segoe UI", 9))
            return
        ty = y0 + self.u(16)
        for seg in sched.segments:
            if seg.out_end < t0 or seg.out_start > t1:
                continue
            xa, xb = self.x_of(seg.out_start), self.x_of(seg.out_end)
            sp = seg.speed
            self.create_rectangle(xa, ty, max(xa + 1, xb), y1 - self.u(4),
                                  fill=SEG_COLORS.get(seg.kind, "#555"), outline="#111")
            if xb - xa > self.u(34):
                self.create_text((xa + xb) / 2, (ty + y1 - self.u(4)) / 2, text=f"{sp:.2f}x",
                                 fill="#e8ecf2", font=("Segoe UI", 7))
        for c in sched.cuts:
            x = self.x_of(c)
            self.create_line(x, y0 + self.u(12), x, y1, fill=DROP, width=self.u(2))
        # letterbox envelope along the top
        if sched.params.letterbox_enabled and sched.letterbox:
            cols = np.arange(0, w, max(1, self.u(3)))
            pts = [0, ty]
            for x in cols:
                pts += [float(x), ty - self.u(10) * sched.letterbox_amount(self.t_of(x))]
            pts += [float(cols[-1]), ty]
            self.create_polygon(*pts, fill="#c9ced8", outline="")
        if sched.duration:
            for t in (sched.start, sched.end):
                x = self.x_of(t)
                self.create_line(x, y0, x, y1, fill="#ffffff", dash=(3, 3))
        # the part of the song that isn't in the montage
        ya, _ = self._track_y["audio"]
        for a, b in ((0.0, sched.start), (sched.end, 1e6)):
            if b - a > 1e-3 and b > t0 and a < t1:
                self.create_rectangle(self.x_of(max(a, t0 - 1)), ya, self.x_of(min(b, t1 + 1)), y1,
                                      fill=REMOVED, outline="", stipple="gray50")

    def _draw_video(self, w, t0, t1):
        p = self.project
        y0, y1 = self._track_y["video"]
        sync = p.sync
        if sync.intro_enabled and sync.intro_end > sync.intro_start:
            xa, xb = self.x_of(sync.intro_start), self.x_of(sync.intro_end)
            self.create_rectangle(xa, y0 + self.u(14), xb, y1, fill=INTRO, outline="")
            self.create_text(xa + self.u(3), y0 + self.u(16), text="INTRO", anchor="nw", fill="#c7b5ff",
                             font=("Segoe UI", 7, "bold"))
        if p.schedule is not None:
            for a, b in p.schedule.removed_source_ranges():
                if b < t0 or a > t1:
                    continue
                self.create_rectangle(self.x_of(a), y0 + self.u(14), self.x_of(b), y1,
                                      fill=REMOVED, outline="", stipple="gray50")
        if len(self.score):
            self._envelope(self.score_times, np.clip(self.score, 0, 1.2) / 1.2, t0, t1, w,
                           y1 - self.u(2), (y1 - y0) - self.u(40), "#3a3346")
        if p.video is not None:
            x = self.x_of(p.video_duration)
            self.create_line(x, y0, x, y1, fill="#666", dash=(2, 2))
        hits = p.markers.hits
        ids = p.markers.combo_ids()
        sel = self.selected[1] if self.selected and self.selected[0] == "hit" else -1
        # combo spans
        for grp in p.markers.combos():
            a, b = hits[grp[0]].t, hits[grp[-1]].t
            if b < t0 - 1 or a > t1 + 1:
                continue
            col = COMBO_COLORS[ids[grp[0]] % len(COMBO_COLORS)]
            self.create_rectangle(self.x_of(a) - self.u(2), y0 + self.u(30), self.x_of(b) + self.u(2), y0 + self.u(36),
                                  fill=col, outline="")
            if len(grp) > 1 and self.x_of(b) - self.x_of(a) > self.u(30):
                self.create_text(self.x_of(a), y0 + self.u(38), text=f"x{len(grp)}", anchor="nw",
                                 fill=col, font=("Segoe UI", 7, "bold"))
        status = {}
        if p.schedule is not None:
            status = {pl.hit_index: pl for pl in p.schedule.placements}
        for i, h in enumerate(hits):
            if not (t0 - 1 <= h.t <= t1 + 1):
                continue
            x = self.x_of(h.t)
            col = COMBO_COLORS[ids[i] % len(COMBO_COLORS)]
            is_sel = i == sel
            self.create_line(x, y0 + self.u(40), x, y1 - self.u(2), fill=SELECT if is_sel else col,
                             width=self.u(3) if is_sel else self.u(2))
            head = "#3ddc97" if status.get(i) and status[i].locked else (
                "#666" if status.get(i) and status[i].out_t is None else col)
            r, cy = self.u(5 if is_sel else 4), y0 + self.u(40)
            self.create_oval(x - r, cy - r, x + r, cy + r, fill=head,
                             outline=SELECT if is_sel else "")
            if h.combo_start:
                self.create_line(x, y0 + self.u(14), x, y0 + self.u(30), fill=col, width=max(1, self.u(1)), dash=(2, 1))
