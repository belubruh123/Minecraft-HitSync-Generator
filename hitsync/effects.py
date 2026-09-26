"""Montage look: colour filters, motion blur, hit/beat effects, effect ranges,
transitions, start/end fades and captions.

Everything is evaluated per output frame from the schedule (hit times,
beats, cuts) so preview and export look identical. Each effect skips all
work when its amount is zero; look-up tables, vignette masks and caption
sprites are built once and cached.

Order: fetch (interpolation + motion blur) -> zoom/shake -> colour ->
glow -> RGB split -> vignette -> letterbox -> captions -> flash / fades.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np

FILTERS = ["none", "vibrant", "cinematic", "warm", "cool", "punchy", "bw", "vintage", "night"]
RANGE_KINDS = ["flashy", "glow", "rgb", "strobe", "bw", "shake", "zoom", "blur"]
RANGE_LABELS = {"flashy": "Flashy", "glow": "Glow", "rgb": "RGB split", "strobe": "Strobe",
                "bw": "Black & white", "shake": "Shake", "zoom": "Zoom pulse",
                "blur": "Motion blur+"}
TRANSITIONS = ["cut", "flash", "zoom", "whip", "dip"]


@dataclass
class EffectRange:
    kind: str
    start: float              # music time
    end: float
    strength: float = 1.0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(str(d["kind"]), float(d["start"]), float(d["end"]),
                   float(d.get("strength", 1.0)))

    def weight(self, t: float, edge: float = 0.12) -> float:
        """0..1 inside the range, eased over ``edge`` s at both ends."""
        if not (self.start <= t < self.end):
            return 0.0
        e = min(edge, (self.end - self.start) / 3)
        w = min(1.0, (t - self.start) / e if e > 0 else 1.0, (self.end - t) / e if e > 0 else 1.0)
        return float(np.clip(w, 0, 1)) * float(self.strength)


# ------------------------------------------------------------ colour looks
def _curve(x, contrast=0.0, lift=0.0, gamma=1.0, gain=1.0):
    y = np.clip(x, 0, 1) ** gamma
    if contrast:
        s = y * y * (3 - 2 * y)                   # smooth S-curve
        y = y + contrast * (s - y)
    return np.clip(lift + y * (gain - lift), 0, 1)


def build_filter(name: str, strength: float):
    """(lut (256,1,3) uint8 in BGR, saturation) for a named look."""
    x = np.linspace(0, 1, 256)
    b = g = r = x
    sat = 1.0
    if name == "vibrant":
        b = g = r = _curve(x, 0.35)
        sat = 1.35
    elif name == "cinematic":                     # teal shadows, orange highlights
        base = _curve(x, 0.45)
        r = np.clip(base + 0.06 * (x - 0.45), 0, 1)
        g = base
        b = np.clip(base + 0.08 * (0.45 - x), 0, 1)
        sat = 1.1
    elif name == "warm":
        r, g, b = _curve(x, 0.15, gain=1.0), _curve(x, 0.15, gain=0.97), _curve(x, 0.15, gain=0.86)
        sat = 1.12
    elif name == "cool":
        r, g, b = _curve(x, 0.15, gain=0.88), _curve(x, 0.15, gain=0.98), _curve(x, 0.15, gain=1.0)
        sat = 1.05
    elif name == "punchy":
        b = g = r = _curve(x, 0.8, gamma=1.05)
        sat = 1.55
    elif name == "bw":
        b = g = r = _curve(x, 0.5)
        sat = 0.0
    elif name == "vintage":
        r = _curve(x, 0.1, lift=0.08, gain=0.97)
        g = _curve(x, 0.1, lift=0.06, gain=0.93)
        b = _curve(x, 0.1, lift=0.10, gain=0.82)
        sat = 0.8
    elif name == "night":
        r, g, b = _curve(x, 0.3, gamma=1.3, gain=0.75), _curve(x, 0.3, gamma=1.25, gain=0.85), \
            _curve(x, 0.3, gamma=1.1, gain=1.0)
        sat = 0.7
    s = float(np.clip(strength, 0, 1))
    lut = np.stack([x + s * (c - x) for c in (b, g, r)], axis=1)
    lut = np.clip(lut * 255 + 0.5, 0, 255).astype(np.uint8).reshape(256, 1, 3)
    return lut, 1.0 + s * (sat - 1.0)


_LUMA = np.array([0.114, 0.587, 0.299], np.float32)      # BGR


def saturate(frame: np.ndarray, sat: float) -> np.ndarray:
    """Saturation as one 3x3 colour matrix (one SIMD pass, no grey copy)."""
    if abs(sat - 1.0) < 1e-3:
        return frame
    M = np.eye(3, dtype=np.float32) * sat + (1.0 - sat) * np.tile(_LUMA, (3, 1))
    return cv2.transform(frame, M)


# ------------------------------------------------------------- primitives
def warp(frame, zoom: float, dx: float, dy: float):
    """Zoom about the centre plus a shift (px): crop the visible window and
    scale it back up (3x faster than a general affine warp). Nothing
    happens when the change is under half a pixel."""
    h, w = frame.shape[:2]
    if (zoom - 1) * max(w, h) / 2 < 0.5 and abs(dx) < 0.5 and abs(dy) < 0.5:
        return frame
    # never show borders: zoom in at least enough to cover the shift
    zoom = max(zoom, 1.0 + 2.0 * max(abs(dx) / w, abs(dy) / h))
    cw, ch = max(2, int(round(w / zoom))), max(2, int(round(h / zoom)))
    x0 = int(round((w - cw) / 2 - dx / zoom))
    y0 = int(round((h - ch) / 2 - dy / zoom))
    x0, y0 = min(max(0, x0), w - cw), min(max(0, y0), h - ch)
    return cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)


def rgb_split(frame, px: int):
    if px < 1:
        return frame
    out = frame.copy()
    out[:, px:, 2] = frame[:, :-px, 2]           # red right
    out[:, :-px, 0] = frame[:, px:, 0]           # blue left
    return out


def glow(frame, amount: float):
    if amount <= 1e-3:
        return frame
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(8, w // 6), max(8, h // 6)), interpolation=cv2.INTER_AREA)
    bright = cv2.subtract(small, np.full_like(small, 150))
    k = max(3, (w // 6) // 30 * 2 + 1)
    bright = cv2.GaussianBlur(bright, (k, k), 0)
    bright = cv2.resize(bright, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(frame, 1.0, bright, min(2.0, 1.6 * amount), 0.0)


def flash(frame, amount: float, white: bool = True):
    """Fade towards white (or black); invisible amounts are skipped."""
    if amount < 0.01:
        return frame
    a = float(min(1.0, amount))
    return cv2.convertScaleAbs(frame, alpha=1.0 - a, beta=255.0 * a if white else 0.0)


def directional_blur(frame, px: int):
    if px < 2:
        return frame
    h, w = frame.shape[:2]
    scale = 4
    small = cv2.resize(frame, (max(8, w // scale), max(8, h // scale)), interpolation=cv2.INTER_AREA)
    k = max(3, px // scale | 1)
    small = cv2.blur(small, (k, 1))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------ look
def _decay(u, tau):
    return math.exp(-u / tau) if 0 <= u < 6 * tau else 0.0


class Look:
    """Everything needed to finish a frame; built once per schedule/params."""

    def __init__(self, schedule, rparams, hits_fx=None, ranges=(), texts=()):
        self.s = schedule
        self.r = rparams
        self.lut, self.sat = build_filter(rparams.filter, rparams.filter_strength) \
            if rparams.filter != "none" and rparams.filter_strength > 0 else (None, 1.0)
        placed = schedule.placed_hits() if hasattr(schedule, "placed_hits") else []
        default = bool(getattr(rparams, "hit_fx_default", True))
        fx = hits_fx or {}
        self.fx_hits = np.array([p.out_t for p in placed
                                 if fx.get(p.hit_index, default)], float)
        self.beats = np.asarray(getattr(schedule, "beat_times", []), float)
        self.downbeats = np.asarray(getattr(schedule, "downbeats", []), float)
        self.cuts = np.asarray(getattr(schedule, "cuts", []), float)
        self.ranges = list(ranges)
        self.texts = list(texts)
        self._vig = {}

    # ---- helpers
    @staticmethod
    def _last(times, t):
        i = int(np.searchsorted(times, t, side="right")) - 1
        return float(times[i]) if i >= 0 else None

    def _range_weights(self, t):
        w = {}
        for rg in self.ranges:
            v = rg.weight(t)
            if v > 0:
                w[rg.kind] = max(w.get(rg.kind, 0.0), v)
        return w

    def vignette_mask(self, shape, amount):
        key = (shape[0], shape[1], round(amount, 3))
        m = self._vig.get(key)
        if m is None:
            h, w = shape[:2]
            y, x = np.ogrid[-1:1:complex(0, h), -1:1:complex(0, w)]
            d = np.sqrt((x * 0.85) ** 2 + y ** 2) / math.sqrt(0.85 ** 2 + 1)
            m = np.clip(1.0 - amount * np.clip((d - 0.35) / 0.65, 0, 1) ** 1.6, 0, 1)
            m = cv2.merge([(m * 255).astype(np.uint8)] * 3)
            self._vig = {key: m}
        return m

    def blur_samples(self, t, speed):
        """(number of source frames to blend, their weights) at output t."""
        amount = float(self.r.motion_blur) + 0.6 * self._range_weights(t).get("blur", 0.0)
        if amount <= 1e-3 or speed < 0.65:
            return 1
        return int(np.clip(1 + round(amount * 2 * max(1.0, speed)), 1, 6))

    # ---- per-frame parameters
    def params_at(self, t: float, frame_w: int) -> dict:
        r = self.r
        rw = self._range_weights(t)
        zoom, dx, dy, white, rgb, glow_a, sat_boost, bw = 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # hit effects (only hits ticked for effects)
        h = self._last(self.fx_hits, t + 0.02)
        if h is not None:
            u = t - h
            e = _decay(u, 0.12) if u >= 0 else max(0.0, 1 + u / 0.02) * 0.5
            zoom += r.hit_zoom * e
            white = max(white, r.hit_flash * _decay(u, 0.07))
            rgb += r.hit_rgb * e
            if r.hit_shake > 0 and u >= 0:
                k = int(h * 1000) % 997
                amp = r.hit_shake * 0.012 * frame_w * _decay(u, 0.15)
                dx += amp * math.sin(2 * math.pi * 23 * u + k)
                dy += amp * math.cos(2 * math.pi * 19 * u + 2 * k)
        # downbeat pulse
        if r.beat_pulse > 0 and len(self.downbeats):
            d = self._last(self.downbeats, t)
            if d is not None:
                zoom += 0.035 * r.beat_pulse * _decay(t - d, 0.18)
        # effect ranges
        if rw:
            b = self._last(self.beats, t)
            ub = (t - b) if b is not None else 9.0
            f = rw.get("flashy", 0.0)
            if f:
                white = max(white, 0.45 * f * _decay(ub, 0.06))
                sat_boost += 0.45 * f
                glow_a += 0.6 * f
                rgb += 0.6 * f * _decay(ub, 0.1)
                zoom += 0.04 * f * _decay(ub, 0.12)
            if rw.get("strobe"):
                white = max(white, 0.8 * rw["strobe"] * _decay(ub, 0.05))
            glow_a += rw.get("glow", 0.0)
            rgb += rw.get("rgb", 0.0) * (0.5 + 0.5 * _decay(ub, 0.1))
            bw = max(bw, rw.get("bw", 0.0))
            if rw.get("zoom"):
                zoom += 0.06 * rw["zoom"] * _decay(ub, 0.15)
            if rw.get("shake"):
                amp = rw["shake"] * 0.008 * frame_w
                dx += amp * math.sin(2 * math.pi * 11 * t)
                dy += amp * math.cos(2 * math.pi * 13 * t + 1.3)
        # transitions at cuts
        tr = self.s.params.transition
        c = self._last(self.cuts, t) if len(self.cuts) else None
        nxt = None
        if len(self.cuts):
            j = int(np.searchsorted(self.cuts, t, side="right"))
            nxt = float(self.cuts[j]) if j < len(self.cuts) else None
        whip = 0.0
        dark = 0.0
        if tr == "zoom" and c is not None:
            zoom += 0.15 * _decay(t - c, 0.12)
        elif tr == "whip":
            if c is not None and t - c < 0.12:
                whip = 1 - (t - c) / 0.12
                dx += -0.08 * frame_w * whip
            if nxt is not None and nxt - t < 0.12:
                w2 = 1 - (nxt - t) / 0.12
                whip = max(whip, w2)
                dx += 0.08 * frame_w * w2
        elif tr == "dip":
            for x in (c, nxt):
                if x is not None and abs(t - x) < 0.1:
                    dark = max(dark, 1 - abs(t - x) / 0.1)
        return dict(zoom=zoom, dx=dx, dy=dy, white=white, rgb=rgb, glow=glow_a,
                    sat_boost=sat_boost, bw=bw, whip=whip, dark=dark)

    def fade_amounts(self, t: float):
        """(white, black) overlay for the montage's start / end fades."""
        s, r = self.s, self.r
        white = black = 0.0
        if r.fade_in not in ("", "none") and r.fade_in_len > 0:
            u = (t - s.start) / r.fade_in_len
            if u < 1:
                a = 1 - max(0.0, u)
                if r.fade_in == "white":
                    white = a
                else:
                    black = a
        if r.fade_out not in ("", "none") and r.fade_out_len > 0:
            u = (s.end - t) / r.fade_out_len
            if u < 1:
                a = 1 - max(0.0, u)
                if r.fade_out == "white":
                    white = max(white, a)
                else:
                    black = max(black, a)
        return white, black

    # ---- apply
    def finish(self, frame: np.ndarray, t: float, letterbox) -> np.ndarray:
        """Apply everything after the source frame is fetched. ``frame`` may
        be a cached decoder frame: it is never modified in place."""
        r = self.r
        h, w = frame.shape[:2]
        p = self.params_at(t, w)
        owned = False
        out = warp(frame, p["zoom"], p["dx"], p["dy"])
        owned = out is not frame
        if p["whip"] > 0.05:
            out = directional_blur(out, int(0.06 * w * p["whip"]))
            owned = True
        if self.lut is not None:
            out = cv2.LUT(out, self.lut)
            owned = True
        sat = self.sat * (1 + p["sat_boost"]) * (1 - p["bw"])
        if abs(sat - 1.0) > 1e-3:
            out = saturate(out, sat)
            owned = True
        if p["glow"] > 1e-3:
            out = glow(out, p["glow"])
            owned = True
        px = int(round(p["rgb"] * 0.006 * w))
        if px >= 1:
            out = rgb_split(out, px)
            owned = True
        if r.vignette > 1e-3:
            out = cv2.multiply(out, self.vignette_mask(out.shape, r.vignette), scale=1 / 255)
            owned = True
        if letterbox is not None:
            amount, aspect = letterbox
            if amount > 1e-3:
                if not owned:
                    out = out.copy()
                    owned = True
                from .renderer import apply_letterbox

                out = apply_letterbox(out, amount, aspect)
        if self.texts:
            active = [it for it in self.texts if it.start <= t < it.end]
            if active:
                if not owned:
                    out = out.copy()
                    owned = True
                from .text_overlay import draw_caption

                for it in active:
                    draw_caption(out, it, t)
        fw, fb = self.fade_amounts(t)
        white = max(p["white"], fw, self.s.flash_amount(t) if hasattr(self.s, "flash_amount") else 0)
        out = flash(out, white, True)
        out = flash(out, max(fb, p["dark"]), False)
        return out
