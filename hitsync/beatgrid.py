"""Pure-numpy beat grid and tempo-curve helpers (no librosa needed).

These operate on the *edited* beat list, so user corrections on the timeline
immediately change the grid and the tempo curve used for footage matching.
"""
from __future__ import annotations

import numpy as np


def _median_interval(beats: np.ndarray, default: float = 0.5) -> float:
    if len(beats) < 2:
        return default
    d = np.diff(beats)
    d = d[d > 1e-3]
    return float(np.median(d)) if len(d) else default


def extended_beats(beats, start: float, end: float) -> np.ndarray:
    """Beat list extrapolated with the local interval to cover [start, end]."""
    b = np.asarray(sorted(beats), dtype=float)
    if len(b) < 2:
        step = 0.5  # 120 BPM fallback when there is no usable beat info
        first = b[0] if len(b) else 0.0
        n0 = int(np.ceil((first - start) / step)) + 1
        return first + step * np.arange(-n0, int((end - first) / step) + 2)
    head_step = _median_interval(b[:9])
    tail_step = _median_interval(b[-9:])
    pre = []
    t = b[0] - head_step
    while t >= start - head_step:
        pre.append(t)
        t -= head_step
    post = []
    t = b[-1] + tail_step
    while t <= end + tail_step:
        post.append(t)
        t += tail_step
    return np.concatenate([np.array(pre[::-1]), b, np.array(post)])


def _index_beats(b: np.ndarray, period: float):
    """Integer beat numbers, counted from the last beat that sat on the pulse.

    Starts from a stable anchor (several consecutive ~1-period intervals) and
    walks outwards. A beat that lands between pulses (e.g. the tracker
    following 1/8 notes) is marked off-grid and never used as a reference,
    so it can't shift the numbering of everything after it.
    """
    n = len(b)
    k = np.zeros(n)
    keep = np.zeros(n, bool)
    iv = np.diff(b) / period
    good = np.abs(iv - 1) < 0.15
    anchor = 0
    for i in range(max(0, n - 4)):
        if good[i:i + 4].all():
            anchor = i
            break
    keep[anchor] = True
    for direction in (1, -1):
        last_t, last_k = b[anchor], 0.0
        rng = range(anchor + 1, n) if direction == 1 else range(anchor - 1, -1, -1)
        for i in rng:
            r = (b[i] - last_t) / period
            m = round(r)
            k[i] = last_k + m
            if m != 0 and abs(r - m) < 0.25:
                keep[i] = True
                last_t, last_k = b[i], k[i]
    return k, keep


def fit_static_grid(beats, duration: float, bpm: float = 0.0, offset: float = 0.0):
    """Fit one constant tempo + phase to detected beats (robust line fit).

    Beat trackers wander: a section of 1/8 hi-hats can make them report two
    beats per real beat, and per-beat jitter is tens of ms. A montage wants
    the song's pulse to stay put, so we fit t_k = phase + period * k to all
    beats, discard outliers (e.g. doubled beats land half a period off the
    line) and refit. ``bpm`` > 0 forces the tempo; ``offset`` nudges the
    phase (seconds). Returns (grid_times, bpm, phase).
    """
    b = np.asarray(sorted(beats), dtype=float)
    if len(b) < 4 and bpm <= 0:
        return b, (60.0 / _median_interval(b) if len(b) > 1 else 120.0), (b[0] if len(b) else 0.0)
    iv = np.diff(b) if len(b) > 1 else np.array([0.5])
    period = 60.0 / bpm if bpm > 0 else float(np.median(iv[iv > 1e-3]))
    # Number the beats neighbour-to-neighbour: dividing the whole song by a
    # rough period lets a ~0.5% period error push late beats onto the wrong
    # index (hundreds of ms of drift over a 3-minute song).
    k, keep = _index_beats(b, period)
    phase = b[0] if len(b) else 0.0
    for _ in range(6):
        if bpm > 0 or keep.sum() < 3:
            phase = float(np.median((b - period * k)[keep]))
        else:
            A = np.stack([np.ones(keep.sum()), k[keep]], axis=1)
            phase, period = np.linalg.lstsq(A, b[keep], rcond=None)[0]
        # re-index against the refined line, then drop off-grid beats
        k = np.round((b - phase) / period)
        res = b - (phase + period * k)
        keep = np.abs(res) < 0.2 * period
    phase = (phase + offset) % period
    n = int(np.ceil((duration - phase) / period)) + 1
    grid = phase + period * np.arange(max(1, n))
    return grid[grid <= duration + 1e-9], 60.0 / period, float(phase)


def subdivide(beats: np.ndarray, n: int) -> np.ndarray:
    """Insert n-1 evenly spaced points between consecutive beats."""
    n = max(1, int(n))
    if n == 1 or len(beats) < 2:
        return np.asarray(beats, dtype=float)
    pts = [beats[:-1] + (np.diff(beats) * k / n) for k in range(n)]
    grid = np.sort(np.concatenate(pts + [beats[-1:]]))
    return grid


def tempo_curve(beats, window: int = 8):
    """Local BPM at each inter-beat midpoint, smoothed and octave-fixed.

    Returns (times, bpm). Octave errors (a stray half/double interval) are
    folded toward the global median so they don't cause 2x speed jumps, and a
    trimmed mean over the window averages out frame-quantisation jitter in
    beat times (a median would lock onto one of the quantised values).
    """
    b = np.asarray(sorted(beats), dtype=float)
    if len(b) < 3:
        return np.array([0.0]), np.array([120.0])
    iv = np.diff(b)
    iv = np.where(iv <= 1e-3, np.median(iv), iv)
    ref = float(np.median(iv))
    # fold octave errors
    iv = np.where(iv < ref / 1.5, iv * 2, iv)
    iv = np.where(iv > ref * 1.5, iv / 2, iv)
    half = max(1, window // 2)

    def trimmed_mean(w):
        w = np.sort(w)
        k = len(w) // 5
        return float(np.mean(w[k: len(w) - k])) if len(w) > 2 * k else float(np.mean(w))

    smooth = np.array([trimmed_mean(iv[max(0, i - half): i + half + 1]) for i in range(len(iv))])
    times = (b[:-1] + b[1:]) / 2
    return times, 60.0 / smooth


def bpm_at(curve, t):
    times, bpm = curve
    return np.interp(t, times, bpm)
