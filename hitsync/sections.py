"""Where the music "starts": drums coming in, the drop, and the song's end.

Works on bars (downbeat to downbeat) of the beat grid, so every candidate is
a bar line, the place a producer would put a drop. Per bar: kick activity,
snare/hat activity, bass level and overall level. A start candidate is a bar
where kick and bass rise and *stay* up; a snare-roll build raises only the
snares, so it scores lower than the drop it leads into.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Candidate:
    time: float
    kind: str          # "drums_in" | "drop" | "section"
    score: float

    @property
    def label(self) -> str:
        return {"drums_in": "Beat kicks in", "drop": "Drop"}.get(self.kind, "Big moment")

    def to_dict(self):
        return {"time": self.time, "kind": self.kind, "score": self.score}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["time"]), str(d.get("kind", "section")), float(d.get("score", 0.0)))


@dataclass
class Sections:
    candidates: list = field(default_factory=list)   # sorted by time
    best: float = 0.0                                # default start (drop) time
    first_sound: float = 0.0
    song_end: float = 0.0                            # last bar line before the outro

    def to_dict(self):
        return {"candidates": [c.to_dict() for c in self.candidates], "best": self.best,
                "first_sound": self.first_sound, "song_end": self.song_end}

    @classmethod
    def from_dict(cls, d):
        if not d:
            return cls()
        return cls([Candidate.from_dict(c) for c in d.get("candidates", [])],
                   float(d.get("best", 0.0)), float(d.get("first_sound", 0.0)),
                   float(d.get("song_end", 0.0)))

    def top(self, n: int = 3) -> list:
        """The n best candidates (always including the default), by time."""
        ranked = sorted(self.candidates, key=lambda c: -c.score)[:n]
        if self.best and not any(abs(c.time - self.best) < 1e-3 for c in ranked):
            ranked = ranked[:n - 1] + [c for c in self.candidates
                                       if abs(c.time - self.best) < 1e-3]
        return sorted(ranked, key=lambda c: c.time)


def _bar_means(times, curve, bars, end):
    edges = np.append(bars, end)
    idx = np.searchsorted(times, edges)
    out = np.zeros(len(bars))
    for i in range(len(bars)):
        a, b = idx[i], max(idx[i] + 1, idx[i + 1])
        out[i] = float(np.mean(curve[a:b])) if b <= len(curve) and a < len(curve) else 0.0
    return out


def detect(curve_t, level, bass, kick, snare, bars, duration) -> Sections:
    """Start candidates from per-bar features.

    ``bars``: bar start times (downbeats). Curves are 0..1 on ``curve_t``.
    """
    bars = np.asarray(bars, float)
    s = Sections()
    loud = np.nonzero(level > 0.08)[0]
    s.first_sound = float(curve_t[loud[0]]) if len(loud) else 0.0
    if len(bars) < 6:
        s.best = float(bars[0]) if len(bars) else s.first_sound
        s.song_end = duration
        return s
    K = _bar_means(curve_t, kick, bars, duration)
    S = _bar_means(curve_t, snare, bars, duration)
    B = _bar_means(curve_t, bass, bars, duration)
    L = _bar_means(curve_t, level, bars, duration)
    n = len(bars)
    bar_len = float(np.median(np.diff(bars)))
    if duration - bars[-1] < 0.75 * bar_len:
        # the song ends a little into its last bar: that bar's averages are
        # just its first hit, not a louder bar
        for arr in (K, S, B, L):
            arr[-1] = arr[-2]

    # drums come in: first bar whose kick level holds for 4 bars after a quieter stretch
    k_hi = np.percentile(K, 85)
    drums_in = None
    for i in range(1, n - 3):
        if (K[i] >= 0.55 * k_hi and K[i:i + 4].min() >= 0.45 * k_hi
                and K[max(0, i - 2):i].max() < 0.45 * k_hi):
            drums_in = i
            break

    scores = np.zeros(n)
    # a start needs song after it (at least 4 bars): the last bars' rise is
    # only the song ending
    for i in range(1, n - 3):
        pre = slice(max(0, i - 4), i)
        post = slice(i, min(n, i + 8))
        rise = (1.0 * (K[post].mean() - K[pre].mean()) + 0.8 * (B[post].mean() - B[pre].mean())
                + 0.4 * (S[post].mean() - S[pre].mean()) + 0.4 * (L[post].mean() - L[pre].mean()))
        sustain = float(np.mean(K[post] + 0.5 * B[post] >= 0.5 * (k_hi + 0.5 * np.percentile(B, 85))))
        scores[i] = max(0.0, rise) * (0.4 + 0.6 * sustain)
    # phrase lines (every 4 bars from the first musical bar) are likelier
    anchor = drums_in if drums_in is not None else int(np.searchsorted(bars, s.first_sound - 0.05))
    for i in range(n):
        if (i - anchor) % 4 == 0:
            scores[i] *= 1.15
    # non-maximum suppression: one candidate per 8 bars
    order = np.argsort(scores)[::-1]
    picked: list[int] = []
    for i in order:
        if scores[i] < 0.06 or len(picked) >= 5:
            break
        if all(abs(i - j) >= 8 for j in picked):
            picked.append(int(i))
    if drums_in is not None and all(abs(drums_in - j) >= 2 for j in picked):
        picked.append(drums_in)
    top = max((scores[i] for i in picked), default=0.0)
    cands = []
    for i in picked:
        kind = "drums_in" if i == drums_in else ("drop" if scores[i] >= 0.999 * top else "section")
        cands.append(Candidate(float(bars[i]), kind, float(scores[i] / max(top, 1e-9))))
    cands.sort(key=lambda c: c.time)
    s.candidates = cands
    # default: the earliest strong moment in the first 60% of the song
    early = [c for c in cands if c.time <= 0.6 * duration and c.score >= 0.7]
    if early:
        s.best = early[0].time
    elif cands:
        s.best = max(cands, key=lambda c: c.score).time
    else:
        s.best = float(bars[min(n - 1, anchor)])
    if early and early[0].kind != "drums_in":
        early[0].kind = "drop"
    # the beat is there from the first bar: start a few bars in, so the
    # slow-mo intro has song to play over
    bar = float(np.median(np.diff(bars)))
    if s.best - s.first_sound < bar - 1e-3:
        i = int(np.searchsorted(bars, s.best - 1e-3)) + (4 if 4 * bar <= 9.0 else 2)
        if i < n:
            s.best = float(bars[i])
            if not any(abs(c.time - s.best) < 1e-3 for c in cands):
                cands.append(Candidate(s.best, "section", 0.5))
                cands.sort(key=lambda c: c.time)
    # song end: last bar that is still loud (outro fades / silence come after)
    ref = np.percentile(L, 60)
    loud_bars = np.nonzero(L >= ref - 0.25)[0]
    last = int(loud_bars[-1]) if len(loud_bars) else n - 1
    s.song_end = float(bars[last + 1]) if last + 1 < n else float(min(duration, bars[-1] + (
        bars[-1] - bars[-2])))
    return s


def auto_intro_length(start: float, bars, first_sound: float, beat: float) -> float:
    """Seconds of song to use as the slow-mo intro before ``start``.

    A short song intro (the start is within ~12 s) is used whole, from the
    first bar with sound; otherwise 4 bars (2 when 4 bars are > 9 s). Never
    reaches back into silence before the song.
    """
    bars = np.asarray(bars, float)
    before = bars[bars < start - 1e-3]
    music = before[before >= first_sound - 0.25 * beat]
    if not len(music):
        return max(0.0, start - first_sound)
    begin = float(music[0])
    if start <= 12.0 and start - begin >= 2 * beat:
        return start - begin
    bar = 4 * beat
    n_bars = 4 if 4 * bar <= 9.0 else 2
    return min(n_bars * bar, start - begin)
