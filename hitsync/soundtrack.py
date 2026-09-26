"""Several songs back to back, as one music timeline.

The montage runs on "music time". With one song that is simply the song's
own time. With more, song 1 plays until its last loud bar (``end``); the
next song crossfades in over about a bar, playing the lead-up to its own
start point, and reaches that start point exactly on the bar where song 1
hands over: no gap, and the beat grid continues without a hiccup (song 1's
beats up to the hand-over, then song 2's from its start point).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Song:
    path: str
    duration: float
    start: float = 0.0          # song time that lands on the hand-over (songs 2+)
    end: float = 0.0            # song time where it hands over to the next song
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    downbeats: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def bar(self) -> float:
        b = self.downbeats
        return float(np.median(np.diff(b))) if len(b) > 1 else 2.0


@dataclass
class Piece:
    song: int
    offset: float               # music time = song time + offset
    out_a: float                # audible from (music time), incl. fade-in
    out_b: float                # audible until (music time), incl. fade-out
    fade_in: float = 0.0        # seconds of fade-in starting at out_a
    fade_out: float = 0.0       # seconds of fade-out ending at out_b


class MusicTimeline:
    def __init__(self, songs: List[Song], xfade_bars: float = 1.0):
        if not songs:
            raise ValueError("no songs")
        self.songs = songs
        self.pieces: List[Piece] = []
        self.handovers: List[float] = []
        first = songs[0]
        end = first.duration
        if len(songs) > 1:
            end = first.end if first.end > 0 else first.duration
        self.pieces.append(Piece(0, 0.0, 0.0, end))
        for k in range(1, len(songs)):
            prev, song = self.pieces[-1], songs[k]
            T = prev.out_b                                  # hand-over bar line
            xf = float(np.clip(songs[k - 1].bar * xfade_bars, 1.0, 4.0))
            lead = min(xf, song.start)                      # can't start before 0
            last = k == len(songs) - 1
            s_end = song.duration if last or song.end <= song.start else song.end
            offset = T - song.start
            prev.fade_out = min(xf, prev.out_b - prev.out_a)
            self.pieces.append(Piece(k, offset, T - lead, s_end + offset, fade_in=lead))
            self.handovers.append(T)

    # ------------------------------------------------------------ timing
    @property
    def duration(self) -> float:
        return self.pieces[-1].out_b

    def _visible(self, attr: str) -> np.ndarray:
        out = []
        for i, p in enumerate(self.pieces):
            s = self.songs[p.song]
            t = np.asarray(getattr(s, attr), float) + p.offset
            lo = self.handovers[i - 1] if i > 0 else -np.inf
            hi = self.handovers[i] if i < len(self.handovers) else np.inf
            out.append(t[(t >= lo - 1e-6) & (t < hi - 1e-6)])
        return np.concatenate(out) if out else np.zeros(0)

    def beats(self) -> np.ndarray:
        return self._visible("beats")

    def downbeats(self) -> np.ndarray:
        return self._visible("downbeats")

    def song_at(self, t: float) -> tuple[int, float]:
        """(song index, song time) of the song you hear at music time t."""
        k = int(np.searchsorted(self.handovers, t, side="right"))
        p = self.pieces[k]
        return p.song, t - p.offset

    # ------------------------------------------------------------ audio
    def render(self, t0: float, n: int, sr: int, load) -> np.ndarray:
        """Music time [t0, t0 + n/sr) as (n, 2) float32; ``load(path)``
        returns a song's (samples, 2) float32 at ``sr``."""
        out = np.zeros((n, 2), np.float32)
        t1 = t0 + n / sr
        for p in self.pieces:
            a, b = max(t0, p.out_a), min(t1, p.out_b)
            if b <= a:
                continue
            audio = load(self.songs[p.song].path)
            i0, i1 = int(round((a - t0) * sr)), int(round((b - t0) * sr))
            s0 = int(round((a - p.offset) * sr))
            seg = audio[max(0, s0): max(0, s0) + (i1 - i0)]
            if s0 < 0:
                seg = np.concatenate([np.zeros((-s0, 2), np.float32), seg])[: i1 - i0]
            seg = seg.astype(np.float32, copy=True)
            times = a + np.arange(len(seg)) / sr
            gain = np.ones(len(seg), np.float32)
            if p.fade_in > 0:
                u = np.clip((times - p.out_a) / p.fade_in, 0, 1)
                gain *= np.sin(u * np.pi / 2) ** 2          # equal-power-ish crossfade
            if p.fade_out > 0:
                u = np.clip((p.out_b - times) / p.fade_out, 0, 1)
                gain *= np.sin(u * np.pi / 2) ** 2
            out[i0:i0 + len(seg)] += seg * gain[:, None]
        return out


def single(path: str, duration: float, beats=(), downbeats=()) -> MusicTimeline:
    return MusicTimeline([Song(path, duration, 0.0, duration, np.asarray(beats, float),
                               np.asarray(downbeats, float))])
