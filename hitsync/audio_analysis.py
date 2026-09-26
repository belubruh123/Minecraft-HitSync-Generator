"""Music analysis: constant beat grid, downbeats, song sections, loudness.

The grid, downbeats and start/drop candidates come from ``music_grid`` and
``sections`` (numpy only, one STFT pass). A classic beat tracker (librosa)
is only run when it is needed: for songs whose tempo wanders (live bands)
and for the "dynamic grid" mode. Its beats are snapped onto the measured
attacks, so they sit on the kick, not a frame after it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import music_grid, sections as sections_mod
from .music_grid import HOP, SR, Grid
from .sections import Sections


def _arr(d, k, dtype=float):
    v = d.get(k)
    return np.asarray(v if v is not None else [], dtype=dtype)


@dataclass
class AudioAnalysis:
    path: str
    duration: float
    grid: Grid
    sections: Sections
    onsets: np.ndarray             # attack times (s)
    drum_w: np.ndarray             # drum-weighted strength per attack
    env_rate: float
    env: np.ndarray                # onset envelope (beat tracker input)
    curve_t: np.ndarray            # ~23 ms loudness/activity curves, 0..1
    level: np.ndarray
    bass: np.ndarray
    kick: np.ndarray
    snare: np.ndarray
    tracked: np.ndarray | None = field(default=None, repr=False)

    # ----------------------------------------------------------- derived
    @property
    def tempo(self) -> float:
        return self.grid.bpm

    @property
    def rms_times(self) -> np.ndarray:        # display curve (timeline)
        return self.curve_t

    @property
    def rms(self) -> np.ndarray:
        return self.level

    @property
    def drums(self) -> np.ndarray:
        return np.clip(0.6 * self.kick + 0.4 * self.snare, 0, 1)

    @property
    def beats(self) -> np.ndarray:
        """Beat-tracker beats (dynamic grid); computed on first use."""
        if self.tracked is None:
            self.tracked = track_beats(self.env, self.env_rate, self.onsets, self.grid.bpm,
                                       self.duration)
        return self.tracked

    def grid_for(self, bpm: float = 0.0, offset: float = 0.0) -> Grid:
        """The static grid, optionally with a forced tempo and a phase nudge."""
        g = self.grid
        if bpm > 0 and abs(bpm - g.bpm) > 1e-6:
            forced = music_grid.grid_with_tempo(self.onsets, self.drum_w, self.duration, bpm)
            # keep the detected bar lines: of the fitted phase and the one half
            # a beat away (a syncopated riff fits both), the one that puts a
            # beat on the detected downbeat; then the downbeat nearest it
            first_db = g.phase + g.period * g.downbeat
            P = forced.period
            phase = min((forced.phase, (forced.phase + P / 2) % P),
                        key=lambda ph: abs(((first_db - ph) / P + 0.5) % 1 - 0.5))
            k = int(round((first_db - phase) / P)) % 4
            g = Grid(forced.bpm, phase, k, forced.confidence, False, g.candidates)
        if offset:
            P = g.period
            shifted = g.phase + offset
            turns = int(np.floor(shifted / P))
            g = Grid(g.bpm, shifted - turns * P, (g.downbeat - turns) % 4, g.confidence,
                     g.drift, g.candidates)
        return g

    # ----------------------------------------------------------- persist
    def to_dict(self):
        r = lambda a, n=4: None if a is None else np.round(np.asarray(a, float), n).tolist()  # noqa: E731
        return {"path": self.path, "duration": self.duration, "grid": self.grid.to_dict(),
                "sections": self.sections.to_dict(), "onsets": r(self.onsets, 5),
                "drum_w": r(self.drum_w, 3), "env_rate": self.env_rate, "env": r(self.env, 3),
                "curve_t": r(self.curve_t, 4), "level": r(self.level, 3),
                "bass": r(self.bass, 3), "kick": r(self.kick, 3), "snare": r(self.snare, 3),
                "tracked": r(self.tracked, 5)}

    @classmethod
    def from_dict(cls, d):
        if "grid" not in d:
            raise KeyError("old analysis format")
        tracked = d.get("tracked")
        return cls(d["path"], float(d["duration"]), Grid.from_dict(d["grid"]),
                   Sections.from_dict(d.get("sections")), _arr(d, "onsets"), _arr(d, "drum_w"),
                   float(d["env_rate"]), _arr(d, "env", np.float32), _arr(d, "curve_t"),
                   _arr(d, "level", np.float32), _arr(d, "bass", np.float32),
                   _arr(d, "kick", np.float32), _arr(d, "snare", np.float32),
                   None if tracked is None else np.asarray(tracked, float))


def analyze_audio(path: str, progress=None) -> AudioAnalysis:
    report = progress or (lambda *_: None)
    report("Loading audio", 0.05)
    y = music_grid.decode_mono(path, SR)
    ma = music_grid.analyze(y, SR, progress=lambda m, f: report(m, 0.1 + 0.75 * f))
    report("Song sections", 0.9)
    beats = ma.grid.beats(ma.duration)
    bars = beats[ma.grid.is_downbeat(beats)]
    secs = sections_mod.detect(ma.curve_t, ma.level, ma.bass, ma.kick, ma.snare, bars,
                               ma.duration)
    a = AudioAnalysis(path, ma.duration, ma.grid, secs, ma.onsets, ma.drum_w, ma.env_rate,
                      ma.env, ma.curve_t, ma.level, ma.bass, ma.kick, ma.snare)
    if ma.grid.drift:              # live tempo: the tracked beats are the grid
        report("Tracking a live tempo", 0.95)
        a.beats  # noqa: B018  (computes and stores them)
    report("Audio done", 1.0)
    return a


def _fold_octave(bpm: np.ndarray, ref: float) -> np.ndarray:
    out = bpm.copy()
    for _ in range(3):
        out = np.where(out > ref * 1.45, out / 2, out)
        out = np.where(out < ref / 1.45, out * 2, out)
    return out


def track_beats(env, rate, onsets, bpm_hint: float, duration: float) -> np.ndarray:
    """Beat tracking that follows tempo changes, snapped onto attacks."""
    import librosa
    from scipy.ndimage import median_filter

    env = np.asarray(env, float)
    if len(env) < 16:
        return np.zeros(0)
    sr, hop = SR, int(round(SR / rate))
    local = np.atleast_1d(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop,
                                                aggregate=None, ac_size=6.0))
    ref = bpm_hint if bpm_hint > 0 else float(np.median(local))
    local = _fold_octave(local.astype(float), ref)
    local = median_filter(local, size=max(3, int(4 * rate)), mode="nearest")
    try:
        _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop,
                                           bpm=local, units="time", trim=False)
    except Exception:
        _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=hop,
                                           bpm=ref, units="time", trim=False)
    return snap_to_attacks(np.asarray(beats, float), np.asarray(onsets, float))


def snap_to_attacks(beats: np.ndarray, attacks: np.ndarray, max_shift: float = 0.07):
    """Move each beat onto the nearest measured attack (within max_shift)."""
    if not len(beats) or not len(attacks):
        return beats
    i = np.clip(np.searchsorted(attacks, beats), 1, max(1, len(attacks) - 1))
    lo = attacks[i - 1]
    hi = attacks[np.minimum(i, len(attacks) - 1)]
    near = np.where(np.abs(lo - beats) <= np.abs(hi - beats), lo, hi)
    out = np.where(np.abs(near - beats) <= max_shift, near, beats)
    return np.unique(np.round(out, 4))


def click_track(beats, downbeats, n: int, sr: int = 44100) -> np.ndarray:
    """Metronome clicks (accented on downbeats) as a mono float32 track."""
    out = np.zeros(n, np.float32)
    t = np.arange(int(0.03 * sr)) / sr
    for freq, times in ((1500.0, beats), (2500.0, downbeats)):
        click = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 180) * 0.5).astype(np.float32)
        for b in times:
            i = int(round(b * sr))
            if 0 <= i < n:
                j = min(n, i + len(click))
                out[i:j] = click[: j - i]
    return out
