"""Synthetic songs with known beats, bars and sections (for beat-grid tests).

Each generator returns ``(audio, info)`` where ``audio`` is mono float32 at
``SR`` and ``info`` holds the ground truth: ``bpm``, ``first_beat`` (time of
beat 1 of bar 1), ``kick_in`` (where the drums enter) and ``drop``.
"""
from __future__ import annotations

import wave

import numpy as np

SR = 22050


def _t(n):
    return np.arange(n) / SR


def kick(level=0.9):
    t = _t(int(0.2 * SR))
    return (np.sin(2 * np.pi * (50 + 110 * np.exp(-t * 35)) * t) * np.exp(-t * 16) * level)


def clap(rng, level=0.45):
    n = int(0.12 * SR)
    t = _t(n)
    noise = rng.standard_normal(n)
    # crude band-pass: difference of two smoothings keeps ~1-5 kHz
    a = np.convolve(noise, np.ones(3) / 3, "same") - np.convolve(noise, np.ones(15) / 15, "same")
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t * 40) * 0.4
    return (a * 1.6 * np.exp(-t * 32) + body) * level


def hat(rng, level=0.12):
    n = int(0.03 * SR)
    noise = rng.standard_normal(n)
    hp = noise - np.convolve(noise, np.ones(4) / 4, "same")
    return hp * np.exp(-_t(n) * 150) * level


def tone(freq, dur, level=0.25, decay=6.0, harmonics=(1.0, 0.4, 0.15)):
    t = _t(int(dur * SR))
    y = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    env = np.exp(-t * decay) * np.minimum(1.0, t / 0.004)
    return y * env * level


def _add(y, x, t):
    i = int(round(t * SR))
    if i >= len(y) or i + len(x) <= 0:
        return
    j = min(len(y), i + len(x))
    y[max(0, i):j] += x[max(0, -i): j - i]


CHORDS = [(220.0, 277.2, 329.6), (196.0, 246.9, 293.7), (174.6, 220.0, 261.6),
          (164.8, 207.7, 246.9)]


def _pad(y, t0, dur, chord, level):
    for f in chord:
        _add(y, tone(f, dur, level, decay=0.4, harmonics=(1.0, 0.2)), t0)


def shape_like(duration=100.0, bpm=96.0, first_beat=0.37, seed=0):
    """96 BPM, 3-3-2 marimba intro (8 bars, no drums), syncopated kick,
    claps on 2 & 4, chorus lift at bar 24."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    s16 = beat / 4
    bars = int((duration - first_beat) / (4 * beat))
    for b in range(bars):
        t0 = first_beat + b * 4 * beat
        chord = CHORDS[b % 4]
        # marimba riff: 16ths 0 3 6 8 11 14 (the 3-3-2 pattern)
        for k, pos in enumerate((0, 3, 6, 8, 11, 14)):
            f = chord[k % 3] * 2
            _add(y, tone(f, 0.25, 0.22, decay=14), t0 + pos * s16)
        if b >= 8:
            for pos in (0, 6, 8, 14):                 # dancehall kick
                _add(y, kick(0.85 if pos in (0, 8) else 0.6), t0 + pos * s16)
            for pos in (4, 12):                       # claps on 2 and 4
                _add(y, clap(rng), t0 + pos * s16)
            for pos in range(0, 16, 2):               # 8th hats
                _add(y, hat(rng, 0.08 if pos % 4 else 0.12), t0 + pos * s16)
            _add(y, tone(chord[0] / 2, 4 * beat, 0.18, decay=0.8, harmonics=(1.0,)), t0)
        if b >= 24:
            _pad(y, t0, 4 * beat, chord, 0.05)
            _add(y, tone(chord[0] / 4, 4 * beat, 0.22, decay=0.5, harmonics=(1.0, 0.3)), t0)
    info = dict(bpm=bpm, first_beat=first_beat, kick_in=first_beat + 8 * 4 * beat,
                drop=first_beat + 24 * 4 * beat)
    return _finish(y), info


def edm_like(duration=90.0, bpm=128.0, first_beat=0.21, seed=1):
    """128 BPM: 8-bar pad + off-beat hats intro, 8-bar snare-roll build,
    four-on-the-floor drop at bar 16."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    bars = int((duration - first_beat) / (4 * beat))
    for b in range(bars):
        t0 = first_beat + b * 4 * beat
        chord = CHORDS[(b // 2) % 4]
        _pad(y, t0, 4 * beat, chord, 0.06 if b < 16 else 0.08)
        for k in range(4):
            _add(y, hat(rng, 0.09), t0 + (k + 0.5) * beat)      # off-beat hats
        if 8 <= b < 16:
            step = 2 if b < 12 else 4                            # 8ths then 16ths
            for k in range(4 * step):
                _add(y, clap(rng, 0.12 + 0.03 * (b - 8)), t0 + k * beat / step)
        if b >= 16:
            for k in range(4):
                _add(y, kick(0.9), t0 + k * beat)
                _add(y, tone(chord[0] / 4, beat / 2, 0.25, decay=3, harmonics=(1.0, 0.5)),
                     t0 + (k + 0.5) * beat)
            for k in (1, 3):
                _add(y, clap(rng), t0 + k * beat)
    info = dict(bpm=bpm, first_beat=first_beat, kick_in=first_beat + 16 * 4 * beat,
                drop=first_beat + 16 * 4 * beat)
    return _finish(y), info


def backbeat_like(duration=60.0, bpm=101.0, first_beat=0.3, seed=2, swing=0.0,
                  drift=0.0, lead_silence=0.0):
    """Plain rock/pop backbeat: kick 1 & 3, snare 2 & 4, 8th hats.

    ``drift`` (fraction) makes the tempo wander like a live band;
    ``lead_silence`` seconds of silence come first.
    """
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    t = first_beat + lead_silence
    times = []
    k = 0
    while t < duration - 0.3:
        times.append(t)
        wobble = 1 + drift * np.sin(2 * np.pi * k / 48.0)
        t += beat * wobble
        k += 1
    for k, tb in enumerate(times):
        bar_pos = k % 4
        if bar_pos % 2 == 0:
            _add(y, kick(), tb)
        else:
            _add(y, clap(rng), tb)
        _add(y, hat(rng), tb)
        nxt = times[k + 1] if k + 1 < len(times) else tb + beat
        _add(y, hat(rng, 0.08), tb + (nxt - tb) * (0.5 + swing))
        if bar_pos == 0:
            _pad(y, tb, 4 * beat, CHORDS[(k // 4) % 4], 0.04)
    info = dict(bpm=bpm, first_beat=times[0], beats=np.array(times), kick_in=times[0],
                drop=times[0])
    return _finish(y), info


def halftime_like(duration=60.0, bpm=140.0, first_beat=0.12, seed=3):
    """Trap-style half time: kick on 1 (+ a pickup), snare on 3, 16th hats."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    bars = int((duration - first_beat) / (4 * beat))
    for b in range(bars):
        t0 = first_beat + b * 4 * beat
        _add(y, kick(), t0)
        _add(y, kick(0.6), t0 + 2.75 * beat)
        _add(y, clap(rng, 0.55), t0 + 2 * beat)
        for k in range(16):
            _add(y, hat(rng, 0.07 if k % 2 else 0.1), t0 + k * beat / 4)
        _pad(y, t0, 4 * beat, CHORDS[b % 4], 0.04)
    info = dict(bpm=bpm, first_beat=first_beat, kick_in=first_beat, drop=first_beat)
    return _finish(y), info



def pop_syncopated_like(duration=120.0, bpm=96.0, first_beat=0.37, seed=0, claps=0.0,
                        kick_pattern="tresillo", vocals=True, intro_bars=8, jitter=0.008,
                        mastered=False):
    """A harder Shape-of-You-like song: the 3-3-2 marimba riff (a little
    loose in time), vocal-like notes on random 16ths, a 3-3-2 ("tresillo")
    or dancehall kick and quiet or no claps on 2 & 4, so no backbeat marks
    the beat."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    s16 = beat / 4
    bars = int((duration - first_beat) / (4 * beat))
    kicks = (0, 3, 6, 8, 11, 14) if kick_pattern == "tresillo" else (0, 6, 8, 14)
    for b in range(bars):
        t0 = first_beat + b * 4 * beat
        chord = CHORDS[b % 4]
        for k, pos in enumerate((0, 3, 6, 8, 11, 14)):
            _add(y, tone(chord[k % 3] * 2, 0.25, 0.22 * rng.uniform(0.7, 1.1), decay=14),
                 t0 + pos * s16 + rng.normal(0, jitter))
        if vocals and b >= 4:
            for pos in sorted(rng.choice(16, size=int(rng.integers(4, 9)), replace=False)):
                _add(y, tone(rng.choice(chord) * rng.choice([1, 2]), rng.uniform(0.1, 0.35), 0.16,
                             decay=4, harmonics=(1.0, 0.6, 0.4, 0.2)),
                     t0 + pos * s16 + rng.normal(0, 0.02))
        if b >= intro_bars:
            for pos in kicks:
                _add(y, kick(0.7 if pos in (0, 8) else 0.5), t0 + pos * s16 + rng.normal(0, jitter / 2))
            if claps:
                for pos in (4, 12):
                    _add(y, clap(rng, claps), t0 + pos * s16 + rng.normal(0, jitter / 2))
            _add(y, tone(chord[0] / 2, 4 * beat, 0.18, decay=0.8, harmonics=(1.0,)), t0)
        if b >= 24:
            _pad(y, t0, 4 * beat, chord, 0.05)
    info = dict(bpm=bpm, first_beat=first_beat, kick_in=first_beat + intro_bars * 4 * beat,
                drop=first_beat + max(24, intro_bars) * 4 * beat)
    return (master(y, seed) if mastered else _finish(y)), info


def dembow_like(duration=90.0, bpm=95.0, first_beat=0.2, seed=4):
    """Reggaeton: kick on every beat, the dembow snare on 16ths 3 6 11 14."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    s16 = beat / 4
    bars = int((duration - first_beat) / (4 * beat))
    for b in range(bars):
        t0 = first_beat + b * 4 * beat
        chord = CHORDS[b % 4]
        for pos in (0, 4, 8, 12):
            _add(y, kick(0.8), t0 + pos * s16)
        for pos in (3, 6, 11, 14):
            _add(y, clap(rng, 0.35), t0 + pos * s16)
        for pos in range(0, 16, 2):
            _add(y, hat(rng, 0.07), t0 + pos * s16)
        for pos in (0, 3, 6, 8, 11, 14):
            _add(y, tone(chord[0] / 2, 0.2, 0.2, decay=6, harmonics=(1.0, 0.3)), t0 + pos * s16)
        _pad(y, t0, 4 * beat, chord, 0.04)
    info = dict(bpm=bpm, first_beat=first_beat, kick_in=first_beat, drop=first_beat)
    return _finish(y), info


def phonk_like(duration=90.0, bpm=130.0, first_beat=0.15, seed=6):
    """Drift phonk: a syncopated 16th cowbell riff, 808 kicks, a clap on 2
    and 4 and rattling 16th hats."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    s16 = beat / 4
    bars = int((duration - first_beat) / (4 * beat))
    for b in range(bars):
        t0 = first_beat + b * 4 * beat
        chord = CHORDS[(b // 2) % 4]
        for k, pos in enumerate((0, 3, 6, 8, 10, 11, 14)):
            _add(y, tone(chord[k % 3] * 3, 0.12, 0.2, decay=25, harmonics=(1.0, 0.8, 0.5, 0.3)),
                 t0 + pos * s16)
        if b >= 4:
            for pos in (0, 7, 10):
                _add(y, kick(0.85), t0 + pos * s16)
                _add(y, tone(chord[0] / 4, 0.4, 0.25, decay=3, harmonics=(1.0,)), t0 + pos * s16)
            for pos in (4, 12):
                _add(y, clap(rng, 0.4), t0 + pos * s16)
            for pos in range(16):
                _add(y, hat(rng, 0.05 if pos % 2 else 0.08), t0 + pos * s16)
    info = dict(bpm=bpm, first_beat=first_beat, kick_in=first_beat + 4 * 4 * beat,
                drop=first_beat + 4 * 4 * beat)
    return _finish(y), info


def trap_triplet_like(duration=80.0, bpm=140.0, first_beat=0.1, seed=8):
    """Trap: half-time snare (beat 3), sparse 808 kicks and hi-hats in 8th
    note triplets with rolls: reads as 140 or 70."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(duration * SR))
    beat = 60.0 / bpm
    bars = int((duration - first_beat) / (4 * beat))
    for b in range(bars):
        t0 = first_beat + b * 4 * beat
        for pos in (0.0, 1.5, 2.75):
            _add(y, kick(0.85), t0 + pos * beat)
        _add(y, clap(rng, 0.5), t0 + 2 * beat)
        for k in range(12):                                  # 8th-note triplets
            _add(y, hat(rng, 0.09 if k % 3 == 0 else 0.06), t0 + k * beat / 3)
        if b % 2:                                            # a 32nd roll
            for k in range(8):
                _add(y, hat(rng, 0.05), t0 + 3 * beat + k * beat / 8)
        _pad(y, t0, 4 * beat, CHORDS[b % 4], 0.04)
    info = dict(bpm=bpm, first_beat=first_beat, kick_in=first_beat, drop=first_beat)
    return _finish(y), info


def master(y, seed=0):
    """Like a released track: a short room reverb and a soft limiter."""
    from scipy.signal import fftconvolve

    rng = np.random.default_rng(seed + 100)
    n = int(0.35 * SR)
    ir = rng.standard_normal(n) * np.exp(-_t(n) * 14) * 0.05
    ir[0] = 1.0
    y = fftconvolve(y, ir)[: len(y)]
    y = y / max(1e-6, np.percentile(np.abs(y), 99.5))
    return _finish(np.tanh(1.8 * y))

def _finish(y):
    y = y / max(1e-6, np.abs(y).max()) * 0.9
    return y.astype(np.float32)


def write_wav(path, y, sr=SR):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
