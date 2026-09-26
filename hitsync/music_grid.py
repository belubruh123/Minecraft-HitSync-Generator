"""Constant-tempo beat grid, downbeats and onset attacks, straight from audio.

Montage music is almost always produced on a DAW grid: one exact tempo for the
whole song. So instead of fitting a line through a beat tracker's output
(which follows syncopation: the 3-3-2 riff of "Shape of You" reads as a
128 BPM dotted-8th pulse, or drifts to 97 BPM), the tempo and phase are
searched directly: every candidate period folds the song's drum onsets onto
one beat, and the true period is the one where they stack up. A wrong period
smears the onsets across the fold once the pattern repeats.

Everything is numpy/scipy (no librosa), from a single STFT pass.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field

import numpy as np

SR = 22050
HOP = 256            # onset-envelope frame: 11.6 ms
N_FFT = 2048
MIN_BPM, MAX_BPM = 55.0, 215.0


# ---------------------------------------------------------------- decoding
def decode_mono(path: str, sr: int = SR) -> np.ndarray:
    """Any audio/video file -> mono float32 at ``sr`` (ffmpeg pipe, no temp file)."""
    from .ffmpeg_utils import NO_WINDOW, find_ffmpeg

    exe = find_ffmpeg()
    if exe:
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path, "-vn",
               "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
        if proc.returncode == 0 and proc.stdout:
            return np.frombuffer(proc.stdout, np.float32).copy()
        raise RuntimeError(f"Could not decode {path}: "
                           f"{proc.stderr.decode(errors='replace')[-300:]}")
    import soundfile as sf     # no ffmpeg: plain audio files only

    y, file_sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if file_sr != sr:
        from scipy.signal import resample_poly

        g = np.gcd(int(file_sr), sr)
        y = resample_poly(y, sr // g, int(file_sr) // g).astype(np.float32)
    return y


# ------------------------------------------------------------ spectral pass
BANDS = {"sub": (25.0, 90.0), "low": (25.0, 150.0), "mid": (150.0, 1500.0),
         "high": (1500.0, 6000.0), "air": (6000.0, 11025.0)}


def _filterbank(sr: int, n_fft: int, n_bands: int = 40, fmin: float = 25.0):
    """Triangular log-frequency bands (n_bins x n_bands) and their centres."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = np.geomspace(fmin, sr / 2, n_bands + 2)
    fb = np.zeros((len(freqs), n_bands), np.float32)
    for b in range(n_bands):
        lo, c, hi = edges[b], edges[b + 1], edges[b + 2]
        up = (freqs - lo) / max(c - lo, 1e-9)
        down = (hi - freqs) / max(hi - c, 1e-9)
        fb[:, b] = np.clip(np.minimum(up, down), 0, None)
        if not fb[:, b].any():                      # narrow low bands: nearest bin
            fb[int(np.argmin(np.abs(freqs - c))), b] = 1.0
    return fb, edges[1:-1]


def _chroma_matrix(sr: int, n_fft: int):
    """Pitch-class folding of the 180-2500 Hz range (above kick and bass
    sweeps, below cymbal noise), where chords live."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    m = np.zeros((len(freqs), 12), np.float32)
    ok = (freqs >= 180) & (freqs <= 2500)
    pc = np.round(12 * np.log2(freqs[ok] / 440.0) + 69).astype(int) % 12
    m[np.nonzero(ok)[0], pc] = 1.0
    return m


@dataclass
class Spectral:
    """Per-frame features (frame n is centred at n * hop / sr)."""

    sr: int
    hop: int
    n: int
    flux: dict            # band name / "full" -> onset envelope
    level: dict           # band name / "full" -> energy in dB
    chroma: np.ndarray    # (n, 12) power per pitch class

    @property
    def rate(self) -> float:
        return self.sr / self.hop

    def times(self) -> np.ndarray:
        return np.arange(self.n) * (self.hop / self.sr)


def spectral_features(y: np.ndarray, sr: int = SR, hop: int = HOP,
                      n_fft: int = N_FFT) -> Spectral:
    """One windowed-FFT pass reduced to band energies + chroma (blockwise).

    scipy's FFT runs in float32 on all cores (numpy's is float64, one core).
    """
    import scipy.fft

    fb, centres = _filterbank(sr, n_fft)
    M = np.concatenate([fb, _chroma_matrix(sr, n_fft)], axis=1)
    pad = n_fft // 2
    yp = np.pad(y.astype(np.float32), (pad, pad + hop))
    n = 1 + len(y) // hop
    win = np.hanning(n_fft).astype(np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(yp, n_fft)[::hop][:n]
    out = np.empty((n, M.shape[1]), np.float32)
    block = 1024
    for i in range(0, n, block):
        spec = scipy.fft.rfft(frames[i:i + block] * win, axis=1, workers=-1)
        out[i:i + block] = (spec.real ** 2 + spec.imag ** 2) @ M
    bands, chroma = out[:, :fb.shape[1]], out[:, fb.shape[1]:]
    L = 10 * np.log10(bands + 1e-10)
    L = np.maximum(L, L.max() - 80.0)
    d = np.diff(L, axis=0, prepend=L[:1])
    d = np.clip(d, 0, None)
    flux, level = {}, {}
    groups = {k: (centres >= lo) & (centres < hi) for k, (lo, hi) in BANDS.items()}
    groups["full"] = np.ones(len(centres), bool)
    for name, sel in groups.items():
        if not sel.any():
            sel = np.ones(len(centres), bool)
        flux[name] = d[:, sel].mean(axis=1)
        level[name] = 10 * np.log10(bands[:, sel].sum(axis=1) + 1e-10)
    # energy-weighted kick activity: magnitude rise in the sub band (a log
    # flux would also fire on faint transients in an otherwise empty band)
    mag = np.sqrt(bands[:, groups["sub"]]).sum(axis=1)
    flux["kick"] = np.clip(np.diff(mag, prepend=mag[:1]), 0, None)
    # loudness-weighted onsets over all bands: a kick or snare counts for
    # much more than a quiet hat (the log flux treats them alike)
    allmag = np.sqrt(bands)
    flux["energy"] = np.clip(np.diff(allmag, axis=0, prepend=allmag[:1]), 0, None).sum(axis=1)
    return Spectral(sr, hop, n, flux, level, chroma)


# ------------------------------------------------------------ onset attacks
def _norm(x: np.ndarray) -> np.ndarray:
    s = float(np.percentile(x, 99)) if len(x) else 1.0
    return x / s if s > 1e-12 else x


def onset_peaks(env: np.ndarray, rate: float, delta: float = 0.05):
    """Peak-pick an onset envelope (librosa-style), returns frame indices."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    e = _norm(env)
    k_max = max(1, int(round(0.03 * rate)))
    k_avg = max(1, int(round(0.1 * rate)))
    local_max = maximum_filter1d(e, size=2 * k_max + 1, mode="nearest")
    local_avg = uniform_filter1d(e, size=2 * k_avg + 1, mode="nearest")
    cand = np.nonzero((e == local_max) & (e >= local_avg + delta) & (e > 0))[0]
    if not len(cand):
        return cand
    keep = [cand[0]]
    wait = max(1, int(round(0.03 * rate)))
    for c in cand[1:]:
        if c - keep[-1] > wait:
            keep.append(c)
    return np.asarray(keep)


def refine_attacks(y: np.ndarray, sr: int, approx: np.ndarray, search=(0.02, 0.07)):
    """Move each approximate onset onto its audible attack.

    The log spectral flux of a centred 93 ms window reacts as soon as the
    window's leading edge reaches a sound, ~30 ms before the attack. In the
    waveform the attack is the start of the steepest rise of a short energy
    envelope (pre-emphasised so hats and claps count, not just bass).
    """
    if not len(approx):
        return approx
    pre = np.empty_like(y)
    pre[0] = y[0]
    pre[1:] = y[1:] - 0.95 * y[:-1]
    w = max(2, int(0.001 * sr))
    energy = np.convolve(pre.astype(np.float64) ** 2, np.ones(w) / w, "same")
    db = 10 * np.log10(energy + 1e-9)
    step = max(1, int(0.002 * sr))
    out = np.empty(len(approx))
    before, after = int(search[0] * sr), int(search[1] * sr)
    for k, t in enumerate(approx):
        c = int(round(t * sr))
        a, b = max(0, c - before), min(len(db) - step, c + after)
        if b - a < 4:
            out[k] = t
            continue
        rise = db[a + step:b + step] - db[a:b]
        out[k] = (a + int(np.argmax(rise)) + step / 2) / sr
    return out


# ---------------------------------------------------------------- tempo fit
def _fold_scores(times, weights, periods, width: float, seg_len: float | None):
    """Kappa-normalised concentration of onsets on a period's best phase.

    For each period: onsets are folded onto one period (per ``seg_len``
    segment, or over the whole song), the weight inside the best ±width
    window is compared with what a random phase would catch. 1 = every
    onset on the grid, 0 = no better than chance.
    """
    B = 64
    seg = np.zeros(len(times), int) if seg_len is None else (times // seg_len).astype(int)
    nseg = int(seg.max()) + 1 if len(seg) else 1
    seg_w = np.bincount(seg, weights=weights, minlength=nseg)
    total = seg_w.sum()
    scores = np.zeros(len(periods))
    phases = np.zeros(len(periods))
    for i, P in enumerate(periods):
        ph = (times / P) % 1.0
        idx = seg * B + (ph * B).astype(int) % B
        h = np.bincount(idx, weights=weights, minlength=nseg * B).reshape(nseg, B)
        k = max(1, int(round(2 * width / P * B)))
        # circular box sum of width k
        hh = np.concatenate([h, h[:, :k]], axis=1)
        cs = np.cumsum(hh, axis=1)
        box = cs[:, k - 1:k - 1 + B] - np.concatenate([np.zeros((nseg, 1)), cs[:, :B - 1]], axis=1)
        best = box.max(axis=1)
        chance = min(0.95, (k / B))
        frac = best / np.maximum(seg_w, 1e-12)
        kappa = (frac - chance) / (1 - chance)
        scores[i] = float((kappa * seg_w).sum() / max(total, 1e-12))
        if nseg == 1:
            j = int(np.argmax(box[0]))
            phases[i] = ((j + k / 2) / B) % 1.0 * P
    return scores, phases


def _prior(bpm):
    return np.exp(-0.5 * (np.log2(np.asarray(bpm) / 110.0) / 0.75) ** 2)


@dataclass
class Grid:
    bpm: float
    phase: float                 # time of a beat (s), 0 <= phase < period
    downbeat: int = 0            # beat index (mod 4) that starts a bar
    confidence: float = 0.0      # 0..1
    drift: bool = False          # tempo wanders: use tracked beats instead
    candidates: list = field(default_factory=list)   # [(bpm, score)] runner-ups

    @property
    def period(self) -> float:
        return 60.0 / self.bpm

    def beats(self, duration: float) -> np.ndarray:
        P = self.period
        n = int(np.ceil((duration - self.phase) / P)) + 1
        g = self.phase + P * np.arange(max(1, n))
        return g[g <= duration + 1e-9]

    def is_downbeat(self, beats: np.ndarray) -> np.ndarray:
        k = np.round((np.asarray(beats) - self.phase) / self.period).astype(int)
        return (k - self.downbeat) % 4 == 0

    def to_dict(self):
        return {"bpm": self.bpm, "phase": self.phase, "downbeat": self.downbeat,
                "confidence": self.confidence, "drift": self.drift,
                "candidates": [list(c) for c in self.candidates]}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["bpm"]), float(d["phase"]), int(d.get("downbeat", 0)),
                   float(d.get("confidence", 0.0)), bool(d.get("drift", False)),
                   [tuple(c) for c in d.get("candidates", [])])


def _local_maxima(x, min_sep):
    order = np.argsort(x)[::-1]
    picked = []
    for i in order:
        if all(abs(i - j) >= min_sep for j in picked):
            picked.append(i)
        if len(picked) >= 10:
            break
    return picked


def estimate_grid(onset_t: np.ndarray, onset_w: np.ndarray, beat_t: np.ndarray,
                  beat_w: np.ndarray, duration: float, bpm: float = 0.0,
                  accent_w: np.ndarray | None = None) -> Grid:
    """Tempo + phase of the constant grid that best fits the onsets.

    ``onset_*``: every attack (all instruments) with its strength;
    ``beat_*``: the same attacks weighted for drums (kick/snare), which
    decide the metrical level and phase. ``bpm`` > 0 fixes the tempo.
    """
    if len(beat_t) < 8:
        return Grid(bpm or 120.0, 0.0, 0, 0.0, True)
    accent_w = beat_w ** 2 if accent_w is None else accent_w
    width = 0.03
    if bpm > 0:
        chosen = float(bpm)
        cands = [(chosen, 1.0)]
    else:
        # 1. coarse: per-segment folds (tolerant of slight drift), 0.1 BPM
        coarse = np.arange(MIN_BPM, MAX_BPM, 0.1)
        s_all, _ = _fold_scores(onset_t, onset_w, 60.0 / coarse, width, 12.0)
        s_drum, _ = _fold_scores(beat_t, beat_w, 60.0 / coarse, width, 12.0)
        score = 0.5 * s_all + s_drum
        peaks = _local_maxima(score, 15)
        # 2. fine: whole-song coherence around each peak
        fine = []
        for i in peaks[:6]:
            b0 = coarse[i]
            grid = np.arange(b0 - 0.12, b0 + 0.12, 0.004)
            sf, _ = _fold_scores(beat_t, beat_w, 60.0 / grid, width, None)
            top = np.nonzero(sf >= sf.max() - 1e-3)[0]      # middle of a flat top
            j = int(top[len(top) // 2])
            fine.append((float(grid[j]), float(score[i]), float(sf[j])))
        # 3. metrical level: fold strength x tempo prior x "not a subdivision"
        scored = [(b, _prior(b) * _level_score(b, beat_t, beat_w, accent_w))
                  for b, _, _ in fine]
        chosen = max(scored, key=lambda c: c[1])[0]
        cands = []
        for b, sc in sorted(scored, key=lambda c: -c[1]):
            if all(abs(b / c[0] - 1) > 0.015 for c in cands):     # distinct tempos only
                cands.append((round(b, 2), round(float(sc), 3)))
        cands = cands[:4]
    P = 60.0 / chosen
    period, phase, conf, drift = _fit_line(beat_t, beat_w, P, fixed=bpm > 0,
                                           duration=duration, phase_w=accent_w)
    if bpm <= 0:
        # A steady song folds as well over the whole song as over 6 s
        # windows; a live, wandering tempo only lines up locally.
        local, _ = _fold_scores(beat_t, beat_w, [period], width, 6.0)
        whole, _ = _fold_scores(beat_t, beat_w, [period], width, None)
        if whole[0] < 0.75 * local[0]:
            drift, conf = True, conf * 0.3
    return Grid(60.0 / period, phase, 0, conf, drift, cands)


def _level_score(bpm, beat_t, beat_w, accent_w) -> float:
    """Fold strength of a tempo, penalised when it is really a subdivision.

    At the true beat level consecutive beats carry similar accents (kick,
    snare, kick, snare). At the 8th-note level every other grid point is a
    weaker off-beat: fold at two periods and compare the two halves, using
    squared kick+snare strength so loud accents dominate.
    """
    P = 60.0 / bpm
    sc, _ = _fold_scores(beat_t, beat_w, [P], 0.03, 12.0)
    _, ph0 = _fold_scores(beat_t, accent_w, [P], 0.03, None)
    ph = ((beat_t - ph0[0]) / P) % 2.0
    on_grid = np.minimum(ph % 1.0, 1 - ph % 1.0) * P < 0.035
    first = (ph < 0.5) | (ph > 1.5)
    a = accent_w[on_grid & first].sum()
    b = accent_w[on_grid & ~first].sum()
    ratio = max(a, b) / max(min(a, b), 1e-9)
    alt = 1.0 / (1.0 + max(0.0, ratio - 1.2))
    return float(max(sc[0], 1e-3) * alt)


def _fit_line(times, weights, P, fixed: bool, duration: float, iters: int = 5,
              phase_w=None):
    """Robust weighted line fit t_k = phase + period * k through the attacks
    nearest each grid point. The starting phase comes from the loud accents
    (``phase_w``), so off-beat hats can't pull the grid onto the off-beats.
    Returns (period, phase, confidence, drift)."""
    _, ph = _fold_scores(times, weights if phase_w is None else phase_w, [P], 0.03, None)
    phase = float(ph[0])
    period = P
    tol = min(0.07, P / 5)
    used = None
    for _ in range(iters):
        k = np.round((times - phase) / period)
        res = times - (phase + period * k)
        near = np.abs(res) < tol
        if near.sum() < 4:
            break
        # one attack per grid point: the strongest
        kk, tt, ww = k[near], times[near], weights[near]
        order = np.lexsort((-ww, kk))
        first = np.concatenate([[True], np.diff(kk[order]) != 0])
        kk, tt, ww = kk[order][first], tt[order][first], ww[order][first]
        if fixed:
            phase = float(np.average(tt - period * kk, weights=ww))
        else:
            A = np.stack([np.ones_like(kk), kk], axis=1) * np.sqrt(ww)[:, None]
            sol = np.linalg.lstsq(A, tt * np.sqrt(ww), rcond=None)[0]
            phase, period = float(sol[0]), float(sol[1])
        res = tt - (phase + period * kk)
        mad = float(np.median(np.abs(res - np.median(res)))) + 1e-4
        tol = float(np.clip(4 * mad, 0.012, min(0.07, P / 5)))
        used = (kk, tt, ww, res)
    phase = phase % period
    conf, drift = _diagnose(times, weights, period, phase, duration)
    return period, phase, conf, drift


def _diagnose(times, weights, period, phase, duration):
    """(confidence, drift) of a fitted grid.

    Each grid beat takes the strongest attack within 70 ms (less than a
    16th note, so syncopation isn't mistaken for drift). Over
    the part of the song that has attacks, confidence is the share of beats
    with an attack within 20 ms. Drift = 16-beat windows sitting at clearly
    different offsets from the line (a live, wandering tempo).
    """
    k = np.round((times - phase) / period)
    res = times - (phase + period * k)
    near = np.abs(res) < min(0.07, period / 5)
    if near.sum() < 8:
        return 0.0, True
    kk, rr, ww = k[near], res[near], weights[near]
    order = np.lexsort((-ww, kk))
    first = np.concatenate([[True], np.diff(kk[order]) != 0])
    kk, rr, ww = kk[order][first], rr[order][first], ww[order][first]
    span = kk.max() - kk.min() + 1
    tight = np.abs(rr) < 0.02
    coverage = tight.sum() / max(1.0, span)
    windows = np.floor(kk / 16).astype(int)
    offs = np.array([np.median(rr[windows == w]) for w in np.unique(windows)
                     if (windows == w).sum() >= 6])
    drift = bool(len(offs) >= 3 and (np.mean(np.abs(offs) > 0.03) > 0.2
                                     or np.ptp(offs) > 0.06))
    spread = float(np.median(np.abs(rr[tight]))) if tight.any() else 0.02
    conf = float(np.clip(coverage * 1.5, 0, 1) * np.clip(1 - spread / 0.02, 0.2, 1))
    if drift:
        conf *= 0.3
    return conf, drift


def find_downbeat(grid: Grid, spec: Spectral, duration: float) -> int:
    """Which beat (mod 4) starts the bar: kick + chord changes, not snare."""
    beats = grid.beats(duration)
    if len(beats) < 8:
        return 0
    rate = spec.rate
    idx = np.clip(np.round(beats * rate).astype(int), 0, spec.n - 1)
    w = max(1, int(0.05 * rate))

    def at_beats(env):
        from scipy.ndimage import maximum_filter1d

        return maximum_filter1d(env, size=2 * w + 1, mode="nearest")[idx]

    kick = _norm(at_beats(spec.flux["low"]))
    snare = _norm(at_beats(spec.flux["high"]))
    # harmony change: median chroma of each beat vs the beat before (the
    # median ignores drum transients inside the beat)
    chroma = np.sqrt(spec.chroma)
    per = np.stack([np.median(chroma[a:max(a + 1, b)], axis=0)
                    for a, b in zip(idx, np.append(idx[1:], spec.n))])
    per = per / (np.linalg.norm(per, axis=1, keepdims=True) + 1e-9)
    change = np.zeros(len(beats))
    change[1:] = 1 - np.sum(per[1:] * per[:-1], axis=1)
    k = np.round((beats - grid.phase) / grid.period).astype(int)
    score = []
    for o in range(4):
        sel = (k % 4) == o
        score.append(kick[sel].mean() - 0.5 * snare[sel].mean() + 2.0 * change[sel].mean())
    return int(np.argmax(score))


# -------------------------------------------------------------- one call
@dataclass
class MusicAnalysis:
    duration: float
    grid: Grid
    onsets: np.ndarray             # attack times (s)
    strengths: np.ndarray          # full-band onset strength of each attack
    drum_w: np.ndarray             # kick/snare-weighted strength of each attack
    env_rate: float
    env: np.ndarray                # full-band onset envelope (for the beat tracker)
    curve_t: np.ndarray            # ~23 ms curves, all 0..1:
    level: np.ndarray              #   whole mix loudness
    bass: np.ndarray               #   sub band (kick + bass) loudness
    kick: np.ndarray               #   sub band onset activity
    snare: np.ndarray              #   snare/hat onset activity

    @property
    def drums(self) -> np.ndarray:
        return np.clip(0.6 * self.kick + 0.4 * self.snare, 0, 1)


def analyze(y: np.ndarray, sr: int = SR, bpm: float = 0.0, progress=None) -> MusicAnalysis:
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    report = progress or (lambda *_: None)
    duration = len(y) / sr
    report("Spectrum", 0.1)
    spec = spectral_features(y, sr)
    rate = spec.rate
    report("Onsets", 0.45)
    full = spec.flux["full"]
    peaks = onset_peaks(full, rate)
    attacks = refine_attacks(y, sr, peaks / rate)
    s_full = _norm(full)[peaks]

    def strength(name):
        return maximum_filter1d(_norm(spec.flux[name]), size=5, mode="nearest")[peaks]

    low, high = strength("low"), strength("high")
    drum_w = low + 0.8 * high + 0.25 * strength("air")
    loud = strength("energy")
    report("Tempo", 0.6)
    grid = estimate_grid(attacks, s_full, attacks, drum_w, duration, bpm,
                         (low + high) ** 2 * (0.25 + loud))
    grid.downbeat = find_downbeat(grid, spec, duration)
    report("Levels", 0.9)
    k = 2                                   # ~23 ms curve resolution
    n = spec.n // k * k
    curve_t = (np.arange(n // k) * k + (k - 1) / 2) / rate

    def coarse(x):
        return x[:n].reshape(-1, k).mean(axis=1)

    def loudness(db):
        c = coarse(db)
        lo, hi = np.percentile(c, 2), c.max()
        return np.clip((c - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)

    def activity(flux):
        c = uniform_filter1d(coarse(_norm(flux)), size=max(1, int(0.3 * rate / k)))
        return np.clip(c / max(1e-9, np.percentile(c, 98)), 0, 1).astype(np.float32)

    report("Grid done", 1.0)
    return MusicAnalysis(duration, grid, attacks, s_full, drum_w, rate, full.astype(np.float32),
                         curve_t, loudness(spec.level["full"]), loudness(spec.level["sub"]),
                         activity(spec.flux["kick"]),
                         activity(spec.flux["high"] + 0.5 * spec.flux["air"]))


def grid_with_tempo(onsets, drum_w, duration: float, bpm: float) -> Grid:
    """Re-fit the phase for a forced tempo (BPM override / tap tempo)."""
    return estimate_grid(onsets, drum_w, onsets, drum_w, duration, bpm)


def tap_tempo(taps) -> tuple[float, float] | None:
    """(bpm, phase) from tapped times; None with fewer than 4 taps."""
    t = np.asarray(sorted(taps), float)
    if len(t) < 4:
        return None
    iv = np.diff(t)
    P = float(np.median(iv))
    k = np.round((t - t[0]) / P)
    A = np.stack([np.ones_like(k), k], axis=1)
    phase, period = np.linalg.lstsq(A, t, rcond=None)[0]
    return 60.0 / period, float(phase % period)
