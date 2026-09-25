"""Music analysis: beats (with time-varying tempo), RMS energy, drop detection."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from .ffmpeg_utils import extract_audio_wav, find_ffmpeg

SR = 22050
HOP = 256          # ~11.6 ms beat-time resolution


@dataclass
class AudioAnalysis:
    path: str
    duration: float
    tempo: float                 # global BPM estimate
    beats: np.ndarray            # beat times (s)
    rms_times: np.ndarray
    rms: np.ndarray              # RMS normalised to 0..1
    tempo_times: np.ndarray      # librosa frame-wise tempo (for display)
    tempo_bpm: np.ndarray
    perc_rms: np.ndarray | None = None   # drums-only RMS (same frames as rms)

    def to_dict(self):
        return {"path": self.path, "duration": self.duration, "tempo": self.tempo,
                "beats": self.beats.tolist(), "rms_times": self.rms_times.tolist(),
                "rms": self.rms.tolist(), "tempo_times": self.tempo_times.tolist(),
                "tempo_bpm": self.tempo_bpm.tolist(),
                "perc_rms": None if self.perc_rms is None else self.perc_rms.tolist()}

    @classmethod
    def from_dict(cls, d):
        arr = lambda k: np.asarray(d.get(k, []), dtype=float)  # noqa: E731
        return cls(d["path"], float(d["duration"]), float(d["tempo"]), arr("beats"),
                   arr("rms_times"), arr("rms"), arr("tempo_times"), arr("tempo_bpm"),
                   arr("perc_rms") if d.get("perc_rms") is not None else None)


def load_audio(path: str, sr: int = SR):
    import librosa

    if find_ffmpeg():
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            extract_audio_wav(path, tmp, sr=sr)
            y, _ = librosa.load(tmp, sr=sr, mono=True)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    else:
        y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr


def _fold_octave(bpm: np.ndarray, ref: float) -> np.ndarray:
    out = bpm.copy()
    for _ in range(3):
        out = np.where(out > ref * 1.45, out / 2, out)
        out = np.where(out < ref / 1.45, out * 2, out)
    return out


def analyze_audio(path: str, progress=None) -> AudioAnalysis:
    import librosa
    from scipy.ndimage import median_filter

    report = progress or (lambda *_: None)
    report("Loading audio", 0.05)
    y, sr = load_audio(path)
    duration = len(y) / sr

    report("Onset envelope", 0.25)
    # One mel spectrogram feeds both the onset envelope and the drum split
    # (same result as onset_strength(y=...), without computing it twice).
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP)
    oenv = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max), sr=sr,
                                        hop_length=HOP, aggregate=np.median)

    report("Tempo tracking", 0.45)
    global_tempo = float(np.atleast_1d(
        librosa.feature.tempo(onset_envelope=oenv, sr=sr, hop_length=HOP))[0])
    local = np.atleast_1d(librosa.feature.tempo(
        onset_envelope=oenv, sr=sr, hop_length=HOP, aggregate=None, ac_size=6.0))
    local = _fold_octave(local.astype(float), global_tempo)
    frames_per_sec = sr / HOP
    local = median_filter(local, size=max(3, int(4 * frames_per_sec)), mode="nearest")

    report("Beat tracking", 0.65)
    try:
        # librosa >= 0.10 accepts a time-varying tempo curve here.
        _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP,
                                           bpm=local, units="time", trim=False)
    except Exception:
        _, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=HOP,
                                           bpm=global_tempo, units="time", trim=False)
    beats = np.asarray(beats, dtype=float)

    report("Energy", 0.85)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    beats = snap_beats_to_onsets(beats, oenv, sr)
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=HOP)
    rms = rms / (rms.max() + 1e-9)
    tempo_times = librosa.frames_to_time(np.arange(len(local)), sr=sr, hop_length=HOP)

    # The drop is where the drums come in, which vocals/pads can mask in a
    # full-mix energy curve: keep a percussive-only RMS for drop detection.
    report("Drum energy", 0.92)
    perc = librosa.feature.rms(y=percussive(y), hop_length=HOP)[0]
    perc = perc[: len(rms)] / (perc.max() + 1e-9)

    report("Audio done", 1.0)
    return AudioAnalysis(path, duration, global_tempo, beats, rms_times, rms,
                         tempo_times, local, perc)


def _median_filter(S: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """scipy median_filter split across threads (it releases the GIL).

    The array is cut along the axis the kernel doesn't span, so every block
    sees exactly the neighbours it would see unsplit: identical output.
    """
    from concurrent.futures import ThreadPoolExecutor

    from scipy.ndimage import median_filter

    other = 0 if size[0] == 1 else 1
    n = int(np.clip(os.cpu_count() or 1, 1, 16))
    edges = np.linspace(0, S.shape[other], n + 1).astype(int)

    def block(k):
        sl = [slice(None), slice(None)]
        sl[other] = slice(edges[k], edges[k + 1])
        return median_filter(S[tuple(sl)], size=size, mode="reflect")

    with ThreadPoolExecutor(n) as ex:
        return np.concatenate(list(ex.map(block, range(n))), axis=other)


def percussive(y: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """Same as librosa.effects.percussive(y), with the two median filters
    (90% of its time) run in parallel."""
    import librosa

    D = librosa.stft(y)
    S, phase = librosa.magphase(D)
    harm = _median_filter(S, (1, kernel_size))
    perc = _median_filter(S, (kernel_size, 1))
    mask = librosa.util.softmask(perc, harm, power=2.0, split_zeros=True)
    return librosa.istft((S * mask) * phase, dtype=y.dtype, length=len(y))


def snap_beats_to_onsets(beats, oenv, sr, max_shift: float = 0.07) -> np.ndarray:
    """Move each beat onto the attack of the nearest detected onset.

    The beat tracker reports the onset-strength peak, which trails the audible
    transient by a frame or two. Backtracking onsets to the preceding minimum
    of the onset envelope gives the true attack (~±4 ms in testing), so hits
    lock to the start of the kick. (Backtracking on wide-frame RMS overshoots.)
    """
    import librosa

    if not len(beats):
        return beats
    frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=HOP,
                                        units="frames")
    if not len(frames):
        return beats
    frames = librosa.onset.onset_backtrack(frames, oenv)
    onsets = librosa.frames_to_time(frames, sr=sr, hop_length=HOP)
    idx = np.clip(np.searchsorted(onsets, beats), 1, len(onsets) - 1) if len(onsets) > 1 else None
    out = beats.copy()
    for k, b in enumerate(beats):
        cand = onsets if idx is None else onsets[idx[k] - 1: idx[k] + 1]
        near = cand[np.argmin(np.abs(cand - b))]
        if abs(near - b) <= max_shift:
            out[k] = near
    return np.unique(np.round(out, 4))


def detect_drop(rms_times, rms, beats, search_start: float = 2.0,
                search_end: float | None = None, window: float = 4.0) -> float:
    """Find the beat where energy jumps the most (the "drop").

    Score for each candidate beat = mean dB energy in the ``window`` after it
    minus the mean in the window before it, favouring loud post-drop sections.
    """
    rms_times = np.asarray(rms_times, float)
    rms = np.asarray(rms, float)
    if len(rms) < 4:
        return 0.0
    duration = float(rms_times[-1])
    if search_end is None:
        search_end = min(duration * 0.6, 120.0)
    db = 20 * np.log10(rms + 1e-4)
    dt = rms_times[1] - rms_times[0]
    k = max(1, int(0.25 / dt))
    db = np.convolve(db, np.ones(k) / k, mode="same")
    csum = np.concatenate([[0.0], np.cumsum(db)])

    def mean_between(a, b):
        i, j = np.searchsorted(rms_times, [a, b])
        j = max(j, i + 1)
        return (csum[min(j, len(db))] - csum[i]) / max(1, min(j, len(db)) - i)

    candidates = np.asarray(beats, float)
    if len(candidates) < 4:
        candidates = np.arange(search_start, search_end, 0.1)
    candidates = candidates[(candidates >= search_start) & (candidates <= search_end)]
    if not len(candidates):
        return 0.0
    loud = np.percentile(db, 60)
    best_t, best = float(candidates[0]), -np.inf
    for t in candidates:
        pre = mean_between(max(0.0, t - window), t)
        post = mean_between(t, min(duration, t + window))
        score = (post - pre) + 0.25 * min(0.0, post - loud)
        if score > best:
            best, best_t = score, float(t)
    return best_t
