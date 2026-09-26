"""Soundtrack building: music + a Minecraft hit sound on every placed hit.

Hit sounds are loaded from the user's own Minecraft Java install (the asset
index maps sound names to hashed object files), so no game assets ship with
this project. Without an install we fall back to a synthesized punch, and a
custom sound file can always be chosen instead.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import wave
from functools import lru_cache

import numpy as np

from .ffmpeg_utils import NO_WINDOW, require_ffmpeg

SR = 44100

# preset -> (preferred asset indexes, sound names)
PRESETS = {
    "classic": (["1.8"], [f"minecraft/sounds/damage/hit{i}.ogg" for i in (1, 2, 3)]),
    "strong": ([], [f"minecraft/sounds/entity/player/attack/strong{i}.ogg" for i in range(1, 7)]),
    "crit": ([], [f"minecraft/sounds/entity/player/attack/crit{i}.ogg" for i in (1, 2, 3)]),
    "knockback": ([], [f"minecraft/sounds/entity/player/attack/knockback{i}.ogg"
                       for i in range(1, 5)]),
}
HIT_SOUND_CHOICES = ["original", "classic", "strong", "crit", "knockback", "custom", "off"]


def find_minecraft_dir() -> str | None:
    env = os.environ.get("HITSYNC_MINECRAFT_DIR")
    cands = [env] if env else []
    if sys.platform == "win32":
        cands.append(os.path.join(os.environ.get("APPDATA", ""), ".minecraft"))
    elif sys.platform == "darwin":
        cands.append(os.path.expanduser("~/Library/Application Support/minecraft"))
    else:
        cands.append(os.path.expanduser("~/.minecraft"))
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "assets", "indexes")):
            return c
    return None


def _index_order(indexes: list[str], preferred: list[str]) -> list[str]:
    def version_key(path):
        name = os.path.splitext(os.path.basename(path))[0]
        nums = [int(x) for x in name.replace("-", ".").split(".") if x.isdigit()]
        return nums or [0]

    pref = [p for p in indexes if os.path.splitext(os.path.basename(p))[0] in preferred]
    rest = sorted((p for p in indexes if p not in pref), key=version_key, reverse=True)
    return pref + rest


@lru_cache(maxsize=16)
def minecraft_sound_files(preset: str) -> tuple[str, ...]:
    """Absolute paths of a preset's .ogg objects in the local Minecraft install."""
    mc = find_minecraft_dir()
    if not mc or preset not in PRESETS:
        return ()
    preferred, names = PRESETS[preset]
    indexes = glob.glob(os.path.join(mc, "assets", "indexes", "*.json"))
    for idx_path in _index_order(indexes, preferred):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                objects = json.load(f).get("objects", {})
        except (OSError, ValueError):
            continue
        found = []
        for name in names:
            h = objects.get(name, {}).get("hash")
            if h:
                p = os.path.join(mc, "assets", "objects", h[:2], h)
                if os.path.isfile(p):
                    found.append(p)
        if found:
            return tuple(found)
    return ()


def decode_audio(path: str, sr: int = SR) -> np.ndarray:
    """Decode any audio file to float32 stereo (N, 2) via ffmpeg."""
    cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", path,
           "-vn", "-f", "f32le", "-ac", "2", "-ar", str(sr), "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    if proc.returncode != 0:
        raise RuntimeError(f"Could not decode {path}: {proc.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).reshape(-1, 2).copy()


_music_cache: dict = {}


def load_music(path: str) -> np.ndarray:
    """Decoded song (44.1 kHz stereo float32); the last few are kept."""
    key = (path, os.path.getmtime(path) if os.path.exists(path) else 0)
    if key not in _music_cache:
        while len(_music_cache) >= 4:
            _music_cache.pop(next(iter(_music_cache)))
        _music_cache[key] = decode_audio(path)
    return _music_cache[key]


def _trim_silence(x: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    loud = np.nonzero(np.abs(x).max(axis=1) > thresh)[0]
    return x[max(0, loud[0] - 8):] if len(loud) else x


def synth_punch(sr: int = SR) -> np.ndarray:
    """Fallback hit: a short pitched-down thud with a noisy transient."""
    n = int(0.16 * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(7)
    body = np.sin(2 * np.pi * (180 * np.exp(-t * 25) + 70) * t) * np.exp(-t * 28)
    click = rng.standard_normal(n) * np.exp(-t * 120) * 0.5
    y = (body + click) * 0.8
    return np.stack([y, y], axis=1).astype(np.float32)


@lru_cache(maxsize=8)
def _load_samples(preset: str, custom_file: str) -> tuple:
    if preset == "custom" and custom_file and os.path.isfile(custom_file):
        return (_trim_silence(decode_audio(custom_file)),), f"custom: {os.path.basename(custom_file)}"
    files = minecraft_sound_files(preset if preset != "custom" else "classic")
    if files:
        return tuple(_trim_silence(decode_audio(f)) for f in files), \
            f"Minecraft assets ({len(files)} variants)"
    return (synth_punch(),), "synthesized punch (Minecraft install not found)"


def hit_samples(preset: str, custom_file: str = ""):
    """(samples, description) for a preset; samples is () when sounds are off."""
    if preset == "off":
        return (), "hit sounds off"
    if preset == "original":
        return (), "original hit sounds from the gameplay recording"
    return _load_samples(preset, custom_file or "")


@lru_cache(maxsize=2)
def _game_audio(video_path: str, mtime: float):
    """(samples, seconds-offset) of the recording's own audio track.

    The offset converts our source time (relative to the first video
    frame) into an index in the decoded audio, whose first sample sits at
    the audio stream's own start timestamp.
    """
    from .ffmpeg_utils import first_audio_pts, first_video_pts

    a0 = first_audio_pts(video_path)
    if a0 is None:
        return None, 0.0
    return decode_audio(video_path), first_video_pts(video_path) - a0


@lru_cache(maxsize=2)
def _sound_bursts(video_path: str, mtime: float):
    """Attack times (s, audio timeline) and strengths of every distinct sound.

    Onsets are found on the spectral flux, then moved onto the audible
    attack in the waveform (the flux of a 46 ms window reacts ~20-30 ms
    early), so a cut there starts right at the hit sound.
    """
    import librosa

    from .music_grid import refine_attacks

    game, _ = _game_audio(video_path, mtime)
    if game is None or not len(game):
        return np.zeros(0), np.zeros(0)
    y = game.mean(axis=1).astype(np.float32)
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop,
                                        units="frames", backtrack=False)
    strength = env[frames] if len(frames) else np.zeros(0)
    times = librosa.frames_to_time(frames, sr=SR, hop_length=hop)
    return refine_attacks(y, SR, times, search=(0.05, 0.04)), strength


def _hit_onsets(video_path: str, src_times, sr: int = SR):
    """Locate each hit's own sound in the recording's audio.

    Returns (game_audio, onsets, found) or None if the video has no audio.
    ``onsets[i]`` is the sample index where hit i's sound starts.

    Screen recordings don't keep sound and picture in lockstep: in a Game
    Bar capture the hit sound led the red flash by 30-50 ms in places and by
    up to ~340 ms in others, varying hit to hit. So instead of searching a
    fixed window, every distinct sound burst is detected and the hits are
    matched to bursts in order: each hit takes an unused burst from 450 ms
    before to 120 ms after its flash, the one closest to the running
    offset. Where no burst is found the running offset is used, so every
    hit still gets its sound (``found[i]`` is False then).
    """
    if not video_path or not os.path.isfile(video_path):
        return None
    mtime = os.path.getmtime(video_path)
    game, offset = _game_audio(video_path, mtime)
    if game is None or not len(game):
        return None
    bursts, strength = _sound_bursts(video_path, mtime)
    if len(strength):
        keep = strength >= 0.15 * np.percentile(strength, 95)   # ignore faint rustle
        bursts, strength = bursts[keep], strength[keep]
    flashes = np.asarray(src_times, float) + offset
    order = np.argsort(flashes)

    def candidates(t, after):
        m = (bursts >= t - 0.45) & (bursts <= t + 0.12) & (bursts > after)
        return bursts[m], strength[m]

    # starting offset: median lag of the loudest nearby burst per hit
    lags = []
    for t in flashes:
        b, st = candidates(t, -np.inf)
        if len(b):
            lags.append(b[np.argmax(st)] - t)
    run = float(np.median(lags)) if lags else -0.04
    onsets = np.zeros(len(flashes), int)
    found = np.zeros(len(flashes), bool)
    last = -np.inf
    for i in order:
        t = flashes[i]
        b, st = candidates(t, last + 0.2)
        if len(b):
            strong = st >= 0.3 * st.max()
            b = b[strong]
            pick = float(b[np.argmin(np.abs((b - t) - run))])
            run = 0.7 * run + 0.3 * (pick - t)
            last, found[i] = pick, True
        else:
            pick = t + run
        onsets[i] = int(np.clip(round(pick * sr), 0, len(game) - 1))
    return game, list(onsets), list(found)


def original_hit_snippets(video_path: str, src_times, length: float = 0.35, sr: int = SR):
    """Short clips of each hit's sound (None where no clear transient)."""
    located = _hit_onsets(video_path, src_times, sr)
    if located is None:
        return None
    game, onsets, found = located
    pre, n_len = int(0.008 * sr), int(length * sr)
    out = []
    for o, ok in zip(onsets, found):
        if not ok:
            out.append(None)
            continue
        snip = game[max(0, o - pre): max(0, o - pre) + n_len]
        out.append(np.pad(snip, ((0, n_len - len(snip)), (0, 0))).astype(np.float32))
    return out


def original_hit_track(video_path: str, placed, t0: float, n: int, volume: float,
                       sr: int = SR, tail: float = 1.0):
    """The recording's own sound for every hit, kept whole, attacks on the beat.

    Each hit plays the gameplay audio from its attack up to the next hit's
    attack (so the entire hit sound and anything after it - crit, sweep,
    knockback - is kept), placed so the attack lands on the hit's beat.
    Where the edit's gap is shorter than the recording's, only the last few
    ms before the next hit are crossfaded away; the last hit of a combo rings
    out for ``tail`` seconds. One gain for all hits keeps natural dynamics.
    Returns an (n, 2) array or None if the video has no audio.
    """
    placed = sorted(placed, key=lambda p: p.out_t)
    located = _hit_onsets(video_path, [p.src_t for p in placed], sr)
    if located is None:
        return None
    game, onsets, _ = located
    track = np.zeros((n, 2), np.float32)
    if not placed:
        return track
    pre, xf = int(0.002 * sr), int(0.015 * sr)
    peaks = [float(np.abs(game[o: o + int(0.08 * sr)]).max()) for o in onsets]
    gain = volume * 0.7 / max(1e-4, float(np.median(peaks)))
    for i, (pl, o) in enumerate(zip(placed, onsets)):
        nxt = placed[i + 1] if i + 1 < len(placed) else None
        continuous = (nxt is not None and 0 < nxt.src_t - pl.src_t < 1.2
                      and 0 < nxt.out_t - pl.out_t < 1.2)
        if continuous:
            length = min(onsets[i + 1] - o, int(round((nxt.out_t - pl.out_t) * sr))) + xf
            fade = xf
        elif nxt is not None and nxt.out_t - pl.out_t < tail:
            # last hit before a cut: ring out until the next combo's first hit
            # starts (never over it), with a short crossfade
            length, fade = int(round((nxt.out_t - pl.out_t) * sr)) + xf, int(0.05 * sr)
        else:
            length, fade = int(tail * sr), int(tail * 0.4 * sr)
        seg = game[max(0, o - pre): o + length].astype(np.float32)
        if len(seg) < 2:
            continue
        env = np.ones(len(seg), np.float32)
        a = min(len(seg), int(0.003 * sr))
        env[:a] = np.linspace(0, 1, a)
        f = min(len(seg), fade)
        env[len(seg) - f:] *= np.linspace(1, 0, f)
        # line up the attack itself (first sample at half the sound's peak)
        # with the beat; the quiet lead-in before it is kept as well
        head = np.abs(game[o: o + int(0.08 * sr)]).max(axis=1)
        attack = int(np.argmax(head > 0.5 * head.max())) if len(head) and head.max() > 0 else 0
        i0 = int(round((pl.out_t - t0) * sr)) - min(pre, o) - attack
        a0, b0 = max(0, i0), min(n, i0 + len(seg))
        if b0 > a0:
            track[a0:b0] += (seg * env[:, None])[a0 - i0: b0 - i0] * gain
    return track


def _repitch(x: np.ndarray, rate: float) -> np.ndarray:
    """Resample so playback is ``rate`` times faster (Minecraft varies pitch)."""
    if abs(rate - 1.0) < 1e-3:
        return x
    n = max(2, int(len(x) / rate))
    src = np.linspace(0, len(x) - 1, n)
    idx = np.arange(len(x))
    return np.stack([np.interp(src, idx, x[:, c]) for c in range(x.shape[1])], axis=1)


def _soft_limit(y: np.ndarray, knee: float = 0.9) -> np.ndarray:
    a = np.abs(y)
    over = a > knee
    if over.any():
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return y


def build_soundtrack(music, schedule, rparams, duration: float | None = None,
                     sr: int = SR, video_path: str | None = None, click: bool = False) -> np.ndarray:
    """The montage soundtrack: music, hit sounds on every placed hit, fades.

    ``music`` is a song path or a ``soundtrack.MusicTimeline`` (several
    songs crossfading). ``hit_sound == "original"`` uses the real hit sound
    from the recording (``video_path``); without usable audio there it
    falls back to the classic Minecraft sound. ``click`` adds a metronome
    on the montage's beats (to check the beat grid by ear).
    """
    from .soundtrack import MusicTimeline

    duration = schedule.duration if duration is None else duration
    n = int(round(duration * sr))
    t0 = float(getattr(schedule, "start", 0.0))      # song before this is cut
    if isinstance(music, MusicTimeline):
        out = music.render(t0, n, sr, load_music) * float(rparams.music_volume)
    else:
        m0 = int(round(t0 * sr))
        song = load_music(music)[m0: m0 + n] * float(rparams.music_volume)
        out = np.zeros((n, 2), np.float32)
        out[: len(song)] = song
    fade_in = _fade_len(rparams, "in", duration)
    k = min(n, int(max(0.03 if t0 > 0 else 0.0, fade_in) * sr))
    if k > 0:                                         # no click where the song is cut
        out[:k] *= np.linspace(0, 1, k, dtype=np.float32)[:, None]
    end = t0 + duration
    placed = [pl for pl in schedule.placements
              if pl.out_t is not None and t0 <= pl.out_t < end]
    preset = rparams.hit_sound
    if preset == "original":
        track = original_hit_track(video_path, placed, t0, n, float(rparams.hit_volume), sr)
        if track is not None:
            out += track
            preset = None
        else:
            preset = "classic"
    samples, _ = hit_samples(preset, rparams.hit_sound_file) if preset else ((), "")
    if samples:
        rng = np.random.default_rng(1234)   # deterministic: preview == export
        vol = float(rparams.hit_volume)
        for pl in sorted(placed, key=lambda p: p.out_t):
            s = samples[rng.integers(len(samples))]
            if rparams.hit_pitch_variation:
                s = _repitch(s, float(rng.uniform(0.85, 1.15)))
            i = int(round((pl.out_t - t0) * sr))
            j = min(n, i + len(s))
            out[i:j] += s[: j - i] * vol
    if click:
        from .audio_analysis import click_track

        beats = np.asarray(getattr(schedule, "beat_times", []), float) - t0
        downs = np.asarray(getattr(schedule, "downbeats", []), float) - t0
        out += click_track(beats, downs, n, sr)[:, None] * 0.8
    fade = _fade_len(rparams, "out", duration)
    if fade <= 0:
        fade = min(float(rparams.audio_fade_out), duration / 3)
        if placed:        # fade after the last hit only, never over the combo
            fade = min(fade, max(0.1, end - max(pl.out_t for pl in placed)))
    if fade > 0:
        k = min(n, int(fade * sr))
        out[n - k:] *= np.linspace(1, 0, k, dtype=np.float32)[:, None]
    return _soft_limit(out)


def _fade_len(rparams, which: str, duration: float) -> float:
    kind = getattr(rparams, f"fade_{which}", "none")
    if not kind or kind == "none":
        return 0.0
    return float(min(getattr(rparams, f"fade_{which}_len", 1.0), duration / 3))


def write_wav(path: str, audio: np.ndarray, sr: int = SR):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
