"""Synthetic test media: a click-track song with a drop + fake PvP footage."""
from __future__ import annotations

import wave

import cv2
import numpy as np

SR = 22050
DROP = 4.0
TEMPO_CHANGE = 14.0


def song_beats(duration=24.0):
    """120 BPM until TEMPO_CHANGE, then 150 BPM."""
    beats, t = [], 0.0
    while t < duration:
        beats.append(t)
        t += 0.5 if t < TEMPO_CHANGE - 1e-9 else 0.4
    return np.array(beats)


def write_song(path, duration=24.0):
    n = int(duration * SR)
    y = np.zeros(n, np.float32)
    rng = np.random.default_rng(0)
    kick_t = np.arange(int(0.18 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 + 90 * np.exp(-kick_t * 30)) * kick_t) * np.exp(-kick_t * 18)
    tick_t = np.arange(int(0.05 * SR)) / SR
    tick = rng.standard_normal(len(tick_t)) * np.exp(-tick_t * 80)
    for b in song_beats(duration):
        i = int(b * SR)
        if b < DROP:
            seg = tick * 0.08
        else:
            seg = np.zeros(len(kick))
            seg[:] = kick * 0.9
            seg[:len(tick)] += tick * 0.3
        j = min(n, i + len(seg))
        y[i:j] += seg[: j - i]
    # post-drop bass pad to lift RMS energy
    t = np.arange(n) / SR
    y += (t >= DROP) * 0.15 * np.sin(2 * np.pi * 55 * t)
    y = np.clip(y, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


# Ground-truth hits: three combos with human jitter, long walking gaps between.
HITS = [5.0, 5.53, 6.02, 6.47, 7.04,
        12.0, 12.55, 13.03, 13.52,
        19.0, 19.47, 20.02]


def write_video(path, duration=26.0, fps=30, size=(320, 180)):
    w, h = size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rng = np.random.default_rng(1)
    tex = cv2.resize((rng.random((h // 4, w // 2)) * 120 + 40).astype(np.uint8), (w * 2, h),
                     interpolation=cv2.INTER_NEAREST)
    tex = cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)
    tex[..., 1] = np.clip(tex[..., 1].astype(int) + 40, 0, 255)  # greenish "grass"
    hit_frames = {int(round(t * fps)) for t in HITS}
    red_until = -1
    for k in range(int(duration * fps)):
        x = int(k * 3) % w
        frame = tex[:, x:x + w].copy()
        cv2.putText(frame, f"{k / fps:5.2f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
        # opponent in the crosshair
        cx, cy = w // 2, h // 2
        if k in hit_frames:
            red_until = k + 7          # hurt tint lasts ~0.25 s
        color = (40, 40, 230) if k <= red_until else (150, 110, 70)
        cv2.rectangle(frame, (cx - 18, cy - 30), (cx + 18, cy + 30), color, -1)
        vw.write(frame)
    vw.release()
