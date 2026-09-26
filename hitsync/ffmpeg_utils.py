"""Locating and invoking the ffmpeg CLI.

Resolution order: ``HITSYNC_FFMPEG`` env var, ``ffmpeg`` on PATH, then the
static binary bundled with the ``imageio-ffmpeg`` wheel (so the tool works on
machines without a system-wide ffmpeg install).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from functools import lru_cache

import numpy as np

# Keep ffmpeg from flashing a console window when launched from the GUI.
NO_WINDOW = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


@lru_cache(maxsize=1)
def find_ffmpeg() -> str | None:
    env = os.environ.get("HITSYNC_FFMPEG")
    if env and os.path.isfile(env):
        return env
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return None


def require_ffmpeg() -> str:
    exe = find_ffmpeg()
    if not exe:
        hint = ("`brew install ffmpeg`" if sys.platform == "darwin" else
                "`winget install Gyan.FFmpeg`" if sys.platform == "win32" else
                "your package manager (e.g. `sudo apt install ffmpeg`)")
        raise RuntimeError(
            f"ffmpeg not found. Install it with {hint}, run `pip install imageio-ffmpeg`, "
            "or set HITSYNC_FFMPEG to the ffmpeg executable.")
    return exe


def run_ffmpeg(args: list[str]) -> subprocess.CompletedProcess:
    cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, creationflags=NO_WINDOW)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({proc.returncode}): {proc.stderr.strip()[-800:]}")
    return proc


def first_video_pts(path: str) -> float:
    """Timestamp of the first video frame on ffmpeg's seek timeline.

    OpenCV reports frame times relative to the first frame, but ffmpeg's
    ``-ss`` is relative to the container start, and screen recordings often
    begin their video stream later (e.g. 0.0667 s). Seeks must add this.
    """
    import re

    return _first_pts(path, "v")


def first_audio_pts(path: str) -> float | None:
    """Timestamp of the first audio sample (None when there's no audio)."""
    return _first_pts(path, "a")


def _first_pts(path: str, kind: str):
    import re

    flt = ["-frames:v", "1", "-vf", "showinfo"] if kind == "v" else ["-af", "ashowinfo", "-t", "0.2"]
    cmd = [require_ffmpeg(), "-hide_banner", "-nostdin", "-i", path, "-map", f"0:{kind}:0",
           *flt, "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                          creationflags=NO_WINDOW)
    m = re.search(r"pts_time:\s*(-?[0-9.]+)", proc.stderr)
    if m:
        return float(m.group(1))
    return 0.0 if kind == "v" else None


@lru_cache(maxsize=1)
def hwaccel_args() -> list[str]:
    """ffmpeg input options for GPU decoding, or [] when none works here.

    ``-hwaccel auto`` picks D3D11VA/DXVA2/CUDA on Windows, VideoToolbox on
    macOS, VAAPI/CUDA on Linux, and quietly decodes on the CPU when the GPU
    can't. One tiny test decode proves the GPU path actually opens.
    Set HITSYNC_HWACCEL=0 to disable.
    """
    if os.environ.get("HITSYNC_HWACCEL", "1") == "0":
        return []
    exe = find_ffmpeg()
    if not exe:
        return []
    method = "videotoolbox" if sys.platform == "darwin" else "auto"
    cmd = [exe, "-hide_banner", "-loglevel", "verbose", "-nostdin", "-hwaccel", method,
           "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:d=0.3", "-f", "null", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=20,
                           creationflags=NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    # lavfi input isn't hardware-decodable; this only proves the option is
    # accepted. Actual GPU use is decided per file, with a CPU retry.
    return ["-hwaccel", method]


HW_ENCODERS = {"nvenc": "h264_nvenc", "amf": "h264_amf", "qsv": "h264_qsv",
               "videotoolbox": "h264_videotoolbox"}


@lru_cache(maxsize=8)
def encoder_works(name: str) -> bool:
    """Can this ffmpeg actually open the (hardware) encoder on this machine?

    Listing it isn't enough: builds ship every GPU encoder, but only the
    one matching the installed GPU/driver opens. One tiny test encode.
    """
    exe = find_ffmpeg()
    if not exe:
        return False
    cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "lavfi",
           "-i", "color=c=gray:s=256x256:r=30:d=0.2", "-c:v", name, "-pix_fmt", "yuv420p",
           "-f", "null", "-"]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=20,
                              creationflags=NO_WINDOW).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def video_encoder_args(choice: str, crf: int, preset: str) -> tuple[str, list[str]]:
    """(label, ffmpeg args) for the chosen encoder, at roughly equal quality."""
    choice = (choice or "auto").lower()
    auto = ["videotoolbox"] if sys.platform == "darwin" else ["nvenc", "amf", "qsv"]
    order = auto if choice == "auto" else [choice] if choice in HW_ENCODERS else []
    for key in order:
        name = HW_ENCODERS[key]
        if not encoder_works(name):
            continue
        q = str(int(crf))
        if key == "nvenc":
            return name, ["-c:v", name, "-preset", "p5", "-rc", "vbr", "-cq", q, "-b:v", "0"]
        if key == "amf":
            return name, ["-c:v", name, "-quality", "quality", "-rc", "cqp",
                          "-qp_i", q, "-qp_p", str(int(crf) + 2)]
        if key == "videotoolbox":
            # quality scale 1-100; crf 18 -> ~70
            return name, ["-c:v", name, "-q:v", str(int(np.clip(125 - 3 * int(crf), 40, 90))),
                          "-allow_sw", "1"]
        return name, ["-c:v", name, "-preset", "medium", "-global_quality", q]
    return "libx264", ["-c:v", "libx264", "-preset", preset, "-crf", str(int(crf))]


def open_path(path: str):
    """Open a file or folder with the system's default app."""
    if sys.platform == "win32":
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", path])
    else:
        subprocess.Popen(["xdg-open", path])


def extract_audio_wav(src: str, dst: str, sr: int = 22050, mono: bool = True):
    """Decode any audio/video container to a PCM wav librosa can read."""
    args = ["-i", src, "-vn", "-ar", str(sr)]
    if mono:
        args += ["-ac", "1"]
    args += ["-c:a", "pcm_s16le", dst]
    run_ffmpeg(args)
