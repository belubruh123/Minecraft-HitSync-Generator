"""Gameplay analysis: centre-screen damage tint + screen velocity -> hits.

Raw per-frame signals are cached in ``VideoAnalysis`` so the user can change
sensitivity and re-detect hits instantly without decoding the video again.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectParams
from .models import Hit


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0

    def to_dict(self):
        return dict(path=self.path, fps=self.fps, frame_count=self.frame_count,
                    width=self.width, height=self.height)

    @classmethod
    def from_dict(cls, d):
        return cls(d["path"], float(d["fps"]), int(d["frame_count"]),
                   int(d["width"]), int(d["height"]))


def probe_video(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if not (1.0 < fps < 1000.0):
        fps = 30.0
    info = VideoInfo(path, float(fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


@dataclass
class VideoAnalysis:
    info: VideoInfo
    times: np.ndarray      # seconds of each analysed frame
    red: np.ndarray        # fraction of damage-tinted pixels in the centre ROI
    motion: np.ndarray     # global screen velocity (px/frame at analysis scale)
    # Presentation timestamp of *every* frame. Screen recorders (Game Bar,
    # OBS) produce variable frame rate video where index/fps drifts by
    # hundreds of ms, so all time<->frame mapping goes through this.
    frame_times: np.ndarray | None = None
    # first frame's timestamp on ffmpeg's seek timeline (see first_video_pts)
    pts_offset: float | None = None

    @property
    def duration(self) -> float:
        if self.frame_times is not None and len(self.frame_times):
            return float(self.frame_times[-1]) + 1.0 / max(1.0, self.info.fps)
        return self.info.duration

    @property
    def sample_rate(self) -> float:
        if len(self.times) < 2:
            return self.info.fps
        return 1.0 / float(np.median(np.diff(self.times)))

    def to_dict(self):
        return {"info": self.info.to_dict(), "times": self.times.tolist(),
                "red": self.red.tolist(), "motion": self.motion.tolist(),
                "frame_times": None if self.frame_times is None else self.frame_times.tolist(),
                "pts_offset": self.pts_offset}

    @classmethod
    def from_dict(cls, d):
        ft = d.get("frame_times")
        return cls(VideoInfo.from_dict(d["info"]), np.asarray(d["times"], float),
                   np.asarray(d["red"], float), np.asarray(d["motion"], float),
                   None if ft is None else np.asarray(ft, float), d.get("pts_offset"))


def damage_tint_fraction(roi_bgr: np.ndarray) -> float:
    """Fraction of pixels showing Minecraft's red hurt overlay."""
    roi = roi_bgr.astype(np.int16)
    b, g, r = roi[..., 0], roi[..., 1], roi[..., 2]
    mask = (r > 90) & (r > g * 1.6 + 10) & (r > b * 1.6 + 10)
    return float(mask.mean())


class _Scanner:
    """Per-frame signals at analysis resolution (shared by both decoders)."""

    def __init__(self, params: DetectParams, aw: int, ah: int):
        self.aw, self.ah = aw, ah
        self.rx0 = int(aw * (0.5 - params.roi_w / 2)); self.rx1 = int(aw * (0.5 + params.roi_w / 2))
        self.ry0 = int(ah * (0.5 - params.roi_h / 2)); self.ry1 = int(ah * (0.5 + params.roi_h / 2))
        self.window = cv2.createHanningWindow((aw // 2, ah // 2), cv2.CV_32F)
        self.prev = None

    def __call__(self, small: np.ndarray) -> tuple[float, float]:
        red = damage_tint_fraction(small[self.ry0:self.ry1, self.rx0:self.rx1])
        gray = cv2.cvtColor(cv2.resize(small, (self.aw // 2, self.ah // 2),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion = 0.0
        if self.prev is not None:
            (dx, dy), _ = cv2.phaseCorrelate(self.prev, gray, self.window)
            motion = float(np.hypot(dx, dy)) * 2.0
        self.prev = gray
        return red, motion


def _analysis_size(params: DetectParams, info: VideoInfo) -> tuple[int, int]:
    aw = int(params.analysis_width) // 2 * 2
    ah = max(2, int(round(aw * info.height / max(1, info.width))) // 2 * 2)
    return aw, ah


def analyze_video(path: str, params: DetectParams | None = None, progress=None,
                  cancel=None, chunks: int | None = None) -> VideoAnalysis:
    """Scan the video for the per-frame damage tint and screen velocity.

    ffmpeg decodes and downscales (the expensive part) in native code and
    streams tiny frames to us (~8x faster than decoding full frames with
    OpenCV); long videos are split into two time chunks decoded in parallel.
    Falls back to OpenCV when ffmpeg isn't available.
    """
    from .ffmpeg_utils import find_ffmpeg

    params = params or DetectParams()
    if find_ffmpeg() is None:
        return _analyze_video_cv(path, params, progress, cancel)
    info = probe_video(path)
    report = progress or (lambda *_: None)
    duration = info.duration if info.duration > 0 else 0.0
    # One ffmpeg already keeps most cores busy decoding; a second chunk
    # fills the gaps (~10% faster on long 1440p captures), more doesn't help.
    n = chunks or (2 if duration > 60 and (os.cpu_count() or 1) >= 4 else 1)
    bounds = [duration * k / n for k in range(n + 1)]
    bounds[0], bounds[-1] = -np.inf, np.inf
    total = max(1, info.frame_count)
    done = [0] * n

    def tick(k, count):
        done[k] = count
        report(f"Scanning video {sum(done)}/{total}", min(0.99, sum(done) / total))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(n) as ex:
        parts = list(ex.map(lambda k: _scan_chunk(path, info, params, bounds[k], bounds[k + 1],
                                                  lambda c: tick(k, c), cancel), range(n)))
    pts = np.concatenate([p[0] for p in parts]) if parts else np.zeros(0)
    red = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0)
    motion = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0)
    if len(pts) > 1:                        # chunk edges: keep one copy per frame
        keep = np.concatenate([[True], np.diff(pts) > 1e-6])
        pts, red, motion = pts[keep], red[keep], motion[keep]
    if len(motion):
        motion[0] = 0.0
    offset = float(pts[0]) if len(pts) else 0.0
    frame_times = pts - offset
    info.frame_count = len(pts)
    # Keep at most max_analysis_fps samples/s (high-fps recordings).
    step = max(1, int(round(info.fps / max(1.0, params.max_analysis_fps))))
    sel = slice(None, None, step)
    if step > 1:
        # motion between analysed samples = sum of the per-frame motions
        motion = np.add.reduceat(motion, np.arange(0, len(motion), step)) if len(motion) else motion
        motion[0] = 0.0
    report("Video scan done", 1.0)
    return VideoAnalysis(info, frame_times[sel].copy(), red[sel].copy(), motion,
                         frame_times, offset)


def _scan_chunk(path, info, params, t0, t1, tick, cancel):
    """Decode frames with pts in [t0, t1) and measure them.

    Returns (pts, red, motion) on ffmpeg's timeline. The chunk starts a
    little early so the first frame's motion has a real predecessor.
    """
    import io
    import queue
    import re
    import subprocess
    import threading

    from .ffmpeg_utils import NO_WINDOW, find_ffmpeg

    aw, ah = _analysis_size(params, info)
    scan = _Scanner(params, aw, ah)
    seek = [] if not np.isfinite(t0) else ["-ss", f"{max(0.0, t0 - 0.5):.6f}"]
    cmd = [find_ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "info", "-nostdin",
           *seek, "-copyts", "-i", path, "-map", "0:v:0",
           "-fps_mode", "passthrough", "-vf", f"scale={aw}:{ah}:flags=area,showinfo=checksum=0",
           "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL, bufsize=0, creationflags=NO_WINDOW)
    stamps: queue.Queue = queue.Queue()
    pat = re.compile(rb" n:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*(-?[0-9.]+)")

    def read_stderr():
        try:
            with io.BufferedReader(proc.stderr, 1 << 16) as err:   # (pipe is unbuffered)
                for line in err:
                    m = pat.search(line)
                    if m:
                        stamps.put(float(m.group(1)))
        except (OSError, ValueError):
            pass
        stamps.put(None)

    reader = threading.Thread(target=read_stderr, daemon=True)
    reader.start()
    size = aw * ah * 3
    buf = bytearray(size)
    view = memoryview(buf)
    pts, red, motion = [], [], []
    count = 0
    try:
        while True:
            if cancel is not None and cancel.is_set():
                break
            got = 0
            while got < size:
                r = proc.stdout.readinto(view[got:])
                if not r:
                    break
                got += r
            if got < size:
                break
            t = stamps.get()
            if t is None:
                break
            if t >= t1:
                break
            r_, m_ = scan(np.frombuffer(buf, np.uint8).reshape(ah, aw, 3))
            count += 1
            if count % 120 == 0:
                tick(count)
            if t < t0:
                continue                    # warm-up frame before this chunk
            pts.append(t); red.append(r_); motion.append(m_)
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()
        reader.join(timeout=5)
        proc.stderr.close()
    return np.asarray(pts, float), np.asarray(red, float), np.asarray(motion, float)


def _analyze_video_cv(path: str, params: DetectParams | None = None, progress=None,
                      cancel=None) -> VideoAnalysis:
    """OpenCV fallback when ffmpeg isn't available (much slower)."""
    params = params or DetectParams()
    report = progress or (lambda *_: None)
    info = probe_video(path)
    cap = cv2.VideoCapture(path)
    step = max(1, int(round(info.fps / params.max_analysis_fps)))
    aw = int(params.analysis_width)
    ah = max(2, int(round(aw * info.height / max(1, info.width))))
    rx0 = int(aw * (0.5 - params.roi_w / 2)); rx1 = int(aw * (0.5 + params.roi_w / 2))
    ry0 = int(ah * (0.5 - params.roi_h / 2)); ry1 = int(ah * (0.5 + params.roi_h / 2))
    window = cv2.createHanningWindow((aw // 2, ah // 2), cv2.CV_32F)

    times, reds, motions, pts_all = [], [], [], []
    prev_gray = None
    idx = 0
    total = max(1, info.frame_count)

    def pts() -> float:
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        fallback = idx / info.fps
        if not np.isfinite(t) or (idx > 0 and (t <= 0 or (pts_all and t <= pts_all[-1]))):
            t = (pts_all[-1] + 1.0 / info.fps) if pts_all else fallback
        return t

    while True:
        if cancel is not None and cancel.is_set():
            break
        if idx % step:
            if not cap.grab():
                break
            pts_all.append(pts())
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        pts_all.append(pts())
        small = cv2.resize(frame, (aw, ah), interpolation=cv2.INTER_AREA)
        reds.append(damage_tint_fraction(small[ry0:ry1, rx0:rx1]))
        gray = cv2.cvtColor(cv2.resize(small, (aw // 2, ah // 2),
                                       interpolation=cv2.INTER_AREA),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is None:
            motions.append(0.0)
        else:
            (dx, dy), _ = cv2.phaseCorrelate(prev_gray, gray, window)
            motions.append(float(np.hypot(dx, dy)) * 2.0)
        prev_gray = gray
        times.append(pts_all[-1])
        idx += 1
        if idx % 60 == 0:
            report(f"Scanning video {idx}/{total}", min(0.99, idx / total))
    cap.release()
    if info.frame_count <= 0 or abs(idx - info.frame_count) > 2:
        info.frame_count = idx  # container frame counts can lie; trust decoding
    report("Video scan done", 1.0)
    from .ffmpeg_utils import find_ffmpeg, first_video_pts

    offset = first_video_pts(path) if find_ffmpeg() else 0.0
    return VideoAnalysis(info, np.asarray(times), np.asarray(reds), np.asarray(motions),
                         np.asarray(pts_all, float), offset)


def _robust_norm(x: np.ndarray) -> np.ndarray:
    scale = np.percentile(x, 99.5) if len(x) else 1.0
    if scale <= 1e-9:
        scale = x.max() if len(x) and x.max() > 0 else 1.0
    return np.clip(x / scale, 0.0, 1.5)


def hit_score(va: VideoAnalysis, params: DetectParams) -> np.ndarray:
    """Combined per-frame hit likelihood (roughly 0..1.5)."""
    if len(va.red) == 0:
        return np.zeros(0)
    from scipy.ndimage import minimum_filter1d

    from scipy.ndimage import maximum_filter1d

    fs = va.sample_rate
    k = max(2, int(round(0.1 * fs)))
    # Onset of red tint: rise above the minimum of the previous ~100 ms.
    prev = np.concatenate([[va.red[0]], va.red[:-1]])
    base = minimum_filter1d(prev, size=k, origin=(k - 1) // 2, mode="nearest")
    rise = np.clip(va.red - base, 0, None)
    # A far-away opponent covers few pixels, so its flash is small in absolute
    # terms. Normalise each rise by the local tint level (±1.5 s) so distant
    # hits score like close ones; the floor stops noise being amplified when
    # nothing red is on screen.
    w = max(3, int(round(3.0 * fs)) | 1)
    local = maximum_filter1d(va.red, size=w, mode="nearest")
    floor = max(0.004, float(np.percentile(va.red, 60)))
    rel = rise / np.maximum(local, floor)
    red_onset = np.clip(0.75 * np.clip(rel / 0.6, 0, 1.5) + 0.25 * _robust_norm(rise), 0, 1.5)
    # Screen-velocity change (camera snap / knockback on impact).
    jerk = np.abs(np.diff(va.motion, prepend=va.motion[:1]))
    jerk = _robust_norm(jerk)
    total_w = max(1e-6, params.red_weight + params.motion_weight)
    return (params.red_weight * red_onset + params.motion_weight * jerk) / total_w


def detect_hits(va: VideoAnalysis, params: DetectParams) -> list[Hit]:
    from scipy.signal import find_peaks

    score = hit_score(va, params)
    if not len(score):
        return []
    threshold = 0.12 + (1.0 - float(np.clip(params.sensitivity, 0, 1))) * 0.55
    dist = max(1, int(params.min_hit_interval * va.sample_rate))
    peaks, props = find_peaks(score, height=threshold, distance=dist)
    peaks = _fill_missed(score, list(peaks), threshold * 0.45, va.sample_rate)
    return [Hit(float(va.times[_leading_edge(score, p, va.sample_rate)]), float(score[p]))
            for p in peaks]


def _fill_missed(score: np.ndarray, peaks: list, low: float, fs: float) -> list:
    """Recover faint hits hiding in a combo gap of ~2x the hit rhythm.

    A far-away opponent's flash can fall under the threshold and split a
    combo in two. When a gap is 1.6-2.4x the typical hit period, the best
    peak near the gap's midpoint is accepted at a lower threshold.
    """
    if len(peaks) < 4:
        return peaks
    gaps = np.diff(peaks) / fs
    normal = gaps[(gaps > 0.4) & (gaps < 1.0)]
    if len(normal) < 3:
        return peaks
    period = float(np.median(normal))
    out = list(peaks)
    for a, b in zip(peaks, peaks[1:]):
        if not 1.6 * period <= (b - a) / fs <= 2.4 * period:
            continue
        mid, half = (a + b) // 2, int(0.3 * period * fs)
        seg = score[mid - half: mid + half + 1]
        if len(seg) and seg.max() >= low:
            out.append(mid - half + int(np.argmax(seg)))
    return sorted(out)


def _leading_edge(score: np.ndarray, p: int, fs: float) -> int:
    """The onset score plateaus for the baseline window; the impact is the
    first frame of that plateau, not wherever find_peaks lands on it."""
    limit = max(0, p - max(2, int(round(0.1 * fs))))
    while p > limit and score[p - 1] >= 0.7 * score[p]:
        p -= 1
    return p


def snap_to_peak(va: VideoAnalysis | None, params: DetectParams, t: float,
                 radius: float = 0.12) -> float:
    """Snap a manually placed hit to the strongest score peak nearby."""
    if va is None or not len(va.times):
        return t
    score = hit_score(va, params)
    i0, i1 = np.searchsorted(va.times, [t - radius, t + radius])
    if i1 <= i0:
        return t
    seg = score[i0:i1]
    if seg.max() < 0.08:
        return t
    return float(va.times[_leading_edge(score, i0 + int(np.argmax(seg)), va.sample_rate)])


def read_frame(path: str, t: float, size: tuple[int, int] | None = None,
               pts_offset: float = 0.0):
    """Frame-accurate single frame at source time t (for previews).

    Uses ffmpeg's decode-accurate timestamp seek; OpenCV's seeking returns
    wrong frames on many variable-frame-rate recordings.
    """
    import subprocess

    from .ffmpeg_utils import NO_WINDOW, find_ffmpeg

    exe = find_ffmpeg()
    if exe is None:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    import re

    info = probe_video(path)
    w, h = size or (info.width, info.height)
    # ffmpeg's accurate seek can drop the wanted frame on VFR captures, so
    # decode a short lead-in with real timestamps and keep the frame on
    # screen at t: the last one whose timestamp is <= t.
    target = max(0.0, t) + pts_offset
    lead = 0.3
    start = max(0.0, target - lead)
    cmd = [exe, "-hide_banner", "-nostats", "-loglevel", "info", "-nostdin",
           "-ss", f"{start:.6f}", "-t", f"{target - start + 0.05:.6f}", "-copyts",
           "-i", path, "-map", "0:v:0",
           "-vf", f"select='between(t\\,{target - lead:.6f}\\,{target + 1e-4:.6f})',"
                  f"scale={w}:{h}:flags=area,showinfo=checksum=0",
           "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    r = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW)
    n = len(r.stdout) // (w * h * 3)
    if n == 0:        # nothing at/before t (e.g. t before the first frame): first frame
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
               "-map", "0:v:0", "-frames:v", "1", "-vf", f"scale={w}:{h}:flags=area",
               "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True, creationflags=NO_WINDOW).stdout
        if len(out) < w * h * 3:
            return None
        return np.frombuffer(out[: w * h * 3], np.uint8).reshape(h, w, 3).copy()
    stamps = [float(x) for x in re.findall(rb" n:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*(-?[0-9.]+)",
                                             r.stderr)]
    k = n - 1
    if len(stamps) == n:
        k = max(i for i in range(n) if stamps[i] <= target + 1e-4) \
            if any(s <= target + 1e-4 for s in stamps) else 0
    frame = r.stdout[k * w * h * 3: (k + 1) * w * h * 3]
    return np.frombuffer(frame, np.uint8).reshape(h, w, 3).copy()
