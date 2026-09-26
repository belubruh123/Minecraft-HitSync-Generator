"""Frame-accurate renderer: OpenCV time-remapping piped into ffmpeg.

For every output frame we evaluate the schedule (source time, speed,
letterbox amount, flash), fetch/synthesise the source frame, composite, and
stream raw BGR into an ffmpeg encoder that also muxes the music track.
Source frames come from an ffmpeg decode pipe (see ``FrameSource``): reads
are forward-sequential (cuts only skip forward) and long jumps re-seek by
timestamp, which stays frame-accurate on variable-frame-rate recordings.
"""
from __future__ import annotations

import io
import math
import os
import re
import subprocess
import tempfile
import queue
import threading
from collections import OrderedDict

import cv2
import numpy as np

from .config import RenderParams
from .ffmpeg_utils import NO_WINDOW, require_ffmpeg, video_encoder_args
from .sync_engine import Schedule
from .video_analysis import VideoInfo


class RenderCancelled(Exception):
    pass


class FrameSource:
    """Frame-accurate source reader backed by an ffmpeg decode pipe.

    OpenCV's frame seeking returns the wrong frame on many real recordings
    (H.264 with variable frame rate, e.g. Windows Game Bar captures), which
    would put every cut out of sync. ffmpeg's own "accurate" seek isn't
    exact either: on those files it sometimes drops the first wanted frame,
    which made every frame after that seek one frame (33 ms) late. So the
    decoder starts a little *before* the target and every decoded frame is
    identified by its real timestamp (``showinfo``, ``-copyts``), looked up
    in the analysis frame list - frame numbers can't drift.
    """

    RESTART_SECONDS = 3.0   # forward jumps longer than this re-seek instead of decoding through

    def __init__(self, path: str, info: VideoInfo, cache_size: int = 6, frame_times=None,
                 size: tuple[int, int] | None = None, pts_offset: float = 0.0):
        self.path = path
        self.pts_offset = float(pts_offset or 0.0)
        self.fps = info.fps
        self.w, self.h = size or (info.width, info.height)
        self.w, self.h = int(self.w), int(self.h)
        self.native = (self.w, self.h) == (info.width, info.height)
        # Full-size frames travel as yuv420p (half the bytes of BGR through
        # the pipe) and are converted by OpenCV in the prefetch thread.
        self.yuv = self.native and self.w % 2 == 0 and self.h % 2 == 0
        if frame_times is not None and len(frame_times) >= 2:
            self.ft = np.asarray(frame_times, float)
        else:
            self.ft = None
        n = len(self.ft) if self.ft is not None else info.frame_count
        self.n = max(1, int(n))
        self.pos = -1                       # index of the last decoded frame
        self.proc = None
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.cache_size = cache_size
        self.last_good = None
        self._flow_key = None
        self._flows = None
        self._dis = None

    def _pts(self, idx: int) -> float:
        return float(self.ft[idx]) if self.ft is not None else idx / self.fps

    SEEK_LEAD = 0.5         # s decoded before a seek target (discarded)
    PREFETCH = 6            # decoded frames buffered ahead of the renderer
    _SHOWINFO = re.compile(rb" n:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*(-?[0-9.]+)")

    def _index_of(self, t: float) -> int:
        """Analysis frame index of a decoded frame's timestamp (source time)."""
        if self.ft is None:
            return int(round(t * self.fps))
        i = int(np.searchsorted(self.ft, t))
        if i > 0 and (i >= len(self.ft) or t - self.ft[i - 1] < self.ft[i] - t):
            i -= 1
        return i

    def _start(self, idx: int):
        self._stop()
        target = self._pts(idx) + self.pts_offset
        t = target - self.SEEK_LEAD
        cmd = [require_ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "info", "-nostdin"]
        if t > 0:
            cmd += ["-ss", f"{t:.6f}"]
        vf = ("" if self.native else f"scale={self.w}:{self.h}:flags=area,") + "showinfo=checksum=0"
        cmd += ["-copyts", "-i", self.path, "-map", "0:v:0", "-an", "-sn",
                "-fps_mode", "passthrough", "-vf", vf,
                "-pix_fmt", "yuv420p" if self.yuv else "bgr24", "-f", "rawvideo", "pipe:1"]
        # unbuffered: frames are read straight into their numpy arrays
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     stdin=subprocess.DEVNULL, bufsize=0, creationflags=NO_WINDOW)
        self.pos = idx - 2
        self._queue = queue.Queue(maxsize=self.PREFETCH)
        self._halt = threading.Event()
        stamps: queue.Queue = queue.Queue()

        def read_stamps(proc=self.proc):
            try:
                with io.BufferedReader(proc.stderr, 1 << 16) as err:   # (pipe is unbuffered)
                    for line in err:
                        m = self._SHOWINFO.search(line)
                        if m:
                            stamps.put(float(m.group(1)))
            except (OSError, ValueError):
                pass
            stamps.put(None)

        self._stamp_thread = threading.Thread(target=read_stamps, daemon=True)
        self._stamp_thread.start()
        self._thread = threading.Thread(target=self._prefetch,
                                        args=(self.proc, self._queue, self._halt, stamps,
                                              idx - 1), daemon=True)
        self._thread.start()

    def _prefetch(self, proc, q, halt, stamps, first):
        """Decode ahead in the background (ffmpeg + pipe reads release the GIL).

        Queues (frame index, BGR frame); lead-in frames before ``first`` are
        dropped here without colour conversion.
        """
        shape = (self.h * 3 // 2, self.w) if self.yuv else (self.h, self.w, 3)
        nbytes = int(np.prod(shape))
        while not halt.is_set():
            frame = np.empty(shape, np.uint8)
            view, got = memoryview(frame).cast("B"), 0
            try:
                while got < nbytes:
                    r = proc.stdout.readinto(view[got:])
                    if not r:
                        break
                    got += r
            except (OSError, ValueError):
                got = 0
            item = None
            if got == nbytes:
                try:
                    ts = stamps.get(timeout=10)   # showinfo logs before the frame is written
                except queue.Empty:
                    ts = None
                if ts is not None:
                    j = self._index_of(ts - self.pts_offset)
                    if j < first:
                        continue
                    item = (j, cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420) if self.yuv else frame)
            while not halt.is_set():
                try:
                    q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if item is None:
                return

    def _stop(self):
        if self.proc is not None:
            self._halt.set()
            self.proc.kill()
            self._thread.join(timeout=5)
            self._stamp_thread.join(timeout=5)
            for pipe in (self.proc.stdout, self.proc.stderr):
                try:
                    pipe.close()
                except Exception:
                    pass
            self.proc.wait()
            self.proc = None

    def close(self):
        self._stop()

    def _read_next(self):
        if self.proc is None:
            return None
        return self._queue.get()

    def get(self, idx: int) -> np.ndarray:
        idx = int(min(max(idx, 0), self.n - 1))
        if idx in self.cache:
            self.cache.move_to_end(idx)
            return self.cache[idx]
        if self.proc is None or idx <= self.pos or \
                self._pts(idx) - self._pts(max(self.pos, 0)) > self.RESTART_SECONDS:
            self._start(idx)
        frame = None
        while self.pos < idx:
            item = self._read_next()
            if item is None:
                self.n = max(1, min(self.n, self.pos + 1))   # stream ended early
                break
            self.pos, frame = item
            if self.pos >= idx - 1:       # keep the pair needed for blending
                self._remember(self.pos, frame)
        if idx in self.cache:
            frame = self.cache[idx]
        elif frame is None or self.pos < idx:
            frame = self.last_good if self.last_good is not None else \
                np.zeros((self.h, self.w, 3), np.uint8)
        # (a frame missing from the stream: the next one stands in)
        self.last_good = frame
        return frame

    def _remember(self, idx, frame):
        self.cache[idx] = frame
        self.cache.move_to_end(idx)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _frame_pos(self, t: float) -> float:
        """Fractional frame index shown at source time t."""
        if self.ft is None:
            return max(0.0, t * self.fps)
        ft = self.ft
        i = int(np.searchsorted(ft, t, side="right")) - 1
        if i < 0:
            return 0.0
        if i >= len(ft) - 1:
            return float(len(ft) - 1)
        return i + float((t - ft[i]) / max(1e-6, ft[i + 1] - ft[i]))

    def frame_at(self, t: float, mode: str = "nearest") -> np.ndarray:
        f = self._frame_pos(t)
        i0 = int(math.floor(f))
        a = f - i0
        if mode == "nearest" or a < 0.05 or a > 0.95 or i0 + 1 >= self.n:
            return self.get(int(round(f)))
        f0 = self.get(i0)
        f1 = self.get(i0 + 1)
        if mode == "flow":
            return self._flow_interp(i0, f0, f1, a)
        return cv2.addWeighted(f0, 1.0 - a, f1, a, 0.0)

    FLOW_WIDTH = 640        # optical flow is estimated at this width

    def _flow_interp(self, i0, f0, f1, t):
        """Optical-flow frame interpolation (Super-SloMo style flow approx).

        Flow is estimated and mixed at <= 640 px wide (motion fields are
        smooth) and only the final remap maps are upscaled to full size, so a
        1440p in-between frame costs two remaps instead of full-res math.
        """
        h, w = f0.shape[:2]
        if self._flow_key != i0:
            if self._dis is None:
                self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
            sw = min(w, self.FLOW_WIDTH)
            sh = max(16, int(round(h * sw / w)))
            g0 = cv2.cvtColor(cv2.resize(f0, (sw, sh), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2GRAY)
            g1 = cv2.cvtColor(cv2.resize(f1, (sw, sh), interpolation=cv2.INTER_AREA),
                              cv2.COLOR_BGR2GRAY)
            scale = np.array([w / sw, h / sh], np.float32)
            self._flows = (self._dis.calc(g0, g1, None) * scale,
                           self._dis.calc(g1, g0, None) * scale)
            self._flow_key = i0
        f01, f10 = self._flows
        sh, sw = f01.shape[:2]
        grid = self._grid(sw, sh, w, h)
        ft0 = -(1 - t) * t * f01 + t * t * f10
        ft1 = (1 - t) * (1 - t) * f01 - t * (1 - t) * f10
        m0 = cv2.resize(grid + ft0, (w, h), interpolation=cv2.INTER_LINEAR)
        m1 = cv2.resize(grid + ft1, (w, h), interpolation=cv2.INTER_LINEAR)
        w0 = cv2.remap(f0, m0, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w1 = cv2.remap(f1, m1, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.addWeighted(w0, 1.0 - t, w1, t, 0.0)

    def _grid(self, sw, sh, w, h):
        """Full-res pixel coordinates sampled on the flow grid (cached)."""
        key = (sw, sh, w, h)
        if getattr(self, "_grid_key", None) != key:
            xs = (np.arange(sw, dtype=np.float32) + 0.5) * (w / sw) - 0.5
            ys = (np.arange(sh, dtype=np.float32) + 0.5) * (h / sh) - 0.5
            gx, gy = np.meshgrid(xs, ys)
            self._grid_cache = np.dstack([gx, gy]).astype(np.float32)
            self._grid_key = key
        return self._grid_cache


def apply_letterbox(frame: np.ndarray, amount: float, aspect: float) -> np.ndarray:
    """Draw eased black bars; fractional edge row is blended for smoothness."""
    if amount <= 1e-3:
        return frame
    h, w = frame.shape[:2]
    full = max(0.0, (h - w / aspect) / 2.0)
    bar = full * min(1.0, amount)
    whole = int(bar)
    frac = bar - whole
    if whole:
        frame[:whole] = 0
        frame[h - whole:] = 0
    if frac > 0.01 and whole < h // 2:
        k = 1.0 - frac
        frame[whole] = (frame[whole] * k).astype(frame.dtype)
        frame[h - whole - 1] = (frame[h - whole - 1] * k).astype(frame.dtype)
    return frame


def apply_flash(frame: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 1e-3:
        return frame
    return cv2.addWeighted(frame, 1.0 - amount, np.full_like(frame, 255), amount, 0.0)


def compose_frame(source: FrameSource, schedule: Schedule, t: float, rparams: RenderParams,
                  size: tuple[int, int] | None = None) -> np.ndarray:
    """Build one output frame at output time ``t`` (also used for GUI preview)."""
    src_t = schedule.src_time(t)
    speed = schedule.speed_at(t)
    # optical flow only pays off in real slow motion; a mild slow-down (a
    # combo stretched onto the beat) blends neighbouring frames instead
    mode = rparams.interp if speed < 0.65 else "blend" if speed < 0.9 and \
        rparams.interp != "nearest" else "nearest"
    frame = source.frame_at(src_t, mode)
    resized = bool(size) and (frame.shape[1], frame.shape[0]) != size
    if resized:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    bars = schedule.letterbox_amount(t)
    if bars > 1e-3 and not resized:
        frame = frame.copy()        # decoded source frames are cached: never draw on them
    frame = apply_letterbox(frame, bars, schedule.params.letterbox_aspect)
    return apply_flash(frame, schedule.flash_amount(t))


def render(video_path: str, music_path: str, out_path: str, schedule: Schedule,
           info: VideoInfo, rparams: RenderParams | None = None, progress=None,
           cancel: threading.Event | None = None, frame_times=None,
           pts_offset: float = 0.0, workers: int | None = None) -> str:
    """Render the montage to ``out_path``.

    Everything that can overlap does: the soundtrack is mixed while the
    video renders, and the video is rendered as a few consecutive parts at
    once (each with its own decoder and encoder), which are then joined
    losslessly and muxed with the soundtrack.
    """
    rparams = rparams or RenderParams()
    report = progress or (lambda *_: None)
    if schedule.duration <= 0 or not schedule.segments:
        raise ValueError("Schedule is empty - nothing to render.")
    ffmpeg = require_ffmpeg()
    # VFR screen captures report odd nominal rates (e.g. 29.615); export at
    # the nearest standard rate instead.
    fps = rparams.output_fps or min((24.0, 25.0, 30.0, 50.0, 60.0),
                                    key=lambda r: abs(r - min(info.fps, 60.0)))
    width = int(round(info.width * rparams.scale / 2)) * 2
    height = int(round(info.height * rparams.scale / 2)) * 2
    n_frames = max(1, int(round(schedule.duration * fps)))
    duration = n_frames / fps
    enc_name, enc_args = video_encoder_args(rparams.encoder, rparams.crf, rparams.preset)
    if workers is None:
        workers = default_workers(n_frames / fps, width * height)
    workers = max(1, min(int(workers), n_frames // 30 or 1))
    bounds = [round(n_frames * i / workers) for i in range(workers + 1)]

    tmp = tempfile.mkdtemp(prefix="hitsync_render_")
    audio_path = os.path.join(tmp, "audio.wav")
    parts = [os.path.join(tmp, f"part{i}.mp4") for i in range(workers)]
    done = [0] * workers
    lock = threading.Lock()

    def tick(i, count):
        with lock:
            done[i] = count
            k = sum(done)
        report(f"Rendering frame {k}/{n_frames} ({enc_name}, {workers} at once)", k / n_frames)

    def soundtrack():
        # Music + hit sounds, mixed in numpy so preview playback and export match.
        from .audio_mix import build_soundtrack, write_wav

        write_wav(audio_path, build_soundtrack(music_path, schedule, rparams, duration,
                                               video_path=video_path))

    from concurrent.futures import ThreadPoolExecutor

    stop = threading.Event()               # a failed part stops the others
    halt = _AnyEvent(cancel, stop)
    try:
        report(f"Rendering with {enc_name}", 0.0)
        with ThreadPoolExecutor(workers + 1) as ex:
            futs = [ex.submit(soundtrack)]
            futs += [ex.submit(_render_part, video_path, info, schedule, rparams, fps,
                               (width, height), bounds[i], bounds[i + 1], parts[i], enc_args,
                               frame_times, pts_offset, (lambda c, i=i: tick(i, c)), halt)
                     for i in range(workers)]
            try:
                for fu in futs:
                    fu.result()
            except BaseException:
                stop.set()
                raise
        if cancel is not None and cancel.is_set():
            raise RenderCancelled()
        report("Joining parts and muxing audio", 0.99)
        _mux(ffmpeg, parts, [(bounds[i + 1] - bounds[i]) / fps for i in range(workers)],
             audio_path, duration, out_path, tmp)
    except BaseException:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)
    report("Render complete", 1.0)
    return out_path


def default_workers(seconds: float, pixels: int) -> int:
    """Parallel render parts: enough to keep decoder, CPU and encoder busy."""
    if seconds < 8:
        return 1
    cores = os.cpu_count() or 2
    return int(np.clip(cores // 5, 1, 3)) if pixels >= 1920 * 1080 else int(np.clip(cores // 4, 1, 4))


class _AnyEvent:
    """is_set() of any of several events (user cancel or a sibling failure)."""

    def __init__(self, *events):
        self.events = [e for e in events if e is not None]

    def is_set(self) -> bool:
        return any(e.is_set() for e in self.events)


def _render_part(video_path, info, schedule, rparams, fps, size, k0, k1, part_path, enc_args,
                 frame_times, pts_offset, tick, cancel):
    """Encode output frames [k0, k1) as a video-only file."""
    width, height = size
    cmd = [require_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}",
           "-r", f"{fps:.6f}", "-i", "pipe:0", "-an", *enc_args, "-pix_fmt", "yuv420p",
           part_path]
    err_file = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=err_file, creationflags=NO_WINDOW)
    source = FrameSource(video_path, info, frame_times=frame_times, size=size,
                         pts_offset=pts_offset)
    # A writer thread feeds the encoder while the next frames are composed
    # (pipe writes release the GIL), so decode, compose and encode overlap.
    frames: queue.Queue = queue.Queue(maxsize=4)
    broken = threading.Event()

    def writer():
        while True:
            f = frames.get()
            if f is None:
                return
            if broken.is_set():
                continue
            try:
                # BGR -> yuv420p here (fast, multi-threaded) halves the bytes
                # sent to the encoder and skips its single-threaded converter
                proc.stdin.write(memoryview(cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420)).cast("B"))
            except (BrokenPipeError, OSError, ValueError):
                broken.set()

    wt = threading.Thread(target=writer, daemon=True)
    wt.start()
    try:
        try:
            for k in range(k0, k1):
                if cancel is not None and cancel.is_set():
                    raise RenderCancelled()
                if broken.is_set():
                    break
                frames.put(compose_frame(source, schedule, schedule.start + k / fps, rparams, size))
                if (k - k0) % 10 == 0:
                    tick(k - k0)
            tick(k1 - k0)
        finally:
            frames.put(None)
            wt.join()
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0 or broken.is_set():
            err_file.seek(0)
            err = err_file.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg exited with {rc}: {err[-800:]}")
    except BaseException:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        proc.wait()
        raise
    finally:
        source.close()
        err_file.close()


def _mux(ffmpeg, parts, lengths, audio_path, duration, out_path, tmp):
    """Join the parts without re-encoding and add the soundtrack.

    Each part's exact length (frames / fps) is given to the concat demuxer:
    left to itself it offsets the next part by the container duration,
    which can be off by a fraction of a frame and shift everything after
    the join by one frame (hits 33 ms off the beat).
    """
    listing = os.path.join(tmp, "parts.txt")
    with open(listing, "w", encoding="utf-8") as f:
        for p, length in zip(parts, lengths):
            f.write("file '" + p.replace("\\", "/").replace("'", "'\\''") + "'\n")
            f.write(f"duration {length:.6f}\n")
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
           "-f", "concat", "-safe", "0", "-i", listing, "-i", audio_path,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{duration:.3f}", "-movflags", "+faststart", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace",
                       creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {r.stderr.strip()[-800:]}")
