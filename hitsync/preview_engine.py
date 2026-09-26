"""Real-time preview playback, independent of the GUI toolkit.

A producer thread composes frames of the edit *ahead* of the playhead into
a small buffer (so a decoder restart at a cut is absorbed), and pre-starts a
second decoder at the next cut that jumps backwards or far ahead, so the
switch is instant. The audio device is the master clock: the GUI asks for
``current_frame()`` on a timer and gets the newest frame that is due; late
frames are dropped, so sound and picture never drift apart.

The same engine plays three things:
  * the montage (schedule + look + soundtrack),
  * the raw gameplay recording with its own sound (to mark "combat starts"),
  * a song on its own (to mark "music starts"; no frames).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

SR = 44100


# ------------------------------------------------------------------ clocks
class AudioClock:
    """Plays an (n, 2) float32 buffer from a position and tells the time.

    Uses ``sounddevice`` (PortAudio: WASAPI on Windows, CoreAudio on macOS)
    with a callback, so seeking is instant and the clock follows the samples
    actually handed to the device. Without an audio device it runs silently
    on the wall clock.
    """

    def __init__(self, sr: int = SR):
        self.sr = sr
        self._audio = None
        self._pos = 0              # next sample index
        self._t0 = 0.0             # buffer time of sample 0
        self._stream = None
        self._wall0 = None
        self._start_pos = 0.0
        self._lock = threading.Lock()
        self.volume = 1.0
        self.device_ok = True

    def start(self, audio: Optional[np.ndarray], pos: float, t0: float = 0.0):
        """Play ``audio`` (whose first sample is time ``t0``) from time ``pos``."""
        self.stop()
        self._audio = audio
        self._t0 = t0
        self._start_pos = pos
        self._pos = max(0, int(round((pos - t0) * self.sr)))
        self._wall0 = time.perf_counter()
        if audio is None or not len(audio):
            return
        try:
            import sounddevice as sd

            self._written = 0
            self._stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32",
                                           callback=self._callback, latency="low")
            self._latency = float(self._stream.latency or 0.0)
            self._stream.start()
        except Exception:                        # no device / no PortAudio
            self._stream = None
            self.device_ok = False

    def _callback(self, out, frames, _time, _status):
        with self._lock:
            a = self._audio
            i = self._pos
            chunk = a[i:i + frames] if a is not None else np.zeros((0, 2), np.float32)
            n = len(chunk)
            out[:n] = chunk * self.volume
            out[n:] = 0
            self._pos = i + frames
            self._written += frames

    def time(self) -> float:
        """Current playback time (buffer clock)."""
        if self._stream is not None:
            with self._lock:
                played = self._written / self.sr - self._latency
            return self._start_pos + max(0.0, played)
        if self._wall0 is None:
            return self._start_pos
        return self._start_pos + (time.perf_counter() - self._wall0)

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._wall0 = None


# ---------------------------------------------------------------- programs
@dataclass
class Program:
    """What to play: an edit (schedule), or raw footage, or a song alone."""

    kind: str                       # "montage" | "source" | "song"
    start: float
    end: float
    schedule: object = None         # montage (for "source": an identity schedule)
    rparams: object = None
    audio: Optional[np.ndarray] = None
    audio_t0: float = 0.0           # timeline time of audio[0]
    video_path: str = ""
    video_info: object = None
    frame_times: object = None
    pts_offset: float = 0.0


def identity_schedule(duration: float):
    """out time == source time: plays the recording as it is."""
    from .config import SyncParams
    from .sync_engine import Schedule, Segment

    s = Schedule(params=SyncParams(letterbox_enabled=False, letterbox_mode="off",
                                   intro_flash=False))
    s.segments = [Segment(np.array([0.0, duration]), np.array([0.0, duration]), "source")]
    s.start, s.duration = 0.0, duration
    return s


# ------------------------------------------------------------------ engine
class PreviewEngine:
    BUFFER_S = 0.8                  # composed ahead of the playhead
    PREWARM_S = 1.2                 # start the next cut's decoder this early
    MAX_W = 1280                    # frames are composed at most this wide; the
                                    # screen scales them (fullscreen stays real time)

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.size: Optional[tuple] = None
        self.program: Optional[Program] = None
        self.clock = AudioClock()
        self.playing = False
        self.position = 0.0
        self._buf: deque = deque()
        self._cv = threading.Condition()
        self._thread = None
        self._stop = threading.Event()
        self._src = None
        self._warm: dict = {}           # segment index -> (FrameSource, thread)
        self._last = None
        self.dropped = 0
        self.on_error: Callable[[Exception], None] = lambda e: None

    # ------------------------------------------------------------ control
    def load(self, program: Program, size: Optional[tuple] = None):
        was = self.playing
        self.pause()
        self.program = program
        if size:
            self.size = tuple(size)
        self.position = float(np.clip(self.position, program.start, program.end))
        self._close_sources()
        if was:
            self.play()

    def fit_size(self, w: int, h: int, aspect: float) -> tuple:
        """Frame size for a w x h view of a video with this aspect ratio."""
        w, h = max(16, w), max(16, h)
        if w / h > aspect:
            w = int(h * aspect)
        else:
            h = int(w / aspect)
        if w > self.MAX_W:
            w, h = self.MAX_W, int(self.MAX_W / aspect)
        return max(16, w // 2 * 2), max(16, h // 2 * 2)

    def set_size(self, size):
        size = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)
        if size != self.size:
            was = self.playing
            self.pause()
            self.size = size
            self._close_sources()
            if was:
                self.play()

    def play(self, t: Optional[float] = None):
        p = self.program
        if p is None:
            return
        self.pause()
        if t is not None:
            self.position = t
        if not p.start <= self.position < p.end - 0.05:
            self.position = p.start
        self._stop.clear()
        self._buf.clear()
        self.dropped = 0
        if p.kind != "song" and p.video_path:
            self._thread = threading.Thread(target=self._produce, args=(self.position,),
                                            daemon=True)
            self._thread.start()
            # let the first frame arrive before the clock starts (decoder spawn)
            deadline = time.perf_counter() + 1.5
            with self._cv:
                while not self._buf and time.perf_counter() < deadline and self._thread.is_alive():
                    self._cv.wait(0.02)
        self.clock.start(p.audio, self.position, p.audio_t0)
        self.playing = True

    def pause(self):
        if not self.playing and self._thread is None:
            return
        self.position = self.time()
        self.playing = False
        self.clock.stop()
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        self._buf.clear()

    def seek(self, t: float):
        was = self.playing
        self.pause()
        p = self.program
        self.position = float(np.clip(t, p.start, p.end)) if p else t
        if was:
            self.play()

    def toggle(self):
        self.pause() if self.playing else self.play()

    def time(self) -> float:
        return self.clock.time() if self.playing else self.position

    def close(self):
        self.pause()
        self._close_sources()

    # ------------------------------------------------------------ frames
    def current_frame(self):
        """(t, frame) due now (None if nothing new); ends playback at the end."""
        p = self.program
        if p is None:
            return None
        now = self.time()
        if self.playing and now >= p.end:
            self.pause()
            self.position = p.end
            return None
        if not self.playing:
            return None
        got = None
        with self._cv:
            while self._buf and self._buf[0][0] <= now + 0.5 / self.fps:
                got = self._buf.popleft()
                if self._buf and self._buf[0][0] <= now + 0.5 / self.fps:
                    self.dropped += 1
            self._cv.notify_all()
        return got

    def still(self, t: float):
        """One composed frame at t (while paused / scrubbing)."""
        p = self.program
        if p is None or p.kind == "song" or not p.video_path:
            return None
        from .renderer import compose_frame

        src = self._source_for(t)
        return compose_frame(src, p.schedule, t, p.rparams, self.size, self.fps, fast=True)

    # ---------------------------------------------------------- producer
    def _new_source(self):
        from .renderer import FrameSource

        p = self.program
        return FrameSource(p.video_path, p.video_info, frame_times=p.frame_times,
                           size=self.size, pts_offset=p.pts_offset)

    def _source_for(self, t: float):
        if self._src is None:
            self._src = self._new_source()
        return self._src

    def _close_sources(self):
        for src, th in list(self._warm.values()):
            th.join(timeout=3)
            src.close()
        self._warm.clear()
        if self._src is not None:
            self._src.close()
            self._src = None

    def _jumps(self, sched):
        """Segment indexes whose start needs a decoder restart (a cut that
        goes back in the footage or skips far ahead)."""
        from .renderer import FrameSource

        out = []
        segs = sched.segments
        for k in range(1, len(segs)):
            gap = segs[k].src_start - segs[k - 1].src_end
            if gap < -0.02 or gap > FrameSource.RESTART_SECONDS:
                out.append(k)
        return out

    def _prewarm(self, sched, k):
        """Start a decoder at segment k's first frame in the background."""
        if k in self._warm:
            return
        src = self._new_source()
        t = sched.segments[k].src_start

        def warm():
            try:
                src.get(src._index_of(max(0.0, t)))
            except Exception:
                pass

        th = threading.Thread(target=warm, daemon=True)
        th.start()
        self._warm[k] = (src, th)

    def _produce(self, t0: float):
        from .renderer import compose_frame

        p = self.program
        sched, rp = p.schedule, p.rparams
        jumps = self._jumps(sched)
        k = 0
        src = self._source_for(t0)
        seg_idx = sched._index(t0)
        try:
            while not self._stop.is_set():
                t = t0 + k / self.fps
                if t >= p.end:
                    break
                # don't fall behind the clock: skip to the playhead
                now = self.clock.time() if self.playing else t0
                if t < now - 1.0 / self.fps:
                    k = int(np.ceil((now - t0) * self.fps))
                    continue
                # keep the buffer ~BUFFER_S ahead
                with self._cv:
                    while (not self._stop.is_set() and self._buf
                           and self._buf[-1][0] - (self.clock.time() if self.playing else t0)
                           > self.BUFFER_S):
                        self._cv.wait(0.01)
                if self._stop.is_set():
                    break
                idx = sched._index(t)
                if idx != seg_idx:
                    if idx in self._warm:           # switch to the pre-started decoder
                        nsrc, th = self._warm.pop(idx)
                        th.join()
                        if self._src is not None:
                            self._src.close()
                        self._src = src = nsrc
                    seg_idx = idx
                for j in jumps:                      # pre-start upcoming jumps
                    start = sched.segments[j].out_start
                    if t < start <= t + self.PREWARM_S:
                        self._prewarm(sched, j)
                frame = compose_frame(src, sched, t, rp, self.size, self.fps, fast=True)
                with self._cv:
                    self._buf.append((t, frame))
                    self._cv.notify_all()
                k += 1
        except Exception as exc:                     # decoding problems: stop quietly
            self.on_error(exc)
        finally:
            for j in list(self._warm):
                if sched.segments[j].out_start <= t0:
                    s2, th = self._warm.pop(j)
                    th.join(timeout=3)
                    s2.close()


# ------------------------------------------------------------ builders
def montage_program(project, audio: Optional[np.ndarray]) -> Program:
    s = project.schedule
    v = project.video
    return Program("montage", s.start, s.end, s, project.render_params, audio, s.start,
                   project.video_path, v.info, v.frame_times, v.pts_offset or 0.0)


def source_program(project) -> Program:
    """The raw recording with its own sound (for 'combat starts here')."""
    from .audio_mix import _game_audio
    from .config import RenderParams
    import os

    v = project.video
    audio, t0 = None, 0.0
    try:
        game, offset = _game_audio(project.video_path, os.path.getmtime(project.video_path))
        if game is not None:
            audio, t0 = game, -offset          # source t -> audio index (t + offset) * sr
    except Exception:
        pass
    dur = project.video_duration
    return Program("source", 0.0, dur, identity_schedule(dur), RenderParams(), audio, t0,
                   project.video_path, v.info, v.frame_times, v.pts_offset or 0.0)


def song_program(path: str, duration: float) -> Program:
    from .audio_mix import load_music

    return Program("song", 0.0, duration, audio=load_music(path), audio_t0=0.0)
