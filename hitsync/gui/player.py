"""In-app playback of the edited montage (video + music + hit sounds).

Video frames are composed on the Tk thread against a wall clock (frames are
dropped when decoding falls behind, so audio never drifts). Audio is the same
soundtrack the exporter uses, played from the start position through
``winsound`` on Windows (stdlib, no extra dependency) or ``sounddevice`` when
installed elsewhere.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import numpy as np

from ..audio_mix import SR, build_soundtrack, write_wav


class AudioOut:
    def __init__(self):
        self._tmp = None
        self._sd = None
        if sys.platform != "win32":
            try:
                import sounddevice  # type: ignore

                self._sd = sounddevice
            except Exception:
                pass

    def play(self, audio: np.ndarray, start: float):
        self.stop()
        chunk = audio[int(start * SR):]
        if not len(chunk):
            return
        if sys.platform == "win32":
            import winsound

            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            write_wav(path, chunk)
            self._tmp = path
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC
                               | winsound.SND_NODEFAULT)
        elif self._sd is not None:
            self._sd.play(chunk, SR)

    def stop(self):
        if sys.platform == "win32":
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        elif self._sd is not None:
            self._sd.stop()
        if self._tmp:
            path, self._tmp = self._tmp, None
            # the file may still be locked for a moment after PlaySound stops
            threading.Timer(1.0, lambda: _remove(path)).start()


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


class Player:
    """Drives playback; the app supplies callbacks for frames and position."""

    def __init__(self, tk_root, get_project, show_frame, on_position, on_state):
        self.root = tk_root
        self.get_project = get_project
        self.show_frame = show_frame      # (bgr_frame, t) -> None
        self.on_position = on_position    # (t) -> None
        self.on_state = on_state          # (playing: bool) -> None
        self.audio = AudioOut()
        self.position = 0.0
        self.playing = False
        self._source = None
        self._job = None
        self._t0 = 0.0
        self._clock0 = 0.0
        self._soundtrack = None
        self._soundtrack_key = None
        self.preview_size = None          # (w, h) to compose frames at

    # ---------------------------------------------------------------- api
    def toggle(self):
        self.pause() if self.playing else self.play()

    def play(self):
        from ..renderer import FrameSource

        p = self.get_project()
        sched = p.schedule
        if sched is None or not sched.segments or p.video is None:
            return
        if not sched.start <= self.position < sched.end - 0.05:
            self.position = sched.start
        try:
            audio = self._get_soundtrack(p)
        except Exception as exc:  # audio problems must not block video preview
            print(f"[player] audio unavailable: {exc}")
            audio = None
        self._source = FrameSource(p.video_path, p.video.info, frame_times=p.video.frame_times,
                                   size=self.preview_size,
                                   pts_offset=p.video.pts_offset or 0.0)
        self._t0 = self.position
        # Decode the first frame before starting the clock: spawning ffmpeg and
        # seeking takes ~1 s on long 1440p captures, which would otherwise be
        # skipped over.
        self._source.frame_at(sched.src_time(self._t0), "nearest")
        if audio is not None:
            self.audio.play(audio, self._t0 - sched.start)   # soundtrack begins at start
        self._clock0 = time.perf_counter()
        self.playing = True
        self.on_state(True)
        self._tick()

    def pause(self):
        if not self.playing:
            return
        self.playing = False
        if self._job:
            self.root.after_cancel(self._job)
            self._job = None
        self.audio.stop()
        if self._source:
            self._source.close()
            self._source = None
        self.on_state(False)

    def seek(self, t: float):
        was = self.playing
        self.pause()
        self.position = max(0.0, t)
        self.on_position(self.position)
        if was:
            self.play()

    def invalidate(self):
        """Schedule or sound settings changed: stop and rebuild audio next time."""
        self._soundtrack_key = None
        self.pause()

    # ------------------------------------------------------------ internals
    def _get_soundtrack(self, p):
        r = p.render_params
        key = (id(p.schedule), p.music_path, r.hit_sound, r.hit_sound_file, r.hit_volume,
               r.music_volume, r.hit_pitch_variation)
        if key != self._soundtrack_key:
            self._soundtrack = build_soundtrack(p.music_path, p.schedule, r,
                                                video_path=p.video_path)
            self._soundtrack_key = key
        return self._soundtrack

    def _tick(self):
        if not self.playing:
            return
        from ..renderer import compose_frame

        p = self.get_project()
        sched = p.schedule
        t = self._t0 + (time.perf_counter() - self._clock0)
        if sched is None or t >= sched.end:
            self.position = 0.0 if sched is None else sched.end
            self.pause()
            self.on_position(self.position)
            return
        self.position = t
        rp = p.render_params
        # optical flow is too slow for real time; blending looks close enough
        interp = "blend" if rp.interp == "flow" else rp.interp
        preview = type(rp)(**{**rp.to_dict(), "interp": interp})
        frame = compose_frame(self._source, sched, t, preview, self.preview_size)
        self.show_frame(frame, t)
        self.on_position(t)
        fps = min(p.video.info.fps, 30.0)
        spent = (time.perf_counter() - self._clock0) - (t - self._t0)
        delay = max(1, int((1.0 / fps - max(0.0, spent)) * 1000))
        self._job = self.root.after(delay, self._tick)
