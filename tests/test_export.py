"""Export path: frame-exact seeking on VFR video, parallel parts, encoders."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync.config import DetectParams  # noqa: E402
from hitsync.ffmpeg_utils import find_ffmpeg, video_encoder_args  # noqa: E402
from hitsync.project import Project  # noqa: E402
from hitsync.renderer import FrameSource, render  # noqa: E402
from hitsync.video_analysis import analyze_video, read_frame  # noqa: E402
from tests import synth  # noqa: E402

W, H, BITS = 320, 96, 10


def numbered_frame(k: int) -> np.ndarray:
    """Frame whose index is written as 10 big black/white blocks."""
    f = np.full((H, W, 3), 128, np.uint8)
    for b in range(BITS):
        f[16:80, b * 32 + 4: b * 32 + 28] = 255 if (k >> b) & 1 else 0
    return f


def read_number(frame: np.ndarray) -> int:
    g = frame.mean(axis=2)
    return sum(1 << b for b in range(BITS) if g[30:66, b * 32 + 8: b * 32 + 24].mean() > 128)


def write_vfr_video(path: str, n: int = 601):
    """H.264 with Game-Bar-like timing: 60 fps with a doubled gap every 7th
    frame, B-frames, keyframes every 30 frames, stream starting at 0.0667 s.
    (ffmpeg drops the very last frame of such a file, so one extra is written.)"""
    cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "60", "-i", "-",
           "-vf", "setpts=(N+floor(N/7))/60/TB+0.0667/TB", "-fps_mode", "passthrough",
           "-c:v", "libx264", "-preset", "veryfast", "-g", "30", "-bf", "2",
           "-pix_fmt", "yuv420p", path]
    data = b"".join(numbered_frame(k).tobytes() for k in range(n))
    subprocess.run(cmd, input=data, check=True)


@unittest.skipUnless(find_ffmpeg(), "ffmpeg required")
class FrameExactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hitsync_export_")
        cls.video = os.path.join(cls.tmp, "numbered.mp4")
        write_vfr_video(cls.video)
        cls.va = analyze_video(cls.video, DetectParams())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def source(self):
        return FrameSource(self.video, self.va.info, frame_times=self.va.frame_times,
                           size=(W, H), pts_offset=self.va.pts_offset)

    def test_scan_sees_every_frame_with_its_timestamp(self):
        self.assertGreaterEqual(len(self.va.frame_times), 600)
        self.assertAlmostEqual(self.va.pts_offset, 0.0667, delta=0.002)   # (timebase rounding)
        expect = (np.arange(600) + np.arange(600) // 7) / 60
        np.testing.assert_allclose(self.va.frame_times[:600], expect, atol=2e-3)

    def test_sequential_reads_are_the_right_frames(self):
        src = self.source()
        try:
            got = [read_number(src.get(i)) for i in range(600)]
        finally:
            src.close()
        self.assertEqual(got, list(range(600)))

    def test_every_seek_lands_on_the_exact_frame(self):
        """ffmpeg's own accurate seek sometimes drops the wanted frame on VFR
        files; every seek position must still return exactly that frame and
        keep counting correctly afterwards."""
        rng = np.random.default_rng(3)
        for idx in sorted(set(rng.integers(1, 590, 60).tolist()) | {1, 29, 30, 31, 7, 8}):
            src = self.source()
            try:
                got = [read_number(src.get(i)) for i in range(idx, idx + 4)]
            finally:
                src.close()
            self.assertEqual(got, list(range(idx, idx + 4)), f"seek to {idx}")

    def test_jumps_forward_and_back(self):
        src = self.source()
        try:
            for idx in (500, 20, 21, 400, 10, 590, 0):          # restarts + short skips
                self.assertEqual(read_number(src.get(idx)), idx)
        finally:
            src.close()

    def test_preview_frame_is_the_one_on_screen(self):
        ft, off = self.va.frame_times, self.va.pts_offset
        for idx in (0, 7, 8, 250, 599):
            nxt = ft[idx + 1]
            for t in (ft[idx], (ft[idx] + nxt) / 2):
                f = read_frame(self.video, t, (W, H), off)
                self.assertEqual(read_number(f), idx, f"t={t:.4f}")


@unittest.skipUnless(find_ffmpeg(), "ffmpeg required")
class ParallelExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hitsync_par_")
        cls.video = os.path.join(cls.tmp, "pvp.mp4")
        cls.music = os.path.join(cls.tmp, "song.wav")
        synth.write_video(cls.video)
        synth.write_song(cls.music)
        p = Project(video_path=cls.video, music_path=cls.music)
        p.sync.min_combo_len = 3
        p.analyze()
        p.render_params.encoder = "x264"
        cls.p, cls.sched = p, p.recalculate()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, workers):
        out = os.path.join(self.tmp, f"w{workers}.mp4")
        p = self.p
        render(p.video_path, p.music_path, out, self.sched, p.video.info, p.render_params,
               None, None, p.video.frame_times, p.video.pts_offset, workers=workers)
        raw = subprocess.run([find_ffmpeg(), "-loglevel", "error", "-i", out, "-vf", "scale=160:90",
                              "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True).stdout
        return out, np.frombuffer(raw, np.uint8).reshape(-1, 90, 160).astype(int)

    def test_parts_join_without_dropped_or_shifted_frames(self):
        _, one = self._render(1)
        out, three = self._render(3)
        self.assertEqual(len(one), len(three))
        self.assertEqual(len(one), int(round(self.sched.duration * 30)))
        diff = np.abs(one - three).mean(axis=(1, 2))
        self.assertLess(diff.max(), 3.0, int(diff.argmax()))      # encoder noise only
        info = subprocess.run([find_ffmpeg(), "-i", out], capture_output=True, text=True,
                              errors="replace").stderr
        self.assertIn("Audio: aac", info)

    def test_encoder_choice(self):
        self.assertEqual(video_encoder_args("x264", 18, "fast")[0], "libx264")
        name, args = video_encoder_args("auto", 18, "fast")
        self.assertIn(name, ("libx264", "h264_nvenc", "h264_amf", "h264_qsv"))
        self.assertEqual(args[:2], ["-c:v", name])


if __name__ == "__main__":
    unittest.main()
