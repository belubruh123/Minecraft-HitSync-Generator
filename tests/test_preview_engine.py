"""Real-time preview: frames come out in order, on the clock, and the
decoder for a cut that jumps back in the footage is started ahead."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync.ffmpeg_utils import find_ffmpeg  # noqa: E402


@unittest.skipUnless(find_ffmpeg(), "ffmpeg unavailable")
class PreviewEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hitsync.project import Project
        from tests import synth

        d = tempfile.mkdtemp()
        os.environ["HITSYNC_CACHE"] = os.path.join(d, "cache")
        video, music = os.path.join(d, "v.mp4"), os.path.join(d, "s.wav")
        synth.write_video(video)
        synth.write_song(music)
        p = Project(video_path=video, music_path=music)
        p.sync.min_combo_len = 3
        p.analyze()
        # play the last combo first: a backwards jump in the footage
        entries = p.plan_entries()
        p.set_plan([c for c, _ in entries][::-1])
        p.recalculate()
        cls.p = p

    def engine(self):
        from hitsync.preview_engine import PreviewEngine, montage_program

        e = PreviewEngine(fps=30)
        e.clock.start = self._silent(e.clock.start)
        e.load(montage_program(self.p, None), size=(160, 90))
        return e

    @staticmethod
    def _silent(start):
        def run(audio, pos, t0=0.0):
            return start(None, pos, t0)      # wall clock (CI has no sound card)
        return run

    def test_frames_in_order_and_on_time(self):
        e = self.engine()
        s = self.p.schedule
        e.play(s.start)
        got = []
        t_end = time.perf_counter() + 2.5
        while time.perf_counter() < t_end:
            f = e.current_frame()
            if f is not None:
                got.append((f[0], e.time()))
                self.assertEqual(f[1].shape, (90, 160, 3))
            time.sleep(0.005)
        e.close()
        ts = np.array([g[0] for g in got])
        self.assertGreater(len(ts), 40)                       # ~30 fps
        self.assertTrue(np.all(np.diff(ts) > 0))
        lag = np.array([c - t for t, c in got])
        self.assertLess(np.median(np.abs(lag)), 0.05)          # shown when due

    def test_backward_cut_is_prestarted(self):
        from hitsync.preview_engine import PreviewEngine

        e = self.engine()
        s = self.p.schedule
        jumps = e._jumps(s)
        self.assertTrue(jumps)
        j = jumps[0]
        t_cut = s.segments[j].out_start
        started = []
        orig = PreviewEngine._prewarm

        def spy(self_, sched, k):
            started.append((k, self_.time()))
            return orig(self_, sched, k)

        e._prewarm = spy.__get__(e)
        e.play(max(s.start, t_cut - 1.5))
        t_stop = time.perf_counter() + 2.5
        frames = []
        while time.perf_counter() < t_stop:
            f = e.current_frame()
            if f is not None:
                frames.append(f)
            time.sleep(0.005)
        e.close()
        self.assertTrue(any(k == j for k, _ in started), started)
        k, when = next(x for x in started if x[0] == j)
        self.assertLess(when, t_cut)                            # before the cut
        after = [t for t, _ in frames if t >= t_cut]
        self.assertTrue(after)

    def test_still_frame_matches_export_composition(self):
        from hitsync.renderer import FrameSource, compose_frame

        e = self.engine()
        s = self.p.schedule
        t = s.placed_hits()[2].out_t + 0.1
        a = e.still(t)
        v = self.p.video
        src = FrameSource(self.p.video_path, v.info, frame_times=v.frame_times, size=(160, 90),
                          pts_offset=v.pts_offset or 0.0)
        b = compose_frame(src, s, t, self.p.render_params, (160, 90))
        src.close()
        e.close()
        np.testing.assert_array_equal(a, b)

    def test_source_and_song_programs(self):
        from hitsync.preview_engine import PreviewEngine, song_program, source_program

        prog = source_program(self.p)
        self.assertEqual(prog.kind, "source")
        self.assertAlmostEqual(prog.end, self.p.video_duration)
        e = PreviewEngine()
        e.load(prog, (160, 90))
        f = e.still(5.0)
        self.assertEqual(f.shape, (90, 160, 3))
        e.close()
        sp = song_program(self.p.music_path, self.p.music_duration)
        self.assertEqual(sp.kind, "song")
        self.assertIsNotNone(sp.audio)


if __name__ == "__main__":
    unittest.main()
