"""End-to-end: synthetic media -> analysis -> alignment -> rendered mp4."""
import os
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync.ffmpeg_utils import find_ffmpeg  # noqa: E402
from hitsync.project import Project  # noqa: E402
from tests import synth  # noqa: E402


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hitsync_test_")
        cls.video = os.path.join(cls.tmp, "pvp.mp4")
        cls.music = os.path.join(cls.tmp, "song.wav")
        synth.write_video(cls.video)
        synth.write_song(cls.music)
        cls.p = Project(video_path=cls.video, music_path=cls.music,
                        output_path=os.path.join(cls.tmp, "out.mp4"))
        cls.p.sync.min_combo_len = 3          # synthetic combos are 5/4/3 hits
        cls.p.analyze()
        # the synthetic song changes tempo on purpose: use the tracked beats
        cls.p.sync.static_grid = False
        cls.p.apply_beat_grid()

    def test_beats_and_tempo(self):
        beats = np.array(self.p.audio.beats)
        post = beats[(beats > synth.DROP + 1) & (beats < synth.TEMPO_CHANGE - 1)]
        self.assertGreater(len(post), 10)
        self.assertAlmostEqual(float(np.median(np.diff(post))), 0.5, delta=0.03)
        late = beats[beats > synth.TEMPO_CHANGE + 1.5]
        self.assertAlmostEqual(float(np.median(np.diff(late))), 0.4, delta=0.03)
        # beats sit on the kick attack, not trailing it
        true = synth.song_beats()
        err = np.array([b - true[np.argmin(np.abs(true - b))] for b in post])
        self.assertLess(abs(float(np.mean(err))), 0.010, err)
        self.assertLess(float(np.max(np.abs(err))), 0.025, err)

    def test_drop_detected(self):
        self.assertAlmostEqual(self.p.sync.drop_time, synth.DROP, delta=0.3)

    def test_hits_detected(self):
        got = np.array([h.t for h in self.p.markers.hits])
        for t in synth.HITS:
            self.assertLess(np.min(np.abs(got - t)), 0.05, f"missed hit at {t}: {got}")
        self.assertEqual(len(got), len(synth.HITS), got)
        self.assertEqual(len(self.p.markers.combos()), 3)

    def test_auto_intro(self):
        # every-beat mode: the intro ramps straight into the first combo hit
        self.assertAlmostEqual(self.p.sync.intro_end, synth.HITS[0], delta=0.04)
        self.assertLess(self.p.sync.intro_start, self.p.sync.intro_end)

    def test_render(self):
        if not find_ffmpeg():
            self.skipTest("ffmpeg unavailable")
        sched = self.p.recalculate()
        locked = [pl for pl in sched.placements if pl.locked]
        self.assertGreaterEqual(len(locked), len(synth.HITS) - 1, sched.summary())
        self.p.render_params.interp = "flow"
        out = self.p.render()
        cap = cv2.VideoCapture(out)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.assertAlmostEqual(n / fps, sched.duration, delta=0.2)
        # letterbox bars during the slow-mo intro only (spans are in song
        # time; the video starts at sched.start)
        self.assertEqual(sched.letterbox_snap, [True])
        start, end = sched.letterbox[0]
        cap.set(cv2.CAP_PROP_POS_MSEC, ((start + end) / 2 + 0.2 - sched.start) * 1000)
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertLess(frame[:5].mean(), 8)
        self.assertLess(frame[-5:].mean(), 8)
        # the first combo plays full frame
        a, b, _ = sched.combo_spans[0]
        cap.set(cv2.CAP_PROP_POS_MSEC, ((a + b) / 2 - sched.start) * 1000)
        ok, frame = cap.read()
        self.assertTrue(ok)
        self.assertGreater(frame[:5].mean(), 30)
        cap.release()


if __name__ == "__main__":
    unittest.main()
