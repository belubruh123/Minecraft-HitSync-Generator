"""Hit-sound mixing and variable-frame-rate frame mapping."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync import audio_mix  # noqa: E402
from hitsync.config import RenderParams  # noqa: E402
from hitsync.ffmpeg_utils import find_ffmpeg  # noqa: E402
from hitsync.renderer import FrameSource  # noqa: E402
from hitsync.video_analysis import VideoInfo  # noqa: E402
from tests import synth  # noqa: E402
from tests.test_engine import make  # noqa: E402
from hitsync.sync_engine import build_schedule  # noqa: E402


@unittest.skipUnless(find_ffmpeg(), "ffmpeg unavailable")
class HitSoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.music = os.path.join(tempfile.mkdtemp(), "song.wav")
        synth.write_song(cls.music, duration=40)
        m, p = make()
        cls.sched = build_schedule(m, p, 40, 40)

    def mix(self, **kw):
        r = RenderParams(**kw)
        return audio_mix.build_soundtrack(self.music, self.sched, r)

    def test_hit_sounds_land_on_hit_times(self):
        # "custom" with a synthesized file keeps the test independent of any
        # local Minecraft install
        path = os.path.join(tempfile.mkdtemp(), "punch.wav")
        audio_mix.write_wav(path, audio_mix.synth_punch())
        on = self.mix(hit_sound="custom", hit_sound_file=path, hit_pitch_variation=False)
        off = self.mix(hit_sound="off")
        diff = np.abs(on - off).max(axis=1)
        sr = audio_mix.SR
        hits = [pl.out_t for pl in self.sched.placements if pl.out_t is not None]
        self.assertTrue(hits)
        for t in hits:
            i = int(t * sr)
            self.assertGreater(diff[i:i + int(0.05 * sr)].max(), 0.1, t)
            self.assertLess(diff[max(0, i - int(0.05 * sr)):i - 5].max(), 1e-3, t)
        # nothing added far from any hit
        mask = np.ones(len(diff), bool)
        for t in hits:
            mask[int(t * sr) - 10: int((t + 0.3) * sr)] = False
        self.assertLess(diff[mask].max(), 1e-3)

    def test_soundtrack_matches_edit_length_and_fades(self):
        out = self.mix(hit_sound="off")
        self.assertEqual(len(out), int(round(self.sched.duration * audio_mix.SR)))
        self.assertLess(np.abs(out[-50:]).max(), 0.01)
        self.assertLessEqual(np.abs(out).max(), 1.0)

    def test_minecraft_sounds_when_installed(self):
        files = audio_mix.minecraft_sound_files("classic")
        if not files:
            self.skipTest("no local Minecraft install")
        samples, desc = audio_mix.hit_samples("classic")
        self.assertIn("Minecraft", desc)
        self.assertTrue(all(len(s) > 1000 for s in samples))


class VfrMappingTests(unittest.TestCase):
    def test_timestamps_drive_frame_lookup(self):
        # 30 fps nominal, but frames 10.. arrive 200 ms late (VFR drift)
        ft = np.arange(40) / 30.0
        ft[10:] += 0.2
        src = FrameSource.__new__(FrameSource)
        src.ft, src.fps, src.n = ft, 30.0, 40
        self.assertEqual(src._frame_pos(ft[5]), 5)
        self.assertEqual(src._frame_pos(ft[20]), 20)       # index/fps would say 26
        self.assertAlmostEqual(src._frame_pos((ft[20] + ft[21]) / 2), 20.5)
        self.assertEqual(src._frame_pos(ft[9] + 0.1), 9 + 0.1 / (ft[10] - ft[9]))
        self.assertEqual(src._frame_pos(99.0), 39)
        self.assertEqual(src._frame_pos(-1.0), 0.0)

    def test_constant_fps_fallback(self):
        src = FrameSource.__new__(FrameSource)
        src.ft, src.fps, src.n = None, 30.0, 40
        self.assertAlmostEqual(src._frame_pos(1.0), 30.0)


if __name__ == "__main__":
    unittest.main()
