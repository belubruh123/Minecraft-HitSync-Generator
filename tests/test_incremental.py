"""Fast analysis paths: ffmpeg scanner, parallel HPSS, cache, partial re-analysis."""
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync import project as project_mod  # noqa: E402
from hitsync.config import DetectParams  # noqa: E402
from hitsync.ffmpeg_utils import find_ffmpeg  # noqa: E402
from hitsync.project import Project  # noqa: E402
from hitsync.video_analysis import _analyze_video_cv, analyze_video  # noqa: E402
from tests import synth  # noqa: E402


@unittest.skipUnless(find_ffmpeg(), "ffmpeg required")
class FastAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="hitsync_inc_")
        cls.video = os.path.join(cls.tmp, "pvp.mp4")
        cls.video2 = os.path.join(cls.tmp, "pvp2.mp4")
        cls.music = os.path.join(cls.tmp, "song.wav")
        cls.music2 = os.path.join(cls.tmp, "song2.wav")
        synth.write_video(cls.video)
        synth.write_song(cls.music)
        shutil.copy(cls.video, cls.video2)
        shutil.copy(cls.music, cls.music2)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _project(self):
        p = Project(video_path=self.video, music_path=self.music)
        p.sync.min_combo_len = 3
        return p

    # ----------------------------------------------------------- scanner
    def test_ffmpeg_scanner_matches_opencv(self):
        fast = analyze_video(self.video, DetectParams())
        slow = _analyze_video_cv(self.video, DetectParams())
        self.assertEqual(len(fast.frame_times), len(slow.frame_times))
        np.testing.assert_allclose(fast.frame_times, slow.frame_times, atol=1e-4)
        self.assertGreater(np.corrcoef(fast.red, slow.red)[0, 1], 0.97)
        self.assertAlmostEqual(fast.pts_offset, slow.pts_offset, places=4)

    def test_parallel_chunks_are_seamless(self):
        one = analyze_video(self.video, DetectParams(), chunks=1)
        three = analyze_video(self.video, DetectParams(), chunks=3)
        np.testing.assert_allclose(three.frame_times, one.frame_times)
        np.testing.assert_allclose(three.red, one.red)
        np.testing.assert_allclose(three.motion, one.motion, atol=1e-6)

    # ----------------------------------------------------------- incremental
    def test_changing_music_only_reanalyses_music(self):
        p = self._project()
        self.assertEqual(p.analyze(), (True, True))
        p.markers.add_hit(12.34, p.sync.combo_gap)          # user edit on the video side
        video = p.video
        p.music_path = self.music2
        with mock.patch.object(project_mod, "analyze_video",
                               side_effect=AssertionError("video re-scanned")):
            self.assertEqual(p.analyze(), (True, False))
        self.assertIs(p.video, video)
        self.assertTrue(any(abs(h.t - 12.34) < 1e-6 for h in p.markers.hits))
        self.assertGreater(len(p.markers.beats), 10)

    def test_changing_video_only_reanalyses_video(self):
        p = self._project()
        p.analyze()
        p.sync.drop_time = p.markers.beats[5]                # user moved the drop
        drop, beats, audio = p.sync.drop_time, list(p.markers.beats), p.audio
        p.video_path = self.video2
        with mock.patch.object(project_mod, "analyze_audio",
                               side_effect=AssertionError("music re-analysed")):
            self.assertEqual(p.analyze(), (False, True))
        self.assertIs(p.audio, audio)
        self.assertEqual(p.sync.drop_time, drop)
        self.assertEqual(p.markers.beats, beats)
        self.assertGreater(len(p.markers.hits), 5)

    def test_unchanged_files_load_from_disk_cache(self):
        self._project().analyze()
        p = self._project()                                   # fresh project, same files
        with mock.patch.object(project_mod, "analyze_audio", side_effect=AssertionError), \
                mock.patch.object(project_mod, "analyze_video", side_effect=AssertionError):
            p.analyze()
        self.assertGreater(len(p.markers.hits), 5)
        self.assertGreater(len(p.markers.beats), 10)

    def test_edited_file_is_not_served_from_cache(self):
        p = self._project()
        p.analyze()
        st = os.stat(self.music)
        os.utime(self.music, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000_000))
        self.assertEqual(p.stale(), (True, False))

    def test_project_file_remembers_what_was_analysed(self):
        p = self._project()
        p.analyze()
        path = os.path.join(self.tmp, "proj.json")
        p.save(path)
        q = Project.load(path)
        self.assertEqual(q.stale(), (False, False))
        q.music_path = self.music2
        self.assertEqual(q.stale(), (True, False))


if __name__ == "__main__":
    unittest.main()
