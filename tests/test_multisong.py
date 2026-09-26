"""Several songs: the next song fades in, lands its start point on the bar
where the previous one hands over (no gap), and the beat grid continues."""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync.soundtrack import MusicTimeline, Song  # noqa: E402

SR = 8000


def grid(first, period, dur):
    b = first + period * np.arange(int((dur - first) / period) + 1)
    return b, b[::4]


class TimelineTests(unittest.TestCase):
    def make(self):
        b1, d1 = grid(0.2, 0.5, 60.0)              # 120 BPM
        b2, d2 = grid(0.1, 0.6, 50.0)              # 100 BPM
        s1 = Song("a", 60.0, 0.0, float(d1[25]), b1, d1)       # hands over at bar 25
        s2 = Song("b", 50.0, float(d2[4]), 45.0, b2, d2)        # starts at its bar 4
        return MusicTimeline([s1, s2]), s1, s2

    def test_start_point_lands_on_the_handover(self):
        tl, s1, s2 = self.make()
        T = tl.handovers[0]
        self.assertAlmostEqual(T, s1.end)
        self.assertEqual(tl.song_at(T - 0.01)[0], 0)
        k, t = tl.song_at(T + 1e-6)
        self.assertEqual(k, 1)
        self.assertAlmostEqual(t, s2.start, places=4)
        self.assertAlmostEqual(tl.duration, T + (s2.duration - s2.start))

    def test_beats_continue_without_a_gap(self):
        tl, s1, s2 = self.make()
        beats = tl.beats()
        T = tl.handovers[0]
        self.assertIn(round(T, 6), np.round(beats, 6))
        d = np.diff(beats)
        before, after = d[beats[1:] <= T + 1e-9], d[beats[:-1] >= T - 1e-9]
        self.assertTrue(np.allclose(before, 0.5))
        self.assertTrue(np.allclose(after, 0.6))
        self.assertTrue(np.all(np.isin(np.round(tl.downbeats(), 6), np.round(beats, 6))))

    def test_audio_crossfades_with_no_silence(self):
        tl, s1, s2 = self.make()
        tone = {"a": np.full((60 * SR, 2), 0.5, np.float32),
                "b": np.full((50 * SR, 2), -0.5, np.float32)}
        out = tl.render(0.0, int(tl.duration * SR), SR, tone.__getitem__)
        T = tl.handovers[0]
        xf = tl.pieces[1].fade_in
        self.assertGreater(xf, 0.9)
        lvl = np.abs(out[:, 0])
        # song 1 alone before, song 2 alone after, both in the crossfade
        self.assertAlmostEqual(out[int((T - xf - 1) * SR), 0], 0.5, places=4)
        self.assertAlmostEqual(out[int((T + 1) * SR), 0], -0.5, places=4)
        # the hand-over itself isn't silent (sin^2 fades: level 0 only exactly
        # where the two cross, and just for a moment)
        window = lvl[int((T - xf) * SR): int(T * SR)]
        self.assertLess(np.mean(window < 0.05), 0.1)

    def test_single_song_is_plain(self):
        b, d = grid(0.2, 0.5, 30.0)
        tl = MusicTimeline([Song("a", 30.0, 0.0, 25.0, b, d)])
        self.assertEqual(tl.duration, 30.0)
        self.assertTrue(np.allclose(tl.beats(), b))

    def test_cut_songs(self):
        b, d = grid(0.2, 0.5, 30.0)
        # a single song cut at 20.2 s ends there, with a tiny de-click fade
        tl = MusicTimeline([Song("a", 30.0, 0.0, 20.2, b, d, cut=True)])
        self.assertAlmostEqual(tl.duration, 20.2)
        self.assertTrue(0 < tl.pieces[0].fade_out <= 0.05)
        self.assertLess(tl.beats().max(), 20.2 + 1e-6)
        # the last of two songs, cut: the music ends at its cut
        tl2, s1, s2 = self.make()
        s2.end, s2.cut = float(s2.downbeats[20]), True
        tl2 = MusicTimeline([s1, s2])
        T = tl2.handovers[0]
        self.assertAlmostEqual(tl2.duration, T + s2.end - s2.start)


class ProjectTests(unittest.TestCase):
    def fresh(self):
        """A new project on the same files (analyses come from the cache)."""
        from hitsync.project import Project

        p = Project(video_path=self.video, music_path=self.songs[0][0])
        p.sync.min_combo_len = 3
        p.add_song(self.songs[1][0])
        p.analyze()
        return p

    @classmethod
    def setUpClass(cls):
        from hitsync.ffmpeg_utils import find_ffmpeg

        if not find_ffmpeg():
            raise unittest.SkipTest("ffmpeg unavailable")
        from hitsync.project import Project
        from tests import synth, synth_music as sm

        d = tempfile.mkdtemp()
        os.environ["HITSYNC_CACHE"] = os.path.join(d, "cache")
        cls.video = os.path.join(d, "v.mp4")
        synth.write_video(cls.video, duration=26)
        cls.songs = []
        for name, (y, info) in (("one", sm.backbeat_like(duration=40.0, bpm=110)),
                                ("two", sm.edm_like(duration=60.0))):
            path = os.path.join(d, name + ".wav")
            sm.write_wav(path, y)
            cls.songs.append((path, info))
        p = Project(video_path=cls.video, music_path=cls.songs[0][0])
        p.sync.min_combo_len = 3
        p.add_song(cls.songs[1][0])
        p.analyze()
        cls.p = p

    def test_second_song_takes_over_at_its_drop(self):
        p = self.p
        tl = p.timeline()
        self.assertIsNotNone(tl)
        T = tl.handovers[0]
        a1 = p.audio
        self.assertLessEqual(T, a1.duration)
        # song 2 starts at its detected drop (bar 16 of the EDM song)
        drop2 = self.songs[1][1]["drop"]
        self.assertAlmostEqual(p.song_start(1), drop2, delta=0.02)
        k, t = tl.song_at(T + 0.001)
        self.assertEqual(k, 1)
        self.assertAlmostEqual(t, drop2, delta=0.02)
        self.assertGreater(p.music_duration, a1.duration)
        # the grid on the timeline runs 110 BPM, then 128 BPM from the hand-over
        beats = np.asarray(p.markers.beats)
        self.assertAlmostEqual(np.median(np.diff(beats[beats < T])), 60 / 110, places=3)
        self.assertAlmostEqual(np.median(np.diff(beats[beats > T])), 60 / 128, places=3)

    def test_marking_the_second_songs_start(self):
        p = self.p
        old = p.song_start(1)
        new = p.set_song_start(1, old - 4 * 60 / 128 + 0.05)     # 4 beats earlier
        self.assertAlmostEqual(new, old - 4 * 60 / 128, delta=0.01)
        p.set_song_start(1, -1.0)                                 # back to automatic
        self.assertAlmostEqual(p.song_start(1), old, places=6)

    def test_soundtrack_and_save(self):
        from hitsync import audio_mix

        p = self.p
        s = p.recalculate()
        mix = audio_mix.build_soundtrack(p.music_source(), s, p.render_params,
                                         video_path=p.video_path)
        self.assertEqual(len(mix), int(round(s.duration * audio_mix.SR)))
        path = os.path.join(tempfile.mkdtemp(), "p.json")
        p.save(path)
        from hitsync.project import Project

        q = Project.load(path)
        self.assertEqual(q.extra_music, p.extra_music)
        self.assertIsNotNone(q.timeline())
        self.assertAlmostEqual(q.music_duration, p.music_duration, places=6)

    def test_per_song_beat_fix(self):
        p = self.p
        before = np.asarray(p.markers.beats)
        p.set_tempo_factor(0.5, song=1)
        T = p.timeline().handovers[0]
        after = np.asarray(p.markers.beats)
        self.assertTrue(np.allclose(before[before < T], after[after < T]))    # song 1 intact
        self.assertAlmostEqual(np.median(np.diff(after[after > T])), 2 * 60 / 128, places=3)
        p.reset_grid(song=1)
        self.assertTrue(np.allclose(np.asarray(p.markers.beats), before))

    def test_cut_song_one_switches_there(self):
        p = self.fresh()
        _, downs, _ = p.song_grid(0)
        want = float(downs[np.searchsorted(downs, p.sync.drop_time + 8.0)])
        got = p.set_song_end(0, want + 0.2)                   # snaps to the bar line
        self.assertAlmostEqual(got, want, places=6)
        self.assertEqual(p.song_end(0), (got, False))
        self.assertAlmostEqual(p.timeline().handovers[0], got, places=6)
        k, t = p.timeline().song_at(got + 0.01)
        self.assertEqual(k, 1)
        # the beat grid switches tempo right there
        beats = np.asarray(p.markers.beats)
        self.assertAlmostEqual(np.median(np.diff(beats[beats > got])), 60 / 128, places=3)
        # too early (before the drop + a bar): refused
        self.assertIsNone(p.set_song_end(0, p.sync.drop_time))
        p.clear_song_end(0)
        self.assertTrue(p.song_end(0)[1])
        self.assertAlmostEqual(p.timeline().handovers[0], self.p.timeline().handovers[0])

    def test_cut_last_song_ends_the_music(self):
        p = self.fresh()
        T = p.timeline().handovers[0]
        start = p.song_start(1)
        _, d2, _ = p.song_grid(1)
        cut = float(d2[np.searchsorted(d2, start + 10.0)])
        self.assertAlmostEqual(p.set_song_end(1, cut), cut)
        self.assertAlmostEqual(p.music_duration, T + cut - start, places=4)
        self.assertLessEqual(max(p.markers.beats), p.music_duration + 1e-6)
        # a single song can be cut too
        p.remove_song(1)
        self.assertIsNone(p.timeline())
        full = p.music_duration
        _, d1, _ = p.song_grid(0)
        c1 = float(d1[np.searchsorted(d1, p.sync.drop_time + 6.0)])
        p.set_song_end(0, c1)
        self.assertIsNotNone(p.timeline())
        self.assertAlmostEqual(p.music_duration, c1, places=4)
        self.assertLess(p.music_duration, full)

    def test_reorder_and_remove_the_first_song(self):
        p = self.fresh()
        one, two = self.songs[0][0], self.songs[1][0]
        a1, a2 = p.audio, p.extra_audio[0]
        drop1 = p.sync.drop_time
        p.set_tempo_factor(0.5, song=1)                        # a beat fix on song 2
        start2 = p.song_start(1)
        self.assertEqual(p.move_song(1, -1), 0)
        self.assertEqual(p.music_paths, [two, one])
        self.assertIs(p.audio, a2)                             # analyses move along
        self.assertIs(p.extra_audio[0], a1)
        self.assertEqual(p.stale(), (False, False))            # nothing to re-analyse
        self.assertAlmostEqual(p.sync.drop_time, start2, delta=0.02)
        self.assertAlmostEqual(p.song_start(1), drop1, delta=0.02)
        self.assertGreater(p.sync.bpm_override, 0)             # the fix came along
        self.assertEqual(p.grid_fix, {})
        self.assertAlmostEqual(p.grid_bpm, 64.0, delta=0.1)
        # delete the (new) first song: song 2 becomes the only one
        self.assertTrue(p.remove_song(0))
        self.assertEqual(p.music_paths, [one])
        self.assertIs(p.audio, a1)
        self.assertIsNone(p.timeline())
        self.assertAlmostEqual(p.sync.drop_time, drop1, delta=0.02)
        self.assertEqual(p.sync.bpm_override, 0.0)
        p.recalculate()
        self.assertGreater(p.schedule.duration, 0)
        # removing the only song leaves no music
        self.assertTrue(p.remove_song(0))
        self.assertEqual((p.music_path, p.extra_music), ("", []))
        self.assertIsNone(p.audio)
        self.assertIsNone(p.schedule)
        self.assertEqual(p.markers.beats, [])
        self.assertEqual(p.music_duration, 0.0)
        self.assertFalse(p.remove_song(0))
        # dropping it again brings it back from the cache
        p.music_path = one
        self.assertEqual(p.analyze(), (True, False))
        self.assertAlmostEqual(p.sync.drop_time, drop1, delta=0.02)
        self.assertGreater(p.recalculate().duration, 0)

    def test_type_or_pick_a_tempo(self):
        p = self.fresh()
        choices = p.tempo_choices(0)
        self.assertTrue(choices)
        self.assertTrue(all(abs(c / p.grid_bpm - 1) > 0.015 for c in choices))
        p.set_song_bpm(55.0, 0)                                # half of 110
        self.assertAlmostEqual(p.grid_bpm, 55.0, places=3)
        beats = np.asarray(p.markers.beats)
        self.assertAlmostEqual(np.median(np.diff(beats[beats < p.timeline().handovers[0]])),
                               60 / 55.0, places=3)
        p.set_song_bpm(64.0, 1)                                # song 2: half of 128
        T = p.timeline().handovers[0]
        beats = np.asarray(p.markers.beats)
        self.assertAlmostEqual(np.median(np.diff(beats[beats > T])), 60 / 64.0, places=3)
        p.reset_grid(0)
        p.reset_grid(1)
        self.assertAlmostEqual(p.grid_bpm, 110.0, delta=0.05)

    def test_cuts_and_order_are_saved(self):
        from hitsync.project import Project

        p = self.fresh()
        _, downs, _ = p.song_grid(0)
        cut = p.set_song_end(0, float(downs[np.searchsorted(downs, p.sync.drop_time + 8.0)]))
        p.move_song(0, 1)
        path = os.path.join(tempfile.mkdtemp(), "p.json")
        p.save(path)
        q = Project.load(path)
        self.assertEqual(q.music_paths, p.music_paths)
        self.assertEqual(q.extra_ends, [cut])
        self.assertAlmostEqual(q.music_duration, p.music_duration, places=6)


if __name__ == "__main__":
    unittest.main()
