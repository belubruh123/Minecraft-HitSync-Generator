"""Beat grid, downbeats and music-start detection on songs with known answers.

The songs are synthesized (tests/synth_music.py) to reproduce what trips up
beat trackers: the 3-3-2 riff of "Shape of You" (reads as 128 BPM), a
drumless intro, off-beat hats, snare-roll builds, swing, silence before the
song, half-time trap and a live band whose tempo wanders.
"""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync import music_grid, sections  # noqa: E402
from hitsync.audio_analysis import click_track, snap_to_attacks  # noqa: E402
from tests import synth_music as sm  # noqa: E402


def analyse(fn, **kw):
    y, info = fn(**kw)
    ma = music_grid.analyze(y)
    beats = ma.grid.beats(ma.duration)
    bars = beats[ma.grid.is_downbeat(beats)]
    secs = sections.detect(ma.curve_t, ma.level, ma.bass, ma.kick, ma.snare, bars, ma.duration)
    return ma, beats, bars, secs, info


def grid_error(beats, info, start=None):
    """Distance (s) of each grid beat from the nearest true beat."""
    P = 60.0 / info["bpm"]
    true = info.get("beats")
    if true is None:
        true = info["first_beat"] + P * np.arange(int(200 / P))
    lo = info["kick_in"] if start is None else start
    inside = beats[(beats >= lo - 0.05) & (beats <= true[-1] + 0.05)]
    return np.array([b - true[np.argmin(np.abs(true - b))] for b in inside])


class GridTests(unittest.TestCase):
    def assertOnBeats(self, ma, beats, info, ms=5.0):
        self.assertAlmostEqual(ma.grid.bpm, info["bpm"], delta=0.05)
        err = grid_error(beats, info)
        self.assertGreater(len(err), 20)
        self.assertLess(np.abs(err).max() * 1000, ms, err)

    def test_shape_of_you_rhythm_is_96_on_the_beat(self):
        ma, beats, bars, secs, info = analyse(sm.shape_like)
        self.assertOnBeats(ma, beats, info)
        self.assertFalse(ma.grid.drift)
        self.assertGreater(ma.grid.confidence, 0.8)
        # bar lines on beat 1
        self.assertLess(abs(((bars[0] - info["first_beat"]) / (4 * 60 / 96) + 0.5) % 1 - 0.5), 0.01)

    def test_edm_offbeat_hats_and_build(self):
        ma, beats, bars, secs, info = analyse(sm.edm_like)
        self.assertOnBeats(ma, beats, info)

    def test_backbeat_swing_silence_and_dnb(self):
        for kw in (dict(), dict(bpm=88, swing=0.17, seed=5), dict(bpm=123, lead_silence=3.3),
                   dict(bpm=174, seed=7)):
            with self.subTest(**kw):
                ma, beats, bars, secs, info = analyse(sm.backbeat_like, **kw)
                self.assertOnBeats(ma, beats, info)
                # downbeat = the kick + chord change, not the snare
                k = round((bars[0] - info["first_beat"]) / (60.0 / info["bpm"]))
                self.assertEqual(k % 4, 0)

    def test_half_time_reads_as_either_level(self):
        ma, beats, bars, secs, info = analyse(sm.halftime_like)
        ratio = ma.grid.bpm / info["bpm"]
        self.assertTrue(abs(ratio - 1) < 1e-3 or abs(ratio - 0.5) < 1e-3, ma.grid.bpm)
        self.assertLess(np.abs(grid_error(beats, info)).max() * 1000, 5.0)

    def test_wandering_live_tempo_is_flagged(self):
        ma, *_ = analyse(sm.backbeat_like, drift=0.03, duration=90)
        self.assertTrue(ma.grid.drift)
        self.assertLess(ma.grid.confidence, 0.5)

    def test_forced_tempo_refits_the_phase(self):
        y, info = sm.backbeat_like(bpm=101)
        ma = music_grid.analyze(y)
        g = music_grid.grid_with_tempo(ma.onsets, ma.drum_w, ma.duration, 101.0)
        self.assertAlmostEqual(g.bpm, 101.0, places=6)
        self.assertLess(np.abs(grid_error(g.beats(ma.duration), info)).max(), 0.005)

    def test_tap_tempo(self):
        taps = 0.25 + 0.6 * np.arange(8) + np.random.default_rng(0).normal(0, 0.01, 8)
        bpm, phase = music_grid.tap_tempo(taps)
        self.assertAlmostEqual(bpm, 100.0, delta=1.0)
        self.assertLess(abs((phase - 0.25 + 0.3) % 0.6 - 0.3), 0.02)

    def test_fast(self):
        y, _ = sm.backbeat_like(duration=180.0)
        t = time.perf_counter()
        music_grid.analyze(y)
        self.assertLess(time.perf_counter() - t, 4.0)   # generous for slow CI machines


class SectionTests(unittest.TestCase):
    def test_music_starts_where_the_beat_kicks_in(self):
        ma, beats, bars, secs, info = analyse(sm.shape_like)
        self.assertAlmostEqual(secs.best, info["kick_in"], delta=0.05)

    def test_drop_beats_the_build(self):
        ma, beats, bars, secs, info = analyse(sm.edm_like)
        self.assertAlmostEqual(secs.best, info["drop"], delta=0.05)
        self.assertGreater(secs.song_end, 0.9 * ma.duration)

    def test_silence_before_the_song(self):
        ma, beats, bars, secs, info = analyse(sm.backbeat_like, bpm=123, lead_silence=3.3)
        self.assertAlmostEqual(secs.first_sound, info["first_beat"], delta=0.1)
        # beat from the first bar: start a few bars in so there's an intro
        self.assertGreater(secs.best, info["first_beat"] + 2)
        length = sections.auto_intro_length(secs.best, bars, secs.first_sound, 60 / 123)
        self.assertGreaterEqual(secs.best - length, secs.first_sound - 0.2)

    def test_intro_length_follows_the_song(self):
        bars = 0.5 + 2.0 * np.arange(40)          # 120 BPM
        # short song intro (start at 8.5 s): all of it
        self.assertAlmostEqual(sections.auto_intro_length(8.5, bars, 0.4, 0.5), 8.0)
        # long intro: 4 bars
        self.assertAlmostEqual(sections.auto_intro_length(40.5, bars, 0.4, 0.5), 8.0)
        # slow song, 4 bars > 9 s: 2 bars
        slow = 0.5 + 2.5 * np.arange(40)
        self.assertAlmostEqual(sections.auto_intro_length(50.5, slow, 0.4, 0.625), 5.0)


class HelperTests(unittest.TestCase):
    def test_snap_to_attacks(self):
        beats = np.array([1.0, 2.03, 3.2])
        att = np.array([0.99, 2.0, 3.0])
        np.testing.assert_allclose(snap_to_attacks(beats, att), [0.99, 2.0, 3.2])

    def test_click_track(self):
        c = click_track([0.5, 1.0], [0.5], 44100 * 2)
        self.assertGreater(np.abs(c[22050:22100]).max(), 0.1)
        self.assertEqual(np.abs(c[:22000]).max(), 0.0)


if __name__ == "__main__":
    unittest.main()
