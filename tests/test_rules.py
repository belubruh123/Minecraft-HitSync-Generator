"""The montage rules: static grid, real combos only, a hit on every beat,
combo spacing from the player's rhythm, original gameplay hit sounds."""
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync import audio_mix, beatgrid  # noqa: E402
from hitsync.config import SyncParams  # noqa: E402
from hitsync.ffmpeg_utils import find_ffmpeg  # noqa: E402
from hitsync.models import Hit, Markers  # noqa: E402
from hitsync.sync_engine import build_schedule  # noqa: E402

PERIOD = 60 / 101.0          # 101 BPM, like "Wait a Minute!"
GRID = 0.3 + PERIOD * np.arange(400)


def combo(start, n, period=0.6, jitter=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return list(start + period * np.arange(n) + rng.uniform(-jitter, jitter, n))


class StaticGridTests(unittest.TestCase):
    def test_fit_ignores_jitter_doubling_and_missing_beats(self):
        rng = np.random.default_rng(1)
        beats = GRID[:330] + rng.normal(0, 0.012, 330)
        beats = np.delete(beats, [40, 41, 200])                     # tracker misses
        doubled = GRID[100:130] + PERIOD / 2                          # 1/8 section
        grid, bpm, _ = beatgrid.fit_static_grid(np.concatenate([beats, doubled]), 200.0)
        self.assertAlmostEqual(bpm, 101.0, delta=0.02)
        # every true beat is on the fitted grid, with no drift at the end
        err = np.array([np.min(np.abs(grid - t)) for t in GRID[:330]])
        self.assertLess(err.max(), 0.01)
        self.assertLess(abs(np.median(np.diff(grid)) - PERIOD), 1e-4)

    def test_bpm_override_and_offset(self):
        grid, bpm, _ = beatgrid.fit_static_grid(GRID[:100], 60.0, bpm=101.0, offset=0.02)
        self.assertAlmostEqual(bpm, 101.0)
        self.assertAlmostEqual(np.min(np.abs(grid - (GRID[10] + 0.02))), 0.0, places=6)


class ComboRuleTests(unittest.TestCase):
    def markers(self, hits):
        m = Markers()
        m.set_beats(GRID)
        m.set_hits([Hit(t) for t in hits], combo_gap=1.2, regularity=0.3)
        return m

    def test_irregular_hits_break_a_combo(self):
        hits = combo(20, 12) + [28.0, 28.35, 29.2, 29.6] + combo(40, 11, seed=2)
        m = self.markers(hits)
        sizes = [len(g) for g in m.combos()]
        self.assertIn(12, sizes)
        self.assertIn(11, sizes)
        self.assertTrue(all(s < 10 for s in sizes if s not in (11, 12)))

    def test_only_combos_of_10_plus_make_the_edit(self):
        hits = combo(10, 5) + combo(20, 12, seed=1) + combo(35, 9, seed=2) + combo(50, 10, seed=3)
        m = self.markers(hits)
        p = SyncParams(intro_start=14.0, intro_end=20.0, drop_time=GRID[27])
        s = build_schedule(m, p, 70, 120)
        placed = [pl for pl in s.placements if pl.out_t is not None]
        self.assertEqual(len(placed), 22)                  # 12 + 10
        self.assertEqual(s.combos_kept, 2)
        self.assertTrue(all(pl.status == "dropped" for pl in s.placements
                            if pl.src_t < 19 or 34 < pl.src_t < 41))

    def test_every_beat_gets_a_hit_and_first_hit_is_on_the_drop(self):
        hits = combo(20, 12, seed=4) + combo(40, 11, seed=5) + combo(70, 10, seed=6)
        m = self.markers(hits)
        drop = GRID[27]
        p = SyncParams(intro_start=14.0, intro_end=20.0, drop_time=drop + 0.1)  # snaps to grid
        s = build_schedule(m, p, 90, 120)
        outs = np.array(sorted(pl.out_t for pl in s.placements if pl.out_t is not None))
        self.assertEqual(len(outs), 33)
        self.assertAlmostEqual(outs[0], drop, places=6)
        idx = np.round((outs - GRID[0]) / PERIOD)
        self.assertTrue(np.all(np.diff(idx) == 1), np.diff(idx))           # no empty beat
        self.assertLess(np.max(np.abs(outs - (GRID[0] + idx * PERIOD))), 1e-6)
        self.assertEqual(len(s.cuts), 2)                                     # between combos
        # the schedule really shows each hit's frame at its beat
        for pl in s.placements:
            if pl.out_t is not None:
                self.assertAlmostEqual(s.src_time(pl.out_t), pl.src_t, places=4)
        # slow-mo intro ramps into the first hit
        intro = s.segments[0]
        self.assertEqual(intro.kind, "intro")
        self.assertLess(s.intro_speed, 0.8)
        self.assertAlmostEqual(intro.src_end, hits[0], places=6)
        self.assertAlmostEqual(intro.speed_at(drop - 1e-3), 1.0, places=2)

    def test_spacing_comes_from_average_hit_gap(self):
        m = self.markers(combo(20, 12, period=2 * PERIOD, jitter=0.03))
        # (with the static beat unticked; a static beat always has one hit per beat)
        p = SyncParams(intro_enabled=False, combo_gap=2.0, static_grid=False)
        m.auto_group(2.0, 0.3)
        s = build_schedule(m, p, 60, 120)
        self.assertEqual(s.beats_per_hit, 2.0)
        outs = np.array([pl.out_t for pl in s.placements if pl.out_t is not None])
        self.assertTrue(np.allclose(np.diff(outs), 2 * PERIOD, atol=1e-6))

        m = self.markers(combo(20, 12, period=PERIOD, jitter=0.03))
        s = build_schedule(m, SyncParams(intro_enabled=False), 60, 120)
        self.assertEqual(s.beats_per_hit, 1.0)

    def test_static_beat_is_one_unbroken_chain_of_locked_hits(self):
        """Static beat: every step (beats per hit) from the drop to the end
        has a locked hit, however sloppy the rhythm."""
        hits = (combo(20, 12, period=2 * PERIOD, jitter=0.03, seed=7)   # slow player
                + combo(60, 11, jitter=0.12, seed=8)                     # sloppy player
                + combo(90, 10, seed=9)[:7] + [95.5, 96.1, 96.7])        # a missed hit
        m = Markers()
        m.set_beats(GRID)
        m.set_hits([Hit(t) for t in hits], combo_gap=2.0, regularity=0.6)
        drop = GRID[27]
        for extra in ({}, {"fill_every_beat": False}, {"combo_spacing": "2"},
                      {"combo_spacing": "0.5"}, {"jitter_tolerance_ms": 0.0},
                      {"lock_mode": "trim"}, {"ramp_max_speed": 1.1}, {"velocity": 0.6}):
            p = SyncParams(intro_start=14.0, intro_end=20.0, drop_time=drop, combo_gap=2.0,
                           combo_regularity=0.6, **extra)
            step = float(extra.get("combo_spacing", 1))
            s = build_schedule(m, p, 120, 200)
            placed = [pl for pl in s.placements if pl.out_t is not None]
            self.assertTrue(placed, extra)
            self.assertTrue(all(pl.locked for pl in placed), extra)          # all green
            outs = np.sort([pl.out_t for pl in placed])
            idx = np.round((outs - drop) / (PERIOD * step))           # steps from the drop
            self.assertLess(np.max(np.abs(outs - (drop + idx * PERIOD * step))), 1e-6, extra)
            self.assertAlmostEqual(outs[0], drop, places=6)
            self.assertTrue(np.all(np.diff(idx) == 1), (extra, np.diff(idx)))  # no gap
            # the montage ends one step after the last hit: no empty beats
            self.assertAlmostEqual(s.end, outs[-1] + PERIOD * step, places=6)
            self.assertEqual(s.beats_per_hit, step)


class ShortIntroTests(unittest.TestCase):
    def build(self, drop, first_hit, intro_length=5.0):
        m = Markers()
        m.set_beats(GRID)
        m.set_hits([Hit(t) for t in combo(first_hit, 12, seed=7)], 1.2, 0.3)
        p = SyncParams(intro_end=first_hit, drop_time=drop, intro_length=intro_length)
        return m, p, build_schedule(m, p, first_hit + 30, 200)

    def test_song_and_footage_before_the_intro_are_cut(self):
        drop = GRID[50]                                     # drop ~29.9 s into the song
        m, p, s = self.build(drop, first_hit=19.0)
        n = round(5.0 / PERIOD)
        self.assertAlmostEqual(s.start, drop - n * PERIOD, places=6)   # whole beats before drop
        self.assertAlmostEqual(drop - s.start, 5.0, delta=PERIOD / 2)
        intro = s.segments[0]
        self.assertAlmostEqual(intro.out_start, s.start, places=6)
        self.assertAlmostEqual(intro.out_end, drop, places=6)
        # only ~2 s of footage fills the 5 s intro at 0.4x: the rest is cut
        self.assertAlmostEqual(intro.src_end, 19.0 + (m.hits[0].t - 19.0), places=6)
        used = intro.src_end - intro.src_start
        self.assertTrue(1.5 < used < 3.0, used)
        self.assertAlmostEqual(s.intro_speed, 0.4, places=2)
        self.assertAlmostEqual(s.speed_at(drop - 1e-3), 1.0, places=2)
        # the montage is the short intro plus the combo
        self.assertAlmostEqual(s.duration, s.end - s.start)
        self.assertLess(s.duration, 5.0 + 12 * PERIOD + 3)

    def test_zero_keeps_the_whole_music_intro(self):
        m, p, s = self.build(GRID[50], first_hit=19.0, intro_length=0.0)
        self.assertEqual(s.start, 0.0)
        self.assertAlmostEqual(s.segments[0].out_start, 0.0)

    @unittest.skipUnless(find_ffmpeg(), "ffmpeg unavailable")
    def test_soundtrack_starts_where_the_song_is_cut(self):
        from hitsync.config import RenderParams

        drop = GRID[50]
        m, p, s = self.build(drop, first_hit=19.0)
        sr = audio_mix.SR
        song = np.zeros((int(60 * sr), 2), np.float32)
        song[int(drop * sr): int(drop * sr) + 200] = 0.9      # a click exactly at the drop
        path = os.path.join(tempfile.mkdtemp(), "song.wav")
        audio_mix.write_wav(path, song)
        mix = audio_mix.build_soundtrack(path, s, RenderParams(hit_sound="off"))
        self.assertEqual(len(mix), int(round(s.duration * sr)))
        click = np.argmax(np.abs(mix).max(axis=1) > 0.5) / sr
        self.assertAlmostEqual(click, drop - s.start, delta=0.002)   # drop at 5 s in


@unittest.skipUnless(find_ffmpeg(), "ffmpeg unavailable")
class OriginalHitSoundTests(unittest.TestCase):
    def test_snippets_start_at_the_recorded_hit_sound(self):
        from tests import synth

        d = tempfile.mkdtemp()
        video, wav, muxed = (os.path.join(d, n) for n in ("v.mp4", "a.wav", "va.mp4"))
        synth.write_video(video, duration=6)
        sr = audio_mix.SR
        y = np.random.default_rng(0).normal(0, 0.003, (6 * sr, 2)).astype(np.float32)
        hit = audio_mix.synth_punch()
        for t in (2.0, 3.5):                         # "hit sounds" 20 ms after the frame
            i = int((t + 0.02) * sr)
            y[i:i + len(hit)] += hit
        audio_mix.write_wav(wav, y)
        subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", video, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", muxed], check=True)
        snips = audio_mix.original_hit_snippets(muxed, [2.0, 3.5, 5.0])
        self.assertIsNotNone(snips)
        for s in snips[:2]:
            loud = np.nonzero(np.abs(s).max(axis=1) > 0.1)[0]
            self.assertGreater(len(loud), 0)
            self.assertLess(loud[0] / sr, 0.03)        # attack right at the start
        self.assertIsNone(snips[2])                    # silence: no fake hit sound

    def test_track_keeps_entire_sounds_and_never_skips_a_hit(self):
        from types import SimpleNamespace as NS
        from tests import synth

        d = tempfile.mkdtemp()
        video, wav, muxed = (os.path.join(d, n) for n in ("v.mp4", "a.wav", "va.mp4"))
        synth.write_video(video, duration=8)
        sr = audio_mix.SR
        y = np.zeros((8 * sr, 2), np.float32)
        tone = np.sin(2 * np.pi * 300 * np.arange(int(0.45 * sr)) / sr)[:, None] * np.ones(2)
        levels = {2.0: 0.5, 2.6: 0.5, 3.2: 0.02, 3.8: 0.5}      # 3.2 s is a very quiet hit
        for t, lvl in levels.items():
            i = int(t * sr)
            y[i:i + len(tone)] += tone * lvl                     # 0.45 s long sounds
        audio_mix.write_wav(wav, y)
        subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", video, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", muxed], check=True)
        # edit places them one beat (0.594 s) apart starting at 10 s
        placed = [NS(src_t=t, out_t=10 + k * PERIOD) for k, t in enumerate(levels)]
        track = audio_mix.original_hit_track(muxed, placed, 9.0, int(6 * sr), 1.0)
        env = np.abs(track).max(axis=1)
        for k, (t, lvl) in enumerate(levels.items()):
            at = int((1 + k * PERIOD) * sr)
            self.assertGreater(env[at: at + int(0.02 * sr)].max(), 0.0, t)       # never skipped
            # the whole 0.45 s sound is there, not faded out early
            late = env[at + int(0.40 * sr): at + int(0.43 * sr)].max()
            self.assertGreater(late, 0.5 * env[at: at + int(0.1 * sr)].max(), t)

    def test_drifting_sound_offsets_are_matched_hit_by_hit(self):
        from tests import synth

        d = tempfile.mkdtemp()
        video, wav, muxed = (os.path.join(d, n) for n in ("v.mp4", "a.wav", "va.mp4"))
        synth.write_video(video, duration=10)
        sr = audio_mix.SR
        y = np.random.default_rng(3).normal(0, 0.002, (10 * sr, 2)).astype(np.float32)
        hit = audio_mix.synth_punch()
        flashes = 2.0 + 0.6 * np.arange(10)
        lags = np.array([-0.04, -0.05, -0.17, -0.34, -0.26, -0.12, -0.05, -0.30, -0.08, -0.04])
        for t, lag in zip(flashes, lags):                 # sound leads the flash, irregularly
            i = int((t + lag) * sr)
            y[i:i + len(hit)] += hit
        audio_mix.write_wav(wav, y)
        subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", video, "-i", wav,
                        "-c:v", "copy", "-c:a", "aac", "-shortest", muxed], check=True)
        audio_mix._game_audio.cache_clear()
        audio_mix._sound_bursts.cache_clear()
        _, onsets, found = audio_mix._hit_onsets(muxed, list(flashes))
        self.assertTrue(all(found))
        err = np.array(onsets) / sr - (flashes + lags)
        # cut points sit just before each attack; a wrong match would be >= 100 ms off
        self.assertLess(np.abs(err).max(), 0.025, err)

    def test_no_audio_track_falls_back(self):
        from tests import synth

        video = os.path.join(tempfile.mkdtemp(), "v.mp4")
        synth.write_video(video, duration=2)
        self.assertIsNone(audio_mix.original_hit_snippets(video, [1.0]))


if __name__ == "__main__":
    unittest.main()
