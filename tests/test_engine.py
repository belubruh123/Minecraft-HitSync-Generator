"""Pure alignment-engine tests (no media needed)."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync.config import SyncParams  # noqa: E402
from hitsync.models import Hit, Markers  # noqa: E402
from hitsync.sync_engine import build_schedule  # noqa: E402

BEATS = np.arange(0, 40, 0.5)           # 120 BPM
HITS = [10.0, 10.53, 11.02, 11.47, 12.04,   # combo 1 (jittery)
        20.0, 20.55, 21.03,                  # combo 2 after a long walk
        30.0]                                # lone hit


def legacy(**kw):
    """Params for the original behaviour: any combo length, no every-beat fill."""
    kw.setdefault("intro_start", 5.0)
    kw.setdefault("intro_end", 9.5)
    kw.setdefault("drop_time", 8.0)
    kw.setdefault("intro_length", 0.0)       # whole music intro
    kw.setdefault("letterbox_mode", "combos")  # bars on every combo
    return SyncParams(min_combo_len=1, fill_every_beat=False, static_grid=False, **kw)


def make(params=None, hits=HITS, beats=BEATS):
    m = Markers()
    m.set_beats(beats)
    p = params or legacy()
    m.set_hits([Hit(t) for t in hits], p.combo_gap)
    return m, p


def on_grid(t, grid, tol=1e-6):
    return np.min(np.abs(np.asarray(grid) - t)) < tol


class EngineTests(unittest.TestCase):
    def test_combos_grouped(self):
        m, _ = make()
        self.assertEqual([len(c) for c in m.combos()], [5, 3, 1])

    def test_intro_slowmo_ramps_to_realtime_at_drop(self):
        m, p = make()
        s = build_schedule(m, p, 40, 40)
        intro = s.segments[0]
        self.assertEqual(intro.kind, "intro")
        self.assertAlmostEqual(intro.out_end, p.drop_time, places=6)
        self.assertAlmostEqual(intro.src_end, p.intro_end, places=6)
        self.assertAlmostEqual(intro.src_start, p.intro_start, places=3)
        self.assertLess(s.intro_speed, 0.7)
        self.assertAlmostEqual(intro.speed_at(p.drop_time - 1e-4), 1.0, places=2)
        self.assertAlmostEqual(intro.speed_at(0.1), s.intro_speed, places=6)

    def test_first_hit_after_drop_on_beat_when_gap_is_off_grid(self):
        # 0.35 s between intro end and first hit is not a beat multiple
        p = legacy(intro_end=9.65)
        m, p = make(p)
        s = build_schedule(m, p, 40, 40)
        first = s.placements[0]
        self.assertTrue(first.locked, first)
        self.assertTrue(on_grid(first.out_t, BEATS), first)
        self.assertAlmostEqual(s.segments[0].out_end, p.drop_time)
        self.assertAlmostEqual(s.speed_at(p.drop_time + 0.01), 1.0, places=6)

    def test_hits_lock_to_beats_within_tolerance(self):
        m, p = make()
        s = build_schedule(m, p, 40, 40)
        fine = np.arange(0, 60, 0.25)
        placed = [pl for pl in s.placements if pl.out_t is not None]
        self.assertEqual(len(placed), len(HITS))
        for pl in placed:
            self.assertTrue(pl.locked, pl)
            self.assertTrue(on_grid(pl.out_t, fine), pl)
            self.assertLessEqual(abs(pl.error_ms), p.jitter_tolerance_ms + 1e-6)
        # schedule maps each hit's source time to its output time
        for pl in placed:
            self.assertAlmostEqual(s.src_time(pl.out_t), pl.src_t, places=4)

    def test_combo_first_hits_land_on_full_beats(self):
        m, p = make()
        s = build_schedule(m, p, 40, 40)
        for grp in m.combos():
            pl = s.placements[grp[0]]
            self.assertTrue(on_grid(pl.out_t, BEATS), pl)

    def test_long_gaps_are_cut(self):
        m, p = make()
        s = build_schedule(m, p, 40, 40)
        self.assertGreaterEqual(len(s.cuts), 2)
        removed = s.removed_source_ranges()
        self.assertTrue(any(a < 15 < b for a, b in removed), removed)
        # output is much shorter than walking through the footage would be
        last = s.placements[-1].out_t
        self.assertLess(last - p.drop_time, 30.0 - p.intro_end - 8)

    def test_source_time_is_monotonic(self):
        m, p = make()
        s = build_schedule(m, p, 40, 40)
        ts = np.arange(0, s.duration, 1 / 60)
        src = np.array([s.src_time(t) for t in ts])
        self.assertTrue(np.all(np.diff(src) >= -1e-9))

    def test_letterbox_follows_combos(self):
        m, p = make()
        s = build_schedule(m, p, 40, 40)
        self.assertEqual(len(s.letterbox), 2)       # lone hit gets no bars
        start, end = s.letterbox[0]
        mid = (start + end) / 2
        self.assertAlmostEqual(s.letterbox_amount(mid), 1.0)
        self.assertEqual(s.letterbox_amount(start - 0.01), 0.0)
        self.assertEqual(s.letterbox_amount(end + p.letterbox_fade + 0.01), 0.0)
        a1 = s.letterbox_amount(start + p.letterbox_fade * 0.3)
        a2 = s.letterbox_amount(start + p.letterbox_fade * 0.7)
        self.assertTrue(0 < a1 < a2 < 1)            # eased in

    def test_small_gaps_are_speed_ramped_onto_the_beat(self):
        m, p = make(hits=[10.0, 10.5, 11.13, 11.6])  # third hit 130 ms late
        s = build_schedule(m, p, 40, 40)
        pl = s.placements[2]
        self.assertTrue(pl.locked)
        self.assertEqual(pl.status, "speed")
        self.assertTrue(on_grid(pl.out_t, BEATS))
        self.assertAlmostEqual(s.src_time(pl.out_t), pl.src_t, places=4)

    def test_gaps_beyond_ramp_bounds_are_not_forced(self):
        m, p = make(hits=[10.0, 10.5, 11.25, 11.75])  # 250 ms off: 1.5x needed
        p.ramp_max_speed = 1.2
        p.ramp_min_speed = 0.9
        s = build_schedule(m, p, 40, 40)
        self.assertEqual(s.placements[2].status, "unsynced")

    def test_trim_mode_drops_late_milliseconds(self):
        p = legacy(lock_mode="trim")
        m, p = make(p)
        s = build_schedule(m, p, 40, 40)
        self.assertTrue(any(seg.kind == "trim" for seg in s.segments))
        for pl in s.placements:
            if pl.status == "trim":
                self.assertGreater(pl.error_ms, 0)
                self.assertAlmostEqual(s.src_time(pl.out_t), pl.src_t, places=4)

    def test_tempo_matching_scales_speed(self):
        beats = np.concatenate([np.arange(0, 20, 0.5), 20 + np.arange(0, 20, 0.4)])
        m, p = make(hits=[], beats=beats)
        p.intro_enabled = False
        s = build_schedule(m, p, 60, 40)
        self.assertAlmostEqual(s.speed_at(5.0), 1.0, places=2)
        self.assertAlmostEqual(s.speed_at(30.0), 1.25, places=2)
        p.tempo_matching = False
        s = build_schedule(m, p, 60, 40)
        self.assertAlmostEqual(s.speed_at(30.0), 1.0, places=6)

    def test_recalculate_after_edits(self):
        m, p = make()
        s1 = build_schedule(m, p, 40, 40)
        m.delete_hit(2)                 # false positive
        m.split_combo_at(2)             # split combo 1
        m.add_hit(25.0, p.combo_gap)    # missing hit
        s2 = build_schedule(m, p, 40, 40)
        self.assertEqual(len(m.combos()), 5)
        self.assertEqual(len(s2.placements), len(HITS))
        self.assertNotEqual(len(s1.letterbox), len(s2.letterbox))
        m.merge_with_previous(2)
        self.assertEqual(len(m.combos()), 4)

    def test_no_intro_starts_on_beat(self):
        m, p = make()
        p.intro_enabled = False
        s = build_schedule(m, p, 40, 40)
        pl = s.placements[0]
        self.assertTrue(pl.locked)
        self.assertTrue(on_grid(pl.out_t, BEATS))
        self.assertAlmostEqual(s.segments[0].out_start, 0.0)


if __name__ == "__main__":
    unittest.main()
