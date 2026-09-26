"""Combo picker (order, on/off, slow-mo lead-in), letterbox on slow-mo parts,
velocity edit and per-hit effect choices."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync.config import SyncParams  # noqa: E402
from hitsync.models import ComboChoice, Hit, Markers  # noqa: E402
from hitsync.project import Project  # noqa: E402
from hitsync.sync_engine import build_schedule  # noqa: E402

PERIOD = 60 / 100.0
GRID = 0.2 + PERIOD * np.arange(400)
DROP = GRID[20]


def combo(start, n, period=0.6, jitter=0.03, seed=0):
    rng = np.random.default_rng(seed)
    return list(start + period * np.arange(n) + rng.uniform(-jitter, jitter, n))


def markers():
    m = Markers()
    m.set_beats(GRID)
    hits = combo(20, 10, seed=1) + combo(40, 12, seed=2) + combo(70, 11, seed=3)
    m.set_hits([Hit(t) for t in hits], combo_gap=1.2, regularity=0.3)
    return m


def params(**kw):
    kw.setdefault("drop_time", DROP)
    kw.setdefault("intro_end", 20.0)
    kw.setdefault("intro_start", 17.0)
    return SyncParams(**kw)


def on_grid(t, tol=1e-6):
    return np.min(np.abs(GRID - t)) < tol


class PlanTests(unittest.TestCase):
    def test_reordered_and_disabled_combos(self):
        m = markers()
        c1, c2, c3 = m.combos()
        s = build_schedule(m, params(), 100, 200, plan=[(c3, False), (c1, False)])
        placed = s.placed_hits()
        self.assertEqual(len(placed), len(c3) + len(c1))
        self.assertTrue(all(s.placements[i].status == "dropped" for i in c2))
        # c3 comes first (right on the drop), then c1 - earlier in the footage
        self.assertEqual(placed[0].hit_index, c3[0])
        self.assertAlmostEqual(placed[0].out_t, DROP, places=6)
        self.assertEqual(placed[len(c3)].hit_index, c1[0])
        # going back in the footage is a cut, and every step still has a hit
        self.assertGreaterEqual(len(s.cuts), 1)
        outs = np.array([p.out_t for p in placed])
        self.assertTrue(np.allclose(np.diff(outs), PERIOD, atol=1e-6))
        for p in placed:
            self.assertAlmostEqual(s.src_time(p.out_t), p.src_t, places=4)
        # never plays footage backwards
        for seg in s.segments:
            self.assertTrue(np.all(np.diff(seg.src_knots) >= -1e-9), seg.kind)

    def test_intro_plays_the_footage_before_the_first_planned_combo(self):
        m = markers()
        c1, c2, c3 = m.combos()
        s = build_schedule(m, params(), 100, 200, plan=[(c2, False), (c1, False)])
        intro = s.segments[0]
        self.assertEqual(intro.kind, "intro")
        self.assertAlmostEqual(intro.src_end, m.hits[c2[0]].t, places=6)
        self.assertLess(intro.src_start, intro.src_end)

    def test_slowmo_lead_in_lands_the_combo_on_the_beat(self):
        m = markers()
        c1, c2, _ = m.combos()
        p = params(lead_in_beats=4, lead_in_speed=0.45)
        s = build_schedule(m, p, 100, 200, plan=[(c1, False), (c2, True)])
        self.assertEqual(len(s.leadins), 1)
        a, b = s.leadins[0]
        first = s.placements[c2[0]]
        self.assertAlmostEqual(first.out_t, b, places=6)
        self.assertTrue(on_grid(first.out_t))
        self.assertAlmostEqual(s.src_time(first.out_t), first.src_t, places=4)
        # 4 beats after the step that would have held the hit
        last_c1 = s.placements[c1[-1]].out_t
        self.assertAlmostEqual(b - last_c1, 5 * PERIOD, places=6)
        # slow motion that ramps back to real time on the hit
        mid = (a + b) / 2 - 0.3
        self.assertAlmostEqual(s.speed_at(mid), 0.45, places=2)
        self.assertAlmostEqual(s.speed_at(b - 1e-3), 1.0, places=1)
        seg = s.segment_at(mid)
        self.assertEqual(seg.kind, "leadin")
        self.assertTrue(seg.cut_before)
        # the footage is what comes right before the combo
        self.assertAlmostEqual(seg.src_end, first.src_t, places=6)
        self.assertLess(first.src_t - seg.src_start, 4 * PERIOD)

    def test_letterbox_only_on_slow_motion_with_a_flash(self):
        m = markers()
        c1, c2, _ = m.combos()
        s = build_schedule(m, params(), 100, 200, plan=[(c1, False), (c2, True)])
        (i0, i1), (l0, l1) = s.letterbox
        self.assertEqual(s.letterbox_snap, [True, True])
        self.assertAlmostEqual(i1, DROP, places=6)
        self.assertEqual(s.letterbox_amount(DROP - 1.0), 1.0)      # bars in the intro
        self.assertEqual(s.letterbox_amount(DROP), 0.0)            # full frame on the hit
        self.assertEqual(s.letterbox_amount(DROP + 2 * PERIOD), 0.0)
        self.assertEqual(s.letterbox_amount((l0 + l1) / 2), 1.0)  # and in the lead-in
        # a quick flash peaks as each slow-mo part ends
        self.assertGreater(s.flash_amount(DROP), 0.75)
        self.assertEqual(s.flash_amount(DROP + 0.5), 0.0)
        self.assertEqual(s.flash_amount(DROP - 0.5), 0.0)
        self.assertGreater(s.flash_amount(l1), 0.75)
        # modes
        s2 = build_schedule(m, params(letterbox_mode="first combo"), 100, 200,
                            plan=[(c1, False), (c2, False)])
        a, b, _ = s2.combo_spans[0]
        self.assertEqual(s2.letterbox_amount((a + b) / 2), 1.0)
        c, d, _ = s2.combo_spans[1]
        self.assertEqual(s2.letterbox_amount((c + d) / 2), 0.0)
        s3 = build_schedule(m, params(letterbox_mode="off"), 100, 200, plan=[(c1, False)])
        self.assertEqual(s3.letterbox_amount(DROP - 1.0), 0.0)

    def test_focus_bars_on_a_chosen_combo(self):
        m = markers()
        c1, c2, c3 = m.combos()
        for mode in ("slowmo", "off"):
            s = build_schedule(m, params(letterbox_mode=mode), 100, 200,
                               plan=[(c1, False, False), (c2, False, True), (c3, False, False)])
            spans = s.combo_spans
            mid = [(a + b) / 2 for a, b, _ in spans]
            self.assertEqual(s.letterbox_amount(mid[0]), 0.0, mode)
            self.assertEqual(s.letterbox_amount(mid[1]), 1.0, mode)    # the focused combo
            self.assertEqual(s.letterbox_amount(mid[2]), 0.0, mode)
            # fully in by its first hit, and gone before the next combo
            self.assertEqual(s.letterbox_amount(s.placements[c2[0]].out_t), 1.0, mode)
            self.assertEqual(s.letterbox_amount(spans[2][0]), 0.0, mode)
        # "off" keeps the intro full frame; only the focused combo gets bars
        self.assertEqual(s.letterbox_amount(DROP - 1.0), 0.0)
        # old 2-tuples still work (no focus)
        s2 = build_schedule(m, params(letterbox_mode="off"), 100, 200, plan=[(c1, False)])
        self.assertEqual(s2.letterbox, [])

    def test_focus_after_slowmo_keeps_the_bars_on(self):
        m = markers()
        c1, c2, _ = m.combos()
        # the first combo, after the slow-mo intro, and a lead-in combo
        s = build_schedule(m, params(), 100, 200, plan=[(c1, False, True), (c2, True, True)])
        a1, b1, _ = s.combo_spans[0]
        l0, l1 = s.leadins[0]
        for t in np.linspace(DROP - 1.0, (a1 + b1) / 2, 40):
            self.assertEqual(s.letterbox_amount(t), 1.0, t)       # no dip on the drop
        for t in np.linspace((l0 + l1) / 2, l1 + 1.0, 40):
            self.assertEqual(s.letterbox_amount(t), 1.0, t)       # nor after the lead-in
        self.assertGreater(s.flash_amount(DROP), 0.75)            # the flash still marks it

    def test_velocity_edit_keeps_hits_on_the_beat(self):
        m = markers()
        c1 = m.combos()[0]
        s0 = build_schedule(m, params(velocity=0.0), 100, 200, plan=[(c1, False)])
        s1 = build_schedule(m, params(velocity=0.6), 100, 200, plan=[(c1, False)])
        for a, b in zip(s0.placed_hits(), s1.placed_hits()):
            self.assertAlmostEqual(a.out_t, b.out_t, places=6)
            self.assertAlmostEqual(s1.src_time(b.out_t), b.src_t, places=4)
        h = s1.placed_hits()
        t0, t1 = h[3].out_t, h[4].out_t
        # slow on the impact, faster in between
        self.assertLess(s1.speed_at(t0 + 0.01), 0.6)
        self.assertGreater(s1.speed_at((t0 + t1) / 2), 1.1)

    def test_beats_per_hit(self):
        m = markers()
        c1 = m.combos()[0]
        for spacing, step in (("0.5", 0.5), ("1", 1.0), ("2", 2.0)):
            s = build_schedule(m, params(combo_spacing=spacing), 100, 200, plan=[(c1, False)])
            outs = np.array([p.out_t for p in s.placed_hits()])
            self.assertTrue(np.allclose(np.diff(outs), PERIOD * step, atol=1e-6), spacing)
        # auto: a fast song (0.3 s beats) gets a hit every other beat
        fast = Markers()
        fast.set_beats(0.1 + 0.3 * np.arange(800))
        fast.set_hits([Hit(t) for t in combo(20, 12, seed=4)], 1.2, 0.3)
        s = build_schedule(fast, params(drop_time=6.1), 100, 200, plan=[(fast.combos()[0], False)])
        self.assertEqual(s.beats_per_hit, 2.0)


class ProjectPlanTests(unittest.TestCase):
    def project(self):
        p = Project()
        p.markers = markers()
        p.sync = params()
        return p

    def test_default_plan_is_every_real_combo_in_time_order(self):
        p = self.project()
        plan = p.engine_plan()
        self.assertEqual([e[0] for e in plan], p.real_combos())

    def test_plan_survives_redetection(self):
        p = self.project()
        entries = p.plan_entries()
        (a, _), (b, _), (c, _) = entries
        b.enabled = False
        c.lead_in = True
        c_key = c.key
        p.set_plan([c, a, b])
        # re-detect: same hits, slightly moved, plus a brand-new combo
        hits = [Hit(h.t + 0.01) for h in p.markers.hits] + [Hit(t) for t in combo(90, 10, seed=9)]
        p.markers.set_hits(hits, 1.2, 0.3)
        plan = p.plan_entries()
        self.assertEqual(len(plan), 4)
        self.assertAlmostEqual(plan[0][0].key, c_key + 0.01, places=6)
        self.assertTrue(plan[0][0].lead_in)
        self.assertFalse(plan[2][0].enabled)
        self.assertTrue(plan[3][0].enabled)         # new combo appended
        used = p.engine_plan()
        self.assertEqual(len(used), 3)

    def test_combat_starts_here(self):
        p = self.project()
        c1, c2, c3 = p.real_combos()
        key = p.combat_starts_at(p.markers.hits[c3[0]].t - 0.1)
        self.assertAlmostEqual(key, p.markers.hits[c3[0]].t)
        self.assertEqual(p.engine_plan()[0][0], c3)
        # mid-combo: the combo is split there and the later part opens
        mid = c2[4]
        p.combat_starts_at(p.markers.hits[mid].t)
        self.assertEqual(p.engine_plan()[0][0][0], mid)

    def test_rank_and_auto_pick(self):
        p = self.project()
        ranks = p.rank_combos()
        self.assertEqual(len(ranks), 3)
        self.assertAlmostEqual(max(ranks.values()), 1.0)
        best = max(ranks, key=ranks.get)
        p.audio = None
        p.markers.beats = list(GRID[:46])          # room for 25 hits after the drop
        p.auto_pick()
        on = [c for c in p.combo_plan if c.enabled]
        self.assertIn(best, [c.key for c in on])
        self.assertLessEqual(sum(len(e[0]) for e in p.engine_plan()), 25)
        self.assertEqual([c.key for c in on], sorted(c.key for c in on))

    def test_per_hit_effects_survive_redetection(self):
        m = markers()
        m.hits[3].fx = False
        m.hits[5].fx = True
        t3, t5 = m.hits[3].t, m.hits[5].t
        m.set_hits([Hit(h.t + 0.02) for h in m.hits], 1.2, 0.3)
        by_t = {round(h.t - 0.02, 6): h.fx for h in m.hits}
        self.assertIs(by_t[round(t3, 6)], False)
        self.assertIs(by_t[round(t5, 6)], True)
        d = m.hits[3].to_dict()
        self.assertEqual(Hit.from_dict(d).fx, False)

    def test_plan_is_saved(self):
        import tempfile

        p = self.project()
        entries = p.plan_entries()
        entries[1][0].lead_in = True
        entries[1][0].focus = True
        p.set_plan([c for c, _ in entries][::-1])
        path = os.path.join(tempfile.mkdtemp(), "p.json")
        p.save(path)
        q = Project.load(path)
        self.assertTrue(q.plan_custom)
        self.assertEqual([c.key for c in q.combo_plan], [c.key for c in p.combo_plan])
        self.assertTrue(q.combo_plan[1].lead_in)
        self.assertTrue(q.combo_plan[1].focus)
        self.assertFalse(q.combo_plan[0].focus)
        self.assertEqual(q.engine_plan()[1][1:], (True, True))
        self.assertIsInstance(q.combo_plan[0], ComboChoice)


if __name__ == "__main__":
    unittest.main()
