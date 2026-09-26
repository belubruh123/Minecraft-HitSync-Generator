"""Montage look: filters, hit effects (only on ticked hits), effect ranges,
transitions, start/end fades, motion blur, captions and styles."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hitsync import effects, styles  # noqa: E402
from hitsync.config import RenderParams, SyncParams  # noqa: E402
from hitsync.effects import EffectRange, Look  # noqa: E402
from hitsync.models import Hit, Markers  # noqa: E402
from hitsync.sync_engine import build_schedule  # noqa: E402
from hitsync.text_overlay import TextItem, animation_state, draw_caption  # noqa: E402

PERIOD = 0.6
GRID = 0.2 + PERIOD * np.arange(300)
DROP = GRID[20]


def schedule(**sync):
    m = Markers()
    m.set_beats(GRID)
    rng = np.random.default_rng(0)
    hits = list(20 + 0.6 * np.arange(10) + rng.uniform(-0.02, 0.02, 10)) + \
        list(40 + 0.6 * np.arange(10) + rng.uniform(-0.02, 0.02, 10))
    m.set_hits([Hit(t) for t in hits], 1.2, 0.3)
    p = SyncParams(drop_time=DROP, intro_end=20.0, intro_start=17.0, min_combo_len=5,
                   letterbox_mode="off", intro_flash=False, **sync)
    return build_schedule(m, p, 70, 170, plan=[(g, False) for g in m.combos()],
                          downbeats=GRID[::4]), m


def frame():
    rng = np.random.default_rng(1)
    f = (rng.random((90, 160, 3)) * 200 + 20).astype(np.uint8)
    f[30:60, 60:100] = (40, 40, 220)
    return f


class LookTests(unittest.TestCase):
    def test_clean_look_changes_nothing(self):
        s, _ = schedule()
        f = frame()
        look = Look(s, RenderParams())
        for t in (s.start + 0.5, s.placed_hits()[3].out_t, s.end - 0.1):
            out = look.finish(f, t, None)
            np.testing.assert_array_equal(out, f)

    def test_never_draws_on_the_source_frame(self):
        s, _ = schedule()
        f = frame()
        keep = f.copy()
        look = Look(s, RenderParams(filter="vibrant", vignette=0.4, hit_zoom=0.08, hit_flash=0.5))
        look.finish(f, s.placed_hits()[2].out_t + 0.01, (1.0, 2.39))
        np.testing.assert_array_equal(f, keep)

    def test_hit_effects_only_on_ticked_hits(self):
        s, _ = schedule()
        placed = s.placed_hits()
        r = RenderParams(hit_zoom=0.1, hit_flash=0.6, hit_fx_default=True)
        off = {placed[4].hit_index: False}
        look = Look(s, r, hits_fx=off)
        f = frame()
        t_on, t_off = placed[3].out_t + 0.01, placed[4].out_t + 0.01
        self.assertFalse(np.array_equal(look.finish(f, t_on, None), f))
        # the unticked hit gets nothing (the previous hit's effect has died out)
        np.testing.assert_array_equal(look.finish(f, t_off, None), f)
        # default off, one hit ticked on
        look = Look(s, RenderParams(hit_zoom=0.1, hit_fx_default=False),
                    hits_fx={placed[4].hit_index: True})
        self.assertFalse(np.array_equal(look.finish(f, t_off, None), f))
        np.testing.assert_array_equal(look.finish(f, t_on, None), f)

    def test_effect_range_applies_only_inside(self):
        s, _ = schedule()
        a, b = GRID[24], GRID[32]
        look = Look(s, RenderParams(), ranges=[EffectRange("flashy", a, b)])
        f = frame()
        inside = look.finish(f, GRID[28] + 0.01, None)
        self.assertFalse(np.array_equal(inside, f))
        np.testing.assert_array_equal(look.finish(f, b + 0.5, None), f)
        np.testing.assert_array_equal(look.finish(f, a - 0.5, None), f)
        for kind in effects.RANGE_KINDS:
            out = Look(s, RenderParams(), ranges=[EffectRange(kind, a, b)]).finish(
                f, GRID[28] + 0.01, None)
            self.assertEqual(out.shape, f.shape, kind)

    def test_deterministic(self):
        s, _ = schedule()
        r = RenderParams(**styles.STYLES["hype"]["render"])
        f = frame()
        t = s.placed_hits()[5].out_t + 0.03
        a = Look(s, r).finish(f, t, (0.5, 2.39))
        b = Look(s, r).finish(f, t, (0.5, 2.39))
        np.testing.assert_array_equal(a, b)

    def test_start_and_end_fades(self):
        s, _ = schedule()
        f = frame()
        look = Look(s, RenderParams(fade_in="black", fade_in_len=1.0,
                                    fade_out="white", fade_out_len=1.0))
        self.assertLess(look.finish(f, s.start, None).mean(), 2)
        self.assertGreater(look.finish(f, s.end - 1e-3, None).mean(), 250)
        mid = look.finish(f, (s.start + s.end) / 2, None)
        np.testing.assert_array_equal(mid, f)

    def test_transitions_at_cuts(self):
        f = frame()
        for tr in ("zoom", "whip", "dip"):
            s, _ = schedule(transition=tr)
            self.assertTrue(s.cuts, tr)
            c = s.cuts[0]
            out = Look(s, RenderParams()).finish(f, c + 0.02, None)
            self.assertFalse(np.array_equal(out, f), tr)
            np.testing.assert_array_equal(Look(s, RenderParams()).finish(f, c + 1.0, None), f)

    def test_filters(self):
        for name in effects.FILTERS:
            lut, sat = effects.build_filter(name, 0.0)
            np.testing.assert_array_equal(lut[:, 0, 0], np.arange(256))
            self.assertAlmostEqual(sat, 1.0)
        lut, sat = effects.build_filter("bw", 1.0)
        self.assertEqual(sat, 0.0)
        out = effects.saturate(frame(), 0.0)
        self.assertLessEqual(np.abs(out.astype(int)[..., 0] - out[..., 2]).max(), 1)

    def test_motion_blur_samples(self):
        s, _ = schedule()
        look = Look(s, RenderParams(motion_blur=0.5))
        t = s.placed_hits()[3].out_t + 0.2
        self.assertGreater(look.blur_samples(t, 1.0), 1)
        self.assertEqual(look.blur_samples(t, 0.4), 1)          # never in slow-mo
        self.assertEqual(Look(s, RenderParams()).blur_samples(t, 1.0), 1)


class CaptionTests(unittest.TestCase):
    def test_pop_animation(self):
        it = TextItem("GG", 10.0, 12.0, animation="pop")
        self.assertFalse(animation_state(it, 9.99)[0])
        v, a0, s0, _, _ = animation_state(it, 10.01)
        v, a1, s1, _, _ = animation_state(it, 11.0)
        v, a2, _, _, _ = animation_state(it, 11.99)
        self.assertLess(a0, 0.2)
        self.assertEqual(a1, 1.0)
        self.assertAlmostEqual(s1, 1.0, places=3)
        self.assertLess(s0, 0.7)
        self.assertLess(a2, 0.1)
        self.assertFalse(animation_state(it, 12.0)[0])

    def test_typewriter_reveals_letters(self):
        it = TextItem("COMBO", 0.0, 3.0, animation="typewriter")
        counts = [animation_state(it, t)[4] for t in (0.01, 0.1, 0.2, 1.0)]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[-1], 5)
        self.assertLess(counts[0], 5)

    def test_positions_and_inside_frame(self):
        for pos, (lo, hi) in (("top", (0, 0.35)), ("bottom", (0.65, 1.0)), ("center", (0.3, 0.7))):
            f = np.zeros((360, 640, 3), np.uint8)
            draw_caption(f, TextItem("INSANE COMBO", 0, 2, pos), 1.0)
            ys = np.nonzero(f.max(axis=(1, 2)) > 0)[0]
            self.assertTrue(len(ys), pos)
            self.assertGreaterEqual(ys.min() / 360, lo - 0.02, pos)
            self.assertLessEqual(ys.max() / 360, hi + 0.02, pos)
        # very long text wraps instead of running off the screen
        f = np.zeros((360, 640, 3), np.uint8)
        draw_caption(f, TextItem("THIS IS A VERY LONG CAPTION " * 3, 0, 2, "center"), 1.0)
        xs = np.nonzero(f.max(axis=(0, 2)) > 0)[0]
        self.assertGreater(xs.min(), 0)
        self.assertLess(xs.max(), 639)

    def test_outline_is_drawn(self):
        f = np.full((360, 640, 3), 128, np.uint8)
        draw_caption(f, TextItem("GG", 0, 2, "center", "youtuber", "none"), 1.0)
        self.assertGreater((f.min(axis=2) > 240).sum(), 50)      # white fill
        self.assertGreater((f.max(axis=2) < 20).sum(), 50)       # black outline

    def test_caption_through_look(self):
        s, _ = schedule()
        t = s.start + 2.0
        look = Look(s, RenderParams(), texts=[TextItem("HELLO", t - 1, t + 1, animation="none")])
        f = frame()
        self.assertFalse(np.array_equal(look.finish(f, t, None), f))
        np.testing.assert_array_equal(look.finish(f, t + 2, None), f)


class StyleTests(unittest.TestCase):
    def test_new_project_uses_the_default_style(self):
        from hitsync.project import Project

        p = Project.new()
        self.assertEqual(p.render_params.style, styles.DEFAULT)
        self.assertEqual(p.render_params.filter, styles.STYLES[styles.DEFAULT]["render"]["filter"])
        styles.apply_style(p, "cinematic")
        self.assertEqual(p.sync.letterbox_mode, "first combo")
        self.assertEqual(p.render_params.style, "cinematic")
        self.assertIn("hit_zoom", styles.style_keys())

    def test_overlays_saved_with_the_project(self):
        import tempfile

        from hitsync.project import Project

        p = Project.new()
        p.markers.set_beats(GRID)
        p.add_text("GG", 10.03, bars=2, position="top")
        p.add_effect("flashy", 12.0, 15.0)
        path = os.path.join(tempfile.mkdtemp(), "p.json")
        p.save(path)
        q = Project.load(path)
        self.assertEqual(q.texts[0].text, "GG")
        self.assertEqual(q.texts[0].position, "top")
        self.assertAlmostEqual(q.texts[0].end - q.texts[0].start, 8 * PERIOD, places=6)
        self.assertEqual(q.effects[0].kind, "flashy")
        self.assertIn(round(q.effects[0].start, 6), np.round(GRID, 6))


if __name__ == "__main__":
    unittest.main()
