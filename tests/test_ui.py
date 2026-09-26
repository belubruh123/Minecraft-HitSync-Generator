"""The Qt app, driven like a user (offscreen): drop files, pick styles and
combos, mark the starts, add text and effects, play and export."""
import os
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception:                                   # PySide6 / its system libs missing
    HAVE_QT = False

from hitsync.ffmpeg_utils import find_ffmpeg  # noqa: E402


@unittest.skipUnless(HAVE_QT and find_ffmpeg(), "Qt or ffmpeg unavailable")
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests import synth, synth_music as sm

        d = tempfile.mkdtemp()
        os.environ["HITSYNC_CACHE"] = os.path.join(d, "cache")
        rng = np.random.default_rng(3)
        hits = []
        for start, n in ((6, 11), (16, 12), (27, 10)):
            hits += list(start + 0.55 * np.arange(n) + rng.uniform(-0.03, 0.03, n))
        old = synth.HITS
        synth.HITS = hits
        cls.video = os.path.join(d, "pvp.mp4")
        synth.write_video(cls.video, duration=36.0, size=(320, 180))
        synth.HITS = old
        cls.song = os.path.join(d, "song.wav")
        y, cls.info = sm.shape_like(duration=70.0)
        sm.write_wav(cls.song, y)
        cls.dir = d
        from hitsync.ui import theme
        from hitsync.ui.main_window import MainWindow

        cls.app = QApplication.instance() or QApplication([])
        theme.apply(cls.app)
        cls.w = MainWindow()
        cls.w.resize(1300, 850)
        cls.w.show()
        cls.w.drop_files([cls.video, cls.song])
        assert cls.wait_idle(), "analysis did not finish"

    @classmethod
    def tearDownClass(cls):
        cls.w.close()

    @classmethod
    def pump(cls, sec=0.1):
        end = time.time() + sec
        while time.time() < end:
            cls.app.processEvents(QEventLoop.AllEvents, 20)
            time.sleep(0.005)

    @classmethod
    def wait_idle(cls, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            cls.pump(0.05)
            if not cls.w.busy:
                cls.pump(0.2)
                return True
        return False

    def test_1_drop_analyses_everything(self):
        p = self.w.project
        self.assertAlmostEqual(p.grid_bpm, 96.0, delta=0.05)
        self.assertEqual(len(p.real_combos()), 3)
        self.assertAlmostEqual(p.sync.drop_time, self.info["kick_in"], delta=0.05)
        self.assertIsNotNone(p.schedule)
        self.assertIsNotNone(self.w.view.img)                 # a poster frame, not black
        chips = [self.w.start_chips.itemAt(i).widget() for i in range(self.w.start_chips.count())]
        self.assertTrue(any(c.isChecked() for c in chips))
        self.assertEqual(self.w.combos.list.count(), 3)

    def test_2_styles_and_quick_settings(self):
        w = self.w
        w.set_style("hype")
        self.assertEqual(w.project.render_params.filter, "punchy")
        w.set_style("montage")
        w.bph.changed.emit("2")
        self.pump(0.3)
        self.assertEqual(w.project.sync.combo_spacing, "2")
        self.assertEqual(w.project.schedule.beats_per_hit, 2.0)
        w.bph.changed.emit("auto")
        w.hit_vol.set(0.5, emit=True)
        self.assertAlmostEqual(w.project.render_params.hit_volume, 0.5)
        # editing a style setting makes the style custom
        w.fx_panel._push("render", "hit_zoom", 0.1)
        self.assertEqual(w.project.render_params.style, "custom")
        w.set_style("montage")

    def test_3_combo_picker(self):
        w = self.w
        choices = list(w.combos.choices)
        choices[1].enabled = False
        choices[0].lead_in = True          # 2nd in the new order (the 1st has the intro)
        w.combos.plan_changed.emit([choices[2], choices[0], choices[1]])
        self.pump(0.4)
        s = w.project.schedule
        self.assertEqual(s.combos_kept, 2)
        self.assertEqual(len(s.leadins), 1)
        first = s.placed_hits()[0]
        self.assertAlmostEqual(first.src_t, choices[2].key, places=4)
        # per-hit effects from the chips
        row = w.combos.list.itemWidget(w.combos.list.item(0))
        row._quick(lambda k, n: k == 0)
        self.pump(0.2)
        g = w.project.engine_plan()[0][0]
        self.assertTrue(w.project.markers.hits[g[0]].fx)
        self.assertFalse(w.project.markers.hits[g[1]].fx)
        w._reset_plan()
        self.pump(0.3)
        self.assertEqual(w.project.schedule.combos_kept, 3)

    def test_4_mark_music_start_by_listening(self):
        w = self.w
        w.set_mode("song")
        self.pump(0.2)
        target = self.info["kick_in"] + 4 * 2.5 + 0.07      # 4 bars later, a bit off
        w.seek(target)
        w.mark()
        self.pump(0.3)
        self.assertAlmostEqual(w.project.sync.drop_time, target - 0.07, delta=0.02)
        w._set_start(0, self.info["kick_in"])
        self.pump(0.2)
        w.set_mode("montage")

    def test_5_mark_combat_start_while_watching(self):
        w = self.w
        w.set_mode("gameplay")
        self.pump(0.2)
        second = w.project.real_combos()[1]
        t = w.project.markers.hits[second[0]].t - 0.2
        w.seek(t)
        w.mark()
        self.pump(0.3)
        plan = w.project.engine_plan()
        self.assertEqual(plan[0][0], second)
        self.assertAlmostEqual(w.project.schedule.segments[0].src_end,
                               w.project.markers.hits[second[0]].t, places=4)
        w._reset_plan()
        w.set_mode("montage")
        self.pump(0.2)

    def test_6_text_and_effect_ranges(self):
        w = self.w
        w.seek(w.project.schedule.start + 4.0)
        w._add_text()
        self.pump(0.2)
        self.assertEqual(len(w.project.texts), 1)
        row = w.texts.rows[0]
        row.edit.setText("GG EZ")
        self.pump(0.2)
        self.assertEqual(w.project.texts[0].text, "GG EZ")
        w._add_range("flashy", "combo")
        self.pump(0.2)
        self.assertEqual(w.project.effects[0].kind, "flashy")
        a, b, _ = w.project.schedule.combo_spans[0]
        self.assertLessEqual(w.project.effects[0].start, a + 1e-6)
        # the timeline shows them; deleting from the timeline removes them
        w.timeline.selected = ("text", 0)
        w.timeline.delete_selected()
        self.pump(0.2)
        self.assertEqual(len(w.project.texts), 0)
        w.project.effects.clear()
        w._overlays_changed()

    def test_7_playback_delivers_frames(self):
        w = self.w
        w.set_mode("montage")
        w.engine.clock.start = (lambda start: (lambda a, pos, t0=0.0: start(None, pos, t0)))(
            w.engine.clock.start)                            # silent clock on CI
        w.seek(w.project.schedule.start + 1.0)
        shown = []
        orig = w._show_frame
        w._show_frame = lambda f: (shown.append(f.shape), orig(f))
        w.toggle_play()
        self.pump(1.5)
        w.toggle_play()
        w._show_frame = orig
        self.assertGreater(len(shown), 20)
        self.assertGreater(w.engine.time(), w.project.schedule.start + 2.0)

    def test_8_export(self):
        from unittest import mock

        from PySide6.QtWidgets import QFileDialog, QMessageBox

        w = self.w
        out = os.path.join(self.dir, "montage.mp4")
        with mock.patch.object(QFileDialog, "getSaveFileName", return_value=(out, "")), \
                mock.patch.object(QMessageBox, "exec", return_value=0):
            w.export()
            self.assertTrue(self.wait_idle(300))
        self.assertTrue(os.path.isfile(out))
        import cv2

        cap = cv2.VideoCapture(out)
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.assertAlmostEqual(n / fps, w.project.schedule.duration, delta=0.2)

    def test_9_save_and_open(self):
        w = self.w
        path = os.path.join(self.dir, "p.hitsync.json")
        w.project_path = path
        w.save_project()
        w.open_project(path)
        self.assertTrue(self.wait_idle())
        self.assertIsNotNone(w.project.schedule)
        self.assertEqual(w.combos.list.count(), 3)


if __name__ == "__main__":
    unittest.main()
