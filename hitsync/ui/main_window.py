"""The Hit-Sync window: drop a video and a song, get a beat-synced montage.

Layout (top to bottom):
  smart bar ..... drop cards (video / music), style, Export, Advanced
  info row ...... tempo check, "music starts" choices, combo count, progress
  centre ........ big preview (Montage / Song / Gameplay) + side panels
  quick row ..... hit volume, music volume, beats per hit, timeline toggle
  timeline ...... (hidden until asked for)
"""
from __future__ import annotations

import os
import threading
import time
import traceback

import numpy as np
from PySide6.QtCore import QObject, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSlider, QSplitter,
                               QStackedWidget, QTabWidget, QVBoxLayout, QWidget)

from .. import styles
from ..project import Project, default_output_path
from . import theme
from .advanced import AdvancedDialog
from .panels import CombosPanel, EffectsPanel, MusicPanel, TextPanel
from .timeline import Timeline
from .video_view import FullscreenPlayer, SongView, VideoView
from .widgets import (DropCard, ElidedLabel, FlowLayout, Segmented, ValueSlider, button,
                      install_wheel_guard, kind_of, label)

VIDEO_FILTER = "Video (*.mp4 *.mkv *.mov *.avi *.webm *.flv *.m4v);;All files (*)"
AUDIO_FILTER = "Audio (*.mp3 *.wav *.ogg *.flac *.m4a *.aac *.opus);;All files (*)"


class Bridge(QObject):
    """Worker threads -> GUI thread."""
    progress = Signal(str, str, float)
    done = Signal(str, object)
    failed = Signal(str, str)
    thumb = Signal(float, object)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        from ..preview_engine import PreviewEngine

        self.setWindowTitle("Hit-Sync · Minecraft PvP montage maker")
        self.setAcceptDrops(True)
        self.settings = QSettings("HitSync", "HitSync")
        self.project = Project.new()
        self.project_path: str | None = None
        self.engine = PreviewEngine(fps=30)
        self.engine.on_error = lambda e: self.bridge.failed.emit("preview", str(e))
        self.bridge = Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.done.connect(self._on_done)
        self.bridge.failed.connect(self._on_failed)
        self.bridge.thumb.connect(self._on_thumb)
        self.busy: dict = {}
        self.pending_analyze = False
        self.cancel = threading.Event()
        self.mode = "montage"
        self.song_index = 0
        self.taps: list = []
        self.soundtrack = None             # (key, audio)
        self.fs: FullscreenPlayer | None = None
        self._last_ui = 0.0

        self._build()
        self._menus()
        self._shortcuts()
        self.frame_timer = QTimer(self, interval=8, timeout=self._tick)
        self.frame_timer.start()
        self.recalc_timer = QTimer(self, singleShot=True, interval=120,
                                   timeout=lambda: self.recalc(now=True))
        self.sound_timer = QTimer(self, singleShot=True, interval=250,
                                  timeout=self._build_soundtrack)
        self.resize_timer = QTimer(self, singleShot=True, interval=150, timeout=self._fit_preview)
        geo = self.settings.value("geometry")
        if geo is not None:
            self.restoreGeometry(geo)
        else:
            self.resize(1440, 900)
        self._pull_all()
        self._update_mode_ui()
        self._check_ffmpeg()

    # ================================================================ layout
    def _build(self):
        # the whole window scrolls when the screen is too small for it (no
        # scrollbars otherwise); the wheel over a slider scrolls the panels
        install_wheel_guard(QApplication.instance())
        root = QWidget()
        self.page = QScrollArea()
        self.page.setWidgetResizable(True)
        self.page.setFrameShape(QFrame.NoFrame)
        self.page.setWidget(root)
        self.setCentralWidget(self.page)
        v = QVBoxLayout(root)
        v.setContentsMargins(12, 10, 12, 6)
        v.setSpacing(8)

        # --- smart bar
        bar = QHBoxLayout()
        self.video_card = DropCard("🎮", "Gameplay video", "Drop a recording here, or click")
        self.video_card.clicked.connect(self.pick_video)
        self.video_card.dropped.connect(self.drop_files)
        self.music_card = DropCard("🎵", "Music", "Drop a song here (more songs play back to back)")
        self.music_card.clicked.connect(self.pick_music)
        self.music_card.dropped.connect(self.drop_files)
        bar.addWidget(self.video_card, 3)
        bar.addWidget(self.music_card, 3)
        style_box = QVBoxLayout()
        style_box.setSpacing(2)
        style_box.addWidget(label("STYLE", h2=True))
        self.style_seg = Segmented([(k, styles.LABELS[k]) for k in styles.STYLES])
        self.style_seg.changed.connect(self.set_style)
        style_box.addWidget(self.style_seg)
        bar.addLayout(style_box)
        self.export_btn = button("⬇  Export video", f"Save the montage as an .mp4 ({theme.MOD}E)",
                                 primary=True, slot=self.export)
        self.export_btn.setMinimumHeight(44)
        bar.addWidget(self.export_btn)
        bar.addWidget(button("⚙", "Advanced settings", slot=self.show_advanced))
        v.addLayout(bar)

        # --- info row: tempo check + music start choices + combos
        info = QHBoxLayout()
        info.setSpacing(6)
        self.tempo_chip = button("Tempo –", "Beat looks off? Fix it in the Song tab", chip=True,
                                 slot=lambda: self.set_mode("song"))
        info.addWidget(self.tempo_chip)
        info.addWidget(label("Music starts:", muted=True))
        chips = QWidget()
        self.start_chips = FlowLayout(chips, spacing=6)      # wraps when narrow
        info.addWidget(chips, 3)
        mark = button("🎧 Choose by listening", "Play the song and click where it should start",
                      flat=True, slot=lambda: self.set_mode("song"))
        info.addWidget(mark)
        self.combo_lbl = ElidedLabel("", muted=True)
        self.combo_lbl.setAlignment(Qt.AlignRight)
        info.addWidget(self.combo_lbl, 2)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(160)
        self.progress.setVisible(False)
        info.addWidget(self.progress)
        self.cancel_btn = button("Cancel", flat=True, slot=self._cancel)
        self.cancel_btn.setVisible(False)
        info.addWidget(self.cancel_btn)
        v.addLayout(info)

        # --- centre: preview + side panels
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        modes = QHBoxLayout()
        self.mode_seg = Segmented([("montage", "▶ Montage"), ("song", "🎵 Song"),
                                   ("gameplay", "🎮 Gameplay")])
        self.mode_seg.changed.connect(self.set_mode)
        modes.addWidget(self.mode_seg)
        modes.addStretch(1)
        self.mode_hint = ElidedLabel("", muted=True)
        self.mode_hint.setAlignment(Qt.AlignRight)
        modes.addWidget(self.mode_hint, 1)
        lv.addLayout(modes)
        self.stack = QStackedWidget()
        self.view = VideoView()
        self.view.double_clicked.connect(self.toggle_fullscreen)
        self.view.clicked.connect(self.toggle_play)
        self.song_view = SongView()
        self.song_view.seek.connect(self.seek)
        self.stack.addWidget(self.view)
        self.stack.addWidget(self.song_view)
        lv.addWidget(self.stack, 1)
        lv.addWidget(self._transport())
        self.actions = QStackedWidget()
        self.actions.addWidget(self._montage_actions())
        self.actions.addWidget(self._song_actions())
        self.actions.addWidget(self._gameplay_actions())
        lv.addWidget(self.actions)
        split.addWidget(left)

        self.side = QTabWidget()
        self.combos = CombosPanel()
        self.combos.plan_changed.connect(self._plan_changed)
        self.combos.fx_changed.connect(self._fx_changed)
        self.combos.auto_pick.connect(self._auto_pick)
        self.combos.reset.connect(self._reset_plan)
        self.combos.select.connect(self._combo_selected)
        self.texts = TextPanel()
        self.texts.add.connect(self._add_text)
        self.texts.changed.connect(self._overlays_changed)
        self.texts.seek.connect(self.seek)
        self.texts.set_time.connect(self._text_time)
        self.music_panel = MusicPanel()
        self.music_panel.add.connect(self.pick_music)
        self.music_panel.select.connect(self._song_selected)
        self.music_panel.move.connect(self._move_song)
        self.music_panel.order.connect(self._song_order)
        self.music_panel.remove.connect(self._remove_song)
        self.music_panel.reset_cut.connect(self._reset_cut)
        self.fx_panel = EffectsPanel(lambda: self.project)
        self.fx_panel.changed.connect(self._look_changed)
        self.fx_panel.add_range.connect(self._add_range)
        self.fx_panel.seek.connect(self.seek)
        self.side.addTab(self.combos, "Combos")
        self.side.addTab(self.music_panel, "Music")
        self.side.addTab(self.texts, "Text")
        self.side.addTab(self.fx_panel, "Effects")
        self.side.setMinimumWidth(300)
        split.addWidget(self.side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([1000, 400])
        v.addWidget(split, 1)

        # --- quick settings
        q = QFrame()
        q.setProperty("panel", True)
        ql = QHBoxLayout(q)
        ql.setContentsMargins(12, 6, 12, 6)
        self.hit_vol = ValueSlider("Hit volume", 0, 1.5, 0.8, percent=True, label_width=80)
        self.hit_vol.changed.connect(lambda x: self._audio_param("hit_volume", x))
        self.music_vol = ValueSlider("Music volume", 0, 1.5, 1.0, percent=True, label_width=90)
        self.music_vol.changed.connect(lambda x: self._audio_param("music_volume", x))
        ql.addWidget(self.hit_vol, 2)
        ql.addSpacing(16)
        ql.addWidget(self.music_vol, 2)
        ql.addSpacing(16)
        ql.addWidget(QLabel("Beats per hit"))
        self.bph = Segmented([("auto", "Auto"), ("0.5", "½"), ("1", "1"), ("2", "2")])
        self.bph.changed.connect(self._bph_changed)
        ql.addWidget(self.bph)
        ql.addStretch(1)
        self.tl_btn = button("Timeline ▾", "Show the detailed timeline (for fine edits)",
                             flat=True, checkable=True)
        self.tl_btn.toggled.connect(self._toggle_timeline)
        ql.addWidget(self.tl_btn)
        v.addWidget(q)

        # --- timeline
        self.timeline = Timeline()
        self.timeline.project = self.project
        self.timeline.changed.connect(self._markers_changed)
        self.timeline.clicked.connect(self._timeline_click)
        self.timeline.overlay_changed.connect(self._overlays_changed)
        self.timeline.selected_changed.connect(self._timeline_selected)
        self.timeline.setVisible(False)
        v.addWidget(self.timeline)
        self.timeline.hbar.setVisible(False)
        v.addWidget(self.timeline.hbar)
        self.statusBar().showMessage("Drop a gameplay video and a song to begin.")
        self.advanced = AdvancedDialog(lambda: self.project, self)
        self.advanced.changed.connect(self._advanced_changed)
        self.advanced.redetect.connect(self._redetect)
        self.advanced.custom_sound.connect(self._pick_sound)

    def _transport(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(button("⏮", "Back to the start (Home)", flat=True,
                           slot=lambda: self.seek(self._program_start())))
        self.play_btn = button("▶", "Play / pause (Space)", slot=self.toggle_play)
        self.play_btn.setFixedWidth(48)
        h.addWidget(self.play_btn)
        self.time_lbl = label("0:00.00 / 0:00.00")
        self.time_lbl.setStyleSheet("font-family: monospace;")
        h.addWidget(self.time_lbl)
        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 10000)
        self.scrub.sliderMoved.connect(self._scrubbed)
        h.addWidget(self.scrub, 1)
        self.click_cb = QCheckBox("Beat click")
        self.click_cb.setToolTip("Hear a click on every beat, to check the beat grid")
        self.click_cb.toggled.connect(self._click_toggled)
        h.addWidget(self.click_cb)
        h.addWidget(button("⛶", "Fullscreen (F)", slot=self.toggle_fullscreen))
        return w

    def _montage_actions(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        self.summary = ElidedLabel("", muted=True)
        h.addWidget(self.summary, 1)
        return w

    def _song_actions(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)
        r1 = QHBoxLayout()
        self.song_pick = QComboBox()
        self.song_pick.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.song_pick.setMinimumContentsLength(10)
        self.song_pick.currentIndexChanged.connect(self._song_picked)
        r1.addWidget(self.song_pick, 1)
        self.mark_music = button("✔ Music starts here  (M)",
                                 "Play the song, and click this where the montage's first "
                                 "hit should land", primary=True, slot=self.mark)
        r1.addWidget(self.mark_music)
        self.cut_btn = button("✂ Switch to next song here  (C)",
                              "Play the song, and click this where it should stop; the next "
                              "song takes over on this bar", slot=self.cut)
        r1.addWidget(self.cut_btn)
        v.addLayout(r1)
        fixes = QWidget()
        r2 = FlowLayout(fixes, spacing=4)                   # wraps when narrow
        self.grid_lbl = label("", muted=True)
        r2.addWidget(self.grid_lbl)
        r2.addWidget(label("  Beat looks off?", muted=True))
        for text, tip, fn in (
                ("Tap (T)", "Tap along with the beat, 8 times or more", self.tap),
                ("½×", "Half the tempo", lambda: self._grid_fix("half")),
                ("2×", "Double the tempo", lambda: self._grid_fix("double")),
                ("Shift ½ beat", "The clicks sit between the beats", lambda: self._grid_fix("shift")),
                ("◀ 10 ms", "Clicks a little late", lambda: self._grid_fix(-10)),
                ("10 ms ▶", "Clicks a little early", lambda: self._grid_fix(10)),
                ("Reset", "Back to what was detected", lambda: self._grid_fix("reset"))):
            r2.addWidget(button(text, tip, flat=True, slot=fn))
        v.addWidget(fixes)
        return w

    def _gameplay_actions(self):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(label("Play your recording and click where the action should start. "
                          "The footage before it becomes the slow-mo intro.", muted=True,
                          wrap=True), 1)
        h.addWidget(button("✔ Combat starts here  (M)", primary=True, slot=self.mark))
        return w

    def _menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        for text, key, fn in (("Choose gameplay video…", None, self.pick_video),
                              ("Add music…", None, self.pick_music),
                              (None, None, None),
                              ("Open project…", QKeySequence.Open, self.open_project),
                              ("Save project", QKeySequence.Save, self.save_project),
                              ("Save project as…", QKeySequence.SaveAs,
                               lambda: self.save_project(True)),
                              (None, None, None),
                              ("Export video…", QKeySequence("Ctrl+E"), self.export),
                              (None, None, None),
                              ("Quit", QKeySequence.Quit, self.close)):
            if text is None:
                fm.addSeparator()
                continue
            a = QAction(text, self)
            if key is not None:
                a.setShortcut(key)
            a.triggered.connect(fn)
            fm.addAction(a)
        vm = mb.addMenu("&View")
        a = QAction("Fullscreen preview", self, shortcut=QKeySequence("F"))
        a.triggered.connect(self.toggle_fullscreen)
        vm.addAction(a)
        a = QAction("Show timeline", self, checkable=True)
        a.toggled.connect(self.tl_btn.setChecked)
        vm.addAction(a)
        a = QAction("Advanced settings…", self, shortcut=QKeySequence.Preferences)
        a.triggered.connect(self.show_advanced)
        vm.addAction(a)
        hm = mb.addMenu("&Help")
        a = QAction("Keyboard shortcuts", self)
        a.triggered.connect(self._help)
        hm.addAction(a)

    def _shortcuts(self):
        for key, fn in ((Qt.Key_Space, self.toggle_play), (Qt.Key_M, self.mark),
                        (Qt.Key_T, self.tap), (Qt.Key_C, self.cut),
                        (Qt.Key_Left, lambda: self.step(-1)),
                        (Qt.Key_Right, lambda: self.step(1)),
                        (Qt.Key_Home, lambda: self.seek(self._program_start()))):
            QShortcut(QKeySequence(key), self, activated=fn)

    # ============================================================= files
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        self.drop_files(paths)

    def drop_files(self, paths):
        videos = [p for p in paths if kind_of(p) == "video"]
        audios = [p for p in paths if kind_of(p) == "audio"]
        projects = [p for p in paths if p.lower().endswith(".json")]
        if projects:
            return self.open_project(projects[0])
        if not videos and not audios:
            QMessageBox.information(self, "Hit-Sync", "Drop a video (mp4, mkv, mov…) or a song "
                                                      "(mp3, wav, ogg, flac…).")
            return
        if videos:
            self.project.video_path = videos[0]
            if not self.project.output_path:
                self.project.output_path = default_output_path(videos[0])
        for a in audios:
            if not self.project.music_path or not os.path.isfile(self.project.music_path):
                self.project.music_path = a
            elif a not in self.project.music_paths:
                self.project.add_song(a)
        self._refresh_cards()
        self.analyze()

    def pick_video(self):
        p, _ = QFileDialog.getOpenFileName(self, "Gameplay video", self._dir(), VIDEO_FILTER)
        if p:
            self._remember_dir(p)
            self.drop_files([p])

    def pick_music(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Music", self._dir(), AUDIO_FILTER)
        if paths:
            self._remember_dir(paths[0])
            self.drop_files(paths)

    def _dir(self):
        return self.settings.value("last_dir", os.path.expanduser("~"))

    def _remember_dir(self, path):
        self.settings.setValue("last_dir", os.path.dirname(path))

    def _refresh_cards(self):
        p = self.project
        if p.video_path:
            name = os.path.basename(p.video_path)
            st = (f"{theme.fmt_short(p.video_duration)} · {len(p.markers.hits)} hits · "
                  f"{len(p.real_combos())} combos" if p.video else "Analysing…")
            self.video_card.set_state(name, st)
        if p.music_path:
            names = [os.path.basename(x) for x in p.music_paths]
            title = names[0] if len(names) == 1 else f"{names[0]}  +{len(names) - 1} more"
            st = (f"{p.grid_bpm:.1f} BPM · {theme.fmt_short(p.music_duration)}"
                  if p.audio else "Analysing…")
            self.music_card.set_state(title, st)

    # ========================================================== analysis
    def analyze(self):
        if "analyze" in self.busy:
            self.pending_analyze = True
            return
        p = self.project
        if not (p.video_path or p.music_path):
            return
        self.engine.pause()
        self.cancel.clear()

        def work(progress):
            return p.analyze(progress=progress, cancel=self.cancel)

        self._run("analyze", work, "Analysing…")

    def _run(self, name, fn, message):
        self.busy[name] = True
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.cancel_btn.setVisible(name in ("analyze", "export"))
        self.statusBar().showMessage(message)

        def progress(msg, frac):
            self.bridge.progress.emit(name, msg, float(frac))

        def worker():
            try:
                self.bridge.done.emit(name, fn(progress))
            except Exception as exc:                      # noqa: BLE001
                traceback.print_exc()
                self.bridge.failed.emit(name, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, name, msg, frac):
        self.progress.setValue(int(np.clip(frac, 0, 1) * 100))
        self.statusBar().showMessage(msg)
        if name == "analyze":
            card = self.video_card if "video" in msg else self.music_card
            card.set_state(progress=frac)

    def _finish(self, name):
        self.busy.pop(name, None)
        if not self.busy:
            self.progress.setVisible(False)
            self.cancel_btn.setVisible(False)
        for c in (self.video_card, self.music_card):
            c.set_state(progress=0.0)

    def _on_done(self, name, result):
        self._finish(name)
        if name == "analyze":
            self._analysis_done()
            if self.pending_analyze:
                self.pending_analyze = False
                self.analyze()
        elif name == "export":
            self._export_done(result)
        elif name == "soundtrack":
            key, audio = result
            self.soundtrack = (key, audio)
            self._attach_audio()

    def _on_failed(self, name, message):
        self._finish(name)
        if name == "preview":
            self.statusBar().showMessage(f"Preview problem: {message}")
            return
        if name == "soundtrack":
            self.statusBar().showMessage(f"No sound in the preview: {message}")
            return
        if self.cancel.is_set():
            self.statusBar().showMessage("Cancelled")
            return
        QMessageBox.warning(self, "Hit-Sync", f"{name.capitalize()} failed:\n{message}")
        self.statusBar().showMessage(f"{name.capitalize()} failed: {message}")

    def _cancel(self):
        self.cancel.set()

    def _analysis_done(self):
        from ..video_analysis import hit_score

        p = self.project
        if p.video is not None:
            self.timeline.set_score(p.video.times, hit_score(p.video, p.detect))
        self._refresh_cards()
        self._pull_all()
        self.recalc(now=True)
        self.timeline.zoom_fit()
        self._load_thumbs()
        if p.audio is not None and p.video is not None:
            self.statusBar().showMessage(
                f"Ready: {p.grid_bpm:.1f} BPM, {len(p.real_combos())} combos. "
                "Press Space to play, F for fullscreen.")
            self.mode = "montage"
            self._update_mode_ui()
            self._load_program(keep=False)          # from the start, poster on the drop
        elif p.video is not None:
            self.statusBar().showMessage("Video ready. Now drop a song.")
        elif p.audio is not None:
            self.statusBar().showMessage("Song ready. Now drop your gameplay video.")

    def _load_thumbs(self):
        p = self.project
        if p.video is None:
            return
        keys = [c.key for c, _ in p.plan_entries()]
        path, off = p.video_path, p.video.pts_offset or 0.0

        def work():
            from ..video_analysis import read_frame

            for k in keys:
                try:
                    f = read_frame(path, k, (176, 100), off)
                except Exception:                         # noqa: BLE001
                    f = None
                if f is not None:
                    self.bridge.thumb.emit(k, f)

        threading.Thread(target=work, daemon=True).start()

    def _on_thumb(self, key, frame):
        h, w = frame.shape[:2]
        img = QImage(np.ascontiguousarray(frame).data, w, h, frame.strides[0],
                     QImage.Format_BGR888).copy()
        self.combos.set_thumb(key, QPixmap.fromImage(img))

    # ============================================================ recalc
    def recalc(self, now=False):
        if not now:
            self.recalc_timer.start()
            return
        self.recalc_timer.stop()
        p = self.project
        if p.audio is None or p.video is None or not p.markers.beats:
            self._update_info()
            return
        try:
            p.recalculate()
        except Exception as exc:                          # noqa: BLE001
            traceback.print_exc()
            self.statusBar().showMessage(f"Alignment failed: {exc}")
            return
        self._update_info()
        self._refresh_plan()
        # caption / effect times are shown on the montage clock (like the player)
        if abs(self.texts.origin - p.schedule.start) > 1e-6:
            self.texts.origin = p.schedule.start
            self.texts.refresh_times()
            self.fx_panel.refresh_ranges()
        self.timeline.update()
        if self.mode == "montage":
            self._load_program(keep=True)
        self.sound_timer.start()

    def _update_info(self):
        p = self.project
        # tempo chip
        if p.audio is not None:
            g = p.audio.grid
            ok = g.confidence >= 0.6 and not g.drift
            mark = "✓" if ok else ("~" if g.drift else "?")
            self.tempo_chip.setText(f"{p.grid_bpm:.1f} BPM {mark}")
            self.tempo_chip.setStyleSheet(f"color: {theme.GOOD if ok else theme.WARN};")
            self.tempo_chip.setToolTip(
                ("Tempo wanders (live band): following the drummer. " if g.drift else "") +
                f"Detection confidence {g.confidence:.0%}. Beat looks off? Fix it in the "
                "Song tab.")
        # music start chips
        while self.start_chips.count():
            wdg = self.start_chips.takeAt(0).widget()
            if wdg is not None:
                wdg.setParent(None)                  # gone now, not when Qt gets to it
                wdg.deleteLater()
        if p.audio is not None:
            for t, text in p.start_candidates():
                b = button(f"{theme.fmt_short(t)} {text}", "Start the combos here", chip=True,
                           checkable=True)
                b.setChecked(abs(t - p.sync.drop_time) < 0.05)
                b.clicked.connect(lambda _=False, t=t: self._set_start(0, t))
                self.start_chips.addWidget(b)
            if not any(abs(t - p.sync.drop_time) < 0.05 for t, _ in p.start_candidates()):
                b = button(f"{theme.fmt_short(p.sync.drop_time)} (yours)", chip=True,
                           checkable=True)
                b.setChecked(True)
                self.start_chips.addWidget(b)
        s = p.schedule
        if s is not None:
            placed = s.placed_hits()
            self.combo_lbl.setText(f"{s.combos_kept} combos · {len(placed)} hits · "
                                   f"{theme.fmt_short(s.duration)} montage")
            locked = sum(1 for x in placed if x.locked)
            self.summary.setText(f"Montage {theme.fmt_time(s.duration)} · {s.combos_kept} combos"
                                 f" · {locked}/{len(placed)} hits on the beat · "
                                 f"{s.ref_bpm:.1f} BPM · beats per hit {s.beats_per_hit:g}"
                                 + (f" · ⚠ {s.warnings[0]}" if s.warnings else ""))
        self._refresh_cards()

    def _refresh_plan(self):
        p = self.project
        if p.video is None:
            self.combos.set_entries([], p.markers, {}, True)
            return
        entries = p.plan_entries()
        sig = tuple((round(c.key, 4), c.enabled, c.lead_in, c.focus, len(g)) for c, g in entries) + \
            tuple(h.fx for h in p.markers.hits)
        if sig != getattr(self, "_plan_sig", None):
            self._plan_sig = sig
            self.combos.set_entries(entries, p.markers, p.rank_combos(),
                                    p.render_params.hit_fx_default)

    # ============================================================ params
    def _pull_all(self):
        p = self.project
        r = p.render_params
        self.hit_vol.set(r.hit_volume)
        self.music_vol.set(r.music_volume)
        self.bph.set(p.sync.combo_spacing if p.sync.combo_spacing != "1.0" else "1")
        self.style_seg.set(r.style)
        self.fx_panel.pull()
        self.advanced.pull()
        self.texts.set_items(p.texts)
        self.timeline.project = p
        self._refresh_song_list()

    def set_style(self, name):
        styles.apply_style(self.project, name)
        self.fx_panel.pull()
        self.advanced.pull()
        self.recalc()

    def _mark_custom(self, attr):
        if attr in styles.style_keys() and self.project.render_params.style != "custom":
            self.project.render_params.style = "custom"
            self.style_seg.set("custom")

    def _look_changed(self, attr):
        if attr:
            self._mark_custom(attr)
        self.recalc()

    def _advanced_changed(self, attr):
        p = self.project
        self._mark_custom(attr)
        if attr in ("static_grid", "bpm_override", "grid_offset_ms"):
            p.apply_beat_grid()
        elif attr in ("combo_gap", "combo_regularity"):
            p.regroup()
        elif attr in ("sensitivity", "motion_weight", "min_hit_interval"):
            self._redetect()
            return
        elif attr in ("intro_auto",) and p.sync.intro_auto:
            p.set_drop(p.sync.drop_time)
        if attr in ("intro_length", "intro_target_speed", "intro_ramp", "min_combo_len",
                    "combo_gap", "combo_regularity"):
            p.auto_intro()
        if attr == "intro_length":
            p.sync.intro_auto = False
        self.fx_panel.pull()
        self.recalc()

    def _audio_param(self, attr, value):
        setattr(self.project.render_params, attr, float(value))
        self.sound_timer.start()

    def _bph_changed(self, value):
        self.project.sync.combo_spacing = str(value)
        self.recalc()

    def _redetect(self):
        p = self.project
        if p.video is None:
            return
        from ..video_analysis import hit_score

        p.redetect_hits()
        p.auto_intro()
        self.timeline.set_score(p.video.times, hit_score(p.video, p.detect))
        self.recalc()

    def _pick_sound(self):
        path, _ = QFileDialog.getOpenFileName(self, "Hit sound", self._dir(), AUDIO_FILTER)
        if path:
            self.project.render_params.hit_sound_file = path
            self.project.render_params.hit_sound = "custom"
            self.advanced.pull()
            self.sound_timer.start()

    def show_advanced(self):
        self.advanced.pull()
        self.advanced.show()
        self.advanced.raise_()

    # ============================================================ combos
    def _plan_changed(self, choices):
        self.project.set_plan(choices)
        self._plan_sig = None
        self.recalc()

    def _fx_changed(self, indices, value):
        self.project.set_hit_fx(indices, value)
        self.recalc()

    def _auto_pick(self):
        self.project.auto_pick()
        self._plan_sig = None
        self.recalc()
        self.statusBar().showMessage("Picked the best combos that fit the song.")

    def _reset_plan(self):
        self.project.reset_plan()
        self._plan_sig = None
        self.recalc()

    def _combo_selected(self, key):
        p = self.project
        if self.mode == "gameplay":
            self.seek(max(0.0, key - 1.0))
            return
        entries = [c for c, _ in p.plan_entries() if c.enabled]
        k = next((i for i, c in enumerate(entries) if abs(c.key - key) < 1e-6), None)
        if k is not None and p.schedule is not None:
            span = p.combo_out_span(k)
            if span:
                self.seek(max(p.schedule.start, span[0] - 0.4))

    def _markers_changed(self, what):
        self.statusBar().showMessage(what)
        self._plan_sig = None
        self.recalc()

    def _timeline_click(self, track, t):
        if track in ("audio", "edit") and self.mode == "montage":
            self.seek(t)
        elif track == "video" and self.mode == "gameplay":
            self.seek(t)

    def _timeline_selected(self, sel):
        if sel and sel[0] == "text":
            self.side.setCurrentWidget(self.texts)
        elif sel and sel[0] == "effect":
            self.side.setCurrentWidget(self.fx_panel)

    # ========================================================== overlays
    def _add_text(self):
        p = self.project
        if not p.markers.beats:
            QMessageBox.information(self, "Hit-Sync", "Add a song first.")
            return
        t = self.engine.time() if self.mode == "montage" else p.sync.drop_time
        p.add_text("YOUR TEXT", t, bars=2, position="bottom")
        self.texts.set_items(p.texts)
        self.side.setCurrentWidget(self.texts)
        if self.texts.rows:
            row = next((r for r in self.texts.rows if r.item.text == "YOUR TEXT"), None)
            if row:
                row.edit.setFocus()
                row.edit.selectAll()
        self._overlays_changed()
        item = p.texts[-1] if p.texts else None
        if item is not None and not self.engine.playing:
            self.recalc(now=True)
            self._show_still(item.start + 0.5)          # the caption fully in

    def _text_time(self, item, which):
        t = self.engine.time()
        if which == "start":
            item.start = self.project._snap_beat(t)
            if item.end <= item.start:
                item.end = item.start + 4 * self.project.beat_period
        else:
            item.end = max(item.start + 0.2, self.project._snap_beat(t))
        self.texts.refresh_times()
        self._overlays_changed()

    def _add_range(self, kind, where):
        p = self.project
        if p.schedule is None:
            return
        t = self.engine.time()
        bar = 4 * p.beat_period
        if where == "combo":
            span = next(((a, b) for a, b, _ in p.schedule.combo_spans if a - 0.1 <= t <= b + 0.1),
                        None)
            if span is None and p.schedule.combo_spans:
                a, b, _ = min(p.schedule.combo_spans, key=lambda s: abs(s[0] - t))
                span = (a, b)
            start, end = span if span else (t, t + 4 * bar)
            end += p.beat_period * 0.5
        elif where == "end":
            start, end = t, p.schedule.end
        else:
            start, end = t, t + int(where) * bar
        p.add_effect(kind, start, end)
        self.fx_panel.refresh_ranges()
        self._overlays_changed()
        o = self.project.schedule.start if self.project.schedule is not None else 0.0
        self.statusBar().showMessage(f"Added {kind} from {theme.fmt_time(start - o)} to "
                                     f"{theme.fmt_time(end - o)}")

    def _overlays_changed(self):
        self.texts.refresh_times()
        self.fx_panel.refresh_ranges()
        self.timeline.update()
        self.recalc()

    # ========================================================= songs/marks
    def _refresh_song_list(self):
        p = self.project
        self.song_pick.blockSignals(True)
        self.song_pick.clear()
        for k, path in enumerate(p.music_paths):
            if path:
                self.song_pick.addItem(f"Song {k + 1}: {os.path.basename(path)}", k)
        self.song_index = min(self.song_index, max(0, self.song_pick.count() - 1))
        self.song_pick.setCurrentIndex(self.song_index)
        self.song_pick.blockSignals(False)
        self._update_cut_button()
        self._refresh_music_panel()

    def _update_cut_button(self):
        last = self.song_index >= self.project.song_count - 1
        self.cut_btn.setText("✂ End the music here  (C)" if last
                             else "✂ Switch to next song here  (C)")

    def _song_infos(self):
        p = self.project
        infos = []
        n = p.song_count
        for k, path in enumerate(p.music_paths):
            if not path:
                continue
            info = {"name": os.path.basename(path), "detail": "Analysing…", "span": ""}
            a = p._song(k)
            if a is not None:
                _, downs, bpm = p.song_grid(k)
                end, auto = p.song_end(k, downs)
                info["detail"] = f"{bpm:.1f} BPM · {theme.fmt_short(a.duration)}"
                start = p.song_start(k)
                verb = "ends at" if k == n - 1 else "switches at"
                info["span"] = (f"{'First hit' if k == 0 else 'Starts'} at "
                                f"{theme.fmt_short(start)} · {verb} {theme.fmt_short(end)}"
                                + (" (auto)" if auto else " ✂"))
                info["cut_manual"] = not auto
            infos.append(info)
        return infos

    def _refresh_music_panel(self, force=False):
        infos = self._song_infos()
        sig = (tuple(tuple(sorted(i.items())) for i in infos), self.song_index)
        if force or sig != getattr(self, "_music_sig", None):
            self._music_sig = sig
            self.music_panel.set_songs(infos, self.song_index)

    def _song_picked(self, i):
        self.song_index = max(0, i)
        self._update_cut_button()
        self._refresh_music_panel()
        if self.mode == "song":
            self._song_view_update()
            self._load_program(keep=False)

    def _song_selected(self, k):
        """A song clicked in the Music panel: listen to it in the Song view."""
        self.song_index = k
        self.song_pick.blockSignals(True)
        self.song_pick.setCurrentIndex(k)
        self.song_pick.blockSignals(False)
        self._update_cut_button()
        self._refresh_music_panel()
        if self.mode != "song":
            self.set_mode("song")
        else:
            self._song_view_update()
            self._load_program(keep=False)

    def _songs_busy(self):
        if "analyze" in self.busy:
            self.statusBar().showMessage("Wait for the songs to finish analysing.", 4000)
            return True
        return False

    def _songs_changed(self, message=""):
        self.engine.pause()
        self._refresh_song_list()
        self._refresh_cards()
        self._refresh_music_panel(force=True)
        self.recalc(now=True)
        if self.mode == "song":
            self._song_view_update()
        self._load_program(keep=self.mode != "song")
        if message:
            self.statusBar().showMessage(message, 5000)

    def _move_song(self, k, d):
        if self._songs_busy():
            return
        j = self.project.move_song(k, d)
        if j != k:
            if self.song_index == k:
                self.song_index = j
            elif self.song_index == j:
                self.song_index = k
            self._songs_changed(f"Song moved to place {j + 1}")

    def _song_order(self, order):
        if self._songs_busy():
            self._refresh_music_panel(force=True)
            return
        cur = self.song_index
        self.project.set_song_order(order)
        self.song_index = order.index(cur) if cur in order else 0
        self._songs_changed("New song order")

    def _remove_song(self, k=None):
        if self._songs_busy():
            return
        k = self.song_index if k is None else k
        name = os.path.basename(self.project.music_paths[k])
        if self.project.remove_song(k):
            if self.song_index > k or self.song_index >= self.project.song_count:
                self.song_index = max(0, self.song_index - 1)
            self._songs_changed(f"Removed {name}")

    def _reset_cut(self, k):
        self.project.clear_song_end(k)
        self._songs_changed("Back to the automatic end")

    def cut(self):
        """✂ at the playhead: the song switches to the next one (or the music
        ends) on this bar."""
        p = self.project
        if self.mode != "song" or p._song(self.song_index) is None:
            if p.audio is not None:
                self.statusBar().showMessage("Open the Song view (Music tab: click a song), "
                                             "play it, then press ✂ where it should stop.", 5000)
            return
        k = self.song_index
        t = p.set_song_end(k, self.engine.time())
        if t is None:
            self.statusBar().showMessage("Too early: cut at least a bar after the song's start.",
                                         5000)
            return
        last = k >= p.song_count - 1
        self._songs_changed(f"{'Music ends' if last else 'Next song takes over'} at "
                            f"{theme.fmt_time(t)} (song {k + 1})")

    def _set_start(self, k, t):
        p = self.project
        t = p.set_song_start(k, t)
        p.auto_intro()
        self.recalc(now=True)
        self._song_view_update()
        self.statusBar().showMessage(f"Music starts at {theme.fmt_time(t)}"
                                     + ("" if k == 0 else f" (song {k + 1})"))

    def mark(self):
        p = self.project
        t = self.engine.time()
        if self.mode == "song" and p.audio is not None:
            self._set_start(self.song_index, t)
        elif self.mode == "gameplay" and p.video is not None:
            key = p.combat_starts_at(t)
            if key is None:
                QMessageBox.information(self, "Hit-Sync", "No hits after this point.")
                return
            self._plan_sig = None
            self.recalc(now=True)
            self.statusBar().showMessage(f"Combat starts at {theme.fmt_time(key)} in the video")

    def tap(self):
        if self.mode != "song" or self.project.audio is None:
            return
        if not self.engine.playing:
            self.toggle_play()
            return
        t = self.engine.time()
        if self.taps and t - self.taps[-1] > 2.5:
            self.taps = []
        self.taps.append(t)
        self.song_view.taps = list(self.taps)
        if len(self.taps) >= 8 and self.project.tap_tempo(self.taps[-12:], self.song_index):
            self.statusBar().showMessage(f"Tapped tempo: {self._song_bpm():.1f} BPM")
            self.recalc(now=True)
            self._song_view_update()
        else:
            self.statusBar().showMessage(f"Tap {len(self.taps)}/8…")

    def _grid_fix(self, what):
        p, k = self.project, self.song_index
        if p.audio is None:
            return
        if what == "half":
            p.set_tempo_factor(0.5, k)
        elif what == "double":
            p.set_tempo_factor(2.0, k)
        elif what == "shift":
            p.shift_half_beat(k)
        elif what == "reset":
            p.reset_grid(k)
        else:
            p.nudge_grid(float(what), k)
        if k == 0:
            p.set_drop(p.sync.drop_time)
        p.auto_intro()
        self.advanced.pull()
        self.recalc(now=True)
        self._song_view_update()
        if self.mode == "song":
            self._load_program(keep=True)

    def _song_bpm(self):
        p = self.project
        return p.song_grid(self.song_index)[2] if p._song(self.song_index) is not None else 0.0

    def _song_view_update(self):
        p = self.project
        a = p._song(self.song_index)
        if a is None:
            return
        beats, downs, bpm = p.song_grid(self.song_index)
        start = p.sync.drop_time if self.song_index == 0 else p.song_start(self.song_index)
        cands = [(float(beats[np.argmin(np.abs(beats - c.time))]) if len(beats) else c.time,
                  c.label) for c in a.sections.top(3)]
        name = os.path.basename(p.music_paths[self.song_index])
        end, auto = p.song_end(self.song_index, downs)
        last = self.song_index >= p.song_count - 1
        self.song_view.set_song(a.curve_t, a.level, beats, downs, a.duration, start, cands,
                                f"Song {self.song_index + 1}: {name}", end,
                                "MUSIC ENDS" if last else "NEXT SONG", auto)
        g = a.grid
        self.grid_lbl.setText(f"{bpm:.2f} BPM · confidence {g.confidence:.0%}"
                              + (" · tempo wanders, following the drummer" if g.drift else ""))
        self._refresh_music_panel()

    # ============================================================ playback
    def set_mode(self, mode):
        if mode == self.mode and self.engine.program is not None:
            self.mode_seg.set(mode)
            return
        self.engine.pause()
        self.mode = mode
        self.mode_seg.set(mode)
        self._update_mode_ui()
        self._load_program(keep=False)

    def _update_mode_ui(self):
        idx = {"montage": 0, "song": 1, "gameplay": 2}[self.mode]
        self.stack.setCurrentIndex(1 if self.mode == "song" else 0)
        self.actions.setCurrentIndex(idx)
        self.mode_seg.set(self.mode)
        self.mode_hint.setText({
            "montage": "The finished montage",
            "song": "Listen and mark where the music starts; check the beat",
            "gameplay": "Your raw recording: mark where the combat starts"}[self.mode])
        if self.mode == "song":
            self._refresh_song_list()
            self._song_view_update()

    def _program_start(self):
        pr = self.engine.program
        return pr.start if pr else 0.0

    def _load_program(self, keep=True):
        from ..preview_engine import montage_program, song_program, source_program

        p = self.project
        pos = self.engine.time() if keep else None
        prog = None
        if self.mode == "montage" and p.schedule is not None and p.video is not None:
            prog = montage_program(p, self._audio_for(p.schedule))
            if pos is None:
                pos = p.schedule.start
            if p.schedule.duration <= 0:
                prog = None
        elif self.mode == "gameplay" and p.video is not None:
            prog = source_program(p)
            if pos is None:
                pos = max(0.0, p.sync.intro_end - 3.0)
        elif self.mode == "song" and p._song(self.song_index) is not None:
            path = p.music_paths[self.song_index]
            a = p._song(self.song_index)
            prog = song_program(path, a.duration)
            if self.click_cb.isChecked():
                from ..audio_analysis import click_track

                beats, downs, _ = p.song_grid(self.song_index)
                prog.audio = prog.audio * 0.8 + click_track(beats, downs, len(prog.audio))[:, None]
            if pos is None:
                start = p.sync.drop_time if self.song_index == 0 else p.song_start(self.song_index)
                pos = max(0.0, start - 4.0)
        if prog is None:
            self.engine.program = None
            if self.mode == "song":
                self.song_view.update()
            else:
                self.view.set_placeholder(*self._placeholder())
            return
        info = p.video.info if (p.video and prog.kind != "song") else None
        if info is not None:
            self.engine.size = self.engine.fit_size(self.view.width(), self.view.height(),
                                                    info.width / max(1, info.height))
        self.engine.position = float(np.clip(pos, prog.start, prog.end))
        self.engine.load(prog)
        if not self.engine.playing:
            poster = None
            if prog.kind == "montage" and not keep:
                hits = prog.schedule.placed_hits()          # not the black fade-in
                poster = hits[0].out_t + 0.3 if hits else prog.start + 1.0
            self._show_still(poster)

    def _placeholder(self):
        p = self.project
        if not p.video_path:
            return ("Drop your gameplay video and a song",
                    "That's it: beats, hits and combos are found automatically.")
        if not p.music_path:
            return ("Now drop a song", "Any mp3, wav, ogg, flac or m4a.")
        if "analyze" in self.busy:
            return ("Analysing…", "Beats, hits and combos are being found.")
        if p.schedule is not None and not p.schedule.placed_hits():
            return ("No combos found",
                    "Try Advanced → Combos: fewer 'Min hits per combo', or check the hits "
                    "on the timeline.")
        return ("", "")

    def _audio_for(self, sched):
        key = self._sound_key(sched)
        if self.soundtrack and self.soundtrack[0] == key:
            return self.soundtrack[1]
        return None

    def _sound_key(self, sched):
        r = self.project.render_params
        return (id(sched), tuple(self.project.music_paths), r.hit_sound, r.hit_sound_file,
                round(r.hit_volume, 3), round(r.music_volume, 3), r.hit_pitch_variation,
                r.fade_in, r.fade_in_len, r.fade_out, r.fade_out_len, self.click_cb.isChecked())

    def _build_soundtrack(self):
        p = self.project
        sched = p.schedule
        if sched is None or p.audio is None:
            return
        key = self._sound_key(sched)
        if self.soundtrack and self.soundtrack[0] == key:
            return
        music, rp, video = p.music_source(), p.render_params, p.video_path
        click = self.click_cb.isChecked()

        def work(_progress):
            from ..audio_mix import build_soundtrack

            return key, build_soundtrack(music, sched, rp, video_path=video, click=click)

        self._run("soundtrack", work, "Preparing the sound…")

    def _attach_audio(self):
        pr = self.engine.program
        if pr is None or pr.kind != "montage" or not self.soundtrack:
            return
        if self.soundtrack[0] != self._sound_key(pr.schedule):
            return
        pr.audio = self.soundtrack[1]
        if self.engine.playing:
            self.engine.play(self.engine.time())
        self.statusBar().showMessage("Ready", 2000)

    def _click_toggled(self, _on):
        if self.mode == "song":
            self._load_program(keep=True)
        else:
            self.sound_timer.start()

    def toggle_play(self):
        pr = self.engine.program
        if pr is None:
            self._load_program(keep=False)
            pr = self.engine.program
            if pr is None:
                return
        if pr.kind == "montage" and pr.audio is None:
            self._build_soundtrack()
        self.engine.toggle()

    def seek(self, t):
        self.engine.seek(t)
        if not self.engine.playing:
            self._show_still()
        self._update_time_ui(force=True)

    def step(self, direction):
        """Left/right: one beat back/forward (montage & song), 1 s in gameplay."""
        p = self.project
        t = self.engine.time()
        if self.mode == "gameplay":
            self.seek(t + direction)
            return
        beats = np.asarray(p.song_grid(self.song_index)[0] if self.mode == "song"
                           else p.markers.beats, float)
        if not len(beats):
            return
        if direction > 0:
            nxt = beats[beats > t + 1e-3]
            self.seek(float(nxt[0]) if len(nxt) else t)
        else:
            prv = beats[beats < t - 1e-3]
            self.seek(float(prv[-1]) if len(prv) else t)

    def _scrubbed(self, v):
        pr = self.engine.program
        if pr is not None:
            self.seek(pr.start + (pr.end - pr.start) * v / 10000.0)

    def _show_still(self, t=None):
        pr = self.engine.program
        if pr is None or pr.kind == "song":
            self.song_view.playhead = self.engine.time()
            self.song_view.update()
            return
        try:
            f = self.engine.still(self.engine.time() if t is None else t)
        except Exception as exc:                          # noqa: BLE001
            self.statusBar().showMessage(f"Preview problem: {exc}")
            return
        if f is not None:
            self._show_frame(f)

    def _show_frame(self, frame):
        if self.fs is not None and self.fs.isVisible():
            self.fs.view.set_frame(frame)
        else:
            self.view.set_frame(frame)

    def _tick(self):
        e = self.engine
        if e.playing:
            got = e.current_frame()
            if got is not None:
                self._show_frame(got[1])
        self._update_time_ui()

    def _update_time_ui(self, force=False):
        now = time.perf_counter()
        if not force and now - self._last_ui < 1 / 30:
            return
        self._last_ui = now
        e = self.engine
        pr = e.program
        t = e.time()
        playing = e.playing
        self.play_btn.setText("⏸" if playing else "▶")
        if pr is None:
            self.time_lbl.setText("0:00.00 / 0:00.00")
            return
        rel, dur = t - pr.start, pr.end - pr.start
        self.time_lbl.setText(f"{theme.fmt_time(rel)} / {theme.fmt_time(dur)}")
        if not self.scrub.isSliderDown() and dur > 0:
            self.scrub.blockSignals(True)
            self.scrub.setValue(int(10000 * rel / dur))
            self.scrub.blockSignals(False)
        if self.fs is not None and self.fs.isVisible():
            self.fs.set_state(playing, rel, dur)
        if pr.kind == "song":
            self.song_view.playhead = t
            self.song_view.update()
        elif pr.kind == "montage" and self.timeline.isVisible():
            self.timeline.set_playhead(t, pr.schedule.src_time(t), follow=playing)
        elif pr.kind == "source" and self.timeline.isVisible():
            self.timeline.set_playhead(None, t, follow=False)

    # ========================================================= fullscreen
    def toggle_fullscreen(self):
        if self.mode == "song":
            return
        if self.fs is not None and self.fs.isVisible():
            self.fs.close()
            return
        if self.fs is None:
            self.fs = FullscreenPlayer()
            self.fs.toggle_play.connect(self.toggle_play)
            self.fs.seek.connect(lambda f: self._scrubbed(int(f * 10000)))
            self.fs.closed.connect(self._fs_closed)
        self.fs.view.img = self.view.img
        screen = self.screen().geometry() if self.screen() else None
        self.fs.showFullScreen()
        self.fs.show_controls()
        p = self.project
        if screen is not None and p.video is not None:
            info = p.video.info
            self.engine.set_size(self.engine.fit_size(screen.width(), screen.height(),
                                                      info.width / max(1, info.height)))
        if not self.engine.playing:
            self._show_still()

    def _fs_closed(self):
        self.view.img = self.fs.view.img if self.fs else self.view.img
        self._fit_preview()
        self.view.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.resize_timer.start()

    def _fit_preview(self):
        p = self.project
        if p.video is None or (self.fs is not None and self.fs.isVisible()):
            return
        info = p.video.info
        size = self.engine.fit_size(self.view.width(), self.view.height(),
                                    info.width / max(1, info.height))
        if size != self.engine.size:
            self.engine.set_size(size)
            if not self.engine.playing:
                self._show_still()

    def _toggle_timeline(self, on):
        self.timeline.setVisible(on)
        self.timeline.hbar.setVisible(on)
        self.tl_btn.setText("Timeline ▴" if on else "Timeline ▾")
        if on:
            QTimer.singleShot(0, self.timeline.zoom_fit)

    # ============================================================ export
    def export(self):
        p = self.project
        if p.schedule is None or p.video is None:
            QMessageBox.information(self, "Hit-Sync", "Drop a video and a song first.")
            return
        if "export" in self.busy:
            return
        out = p.output_path or default_output_path(p.video_path)
        path, _ = QFileDialog.getSaveFileName(self, "Export video", out, "MP4 video (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"
        p.output_path = path
        self.engine.pause()
        self.cancel.clear()
        self.recalc(now=True)

        def work(progress):
            return p.render(progress=progress, cancel=self.cancel, out_path=path)

        self._run("export", work, "Exporting…")
        self.export_btn.setEnabled(False)

    def _export_done(self, path):
        from ..ffmpeg_utils import open_path

        self.export_btn.setEnabled(True)
        self.statusBar().showMessage(f"Exported {os.path.basename(path)}")
        box = QMessageBox(self)
        box.setWindowTitle("Export finished")
        box.setText(f"Your montage is ready:\n{path}")
        play = box.addButton("Play it", QMessageBox.AcceptRole)
        folder = box.addButton("Show in folder", QMessageBox.ActionRole)
        box.addButton("Close", QMessageBox.RejectRole)
        box.exec()
        try:
            if box.clickedButton() is play:
                open_path(path)
            elif box.clickedButton() is folder:
                open_path(os.path.dirname(path))
        except OSError as exc:
            self.statusBar().showMessage(str(exc))

    # ============================================================ project
    def save_project(self, save_as=False):
        p = self.project
        path = self.project_path
        if save_as or not path:
            base = os.path.splitext(os.path.basename(p.video_path or "montage"))[0]
            path, _ = QFileDialog.getSaveFileName(self, "Save project",
                                                  os.path.join(self._dir(), f"{base}.hitsync.json"),
                                                  "Hit-Sync project (*.json)")
            if not path:
                return
        p.save(path)
        self.project_path = path
        self.statusBar().showMessage(f"Saved {os.path.basename(path)}")

    def open_project(self, path=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Open project", self._dir(),
                                                  "Hit-Sync project (*.json)")
            if not path:
                return
        try:
            proj = Project.load(path)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self, "Hit-Sync", f"Could not open the project:\n{exc}")
            return
        self.engine.close()
        self.project = proj
        self.project_path = path
        self.soundtrack = None
        self._plan_sig = None
        self._pull_all()
        if proj.video is not None:
            from ..video_analysis import hit_score

            self.timeline.set_score(proj.video.times, hit_score(proj.video, proj.detect))
        need = proj.stale()
        if proj.audio is None or proj.video is None or any(need):
            self.analyze()
        else:
            self._analysis_done()
        self.statusBar().showMessage(f"Opened {os.path.basename(path)}")

    # ============================================================== misc
    def _check_ffmpeg(self):
        from ..ffmpeg_utils import find_ffmpeg

        if not find_ffmpeg():
            QMessageBox.warning(self, "Hit-Sync", "ffmpeg was not found. Run the setup script "
                                                  "again, or install ffmpeg.")

    def _help(self):
        QMessageBox.information(self, "Keyboard shortcuts", (
            "Space  play / pause\n"
            "F  fullscreen preview (Esc to leave)\n"
            "← →  one beat back / forward\n"
            "M  mark: music starts here (Song) / combat starts here (Gameplay)\n"
            "T  tap along with the beat (Song)\n"
            "C  cut: the next song takes over here / the music ends here (Song)\n"
            f"{theme.MOD}S  save project · {theme.MOD}O open · {theme.MOD}E export\n\n"
            "Timeline: double-click adds a beat/hit, right-click deletes, S splits a combo,\n"
            "M merges it with the previous one, E toggles effects on a hit.\n"
            f"The wheel scrolls the timeline; {theme.MOD}wheel zooms."))

    def closeEvent(self, e):
        self.settings.setValue("geometry", self.saveGeometry())
        self.cancel.set()
        self.engine.close()
        if self.fs is not None:
            self.fs.close()
        super().closeEvent(e)


__all__ = ["MainWindow", "QPushButton", "QSizePolicy"]
