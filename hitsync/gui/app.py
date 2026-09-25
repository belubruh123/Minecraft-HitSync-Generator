"""customtkinter desktop GUI for Hit-Sync."""
from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
import traceback
from tkinter import filedialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from ..audio_mix import HIT_SOUND_CHOICES, hit_samples
from ..ffmpeg_utils import find_ffmpeg
from ..project import Project, default_output_path
from ..video_analysis import hit_score, read_frame, snap_to_peak
from .player import Player
from .timeline import Timeline

VIDEO_TYPES = [("Video", "*.mp4 *.mkv *.mov *.avi *.webm *.flv"), ("All files", "*.*")]
AUDIO_TYPES = [("Audio", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac *.opus"), ("All files", "*.*")]
PREVIEW_W, PREVIEW_H = 400, 225


class Param:
    """Binds a tk variable to an attribute of a params dataclass."""

    def __init__(self, obj_getter, attr, var):
        self.obj_getter, self.attr, self.var = obj_getter, attr, var

    def push(self):          # variable -> dataclass
        try:
            val = self.var.get()
        except (tk.TclError, ValueError):
            return
        obj = self.obj_getter()
        cur = getattr(obj, self.attr)
        setattr(obj, self.attr, type(cur)(val))

    def pull(self):          # dataclass -> variable
        self.var.set(getattr(self.obj_getter(), self.attr))


class HitSyncApp(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()
        self.title("Hit-Sync · Minecraft PvP Montage Editor")
        self._fit_to_screen(1480, 940)

        self.project = Project()
        self.events: "queue.Queue[tuple]" = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.params: list[Param] = []
        self._recalc_job = None
        self._preview_img = None
        self.project_path: str | None = None

        # Row 1 (preview) absorbs any shortage of height so the timeline
        # editor always stays fully visible.
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_files()
        self._build_sidebar()
        self._build_main()
        self._build_timeline()
        self.player = Player(self, lambda: self.project, self._player_frame,
                             self._player_position, self._player_state)
        self._build_menu()
        self._pull_params()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll)
        self._check_deps()

    # ================================================================ layout
    def _fit_to_screen(self, want_w, want_h):
        """Size the window in logical units so it fits high-DPI screens."""
        scale = ctk.ScalingTracker.get_window_scaling(self)
        avail_w = self.winfo_screenwidth() / scale - 40
        avail_h = self.winfo_screenheight() / scale - 90   # taskbar + title bar
        w, h = int(min(want_w, avail_w)), int(min(want_h, avail_h))
        self.geometry(f"{w}x{h}+10+10")
        self.minsize(min(1000, w), min(640, h))

    def _build_menu(self):
        bar = tk.Menu(self)
        fm = tk.Menu(bar, tearoff=0)
        fm.add_command(label="Choose Gameplay Video…", command=self._pick_video)
        fm.add_command(label="Choose Music…", command=self._pick_music)
        fm.add_separator()
        fm.add_command(label="Open Project (.json)…", accelerator="Ctrl+O",
                       command=self._load_project)
        fm.add_command(label="Save Project", accelerator="Ctrl+S", command=self._save_project)
        fm.add_command(label="Save Project As (.json)…", accelerator="Ctrl+Shift+S",
                       command=lambda: self._save_project(save_as=True))
        fm.add_separator()
        fm.add_command(label="Export Video As (.mp4)…", accelerator="Ctrl+E",
                       command=self._export)
        fm.add_separator()
        fm.add_command(label="Exit", command=self._on_close)
        bar.add_cascade(label="File", menu=fm)
        pm = tk.Menu(bar, tearoff=0)
        pm.add_command(label="Play / Pause", accelerator="Space", command=self._toggle_play)
        pm.add_command(label="Go to Start", accelerator="Home", command=lambda: self.player.seek(0))
        bar.add_cascade(label="Playback", menu=pm)
        self.configure(menu=bar)
        self.bind_all("<Control-o>", lambda e: self._load_project())
        self.bind_all("<Control-s>", lambda e: self._save_project())
        self.bind_all("<Control-S>", lambda e: self._save_project(save_as=True))
        self.bind_all("<Control-e>", lambda e: self._export())
        self.bind_all("<space>", self._space)
        self.bind_all("<Home>", lambda e: None if self._typing() else self.player.seek(0))

    def _typing(self) -> bool:
        return isinstance(self.focus_get(), (tk.Entry, tk.Text))

    def _space(self, _e):
        if self._typing():
            return None
        self._toggle_play()
        return "break"

    def _build_files(self):
        f = ctk.CTkFrame(self)
        f.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 4))
        f.grid_columnconfigure(1, weight=1)
        self.video_var = tk.StringVar()
        self.music_var = tk.StringVar()
        self.out_var = tk.StringVar()
        rows = [("Gameplay video", self.video_var, self._pick_video),
                ("Music track", self.music_var, self._pick_music),
                ("Output file", self.out_var, self._pick_output)]
        for r, (label, var, cmd) in enumerate(rows):
            ctk.CTkLabel(f, text=label, width=110, anchor="w").grid(row=r, column=0, padx=6, pady=2)
            ctk.CTkEntry(f, textvariable=var).grid(row=r, column=1, sticky="ew", padx=4, pady=2)
            ctk.CTkButton(f, text="Browse…", width=80, command=cmd).grid(row=r, column=2, padx=4)
        btns = ctk.CTkFrame(f, fg_color="transparent")
        btns.grid(row=0, column=3, rowspan=3, padx=8)
        self.analyze_btn = ctk.CTkButton(btns, text="Analyze Media", height=40, width=150,
                                         command=self._analyze)
        self.analyze_btn.pack(pady=2)
        row = ctk.CTkFrame(btns, fg_color="transparent")
        row.pack(pady=2)
        ctk.CTkButton(row, text="Open Project", width=100,
                      command=self._load_project).pack(side="left", padx=2)
        ctk.CTkButton(row, text="Save Project", width=100,
                      command=self._save_project).pack(side="left", padx=2)

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").pack(fill="x", padx=6, pady=(12, 2))

    def _slider(self, parent, label, obj, attr, lo, hi, fmt="{:.2f}", steps=None, integer=False):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=1)
        var = tk.IntVar() if integer else tk.DoubleVar()
        ctk.CTkLabel(row, text=label, anchor="w", width=150).pack(side="left")
        # Type an exact value (Enter / Tab / click away applies it, Esc undoes),
        # or drag the slider. Typed values aren't limited to the slider's steps.
        entry = ctk.CTkEntry(row, width=64, justify="right")
        entry.pack(side="right")
        slider = ctk.CTkSlider(row, from_=lo, to=hi, variable=var, number_of_steps=steps)
        slider.pack(side="right", fill="x", expand=True, padx=4)
        param = Param(obj, attr, var)
        percent = "%" in fmt
        normal_border = entry.cget("border_color")

        def show():
            try:
                v = var.get()
                text = fmt.format(v)
                if not percent and abs(float(text) - v) > 1e-9:   # keep typed precision
                    text = f"{v:.4f}".rstrip("0").rstrip(".")
            except (tk.TclError, ValueError):
                return
            entry.delete(0, "end")
            entry.insert(0, text)

        def changed(*_):
            if self.focus_get() is not getattr(entry, "_entry", entry):   # not while typing
                show()
            param.push()
            self._param_changed(attr)

        def apply(_event=None):
            text = entry.get().strip().replace(",", ".")
            try:
                is_pct = text.endswith("%")
                value = float(text.rstrip("%").strip())
                if percent and (is_pct or abs(value) > 1.0):
                    value /= 100.0                 # "30" or "30%" -> 0.30
                if lo >= 0:
                    value = max(value, 0.0)        # no negative lengths/speeds
                value = int(round(value)) if integer else value
            except ValueError:
                entry.configure(border_color="#d9534f")   # not a number: undo
                self.after(600, lambda: entry.configure(border_color=normal_border))
                show()
                return "break"
            try:
                same = abs(float(var.get()) - value) < 1e-12
            except (tk.TclError, ValueError):
                same = False
            if not same:
                var.set(value)
            show()
            return None

        def revert(_event=None):
            show()
            self.focus_set()
            return "break"

        entry.bind("<Return>", apply)
        entry.bind("<KP_Enter>", apply)
        entry.bind("<FocusOut>", apply)
        entry.bind("<Escape>", revert)
        var.trace_add("write", changed)
        self.params.append(param)
        slider.entry = entry
        return slider, var

    def _check(self, parent, label, obj, attr):
        var = tk.BooleanVar()
        cb = ctk.CTkCheckBox(parent, text=label, variable=var)
        cb.pack(anchor="w", padx=8, pady=2)
        self._widgets[attr] = cb
        param = Param(obj, attr, var)
        var.trace_add("write", lambda *_: (param.push(), self._param_changed(attr)))
        self.params.append(param)
        return var

    def _choice(self, parent, label, obj, attr, values, cast=str, menu=False):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=2)
        ctk.CTkLabel(row, text=label, anchor="w", width=150).pack(side="left")
        var = tk.StringVar()
        widget = ctk.CTkOptionMenu if menu else ctk.CTkSegmentedButton
        w = widget(row, values=values, variable=var)
        w.pack(side="right", fill="x", expand=True)
        self._widgets[attr] = w

        class _P(Param):
            def push(s):
                setattr(s.obj_getter(), attr, cast(var.get()))

            def pull(s):
                var.set(str(getattr(s.obj_getter(), attr)))

        param = _P(obj, attr, var)
        var.trace_add("write", lambda *_: (param.push(), self._param_changed(attr)))
        self.params.append(param)
        return var

    def _build_sidebar(self):
        self._widgets = {}
        sb = ctk.CTkScrollableFrame(self, width=360, label_text="Settings")
        sb.grid(row=1, column=0, rowspan=2, sticky="nsw", padx=(8, 4), pady=(4, 8))
        S = lambda: self.project.sync     # noqa: E731
        D = lambda: self.project.detect   # noqa: E731
        R = lambda: self.project.render_params  # noqa: E731

        self._section(sb, "1 · Intro speed ramp")
        self._check(sb, "Enable slow-mo intro", S, "intro_enabled")
        self.intro_start_slider, _ = self._slider(sb, "Intro start (video s)", S, "intro_start", 0, 60)
        self.intro_end_slider, _ = self._slider(sb, "Combat begins (video s)", S, "intro_end", 0, 60)
        self.drop_slider, _ = self._slider(sb, "Drop (music s)", S, "drop_time", 0, 60)
        self._slider(sb, "Ramp to 1.0x (s)", S, "intro_ramp", 0.1, 2.0)
        self._slider(sb, "Intro slow-mo speed", S, "intro_target_speed", 0.1, 0.9)
        self._slider(sb, "Intro length (s, 0=all)", S, "intro_length", 0, 15, "{:.1f}", 150)
        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=(4, 0))
        ctk.CTkButton(row, text="Drop = Playhead", command=self._drop_here).pack(
            side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(row, text="Combat = Selected Hit", command=self._combat_here).pack(
            side="left", expand=True, fill="x", padx=2)
        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(fill="x", padx=6, pady=4)
        ctk.CTkButton(row, text="Auto-Detect Drop", command=self._auto_drop).pack(
            side="left", expand=True, fill="x", padx=2)
        ctk.CTkButton(row, text="Auto Intro Range", command=self._auto_intro).pack(
            side="left", expand=True, fill="x", padx=2)

        self._section(sb, "2 · Beat grid")
        self._check(sb, "Static beat: one tempo, a hit on every beat", S, "static_grid")
        self._slider(sb, "BPM (0 = detected)", S, "bpm_override", 0, 200, "{:.1f}", 400)
        self._slider(sb, "Grid offset (ms)", S, "grid_offset_ms", -150, 150, "{:.0f}", 60)
        self.grid_lbl = ctk.CTkLabel(sb, text="", anchor="w", text_color="#8a93a3")
        self.grid_lbl.pack(fill="x", padx=8)
        self._check(sb, "Tempo matching (dynamic grid only)", S, "tempo_matching")

        self._section(sb, "3 · Cinematic letterbox")
        self._check(sb, "Black bars during combos", S, "letterbox_enabled")
        self._slider(sb, "Bar aspect ratio", S, "letterbox_aspect", 1.85, 3.0)
        self._slider(sb, "Ease in/out (s)", S, "letterbox_fade", 0.05, 1.0)
        self._slider(sb, "Min hits for bars", S, "min_combo_hits", 1, 6, "{:.0f}", 5, True)

        self._section(sb, "4 · Combos & sync")
        self._slider(sb, "Min hits per combo", S, "min_combo_len", 2, 30, "{:.0f}", 28, True)
        self._slider(sb, "Rhythm tolerance (±%)", S, "combo_regularity", 0.05, 0.6,
                     "{:.0%}", 55)
        self._slider(sb, "Max gap in combo (s)", S, "combo_gap", 0.3, 3.0)
        self._check(sb, "A hit on every beat (cut between combos)", S, "fill_every_beat")
        self._choice(sb, "Beats per hit", S, "combo_spacing", ["auto", "0.5", "1", "2"])
        self._slider(sb, "Tolerance (ms)", S, "jitter_tolerance_ms", 0, 200, "{:.0f}", 40)
        self._choice(sb, "Lock method", S, "lock_mode", ["ramp", "trim"])
        self._slider(sb, "Slowest speed-ramp", S, "ramp_min_speed", 0.3, 1.0)
        self._slider(sb, "Fastest speed-ramp", S, "ramp_max_speed", 1.0, 3.0)
        self._slider(sb, "Cut dead gaps > (s)", S, "long_gap", 0.0, 3.0)
        self._slider(sb, "Post-roll (s)", S, "post_roll", 0.0, 1.5)
        self._choice(sb, "Combo transition", S, "transition", ["cut", "flash"])

        self._section(sb, "Hit detection")
        self._slider(sb, "Sensitivity", D, "sensitivity", 0, 1)
        self._slider(sb, "Screen-velocity weight", D, "motion_weight", 0, 1)
        self._slider(sb, "Min hit spacing (s)", D, "min_hit_interval", 0.1, 1.0)
        ctk.CTkButton(sb, text="Re-detect Hits (cached signals)",
                      command=self._redetect).pack(fill="x", padx=8, pady=4)

        self._section(sb, "Hit sounds")
        self.sound_var = self._choice(sb, "Sound", R, "hit_sound", HIT_SOUND_CHOICES, menu=True)
        self.sound_var.trace_add("write", lambda *_: self.after_idle(self._sound_choice_changed))
        self._slider(sb, "Hit volume", R, "hit_volume", 0, 1.5)
        self._slider(sb, "Music volume", R, "music_volume", 0, 1.5)
        self._check(sb, "Random pitch (like in-game)", R, "hit_pitch_variation")
        ctk.CTkButton(sb, text="Custom Sound File…", command=self._pick_sound).pack(
            fill="x", padx=8, pady=4)
        self.sound_src_lbl = ctk.CTkLabel(sb, text="", anchor="w", text_color="#8a93a3",
                                          wraplength=330, justify="left")
        self.sound_src_lbl.pack(fill="x", padx=8)

        self._section(sb, "Render")
        self._choice(sb, "Slow-mo frames", R, "interp", ["nearest", "blend", "flow"])
        self._slider(sb, "Resolution scale", R, "scale", 0.25, 1.0)
        self._slider(sb, "Quality (CRF)", R, "crf", 12, 30, "{:.0f}", 18, True)
        self._choice(sb, "Encoder (auto = GPU)", R, "encoder", ["auto", "x264"])

    def _build_main(self):
        m = ctk.CTkFrame(self)
        m.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)
        m.grid_columnconfigure(1, weight=1)
        m.grid_rowconfigure(0, weight=1)
        self.preview = ctk.CTkLabel(m, text="Preview", width=PREVIEW_W, height=PREVIEW_H,
                                    fg_color="#0e0f12", corner_radius=6)
        self.preview.grid(row=0, column=0, padx=8, pady=8, sticky="n")
        info = ctk.CTkFrame(m, fg_color="transparent")
        info.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        info.grid_rowconfigure(1, weight=1)
        info.grid_columnconfigure(0, weight=1)
        self.preview_caption = ctk.CTkLabel(info, text="Click the timeline to preview frames.",
                                            anchor="w")
        self.preview_caption.grid(row=0, column=0, sticky="ew")
        self.summary = ctk.CTkTextbox(info, font=ctk.CTkFont(family="Consolas", size=12))
        self.summary.grid(row=1, column=0, sticky="nsew")
        self._build_render_bar(info)
        self._set_summary("1. Pick a gameplay video and a music track.\n"
                          "2. Analyze Media (beats, drop, hits).\n"
                          "3. Fix markers on the timeline, then Recalculate Alignment.\n"
                          "4. Render.")

    def _build_timeline(self):
        wrap = ctk.CTkFrame(self)
        wrap.grid(row=2, column=1, sticky="ew", padx=(4, 8), pady=(4, 8))
        wrap.grid_columnconfigure(0, weight=1)
        bar = ctk.CTkFrame(wrap, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        ctk.CTkButton(bar, text="⏮", width=36, command=lambda: self.player.seek(0)).pack(
            side="left", padx=(0, 2))
        self.play_btn = ctk.CTkButton(bar, text="▶ Play", width=86, fg_color="#c0392b",
                                      hover_color="#962d22", command=self._toggle_play)
        self.play_btn.pack(side="left", padx=2)
        self.time_lbl = ctk.CTkLabel(bar, text="0:00.00 / 0:00.00", width=120,
                                     font=ctk.CTkFont(family="Consolas", size=12))
        self.time_lbl.pack(side="left", padx=(4, 10))
        buttons = [("Delete", lambda: self.timeline.delete_selected()),
                   ("Split Combo", lambda: self.timeline.split_selected()),
                   ("Merge ← Prev", lambda: self.timeline.merge_selected()),
                   ("Auto-Group", self._auto_group),
                   ("Fit", lambda: self.timeline.zoom_fit())]
        for text, cmd in buttons:
            ctk.CTkButton(bar, text=text, width=40, command=cmd).pack(side="left", padx=2)
        self.live_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(bar, text="Live", variable=self.live_var, width=60).pack(side="right", padx=4)
        self.recalc_btn = ctk.CTkButton(bar, text="⟳ Recalculate Alignment", width=215,
                                        fg_color="#2e9d6a", hover_color="#237a52",
                                        command=self._recalculate)
        self.recalc_btn.pack(side="right", padx=4)
        self.timeline = Timeline(wrap, self.project, on_change=self._markers_changed,
                                 on_select=self._selected, on_click=self._timeline_click,
                                 snap_hit=self._snap_hit,
                                 scale=ctk.ScalingTracker.get_widget_scaling(wrap))
        self.timeline.grid(row=1, column=0, sticky="ew", padx=4, pady=(2, 4))
        ctk.CTkLabel(wrap, text="Click marker = select · Double-click = add beat/hit · "
                               "Right-click = delete · Wheel = zoom · Drag = pan · "
                               "Space play · Del, S split, M merge, F fit",
                     text_color="#8a93a3", anchor="w").grid(row=2, column=0, sticky="ew", padx=6)

    def _build_render_bar(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        f.grid_columnconfigure(2, weight=1)
        self.render_btn = ctk.CTkButton(f, text="⬇ Export MP4…", width=160, height=34,
                                        command=self._export)
        self.render_btn.grid(row=0, column=0, padx=(0, 6))
        self.cancel_btn = ctk.CTkButton(f, text="Cancel", width=72, fg_color="#7a2e3b",
                                        hover_color="#5c222c", command=self.cancel.set,
                                        state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=4)
        self.progress = ctk.CTkProgressBar(f)
        self.progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.progress.set(0)
        self.status = ctk.CTkLabel(f, text="Ready", anchor="w")
        self.status.grid(row=0, column=2, sticky="ew", padx=(10, 0))

    # ============================================================ params
    def _pull_params(self):
        self._suspend = True
        try:
            for p in self.params:
                p.pull()
        finally:
            self._suspend = False
        self._update_ranges()
        self._refresh_sound_label()
        self._refresh_grid_label()
        self._refresh_static_lock()

    def _refresh_static_lock(self):
        """Static beat = one hit on every beat: those settings are implied."""
        static = self.project.sync.static_grid
        for attr in ("fill_every_beat", "combo_spacing"):
            w = getattr(self, "_widgets", {}).get(attr)
            if w is not None:
                w.configure(state="disabled" if static else "normal")

    def _update_ranges(self):
        vd = max(1.0, self.project.video_duration or 60)
        md = max(1.0, self.project.music_duration or 60)
        for s in (self.intro_start_slider, self.intro_end_slider):
            s.configure(to=vd, number_of_steps=int(vd * 20))
        self.drop_slider.configure(to=md, number_of_steps=int(md * 20))
        # re-apply values so sliders repaint within the new range
        for p in self.params:
            if p.attr in ("intro_start", "intro_end", "drop_time"):
                p.pull()

    def _param_changed(self, attr=None):
        if getattr(self, "_suspend", False):
            return
        if attr in ("static_grid", "bpm_override", "grid_offset_ms"):
            self.project.apply_beat_grid()
            self._refresh_grid_label()
            self._refresh_static_lock()
        elif attr in ("combo_gap", "combo_regularity"):
            self.project.regroup()
        self._refresh_sound_label()
        if self.project.markers.hits or self.project.markers.beats:
            self.timeline.redraw()
            if self.live_var.get():
                self._schedule_recalc()

    def _schedule_recalc(self):
        if self._recalc_job:
            self.after_cancel(self._recalc_job)
        self._recalc_job = self.after(250, self._recalculate)

    # =========================================================== actions
    def _pick_video(self):
        p = filedialog.askopenfilename(title="Gameplay video", filetypes=VIDEO_TYPES)
        if p:
            self.video_var.set(p)
            self._reanalyze_if_analyzed()
            if not self.out_var.get():
                self.out_var.set(default_output_path(p))

    def _pick_music(self):
        p = filedialog.askopenfilename(title="Music track", filetypes=AUDIO_TYPES)
        if p:
            self.music_var.set(p)
            self._reanalyze_if_analyzed()

    def _pick_output(self):
        p = filedialog.asksaveasfilename(title="Output", defaultextension=".mp4",
                                         filetypes=[("MP4", "*.mp4")])
        if p:
            self.out_var.set(p)

    def _sync_paths(self):
        self.project.video_path = self.video_var.get().strip()
        self.project.music_path = self.music_var.get().strip()
        self.project.output_path = self.out_var.get().strip()

    def _reanalyze_if_analyzed(self):
        """A new file was picked after an analysis: analyse just that file."""
        if self.project.analyzed and not self.busy:
            self._analyze()

    def _analyze(self):
        if self.busy:
            return
        self._sync_paths()
        for path, name in ((self.project.video_path, "video"), (self.project.music_path, "music")):
            if not os.path.isfile(path):
                messagebox.showerror("Hit-Sync", f"Please choose a valid {name} file.")
                return
        need_a, need_v = self.project.stale()
        label = ("Analyzing music only" if need_a and not need_v else
                 "Analyzing video only" if need_v and not need_a else "Analyzing")
        self.player.invalidate()
        self._run_bg(label, self.project.analyze, self._analysis_done)

    def _analysis_done(self, result):
        did_a, did_v = result if isinstance(result, tuple) else (True, True)
        self.timeline.set_score(self.project.video.times,
                                hit_score(self.project.video, self.project.detect))
        self._pull_params()
        self._recalculate()
        self.timeline.zoom_fit()
        what = ("Music re-analyzed" if did_a and not did_v else
                "Video re-analyzed" if did_v and not did_a else "Analysis done")
        self._set_status(f"{what} · {len(self.project.markers.beats)} beats · "
                         f"{len(self.project.markers.hits)} hits · "
                         f"{self.project.grid_bpm:.1f} BPM")

    def _auto_drop(self):
        if not self.project.audio:
            return self._need_analysis()
        t = self.project.auto_drop()
        self._pull_params()
        self._set_status(f"Drop detected at {t:.2f}s (largest RMS energy jump)")
        self._recalculate()

    def _auto_intro(self):
        if not self.project.video:
            return self._need_analysis()
        self.project.auto_intro()
        self._pull_params()
        self._recalculate()

    def _redetect(self):
        if not self.project.video:
            return self._need_analysis()
        self.project.redetect_hits()
        self.timeline.set_score(self.project.video.times,
                                hit_score(self.project.video, self.project.detect))
        self.timeline.selected = None
        self._recalculate()
        self._set_status(f"Re-detected {len(self.project.markers.hits)} hits from cached signals")

    def _drop_here(self):
        """Drop = current playhead (snapped to the beat grid)."""
        t = self.player.position
        beats = self.project.markers.beats
        if beats:
            t = min(beats, key=lambda b: abs(b - t))
        self.project.sync.drop_time = t
        self._pull_params()
        self._set_status(f"Drop set to {t:.2f}s")
        self._recalculate()

    def _combat_here(self):
        """Combat begins at the selected hit; intro range resized for slow-mo."""
        sel = self.timeline.selected
        if not sel or sel[0] != "hit":
            messagebox.showinfo("Hit-Sync", "Select the first hit of the combo on the VIDEO track.")
            return
        h = self.project.markers.hits[sel[1]].t
        s = self.project.sync
        L = max(s.drop_time, 1.0)
        if s.intro_length > 0:
            L = min(L, s.intro_length)
        R = min(s.intro_ramp, L / 2)
        want = s.intro_target_speed * (L - R) + R * (s.intro_target_speed + 1) / 2
        s.intro_end, s.intro_start = h, max(0.0, h - want)
        self._pull_params()
        self._set_status(f"Combat begins at {h:.2f}s (intro {s.intro_start:.2f}-{h:.2f}s)")
        self._recalculate()

    def _refresh_grid_label(self):
        if hasattr(self, "grid_lbl"):
            bpm = self.project.grid_bpm
            self.grid_lbl.configure(text=f"Grid tempo: {bpm:.2f} BPM" if bpm else "")

    def _auto_group(self):
        self.project.regroup()
        self._markers_changed("Combos regrouped by gap")

    def _markers_changed(self, what):
        self._set_status(f"{what} · alignment {'updated' if self.live_var.get() else 'stale'}")
        if self.live_var.get():
            self._recalculate()
        else:
            self.recalc_btn.configure(fg_color="#b8860b")

    def _recalculate(self):
        self._recalc_job = None
        if not (self.project.markers.beats or self.project.markers.hits):
            return
        t0 = time.perf_counter()
        try:
            sched = self.project.recalculate()
        except Exception as exc:  # show engine problems instead of crashing the UI
            traceback.print_exc()
            self._set_status(f"Alignment failed: {exc}")
            return
        ms = (time.perf_counter() - t0) * 1000
        self.player.invalidate()
        self._player_position(min(max(self.player.position, sched.start), sched.end))
        self.recalc_btn.configure(fg_color="#2e9d6a")
        self._set_summary(sched.summary() + f"\n(recalculated in {ms:.1f} ms, no media re-read)")
        self.timeline.redraw()

    def _export(self, ask: bool = True):
        """Save-as dialog for the .mp4, then render it in the background."""
        if self.busy:
            return
        self._sync_paths()
        if not self.project.analyzed:
            return self._need_analysis()
        if not find_ffmpeg():
            messagebox.showerror("Hit-Sync", "ffmpeg was not found.")
            return
        out = self.project.output_path or default_output_path(self.project.video_path)
        if ask:
            out = filedialog.asksaveasfilename(
                title="Export video as", defaultextension=".mp4",
                initialdir=os.path.dirname(out) or None, initialfile=os.path.basename(out),
                filetypes=[("MP4 video", "*.mp4")])
            if not out:
                return
        self.out_var.set(out)
        self.project.output_path = out
        self.player.pause()
        self._recalculate()
        self._run_bg("Exporting", self.project.render, self._export_done)

    def _export_done(self, path):
        self._set_status(f"Exported {os.path.basename(path)}")
        if messagebox.askyesno("Hit-Sync", f"Export finished:\n{path}\n\nOpen the video now?"):
            if hasattr(os, "startfile"):
                os.startfile(path)

    def _save_project(self, save_as: bool = False):
        self._sync_paths()
        path = self.project_path
        if save_as or not path:
            base = os.path.splitext(os.path.basename(self.project.video_path or "montage"))[0]
            path = filedialog.asksaveasfilename(
                title="Save project as", defaultextension=".json",
                initialfile=f"{base}.hitsync.json",
                filetypes=[("Hit-Sync project", "*.json")])
            if not path:
                return
        self.project.save(path)
        self.project_path = path
        self._set_status(f"Project saved: {os.path.basename(path)}")

    def _load_project(self):
        p = filedialog.askopenfilename(title="Open project",
                                       filetypes=[("Hit-Sync project", "*.json")])
        if not p:
            return
        try:
            proj = Project.load(p)
        except Exception as exc:
            messagebox.showerror("Hit-Sync", f"Could not load project:\n{exc}")
            return
        self.player.invalidate()
        self.project = proj
        self.project_path = p
        self.timeline.project = proj
        self.video_var.set(proj.video_path)
        self.music_var.set(proj.music_path)
        self.out_var.set(proj.output_path)
        if proj.video is not None:
            self.timeline.set_score(proj.video.times, hit_score(proj.video, proj.detect))
        self._pull_params()
        self._recalculate()
        self.timeline.zoom_fit()
        self._set_status(f"Loaded {p}")

    # ========================================================= timeline cb
    def _snap_hit(self, t):
        return snap_to_peak(self.project.video, self.project.detect, t)

    def _selected(self, sel):
        if not sel:
            return
        kind, i = sel
        m = self.project.markers
        if kind == "hit" and i < len(m.hits):
            h = m.hits[i]
            info = ""
            if self.project.schedule:
                pl = next((p for p in self.project.schedule.placements if p.hit_index == i), None)
                if pl and pl.out_t is not None:
                    info = (f" → output {pl.out_t:.3f}s [{pl.status}"
                            f"{', err %+.0f ms' % pl.error_ms if pl.error_ms else ''}]")
            self._show_source_frame(h.t, f"Hit #{i + 1} @ {h.t:.3f}s (video){info}")
        elif kind == "beat" and i < len(m.beats):
            self._set_status(f"Beat #{i + 1} @ {m.beats[i]:.3f}s (music)")

    def _timeline_click(self, track, t):
        if track == "video":
            self._show_source_frame(t, f"Video @ {t:.3f}s")
        elif track in ("edit", "audio") and self.project.schedule and self.project.video:
            sched = self.project.schedule
            t = min(max(sched.start, t), sched.end)
            self.player.seek(t)
            if not self.player.playing:
                self._show_output_frame(t)

    def _show_source_frame(self, t, caption):
        if not self.project.video_path:
            return
        v = self.project.video
        size = self._preview_px(v.info.width, v.info.height)[1] if v else None
        frame = read_frame(self.project.video_path, t, size,
                           (v.pts_offset or 0.0) if v else 0.0)
        if frame is not None:
            self._set_preview(frame, caption)

    def _show_output_frame(self, t):
        from ..renderer import FrameSource, compose_frame

        sched = self.project.schedule
        if not sched.start <= t <= sched.end:
            return
        info = self.project.video.info
        size = self._preview_px(info.width, info.height)[1]
        src = FrameSource(self.project.video_path, info,
                          frame_times=self.project.video.frame_times, size=size,
                          pts_offset=self.project.video.pts_offset or 0.0)
        try:
            frame = compose_frame(src, sched, t, self.project.render_params, size)
        finally:
            src.close()
        self._set_preview(frame, f"Output @ {t:.3f}s ← video {sched.src_time(t):.3f}s · "
                                 f"speed {sched.speed_at(t):.2f}x · "
                                 f"bars {sched.letterbox_amount(t) * 100:.0f}%")

    def _preview_px(self, w, h):
        """Logical and physical preview sizes for a w x h frame."""
        scale = min(PREVIEW_W / w, PREVIEW_H / h)
        lw, lh = max(2, int(w * scale)), max(2, int(h * scale))
        dpi = ctk.ScalingTracker.get_widget_scaling(self.preview)
        return (lw, lh), (int(lw * dpi), int(lh * dpi))

    def _set_preview(self, frame, caption):
        h, w = frame.shape[:2]
        logical, physical = self._preview_px(w, h)
        if (w, h) != physical:
            frame = cv2.resize(frame, physical, interpolation=cv2.INTER_AREA)
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._preview_img = ctk.CTkImage(light_image=img, dark_image=img, size=logical)
        self.preview.configure(image=self._preview_img, text="")
        if caption is not None:
            self.preview_caption.configure(text=caption)

    # ========================================================== playback
    def _toggle_play(self):
        if self.project.schedule is None or self.project.video is None:
            return self._need_analysis()
        info = self.project.video.info
        self.player.preview_size = self._preview_px(info.width, info.height)[1]
        self.player.toggle()

    def _player_frame(self, frame, t):
        self._set_preview(frame, None)

    def _player_position(self, t):
        sched = self.project.schedule
        dur = sched.duration if sched else 0.0
        rel = max(0.0, t - sched.start) if sched else t
        self.time_lbl.configure(text=f"{_fmt(rel)} / {_fmt(dur)}")
        if sched is not None:
            self.timeline.set_playhead(t, sched.src_time(t), follow=self.player.playing)
            if self.player.playing:
                self.preview_caption.configure(
                    text=f"▶ {_fmt(t)} · video {sched.src_time(t):.2f}s · "
                         f"{sched.speed_at(t):.2f}x")

    def _player_state(self, playing):
        self.play_btn.configure(text="⏸ Pause" if playing else "▶ Play")

    # ============================================================ sound
    def _refresh_sound_label(self):
        if not hasattr(self, "sound_src_lbl"):
            return
        r = self.project.render_params
        try:
            _, desc = hit_samples(r.hit_sound, r.hit_sound_file) if find_ffmpeg() else ((), "")
        except Exception as exc:
            desc = f"could not load: {exc}"
        self.sound_src_lbl.configure(text=f"Source: {desc}" if desc else "")

    def _sound_choice_changed(self):
        r = self.project.render_params
        if r.hit_sound == "custom" and not r.hit_sound_file:
            self._pick_sound()

    def _pick_sound(self):
        p = filedialog.askopenfilename(title="Hit sound", filetypes=AUDIO_TYPES)
        if not p:
            return
        self.project.render_params.hit_sound_file = p
        self.sound_var.set("custom")     # triggers push + refresh
        self._refresh_sound_label()
        self.player.invalidate()

    def _on_close(self):
        self.player.pause()
        self.destroy()

    # ======================================================== background
    def _run_bg(self, label, fn, on_done):
        self.busy = True
        self.cancel.clear()
        for b in (self.analyze_btn, self.render_btn):
            b.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress.set(0)

        def progress(msg, frac):
            self.events.put(("progress", msg, frac))

        def worker():
            try:
                result = fn(progress=progress, cancel=self.cancel)
                self.events.put(("done", on_done, result))
            except Exception as exc:
                traceback.print_exc()
                self.events.put(("error", label, exc))

        threading.Thread(target=worker, daemon=True).start()
        self._set_status(f"{label}…")

    def _poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                if ev[0] == "progress":
                    _, msg, frac = ev
                    self.progress.set(float(np.clip(frac, 0, 1)))
                    self._set_status(msg)
                elif ev[0] == "done":
                    self._finish()
                    if self.cancel.is_set():
                        self._set_status("Cancelled")
                    else:
                        self.progress.set(1)
                        ev[1](ev[2])
                elif ev[0] == "error":
                    self._finish()
                    from ..renderer import RenderCancelled

                    if isinstance(ev[2], RenderCancelled):
                        self._set_status("Render cancelled")
                    else:
                        self._set_status(f"{ev[1]} failed: {ev[2]}")
                        messagebox.showerror("Hit-Sync", f"{ev[1]} failed:\n{ev[2]}")
        except queue.Empty:
            pass
        self.after(50, self._poll)

    def _finish(self):
        self.busy = False
        for b in (self.analyze_btn, self.render_btn):
            b.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    # ============================================================ misc
    def _need_analysis(self):
        messagebox.showinfo("Hit-Sync", "Analyze the media first.")

    def _set_status(self, text):
        self.status.configure(text=text if len(text) <= 60 else text[:59] + "…")

    def _set_summary(self, text):
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", text)
        self.summary.configure(state="disabled")

    def _check_deps(self):
        if not find_ffmpeg():
            self._set_status("⚠ ffmpeg not found - rendering disabled until installed")


def _fmt(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    app = HitSyncApp()
    app.mainloop()


if __name__ == "__main__":
    main()
