"""Advanced settings: every knob, grouped in tabs, built from one spec list.

Regular users never need these; the main window has the three that matter
(hit volume, music volume, beats per hit)."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
                               QTabWidget, QVBoxLayout, QWidget)

from ..audio_mix import HIT_SOUND_CHOICES
from . import theme
from .panels import _scroll
from .widgets import ValueSlider, button, label

# (tab, label, params ("sync"|"detect"|"render"), attr, kind, arg, tooltip)
# kind: slider -> arg = (lo, hi, fmt, integer, percent) ; check ; choice -> [(value, text)]
SPEC = [
    ("Intro & start", "Slow-mo intro", "sync", "intro_enabled", "check", None,
     "A slow-motion intro before the music starts"),
    ("Intro & start", "Intro length follows the song", "sync", "intro_auto", "check", None,
     "Use the song's own intro (or 4 bars) as the slow-mo part"),
    ("Intro & start", "Intro length (s)", "sync", "intro_length", "slider",
     (0, 20, "{:.1f}", False, False), "0 = keep the whole song intro"),
    ("Intro & start", "Intro slow-mo speed", "sync", "intro_target_speed", "slider",
     (0.1, 0.9, "{:.2f}", False, False), ""),
    ("Intro & start", "Ramp back to 1.0x (s)", "sync", "intro_ramp", "slider",
     (0.1, 2.0, "{:.2f}", False, False), ""),
    ("Intro & start", "Flash when slow-mo ends", "sync", "intro_flash", "check", None, ""),
    ("Intro & start", "Lead-in length (beats)", "sync", "lead_in_beats", "slider",
     (1, 16, "{:.0f}", True, False), "Slow-mo lead-in before a combo (tick it in Combos)"),
    ("Intro & start", "Lead-in speed", "sync", "lead_in_speed", "slider",
     (0.15, 0.9, "{:.2f}", False, False), ""),

    ("Beat grid", "Static beat (one tempo)", "sync", "static_grid", "check", None,
     "Off = follow a tempo that changes (tracked beats)"),
    ("Beat grid", "BPM (0 = detected)", "sync", "bpm_override", "slider",
     (0, 220, "{:.2f}", False, False), "Force the tempo of song 1"),
    ("Beat grid", "Grid offset (ms)", "sync", "grid_offset_ms", "slider",
     (-300, 300, "{:.0f}", False, False), "Nudge the beat grid of song 1"),
    ("Beat grid", "Tempo matching (dynamic grid)", "sync", "tempo_matching", "check", None, ""),

    ("Combos & sync", "Min hits per combo", "sync", "min_combo_len", "slider",
     (2, 30, "{:.0f}", True, False), "Shorter runs are not combos"),
    ("Combos & sync", "Rhythm tolerance", "sync", "combo_regularity", "slider",
     (0.05, 0.6, "{:.0%}", False, True), "How steady a combo's rhythm must be"),
    ("Combos & sync", "Max gap in a combo (s)", "sync", "combo_gap", "slider",
     (0.3, 3.0, "{:.2f}", False, False), ""),
    ("Combos & sync", "Tolerance (ms)", "sync", "jitter_tolerance_ms", "slider",
     (0, 200, "{:.0f}", False, False), "Hits this close to the beat get a micro ramp"),
    ("Combos & sync", "Lock method", "sync", "lock_mode", "choice",
     [("ramp", "Micro speed ramp"), ("trim", "Trim late milliseconds")], ""),
    ("Combos & sync", "Slowest speed-ramp", "sync", "ramp_min_speed", "slider",
     (0.3, 1.0, "{:.2f}", False, False), ""),
    ("Combos & sync", "Fastest speed-ramp", "sync", "ramp_max_speed", "slider",
     (1.0, 3.0, "{:.2f}", False, False), ""),
    ("Combos & sync", "Post-roll (s)", "sync", "post_roll", "slider",
     (0.0, 1.5, "{:.2f}", False, False), "Footage kept after a combo's last hit"),

    ("Hit detection", "Sensitivity", "detect", "sensitivity", "slider",
     (0, 1, "{:.2f}", False, False), ""),
    ("Hit detection", "Screen-velocity weight", "detect", "motion_weight", "slider",
     (0, 1, "{:.2f}", False, False), ""),
    ("Hit detection", "Min hit spacing (s)", "detect", "min_hit_interval", "slider",
     (0.1, 1.0, "{:.2f}", False, False), ""),

    ("Hit sounds", "Sound", "render", "hit_sound", "choice",
     [(c, c.capitalize()) for c in HIT_SOUND_CHOICES], "Original = your recording's own hits"),
    ("Hit sounds", "Random pitch (like in-game)", "render", "hit_pitch_variation", "check",
     None, ""),

    ("Letterbox", "Bars", "sync", "letterbox_mode", "choice",
     [("slowmo", "Slow-mo parts only"), ("first combo", "Intro + first combo"),
      ("combos", "Every combo"), ("off", "Off")], ""),
    ("Letterbox", "Bar aspect ratio", "sync", "letterbox_aspect", "slider",
     (1.85, 3.0, "{:.2f}", False, False), ""),
    ("Letterbox", "Ease in/out (s)", "sync", "letterbox_fade", "slider",
     (0.05, 1.0, "{:.2f}", False, False), ""),

    ("Export", "Slow-mo frames", "render", "interp", "choice",
     [("flow", "Optical flow (smooth)"), ("blend", "Blend"), ("nearest", "Nearest")], ""),
    ("Export", "Resolution scale", "render", "scale", "slider",
     (0.25, 1.0, "{:.2f}", False, False), ""),
    ("Export", "Quality (CRF, lower = better)", "render", "crf", "slider",
     (12, 30, "{:.0f}", True, False), ""),
    ("Export", "Encoder", "render", "encoder", "choice",
     [("auto", "Auto (GPU when available)"), ("x264", "x264 (CPU)")], ""),
    ("Export", "x264 speed", "render", "preset", "choice",
     [("veryfast", "Very fast"), ("faster", "Faster"), ("fast", "Fast"),
      ("medium", "Medium")], ""),
]


class AdvancedDialog(QDialog):
    changed = Signal(str)          # attribute name
    redetect = Signal()
    custom_sound = Signal()

    def __init__(self, project_getter, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced settings")
        self.resize(820, 640)
        self.P = project_getter
        lay = QVBoxLayout(self)
        lay.addWidget(label("For fine-tuning. Everything here has good automatic defaults.",
                            muted=True))
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs, 1)
        self.controls = []
        pages = {}
        for tab, text, which, attr, kind, arg, tip in SPEC:
            if tab not in pages:
                inner = QWidget()
                v = QVBoxLayout(inner)
                v.setContentsMargins(4, 8, 8, 8)
                pages[tab] = v
                self.tabs.addTab(_scroll(inner), tab.replace("&", "&&"))
            v = pages[tab]
            if kind == "slider":
                lo, hi, fmt, integer, percent = arg
                w = ValueSlider(text, lo, hi, 0, fmt, integer=integer, percent=percent, tip=tip,
                                label_width=190)
                w.changed.connect(lambda val, wh=which, a=attr, i=integer:
                                  self._push(wh, a, int(val) if i else float(val)))
                v.addWidget(w)
            elif kind == "check":
                w = QCheckBox(text)
                w.setToolTip(tip)
                w.toggled.connect(lambda on, wh=which, a=attr: self._push(wh, a, bool(on)))
                v.addWidget(w)
            else:
                row = QHBoxLayout()
                lb = QLabel(text)
                lb.setMinimumWidth(190)
                row.addWidget(lb)
                w = QComboBox()
                w.setToolTip(tip)
                for val, t in arg:
                    w.addItem(t, val)
                w.currentIndexChanged.connect(lambda _, w=w, wh=which, a=attr:
                                              self._push(wh, a, w.currentData()))
                row.addWidget(w, 1)
                v.addLayout(row)
            self.controls.append((w, which, attr, kind))
        pages["Hit detection"].addWidget(button("Re-detect hits (instant, no re-scan)",
                                                slot=lambda: self.redetect.emit()))
        pages["Hit sounds"].addWidget(button("Choose a custom sound file…",
                                             slot=lambda: self.custom_sound.emit()))
        self.sound_lbl = label("", muted=True, wrap=True)
        pages["Hit sounds"].addWidget(self.sound_lbl)
        for v in pages.values():
            v.addStretch(1)
        close = button("Done", primary=True, slot=self.close)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close)
        lay.addLayout(row)

    def _obj(self, which):
        p = self.P()
        return {"sync": p.sync, "detect": p.detect, "render": p.render_params}[which]

    def _push(self, which, attr, value):
        if getattr(self, "_pulling", False):
            return
        setattr(self._obj(which), attr, value)
        self.changed.emit(attr)

    def pull(self):
        self._pulling = True
        try:
            for w, which, attr, kind in self.controls:
                v = getattr(self._obj(which), attr)
                w.blockSignals(True)
                if kind == "slider":
                    w.set(v)
                elif kind == "check":
                    w.setChecked(bool(v))
                else:
                    i = w.findData(v)
                    w.setCurrentIndex(max(0, i))
                w.blockSignals(False)
        finally:
            self._pulling = False
        try:
            from ..audio_mix import hit_samples
            from ..ffmpeg_utils import find_ffmpeg

            r = self.P().render_params
            desc = hit_samples(r.hit_sound, r.hit_sound_file)[1] if find_ffmpeg() else ""
            self.sound_lbl.setText(f"Source: {desc}" if desc else "")
        except Exception as exc:                          # noqa: BLE001
            self.sound_lbl.setText(f"Could not load the sound: {exc}")
        self.sound_lbl.setStyleSheet(f"color: {theme.MUTED};")


__all__ = ["AdvancedDialog", "Qt"]
