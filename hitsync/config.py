"""Tunable parameters for detection, alignment and rendering.

All three groups are plain dataclasses so they can be edited by the GUI,
serialised into a project file, and passed around without side effects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


def _from_dict(cls, data: dict | None):
    """Build a dataclass from a dict, ignoring unknown keys (forward compat)."""
    obj = cls()
    if not data:
        return obj
    names = {f.name: f for f in fields(cls)}
    for key, value in data.items():
        if key in names:
            default = getattr(obj, key)
            try:
                value = type(default)(value) if default is not None else value
            except (TypeError, ValueError):
                continue
            setattr(obj, key, value)
    return obj


@dataclass
class DetectParams:
    """Hit detection (video analysis) settings."""

    roi_w: float = 0.45            # centre ROI width as fraction of frame
    roi_h: float = 0.60            # centre ROI height as fraction of frame
    analysis_width: int = 320      # frames are downscaled to this width
    max_analysis_fps: float = 60.0
    red_weight: float = 1.0        # weight of damage-tint onset
    motion_weight: float = 0.35    # weight of screen-velocity change
    sensitivity: float = 0.5       # 0 = strict, 1 = catch everything
    # Minecraft entities are invulnerable for 10 ticks (0.5 s) after taking
    # damage, so two real hits can't be closer than ~0.45 s.
    min_hit_interval: float = 0.40  # seconds; peaks closer than this merge

    to_dict = asdict

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)


@dataclass
class SyncParams:
    """Alignment / edit-decision settings (all cheap to recompute)."""

    # --- 1. Intro speed ramp -------------------------------------------------
    intro_enabled: bool = True
    intro_start: float = 0.0        # source seconds
    intro_end: float = 0.0          # source seconds: combat begins here
    drop_time: float = 0.0          # music seconds: the beat drop
    intro_ramp: float = 0.6         # output seconds spent easing back to 1.0x
    intro_min_speed: float = 0.15   # slowest allowed slow-motion factor
    intro_target_speed: float = 0.4  # slow-mo speed of the intro footage
    # Length of the intro in the montage (s). The song is cut to start this
    # long before the drop (snapped to whole beats) and only the footage that
    # fills it in slow-mo is used. 0 = keep the whole music intro.
    intro_length: float = 5.0
    intro_auto: bool = True         # intro length follows the song's own intro

    # --- Beat grid -----------------------------------------------------------
    static_grid: bool = True        # one constant tempo for the whole song
    bpm_override: float = 0.0       # >0 forces the grid tempo
    grid_offset_ms: float = 0.0     # nudge the static grid's phase

    # --- 2. Dynamic tempo matching -------------------------------------------
    tempo_matching: bool = True
    tempo_strength: float = 1.0     # 0 = off, 1 = speed proportional to BPM
    min_speed: float = 0.25
    max_speed: float = 3.0

    # --- 4. Jitter tolerance & auto cutting ----------------------------------
    jitter_tolerance_ms: float = 80.0
    lock_mode: str = "ramp"         # "ramp" (micro speed ramp) | "trim" (drop ms)
    combo_spacing: str = "auto"     # beats per hit in a combo: auto|0.5|1|2|3|4
    ramp_min_speed: float = 0.6     # bigger gaps than the tolerance are fixed by
    ramp_max_speed: float = 1.7     # speeding up/slowing down within these bounds
    combo_gap: float = 1.2          # source s between hits that splits combos
    combo_regularity: float = 0.30  # gap may differ from the combo's rhythm by 30%
    min_combo_len: int = 10         # shorter runs aren't combos: cut from the edit
    # Every static beat after the drop gets a hit: hits step exactly one grid
    # step at a time and the next combo starts on the beat right after the
    # previous combo's last hit (the dead footage is cut inside that beat).
    fill_every_beat: bool = True
    long_gap: float = 0.6           # dead source s between combos that forces a cut
    pre_roll: float = 0.35          # source s kept before a combo's first hit
    post_roll: float = 0.35         # source s kept after a combo's last hit
    outro: float = 1.0              # source s kept after the final combo
    transition: str = "cut"         # "cut" | "flash"
    flash_duration: float = 0.12

    # --- 3. Cinematic letterbox ----------------------------------------------
    letterbox_enabled: bool = True
    letterbox_aspect: float = 2.39  # target aspect ratio when bars are fully in
    letterbox_fade: float = 0.25    # ease in/out duration (s)
    letterbox_lead: float = 0.10    # bars start this long before the first hit
    letterbox_hold: float = 0.15    # bars start leaving this long after last hit
    min_combo_hits: int = 2         # combos shorter than this get no bars

    @property
    def every_beat(self) -> bool:
        """A static beat means *every* beat of it gets a hit, on the beat."""
        return self.static_grid or self.fill_every_beat

    to_dict = asdict

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)


@dataclass
class RenderParams:
    output_fps: float = 0.0         # 0 = same as source (capped at 60)
    scale: float = 1.0              # output resolution scale
    interp: str = "flow"            # slow-mo frame synthesis: nearest|blend|flow
    crf: int = 18                   # quality (lower = better); hardware encoders map it
    preset: str = "medium"          # x264 speed preset
    # "auto" = GPU encoder (NVIDIA/AMD/Intel) when one works, else x264 on the CPU
    encoder: str = "auto"           # auto | x264 | nvenc | amf | qsv
    audio_fade_out: float = 1.5
    # hit sounds (see audio_mix.HIT_SOUND_CHOICES)
    hit_sound: str = "original"     # original|off|classic|strong|crit|knockback|custom
    hit_sound_file: str = ""        # used when hit_sound == "custom"
    hit_volume: float = 0.8
    music_volume: float = 1.0
    hit_pitch_variation: bool = True

    to_dict = asdict

    @classmethod
    def from_dict(cls, d):
        return _from_dict(cls, d)
