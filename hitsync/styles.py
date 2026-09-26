"""One-click looks. A style is just a set of render + edit settings; picking
one overwrites those settings, and changing any of them afterwards turns the
style into "custom"."""
from __future__ import annotations

STYLES = {
    "clean": {
        "render": dict(filter="none", filter_strength=0.0, vignette=0.0, motion_blur=0.0,
                       hit_zoom=0.0, hit_shake=0.0, hit_flash=0.0, hit_rgb=0.0, beat_pulse=0.0,
                       fade_in="black", fade_in_len=0.5, fade_out="black", fade_out_len=1.5),
        "sync": dict(velocity=0.0, transition="cut", letterbox_mode="slowmo", intro_flash=True),
    },
    "montage": {
        "render": dict(filter="vibrant", filter_strength=0.6, vignette=0.2, motion_blur=0.35,
                       hit_zoom=0.04, hit_shake=0.0, hit_flash=0.12, hit_rgb=0.0, beat_pulse=0.0,
                       fade_in="black", fade_in_len=0.6, fade_out="black", fade_out_len=1.5),
        "sync": dict(velocity=0.3, transition="flash", letterbox_mode="slowmo", intro_flash=True),
    },
    "hype": {
        "render": dict(filter="punchy", filter_strength=0.8, vignette=0.3, motion_blur=0.5,
                       hit_zoom=0.07, hit_shake=0.5, hit_flash=0.3, hit_rgb=0.6, beat_pulse=0.4,
                       fade_in="white", fade_in_len=0.4, fade_out="white", fade_out_len=1.0),
        "sync": dict(velocity=0.5, transition="zoom", letterbox_mode="slowmo", intro_flash=True),
    },
    "cinematic": {
        "render": dict(filter="cinematic", filter_strength=0.8, vignette=0.35, motion_blur=0.45,
                       hit_zoom=0.02, hit_shake=0.0, hit_flash=0.0, hit_rgb=0.0, beat_pulse=0.0,
                       fade_in="black", fade_in_len=1.0, fade_out="black", fade_out_len=2.5),
        "sync": dict(velocity=0.2, transition="dip", letterbox_mode="first combo",
                     intro_flash=True),
    },
}
LABELS = {"clean": "Clean", "montage": "Montage", "hype": "Hype", "cinematic": "Cinematic",
          "custom": "Custom"}
DEFAULT = "montage"


def apply_style(project, name: str):
    style = STYLES.get(name)
    if style is None:
        return
    for key, value in style["render"].items():
        setattr(project.render_params, key, value)
    for key, value in style["sync"].items():
        setattr(project.sync, key, value)
    project.render_params.style = name


def style_keys() -> set:
    """Every setting a style controls (editing one makes the style custom)."""
    keys = set()
    for s in STYLES.values():
        keys |= set(s["render"]) | set(s["sync"])
    return keys
