"""YouTuber-style captions: bold text with a thick outline, a glow or a
shadow, at the top, centre or bottom of the screen, animated in and out.

Each caption is rendered once (Pillow, with its stroke and glow) into an RGBA
sprite and cached; per frame it is only scaled/moved/faded and alpha-blended
onto the frame inside its own box, so captions cost next to nothing.
"""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "fonts")


@dataclass(frozen=True)
class CaptionStyle:
    fill: tuple                      # RGBA
    outline: tuple                   # RGBA
    shadow: tuple                    # RGBA (alpha 0 = none)
    outline_w: float                 # fraction of the font size
    font: str
    glow: Optional[tuple] = None     # RGB of the glow around the outline
    glow_radius: float = 0.18        # blur radius, fraction of the font size
    glow_strength: float = 2.0       # >1 = more vivid (the glow is stacked)


STYLES = {
    # white letters, purple outline, vivid purple glow, rounded font
    "glow": CaptionStyle((255, 255, 255, 255), (138, 43, 226, 255), (0, 0, 0, 0), 0.10,
                         "Fredoka", glow=(180, 90, 255)),
    "youtuber": CaptionStyle((255, 255, 255, 255), (0, 0, 0, 255), (0, 0, 0, 170), 0.11, "Anton"),
    "yellow": CaptionStyle((255, 226, 0, 255), (0, 0, 0, 255), (0, 0, 0, 170), 0.11, "Anton"),
    "red": CaptionStyle((255, 59, 59, 255), (255, 255, 255, 255), (0, 0, 0, 160), 0.09, "Anton"),
    "impact": CaptionStyle((255, 255, 255, 255), (0, 0, 0, 255), (0, 0, 0, 0), 0.08, "Montserrat"),
    "minimal": CaptionStyle((255, 255, 255, 255), (0, 0, 0, 200), (0, 0, 0, 120), 0.04,
                            "Montserrat"),
}
LABELS = {"glow": "Glow (purple)", "youtuber": "YouTuber", "yellow": "Yellow", "red": "Red",
          "impact": "Impact", "minimal": "Minimal"}
DEFAULT_STYLE = "glow"
ANIMATIONS = ["pop", "slide", "fade", "typewriter", "none"]
POSITIONS = ["top", "center", "bottom"]


@dataclass
class TextItem:
    text: str
    start: float                 # music time
    end: float
    position: str = "bottom"     # top | center | bottom
    style: str = DEFAULT_STYLE
    animation: str = "pop"
    size: float = 1.0            # 1 = 7.5% of the frame height

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(str(d.get("text", "")), float(d["start"]), float(d["end"]),
                   str(d.get("position", "bottom")), str(d.get("style", DEFAULT_STYLE)),
                   str(d.get("animation", "pop")), float(d.get("size", 1.0)))


# ------------------------------------------------------------------ fonts
def _system_fonts(name: str) -> list[str]:
    if sys.platform == "win32":
        root = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
        files = ["impact.ttf", "ariblk.ttf", "arialbd.ttf"]
    elif sys.platform == "darwin":
        root = "/System/Library/Fonts/Supplemental"
        files = ["Impact.ttf", "Arial Black.ttf", "Arial Bold.ttf"]
    else:
        root = "/usr/share/fonts"
        return sorted(glob.glob(os.path.join(root, "**", "DejaVuSans-Bold.ttf"), recursive=True))
    return [os.path.join(root, f) for f in files]


@lru_cache(maxsize=16)
def font_path(name: str = "Anton") -> str | None:
    """The bundled font, else a bold system font, else None (Pillow default)."""
    bundled = {"Anton": "Anton-Regular.ttf", "Montserrat": "Montserrat-Black.ttf",
               "Fredoka": "Fredoka-Bold.ttf"}
    cands = []
    if os.path.isfile(name):                         # a custom font file
        cands.append(name)
    if name in bundled:
        cands.append(os.path.join(FONT_DIR, bundled[name]))
    cands += [os.path.join(FONT_DIR, f) for f in bundled.values()] + _system_fonts(name)
    return next((c for c in cands if os.path.isfile(c)), None)


@lru_cache(maxsize=64)
def _font(name: str, px: int):
    from PIL import ImageFont

    path = font_path(name)
    if path:
        return ImageFont.truetype(path, px)
    try:
        return ImageFont.load_default(px)
    except TypeError:                                 # old Pillow
        return ImageFont.load_default()


def _wrap(text: str, font, max_w: int) -> str:
    lines = []
    for para in text.split("\n"):
        words, line = para.split(" "), ""
        for w in words:
            trial = (line + " " + w).strip()
            if line and font.getlength(trial) > max_w:
                lines.append(line)
                line = w
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def sprite_pad(style: str, px: int) -> int:
    """Empty margin around the text inside its sprite (room for the outline,
    shadow and the widest glow blur, about 3 sigma, so it's never cut off)."""
    st = STYLES.get(style, STYLES[DEFAULT_STYLE])
    stroke_w = max(1, int(round(px * st.outline_w)))
    pad = stroke_w + int(px * 0.12)
    if st.glow:
        glow_r = max(1.0, px * st.glow_radius)
        pad += int(np.ceil(glow_r * 2 * 3)) + int(round(px * 0.06))
    return pad


@lru_cache(maxsize=128)
def render_sprite(text: str, style: str, px: int, max_w: int) -> np.ndarray:
    """(h, w, 4) float32 premultiplied-alpha RGBA sprite of the caption, BGR order."""
    from PIL import Image, ImageChops, ImageDraw, ImageFilter

    st = STYLES.get(style, STYLES[DEFAULT_STYLE])
    font = _font(st.font, px)
    text = _wrap(text, font, max_w)
    stroke_w = max(1, int(round(px * st.outline_w)))
    spacing = int(px * 0.12)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.multiline_textbbox((0, 0), text, font=font, stroke_width=stroke_w, align="center",
                                   spacing=spacing)
    box = (int(np.floor(box[0])), int(np.floor(box[1])), int(np.ceil(box[2])),
           int(np.ceil(box[3])))                     # (newer Pillow returns floats)
    glow_r = max(1.0, px * st.glow_radius) if st.glow else 0.0
    glow_w = stroke_w + int(round(px * 0.06)) if st.glow else 0
    pad = sprite_pad(style, px)
    w, h = box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad
    origin = (pad - box[0], pad - box[1])
    img = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))

    if st.glow:
        # a soft halo in the glow colour: a tight bright core plus a wide
        # soft one, so it reads as vivid neon (only the alpha is blurred, so
        # the colour stays pure out to the edge)
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).multiline_text(origin, text, font=font, fill=255,
                                            stroke_width=glow_w, stroke_fill=255,
                                            align="center", spacing=spacing)
        k = st.glow_strength
        core = mask.filter(ImageFilter.GaussianBlur(glow_r)).point(lambda v: min(255, int(v * k)))
        wide = mask.filter(ImageFilter.GaussianBlur(glow_r * 2)).point(lambda v: int(v * 0.7))
        halo = Image.new("RGBA", img.size, tuple(st.glow) + (0,))
        halo.putalpha(ImageChops.lighter(core, wide))
        img = Image.alpha_composite(img, halo)
    if st.shadow[3] > 0:
        off = max(1, int(px * 0.06))
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).multiline_text((origin[0] + off, origin[1] + off), text, font=font,
                                          fill=st.shadow, stroke_width=stroke_w,
                                          stroke_fill=st.shadow, align="center", spacing=spacing)
        img = Image.alpha_composite(img, sh.filter(ImageFilter.GaussianBlur(px * 0.05)))
    ImageDraw.Draw(img).multiline_text(origin, text, font=font, fill=st.fill, stroke_width=stroke_w,
                                       stroke_fill=st.outline, align="center", spacing=spacing)
    a = np.asarray(img, np.float32) / 255.0
    rgb = a[..., :3][..., ::-1] * a[..., 3:4]           # BGR, premultiplied
    return np.concatenate([rgb, a[..., 3:4]], axis=2)


# -------------------------------------------------------------- animation
def _ease_out_back(u):
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (u - 1) ** 3 + c1 * (u - 1) ** 2


def _ease_out_cubic(u):
    return 1 - (1 - u) ** 3


def animation_state(item: TextItem, t: float):
    """(visible, alpha, scale, dy_fraction, chars) of a caption at music time t."""
    if not (item.start <= t < item.end) or not item.text.strip():
        return False, 0.0, 1.0, 0.0, 0
    dur = item.end - item.start
    t_in = min(0.3, dur / 3)
    t_out = min(0.2, dur / 4)
    u_in = np.clip((t - item.start) / max(t_in, 1e-3), 0, 1)
    u_out = np.clip((item.end - t) / max(t_out, 1e-3), 0, 1)
    alpha, scale, dy, chars = 1.0, 1.0, 0.0, len(item.text)
    anim = item.animation
    if anim == "pop":
        alpha = min(1.0, u_in * 2.5) * u_out
        scale = (0.55 + 0.45 * _ease_out_back(u_in)) * (0.85 + 0.15 * u_out)
    elif anim == "slide":
        alpha = u_in * u_out
        dy = (1 - _ease_out_cubic(u_in)) * 0.5 + (1 - u_out) * -0.3
    elif anim == "fade":
        alpha = u_in * u_out
    elif anim == "typewriter":
        reveal = min(0.9, dur * 0.4, 0.05 * len(item.text))
        chars = int(np.ceil(len(item.text) * np.clip((t - item.start) / max(reveal, 1e-3), 0, 1)))
        alpha = u_out
    return chars > 0 and alpha > 1e-3, float(alpha), float(scale), float(dy), chars


def draw_caption(frame: np.ndarray, item: TextItem, t: float, top_bar: float = 0.0,
                 bottom_bar: float = 0.0) -> np.ndarray:
    """Draw one caption onto a BGR uint8 frame (in place) at music time t."""
    visible, alpha, scale, dy, chars = animation_state(item, t)
    if not visible:
        return frame
    H, W = frame.shape[:2]
    px = max(8, int(round(0.075 * H * item.size)))
    sprite = render_sprite(item.text[:chars] if chars < len(item.text) else item.text,
                           item.style, px, int(W * 0.9))
    # the text box (without the glow/outline margin) sits `margin` from the edge
    text_h = max(1, sprite.shape[0] - 2 * sprite_pad(item.style, px))
    if abs(scale - 1.0) > 1e-3:
        sh, sw = sprite.shape[:2]
        sprite = cv2.resize(sprite, (max(1, int(sw * scale)), max(1, int(sh * scale))),
                            interpolation=cv2.INTER_LINEAR)
    sh, sw = sprite.shape[:2]
    margin = int(0.06 * H)
    if item.position == "top":
        cy = margin + text_h / 2
    elif item.position == "center":
        cy = H / 2
    else:
        cy = H - margin - text_h / 2
    cy += dy * px
    x0 = int(round(W / 2 - sw / 2))
    y0 = int(round(cy - sh / 2))
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(W, x0 + sw), min(H, y0 + sh)
    if fx1 <= fx0 or fy1 <= fy0:
        return frame
    spr = sprite[fy0 - y0: fy1 - y0, fx0 - x0: fx1 - x0] * alpha
    roi = frame[fy0:fy1, fx0:fx1].astype(np.float32)
    roi = roi * (1 - spr[..., 3:4]) + spr[..., :3] * 255.0
    frame[fy0:fy1, fx0:fx1] = np.clip(roi, 0, 255).astype(np.uint8)
    return frame
