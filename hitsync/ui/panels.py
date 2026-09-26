"""Side panels: Combos (pick / order / lead-in / focus / per-hit effects),
Music (songs: order, cuts, remove), Text (captions) and Effects (look, hit
effects, focus bars, transitions, fades, ranges)."""
from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListView, QListWidget,
                               QListWidgetItem,
                               QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget)

from .. import effects as fx_mod
from ..text_overlay import ANIMATIONS, LABELS as TEXT_LABELS, STYLES as TEXT_STYLES
from . import theme
from .widgets import ElidedLabel, Segmented, ValueSlider, button, label, shrinkable


def _scroll(inner: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidgetResizable(True)
    sa.setWidget(inner)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return sa


def _fit_list(lst: QListWidget):
    """Rows as wide as the panel (never a sideways scrollbar), scrolled
    smoothly up and down."""
    lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    lst.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
    lst.setResizeMode(QListView.Adjust)
    lst.setSpacing(3)


def _row_hint(row: QWidget) -> QSize:
    # width 0: the list stretches every row to its own width
    return QSize(0, row.sizeHint().height())


def _section(lay, text):
    lb = label(text.upper(), h2=True, wrap=True)
    lb.setContentsMargins(0, 10, 0, 2)
    lay.addWidget(lb)


# ================================================================ combos
class ComboRow(QFrame):
    toggled = Signal(float, bool)
    lead_toggled = Signal(float, bool)
    focus_toggled = Signal(float, bool)
    fx_changed = Signal(list, object)          # hit indices, True/False
    selected = Signal(float)
    moved = Signal(float, int)                 # key, -1 up / +1 down

    def __init__(self, n, choice, hits, ranks, default_fx, thumb=None):
        super().__init__()
        self.key = choice.key
        self.setProperty("card", True)
        self.hits = hits                        # [(index, Hit)]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        top = QHBoxLayout()
        handle = QLabel("⠿")
        handle.setToolTip("Drag to change the order")
        handle.setStyleSheet(f"color: {theme.MUTED}; font-size: 16px;")
        top.addWidget(handle)
        self.use = QCheckBox()
        self.use.setChecked(choice.enabled)
        self.use.setToolTip("Use this combo in the montage")
        self.use.toggled.connect(lambda on: self.toggled.emit(self.key, on))
        top.addWidget(self.use)
        self.thumb = QLabel()
        self.thumb.setFixedSize(80, 45)
        self.thumb.setStyleSheet(f"background: {theme.PANEL}; border-radius: 4px;")
        if thumb is not None:
            self.set_thumb(thumb)
        top.addWidget(self.thumb)
        info = QVBoxLayout()
        info.setSpacing(0)
        stars = "★" * max(1, int(round(ranks.get(choice.key, 0.0) * 3))) if ranks else ""
        t = label(f"#{n}  ×{len(hits)} hits  <span style='color:{theme.WARN}'>{stars}</span>")
        t.setTextFormat(Qt.RichText)
        t.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        info.addWidget(t)
        info.addWidget(ElidedLabel(f"at {theme.fmt_short(choice.key)} in the video", muted=True))
        top.addLayout(info, 1)
        ud = QVBoxLayout()
        ud.setSpacing(0)
        for txt, d in (("▲", -1), ("▼", 1)):
            b = button(txt, "Move up" if d < 0 else "Move down", flat=True,
                       slot=lambda _=False, d=d: self.moved.emit(self.key, d))
            b.setFixedSize(22, 18)
            ud.addWidget(b)
        top.addLayout(ud)
        lay.addLayout(top)
        row = QGridLayout()                      # stacked, so it fits a narrow panel
        row.setContentsMargins(0, 0, 0, 0)
        row.setVerticalSpacing(2)
        self.lead = QCheckBox("Slow-mo lead-in")
        self.lead.setToolTip("Show the footage before this combo in slow motion; the combo "
                             "still starts on the beat")
        self.lead.setChecked(choice.lead_in)
        self.lead.toggled.connect(lambda on: self.lead_toggled.emit(self.key, on))
        row.addWidget(self.lead, 0, 0)
        self.focus = QCheckBox("🎬 Focus bars")
        self.focus.setToolTip("Black bars at the top and bottom while this combo plays "
                              "(a cinematic focus on it)")
        self.focus.setChecked(choice.focus)
        self.focus.toggled.connect(lambda on: self.focus_toggled.emit(self.key, on))
        row.addWidget(self.focus, 1, 0)
        self.fx_btn = button("✦ Effects", "Choose which hits get zoom / shake / flash",
                             flat=True, checkable=True)
        row.addWidget(self.fx_btn, 0, 1, Qt.AlignRight)
        row.setColumnStretch(0, 1)
        lay.addLayout(row)
        self.fx_box = QWidget()
        g = QGridLayout(self.fx_box)
        g.setContentsMargins(0, 0, 0, 0)
        g.setSpacing(3)
        self.chips = []
        for k, (i, h) in enumerate(hits):
            b = QPushButton(str(k + 1))
            b.setProperty("chip", True)
            b.setCheckable(True)
            b.setChecked(default_fx if h.fx is None else h.fx)
            b.setFixedSize(34, 24)
            b.setStyleSheet("padding: 0px;")
            b.toggled.connect(lambda on, i=i: self.fx_changed.emit([i], on))
            g.addWidget(b, k // 6, k % 6)
            self.chips.append((i, b))
        quick = QGridLayout()
        for j, (text, pick) in enumerate((("All", lambda k, n: True),
                                          ("None", lambda k, n: False),
                                          ("Every 2nd", lambda k, n: k % 2 == 1),
                                          ("First && last", lambda k, n: k in (0, n - 1)))):
            quick.addWidget(button(text, flat=True, slot=lambda _=False, f=pick: self._quick(f)),
                            j // 2, j % 2)
        g.addLayout(quick, (len(hits) + 5) // 6, 0, 1, 6)
        self.fx_box.setVisible(False)
        self.fx_btn.toggled.connect(self._expand)
        lay.addWidget(self.fx_box)

    def _expand(self, on):
        self.fx_box.setVisible(on)
        self.adjustSize()
        self.parent_item_resize()

    def parent_item_resize(self):
        item = getattr(self, "item", None)
        if item is not None:
            item.setSizeHint(_row_hint(self))

    def _quick(self, pick):
        n = len(self.chips)
        on_list, off_list = [], []
        for k, (i, b) in enumerate(self.chips):
            want = pick(k, n)
            b.blockSignals(True)
            b.setChecked(want)
            b.blockSignals(False)
            (on_list if want else off_list).append(i)
        if on_list:
            self.fx_changed.emit(on_list, True)
        if off_list:
            self.fx_changed.emit(off_list, False)

    def set_thumb(self, pix: QPixmap):
        self.thumb.setPixmap(pix.scaled(self.thumb.size(), Qt.KeepAspectRatioByExpanding,
                                        Qt.SmoothTransformation))

    def mousePressEvent(self, e):
        self.selected.emit(self.key)
        super().mousePressEvent(e)


class CombosPanel(QWidget):
    plan_changed = Signal(list)        # [ComboChoice] in order
    fx_changed = Signal(list, object)
    auto_pick = Signal()
    reset = Signal()
    select = Signal(float)             # combo key (video time)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        top = QHBoxLayout()
        top.addWidget(button("★ Auto-pick best", "Use the best combos that fit the song",
                             slot=lambda: self.auto_pick.emit()))
        top.addWidget(button("Reset", "Every combo, in the order you played them",
                             flat=True, slot=lambda: self.reset.emit()))
        top.addStretch(1)
        lay.addLayout(top)
        self.hint = label("Drag to reorder · tick to use · 🎬 black bars on a combo · "
                          "✦ choose hits for effects", muted=True, wrap=True)
        lay.addWidget(self.hint)
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        _fit_list(self.list)
        self.list.model().rowsMoved.connect(self._rows_moved)
        lay.addWidget(self.list, 1)
        self.empty = label("Combos show up here after the video is analysed.", muted=True,
                           wrap=True)
        lay.addWidget(self.empty)
        self.choices = []
        self.thumbs = {}

    def set_entries(self, entries, markers, ranks, default_fx):
        self.choices = [c for c, _ in entries]
        self.list.clear()
        self.empty.setVisible(not entries)
        for n, (c, g) in enumerate(entries, start=1):
            hits = [(i, markers.hits[i]) for i in g]
            row = ComboRow(n, c, hits, ranks, default_fx, self.thumbs.get(round(c.key, 3)))
            item = QListWidgetItem()
            item.setData(Qt.UserRole, c.key)
            item.setSizeHint(_row_hint(row))
            row.item = item
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            row.toggled.connect(self._toggled)
            row.lead_toggled.connect(self._lead)
            row.focus_toggled.connect(self._focus)
            row.fx_changed.connect(self.fx_changed.emit)
            row.selected.connect(self.select.emit)
            row.moved.connect(self._move)

    def set_thumb(self, key, pix):
        self.thumbs[round(key, 3)] = pix
        for k in range(self.list.count()):
            item = self.list.item(k)
            if abs(item.data(Qt.UserRole) - key) < 1e-3:
                w = self.list.itemWidget(item)
                if w is not None:
                    w.set_thumb(pix)

    def _by_key(self, key):
        return next((c for c in self.choices if abs(c.key - key) < 1e-6), None)

    def _toggled(self, key, on):
        c = self._by_key(key)
        if c:
            c.enabled = on
            self.plan_changed.emit(list(self.choices))

    def _lead(self, key, on):
        c = self._by_key(key)
        if c:
            c.lead_in = on
            self.plan_changed.emit(list(self.choices))

    def _focus(self, key, on):
        c = self._by_key(key)
        if c:
            c.focus = on
            self.plan_changed.emit(list(self.choices))

    def _move(self, key, d):
        i = next((k for k, c in enumerate(self.choices) if abs(c.key - key) < 1e-6), None)
        if i is None or not 0 <= i + d < len(self.choices):
            return
        self.choices[i], self.choices[i + d] = self.choices[i + d], self.choices[i]
        self.plan_changed.emit(list(self.choices))

    def _rows_moved(self, *_):
        order = [self.list.item(k).data(Qt.UserRole) for k in range(self.list.count())]
        by = {round(c.key, 6): c for c in self.choices}
        self.choices = [by[round(k, 6)] for k in order if round(k, 6) in by]
        self.plan_changed.emit(list(self.choices))


# ================================================================== music
class SongRow(QFrame):
    """One song of the soundtrack: name, tempo, where it starts and where it
    switches to the next song, with move / delete / cut-reset."""

    selected = Signal(int)
    moved = Signal(int, int)                   # song index, -1 up / +1 down
    removed = Signal(int)
    reset_cut = Signal(int)

    def __init__(self, k, info: dict, can_remove: bool, current: bool):
        super().__init__()
        self.k = k
        self.setProperty("card", True)
        if current:
            self.setStyleSheet(f"QFrame[card=\"true\"] {{ border: 1px solid {theme.ACCENT}; }}")
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to listen to it in the Song view")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(2)
        top = QHBoxLayout()
        handle = QLabel("⠿")
        handle.setToolTip("Drag to change the order")
        handle.setStyleSheet(f"color: {theme.MUTED}; font-size: 16px;")
        top.addWidget(handle)
        name = label(f"<b>{k + 1}.</b> {info['name']}")
        name.setTextFormat(Qt.RichText)
        name.setMinimumWidth(40)
        name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        top.addWidget(name, 1)
        for txt, d in (("▲", -1), ("▼", 1)):
            b = button(txt, "Play earlier" if d < 0 else "Play later", flat=True,
                       slot=lambda _=False, d=d: self.moved.emit(self.k, d))
            b.setFixedSize(24, 22)
            top.addWidget(b)
        rm = button("✕", "Remove this song" if can_remove else
                    "The only song: drop another song to replace it", flat=True,
                    slot=lambda: self.removed.emit(self.k))
        rm.setFixedSize(24, 22)
        rm.setEnabled(can_remove)
        top.addWidget(rm)
        lay.addLayout(top)
        lay.addWidget(ElidedLabel(info["detail"], muted=True))
        row = QHBoxLayout()
        self.cut_lbl = ElidedLabel(info["span"], muted=not info.get("cut_manual"))
        self.cut_lbl.setToolTip(info["span"])
        row.addWidget(self.cut_lbl, 1)
        if info.get("cut_manual"):
            row.addWidget(button("✂ Auto", "Back to the automatic end", flat=True,
                                 slot=lambda: self.reset_cut.emit(self.k)))
        lay.addLayout(row)

    def mousePressEvent(self, e):
        self.selected.emit(self.k)
        super().mousePressEvent(e)


class MusicPanel(QWidget):
    """The soundtrack: songs in play order. Each one plays from its start
    point (song 1: where the first hit lands) until its cut, where the next
    song takes over on the beat."""

    add = Signal()
    select = Signal(int)
    move = Signal(int, int)
    order = Signal(list)               # new order of song indices (drag)
    remove = Signal(int)
    reset_cut = Signal(int)

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.addWidget(button("＋ Add song", "Another song, played after the others",
                             primary=True, slot=lambda: self.add.emit()))
        lay.addWidget(label("Songs play top to bottom, each switching to the next one on the "
                            "beat. Drag or ▲▼ to reorder. To cut a song: click it, play it, "
                            "and press ✂ where the next song should take over.", muted=True,
                            wrap=True))
        self.list = QListWidget()
        self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.list.setDefaultDropAction(Qt.MoveAction)
        _fit_list(self.list)
        self.list.model().rowsMoved.connect(self._rows_moved)
        lay.addWidget(self.list, 1)
        self.empty = label("Drop a song anywhere on the window.", muted=True, wrap=True)
        lay.addWidget(self.empty)

    def set_songs(self, infos, current: int):
        self.list.clear()
        self.empty.setVisible(not infos)
        for k, info in enumerate(infos):
            row = SongRow(k, info, len(infos) > 1, k == current)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, k)
            item.setSizeHint(_row_hint(row))
            self.list.addItem(item)
            self.list.setItemWidget(item, row)
            row.selected.connect(self.select.emit)
            row.moved.connect(self.move.emit)
            row.removed.connect(self.remove.emit)
            row.reset_cut.connect(self.reset_cut.emit)

    def _rows_moved(self, *_):
        order = [int(self.list.item(k).data(Qt.UserRole)) for k in range(self.list.count())]
        if order != sorted(order):
            self.order.emit(order)


# ================================================================== text
class TextRow(QFrame):
    changed = Signal()
    removed = Signal(object)
    seek = Signal(float)
    set_time = Signal(object, str)          # item, "start" | "end"

    def __init__(self, item, origin=lambda: 0.0):
        super().__init__()
        self.item = item
        self.origin = origin                    # montage start (times shown from it)
        self.setProperty("card", True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.edit = QLineEdit(item.text)
        self.edit.setPlaceholderText("Your text…")
        self.edit.textChanged.connect(self._text)
        top.addWidget(self.edit, 1)
        top.addWidget(button("✕", "Remove", flat=True, slot=lambda: self.removed.emit(item)))
        lay.addLayout(top)
        self.pos = Segmented([("top", "Top"), ("center", "Center"), ("bottom", "Bottom")])
        self.pos.set(item.position)
        self.pos.changed.connect(lambda v: self._set("position", v))
        lay.addWidget(self.pos, 0, Qt.AlignLeft)
        row = QHBoxLayout()                      # style + animation, shrink to fit
        self.style = shrinkable(QComboBox())
        for s in TEXT_STYLES:
            self.style.addItem(TEXT_LABELS.get(s, s.capitalize()), s)
        self.style.setCurrentIndex(max(0, self.style.findData(item.style)))
        self.style.currentIndexChanged.connect(lambda _: self._set("style", self.style.currentData()))
        row.addWidget(self.style, 3)
        self.anim = shrinkable(QComboBox(), 6)
        for a in ANIMATIONS:
            self.anim.addItem(a.capitalize(), a)
        self.anim.setCurrentIndex(max(0, self.anim.findData(item.animation)))
        self.anim.currentIndexChanged.connect(lambda _: self._set("animation", self.anim.currentData()))
        row.addWidget(self.anim, 2)
        lay.addLayout(row)
        tr = QHBoxLayout()
        tr.addWidget(button("⇤ Start", "Start at the playhead", flat=True,
                            slot=lambda: self.set_time.emit(item, "start")))
        self.times = ElidedLabel("", muted=True)
        self.times.setAlignment(Qt.AlignCenter)
        self.times.setCursor(Qt.PointingHandCursor)
        tr.addWidget(self.times, 1)
        tr.addWidget(button("End ⇥", "End at the playhead", flat=True,
                            slot=lambda: self.set_time.emit(item, "end")))
        lay.addLayout(tr)
        self.size = ValueSlider("Size", 0.5, 2.0, item.size, "{:.2f}", label_width=40)
        self.size.changed.connect(lambda v: self._set("size", v))
        lay.addWidget(self.size)
        self.refresh()

    def refresh(self):
        o = self.origin()
        self.times.setText(f"{theme.fmt_time(self.item.start - o)} → "
                           f"{theme.fmt_time(self.item.end - o)}")

    def _text(self, t):
        self.item.text = t
        self.changed.emit()

    def _set(self, attr, v):
        setattr(self.item, attr, v)
        self.changed.emit()

    def mousePressEvent(self, e):
        self.seek.emit(self.item.start + 0.3)
        super().mousePressEvent(e)


class TextPanel(QWidget):
    add = Signal()
    changed = Signal()
    seek = Signal(float)
    set_time = Signal(object, str)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 6, 0, 0)
        outer.addWidget(button("＋ Add text at playhead", "A caption for 2 bars, at the bottom",
                               primary=True, slot=lambda: self.add.emit()))
        outer.addWidget(label("Bold outlined captions, like a YouTube video. Drag the yellow "
                              "bars on the timeline to change when they show.", muted=True,
                              wrap=True))
        inner = QWidget()
        self.lay = QVBoxLayout(inner)
        self.lay.setContentsMargins(0, 0, 4, 0)
        self.lay.addStretch(1)
        outer.addWidget(_scroll(inner), 1)
        self.rows = []
        self.origin = 0.0                        # montage start, in music time

    def set_items(self, items):
        for r in self.rows:
            r.setParent(None)
            r.deleteLater()
        self.rows = []
        for it in items:
            r = TextRow(it, lambda: self.origin)
            r.changed.connect(self.changed.emit)
            r.removed.connect(self._remove)
            r.seek.connect(self.seek.emit)
            r.set_time.connect(self.set_time.emit)
            self.lay.insertWidget(self.lay.count() - 1, r)
            self.rows.append(r)
        self._items = items

    def refresh_times(self):
        for r in self.rows:
            r.refresh()

    def _remove(self, item):
        if item in self._items:
            self._items.remove(item)
        self.set_items(self._items)
        self.changed.emit()


# =============================================================== effects
class EffectsPanel(QWidget):
    changed = Signal(str)                    # attribute name ("" = overlays)
    add_range = Signal(str, str)             # kind, where ("4 bars" | "combo" | "end")
    seek = Signal(float)

    def __init__(self, project_getter):
        super().__init__()
        self.P = project_getter
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 6, 0)
        self.controls = []

        _section(lay, "Flashy effects on part of the montage")
        add = QHBoxLayout()
        self.kind = shrinkable(QComboBox())
        for k in fx_mod.RANGE_KINDS:
            self.kind.addItem(fx_mod.RANGE_LABELS[k], k)
        add.addWidget(self.kind, 1)
        self.where = shrinkable(QComboBox())
        for text, v in (("4 bars from playhead", "4"), ("8 bars from playhead", "8"),
                        ("The combo at the playhead", "combo"), ("Playhead to the end", "end")):
            self.where.addItem(text, v)
        add.addWidget(self.where, 1)
        lay.addLayout(add)
        lay.addWidget(button("＋ Add effect", primary=True,
                             slot=lambda: self.add_range.emit(self.kind.currentData(),
                                                              self.where.currentData())))
        self.ranges_box = QVBoxLayout()
        lay.addLayout(self.ranges_box)

        _section(lay, "Hit effects (on the hits you tick in Combos ✦)")
        self._slider(lay, "Zoom punch", "render", "hit_zoom", 0, 0.15, "{:.2f}")
        self._slider(lay, "Screen shake", "render", "hit_shake", 0, 1, percent=True)
        self._slider(lay, "Flash", "render", "hit_flash", 0, 1, percent=True)
        self._slider(lay, "RGB split", "render", "hit_rgb", 0, 1, percent=True)
        self._check(lay, "Effects on every hit", "render", "hit_fx_default",
                    "Every hit gets the effects unless you untick it in Combos ✦")
        self._slider(lay, "Velocity", "sync", "velocity", 0, 0.9, percent=True,
                     tip="Slow on each impact, fast in between (hits stay on the beat)")

        _section(lay, "Look")
        self._choice(lay, "Filter", "render", "filter",
                     [(f, f.capitalize() if f != "bw" else "Black & white") for f in fx_mod.FILTERS])
        self._slider(lay, "Filter strength", "render", "filter_strength", 0, 1, percent=True)
        self._slider(lay, "Motion blur", "render", "motion_blur", 0, 1, percent=True)
        self._slider(lay, "Vignette", "render", "vignette", 0, 1, percent=True)
        self._slider(lay, "Bar pulse", "render", "beat_pulse", 0, 1, percent=True)

        _section(lay, "Focus bars (black bars top and bottom)")
        self._choice(lay, "Automatic", "sync", "letterbox_mode",
                     [("slowmo", "Slow-mo parts"), ("first combo", "Intro + 1st combo"),
                      ("combos", "Every combo"), ("off", "Off")])
        lay.addWidget(label("Or tick 🎬 Focus bars on any combo in the Combos tab.",
                            muted=True, wrap=True))

        _section(lay, "Between combos")
        self._choice(lay, "Transition", "sync", "transition",
                     [(t, t.capitalize()) for t in fx_mod.TRANSITIONS])

        _section(lay, "Start and end")
        self._choice(lay, "Start", "render", "fade_in", [("none", "No fade"),
                                                          ("black", "Fade from black"),
                                                          ("white", "Flash in")])
        self._slider(lay, "Start fade (s)", "render", "fade_in_len", 0.1, 3.0, "{:.1f}")
        self._choice(lay, "End", "render", "fade_out", [("none", "Cut"),
                                                         ("black", "Fade to black"),
                                                         ("white", "Fade to white")])
        self._slider(lay, "End fade (s)", "render", "fade_out_len", 0.2, 5.0, "{:.1f}")
        lay.addStretch(1)
        outer.addWidget(_scroll(inner), 1)

    # ---- bound controls
    def _obj(self, which):
        p = self.P()
        return p.render_params if which == "render" else p.sync

    def _slider(self, lay, text, which, attr, lo, hi, fmt="{:.2f}", percent=False, tip=""):
        w = ValueSlider(text, lo, hi, 0, fmt, percent=percent, tip=tip, label_width=112)
        w.changed.connect(lambda v: self._push(which, attr, float(v)))
        lay.addWidget(w)
        self.controls.append((w, which, attr, "slider"))

    def _check(self, lay, text, which, attr, tip=""):
        w = QCheckBox(text)
        w.setToolTip(tip)
        w.toggled.connect(lambda on: self._push(which, attr, bool(on)))
        lay.addWidget(w)
        self.controls.append((w, which, attr, "check"))

    def _choice(self, lay, text, which, attr, options):
        row = QHBoxLayout()
        lb = QLabel(text)
        lb.setFixedWidth(112)
        row.addWidget(lb)
        w = shrinkable(QComboBox())
        for v, t in options:
            w.addItem(t, v)
        w.currentIndexChanged.connect(lambda _: self._push(which, attr, w.currentData()))
        row.addWidget(w, 1)
        lay.addLayout(row)
        self.controls.append((w, which, attr, "combo"))

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
                    w.setCurrentIndex(max(0, w.findData(v)))
                w.blockSignals(False)
        finally:
            self._pulling = False
        self.refresh_ranges()

    def refresh_ranges(self):
        while self.ranges_box.count():
            wdg = self.ranges_box.takeAt(0).widget()
            if wdg is not None:
                wdg.setParent(None)                  # gone now, not when Qt gets to it
                wdg.deleteLater()
        p = self.P()
        o = p.schedule.start if p.schedule is not None else 0.0     # montage clock
        for rg in list(p.effects):
            card = QFrame()
            card.setProperty("card", True)
            v = QVBoxLayout(card)
            v.setContentsMargins(8, 4, 8, 4)
            head = QHBoxLayout()
            name = label(f"<b>{fx_mod.RANGE_LABELS.get(rg.kind, rg.kind)}</b>  "
                         f"<span style='color:{theme.MUTED}'>{theme.fmt_time(rg.start - o)} → "
                         f"{theme.fmt_time(rg.end - o)}</span>")
            name.setTextFormat(Qt.RichText)
            name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            head.addWidget(name, 1)
            head.addWidget(button("▶", "Preview it", flat=True,
                                  slot=lambda _=False, r=rg: self.seek.emit(r.start)))
            head.addWidget(button("✕", "Remove", flat=True,
                                  slot=lambda _=False, r=rg: self._remove(r)))
            v.addLayout(head)
            s = ValueSlider("Strength", 0, 1.5, rg.strength, percent=True, label_width=60)
            s.changed.connect(lambda val, r=rg: (setattr(r, "strength", val), self.changed.emit("")))
            v.addWidget(s)
            self.ranges_box.addWidget(card)

    def _remove(self, rg):
        p = self.P()
        if rg in p.effects:
            p.effects.remove(rg)
        self.refresh_ranges()
        self.changed.emit("")


__all__ = ["CombosPanel", "MusicPanel", "TextPanel", "EffectsPanel", "QSize"]
