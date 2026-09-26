"""Editable marker data: beats, hits and combo grouping.

Combos are represented by a ``combo_start`` flag on each hit (hits are kept
sorted by time). A combo is a run of hits from one flagged hit up to the next.
That makes split / merge / delete trivially local edits and keeps manual
grouping stable when the list is recalculated.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Hit:
    t: float                    # source (video) time in seconds
    strength: float = 1.0
    combo_start: bool = False
    manual: bool = False
    fx: Optional[bool] = None   # hit effects on this hit: None = the default

    def to_dict(self):
        d = {"t": self.t, "strength": self.strength,
             "combo_start": self.combo_start, "manual": self.manual}
        if self.fx is not None:
            d["fx"] = self.fx
        return d

    @classmethod
    def from_dict(cls, d):
        fx = d.get("fx")
        return cls(float(d["t"]), float(d.get("strength", 1.0)),
                   bool(d.get("combo_start", False)), bool(d.get("manual", False)),
                   None if fx is None else bool(fx))


@dataclass
class Markers:
    """The user-editable state the sync engine consumes."""

    beats: List[float] = field(default_factory=list)   # music seconds
    hits: List[Hit] = field(default_factory=list)      # source seconds

    # ------------------------------------------------------------------ beats
    def set_beats(self, beats):
        self.beats = sorted(float(b) for b in beats)

    def add_beat(self, t: float) -> int:
        i = bisect.bisect_left(self.beats, t)
        self.beats.insert(i, float(t))
        return i

    def delete_beat(self, index: int):
        if 0 <= index < len(self.beats):
            del self.beats[index]

    # ------------------------------------------------------------------- hits
    def set_hits(self, hits: List[Hit], combo_gap: float, regularity: float | None = None):
        """Replace the hits (e.g. re-detection), keeping per-hit effect
        choices of hits that are still there (matched within 60 ms)."""
        old = [(h.t, h.fx) for h in self.hits if h.fx is not None]
        self.hits = sorted(hits, key=lambda h: h.t)
        for t, fx in old:
            near = min(self.hits, key=lambda h: abs(h.t - t), default=None)
            if near is not None and abs(near.t - t) <= 0.06 and near.fx is None:
                near.fx = fx
        self.auto_group(combo_gap, regularity)

    def auto_group(self, combo_gap: float, regularity: float | None = None):
        """Recompute combo boundaries.

        A gap longer than ``combo_gap`` always splits. With ``regularity``
        (a fraction, e.g. 0.3) a combo must also keep a steady rhythm: a gap
        that differs from the combo's running hit period by more than that
        fraction starts a new combo. The period prior is the typical gap over
        the whole video, so a combo can't start with an odd double-hit.
        """
        gaps = [b.t - a.t for a, b in zip(self.hits, self.hits[1:])]
        short = sorted(g for g in gaps if g <= combo_gap)
        prior = short[len(short) // 2] if short else None
        prev, run = None, []
        for h in self.hits:
            start = prev is None or (h.t - prev) > combo_gap
            if not start and regularity is not None and prior:
                g = h.t - prev
                recent = sorted(run[-6:])
                period = recent[len(recent) // 2] if len(recent) >= 3 else prior
                start = abs(g - period) > regularity * period
            h.combo_start = start
            run = [] if start else run + [h.t - prev]
            prev = h.t

    def typical_hit_period(self, combos: List[List[int]] | None = None) -> float | None:
        """Median gap between consecutive hits inside the given combos."""
        groups = self.combos() if combos is None else combos
        gaps = sorted(self.hits[g[k + 1]].t - self.hits[g[k]].t
                      for g in groups for k in range(len(g) - 1))
        return gaps[len(gaps) // 2] if gaps else None

    def add_hit(self, t: float, combo_gap: float, strength: float = 1.0) -> int:
        """Insert a manual hit; it joins the neighbouring combo when close."""
        times = [h.t for h in self.hits]
        i = bisect.bisect_left(times, t)
        start = i == 0 or (t - self.hits[i - 1].t) > combo_gap
        self.hits.insert(i, Hit(float(t), strength, start, manual=True))
        return i

    def delete_hit(self, index: int):
        if not 0 <= index < len(self.hits):
            return
        was_start = self.hits[index].combo_start
        del self.hits[index]
        # Preserve the boundary: the next hit of the same combo inherits it.
        if was_start and index < len(self.hits) and not self.hits[index].combo_start:
            self.hits[index].combo_start = True
        if self.hits:
            self.hits[0].combo_start = True

    def split_combo_at(self, index: int):
        """Make ``hits[index]`` the first hit of a new combo."""
        if 0 <= index < len(self.hits):
            self.hits[index].combo_start = True

    def merge_with_previous(self, index: int):
        """Merge the combo containing ``hits[index]`` into the previous combo."""
        start = self.combo_start_index(index)
        if start > 0:
            self.hits[start].combo_start = False

    def combo_start_index(self, index: int) -> int:
        i = index
        while i > 0 and not self.hits[i].combo_start:
            i -= 1
        return i

    def combo_ids(self) -> List[int]:
        ids, cid = [], -1
        for i, h in enumerate(self.hits):
            if h.combo_start or i == 0:
                cid += 1
            ids.append(cid)
        return ids

    def combos(self) -> List[List[int]]:
        """Hit indices grouped by combo, in time order."""
        groups: List[List[int]] = []
        for i, h in enumerate(self.hits):
            if h.combo_start or not groups:
                groups.append([])
            groups[-1].append(i)
        return groups

    # ---------------------------------------------------------------- persist
    def to_dict(self):
        return {"beats": list(self.beats), "hits": [h.to_dict() for h in self.hits]}

    @classmethod
    def from_dict(cls, d):
        m = cls()
        m.beats = sorted(float(b) for b in d.get("beats", []))
        m.hits = sorted((Hit.from_dict(h) for h in d.get("hits", [])), key=lambda h: h.t)
        return m


@dataclass
class ComboChoice:
    """One entry of the combo plan: which combo, whether it's used, whether
    it gets a slow-mo lead-in and whether it gets focus bars (black bars top
    and bottom). ``key`` is the combo's first hit time, which survives
    re-grouping and re-detection."""

    key: float
    enabled: bool = True
    lead_in: bool = False
    focus: bool = False

    def to_dict(self):
        return {"key": self.key, "enabled": self.enabled, "lead_in": self.lead_in,
                "focus": self.focus}

    @classmethod
    def from_dict(cls, d):
        return cls(float(d["key"]), bool(d.get("enabled", True)), bool(d.get("lead_in", False)),
                   bool(d.get("focus", False)))
