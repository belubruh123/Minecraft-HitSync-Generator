"""Edit-decision engine: markers + params -> time-remap schedule.

The output timeline is the music timeline (output t == music t). The schedule
is a list of contiguous ``Segment``s, each mapping an output interval onto a
monotonically increasing source interval through piecewise-linear knots
(which is how speed ramps are expressed). A ``cut_before`` flag marks a
discontinuity in source time (a hard cut).

Everything here is pure numpy on small arrays, so "Recalculate Alignment"
is instantaneous and never touches the media files.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import beatgrid
from .config import SyncParams
from .models import Markers

EPS = 1e-6


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


@dataclass
class Segment:
    out_knots: np.ndarray
    src_knots: np.ndarray
    kind: str                      # intro | lead | bridge | combo | trim | tail | outro
    cut_before: bool = False

    @property
    def out_start(self):
        return float(self.out_knots[0])

    @property
    def out_end(self):
        return float(self.out_knots[-1])

    @property
    def src_start(self):
        return float(self.src_knots[0])

    @property
    def src_end(self):
        return float(self.src_knots[-1])

    @property
    def speed(self) -> float:
        d = self.out_end - self.out_start
        return (self.src_end - self.src_start) / d if d > EPS else 1.0

    def src_at(self, t: float) -> float:
        return float(np.interp(t, self.out_knots, self.src_knots))

    def speed_at(self, t: float) -> float:
        i = int(np.clip(np.searchsorted(self.out_knots, t, side="right") - 1,
                        0, len(self.out_knots) - 2))
        do = self.out_knots[i + 1] - self.out_knots[i]
        return float((self.src_knots[i + 1] - self.src_knots[i]) / do) if do > EPS else 1.0


@dataclass
class HitPlacement:
    hit_index: int
    src_t: float
    out_t: Optional[float]         # None when the hit was cut / outside the edit
    target: Optional[float]        # grid point it was locked to (if any)
    locked: bool
    error_ms: float                # natural arrival minus target (ms)
    combo: int
    status: str = ""               # locked | ramp | trim | unsynced | ignored


@dataclass
class Schedule:
    segments: List[Segment] = field(default_factory=list)
    placements: List[HitPlacement] = field(default_factory=list)
    letterbox: List[tuple] = field(default_factory=list)   # (start, end) out times
    cuts: List[float] = field(default_factory=list)        # hard-cut out times
    # The schedule lives on the music timeline; the montage covers music
    # time [start, start + duration] (the song before `start` is cut).
    start: float = 0.0
    duration: float = 0.0
    intro_speed: float = 1.0
    ref_bpm: float = 120.0
    beats_per_hit: float = 1.0
    hit_period: float = 0.0          # measured average spacing inside combos
    combos_kept: int = 0
    combos_dropped: int = 0
    params: SyncParams = field(default_factory=SyncParams)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._starts = None

    @property
    def end(self) -> float:
        return self.start + self.duration

    # ---------------------------------------------------------- evaluation
    def _index(self, t: float) -> int:
        if self._starts is None or len(self._starts) != len(self.segments):
            self._starts = [s.out_start for s in self.segments]
        i = bisect.bisect_right(self._starts, t) - 1
        return int(np.clip(i, 0, len(self.segments) - 1))

    def src_time(self, t: float) -> float:
        if not self.segments:
            return t
        return self.segments[self._index(t)].src_at(t)

    def speed_at(self, t: float) -> float:
        if not self.segments:
            return 1.0
        return self.segments[self._index(t)].speed_at(t)

    def letterbox_amount(self, t: float) -> float:
        p = self.params
        if not p.letterbox_enabled:
            return 0.0
        fade = max(1e-3, p.letterbox_fade)
        amt = 0.0
        for start, end in self.letterbox:
            if t < start or t > end + fade:
                continue
            a = smoothstep((t - start) / fade) if t < end else smoothstep((end + fade - t) / fade)
            if t >= end:
                # never drop below where the ease-in had reached
                a = min(a, smoothstep((end - start) / fade))
            amt = max(amt, float(a))
        return amt

    def flash_amount(self, t: float) -> float:
        p = self.params
        if p.transition != "flash" or not self.cuts:
            return 0.0
        i = bisect.bisect_right(self.cuts, t) - 1
        if i < 0:
            return 0.0
        u = (t - self.cuts[i]) / max(1e-3, p.flash_duration)
        return float((1 - u) ** 2) if 0 <= u < 1 else 0.0

    def removed_source_ranges(self) -> List[tuple]:
        """Source intervals skipped by cuts (for timeline display)."""
        out = []
        for a, b in zip(self.segments, self.segments[1:]):
            if b.src_start > a.src_end + 1e-4:
                out.append((a.src_end, b.src_start))
        return out

    def summary(self) -> str:
        locked = sum(1 for p in self.placements if p.locked)
        placed = sum(1 for p in self.placements if p.out_t is not None)
        combos = len({p.combo for p in self.placements if p.out_t is not None})
        errs = [abs(p.error_ms) for p in self.placements if p.locked]
        ramped = sum(1 for p in self.placements if p.status == "speed")
        bph = self.beats_per_hit
        spacing = ("1 hit per beat" if bph == 1 else
                   f"1 hit per {bph:g} beats" if bph > 1 else f"{1 / bph:g} hits per beat")
        lines = [
            f"Output: {self.duration:.2f}s (song {self.start:.2f}-{self.end:.2f}s)  |  "
            f"cuts {len(self.cuts)}",
            f"Tempo {self.ref_bpm:.1f} BPM  |  intro slow-mo {self.intro_speed:.2f}x",
            f"Combos kept {self.combos_kept}  |  too short/irregular, cut {self.combos_dropped}",
            f"Combo spacing: {spacing} (avg hit gap {self.hit_period * 1000:.0f} ms)",
            f"Hits placed {placed}/{len(self.placements)}  |  on beat {locked}"
            f" ({ramped} via speed ramp)  |  combos {combos}",
        ]
        if errs:
            lines.append(f"Jitter corrected: mean {np.mean(errs):.1f} ms, max {np.max(errs):.1f} ms")
        lines += [f"! {w}" for w in self.warnings]
        return "\n".join(lines)


class SyncEngine:
    def __init__(self, markers: Markers, params: SyncParams, video_duration: float,
                 music_duration: float):
        self.m = markers
        self.p = params
        # Static beat: every grid beat from the drop to the end holds exactly
        # one hit (one hit per beat, never a skipped or off-grid beat).
        self.fill = params.every_beat
        self.video_duration = float(video_duration)
        self.music_duration = float(music_duration)
        beats = np.asarray(markers.beats, float)
        self.beats = beatgrid.extended_beats(beats, 0.0, self.music_duration + 5.0)
        groups = markers.combos()
        self.kept = [g for g in groups if len(g) >= max(1, params.min_combo_len)]
        self.beat_period = float(np.median(np.diff(self.beats))) if len(self.beats) > 1 else 0.5
        self.hit_period = markers.typical_hit_period(self.kept) or \
            markers.typical_hit_period() or self.beat_period
        self.beats_per_hit = self._choose_spacing()
        # Hits inside a combo step along this grid, `combo_step` points apart.
        if self.beats_per_hit < 1:
            k = int(round(1 / self.beats_per_hit))
            self.combo_grid, self.combo_step = beatgrid.subdivide(self.beats, k), 1
        else:
            self.combo_grid, self.combo_step = self.beats, int(round(self.beats_per_hit))
        self.curve = beatgrid.tempo_curve(beats if len(beats) >= 3 else self.beats)
        if params.drop_time > 0 and len(beats):
            # the drop is a beat of the static grid
            params = type(params)(**{**params.to_dict(),
                                     "drop_time": float(beats[np.argmin(np.abs(beats - params.drop_time))])})
            self.p = params
        anchor = params.drop_time if params.intro_enabled and params.drop_time > 0 else 0.0
        self.ref_bpm = float(beatgrid.bpm_at(self.curve, anchor))
        self.tol = params.jitter_tolerance_ms / 1000.0

    def _choose_spacing(self) -> float:
        """Beats per hit: the musical note value closest to the player's rhythm."""
        if self.p.static_grid:
            return 1.0          # one hit per static beat, whatever the notes do
        choice = str(self.p.combo_spacing).strip().lower()
        if choice not in ("", "auto"):
            try:
                return max(0.25, float(choice))
            except ValueError:
                pass
        ratio = self.hit_period / max(1e-6, self.beat_period)
        options = (0.5, 1.0, 2.0, 3.0, 4.0)
        return min(options, key=lambda o: abs(np.log(ratio / o)))

    # ------------------------------------------------------------- helpers
    def base_speed(self, t: float) -> float:
        p = self.p
        if not p.tempo_matching or p.tempo_strength <= 0:
            return 1.0
        ratio = float(beatgrid.bpm_at(self.curve, t)) / max(1e-6, self.ref_bpm)
        return float(np.clip(ratio ** p.tempo_strength, p.min_speed, p.max_speed))

    def _natural(self, out0: float, src0: float, src1: float):
        """Knots for playing src0->src1 at the tempo-matched base speed."""
        outs, srcs = [out0], [src0]
        t, s = out0, src0
        while s < src1 - EPS:
            sp = self.base_speed(t)
            dt = min(0.1, (src1 - s) / sp)
            t += dt
            s = min(src1, s + sp * dt)
            outs.append(t)
            srcs.append(s)
        return np.array(outs), np.array(srcs)

    def _natural_duration(self, out0, src0, src1) -> float:
        outs, _ = self._natural(out0, src0, src1)
        return float(outs[-1] - out0)

    def _push(self, sched: Schedule, outs, srcs, kind, cut_before=False):
        outs = np.asarray(outs, float)
        srcs = np.asarray(srcs, float)
        if len(outs) < 2 or outs[-1] - outs[0] < EPS:
            return
        cut_before = cut_before and bool(sched.segments)  # nothing to cut from at t=0
        sched.segments.append(Segment(outs, srcs, kind, cut_before))
        if cut_before:
            sched.cuts.append(float(outs[0]))

    def _linear(self, sched, out0, src0, out1, src1, kind, cut_before=False):
        self._push(sched, [out0, out1], [src0, src1], kind, cut_before)

    @staticmethod
    def _nearest(grid: np.ndarray, t: float, after: float):
        cand = grid[grid > after + EPS]
        if not len(cand):
            return None
        return float(cand[np.argmin(np.abs(cand - t))])

    def _ramp_ok(self, src_len, out_len, t) -> bool:
        """Is this speed change (relative to the base speed) an acceptable ramp?"""
        if out_len <= EPS:
            return False
        rel = (src_len / out_len) / self.base_speed(t)
        return self.p.ramp_min_speed <= rel <= self.p.ramp_max_speed

    def _speed_ok(self, src_len, out_len) -> bool:
        if out_len <= EPS:
            return False
        s = src_len / out_len
        return self.p.min_speed <= s <= self.p.max_speed

    # ------------------------------------------------------------- intro
    def _intro_end_for(self, first_hit: Optional[float]) -> float:
        """Nudge the intro's source end so the first combat hit lands on a beat.

        After the drop footage plays at 1.0x, so (first_hit - intro_end) must
        be a whole number of beats. Moving the end point by < half a beat is
        invisible inside slow motion, and avoids an off-grid first combo.
        """
        p = self.p
        if first_hit is None:
            return p.intro_end
        if self.fill and (p.intro_length > 0 or first_hit > p.intro_start + 0.2):
            return first_hit            # slow-mo ramps straight into the first hit
        gap = first_hit - p.intro_end
        if gap < 0 or gap > p.long_gap + p.pre_roll:
            return p.intro_end          # long dead gap: the bridge will cut it
        sp = self.base_speed(p.drop_time)
        cand = self.beats[self.beats >= p.drop_time - EPS]
        if not len(cand):
            return p.intro_end
        b = float(cand[np.argmin(np.abs(cand - (p.drop_time + gap / sp)))])
        new_end = first_hit - (b - p.drop_time) * sp
        return new_end if new_end > p.intro_start + 0.2 else p.intro_end

    def intro_window(self) -> float:
        """Music time the montage starts at: whole beats before the drop."""
        p = self.p
        if p.intro_length <= 0:
            return 0.0
        n = max(1, int(round(p.intro_length / self.beat_period)))
        return max(0.0, p.drop_time - n * self.beat_period)

    def _build_intro(self, sched: Schedule, intro_end: float):
        p = self.p
        start = self.intro_window()
        L = p.drop_time - start
        R = min(max(0.0, p.intro_ramp), L * 0.5)
        if p.intro_length > 0:
            # Only the footage that fills the short intro at the slow-mo
            # speed is used; everything before it is cut.
            v = float(np.clip(p.intro_target_speed, p.intro_min_speed, 1.0))
            D = min(intro_end, v * (L - R) + R * (v + 1) / 2)
        else:
            D = intro_end - p.intro_start
        denom = L - R / 2
        s_slow = (D - R / 2) / denom if denom > EPS else 1.0
        if s_slow > 1.0:
            s_slow = 1.0
            sched.warnings.append("Intro footage longer than music intro: its start was trimmed.")
        elif s_slow < p.intro_min_speed:
            s_slow = p.intro_min_speed
            sched.warnings.append(
                f"Intro footage too short for full slow-mo; clamped to {s_slow:.2f}x "
                "and the intro begins later in the footage.")
        used = s_slow * (L - R) + R * (s_slow + 1) / 2 if s_slow < 1.0 else L
        src_start = max(0.0, intro_end - used)

        n = max(8, int(R * 120))
        drop, t_r = p.drop_time, p.drop_time - R
        outs = np.concatenate([[start], np.linspace(t_r, drop, n)]) if R > EPS \
            else np.array([start, drop])
        speed = np.where(outs < t_r, s_slow, s_slow + (1 - s_slow) * smoothstep((outs - t_r) / max(R, EPS)))
        speed[0] = s_slow
        srcs = np.concatenate([[0.0], np.cumsum(np.diff(outs) * (speed[1:] + speed[:-1]) / 2)])
        # remove the small numeric drift so the drop lands exactly on intro_end
        if srcs[-1] > EPS:
            srcs *= (intro_end - src_start) / srcs[-1]
        srcs += src_start
        self._push(sched, outs, srcs, "intro")
        sched.intro_speed = float(s_slow)
        sched.start = start
        return drop, intro_end

    # ------------------------------------------------------------ bridges
    def _try_cut(self, sched, out0, src0, target_src, post, pre):
        """Keep a short tail, hard-cut the dead footage, land target on a beat.

        The first-hit beat is the earliest one that leaves room for the
        pre-roll; any spare time goes to a longer pre-roll. Returns the beat
        time, or None when the gap is too short to cut anything.
        """
        tail = min(post, max(0.0, target_src - src0 - pre))
        t_outs, t_srcs = self._natural(out0, src0, src0 + tail)
        t_end, s_end = float(t_outs[-1]), float(t_srcs[-1])
        sp = self.base_speed(t_end)
        cand = self.beats[self.beats >= t_end + pre / sp - EPS]
        if not len(cand):
            return None
        beat = float(cand[0])
        cut_src = target_src - (beat - t_end) * sp
        if cut_src <= s_end + 0.02:
            return None
        self._push(sched, t_outs, t_srcs, "tail")
        self._linear(sched, t_end, cut_src, beat, target_src, "lead", cut_before=True)
        return beat

    def _bridge(self, sched, out0, src0, target_src, post, pre, info: dict):
        """Move from (out0, src0) to the first hit of the next combo.

        Long dead gaps are cut out; short gaps are kept and micro-locked if
        within tolerance; otherwise a small dead chunk is cut to reach the
        previous beat. Never stretches footage beyond the jitter tolerance.
        """
        p = self.p
        gap = target_src - src0
        if gap <= EPS:                         # already there (intro ends on the hit)
            info.update(target=out0, locked=True, status="locked")
            return out0
        if self.fill:
            return self._bridge_next_beat(sched, out0, src0, target_src, post, info)
        dead = gap - post - pre
        natural = out0 + self._natural_duration(out0, src0, target_src)

        if dead > p.long_gap:
            beat = self._try_cut(sched, out0, src0, target_src, post, pre)
            if beat is not None:
                info.update(target=beat, locked=True, status="cut")
                return beat
        b = self._nearest(self.beats, natural, out0 + 0.05)
        if b is not None and abs(natural - b) <= self.tol and self._speed_ok(gap, b - out0):
            self._lock_segment(sched, out0, src0, b, target_src, natural, "bridge")
            info.update(target=b, error_ms=(natural - b) * 1000, locked=True,
                        status="trim" if p.lock_mode == "trim" and natural > b else "ramp")
            return b
        # A little off: speed up / slow down to the nearest beat that stays
        # within the ramp bounds rather than leave the hit off-beat.
        cands = [x for x in self.beats[self.beats > out0 + 0.05][:4] if self._ramp_ok(gap, x - out0, out0)]
        if cands:
            b = float(min(cands, key=lambda x: abs(x - natural)))
            self._linear(sched, out0, src0, b, target_src, "bridge")
            info.update(target=b, error_ms=(natural - b) * 1000, locked=True, status="speed")
            return b
        if dead > 0.05:
            beat = self._try_cut(sched, out0, src0, target_src, min(post, dead / 2), pre)
            if beat is not None:
                info.update(target=beat, locked=True, status="cut")
                return beat
        # Can't sync without stretching: play it naturally.
        outs, srcs = self._natural(out0, src0, target_src)
        self._push(sched, outs, srcs, "bridge")
        info.update(status="unsynced")
        return float(outs[-1])

    def _next_step(self, out0: float) -> Optional[float]:
        """The combo-grid point one step after out0."""
        G = self.combo_grid
        k0 = int(np.argmin(np.abs(G - out0)))
        if abs(G[k0] - out0) <= 0.05:          # on the grid: exactly one step on
            j = k0 + self.combo_step
        else:                                  # off-grid: first point after it
            j = int(np.searchsorted(G, out0 + 0.05))
        return float(G[j]) if j < len(G) else None

    def _bridge_next_beat(self, sched, out0, src0, target_src, post, info):
        """Land the next combo's first hit on the very next beat (fill mode).

        The beat between the two hits holds a short tail of the old combo and
        a pre-roll into the new one, hard-cut together; if the gap is shorter
        than a beat the footage is simply ramped to fit.
        """
        b = self._next_step(out0)
        if b is None:
            return out0
        span = b - out0
        gap = target_src - src0
        if gap <= span * self.p.ramp_max_speed:      # close enough: ramp, no cut
            self._linear(sched, out0, src0, b, target_src, "bridge")
            info.update(target=b, locked=True,
                        status="ramp" if abs(gap - span) <= self.tol else "speed")
            return b
        tail = min(post, span / 2)
        self._linear(sched, out0, src0, out0 + tail, src0 + tail, "tail")
        self._linear(sched, out0 + tail, target_src - (span - tail), b, target_src, "lead",
                     cut_before=True)
        info.update(target=b, locked=True, status="cut")
        return b

    def _lock_segment(self, sched, out0, src0, out1, src1, natural, kind):
        """Lock src1 onto out1 by micro speed-ramp or (when late) trimming ms."""
        err = natural - out1
        sp = self.base_speed(out0)
        if self.p.lock_mode == "trim" and err > 0:
            trim = err * sp
            keep = (src1 - src0 - trim) / 2
            mid_src = src0 + keep
            outs, srcs = self._natural(out0, src0, mid_src)
            self._push(sched, outs, srcs, kind)
            # a few ms jump inside continuous footage: not flagged as a visible cut
            self._linear(sched, float(outs[-1]), mid_src + trim, out1, src1, "trim")
        else:
            self._linear(sched, out0, src0, out1, src1, kind)

    # ------------------------------------------------------------- build
    def build(self) -> Schedule:
        p = self.p
        sched = Schedule(params=p, ref_bpm=self.ref_bpm)
        hits = self.m.hits
        ids = self.m.combo_ids()
        placements = [HitPlacement(i, h.t, None, None, False, 0.0, ids[i], "ignored")
                      for i, h in enumerate(hits)]
        sched.placements = placements

        use_intro = p.intro_enabled and p.drop_time > 0.05 and p.intro_end > p.intro_start + 0.05
        combat_from = p.intro_end if use_intro else -1.0
        # Only real combos (>= min_combo_len steady hits) make the edit; every
        # other hit is dead footage the bridges cut straight through.
        kept_set = {i for g in self.kept for i in g}
        for i in range(len(hits)):
            if i not in kept_set:
                placements[i].status = "dropped"
        # a hit exactly at 'combat begins' is the first combat hit, not intro
        combos = [[i for i in grp if hits[i].t >= combat_from - 1e-3] for grp in self.kept]
        combos = [c for c in combos if c]
        sched.combos_kept = len(combos)
        sched.combos_dropped = len(self.m.combos()) - len(self.kept)
        sched.beats_per_hit = self.beats_per_hit
        sched.hit_period = self.hit_period
        vid_end = self.video_duration

        def place(i, out_t, info):
            pl = placements[i]
            pl.out_t = out_t
            pl.target = info.get("target")
            pl.error_ms = info.get("error_ms", 0.0)
            pl.locked = info.get("locked", False)
            pl.status = info.get("status", "unsynced")

        out_t, src_t = 0.0, 0.0
        first_already_placed = False
        if use_intro:
            first_hit = hits[combos[0][0]].t if combos else None
            out_t, src_t = self._build_intro(sched, self._intro_end_for(first_hit))
        elif combos:
            # No intro: open on a beat with the first hit, keeping up to pre_roll.
            h0 = hits[combos[0][0]].t
            sp = self.base_speed(0.0)
            cand = self.beats[(self.beats >= min(p.pre_roll, h0) / sp - EPS)
                              & (self.beats * sp <= h0 + EPS)]
            if len(cand):
                b = float(cand[0])
                self._linear(sched, 0.0, h0 - b * sp, b, h0, "lead")
                place(combos[0][0], b, dict(target=b, locked=True, status="locked"))
                out_t, src_t = b, h0
                first_already_placed = True

        post = 0.0  # the drop / video start has no combo tail to keep
        for ci, combo in enumerate(combos):
            if out_t >= self.music_duration:
                break
            if not (ci == 0 and first_already_placed):
                info: dict = {}
                out_t = self._bridge(sched, out_t, src_t, hits[combo[0]].t, post,
                                     p.pre_roll, info)
                src_t = hits[combo[0]].t
                place(combo[0], out_t, info)
            for i in combo[1:]:
                if out_t >= self.music_duration:
                    break
                out_t, src_t = self._step_in_combo(sched, out_t, src_t, i)
            post = p.post_roll

        # Tail / outro after the final combo
        remaining_music = self.music_duration - out_t
        if remaining_music > EPS and src_t < vid_end:
            extra = (p.post_roll + p.outro) if combos else vid_end - src_t
            end_src = min(vid_end, src_t + extra)
            outs, srcs = self._natural(out_t, src_t, end_src)
            b = self._next_step(out_t) if self.fill and combos else None
            if b is not None:
                # Every beat holds a hit, so the montage ends on the beat
                # after the last one: no empty beats at the end.
                end_src = min(vid_end, src_t + (b - out_t) * self.base_speed(out_t))
                outs, srcs = self._natural(out_t, src_t, end_src)
            self._push(sched, outs, srcs, "outro")

        # clamp to music length
        end = min(self.music_duration, sched.segments[-1].out_end if sched.segments else 0.0)
        sched.duration = max(0.0, end - sched.start)
        self._clip_to(sched, end)
        self._letterbox(sched, combos_all=self.kept)
        return sched

    def _step_in_combo(self, sched, out0, src0, i):
        """Advance to the next hit, exactly N combo-steps after the previous one.

        The target is counted in grid steps (the combo spacing), never
        "nearest beat or half beat", so every hit keeps the same musical
        value. Within the jitter tolerance the correction is a micro ramp
        (or trim); beyond it the footage is sped up / slowed down within the
        ramp bounds so the hit still lands on the beat.
        """
        p = self.p
        h = self.m.hits[i].t
        pl = sched.placements[i]
        G, step = self.combo_grid, self.combo_step
        natural = out0 + self._natural_duration(out0, src0, h)
        k0 = int(np.argmin(np.abs(G - out0)))
        local = (G[min(k0 + step, len(G) - 1)] - G[k0]) or self.beat_period
        n0 = max(1, int(round((natural - out0) / local)))
        if self.fill:
            return self._step_fill(sched, out0, src0, h, natural, pl)
        cands = []
        for n in sorted({n0, max(1, n0 - 1), n0 + 1}):
            j = k0 + n * step
            if j < len(G) and G[j] > out0 + 0.05:
                cands.append(float(G[j]))
        within = [g for g in cands if abs(natural - g) <= self.tol and self._speed_ok(h - src0, g - out0)]
        if within:
            g = min(within, key=lambda x: abs(natural - x))
            self._lock_segment(sched, out0, src0, g, h, natural, "combo")
            pl.status = "trim" if p.lock_mode == "trim" and natural > g else "ramp"
        else:
            ramps = [g for g in cands if self._ramp_ok(h - src0, g - out0, out0)]
            if not ramps:
                outs, srcs = self._natural(out0, src0, h)
                self._push(sched, outs, srcs, "combo")
                pl.out_t, pl.target, pl.locked = float(outs[-1]), None, False
                pl.error_ms, pl.status = 0.0, "unsynced"
                return float(outs[-1]), h
            g = min(ramps, key=lambda x: abs(np.log((h - src0) / (x - out0))))
            self._linear(sched, out0, src0, g, h, "combo")
            pl.status = "speed"
        pl.out_t, pl.target, pl.locked = g, g, True
        pl.error_ms = (natural - g) * 1000
        return g, h

    def _step_fill(self, sched, out0, src0, h, natural, pl):
        """Every-beat mode: the hit lands on the very next beat, always.

        Within the tolerance it's a micro ramp/trim; otherwise the footage is
        sped up or slowed down to fit the beat. A gap too long even for the
        fastest speed (e.g. a missed detection) keeps a short tail and
        hard-cuts to the hit inside that beat. Never left off the grid.
        """
        p = self.p
        g = self._next_step(out0)
        if g is None:                         # song over
            outs, srcs = self._natural(out0, src0, h)
            self._push(sched, outs, srcs, "combo")
            pl.out_t, pl.target, pl.locked, pl.status = float(outs[-1]), None, False, "unsynced"
            return float(outs[-1]), h
        span, gap = g - out0, h - src0
        if abs(natural - g) <= self.tol and self._speed_ok(gap, span):
            self._lock_segment(sched, out0, src0, g, h, natural, "combo")
            pl.status = "trim" if p.lock_mode == "trim" and natural > g else "ramp"
        elif gap <= span * max(p.ramp_max_speed, p.max_speed):
            self._linear(sched, out0, src0, g, h, "combo")
            pl.status = "speed"
        else:
            tail = min(p.post_roll, span / 2)
            self._linear(sched, out0, src0, out0 + tail, src0 + tail, "combo")
            self._linear(sched, out0 + tail, h - (span - tail), g, h, "lead", cut_before=True)
            pl.status = "cut"
        pl.out_t, pl.target, pl.locked = g, g, True
        pl.error_ms = (natural - g) * 1000
        return g, h

    def _clip_to(self, sched: Schedule, end: float):
        keep = []
        for seg in sched.segments:
            if seg.out_start >= end - EPS:
                break
            if seg.out_end > end:
                src_end = seg.src_at(end)
                mask = seg.out_knots < end
                seg = Segment(np.append(seg.out_knots[mask], end),
                              np.append(seg.src_knots[mask], src_end), seg.kind, seg.cut_before)
            keep.append(seg)
        sched.segments = keep
        sched.cuts = [c for c in sched.cuts if c < end]
        for pl in sched.placements:
            if pl.out_t is not None and pl.out_t > end:
                pl.out_t, pl.locked, pl.status = None, False, "ignored"

    def _letterbox(self, sched: Schedule, combos_all):
        p = self.p
        spans = []
        for grp in combos_all:
            outs = [sched.placements[i].out_t for i in grp
                    if sched.placements[i].out_t is not None]
            if len(outs) < max(1, p.min_combo_hits):
                continue   # (kept combos are already >= min_combo_len)
            spans.append((max(0.0, min(outs) - p.letterbox_lead), max(outs) + p.letterbox_hold))
        sched.letterbox = spans


def build_schedule(markers: Markers, params: SyncParams, video_duration: float,
                   music_duration: float) -> Schedule:
    return SyncEngine(markers, params, video_duration, music_duration).build()
