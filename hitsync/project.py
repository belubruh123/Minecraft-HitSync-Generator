"""Project state: media paths, parameters, cached analysis, edited markers."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import analysis_cache, sections as sections_mod
from .audio_analysis import AudioAnalysis, analyze_audio
from .config import DetectParams, RenderParams, SyncParams
from .models import ComboChoice, Markers
from .sync_engine import Schedule, build_schedule
from .video_analysis import VideoAnalysis, analyze_video, detect_hits

PROJECT_VERSION = 1


@dataclass
class Project:
    video_path: str = ""
    music_path: str = ""
    output_path: str = ""
    detect: DetectParams = field(default_factory=DetectParams)
    sync: SyncParams = field(default_factory=SyncParams)
    render_params: RenderParams = field(default_factory=RenderParams)
    audio: Optional[AudioAnalysis] = None
    video: Optional[VideoAnalysis] = None
    markers: Markers = field(default_factory=Markers)
    schedule: Optional[Schedule] = None
    grid_bpm: float = 0.0          # tempo of the fitted static grid
    downbeats: list = field(default_factory=list)   # bar starts (music s)
    # Combo picker: order + on/off + slow-mo lead-in. Until the user changes
    # it, the plan follows the detected combos (every real combo, in time order).
    combo_plan: list = field(default_factory=list)
    plan_custom: bool = False
    # Fingerprints of the media the cached analysis came from, so changing
    # only the music (or only the video) re-analyses just that file.
    audio_key: str = ""
    video_key: str = ""

    # ------------------------------------------------------------ analysis
    @property
    def analyzed(self) -> bool:
        return self.audio is not None and self.video is not None

    @property
    def video_duration(self) -> float:
        return self.video.duration if self.video else 0.0

    @property
    def music_duration(self) -> float:
        return self.audio.duration if self.audio else 0.0

    def current_audio_key(self) -> str:
        return analysis_cache.fingerprint("audio", self.music_path)

    def current_video_key(self) -> str:
        d = self.detect
        return analysis_cache.fingerprint("video", self.video_path, w=d.analysis_width,
                                          fps=d.max_analysis_fps, roi=[d.roi_w, d.roi_h])

    def stale(self) -> tuple[bool, bool]:
        """(music needs analysis, video needs analysis)."""
        return (self.audio is None or self.audio_key != self.current_audio_key(),
                self.video is None or self.video_key != self.current_video_key())

    def analyze(self, progress=None, cancel: threading.Event | None = None,
                force: bool = False) -> tuple[bool, bool]:
        """Analyse whatever changed since the last run.

        Only a new/changed music file is re-analysed (beats, grid, drop) and
        only a new/changed video is re-scanned (hits); the other side, with
        any edits made to it, is kept. Unchanged files load from the on-disk
        cache. Music and video run in parallel. When nothing changed,
        everything is re-derived from the cached signals (instant).
        Returns (music analysed, video analysed).
        """
        report = progress or (lambda *_: None)
        need_a, need_v = (True, True) if force else self.stale()
        if not (need_a or need_v):
            need_a = need_v = True              # "Analyze" again = redo from cache
            redo_only = True
        else:
            redo_only = False
        akey, vkey = self.current_audio_key(), self.current_video_key()
        frac = {"a": 0.0, "v": 0.0}
        weight = {"a": 0.2 if need_v else 1.0, "v": 0.8 if need_a else 1.0}

        def sub(kind, label):
            def cb(msg, f):
                frac[kind] = f
                report(f"[{label}] {msg}", sum(weight[k] * frac[k] for k in frac
                                               if (need_a if k == "a" else need_v)))
            return cb

        def do_audio():
            cached = None if force else analysis_cache.load(akey)
            if cached:
                try:
                    return AudioAnalysis.from_dict(cached)
                except (KeyError, TypeError, ValueError):
                    pass
            a = analyze_audio(self.music_path, sub("a", "music"))
            analysis_cache.save(akey, a.to_dict())
            return a

        def do_video():
            cached = None if force else analysis_cache.load(vkey)
            if cached:
                try:
                    return VideoAnalysis.from_dict(cached)
                except (KeyError, TypeError, ValueError):
                    pass
            v = analyze_video(self.video_path, self.detect, sub("v", "video"), cancel)
            if not (cancel is not None and cancel.is_set()):
                analysis_cache.save(vkey, v.to_dict())
            return v

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(2) as ex:
            fa = ex.submit(do_audio) if need_a else None
            fv = ex.submit(do_video) if need_v else None
            audio = fa.result() if fa else None
            video = fv.result() if fv else None
        if cancel is not None and cancel.is_set():
            return False, False
        if audio is not None:
            self.audio, self.audio_key = audio, akey
            self.apply_beat_grid()
            self.auto_drop()
        if video is not None:
            self.video, self.video_key = video, vkey
            self.redetect_hits()
        self.auto_intro()                  # depends on the drop and the hits
        if redo_only:
            return False, False
        return need_a, need_v

    def apply_beat_grid(self):
        """Beats on the timeline = the song's constant grid.

        The grid is measured from the audio itself (tempo, phase, bar lines).
        ``bpm_override`` forces a tempo, ``grid_offset_ms`` nudges the phase.
        Songs whose tempo wanders (or the dynamic mode) use tracked beats.
        """
        if self.audio is None:
            return
        s, a = self.sync, self.audio
        if s.static_grid and not (a.grid.drift and s.bpm_override <= 0):
            g = a.grid_for(s.bpm_override, s.grid_offset_ms / 1000.0)
            beats = g.beats(a.duration)
            self.markers.set_beats(beats)
            self.downbeats = [float(b) for b in beats[g.is_downbeat(beats)]]
            self.grid_bpm = g.bpm
        else:
            beats = a.beats
            self.markers.set_beats(beats)
            self.downbeats = [float(b) for b in beats[::4]]
            self.grid_bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) > 1 else 0.0

    @property
    def beat_period(self) -> float:
        b = self.markers.beats
        return float(np.median(np.diff(b))) if len(b) > 1 else 0.5

    # -------------------------------------------------------- beat fixes
    def set_tempo_factor(self, factor: float):
        """Half / double the grid tempo (the detector picked the wrong level)."""
        if self.grid_bpm > 0:
            self.sync.bpm_override = self.grid_bpm * factor
            self.apply_beat_grid()

    def shift_half_beat(self):
        """The grid sits on the off-beats: move it by half a beat."""
        half = self.beat_period / 2 * 1000.0
        self.sync.grid_offset_ms = (self.sync.grid_offset_ms + half) % (2 * half)
        self.apply_beat_grid()

    def nudge_grid(self, ms: float):
        self.sync.grid_offset_ms += ms
        self.apply_beat_grid()

    def tap_tempo(self, taps) -> bool:
        """Tempo and phase from times tapped along with the music."""
        from .music_grid import tap_tempo

        res = tap_tempo(taps)
        if res is None or self.audio is None:
            return False
        bpm, phase = res
        self.sync.bpm_override = round(bpm, 2)
        self.sync.grid_offset_ms = 0.0
        g = self.audio.grid_for(self.sync.bpm_override)
        # tapped phase wins over the fitted one when they disagree by > 40 ms
        d = (phase - g.phase + g.period / 2) % g.period - g.period / 2
        if abs(d) > 0.04:
            self.sync.grid_offset_ms = d * 1000.0
        self.apply_beat_grid()
        return True

    def reset_grid(self):
        self.sync.bpm_override = 0.0
        self.sync.grid_offset_ms = 0.0
        self.apply_beat_grid()

    def redetect_hits(self):
        """Re-run peak picking on cached signals (no media decoding)."""
        if self.video is None:
            return
        self.markers.set_hits(detect_hits(self.video, self.detect), self.sync.combo_gap,
                              self.sync.combo_regularity)

    def regroup(self):
        self.markers.auto_group(self.sync.combo_gap, self.sync.combo_regularity)

    def real_combos(self):
        """Combos long enough to make the edit."""
        return [g for g in self.markers.combos() if len(g) >= max(1, self.sync.min_combo_len)]

    # ------------------------------------------------------------ combo plan
    def _combo_for_key(self, key: float, groups):
        hits = self.markers.hits
        for k, g in enumerate(groups):
            a, b = hits[g[0]].t, hits[g[-1]].t
            if a - 0.3 <= key <= b + 1e-6:
                return k
        return None

    def plan_entries(self):
        """[(ComboChoice, hit indices)] in montage order, reconciled with the
        current hits: entries whose combo vanished are dropped, new real
        combos are added at the end."""
        groups = self.markers.combos()
        real = self.real_combos()
        if not self.plan_custom:
            self.combo_plan = [ComboChoice(self.markers.hits[g[0]].t, True,
                                           any(c.lead_in and abs(c.key - self.markers.hits[g[0]].t) < 0.3
                                               for c in self.combo_plan))
                               for g in real]
        out, used = [], set()
        for c in self.combo_plan:
            k = self._combo_for_key(c.key, groups)
            if k is None or k in used:
                continue
            used.add(k)
            c.key = self.markers.hits[groups[k][0]].t
            out.append((c, groups[k]))
        for g in real:
            k = groups.index(g)
            if k not in used:
                used.add(k)
                c = ComboChoice(self.markers.hits[g[0]].t, True, False)
                out.append((c, g))
        self.combo_plan = [c for c, _ in out]
        return out

    def engine_plan(self):
        return [(g, c.lead_in) for c, g in self.plan_entries() if c.enabled]

    def set_plan(self, choices):
        """Replace the plan (UI: reorder / tick / lead-in)."""
        self.combo_plan = list(choices)
        self.plan_custom = True
        self.auto_intro()

    def reset_plan(self):
        self.plan_custom = False
        self.combo_plan = []
        self.auto_intro()

    def rank_combos(self) -> dict:
        """{combo key: score 0..1}: more hits, steadier rhythm, cleaner hits."""
        hits = self.markers.hits
        raw = {}
        for g in self.real_combos():
            t = np.array([hits[i].t for i in g])
            gaps = np.diff(t)
            steady = 1.0 - float(np.clip(np.std(gaps) / max(np.mean(gaps), 1e-6), 0, 1)) \
                if len(gaps) > 1 else 0.5
            strength = float(np.mean([min(1.5, hits[i].strength) for i in g]))
            raw[hits[g[0]].t] = len(g) * (0.3 + 0.7 * steady) * (0.5 + 0.5 * strength)
        top = max(raw.values(), default=1.0)
        return {k: v / top for k, v in raw.items()}

    def auto_pick(self):
        """Use the best combos that fit the song, best-first order kept in
        time order (so the story still reads forward)."""
        ranks = self.rank_combos()
        entries = self.plan_entries()
        if not entries:
            return
        bph = max(0.5, float(self.schedule.beats_per_hit) if self.schedule else 1.0)
        end = self.music_end()
        capacity = (end - self.sync.drop_time) / max(1e-6, self.beat_period * bph)
        chosen, n = set(), 0
        for c, g in sorted(entries, key=lambda e: -ranks.get(e[0].key, 0.0)):
            if n + len(g) <= capacity or not chosen:
                chosen.add(c.key)
                n += len(g)
        plan = sorted((c for c, _ in entries), key=lambda c: (c.key not in chosen, c.key))
        for c in plan:
            c.enabled = c.key in chosen
        self.set_plan(plan)

    def combat_starts_at(self, t: float) -> float | None:
        """Gameplay mark: the combo from the first hit at/after ``t`` opens
        the montage (the slow-mo intro plays the footage before it)."""
        hits = self.markers.hits
        idx = next((i for i, h in enumerate(hits) if h.t >= t - 0.25), None)
        if idx is None:
            return None
        if not hits[idx].combo_start:
            self.markers.split_combo_at(idx)
        key = hits[idx].t
        entries = self.plan_entries()
        plan = [c for c, _ in entries if abs(c.key - key) > 1e-6]
        first = next((c for c, _ in entries if abs(c.key - key) <= 1e-6), None) or \
            ComboChoice(key, True, False)
        first.enabled = True
        self.set_plan([first] + plan)
        return key

    def music_end(self) -> float:
        """Where the montage music can run to (end of the last song)."""
        if self.audio is None:
            return max(self.markers.beats, default=0.0)
        end = self.audio.sections.song_end or self.audio.duration
        return min(self.audio.duration, max(end, self.sync.drop_time))

    def start_candidates(self) -> list:
        """[(time, label)] of likely music starts, snapped to the grid."""
        if self.audio is None:
            return []
        return [(self._snap_beat(c.time), c.label) for c in self.audio.sections.top(3)]

    def _snap_beat(self, t: float) -> float:
        b = self.markers.beats
        if not b:
            return t
        arr = np.asarray(b)
        return float(arr[np.argmin(np.abs(arr - t))])

    def auto_drop(self) -> float:
        """Music start (the drop) = the song's first big moment."""
        if self.audio is None:
            return 0.0
        return self.set_drop(self.audio.sections.best)

    def set_drop(self, t: float) -> float:
        """Set where the first combo hit lands (snapped to the beat grid);
        the slow-mo intro length follows the song when it is automatic."""
        t = self._snap_beat(t)
        self.sync.drop_time = t
        if self.sync.intro_auto and self.audio is not None:
            secs = self.audio.sections
            self.sync.intro_length = sections_mod.auto_intro_length(
                t, self.downbeats, secs.first_sound, self.beat_period)
        return t

    def auto_intro(self):
        """Intro = footage before the first combo of the plan, sized for slow-mo.

        Combat begins at the first hit of the first combo that makes the
        edit; the intro range before it is sized so the music intro plays it
        at ``intro_target_speed`` (including the ramp back to 1.0x).
        """
        if not self.markers.hits or self.video is None:
            return
        s = self.sync
        plan = self.engine_plan()
        first = self.markers.hits[plan[0][0][0]].t if plan else self.markers.hits[0].t
        end = first if s.every_beat else max(0.0, first - s.pre_roll)
        L = max(s.drop_time, 1.0)
        if s.intro_length > 0:
            L = min(L, s.intro_length)     # only this much of the song is intro
        R = min(s.intro_ramp, L / 2)
        v = s.intro_target_speed
        want = v * (L - R) + R * (v + 1) / 2
        s.intro_end = end
        s.intro_start = max(0.0, end - want)

    # ------------------------------------------------------------ schedule
    def recalculate(self) -> Schedule:
        vd = self.video_duration or (max((h.t for h in self.markers.hits), default=0.0) + 5)
        md = self.music_duration or (max(self.markers.beats, default=0.0) + 5)
        plan = self.engine_plan() if self.markers.hits else None
        self.schedule = build_schedule(self.markers, self.sync, vd, md, plan=plan,
                                       downbeats=self.downbeats)
        return self.schedule

    def render(self, progress=None, cancel=None, out_path: str | None = None) -> str:
        from .renderer import render

        if self.video is None:
            raise RuntimeError("Analyze the media first.")
        sched = self.schedule or self.recalculate()
        out = out_path or self.output_path or default_output_path(self.video_path)
        return render(self.video_path, self.music_path, out, sched, self.video.info,
                      self.render_params, progress, cancel, self.video.frame_times,
                      self.video.pts_offset or 0.0)

    # ------------------------------------------------------------- persist
    def to_dict(self) -> dict:
        return {
            "version": PROJECT_VERSION,
            "video_path": self.video_path, "music_path": self.music_path,
            "output_path": self.output_path,
            "detect": self.detect.to_dict(), "sync": self.sync.to_dict(),
            "render": self.render_params.to_dict(),
            "markers": self.markers.to_dict(), "downbeats": list(self.downbeats),
            "combo_plan": [c.to_dict() for c in self.combo_plan],
            "plan_custom": self.plan_custom,
            "audio_key": self.audio_key, "video_key": self.video_key,
            "audio": self.audio.to_dict() if self.audio else None,
            "video": self.video.to_dict() if self.video else None,
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, default=_json_default)

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        p = cls(d.get("video_path", ""), d.get("music_path", ""), d.get("output_path", ""),
                DetectParams.from_dict(d.get("detect")), SyncParams.from_dict(d.get("sync")),
                RenderParams.from_dict(d.get("render")))
        p.markers = Markers.from_dict(d.get("markers", {}))
        if len(p.markers.beats) > 1:
            p.grid_bpm = 60.0 / float(np.median(np.diff(p.markers.beats)))
        if d.get("audio"):
            try:
                p.audio = AudioAnalysis.from_dict(d["audio"])
                p.audio_key = d.get("audio_key") or p.current_audio_key()
            except (KeyError, TypeError, ValueError):
                p.audio, p.audio_key = None, ""       # older format: re-analysed on demand
        p.downbeats = list(d.get("downbeats") or p.markers.beats[::4])
        p.combo_plan = [ComboChoice.from_dict(c) for c in d.get("combo_plan", [])]
        p.plan_custom = bool(d.get("plan_custom", False))
        if d.get("video"):
            p.video = VideoAnalysis.from_dict(d["video"])
            p.video_key = d.get("video_key") or p.current_video_key()
            if p.video.pts_offset is None and os.path.isfile(p.video_path):
                from .ffmpeg_utils import find_ffmpeg, first_video_pts

                p.video.pts_offset = first_video_pts(p.video_path) if find_ffmpeg() else 0.0
        return p


def default_output_path(video_path: str) -> str:
    base, _ = os.path.splitext(video_path or "montage")
    return base + "_hitsync.mp4"


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
