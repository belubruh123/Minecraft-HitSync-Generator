"""Headless pipeline: analyze -> align -> render, or launch the GUI.

    python -m hitsync                       # GUI
    python -m hitsync run VIDEO MUSIC -o out.mp4 [--intro 3 8] [--drop 12.5]
    python -m hitsync check                 # dependency report
    python -m hitsync beats SONG [--click out.wav]   # check the beat detection
"""
from __future__ import annotations

import argparse
import importlib
import sys

import numpy as np


def check_dependencies() -> bool:
    ok = True
    print(f"Python {sys.version.split()[0]}")
    for mod, pip_name in [("numpy", "numpy"), ("scipy", "scipy"), ("librosa", "librosa"),
                          ("cv2", "opencv-python"), ("PySide6", "PySide6-Essentials"),
                          ("PIL", "pillow"), ("sounddevice", "sounddevice")]:
        optional = mod == "sounddevice"             # preview plays silently without it
        try:
            m = importlib.import_module(mod)
            print(f"  [ok] {mod:<14} {getattr(m, '__version__', '')}")
        except Exception as exc:
            ok = ok and optional
            tag = "--" if optional else "!!"
            print(f"  [{tag}] {mod:<14} missing -> pip install {pip_name}  ({exc})"
                  + ("  (optional: preview sound)" if optional else ""))
    from .ffmpeg_utils import find_ffmpeg

    exe = find_ffmpeg()
    print(f"  [{'ok' if exe else '!!'}] ffmpeg         {exe or 'not found'}")
    return ok and bool(exe)


def _progress(msg, frac):
    print(f"\r{frac * 100:5.1f}%  {msg:<60}", end="", flush=True)


def beats(args) -> int:
    """Print what was detected in a song; optionally render it with clicks."""
    from .audio_analysis import analyze_audio, click_track
    from .sections import auto_intro_length

    a = analyze_audio(args.song)
    g = a.grid
    grid_beats = g.beats(a.duration)
    bars = grid_beats[g.is_downbeat(grid_beats)]
    print(f"Tempo      {g.bpm:.2f} BPM   (confidence {g.confidence:.0%}"
          f"{', tempo wanders: tracked beats are used' if g.drift else ''})")
    print(f"First beat {g.phase:.3f} s, first bar line {bars[0] if len(bars) else 0:.3f} s")
    if g.candidates:
        print("Other tempos considered: " + ", ".join(f"{b:.2f}" for b, _ in g.candidates[1:]))
    secs = a.sections
    print(f"Sound starts {secs.first_sound:.2f} s, song ends (last loud bar) {secs.song_end:.2f} s")
    print("Music start candidates:")
    for c in secs.top(3):
        mark = "  <- default" if abs(c.time - secs.best) < 1e-3 else ""
        print(f"  {_mmss(c.time)}  {c.label}{mark}")
    intro = auto_intro_length(secs.best, bars, secs.first_sound, g.period)
    print(f"Slow-mo intro: {intro:.2f} s of song before the start")
    if args.click:
        from .audio_mix import load_music, write_wav, SR

        music = load_music(args.song)
        beats_used = a.beats if g.drift else grid_beats
        clicks = click_track(beats_used, bars if not g.drift else beats_used[::4], len(music), SR)
        write_wav(args.click, np.clip(music * 0.7 + clicks[:, None], -1, 1))
        print(f"Wrote {args.click}: the song with a click on every detected beat "
              "(high click = bar start)")
    return 0


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:05.2f}"


def run(args) -> int:
    from .project import Project, default_output_path

    p = Project(video_path=args.video, music_path=args.music,
                output_path=args.output or default_output_path(args.video))
    p.detect.sensitivity = args.sensitivity
    p.sync.jitter_tolerance_ms = args.tolerance
    p.sync.lock_mode = args.lock
    p.sync.transition = args.transition
    p.sync.tempo_matching = not args.no_tempo
    p.render_params.interp = args.interp
    p.render_params.scale = args.scale
    p.render_params.hit_sound = args.hit_sound
    p.render_params.hit_sound_file = args.hit_sound_file or ""
    p.render_params.hit_volume = args.hit_volume
    p.render_params.encoder = args.encoder
    p.sync.min_combo_len = args.min_combo
    p.sync.fill_every_beat = not args.no_fill
    p.sync.bpm_override = args.bpm
    p.analyze(_progress)
    print()
    if args.drop is not None:
        b = p.markers.beats
        p.sync.drop_time = min(b, key=lambda x: abs(x - args.drop)) if b else args.drop
        p.auto_intro()                     # intro length depends on the drop
    if args.intro:
        p.sync.intro_start, p.sync.intro_end = args.intro
    if args.no_intro:
        p.sync.intro_enabled = False
    sched = p.recalculate()
    print(f"Grid {p.grid_bpm:.2f} BPM, hits {len(p.markers.hits)}, "
          f"real combos {len(p.real_combos())}, drop {p.sync.drop_time:.2f}s, "
          f"intro {p.sync.intro_start:.2f}-{p.sync.intro_end:.2f}s")
    print(sched.summary())
    if args.save_project:
        p.save(args.save_project)
        print(f"Project saved to {args.save_project}")
    if args.dry_run:
        return 0
    out = p.render(_progress)
    print(f"\nWrote {out}")
    return 0


def main(argv=None) -> int:
    # Non-UTF-8 Windows consoles (e.g. cp950) can't print every symbol we use.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(prog="hitsync", description="Minecraft PvP montage hit-sync")
    sub = ap.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="launch the desktop app (default)")
    g.add_argument("files", nargs="*", help="video / songs / project to open")
    sub.add_parser("check", help="verify dependencies")
    bt = sub.add_parser("beats", help="show the detected tempo, beats and music start")
    bt.add_argument("song")
    bt.add_argument("--click", help="write the song with a metronome click to this .wav")
    r = sub.add_parser("run", help="headless analyze + render")
    r.add_argument("video")
    r.add_argument("music")
    r.add_argument("-o", "--output")
    r.add_argument("--intro", nargs=2, type=float, metavar=("START", "END"),
                   help="intro footage range in video seconds (default: auto)")
    r.add_argument("--drop", type=float, help="drop time in music seconds (default: auto)")
    r.add_argument("--no-intro", action="store_true")
    r.add_argument("--no-tempo", action="store_true", help="disable tempo matching")
    r.add_argument("--tolerance", type=float, default=80.0, help="jitter tolerance (ms)")
    r.add_argument("--lock", choices=["ramp", "trim"], default="ramp")
    r.add_argument("--transition", choices=["cut", "flash"], default="cut")
    r.add_argument("--sensitivity", type=float, default=0.5)
    r.add_argument("--interp", choices=["nearest", "blend", "flow"], default="blend")
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--hit-sound", default="original",
                   choices=["original", "off", "classic", "strong", "crit", "knockback", "custom"],
                   help="original = the hit sound from your recording; others are read "
                        "from your .minecraft")
    r.add_argument("--min-combo", type=int, default=10,
                   help="hits a steady combo needs to make the edit (default 10)")
    r.add_argument("--no-fill", action="store_true",
                   help="don't force a hit onto every beat")
    r.add_argument("--bpm", type=float, default=0.0, help="force the static grid tempo")
    r.add_argument("--hit-sound-file", help="audio file for --hit-sound custom")
    r.add_argument("--hit-volume", type=float, default=0.8)
    r.add_argument("--encoder", default="auto", choices=["auto", "x264", "nvenc", "amf", "qsv"],
                   help="auto = GPU encoder when available, else x264")
    r.add_argument("--save-project")
    r.add_argument("--dry-run", action="store_true", help="analyze + align only")
    args = ap.parse_args(argv)

    if args.cmd == "check":
        return 0 if check_dependencies() else 1
    if args.cmd == "run":
        return run(args)
    if args.cmd == "beats":
        return beats(args)
    from .ui import main as gui_main

    return gui_main([sys.argv[0]] + list(getattr(args, "files", []) or []))


if __name__ == "__main__":
    sys.exit(main())
