"""Speed benchmark: video scan, music analysis, export and preview fps.

    python tools/bench.py [--seconds 60] [--height 1440] [--keep DIR]

Generates a synthetic screen-capture-like clip (H.264, moving texture, red
hit flashes, an audio track) and a 3-minute song, then times each stage.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_clip(path, seconds, height, fps=30):
    from hitsync.ffmpeg_utils import require_ffmpeg

    width = height * 16 // 9 // 2 * 2
    # moving test pattern + a red box flashing twice a second (the "hits")
    vf = (f"drawbox=x=iw/2-ih/8:y=ih/2-ih/6:w=ih/4:h=ih/3:color=red@1:t=fill:"
          f"enable='lt(mod(t\\,0.5)\\,0.12)'")
    cmd = [require_ffmpeg(), "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}",
           "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000",
           "-t", str(seconds), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "23", "-g", str(fps * 2), "-c:a", "aac", "-shortest", path]
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--keep", help="reuse/keep generated media in this folder")
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--style", default="clean", help="look used for the export timing")
    args = ap.parse_args()
    d = args.keep or tempfile.mkdtemp(prefix="hitsync_bench_")
    os.makedirs(d, exist_ok=True)
    clip = os.path.join(d, f"clip_{args.height}p_{int(args.seconds)}s.mp4")
    song = os.path.join(d, "song_180s.wav")
    if not os.path.exists(clip):
        print("generating clip…", flush=True)
        make_clip(clip, args.seconds, args.height)
    if not os.path.exists(song):
        from tests import synth_music as sm

        y, _ = sm.shape_like(duration=180.0)
        sm.write_wav(song, y)

    os.environ["HITSYNC_CACHE"] = os.path.join(d, "cache_unused")
    from hitsync.audio_analysis import analyze_audio
    from hitsync.config import DetectParams
    from hitsync.video_analysis import analyze_video

    t = time.perf_counter()
    va = analyze_video(clip, DetectParams())
    tv = time.perf_counter() - t
    print(f"video scan   {tv:6.2f} s  ({len(va.frame_times) / tv:.0f} frames/s)")
    t = time.perf_counter()
    analyze_audio(song)
    print(f"music        {time.perf_counter() - t:6.2f} s  (180 s song)")

    if not args.no_export:
        from hitsync.project import Project

        from hitsync.styles import apply_style

        p = Project(video_path=clip, music_path=song, output_path=os.path.join(d, "out.mp4"))
        apply_style(p, args.style)
        p.sync.min_combo_len = 4
        p.analyze()
        sched = p.recalculate()
        t = time.perf_counter()
        p.render()
        te = time.perf_counter() - t
        print(f"export       {te:6.2f} s  ({sched.duration:.1f} s montage, "
              f"{sched.duration * 30 / te:.0f} fps, {args.style} look)")
    if not args.keep:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
