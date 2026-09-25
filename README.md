# Hit-Sync: Minecraft PvP Montage Editor

Hit-Sync locks sword hits to the beat. It takes one gameplay recording and one song, and
renders a montage with:
- a short slow-mo intro (about 5 s) that ramps back to real time, with the first combo hit on the drop
- only real combos: 10 or more hits in a steady rhythm; everything else is cut
- a hit on every beat of the song's fixed tempo, from the drop to the end of the edit
- your recording's own hit sounds, placed on the beat
- letterbox bars during combos

## Run

```
run_gui.bat                                   # desktop editor
.venv\Scripts\python -m hitsync check         # verify dependencies + ffmpeg
.venv\Scripts\python -m hitsync run pvp.mp4 song.mp3 -o montage.mp4 --drop 16
.venv\Scripts\python -m unittest discover -s tests -t .               # tests
```

Setup from scratch: `py -3.13 -m venv .venv && .venv\Scripts\pip install -r requirements.txt`.
If there's no system ffmpeg, the tool uses the static binary that `imageio-ffmpeg` ships.

## Workflow

1. Pick the video and the music, then click **Analyze Media**. This finds the song's
   fixed tempo, the drop, the hits, and the real combos.
   - If the drop is wrong, click the MUSIC track where the music kicks in, then click
     **Drop = Playhead**.
   - If combat should start elsewhere, select the combo's first hit on the VIDEO track,
     then click **Combat = Selected Hit**. The slow-mo intro range is resized to match.
2. Check the markers on the timeline:
   - **Double-click** to add a beat (MUSIC track) or a hit (VIDEO track). A new hit snaps
     to the nearest detection peak.
   - **Right-click** a marker to delete it.
   - Select a hit, then press **S** to split the combo there or **M** to merge its combo
     with the previous one.
3. Click **Recalculate Alignment**. It runs in milliseconds from the markers in memory and
   never re-reads the media. It also runs automatically while **Live** is ticked.
4. Click **Play** (or press **Space**) to watch the edited montage in the preview, with
   the music and hit sounds. Click the MUSIC or EDIT track to move the playhead; the
   dashed playhead on the VIDEO track shows which moment of the source is on screen.
5. Export with **File > Export Video As (.mp4)…** (Ctrl+E) or the **Export MP4…** button.
   Your edits are saved separately with **File > Save Project (.json)** (Ctrl+S). Open a
   saved project with Ctrl+O; the cached analysis is stored in it, so nothing is re-analyzed.

Every slider has a number box next to it. Type an exact value and press Enter (or Tab, or
click away) to apply it, or press Esc to undo. Typed values can go past the slider's range
and aren't rounded to its steps (for example 101.25 BPM). Percent fields accept `25` or
`25%`, and a comma works as the decimal point.

## Static beat = a hit on every beat

With **Static beat** ticked (the default), every beat of the grid from the drop to the end
of the montage has exactly one hit, locked on the beat. On the MUSIC track that shows as one
unbroken chain of green triangles, with no gaps and no orange. What makes that true:

- One hit per beat, always. **A hit on every beat** and **Beats per hit** are implied and
  greyed out.
- A hit never falls off the grid. A small timing error gets a micro ramp. A bigger one is
  sped up or slowed down to fit the beat. A gap too long even for the fastest speed (for
  example a missed detection) keeps a short tail and hard-cuts to the hit inside that beat.
- The montage ends on the beat after the last hit, so there are no empty beats at the end.
  The music fades out over that final beat only, never over the hits.

Untick **Static beat** to get the dynamic grid back, with beats-per-hit spacing and
tempo matching.

## Analysis speed and re-analysis

- **Only what changed is re-analyzed.** Once a pair is analyzed, picking a different
  music file re-analyzes only the music (beats, grid, drop); your hits, combos and hit
  edits stay. Picking a different video re-scans only the video (hits, combos); the beat
  grid and a drop you set by hand stay. This runs automatically when you pick the new
  file, and also when you click **Analyze Media**.
- **Files you've analyzed before load instantly.** Results are cached per file (by path,
  size and modification time) in `%LOCALAPPDATA%\HitSync\cache` (override with
  `HITSYNC_CACHE`). Switching back to an earlier song or clip takes well under a second. An
  edited file is analyzed again.
- **Music and video are analyzed at the same time.**
- **Speed on a 4:45 1440p Game Bar capture plus a 3:16 song:**

  | Step | Before | Now |
  |---|---|---|
  | Video scan | 103 s | 13 s |
  | Music analysis | 21 s | 3.5 s |
  | Full first analysis | ~124 s | ~18 s |
  | Music changed only | ~124 s | ~3.5 s |
  | Previously analyzed files | ~124 s | 0.07 s |

  The video scan is faster because ffmpeg decodes and shrinks the frames to 320 px in
  native code, instead of OpenCV decoding full 1440p frames and resizing them in Python.
  Music analysis is faster because the drum/harmonic split's median filters run on all
  cores, with identical output.

## Export speed

A 56 s montage from a 1440p capture exports in about 30–36 s. It used to take about
190 s (8.7 fps); it now runs at 46–57 fps.

- **GPU encoding:** **Encoder: auto** uses the GPU's H.264 encoder (AMD AMF, NVIDIA NVENC or
  Intel Quick Sync) when one works on your machine. Otherwise it uses x264 on the CPU.
  Choose **x264** to force CPU encoding (`--encoder x264` on the command line).
- **Parallel parts:** the video renders as up to 3 consecutive parts at once, each with its
  own decoder and encoder. The parts are then joined without re-encoding. The soundtrack
  is mixed at the same time.
- **Background decoding:** source frames are decoded ahead in a background thread, straight
  into memory. Full-size frames travel as YUV, half the bytes of RGB.
- **Cheaper slow-mo:** optical flow is computed at 640 px wide and only the final warp runs
  at full resolution.

**Frame accuracy:** ffmpeg's "accurate" seek sometimes drops the wanted frame on
variable-frame-rate captures. When it did, every frame after that seek came out one frame
(33 ms) late: 2 of the 6 seeks in the test montage. Every decoded frame is now identified
by its real timestamp instead of by counting frames, so a seek can't shift the footage.
Renders with 1 part and with 3 parts are now frame-identical.

## Hit sounds

Every hit placed in the edit gets a hit sound, in both the preview and the export.

**`original`** (the default) uses each hit's own sound from your recording, kept whole:
each hit plays the recorded audio from its attack up to the next hit, so the entire sound
and anything right after it (crit, sweep) is kept. The attack is lined up exactly with the
beat. The last hit of a combo rings out until the next combo's first hit starts. One volume
is applied to every hit, so loud and quiet hits keep their natural balance.

Screen recordings don't keep sound and picture in lockstep. In a Game Bar capture the hit
sound came 30 ms to 350 ms before the red flash, and the gap changed from hit to hit. So
the tool detects every distinct sound in the recording and pairs the hits with them in
order, rather than searching a fixed window. If the recording has no audio track, the tool
uses `classic` instead.

The other presets are read from your own Minecraft Java install
(`%APPDATA%\.minecraft`); nothing from the game is bundled here.

| Preset | Sound |
|---|---|
| `classic` | 1.8 PvP hurt sound (`damage/hit1-3`) |
| `strong`, `crit`, `knockback` | 1.9+ attack sounds |
| `custom` | any audio file you choose |
| `off` | no hit sounds |

Like the game, each hit picks a random variant and a random pitch; turn that off with
**Random pitch**. If no Minecraft install is found, a synthesized punch is used instead.
Set `HITSYNC_MINECRAFT_DIR` to point at a non-standard install.

## How the alignment works (`hitsync/sync_engine.py`)

The output timeline is the music timeline. The schedule is a list of time-remap segments
that map output time to source time. A segment can also mark a hard cut.

| Feature | Mechanism |
|---|---|
| Short intro | The montage starts **Intro length** (default 5 s, snapped to whole beats) before the drop, so the song before that is cut. The intro uses only the footage just before "combat begins" that fills those seconds at **Intro slow-mo speed** (about 2 s of video at 0.4x), so the video before that is cut too. Set the length to 0 to keep the whole music intro. |
| Static beat | One tempo and phase is fitted to all the detected beats with a robust line fit. Beats are counted from the last beat known to be on the pulse, so a stretch where the tracker follows 1/8 notes (two detections per beat) is ignored instead of doubling the grid. The drop snaps to this grid. **BPM** and **Grid offset** override the fit. |
| Real combos | Hits are grouped by rhythm: a gap more than ±30% (**Rhythm tolerance**) away from the combo's running hit period starts a new group. Only groups of at least **Min hits per combo** (default 10) make the edit; every other hit is dead footage and gets cut. |
| Combo spacing | The average gap between hits in the real combos is compared with the beat to choose beats-per-hit (½, 1, 2, …). For example, a 0.60 s hit gap against a 0.594 s beat gives 1 hit per beat. |
| A hit on every beat | Each hit goes exactly one step after the previous one, never skipping a beat. The next combo's first hit lands on the beat right after the previous combo's last hit: a short tail and a pre-roll share that one beat, with a hard cut between them. The intro ramps straight into the first combo hit, which lands exactly on the drop. |
| Timing fixes | Within ±tolerance (default 80 ms), a micro speed-ramp (`ramp`) or trimming the late milliseconds (`trim`) locks the hit. Larger gaps are fixed by speeding up or slowing down between **Slowest** and **Fastest speed-ramp** (default 0.6–1.7×). |
| Letterbox | Each combo with at least `min_combo_hits` hits gets bars. They ease in just before the first hit, stay while the combo lasts, and ease out when it breaks. |

## Detection

- **Beats:** librosa beat tracking, with each beat snapped to its backtracked onset attack
  (±4 ms on synthetic audio). These beats then feed the static grid fit.
- **Drop:** the grid beat with the largest jump in drum energy (percussive RMS). Vocals
  or pads can mask the drop in a full-mix energy curve, so the drums are measured alone.
- **Hits:** the onset of the red hurt tint in the centre of the screen, plus changes in
  screen velocity (phase correlation). Hit times use the rising edge of the tint. Each
  flash is normalized against the local red level, so a far-away opponent counts like a
  close one. Hits closer than 0.4 s are merged, because Minecraft's damage invulnerability
  lasts 0.5 s. A gap about twice the hit rhythm is searched again for a faint hit that was
  missed. The raw signals are cached, so changing the sensitivity or clicking
  **Re-detect Hits** doesn't decode the video again.

## Layout

```
hitsync/
  config.py          parameter dataclasses
  models.py          beats/hits/combo editing (split, merge, delete, add)
  beatgrid.py        static beat grid fit, subdivisions, tempo curve (numpy only)
  audio_analysis.py  librosa: beats, onsets, RMS, drop detection
  video_analysis.py  ffmpeg-piped scan: damage tint + screen velocity -> hits
  analysis_cache.py  per-file on-disk cache of analysis results
  sync_engine.py     edit-decision engine -> Schedule
  audio_mix.py       soundtrack: music + original / Minecraft hit sounds
  renderer.py        timestamp-exact frame source, remap (nearest/blend/flow), parallel encode
  project.py         state, analysis orchestration, JSON save/load
  cli.py             `python -m hitsync [gui|run|check]`
  gui/app.py         customtkinter window
  gui/timeline.py    interactive tk.Canvas timeline with playhead
  gui/player.py      in-app playback (video + soundtrack)
tests/               engine, rules, audio/VFR unit tests + end-to-end synthetic-media test
```

**Variable frame rate:** screen recorders such as Game Bar and OBS produce VFR video.
Every time-to-frame lookup uses each frame's real timestamp. Frames are decoded through
ffmpeg with timestamp-accurate seeking, adjusted for the video stream's start offset,
because OpenCV's frame seeking returns the wrong frame on these files.

The output audio is the music plus the hit sounds; the rest of the gameplay audio isn't
mixed in.
