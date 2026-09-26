# Hit-Sync: Minecraft PvP montage maker

Drop in a gameplay recording and a song. Hit-Sync finds the song's beat and
where the music really starts, finds your hits and combos, and makes a montage
where every hit lands on the beat, with a slow-mo intro, montage effects and
text. Windows and macOS (Linux works too).

## Start

| | |
|---|---|
| **Windows** | Double-click `run_gui.bat` |
| **macOS** | Double-click `run_gui.command` (the first time: right-click → Open) |
| **Linux** | `./run_gui.sh` |

The first start sets everything up by itself: a private Python environment
with the requirements (a few minutes, once). You only need
[Python 3.10+](https://www.python.org/downloads/). On Windows, tick "Add python.exe
to PATH" when installing it. ffmpeg is included; a system ffmpeg is used if you
have one.

## Make a montage

1. **Drop your video and a song** anywhere on the window. You can also click
   the two cards at the top. Each file is analysed the moment you drop it.
2. **Press Space** to watch the montage. **F** plays it fullscreen.
3. **Export video** (top right) saves the .mp4.

That's it. Everything below is optional.

### The smart bar

- **Tempo check** (e.g. `96.0 BPM ✓`, or `?` when unsure): click it to pick one
  of the other likely tempos (e.g. `Use 192.0 BPM`), **type the BPM**, or tap
  along.
- **Music starts**: the best guesses for where the song kicks in (e.g.
  `0:20 Beat kicks in`, `1:00 Big moment`). Click one to use it, or pick by
  ear (below).
- **Style**: one-click looks.

  | Style | Look |
  |---|---|
  | Clean | No effects |
  | Montage (default) | Vibrant colours, light motion blur, a soft zoom punch on hits, flash between combos |
  | Hype | Punchy colours, shake, flash and RGB split on hits, zoom transitions |
  | Cinematic | Teal-orange colours, bars through the first combo, dip-to-black transitions |

  Changing any of these settings yourself makes the style *Custom*.
- **Bottom row**: *Hit volume*, *Music volume* and *Beats per hit* (Auto, ½, 1,
  2). Auto picks from your hit speed, so a fast song gets a hit every other
  beat and a slow one two per beat.

### Pick the starts by listening and watching

- **Song** tab: the song's waveform with its beats and bar lines. Play it and
  press **✔ Music starts here (M)** where your first combo hit should land. It
  snaps to the beat.
- **Gameplay** tab: your raw recording with its own sound. Press **✔ Combat
  starts here (M)** where the action should begin. The footage before it
  becomes the slow-mo intro, and that combo opens the montage.

If the beat sounds off, tick **Beat click** to hear a click on every detected
beat. The fixes are in the Song tab:

| Fix | What it does |
|---|---|
| Type BPM | You know the tempo: type it (the beat is fitted to it) |
| Tap (T) | Tap along 8 times or more |
| ½× / 2× | The tempo is half or double what it should be |
| Shift ½ beat | The clicks sit between the beats |
| ◀ 10 ms / 10 ms ▶ | The clicks are a little late or early |
| Reset | Back to what was detected |

### Combos panel

Every real combo (10+ steady hits) is shown with a thumbnail and ★ rating.

- **Drag** combos (or use ▲▼) to change the order.
- **Tick** a combo to use it in the montage.
- **★ Auto-pick best** uses the best combos that fit the song.
- **Slow-mo lead-in**: the footage before that combo plays in slow motion
  (with letterbox bars and a flash), and the combo still starts on the beat.
- **🎬 Focus bars**: black bars at the top and bottom while that combo plays,
  a cinematic focus on it. After a slow-mo part the bars simply stay on.
- **✦ Effects**: choose which hits get the zoom/shake/flash/RGB effects.
  Quick picks: All, None, Every 2nd, First & last. On the timeline, **E** toggles
  a selected hit.

### Text panel

**＋ Add text at playhead** adds a caption. The default style is **Glow**:
white rounded letters (Fredoka) with a purple outline and a vivid purple glow.
For each caption you can:

- type the text
- choose **Top / Center / Bottom**, a style (Glow, YouTuber, Yellow, Red,
  Impact, Minimal) and an animation (Pop, Slide, Fade, Typewriter)
- set **⇤ Start / End ⇥** at the playhead, or drag the yellow bar's edges on
  the timeline (they snap to beats). Times are shown on the montage clock,
  like the player.

Add as many captions as you like, one after another.

### Effects panel

- **Flashy effects on part of the montage**: Flashy (beat strobe, glow, colour
  boost, RGB and zoom pulses), Glow, RGB split, Strobe, Black & white, Shake,
  Zoom pulse, Motion blur+. Choose where: 4 or 8 bars from the playhead, *the
  combo at the playhead*, or *playhead to the end*. Each range has a strength
  and shows on the timeline, where you can drag its edges.
- **Hit effects**: zoom punch, shake, flash, RGB split, and *velocity* (slow on
  each impact, fast in between).
- **Look**: filter (Vibrant, Cinematic, Warm, Cool, Punchy, B&W, Vintage,
  Night) and its strength, motion blur, vignette, bar pulse.
- **Focus bars**: the automatic black bars (slow-mo parts, intro + first
  combo, every combo, or off). Bars on single combos are ticked in Combos.
- **Between combos**: cut, flash, zoom, whip or dip.
- **Start and end**: fade from black or flash in; end with a cut, fade to
  black or fade to white, each with its own length. The montage keeps playing
  past the last hit so the fade never covers a hit.

### Music panel (more songs)

Drop more songs, or press **＋ Add song** in the **Music** tab. They play top
to bottom. The next song fades in over about a bar, playing its own lead-up,
and reaches *its* start point exactly on the bar where the song before it
stops: no gap, and the beat grid continues.

- **Cut a song**: click it in the Music tab (it opens in the Song view), play
  it, and press **✂ Switch to next song here (C)** where the next song should
  take over. For the last song the button is **✂ End the music here**. The
  cut snaps to the bar line. Without a cut, a song plays until its last loud
  bar (the last song plays to its end); **✂ Auto** goes back to that.
- **Change the order**: drag the songs, or use ▲▼. The first song's start
  point is where the first combo hit lands.
- **Remove a song**: ✕ on the song, or ✕ / right-click on the Music card at
  the top (with several songs it asks which one, or all). Any song can go,
  the only one too; drop it again to bring it back.

Nothing is analysed again: each song keeps its beat, start, cut and beat fixes
when it moves.

### Letterbox and flash

The cinematic bars show during the slow-mo intro (and slow-mo lead-ins). A
quick flash marks the first hit, and the combos play full frame. Tick
**🎬 Focus bars** on any combo to keep bars on through it. Effects → Focus
bars switches the automatic bars to "Intro + 1st combo", "Every combo" or off.

### Timeline and Advanced

**Timeline ▾** (bottom right) shows the detailed edit:

- **MUSIC**: beats and bar lines.
- **EDIT**: speeds, cuts, bars, captions and effects.
- **VIDEO**: hits and combos.

What you can do on it:

- Double-click to add a beat or hit; right-click deletes one.
- **S** splits a combo, **M** merges it with the one before.
- The mouse wheel or the scrollbar scrolls; **Ctrl/⌘ + wheel** zooms.

On a small screen the whole window scrolls instead of cutting off, and the
mouse wheel over a slider scrolls the panel (click a slider first to change
it with the wheel).

**⚙ Advanced** has every other setting, all with good defaults:

- intro length and speed, lead-in length
- beat grid overrides
- combo rules and sync tolerance
- hit detection sensitivity
- hit sounds (your recording's own, Minecraft's from your install, or a
  custom file)
- letterbox
- export quality and encoder

| Key | Action |
|---|---|
| Space | Play / pause |
| F | Fullscreen (Esc to leave) |
| ← → | One beat back / forward |
| M | Mark (Song / Gameplay tab) |
| T | Tap tempo (Song tab) |
| C | Cut: the next song takes over here / the music ends here (Song tab) |
| Ctrl/⌘ + S / O / E | Save project / open project / export |

## Command line

```
python -m hitsync                                  # the app (drop files, or pass them)
python -m hitsync beats song.mp3 --click check.wav # what was detected + a click track
python -m hitsync run pvp.mp4 song1.mp3 song2.mp3 -o montage.mp4 --style hype \
    --beats-per-hit auto --text "0:02-0:05=GG EZ@top" --effect "flashy=0:10-0:18" \
    --focus 1,3 --fade-in black --fade-out white [--drop 20.4] [--dry-run] \
    [--save-project p.json]
python -m hitsync check                            # dependencies + ffmpeg
python -m unittest discover -s tests -t .          # tests
python tools/bench.py                              # speed benchmark
```

`--text` and `--effect` times count from the start of the montage. `--focus`
puts focus bars on those combos (numbered in time order).

## How it works

### Beat grid (`music_grid.py`)

Montage songs are made on a DAW grid with one exact tempo. So instead of
following a beat tracker (the 3-3-2 marimba riff of *Shape of You* reads as
128 BPM, or wanders off to 97), Hit-Sync searches the tempo and phase
directly:

- **Tempo**: every candidate tempo folds the song's drum attacks onto a single
  beat. The true tempo is the one where they stack up over the whole song; a
  wrong one smears once the pattern repeats.
- **Right multiple**: a syncopated riff (the 3-3-2 of *Shape of You*,
  reggaeton, phonk cowbells, trap triplets) also folds well at 1.5×, 4/3 or 2×
  the real tempo. So the related tempos of the best candidates are always
  weighed too, and each is scored on:
  - how well the attacks fold;
  - a tempo prior;
  - **bar repetition**: songs repeat every bar and every two bars, so the true
    tempo's 4 and 8 beats line up with the song's own repetition, and a 1.5×
    tempo's don't;
  - a subdivision test: a tempo whose every other beat only has hats is the
    8th-note level. Hit strength is compared by rank, so a clap counts like
    an 808, and a song without a backbeat still keeps its beat.
- **Phase**: the phase comes from the loud kick and snare accents, so off-beat
  hats can't flip it. It is then fitted to the attacks, which are measured on
  the waveform to about ±1 ms. A riff that hits both the beats and the "and"s
  is settled by where the louder attacks and the chord changes are.
- **Downbeats**: the bar lines come from the kick plus chord changes.
- **Live songs**: a song whose tempo wanders (a live band) is detected by the
  local tempo moving more than 2% through the song, and the beat tracker
  follows it instead.

On 48 synthetic songs of 8 styles at random tempos, 47 read the exact tempo
and one read double (still every beat). Before this, 23 of them read a wrong
tempo, such as a 3-3-2 song at 89.3 BPM read as 134.

### Music start (`sections.py`)

Per bar, it measures kick activity, sub-bass loudness, snare/hat activity and
overall loudness.

- **Beat kicks in**: the first bar where the kick holds after a quieter
  stretch.
- **Drop**: the bar line where kick and bass rise and *stay* up. A snare-roll
  build scores lower than the drop it leads into.
- **Default**: the earliest strong moment (with at least 4 bars of song after
  it).
- **Slow-mo intro**: the whole song intro when it is short, otherwise 4 bars
  (2 for slow songs). It never starts in silence.

### Edit (`sync_engine.py`)

The output timeline is the music timeline. The edit is a list of time-remap
segments (speed ramps, cuts, slow-mo) built so that every hit lands exactly
on its beat step:

- The combo plan (order, on/off, lead-ins), beats per hit and velocity curves
  are applied here.
- The slow-mo intro ramps into the first hit on the music start.
- Reordered combos are reached by cuts.

### Look (`effects.py`, `text_overlay.py`)

Everything is evaluated per output frame from the edit (hits, beats, cuts),
so the preview and the export are identical.

- Filters are a colour lookup table plus a colour matrix.
- Motion blur blends recent source frames, never across a cut.
- Captions are rendered once as sprites.
- Invisible effects cost nothing.

### Preview (`preview_engine.py`)

- A background thread builds frames ahead of the playhead.
- Before a cut that jumps back in the footage, a second decoder starts in
  advance.
- The sound card is the clock, so picture and sound never drift.
- Frames are built at most 1280 px wide and the screen scales them, so
  fullscreen stays smooth.

### Hits (`video_analysis.py`)

The onset of the red hurt tint in the centre of the screen, plus camera
snaps, from a 320 px scan decoded by ffmpeg. With two chunks, one decodes on
the GPU (when there is one) while the other uses the CPU.

## Speed

Measured on a 4-core container, with a synthetic 60 s 1440p30 capture and a
3-minute song (`tools/bench.py`):

| Step | Before | Now |
|---|---|---|
| Music analysis (3 min song) | 5.6 s (10.7 s first run) | ~1 s |
| Video scan (60 s 1440p) | 6.5 s | 5.8 s (+ GPU decode on your machine) |
| Export, 79 s montage at 1440p, no effects | 136 s | 66 s |
| Export, same montage, Montage look | – | ~100 s |
| Preview frame building at 960×540, Montage look | – | ~80 fps (2.7× real time) |

On a normal PC export is faster: more cores, and the GPU encoder (NVIDIA, AMD,
Intel, or Apple VideoToolbox) is used automatically.

Analysed files are cached per file (`%LOCALAPPDATA%\HitSync` on Windows,
`~/Library/Caches/HitSync` on macOS, `~/.cache/HitSync` on Linux). Opening
them again is instant.

## Files

```
hitsync/
  music_grid.py      beat grid: tempo/phase search, attacks, downbeats, confidence
  sections.py        music start candidates, song end, automatic intro length
  audio_analysis.py  AudioAnalysis (grid + sections + curves), beat tracker fallback
  video_analysis.py  ffmpeg scan: hurt tint + screen motion -> hits
  sync_engine.py     the edit: plan, intro, lead-ins, steps on the beat, letterbox
  soundtrack.py      several songs as one music timeline (crossfades, grid)
  audio_mix.py       soundtrack: music + hit sounds (recording / Minecraft / custom)
  effects.py         filters, motion blur, hit effects, ranges, transitions, fades
  text_overlay.py    captions (fonts in hitsync/assets/fonts, SIL OFL)
  styles.py          Clean / Montage / Hype / Cinematic
  renderer.py        frame-exact decoding, frame composition, parallel export
  preview_engine.py  real-time playback (decode ahead, audio clock)
  project.py         project state, analysis orchestration, save/load
  cli.py             python -m hitsync [run|beats|check]
  ui/                the Qt app (main window, panels, timeline, advanced settings)
tests/               engine, beat grid, plan, effects, multi-song, preview, UI tests
tools/bench.py       speed benchmark
```

Fonts: Anton, Montserrat Black and Fredoka Bold, under the SIL Open Font
License (see `hitsync/assets/fonts/OFL.txt`).
