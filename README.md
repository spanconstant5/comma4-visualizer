# Comma 4 Drive Visualizer

A standalone GUI program that replays openpilot routes from your comma 4 in a
ui-alt / replay-clipper style view — real camera video, real telemetry, and a
radar top-down, all driven by the actual logs.

## Build the EXE (one time, on Windows)

1. Make sure Python 3.10+ is installed (https://python.org — tick
   **"Add Python to PATH"** during setup).
2. Double-click **`build.bat`**.
3. When it finishes, your program is at **`dist\Comma4Visualizer.exe`**.

That's the only time you touch anything build-related. Move
`Comma4Visualizer.exe` wherever you like and double-click it to run.

## Use it (no command line)

1. Double-click **`Comma4Visualizer.exe`**.
2. Click **Load Route Folder…** and pick a route you downloaded from your
   comma 4, or click **Demo Mode** to see it with sample data.

### Controls — use the menu bar at the top

- **File** — Open Route, Quit
- **Playback** — Play/Pause, Restart, Set In/Out points, Speed
- **View** — Location Panel (privacy toggle)
- **Export** — Export Video (full options dialog)
- **Help** — Controls & info

Drag the timeline at the bottom to scrub anywhere. Keyboard shortcuts still
work (Space, ← →, I/O, E, M, R, Q) but everything is in the menus.

### Alerts / take-over

When openpilot showed an alert during the drive — *"TAKE CONTROL
IMMEDIATELY"*, a disengage prompt, etc. — it's drawn as a banner over the road
camera, colored by severity (red = critical), straight from the log's
`selfdriveState`. So you can see exactly when and why it asked you to take over.

### What it shows from your logs

- **Camera + model**: real road/wide video with the model's **path, lane lines
  and road edges** projected on top using the log's real camera calibration
  (`extrinsicsCalibration`) and the comma road/wide intrinsics — the same math
  openpilot's own UI uses, so the path lines up with the road.
- **DRIVER** (sunnypilot MADS-aware):
  - `OPENPILOT` (green) — fully engaged (lateral + longitudinal)
  - `OP LATERAL` — MADS steering only, you're on the gas/brake
  - `OP LONG` — longitudinal only, you're steering
  - `OVERRIDE` — you're overriding
  - `HUMAN` — disengaged, you're driving
- **MAX**: the set cruise speed (`vCruiseCluster`), in mph.
- **model**: the active driving model's name (sunnypilot `modelManagerSP`).

### Location / privacy

The **LOCATION** panel (bottom-right) shows where the clip was recorded on an
**OpenStreetMap** map (tiles downloaded on the fly and cached), with the drive's
GPS trace and your current position. Offline, it falls back to a trace-only
view. Press **`M`** (or View menu) to hide it, and turn it off in the export
dialog, so a shared clip doesn't reveal where you drove.

### What's on the camera vs. the dashboard

To match the comma 4 screen, the camera view stays clean: path, lane lines,
road edges, turn-signal arrows, the engaged dot, the driver/steering icons,
the MAX (set) speed badge, and take-over alerts — but **not** a constant speed
readout. Current **speed is shown in the TELEMETRY panel** instead.

### Performance

Video decodes on **background threads** (GPU-accelerated) so playback doesn't
stutter, with a RAM frame cache for instant scrubbing. Export has a
**Performance** setting that cycles:

- **Balanced** — one render core, higher quality.
- **Max** — GPU encode + fast preset, decode/encode overlapped.
- **Max — ALL CORES** — renders the drive in parallel across every CPU core
  (separate worker processes) then stitches the parts. Fastest for long
  exports; your PC will be very busy. Falls back to single-core if a worker
  fails.

GPU use is **vendor-agnostic**: HEVC decode uses whatever hardware path the
system exposes (NVDEC / AMF / Quick Sync / D3D11VA / VideoToolbox) and falls
back to software automatically; the export encoder is chosen by a quick
**test-encode probe** that picks the first of NVENC / AMF / Quick Sync /
VideoToolbox that actually runs on your machine, else CPU x264 — so it works on
AMD, NVIDIA, Intel or Apple without configuration.

Note: the overlay drawing itself is CPU (pygame/SDL) — the GPU does the HEVC
decode and encode. "All cores" is what actually saturates the CPU. (ROCm/GPU
compute doesn't apply to 2D UI drawing.)

### Export a high-res clip

Press **`E`** during route playback to open the export dialog. Options:

- **Content**: *Dashboard* (the full view) or *Camera only (sim recording)* —
  just the camera with the openpilot overlays, like comma's own clip but at
  much higher resolution.
- **Camera**: Road (narrow) or Wide, for camera-only.
- **Range**: the whole drive, or a selection — press **`I`** and **`O`** during
  playback to set in/out points (shown on the timeline).
- **Encoder**: **GPU** (auto-detected — AMD AMF on your 9060 XT, or NVENC/QSV)
  or CPU. GPU is much faster.
- **Location overlay**: on/off (privacy).
- **Save to**: pick any folder (defaults to Desktop).

Dashboard exports at 2× (~2880×1872); camera-only at 1920-wide. A live
progress bar shows frame count, encoder, elapsed time and ETA. `Esc` cancels.
Uses a bundled ffmpeg — no extra install.

## Getting a route off your comma 4 (Cyberduck)

1. Connect to the device over **SFTP** in Cyberduck (the IP is on the comma
   screen). Go to `/data/media/0/realdata/`.
2. Each drive is split into numbered **segments** that share a timestamp
   prefix, e.g. `2026-11-20--17-31-42--0`, `...--1`, `...--2`, …
3. Download **all segments of one drive** into a single folder. You can leave
   the names as-is or rename them to `0`, `1`, `2`, …:

   ```
   my_drive/
     0/   (or 2026-11-20--17-31-42--0/)
       rlog.bz2
       fcamera.hevc     <- road camera
       ecamera.hevc     <- wide camera
     1/
       rlog.bz2
       fcamera.hevc
       ecamera.hevc
     ...
   ```
4. In the program, click **Load Route Folder…** and select `my_drive`.

The segments are stitched into one continuous drive automatically.

## Performance / GPU

- HEVC camera decode uses **hardware acceleration** when your GPU/driver
  exposes it (on Windows, D3D11VA/DXVA2 — works with your Radeon RX 9060 XT),
  and falls back to CPU automatically. On a Ryzen 7700X, software decode of the
  20 fps cameras is already trivial, so GPU mainly helps fast scrubbing.
- If a driver misbehaves, disable GPU decode: run from a terminal with
  `Comma4Visualizer.exe --no-gpu`, or set the env var `COMMA4_NO_HWACCEL=1`.
- A progress bar shows while logs are parsed and video is opened.

## Schema / fork

The capnp schemas in `schemas/` are taken from the **sunnypilot `sun`
branch** (spanconstant5/openpilot + spanconstant5/opendbc), so logs recorded on
your fork — including sunnypilot-specific messages — parse correctly. If you
rebase your fork and the log format changes, replace the files in `schemas/`
with your branch's `cereal/{log,custom,deprecated}.capnp`,
`cereal/include/c++.capnp` and `opendbc/car/car.capnp`, then rebuild.

## What's real vs. demo

- **Route mode** (Load Route Folder): camera panels show the actual
  `fcamera.hevc` / `ecamera.hevc` video from the device, and all telemetry
  (speed, steering, gas/brake, radar leads) comes straight from `rlog.bz2`.
- **Demo mode**: synthetic sample data, for when you just want to see the
  layout without downloading anything.

## Notes

- `rlog.bz2` is required for telemetry. If a segment is missing a camera file,
  that panel shows a "no video" placeholder but telemetry still plays.
- The `schemas/` folder next to the scripts holds the openpilot log schema and
  is bundled into the EXE automatically — keep it next to `comma4_viz.py` when
  building.
