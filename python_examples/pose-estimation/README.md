# Pose Estimation Apps

Three interactive, real-time apps built on one shared pose-estimation
engine (`yolov8n-pose.dxnn` running on the DeepX NPU). Custom pipeline -
no Frigate involved, reads directly from a USB camera (lowest possible
latency) or from a video file.

**Status:** complete. On 2026-10-06 the apps ran on a Raspberry Pi 5 against the
author's own test clips (not included in this repo), and the pose engine was
benchmarked there (see "Performance on the Raspberry Pi 5" below). A USB camera
on the Pi 5 was not tested. `install.sh` installs the apps on the Pi.

| App | What it does | Run it |
|---|---|---|
| 🧘 **Pilates Coach** | Counts arm-press reps (elbow angle), gives form feedback, per person | `python fitness_coach.py` |
| 🚨 **Fall Detection** | Flags a likely fall (torso above 70° within 2 s of a fast hip drop), per person | `python fall_detection.py` |
| 🍉 **Fruit Ninja-style pose game** | Slice falling fruit with your hands, score points, multiplayer | `python fruit_ninja_pose.py` |

## Requirements

- A Sixfab AI HAT+ (DX-M1M) or Sixfab Edge AI Expansion Board (DX-M1), with the `dx_engine` Python package installed and working -
  confirmed via `dxrt-cli -s`.
- A USB webcam or a video file.
- `opencv-python`, `numpy` (already used throughout this repo).
- `model/yolov8n-pose.dxnn`. `install.sh` downloads it from the GitHub release
  (`models.txt` lists the asset name and SHA-256).
- On the Pi: a desktop session (the apps open `cv2.imshow` windows).

## Quick start

```bash
cd python_examples/pose-estimation

# Live, from your webcam (camera index 0 by default):
python fitness_coach.py
python fall_detection.py
python fruit_ninja_pose.py

# Or point any of them at a video file instead of a camera (bring your own clip):
python fitness_coach.py <path-to-video.mp4>
python fall_detection.py <path-to-video.mp4>
python fruit_ninja_pose.py <path-to-video.mp4>
```

### Install on a Raspberry Pi 5

```bash
./install.sh                 # downloads the model, copies the apps to ~/pose-estimation, checks dx_engine/cv2/numpy
VENV_DIR=~/my-venv ./install.sh     # use your own venv instead of the shared sixfab-dx one
```

Then, on the Pi's desktop (not over SSH), run any app with your own video file or a camera index:
`DISPLAY=:0 /opt/sixfab-dx/venv/bin/python fall_detection.py <path-to-video.mp4>`.
The windows open at video size and can overflow the screen; drag them by the
title bar.

If you have more than one camera, pass its index instead of a path:
`python fitness_coach.py 1`.

All three windows have the same basic controls:

| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset (rep counters / alerts / restart the game after game over) |

Only one app can use the camera at a time - close one (`q`) before
starting another.

## How each app works

### 🧘 Pilates Coach (`fitness_coach.py`)

Tracks the **elbow angle** (shoulder → elbow → wrist), averaged across
both arms when visible. This is deliberately NOT a squat/knee-angle
counter - most Pilates work (ring press, chest press, arm circles) is
driven by controlled arm extension/flexion rather than leg movement, so a
squat counter never fires on this kind of exercise.

State machine per person: **EXTENDED → BENT → EXTENDED** counts one rep.
The minimum angle reached during the BENT phase drives the feedback
("Nice control!" vs. "Pull in further").

**Multi-person:** every detected person gets their own tracked rep
counter, shown above their own bounding box - built and tested against a
5-person Pilates reformer class video, correctly counted 3 people
independently and kept their identities stable across the clip.

### 🚨 Fall Detection (`fall_detection.py`)

Combines two signals per person, so sitting or crouching down on purpose
does **not** trigger a false alarm:

1. **Torso angle from vertical** (shoulder-midpoint → hip-midpoint line).
   Upright ≈ 0°, lying flat ≈ 90°.
2. **Vertical fall speed** - how fast that person's hips dropped in the
   last ~0.6 seconds. A real fall is fast; sitting down slowly is not.

A fall is flagged when the torso goes above 70° from vertical **within 2
seconds of a fast drop** (more than 0.55 frame heights per second, the old
400 px/s at 720p). The drop is remembered because a person often drops fast
while the torso is still upright and only turns horizontal afterwards, so
requiring both at the same moment never fired on the sample clip. Each
person raises one alert per fall and must stand upright again (below 40°)
before another. A red "FALL DETECTED" banner stays up for 5 seconds (or
until `r` clears it). For video files, time comes from the video clock
(frame number / fps), not the wall clock. The constants are at the top of
`fall_detection.py`; they were set on one sample clip and are not validated.

**Multi-person:** each tracked person has independent history and alert
state - built to watch a room with several people, not just one.

### 🍉 Fruit Ninja-style pose game (`fruit_ninja_pose.py`)

Both wrists act as blades. Each hand always shows a solid colored circle
at its current position (so you can see where your "blade" is even when
your hand is still), plus a glowing motion trail when swiping. Fruit
launches from the bottom of the screen and arcs up and down under gravity,
exactly like the original Fruit Ninja - swipe through it to slice it
(score +1). Miss 3 fruits (let them fall back down uncut) and it's game
over; press `r` to play again.

**Multiplayer:** every person detected in frame gets their own pair of
blades (color-coded per person), so more than one player can slice at
once.

## Shared engine (`pose_engine.py`)

All three apps are built on this one module:

- `PoseEngine` - loads `yolov8n-pose.dxnn`, runs inference, decodes the
  raw (56, 8400) output (box + confidence + 17 COCO keypoints per anchor)
  with NMS in Python, then drops weak duplicates (confidence below 0.6 and
  at least 70% inside a stronger pose's box). Returns a list of `Pose`
  objects (one per detected person), each with a bounding box, confidence,
  and 17 keypoints.
- `update_pose_tracks()` - a nearest-centroid multi-person tracker that
  matches all (track, pose) pairs globally by distance, ages every
  unmatched track and drops tracks after 40 frames, with a free-form
  `.data` dict each app uses to store its own per-person state (rep
  counter, fall-alert history, blade trails, etc).
- `open_source()` / `read_looping()` - opens either a live USB camera or a
  video file (looping) behind the same interface, so every app can take
  either without special-casing.
- `angle()`, `draw_text()`, `FpsCounter` - small shared helpers.

Known limit: when two people overlap heavily, YOLOv8n-pose can return one
merged skeleton for a few frames (about 0.3 s in the sample clip). The
tracker keeps ids stable across it, but the merged skeleton's joints belong
to two people.

## Performance on the Raspberry Pi 5

Measured on 2026-10-06 with `benchmark.py`, against the author's own test clips
(not included in this repo): each cell is the range over three runs.
`infer()` is `PoseEngine.infer()` (preprocessing, the NPU run, decoding and NMS).
The full loop adds reading the frame and updating the tracker.

| Clip | Resolution, source rate | Poses per frame (mean, max) | `infer()` mean | `infer()` p95 | Read + infer + tracking, mean | Full loop |
|---|---|---|---|---|---|---|
| Pilates reformer class, 5 people | 1920x1080, 25 fps | 3.90, 5 | 27.3 to 27.4 ms | 28.1 to 28.3 ms | 31.0 to 31.2 ms | 32.1 to 32.3 FPS |
| Fall clip, 1-2 people | 898x506, 50 fps | 1.00, 2 | 25.4 to 26.5 ms | 26.9 to 27.0 ms | 26.6 to 27.8 ms | 36.0 to 37.6 FPS |

- **Environment:** Raspberry Pi 5 Model B Rev 1.1 (4 cores), Debian GNU/Linux 13
  (trixie), kernel 6.18.50+rpt-rpi-2712, Python 3.13.5. DEEPX DX-M1M, DXRT
  v3.4.2+d803450, RT driver v2.6.0, PCIe driver v2.5.0, firmware v2.7.4. The Frigate
  container was running with all cameras off, the load average was about 0.0, and
  nothing else used the NPU.
- **Method:** 30 warm-up frames, then 300 timed frames per clip (the clip loops),
  `time.perf_counter`, headless: no window, no drawing and no camera capture.
  Three runs per clip.
- **Not measured:** the NPU time alone (`infer()` includes preprocessing and NMS on the
  CPU), a live USB camera on the Pi 5, power, temperature, more than three runs, runs
  with Frigate cameras on. Accuracy of rep counts and fall alerts is not measured.

Run it yourself, from this folder, with nothing else using the NPU (bring your own clips):

```sh
python benchmark.py <path-to-video1.mp4> <path-to-video2.mp4>
```

## Other files in this folder

- `pose_engine.py` - the shared pose engine (model decoding, NMS, tracking) used by the three apps.
- `benchmark.py` - headless benchmark of the pose engine on video files (see above).
- `install.sh`, `models.txt` - Raspberry Pi installer and the release asset list (model only).
