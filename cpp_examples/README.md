# DeepX C++ Examples

Standalone, idiomatic C++ demos for the DeepX NPU — mirroring the Python
examples but built with only OpenCV and the DeepX SDK (`dxrt`). No Qt. No
YAML. Each demo is a single-file executable you can read end-to-end.

## Prerequisites

- Raspberry Pi 5 (or any Linux host with a DeepX-compatible NPU)
- The `sixfab-dx` package installed from the Sixfab APT repository. It pulls in
  the DeepX runtime (`libdxrt-bin`), which provides `libdxrt.so` and the headers
  under `/usr/local/include/dxrt/`
- OpenCV 4.x development headers
- A C++17 compiler, CMake >= 3.14, `pkg-config`

## Quick start

```bash
git clone https://github.com/sixfab/sixfab-dx-examples.git
cd sixfab-dx-examples
./auto-install.sh
```

`auto-install.sh` downloads the sample models, videos and images into
`resources/` at the repository root, then runs `cpp_examples/setup.sh`.
`setup.sh` installs the system packages, configures CMake, and builds every
demo in `cpp_examples/build/`. It does not download models or videos.

Then open the demo menu, or run a demo directly from `cpp_examples/`:

```bash
bash cpp_examples/start.sh

cd cpp_examples
./build/object_detection/yolov8_demo
```

## Demos

| Binary                                    | Description                                    |
| ----------------------------------------- | ---------------------------------------------- |
| `object_detection/yolov8_demo`            | YOLOv8 object detection (PPU, anchor-free)     |
| `object_detection/yolov5_demo`            | YOLOv5 object detection (PPU, anchor-based)    |

`start.sh` lists every demo in the repository, including pose, face, PPU,
async, segmentation, classification and advanced (tracking / trespassing /
smart-traffic) demos.

## Running with a custom source

Each demo reads `configs/<name>.json` for its defaults and then applies CLI
overrides. Supported flags:

```
--model <path>          override model_path
--source <type>         video | camera | libcamera | rtsp | image
--path <path>           override source_path (video / image / RTSP URL)
--labels <path>         one label per line
--conf <float>          score threshold (default 0.3)
--iou  <float>          NMS IoU threshold (default 0.45)
--camera-index <n>      USB camera index for --source camera
--window <title>        OpenCV window title
--no-fps                hide the FPS overlay
```

Examples:

```bash
./build/object_detection/yolov8_demo --source camera --camera-index 0
./build/object_detection/yolov8_demo --source libcamera --path 0:1536:864:30
./build/object_detection/yolov8_demo --source video --path my_clip.mp4 --conf 0.4
```

### Raspberry Pi camera (rpicam)

Verify the camera is detected before use:
```bash
rpicam-hello --list-cameras
```

Use `--source libcamera` with `--path` in `idx:W:H:fps` format (all fields optional):

```bash
# Camera index 0, 1280×720, 30 fps
./build/object_detection/yolov8_demo --source libcamera --path 0:1280:720:30

# Use defaults (1536×864, 30 fps)
./build/object_detection/yolov8_demo --source libcamera --path 0
```

To make rpicam the default for a demo, edit its `configs/<demo>.json`:
```json
{
  "source": "libcamera",
  "source_path": "0:1280:720:30"
}
```

> libcamera support is enabled by default when CMake finds it via `pkg-config`.
> If you get `libcamera support not compiled in`, rebuild with:
> ```bash
> cmake .. -DENABLE_LIBCAMERA=ON && make -j$(nproc)
> ```

## Adding a new demo

1. Put `<demo_name>.cpp` in the matching category directory
   (`object_detection/`, `pose_estimation/`, …).
2. In that directory's `CMakeLists.txt`, add:
   ```cmake
   add_demo(<demo_name> <demo_name>.cpp)
   ```
3. Optionally ship `configs/<demo_name>.json` with sensible defaults.
4. Re-run `cmake .. && make` in `build/`.

Each demo follows the same three-part skeleton — a `Model` class with a
`.infer()` method, a `draw()` lambda, and a `main()` that wires an
`InputSource` into `runDemo()`. See any existing demo file for the pattern.

## Directory layout

```
cpp_examples/
  CMakeLists.txt       -- top-level build: common lib + demo subdirs
  setup.sh             -- installs build dependencies and builds every demo
  start.sh             -- interactive demo menu
  configs/             -- per-demo JSON config files
  common/              -- shared library (sdk, ppu, input, vis, runner, ...)
  object_detection/    -- YOLO demos
../resources/          -- models, videos and images, downloaded by auto-install.sh
```
