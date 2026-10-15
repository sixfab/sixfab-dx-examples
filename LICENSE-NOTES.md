# License notes — sixfab-dx-examples

Per-model license notes for the 2 standalone apps under `python_examples/` in
this repo, copied verbatim from the release staging notes (no wording
changed). See `DEEPX-SDK-LICENSE.txt` in this same folder for the DEEPX SDK
terms referenced in each section below.

---

# License notes — retail-analytics-v1.0.0

## Model file
- `yolox-s_640x640_ppu.dxnn`
- SHA-256: `ea2f3d402c510fd898095522a1e51160883bdb46e78ae55fce0b7b9ea8ac712d`
- Matches `python_examples/retail-analytics/models.txt` in this repo.

## What this is
Ready-compiled, pretrained YOLOX-S object detector (COCO-80 classes),
sourced directly from the **DeepX Model Zoo**, PPU-compiled for the DeepX
NPU. No custom training — person detection (COCO class 0) is used as-is for
foot-traffic counting.

## DEEPX compiler (dxcom)
Compiled with DEEPX's `dxcom` toolchain (in addition to being sourced from
the DeepX Model Zoo). See `DEEPX-SDK-LICENSE.txt` in the parent folder for
DEEPX's license terms.

## License chain
- **Base architecture:** YOLOX (Megvii) → **Apache License 2.0**. DeepX's own
  Model Zoo listing (developer.deepx.ai) lists the YOLOX family
  (YOLOXTiny/S/SLeaky/SWideLeaky/LLeaky) under Apache-2.0, matching upstream
  YOLOX's license.
- **Training data:** COCO 2017 (https://cocodataset.org) — images CC BY 4.0,
  annotations licensed for research and commercial use.
- **Caution:** DeepX's model download page carries a generic warning that
  commercial distribution of a product using their models "may obligate
  disclosure of source code" and disclaims responsibility for license
  disputes. This reads as boilerplate aimed at their AGPL-licensed models
  (e.g. Ultralytics-derived ones) rather than Apache-2.0 YOLOX specifically,
  but it has not been reconciled against a DeepX-AI repo LICENSE/NOTICE
  file. Worth a quick confirmation with DeepX/Sixfab before public release
  if that matters for this project.

## Demo video — not published
`retail-test-street.mp4` is not distributed with this app and stays off
GitHub entirely.

## Attribution text (suggested)
> Object detector based on YOLOX (Apache-2.0, Megvii), via the DeepX Model
> Zoo, trained on the COCO dataset (cocodataset.org).

---

# License notes — pose-estimation-v1.0.0

## Model file
- `yolov8n-pose.dxnn`
- SHA-256: `a5b26c6c20fbbcb3825040b679ee4d5e154c92d087de0c14e6ce479af292814d`
- Matches `python_examples/pose-estimation/models.txt` in this repo.

## What this is
Stock Ultralytics YOLOv8n-pose model (17-point COCO skeleton keypoints),
PPU-compiled for the DeepX NPU. No custom training.

## DEEPX compiler (dxcom)
Compiled with DEEPX's `dxcom` toolchain. See `DEEPX-SDK-LICENSE.txt` in the
parent folder for DEEPX's license terms.

## License chain
- **Base weights/architecture:** Ultralytics YOLOv8-pose → **AGPL-3.0**
  (https://github.com/ultralytics/ultralytics, `LICENSE`). Copyleft:
  distributing this model as part of a product obligates making the
  corresponding source available under AGPL-3.0, unless Sixfab/the publisher
  holds an Ultralytics Enterprise license that supersedes this.
- **Training data:** COCO 2017 keypoints (https://cocodataset.org) — images
  CC BY 4.0, annotations licensed for research and commercial use. No
  additional restriction.

## Demo videos — not published
`pilates.mp4` and `fall.mp4` are the owner's own test clips and are not
distributed with this app.

## Attribution text (suggested)
> Pose estimation based on Ultralytics YOLOv8-pose (AGPL-3.0), trained on
> the COCO keypoints dataset (cocodataset.org).
