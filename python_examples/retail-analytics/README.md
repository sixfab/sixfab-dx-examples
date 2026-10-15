# Retail Analytics

Custom pipeline (not Frigate): `yolox-s_640x640_ppu.dxnn` (DeepX Model Zoo, pretrained) for person detection and tracking, an entry/exit counter, and a live and total heatmap, shown in a window on the Pi's own screen.

## What is in this folder

- `watch_retail_local.py` - the main pipeline.
- `models.txt` - release tag, asset name and SHA-256 of the model.
- `install.sh` - downloads the files from the release and sets up a Python environment with `dx_engine`, OpenCV and numpy.
- `run_on_pi.sh` - starts the pipeline on the Pi's own desktop (`DISPLAY=:0`).

## Install

Install the `sixfab-dx` package first (see the repo root README); it provides the shared Python environment with `dx_engine`. Then:

```bash
cd python_examples/retail-analytics
./install.sh              # downloads the model, copies the script to ~/retail-analytics
```

Then, in a terminal on the Pi's own desktop (not over SSH):

```bash
bash ~/retail-analytics/run_on_pi.sh                       # defaults to the first USB camera (index 0)
bash ~/retail-analytics/run_on_pi.sh <video-file-or-rtsp-url>
```

Keys: `q` quit, `h` toggle the live heatmap, `r` reset the counters. `total_heatmap_final.png` is saved on exit. RTSP URL with a password: it ends up in your shell history, so use a camera account with a viewing-only role.

## Troubleshooting

- `dx_engine` import error: run `./install.sh` again (it rebuilds the Python environment).
- No window opens: check that a desktop session is active (`loginctl list-sessions`).
- `QFontDatabase` warning: harmless.
