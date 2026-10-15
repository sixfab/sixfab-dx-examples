#!/usr/bin/env bash
# Retail Analytics - re-runnable install. Run it ON the Raspberry Pi 5 after the sixfab-dx package is installed.
# Custom pipeline (not Frigate): a Python script that talks to the DX-M1 NPU through
# dx_engine and opens a cv2.imshow window on the Pi's own desktop (DISPLAY=:0), so it is
# started by hand from the desktop, not as a service.
# Usage: ./install.sh [--rtsp-url URL] [--yes] [--dry-run]
# Optional env: DEPLOY_DIR (default ~/retail-analytics), VENV_DIR (default: the sixfab-dx venv)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../_shared/common.sh"

parse_flags "$@"
DEPLOY_DIR="${DEPLOY_DIR:-$HOME/retail-analytics}"
VENV_DIR="${VENV_DIR:-$SIXFAB_VENV}"

echo "== 1/3: model and files (release $(models_tag "$HERE/models.txt")) =="
fetch_assets "$HERE" 0
run mkdir -p "$DEPLOY_DIR/model"
run cp "$HERE/model/yolox-s_640x640_ppu.dxnn" "$DEPLOY_DIR/model/yolox-s_640x640_ppu.dxnn"
run cp "$HERE/watch_retail_local.py" "$DEPLOY_DIR/watch_retail_local.py"
run cp "$HERE/run_on_pi.sh" "$DEPLOY_DIR/run_on_pi.sh"
run chmod +x "$DEPLOY_DIR/run_on_pi.sh"

echo "== 2/3: Python environment (shared sixfab-dx venv) =="
if [ "${DRY_RUN:-0}" = "1" ] || [ "${SIXFAB_TEST:-0}" = "1" ]; then
    say "   skipped (dry run / test)"
else
    SIXFAB_VENV="$VENV_DIR" require_sixfab_venv
    "$VENV_DIR/bin/python" -c 'import cv2, numpy' 2>/dev/null || "$VENV_DIR/bin/python" -m pip install --quiet opencv-python numpy
fi

echo "== 3/3: verification =="
if [ "${DRY_RUN:-0}" = "1" ] || [ "${SIXFAB_TEST:-0}" = "1" ]; then
    say "   skipped (dry run / test)"
else
    "$VENV_DIR/bin/python" -c 'import dx_engine, cv2; print("dx_engine + cv2 OK, cv2", cv2.__version__)'
    dxrt-cli -s | head -3 || true
fi

RUN_INPUT="${RTSP_URL:-<video-file-or-rtsp-url>}"
cat <<EOT

Install complete.

On the Pi's own desktop (not SSH), open a terminal and run:
  VENV_DIR=$VENV_DIR bash $DEPLOY_DIR/run_on_pi.sh "$RUN_INPUT"
EOT
