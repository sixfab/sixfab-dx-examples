#!/usr/bin/env bash
# Pose Estimation (Pilates Coach, Fall Detection, pose game) - re-runnable install.
# Run it ON the Raspberry Pi 5 after the sixfab-dx package is installed. These are custom pipelines (no Frigate):
# Python apps that talk to the DX-M1 NPU through dx_engine and open cv2.imshow windows
# on the Pi's own desktop (labwc/Xwayland, DISPLAY=:0); they are not headless services.
# Usage: ./install.sh [--yes] [--dry-run]
# Optional env:
#   DEPLOY_DIR  where to install the apps   (default: ~/pose-estimation)
#   VENV_DIR    Python virtual environment  (default: the sixfab-dx venv, /opt/sixfab-dx/venv;
#               point it at your own venv that already has dx_engine, opencv and numpy)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/../_shared/common.sh"

parse_flags "$@"
DEPLOY_DIR="${DEPLOY_DIR:-$HOME/pose-estimation}"
VENV_DIR="${VENV_DIR:-$SIXFAB_VENV}"

echo "== 1/3: model and files (release $(models_tag "$HERE/models.txt")) =="
fetch_assets "$HERE" 0
run mkdir -p "$DEPLOY_DIR/model"
run cp "$HERE/model/yolov8n-pose.dxnn" "$DEPLOY_DIR/model/yolov8n-pose.dxnn"
for f in pose_engine.py fitness_coach.py fall_detection.py fruit_ninja_pose.py benchmark.py; do
    run cp "$HERE/$f" "$DEPLOY_DIR/$f"
done

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
    "$VENV_DIR/bin/python" -c 'import dx_engine, cv2, numpy; print("dx_engine + cv2 + numpy OK, cv2", cv2.__version__)'
    dxrt-cli -s | head -3 || true
fi

cat <<EOT

Install complete.

Open a terminal on the Pi's desktop (not SSH) and run one of the apps. The argument is a
video file or a camera index (0 for the first USB camera):

  cd $DEPLOY_DIR
  DISPLAY=:0 $VENV_DIR/bin/python fitness_coach.py <video-file-or-camera-index>
  DISPLAY=:0 $VENV_DIR/bin/python fall_detection.py <video-file-or-camera-index>
  DISPLAY=:0 $VENV_DIR/bin/python fruit_ninja_pose.py <video-file-or-camera-index>

Keys in every window: q = quit, r = reset.
EOT
