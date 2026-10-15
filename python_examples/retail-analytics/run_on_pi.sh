#!/usr/bin/env bash
# Opens the retail analytics window on the Pi's own screen (DISPLAY=:0, labwc/Xwayland).
# Run it from a terminal on the Pi's desktop.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DISPLAY="${DISPLAY:-:0}"
exec "${VENV_DIR:-/opt/sixfab-dx/venv}/bin/python" "$HERE/watch_retail_local.py" "$@"
