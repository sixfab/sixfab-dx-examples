#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
. "$HERE/lib.sh"
export PYTHON="${PYTHON:-python3}" SIXFAB_TEST=1 SIXFAB_NO_TTY=1 PATH="$HERE/fakebin:$PATH"
T="$(mktemp -d)"
export FAKE_CONFIG_DIR="$T/cfg" FAKE_LOG="$T/log" HOME="$T/home" ASSUME_YES=1
mkdir -p "$T/cfg" "$T/home" "$T/rel"
N=0

# stage_app <app>: copy _shared+the app to a fresh dir (REPO), serve fake release assets, rewrite SHAs
stage_app() {
  N=$((N+1)); REPO="$T/repo$N"; mkdir -p "$REPO"; cp -r "$ROOT/_shared" "$ROOT/$1" "$REPO/"
  local tag; tag="$(sed -n 's/^# TAG: //p' "$REPO/$1/models.txt")"
  mkdir -p "$T/rel/$tag"
  { echo "# TAG: $tag"
    grep -v -e '^#' -e '^$' "$REPO/$1/models.txt" | while read -r n p s o; do
      printf 'fake-%s' "$n" > "$T/rel/$tag/$n"
      echo "$n  $p  $(sha256sum "$T/rel/$tag/$n" | cut -d' ' -f1)  $o"; done; } > "$REPO/$1/models.txt.new"
  mv "$REPO/$1/models.txt.new" "$REPO/$1/models.txt"
}
export RELEASE_BASE_URL="file:///$(cygpath -m "$T" 2>/dev/null || echo "${T#/}")/rel"

for app in retail-analytics pose-estimation; do
  stage_app "$app"
  DEPLOY_DIR="$T/deploy-$app" bash "$REPO/$app/install.sh" --yes --dry-run >"$T/out" 2>&1     && pass "$app dry-run" || { fail "$app dry-run"; cat "$T/out"; }
  grep -q 'DRY: ' "$T/out" && pass "$app dry-run prints planned actions" || fail "$app dry-run prints planned actions"
  assert_no_file "$T/deploy-$app" "$app dry-run leaves no deploy dir"
done

# real file handling without the DX-RT wheel: model + app scripts land in DEPLOY_DIR
stage_app pose-estimation
DEPLOY_DIR="$T/deploy-pose" bash "$REPO/pose-estimation/install.sh" --yes >"$T/out" 2>&1   && pass "pose install (files only)" || { fail "pose install (files only)"; cat "$T/out"; }
assert_file "$T/deploy-pose/model/yolov8n-pose.dxnn" "pose model deployed"
assert_file "$T/deploy-pose/fall_detection.py" "pose app scripts deployed"

stage_app pose-estimation
msg="$(SIXFAB_TEST=0 SIXFAB_VENV="$T/no-such-venv" DEPLOY_DIR="$T/deploy-nv" bash "$REPO/pose-estimation/install.sh" --yes 2>&1)"; rc=$?
assert_eq "1" "$rc" "pose install stops when the sixfab-dx venv is missing"
echo "$msg" | grep -q 'Python environment not found' && pass "missing venv message" || fail "missing venv message"
mkdir -p "$T/fakevenv/bin"; printf '#!/usr/bin/env bash
exit 0
' > "$T/fakevenv/bin/python"; chmod +x "$T/fakevenv/bin/python"
SIXFAB_TEST=0 VENV_DIR="$T/fakevenv" DEPLOY_DIR="$T/deploy-v" bash "$REPO/pose-estimation/install.sh" --yes >"$T/out" 2>&1   && pass "pose install with a VENV_DIR override" || { fail "pose install with a VENV_DIR override"; tail -5 "$T/out"; }

finish
