#!/usr/bin/env bash
# Shared helpers for install-frigate.sh and the project install.sh scripts. Source it, do not run it.
LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
RELEASE_REPO="${RELEASE_REPO:-sixfab/sixfab-dx-examples}"
SIXFAB_VENV="${SIXFAB_VENV:-/opt/sixfab-dx/venv}"

die() { echo "ERROR: $*" >&2; exit 1; }
say() { echo "$*"; }

# ask PROMPT [DEFAULT] -> answer on stdout. The prompt goes to stderr so it is
# visible inside $(...). Reads the terminal (works behind curl | bash); with no
# terminal it returns DEFAULT.
ask() {
    local prompt="$1" default="${2:-}" reply=""
    printf '%s' "$prompt" >&2
    if [ "${ASK_FROM_STDIN:-0}" = "1" ]; then
        IFS= read -r reply || reply=""
    elif [ "${SIXFAB_NO_TTY:-0}" != "1" ] && ( : </dev/tty ) 2>/dev/null; then
        IFS= read -r reply </dev/tty || reply=""
    fi
    [ -t 2 ] || printf '\n' >&2
    printf '%s' "${reply:-$default}"
}

confirm() {
    [ "${ASSUME_YES:-0}" = "1" ] && return 0
    case "$(ask "$1 [y/N] ")" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

run() {
    if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY: $*"; else "$@"; fi
}

models_tag() { sed -n 's/^# TAG:[[:space:]]*//p' "$1" | head -1; }

# name <TAB> path <TAB> sha256 <TAB> optional-or-empty
models_rows() {
    grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' "$1" \
        | awk '{printf "%s\t%s\t%s\t%s\n", $1, $2, $3, $4}'
}

release_url() {
    local base="${RELEASE_BASE_URL:-https://github.com/${RELEASE_REPO}/releases/download}"
    echo "$base/$1/$2"
}

# fetch_assets APP_DIR WITH_OPTIONAL(0|1): download every asset of APP_DIR/models.txt
# to APP_DIR/<path>, verifying SHA-256. Existing files with a matching SHA are kept.
fetch_assets() {
    local app="$1" with_optional="${2:-0}" tag name path sha opt dest got
    tag="$(models_tag "$app/models.txt")"
    [ -n "$tag" ] || die "$app/models.txt has no '# TAG:' line"
    while IFS=$'\t' read -r name path sha opt; do
        if [ "$opt" = "optional" ] && [ "$with_optional" != "1" ]; then continue; fi
        dest="$app/$path"
        if [ -f "$dest" ] && [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then
            say "   have $path"; continue
        fi
        if [ "${DRY_RUN:-0}" = "1" ]; then echo "DRY: download $(release_url "$tag" "$name") -> $path"; continue; fi
        mkdir -p "$(dirname "$dest")"
        say "   downloading $name ($tag)"
        curl -fL --progress-bar -o "$dest.part" "$(release_url "$tag" "$name")" \
            || { rm -f "$dest.part"; die "could not download $name from release $tag"; }
        got="$(sha256sum "$dest.part" | cut -d' ' -f1)"
        if [ "$got" != "$sha" ]; then
            rm -f "$dest.part"
            die "SHA-256 mismatch for $name (expected $sha, got $got). The release may have changed."
        fi
        mv "$dest.part" "$dest"
    done < <(models_rows "$app/models.txt")
}

require_sixfab_venv() {
    [ -x "$SIXFAB_VENV/bin/python" ]         || die "Python environment not found at $SIXFAB_VENV. Install the sixfab-dx package first (see the README in the repo root), or set SIXFAB_VENV."
}


# parse_flags is shared (retail-analytics and pose-estimation use --yes/--dry-run/--with-demo;
# the Frigate-only --rtsp-url/--camera-name/--broker/--container/--config-dir flags are simply unused here).
parse_flags() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --rtsp-url)    RTSP_URL="$2"; shift 2 ;;
            --camera-name) CAMERA_NAME="$2"; shift 2 ;;
            --broker)      BROKER="$2"; shift 2 ;;
            --container)   CONTAINER="$2"; shift 2 ;;
            --config-dir)  CONFIG_DIR="$2"; shift 2 ;;
            --yes)         ASSUME_YES=1; shift ;;
            --dry-run)     DRY_RUN=1; shift ;;
            --with-demo)   WITH_DEMO=1; shift ;;
            -h|--help)     echo "Flags: --rtsp-url URL --camera-name NAME --broker HOST --container NAME --config-dir DIR --yes --dry-run --with-demo"; exit 0 ;;
            *) die "unknown flag '$1' (try --help)" ;;
        esac
    done
}
