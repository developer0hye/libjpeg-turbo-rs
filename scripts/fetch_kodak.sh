#!/usr/bin/env bash
# Download the 24-image Kodak PhotoCD test set (public domain) and generate
# q75/q90 JPEGs into tests/fixtures/kodak/.
#
# Source: http://r0k.us/graphics/kodak/ (public domain, redistributable)
#
# The raw PNGs and full-res JPEGs are NOT checked into git — only a small
# subset of derived JPEGs may be committed.  Run this script locally to
# populate the full corpus before running comprehensive Kodak tests.
#
# Usage:
#   scripts/fetch_kodak.sh               # download + generate all 24 x q75 + q90
#   scripts/fetch_kodak.sh --clean       # remove downloaded PNGs after generation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${KODAK_CACHE_DIR:-$REPO_ROOT/target/kodak_cache}"
OUT_DIR="$REPO_ROOT/tests/fixtures/kodak"
BASE_URL="http://r0k.us/graphics/kodak"

mkdir -p "$CACHE_DIR" "$OUT_DIR"

CLEAN=0
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

if ! command -v cjpeg >/dev/null 2>&1; then
    echo "ERROR: cjpeg (libjpeg-turbo) not found in PATH." >&2
    echo "Install via:  brew install jpeg-turbo   (macOS)" >&2
    echo "              apt install libjpeg-turbo-progs   (Debian/Ubuntu)" >&2
    exit 1
fi

# Kodak images are named kodim01.png .. kodim24.png.
for i in $(seq -f '%02g' 1 24); do
    png="$CACHE_DIR/kodim$i.png"
    ppm="$CACHE_DIR/kodim$i.ppm"
    q75="$OUT_DIR/kodim${i}_q75.jpg"
    q90="$OUT_DIR/kodim${i}_q90.jpg"

    if [[ ! -f "$png" ]]; then
        echo "[fetch] kodim$i.png"
        curl -fsSL -o "$png" "$BASE_URL/kodim$i.png"
    fi

    # Decode PNG to PPM so cjpeg can read it (cjpeg does not ingest PNG).
    if [[ ! -f "$ppm" ]] || [[ "$png" -nt "$ppm" ]]; then
        if command -v sips >/dev/null 2>&1; then
            # macOS: use sips + a manual PPM re-encode through Python.
            python3 - "$png" "$ppm" <<'PY'
import struct, sys, zlib
png, ppm = sys.argv[1], sys.argv[2]
with open(png, "rb") as f:
    data = f.read()
assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
# Minimal PNG -> RGB8 decoder via stdlib: use image-agnostic fallback.
# We rely on Pillow if present; otherwise bail with message.
try:
    from PIL import Image
except ImportError:
    sys.stderr.write("ERROR: Python 'Pillow' required to decode PNGs on macOS.\n"
                     "Install via: pip3 install Pillow\n")
    sys.exit(3)
img = Image.open(png).convert("RGB")
w, h = img.size
with open(ppm, "wb") as out:
    out.write(f"P6\n{w} {h}\n255\n".encode())
    out.write(img.tobytes())
PY
        elif command -v pngtopnm >/dev/null 2>&1; then
            pngtopnm "$png" > "$ppm"
        elif command -v magick >/dev/null 2>&1; then
            magick "$png" "$ppm"
        else
            echo "ERROR: need sips+Pillow, pngtopnm, or magick to decode PNG." >&2
            exit 1
        fi
    fi

    if [[ ! -f "$q75" ]]; then
        echo "[encode] kodim${i}_q75.jpg"
        cjpeg -quality 75 -outfile "$q75" "$ppm"
    fi
    if [[ ! -f "$q90" ]]; then
        echo "[encode] kodim${i}_q90.jpg"
        cjpeg -quality 90 -outfile "$q90" "$ppm"
    fi
done

if [[ $CLEAN -eq 1 ]]; then
    rm -rf "$CACHE_DIR"
fi

echo "Kodak corpus ready under $OUT_DIR"
