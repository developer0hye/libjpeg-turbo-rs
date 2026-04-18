#!/usr/bin/env bash
# Download the USC-SIPI "Miscellaneous" subset (lena, mandrill, airplane)
# and generate test JPEGs into tests/fixtures/usc_sipi/.
#
# Source: https://sipi.usc.edu/database/database.php?volume=misc
#
# These images have been the lingua-franca of image-processing research since
# the 1970s and are distributed for academic / research use.  They are NOT
# checked into git — run this script locally to populate the corpus.
#
# Usage:
#   scripts/fetch_usc_sipi.sh            # download + generate q75 + q90 JPEGs
#   scripts/fetch_usc_sipi.sh --clean    # remove cached TIFFs after generation

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="${USC_SIPI_CACHE_DIR:-$REPO_ROOT/target/usc_sipi_cache}"
OUT_DIR="$REPO_ROOT/tests/fixtures/usc_sipi"
BASE_URL="https://sipi.usc.edu/database/misc"

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
    exit 1
fi

# Name -> USC-SIPI filename (all 512x512 colour unless noted).
#   lena     = 4.2.04.tiff
#   mandrill = 4.2.03.tiff
#   airplane = 4.2.05.tiff (F-16)
declare -a IMAGES=(
    "lena:4.2.04.tiff"
    "mandrill:4.2.03.tiff"
    "airplane:4.2.05.tiff"
)

for entry in "${IMAGES[@]}"; do
    name="${entry%%:*}"
    remote="${entry##*:}"
    tiff="$CACHE_DIR/$name.tiff"
    ppm="$CACHE_DIR/$name.ppm"
    q75="$OUT_DIR/${name}_q75.jpg"
    q90="$OUT_DIR/${name}_q90.jpg"

    if [[ ! -f "$tiff" ]]; then
        echo "[fetch] $name  ($remote)"
        curl -fsSL -o "$tiff" "$BASE_URL/$remote"
    fi

    if [[ ! -f "$ppm" ]] || [[ "$tiff" -nt "$ppm" ]]; then
        if command -v tiffcp >/dev/null 2>&1 && command -v tifftopnm >/dev/null 2>&1; then
            tifftopnm "$tiff" > "$ppm"
        elif command -v sips >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
            # macOS: TIFF -> BMP via sips, then parse minimal BMP -> PPM.
            bmp="$CACHE_DIR/$name.bmp"
            sips -s format bmp "$tiff" --out "$bmp" >/dev/null
            python3 - "$bmp" "$ppm" <<'PY'
import struct, sys
with open(sys.argv[1], "rb") as f:
    data = f.read()
assert data[:2] == b"BM", "not a BMP"
pixoff = struct.unpack_from("<I", data, 10)[0]
w = struct.unpack_from("<i", data, 18)[0]
h = struct.unpack_from("<i", data, 22)[0]
bpp = struct.unpack_from("<H", data, 28)[0]
assert bpp in (24, 32), f"unsupported bmp bpp: {bpp}"
row_stride = ((w * (bpp // 8)) + 3) & ~3
top_down = h < 0
h = abs(h)
out = bytearray()
for y in range(h):
    src_y = y if top_down else h - 1 - y
    off = pixoff + src_y * row_stride
    for x in range(w):
        px = off + x * (bpp // 8)
        b, g, r = data[px], data[px+1], data[px+2]
        out.extend((r, g, b))
with open(sys.argv[2], "wb") as f:
    f.write(f"P6\n{w} {h}\n255\n".encode())
    f.write(out)
PY
        else
            echo "ERROR: need tifftopnm or sips+python3 to decode TIFFs." >&2
            exit 1
        fi
    fi

    if [[ ! -f "$q75" ]]; then
        echo "[encode] ${name}_q75.jpg"
        cjpeg -quality 75 -outfile "$q75" "$ppm"
    fi
    if [[ ! -f "$q90" ]]; then
        echo "[encode] ${name}_q90.jpg"
        cjpeg -quality 90 -outfile "$q90" "$ppm"
    fi
done

if [[ $CLEAN -eq 1 ]]; then
    rm -rf "$CACHE_DIR"
fi

echo "USC-SIPI corpus ready under $OUT_DIR"
