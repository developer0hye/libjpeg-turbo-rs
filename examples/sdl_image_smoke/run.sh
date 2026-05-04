#!/usr/bin/env bash
# P2-10: SDL_image decode round-trip smoke against our cdylib.
#
# Stages our cdylib under the libjpeg SONAME, sets LD_PRELOAD (Linux) or
# DYLD_INSERT_LIBRARIES (macOS), and runs the sdl_image_smoke binary.
# The C harness reads a pre-encoded JPEG (encoded outside this path),
# decodes via SDL_image (which calls libjpeg through our cdylib), and
# compares the decoded surface to the reference PPM via PSNR.
#
# Exit codes:
#   0   success
#   1   binary missing (run build.sh first)
#   2   unsupported OS for cdylib injection
#   3   --lib / --binary / --jpeg / --ref / --workdir missing or invalid
#   4   IMG_Init JPG failed (SDL_image build doesn't link libjpeg)
#   5   IMG_Load_RW returned NULL (decoder rejected our cdylib's output)
#   6   dim mismatch
#   7   PSNR below threshold
#   8   macOS SIP blocked DYLD_INSERT_LIBRARIES
#  10   usage error
#  11   SDL_image bound to mozjpeg (libjpeg-turbo fork — runtime layout incompat)
#
# Usage:
#   run.sh --lib <cdylib> --binary <sdl_image_smoke> --jpeg <input.jpg>
#          --ref <reference.ppm> --workdir <dir> [--min-psnr 28.0]

set -euo pipefail

MIN_PSNR="28.0"
LIB=""
BINARY=""
JPEG=""
REF=""
WORKDIR=""

usage() {
  cat >&2 <<EOF
usage: $0 --lib <cdylib> --binary <sdl_image_smoke>
          --jpeg <input.jpg> --ref <reference.ppm>
          --workdir <dir> [--min-psnr $MIN_PSNR]
EOF
  exit 10
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib)      LIB="${2:-}";      shift 2 ;;
    --binary)   BINARY="${2:-}";   shift 2 ;;
    --jpeg)     JPEG="${2:-}";     shift 2 ;;
    --ref)      REF="${2:-}";      shift 2 ;;
    --workdir)  WORKDIR="${2:-}";  shift 2 ;;
    --min-psnr) MIN_PSNR="${2:-}"; shift 2 ;;
    -h|--help)  usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "$LIB" && -n "$BINARY" && -n "$JPEG" && -n "$REF" && -n "$WORKDIR" ]] || usage
[[ -f "$LIB"    ]] || { echo "cdylib not found: $LIB"     >&2; exit 3; }
[[ -x "$BINARY" ]] || { echo "binary not found: $BINARY"  >&2; exit 1; }
[[ -f "$JPEG"   ]] || { echo "jpeg not found: $JPEG"      >&2; exit 3; }
[[ -f "$REF"    ]] || { echo "ref not found: $REF"        >&2; exit 3; }
mkdir -p "$WORKDIR"

# ---- mozjpeg detection (matches gd/libvips/ffmpeg harnesses) ---------
case "$(uname -s)" in
  Linux)
    LINKED="$(ldd "$BINARY" 2>/dev/null | grep -E 'libjpeg\.so' || true)"
    if [[ -z "$LINKED" ]]; then
      LIBSDLIMG_SO="$(ldd "$BINARY" 2>/dev/null | awk '/libSDL2_image/ {print $3}' | head -n1)"
      [[ -n "$LIBSDLIMG_SO" && -f "$LIBSDLIMG_SO" ]] && \
        LINKED="$(ldd "$LIBSDLIMG_SO" 2>/dev/null | grep -E 'libjpeg\.so' || true)"
    fi
    ;;
  Darwin)
    LINKED="$(otool -L "$BINARY" 2>/dev/null | grep -E 'libjpeg\.[0-9.]+\.dylib' || true)"
    if [[ -z "$LINKED" ]]; then
      LIBSDLIMG_DYLIB="$(otool -L "$BINARY" 2>/dev/null | awk '/libSDL2_image-[0-9.]+\.[0-9]+\.dylib/ {print $1}' | head -n1)"
      if [[ -n "$LIBSDLIMG_DYLIB" && -f "$LIBSDLIMG_DYLIB" ]]; then
        LINKED="$(otool -L "$LIBSDLIMG_DYLIB" 2>/dev/null | grep -E 'libjpeg\.[0-9.]+\.dylib' || true)"
      fi
    fi
    ;;
esac
if [[ -n "$LINKED" ]] && echo "$LINKED" | grep -qi "mozjpeg"; then
  echo "SKIP: SDL_image on this host is bound to mozjpeg (libjpeg-turbo fork" >&2
  echo "      with extended jpeg_compress_struct layout)." >&2
  echo "      Linked libjpeg: $(echo $LINKED | tr -s ' ' | head -c 200)" >&2
  exit 11
fi

# ---- Per-OS loader setup --------------------------------------------
SYMLINK_DIR="$WORKDIR/symlinks"
mkdir -p "$SYMLINK_DIR"
BINARY_REAL="$(/usr/bin/readlink -f "$BINARY" 2>/dev/null || readlink -f "$BINARY" 2>/dev/null || echo "$BINARY")"

case "$(uname -s)" in
  Linux)
    ln -sf "$LIB" "$SYMLINK_DIR/libjpeg.so.62"
    ln -sf "libjpeg.so.62" "$SYMLINK_DIR/libjpeg.so"
    export LD_PRELOAD="$SYMLINK_DIR/libjpeg.so.62${LD_PRELOAD:+:$LD_PRELOAD}"
    export LD_LIBRARY_PATH="$SYMLINK_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  Darwin)
    ln -sf "$LIB" "$SYMLINK_DIR/libjpeg.62.dylib"
    ln -sf "libjpeg.62.dylib" "$SYMLINK_DIR/libjpeg.dylib"
    case "$BINARY_REAL" in
      /usr/*|/bin/*|/sbin/*|/System/*|/Applications/*)
        echo "SKIP: macOS SIP blocks DYLD_INSERT_LIBRARIES for $BINARY_REAL" >&2
        exit 8
        ;;
    esac
    export DYLD_INSERT_LIBRARIES="$SYMLINK_DIR/libjpeg.62.dylib${DYLD_INSERT_LIBRARIES:+:$DYLD_INSERT_LIBRARIES}"
    export DYLD_FORCE_FLAT_NAMESPACE=1
    export DYLD_LIBRARY_PATH="$SYMLINK_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    ;;
  *)
    echo "SKIP: unsupported OS for cdylib injection: $(uname -s)" >&2
    exit 2
    ;;
esac

"$BINARY" "$JPEG" "$REF" "$MIN_PSNR"
