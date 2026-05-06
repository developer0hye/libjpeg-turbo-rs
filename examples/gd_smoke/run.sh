#!/usr/bin/env bash
# P2-10: libgd round-trip smoke against our libjpeg-turbo-rs cdylib.
#
# Stages our cdylib under the libjpeg SONAME, sets LD_PRELOAD (Linux) or
# DYLD_INSERT_LIBRARIES (macOS), and runs the gd_smoke binary built by
# build.sh. The C harness performs the actual encode/decode + PSNR check.
#
# Exit codes:
#   0   success
#   1   gd_smoke binary not found (run build.sh first)
#   2   libgd binary skip-with-reason (no cdylib injection mechanism)
#   3   --lib argument missing/invalid
#   4   gd encode step failed (exit 4 from gd_smoke)
#   5   gd decode step failed (exit 5)
#   6   dim mismatch (exit 6)
#   7   PSNR below threshold (exit 7)
#   8   macOS SIP blocked DYLD_INSERT_LIBRARIES
#  10   usage error
#  11   libgd bound to mozjpeg (libjpeg-turbo fork — runtime layout incompat)
#
# Usage:
#   run.sh --lib <cdylib> --binary <gd_smoke> --input <input.ppm>
#          --workdir <dir> [--quality 75] [--min-psnr 30.0]

set -euo pipefail

QUALITY="75"
MIN_PSNR="30.0"
LIB=""
BINARY=""
INPUT=""
WORKDIR=""

usage() {
  cat >&2 <<EOF
usage: $0 --lib <cdylib> --binary <gd_smoke> --input <input.ppm>
          --workdir <dir> [--quality $QUALITY] [--min-psnr $MIN_PSNR]
EOF
  exit 10
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib)      LIB="${2:-}";      shift 2 ;;
    --binary)   BINARY="${2:-}";   shift 2 ;;
    --input)    INPUT="${2:-}";    shift 2 ;;
    --workdir)  WORKDIR="${2:-}";  shift 2 ;;
    --quality)  QUALITY="${2:-}";  shift 2 ;;
    --min-psnr) MIN_PSNR="${2:-}"; shift 2 ;;
    -h|--help)  usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "$LIB" && -n "$BINARY" && -n "$INPUT" && -n "$WORKDIR" ]] || usage
[[ -f "$LIB"    ]] || { echo "cdylib not found: $LIB"     >&2; exit 3; }
[[ -x "$BINARY" ]] || { echo "binary not found: $BINARY"  >&2; exit 1; }
[[ -f "$INPUT"  ]] || { echo "input not found: $INPUT"    >&2; exit 3; }
mkdir -p "$WORKDIR"

# ---- mozjpeg detection -----------------------------------------------
# The gd_smoke binary links to libgd which links to libjpeg. If that
# libjpeg is mozjpeg, the runtime struct layout is incompatible — see
# examples/libvips_smoke/run.sh for the rationale.
case "$(uname -s)" in
  Linux)
    LINKED="$(ldd "$BINARY" 2>/dev/null | grep -E 'libjpeg\.so' || true)"
    if [[ -z "$LINKED" ]]; then
      LIBGD_SO="$(ldd "$BINARY" 2>/dev/null | awk '/libgd\.so/ {print $3}' | head -n1)"
      [[ -n "$LIBGD_SO" && -f "$LIBGD_SO" ]] && \
        LINKED="$(ldd "$LIBGD_SO" 2>/dev/null | grep -E 'libjpeg\.so' || true)"
    fi
    ;;
  Darwin)
    LINKED="$(otool -L "$BINARY" 2>/dev/null | grep -E 'libjpeg\.[0-9.]+\.dylib' || true)"
    if [[ -z "$LINKED" ]]; then
      LIBGD_DYLIB="$(otool -L "$BINARY" 2>/dev/null | awk '/libgd\.[0-9.]+\.dylib/ {print $1}' | head -n1)"
      if [[ -n "$LIBGD_DYLIB" && -f "$LIBGD_DYLIB" ]]; then
        LINKED="$(otool -L "$LIBGD_DYLIB" 2>/dev/null | grep -E 'libjpeg\.[0-9.]+\.dylib' || true)"
      fi
    fi
    ;;
esac
if [[ -n "$LINKED" ]] && echo "$LINKED" | grep -qi "mozjpeg"; then
  echo "SKIP: libgd on this host is bound to mozjpeg (libjpeg-turbo fork" >&2
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

# ---- Run the harness -------------------------------------------------
"$BINARY" "$INPUT" "$QUALITY" "$MIN_PSNR"
