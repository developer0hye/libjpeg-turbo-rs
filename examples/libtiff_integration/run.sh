#!/usr/bin/env bash
# Run the libtiff_integration binary with our cdylib staged as the JPEG
# provider that libtiff resolves at runtime.
#
# libtiff links against libjpeg.so.62 / libjpeg.62.dylib.  By prepending a
# private loader directory (containing symlinks from the canonical SONAME to
# our cdylib) to DYLD_LIBRARY_PATH / LD_LIBRARY_PATH we ensure that libtiff's
# JPEG calls hit our shim rather than the system libjpeg — this is the same
# technique used by examples/stock_djpeg_cjpeg/run.sh.
#
# Exit codes:
#   0  PASS
#   1  FAIL  (pixel mismatch or API failure — real bug in our shim)
#   2  SKIP  (binary or shim not found — environment not set up)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)}"
CAPI_TARGET_DIR="${CAPI_TARGET_DIR:-$REPO_ROOT/target/release}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/build}"
BIN="$OUT_DIR/libtiff_integration"

if [[ ! -x "$BIN" ]]; then
    echo "SKIP: binary not found at $BIN; run build.sh first" >&2
    exit 2
fi

# Locate the cdylib.
case "$(uname -s)" in
    Darwin)
        SHIM_LIB="$CAPI_TARGET_DIR/liblibjpeg_turbo_rs_capi.dylib"
        DYLIB_EXT="dylib"
        ;;
    Linux)
        SHIM_LIB="$CAPI_TARGET_DIR/liblibjpeg_turbo_rs_capi.so"
        DYLIB_EXT="so"
        ;;
    *)
        echo "SKIP: unsupported OS $(uname -s)" >&2
        exit 2
        ;;
esac

if [[ ! -f "$SHIM_LIB" ]]; then
    echo "SKIP: cdylib not found at $SHIM_LIB" >&2
    exit 2
fi

# Absolutize the shim path so the symlink target resolves correctly from any
# working directory (same guard as stock_djpeg_cjpeg/run.sh).
SHIM_LIB="$(cd "$(dirname "$SHIM_LIB")" && pwd)/$(basename "$SHIM_LIB")"

# Stage a private loader directory with canonical-SONAME symlinks pointing at
# our cdylib.  Both libjpeg.so.62 / libjpeg.62.dylib (classic jpeglib ABI)
# and libturbojpeg.so.0 / libturbojpeg.0.dylib (TurboJPEG ABI) are covered
# because libtiff only needs the classic ABI, but the symlink costs nothing.
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

LOADER_DIR="$WORK/loader"
mkdir -p "$LOADER_DIR"

if [[ "$DYLIB_EXT" == "dylib" ]]; then
    # libjpeg.62.dylib — classic ABI SONAME our shim bakes in via @rpath
    ln -sf "$SHIM_LIB" "$LOADER_DIR/libjpeg.62.dylib"
    # libjpeg.8.dylib — Homebrew libtiff links against libjpeg-turbo's .8 SONAME
    ln -sf "$SHIM_LIB" "$LOADER_DIR/libjpeg.8.dylib"
    ln -sf "$SHIM_LIB" "$LOADER_DIR/libturbojpeg.0.dylib"
else
    ln -sf "$SHIM_LIB" "$LOADER_DIR/libjpeg.so.62"
    # libjpeg.so.8 — some Linux libtiff builds link against libjpeg-turbo .8 SONAME
    ln -sf "$SHIM_LIB" "$LOADER_DIR/libjpeg.so.8"
    ln -sf "$SHIM_LIB" "$LOADER_DIR/libturbojpeg.so.0"
fi

# Temp TIFF output path.
TIF_PATH="$WORK/round_trip.tif"

echo "==> Running libtiff_integration with our shim as JPEG provider" >&2
echo "    shim:    $SHIM_LIB" >&2
echo "    tif:     $TIF_PATH" >&2

# Disable set -e for the binary invocation so we can capture and translate exit codes.
set +e
DYLD_LIBRARY_PATH="$LOADER_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
LD_LIBRARY_PATH="$LOADER_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$BIN" "$TIF_PATH"
BIN_RC=$?
set -e

case $BIN_RC in
    0)
        echo "PASS: libtiff_integration" >&2
        exit 0
        ;;
    1)
        # Pixel mismatch — real JPEG round-trip quality failure.
        echo "FAIL: libtiff_integration pixel mismatch (exit 1)" >&2
        exit 1
        ;;
    *)
        # exit 2+ from main.c means API failure (TIFFOpen / TIFFWriteEncodedStrip /
        # TIFFReadEncodedStrip returned an error).  Map to exit 3 so the Rust test
        # can distinguish this real failure from its own "tool not found" skip (exit 2).
        echo "FAIL: libtiff_integration API error (binary exited $BIN_RC, reporting as 3)" >&2
        exit 3
        ;;
esac
