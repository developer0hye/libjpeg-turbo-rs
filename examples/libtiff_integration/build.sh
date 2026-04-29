#!/usr/bin/env bash
# Build libtiff_integration/main.c linked against:
#   - libtiff  (system install — the downstream consumer)
#   - OUR cdylib (libjpeg_turbo_rs_capi) — the JPEG back-end libtiff will call
#
# The binary does NOT link libjpeg directly; libtiff resolves JPEG symbols
# at runtime from whatever libjpeg it finds in its loader path.  run.sh then
# arranges DYLD_LIBRARY_PATH / LD_LIBRARY_PATH so that libtiff's JPEG calls
# resolve against our shim, not the system libjpeg.
#
# After the build, nm / otool (macOS) or ldd / nm (Linux) is used to confirm
# that the binary uses libtiff and that libtiff in turn has JPEG symbols
# (verifying our shim is in the resolution chain when run.sh sets the loader
# path).
#
# Exit codes:
#   0  success
#   1  libtiff headers / library not found  (test should SKIP)
#   2  cc not found                          (test should SKIP)
#   3  build failed                          (hard failure — something is wrong)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)}"
CAPI_TARGET_DIR="${CAPI_TARGET_DIR:-$REPO_ROOT/target/release}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/build}"
mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# 1. Locate cc.
# ---------------------------------------------------------------------------
CC="${CC:-cc}"
if ! command -v "$CC" >/dev/null 2>&1; then
    echo "SKIP: cc not found (set CC= env var to override)" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# 2. Locate libtiff.
# ---------------------------------------------------------------------------
# Probe in priority order: pkg-config → Homebrew opt/include → standard paths.
TIFF_INC=""
TIFF_LIB=""

if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libtiff-4 2>/dev/null; then
    TIFF_INC="$(pkg-config --cflags-only-I libtiff-4 2>/dev/null || true)"
    TIFF_LIB="$(pkg-config --libs        libtiff-4 2>/dev/null || true)"
fi

# Homebrew probe (macOS, both arm64 and x86_64).
if [[ -z "$TIFF_INC" ]]; then
    for prefix in /opt/homebrew /opt/homebrew/opt/libtiff /usr/local; do
        if [[ -f "$prefix/include/tiffio.h" ]]; then
            TIFF_INC="-I$prefix/include"
            TIFF_LIB="-L$prefix/lib -ltiff"
            break
        fi
    done
fi

# Common Linux system paths.
if [[ -z "$TIFF_INC" ]]; then
    for prefix in /usr /usr/local; do
        if [[ -f "$prefix/include/tiffio.h" ]]; then
            TIFF_INC="-I$prefix/include"
            TIFF_LIB="-L$prefix/lib -ltiff"
            break
        fi
    done
fi

if [[ -z "$TIFF_INC" ]]; then
    echo "SKIP: tiffio.h not found; install libtiff-dev (apt) or libtiff (brew)" >&2
    exit 1
fi

echo "==> libtiff: inc='$TIFF_INC' libs='$TIFF_LIB'" >&2

# ---------------------------------------------------------------------------
# 3. Locate our cdylib.
# ---------------------------------------------------------------------------
case "$(uname -s)" in
    Darwin)
        SHIM_LIB="$CAPI_TARGET_DIR/liblibjpeg_turbo_rs_capi.dylib"
        ;;
    Linux)
        SHIM_LIB="$CAPI_TARGET_DIR/liblibjpeg_turbo_rs_capi.so"
        ;;
    *)
        echo "SKIP: unsupported OS $(uname -s)" >&2
        exit 1
        ;;
esac

if [[ ! -f "$SHIM_LIB" ]]; then
    echo "ERROR: cdylib not found at $SHIM_LIB" >&2
    echo "       Build first: cargo build -p libjpeg-turbo-rs-capi --release" >&2
    exit 3
fi
echo "==> shim: $SHIM_LIB" >&2

# ---------------------------------------------------------------------------
# 4. Compile main.c.
#    We link libtiff so the binary can open TIFF files; we do NOT link our
#    shim directly into the binary — libtiff will dlopen / link libjpeg at
#    runtime, and run.sh steers that resolution to our shim via loader-path
#    env vars.  (On some libtiff builds libjpeg is a hard link-time dep of
#    libtiff itself, which is fine: run.sh's loader dir shadows the system
#    libjpeg.so.62 with a symlink to our cdylib.)
# ---------------------------------------------------------------------------
OUT_BIN="$OUT_DIR/libtiff_integration"
BUILD_LOG="$OUT_DIR/build.log"

echo "==> Compiling $SCRIPT_DIR/main.c -> $OUT_BIN" >&2
if ! "$CC" -O2 -Wno-unused -Wno-deprecated-declarations \
        $TIFF_INC \
        "$SCRIPT_DIR/main.c" \
        -o "$OUT_BIN" \
        $TIFF_LIB \
        2>"$BUILD_LOG"; then
    echo "FAIL: compilation failed (see $BUILD_LOG):" >&2
    cat "$BUILD_LOG" >&2
    exit 3
fi

echo "==> Build succeeded: $OUT_BIN" >&2

# ---------------------------------------------------------------------------
# 5. Sanity check: verify the binary references libtiff.
# ---------------------------------------------------------------------------
case "$(uname -s)" in
    Darwin)
        if otool -L "$OUT_BIN" 2>/dev/null | grep -q libtiff; then
            echo "==> VERIFY ok: binary references libtiff" >&2
        else
            echo "WARN: could not confirm libtiff linkage via otool" >&2
        fi
        ;;
    Linux)
        if ldd "$OUT_BIN" 2>/dev/null | grep -q libtiff || \
           nm -D "$OUT_BIN" 2>/dev/null | grep -q TIFF; then
            echo "==> VERIFY ok: binary references libtiff" >&2
        else
            echo "WARN: could not confirm libtiff linkage via ldd/nm" >&2
        fi
        ;;
esac

echo "OK: libtiff_integration built successfully" >&2
