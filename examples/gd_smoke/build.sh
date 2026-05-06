#!/usr/bin/env bash
# P2-10: build the libgd round-trip C harness.
#
# Compiles `main.c` against libgd. Mirrors examples/libtiff_integration/
# build.sh: skip-with-reason when toolchain or library is absent;
# hard-fail only on real compile errors.
#
# Exit codes (consumed by run.sh and the Rust wrapper):
#   0 build succeeded — `gd_smoke` binary in --out-dir
#   1 libgd headers / library not found
#   2 cc compiler not found
#   3 compilation failed (real error)
#
# Usage:
#   build.sh --out-dir <dir>

set -euo pipefail

OUT_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    -h|--help) echo "usage: $0 --out-dir <dir>" >&2; exit 99 ;;
    *) echo "unknown arg: $1" >&2; exit 99 ;;
  esac
done

[[ -n "$OUT_DIR" ]] || { echo "missing --out-dir" >&2; exit 99; }
mkdir -p "$OUT_DIR"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---- Locate cc -------------------------------------------------------
if ! command -v cc >/dev/null 2>&1; then
  echo "cc compiler not found" >&2
  exit 2
fi

# ---- Locate libgd ----------------------------------------------------
# Prefer pkg-config so we pick up Homebrew's keg-only paths and the
# distro's per-arch include dirs without hardcoding. Fall back to a
# couple of canonical locations if pkg-config has no entry.
GD_CFLAGS=""
GD_LIBS=""
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists gdlib 2>/dev/null; then
  GD_CFLAGS="$(pkg-config --cflags gdlib)"
  GD_LIBS="$(pkg-config --libs gdlib)"
elif [[ -f /opt/homebrew/include/gd.h ]]; then
  GD_CFLAGS="-I/opt/homebrew/include"
  GD_LIBS="-L/opt/homebrew/lib -lgd"
elif [[ -f /usr/include/gd.h ]]; then
  GD_CFLAGS="-I/usr/include"
  GD_LIBS="-lgd"
else
  echo "libgd headers not found (tried pkg-config, /opt/homebrew, /usr)" >&2
  exit 1
fi

# We rely on the system libjpeg appearing in libgd's link line for the
# JPEG codec. The actual cdylib is injected by run.sh via LD_PRELOAD /
# DYLD_INSERT_LIBRARIES — we do not link our cdylib here directly.
echo "==> cc -O2 $GD_CFLAGS main.c $GD_LIBS -lm -o $OUT_DIR/gd_smoke" >&2
if ! cc -O2 -Wall -Wextra -Werror $GD_CFLAGS "$SCRIPT_DIR/main.c" $GD_LIBS -lm -o "$OUT_DIR/gd_smoke" 2>"$OUT_DIR/build.err"; then
  echo "compilation failed:" >&2
  sed 's/^/  /' "$OUT_DIR/build.err" >&2 || true
  exit 3
fi

echo "PASS: $OUT_DIR/gd_smoke" >&2
exit 0
