#!/usr/bin/env bash
# P2-10: build the SDL_image decode-round-trip C harness.
#
# Compiles `main.c` against SDL2 + SDL2_image. Mirrors examples/gd_smoke/
# build.sh: skip-with-reason when toolchain or library is absent;
# hard-fail only on real compile errors.
#
# Exit codes (consumed by run.sh / the Rust wrapper):
#   0 build succeeded — `sdl_image_smoke` binary in --out-dir
#   1 SDL2 / SDL2_image headers / library not found
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

if ! command -v cc >/dev/null 2>&1; then
  echo "cc compiler not found" >&2
  exit 2
fi

# ---- Locate SDL2 + SDL2_image ----------------------------------------
# Prefer pkg-config — both packages ship .pc files (sdl2 + SDL2_image).
SDL_CFLAGS=""
SDL_LIBS=""
if command -v pkg-config >/dev/null 2>&1 \
   && pkg-config --exists sdl2 SDL2_image 2>/dev/null; then
  SDL_CFLAGS="$(pkg-config --cflags sdl2 SDL2_image)"
  SDL_LIBS="$(pkg-config --libs sdl2 SDL2_image)"
elif [[ -f /opt/homebrew/include/SDL2/SDL_image.h ]]; then
  SDL_CFLAGS="-I/opt/homebrew/include/SDL2 -I/opt/homebrew/include"
  SDL_LIBS="-L/opt/homebrew/lib -lSDL2 -lSDL2_image"
elif [[ -f /usr/include/SDL2/SDL_image.h ]]; then
  SDL_CFLAGS="-I/usr/include/SDL2 -D_REENTRANT"
  SDL_LIBS="-lSDL2 -lSDL2_image"
else
  echo "SDL2_image headers not found (tried pkg-config, /opt/homebrew, /usr)" >&2
  exit 1
fi

echo "==> cc -O2 $SDL_CFLAGS main.c $SDL_LIBS -lm -o $OUT_DIR/sdl_image_smoke" >&2
if ! cc -O2 -Wall -Wextra $SDL_CFLAGS "$SCRIPT_DIR/main.c" $SDL_LIBS -lm -o "$OUT_DIR/sdl_image_smoke" 2>"$OUT_DIR/build.err"; then
  echo "compilation failed:" >&2
  sed 's/^/  /' "$OUT_DIR/build.err" >&2 || true
  exit 3
fi

echo "PASS: $OUT_DIR/sdl_image_smoke" >&2
exit 0
