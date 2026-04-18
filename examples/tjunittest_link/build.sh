#!/usr/bin/env bash
#
# Build `tjunittest` (from the libjpeg-turbo submodule) against OUR
# `libjpeg-turbo-rs-capi` cdylib. After this script runs you will find
# the binary at `$OUT_DIR/tjunittest`.
#
# Usage:
#   OUT_DIR=build ./build.sh
#
# Optional env:
#   CC          — override the C compiler (default: cc)
#   REPO_ROOT   — repository root (auto-detected by walking up from
#                 this script's location).
#   TARGET_DIR  — cargo target dir (default: $REPO_ROOT/target).
#   PROFILE     — cargo profile subdir (default: release).
#
# Exit status is non-zero if compilation or symlink setup fails.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
TARGET_DIR="${TARGET_DIR:-$REPO_ROOT/target}"
PROFILE="${PROFILE:-release}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/build}"
CC="${CC:-cc}"

REF_SRC="$REPO_ROOT/references/libjpeg-turbo/src"

if [ ! -f "$REF_SRC/tjunittest.c" ]; then
  echo "ERROR: $REF_SRC/tjunittest.c not found. Run:" >&2
  echo "  git submodule update --init --depth 1 references/libjpeg-turbo" >&2
  exit 2
fi

# Locate our cdylib.
if [ "$(uname)" = "Darwin" ]; then
  CDYLIB_NAME="liblibjpeg_turbo_rs_capi.dylib"
  VERSIONED_SO="libturbojpeg.0.dylib"
  SHORT_SO="libturbojpeg.dylib"
else
  CDYLIB_NAME="liblibjpeg_turbo_rs_capi.so"
  VERSIONED_SO="libturbojpeg.so.0"
  SHORT_SO="libturbojpeg.so"
fi

CDYLIB_PATH="$TARGET_DIR/$PROFILE/$CDYLIB_NAME"
if [ ! -f "$CDYLIB_PATH" ]; then
  echo "ERROR: cdylib not found at $CDYLIB_PATH" >&2
  echo "Run: cargo build -p libjpeg-turbo-rs-capi --release" >&2
  exit 3
fi

mkdir -p "$OUT_DIR"
LINK_DIR="$OUT_DIR/linkdir"
rm -rf "$LINK_DIR"
mkdir -p "$LINK_DIR"

# Symlink our cdylib under every name the link-time and load-time
# resolvers look for:
# - short `libturbojpeg.{dylib|so}` so `-lturbojpeg` picks it up
# - versioned `libturbojpeg.0.{dylib|so.0}` (the documented SONAME)
# - the install_name our build.rs bakes in
#   (`libjpeg.62.dylib` on macOS, `libjpeg.so.62` on Linux) so `dyld`/
#   `ld.so` resolve the embedded reference at runtime.
ln -sf "$CDYLIB_PATH" "$LINK_DIR/$VERSIONED_SO"
ln -sf "$CDYLIB_PATH" "$LINK_DIR/$SHORT_SO"
if [ "$(uname)" = "Darwin" ]; then
  ln -sf "$CDYLIB_PATH" "$LINK_DIR/libjpeg.62.dylib"
  ln -sf "$CDYLIB_PATH" "$LINK_DIR/libjpeg.dylib"
else
  ln -sf "$CDYLIB_PATH" "$LINK_DIR/libjpeg.so.62"
  ln -sf "$CDYLIB_PATH" "$LINK_DIR/libjpeg.so"
fi

# Compile: the test uses tjutil.c for getTime(), md5/md5.c + md5/md5hl.c
# for MD5{Init,Update,End,File}.
"$CC" \
  -O2 \
  -I "$SCRIPT_DIR" \
  -I "$REF_SRC" \
  "$REF_SRC/tjunittest.c" \
  "$REF_SRC/tjutil.c" \
  "$REF_SRC/md5/md5.c" \
  "$REF_SRC/md5/md5hl.c" \
  -L "$LINK_DIR" \
  -lturbojpeg \
  -Wl,-rpath,"$LINK_DIR" \
  -o "$OUT_DIR/tjunittest"

echo "Built: $OUT_DIR/tjunittest"
echo "Link dir: $LINK_DIR"
