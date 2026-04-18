#!/usr/bin/env bash
#
# Build (if necessary) and run `tjunittest` against OUR libturbojpeg
# cdylib, capturing per-subtest pass/fail.
#
# Usage:
#   ./run.sh                 # runs the default suite
#   TJU_ARGS="-yuv" ./run.sh # forward args to tjunittest (e.g. -yuv)
#
# Exit status mirrors tjunittest's own exit code so the Rust test harness
# can assert on it.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/build}"
BIN="$OUT_DIR/tjunittest"

if [ ! -x "$BIN" ]; then
  echo "Building tjunittest first..." >&2
  OUT_DIR="$OUT_DIR" bash "$SCRIPT_DIR/build.sh"
fi

LINK_DIR="$OUT_DIR/linkdir"

# macOS uses DYLD_LIBRARY_PATH in addition to the embedded rpath (which is
# already baked in by build.sh). Setting both keeps sanitizer-enabled and
# hardened runtime setups happy.
export LD_LIBRARY_PATH="$LINK_DIR:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$LINK_DIR:${DYLD_LIBRARY_PATH:-}"

# Run in a temp work dir so tjunittest can write intermediate files
# (it drops `test.*` test outputs in cwd).
WORK_DIR="$(mktemp -d -t tjunittest-link.XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT

cd "$WORK_DIR"
# shellcheck disable=SC2086
"$BIN" ${TJU_ARGS:-}
