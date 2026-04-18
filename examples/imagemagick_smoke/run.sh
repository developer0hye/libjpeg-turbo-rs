#!/usr/bin/env bash
# FFI B9-3: ImageMagick smoke test against our libjpeg-turbo-rs cdylib.
#
# Round-trips a fixture PPM through `convert input.ppm -quality 75 out.jpg`
# then `convert out.jpg decoded.ppm` while forcing ImageMagick to load our
# cdylib (libjpeg.so.62 / libjpeg.62.dylib) via LD_PRELOAD (Linux) or
# DYLD_INSERT_LIBRARIES (macOS).
#
# Exit codes:
#   0   success (PSNR above threshold)
#   2   ImageMagick binary (magick/convert) not available on PATH
#   3   our cdylib path (--lib <path>) missing/invalid
#   4   encode convert step failed
#   5   decode convert step failed
#   6   PPM parse error
#   7   PSNR below threshold (round-trip quality regression)
#   8   SIP / dyld injection blocked by the OS (macOS system binaries)
#   10  usage error
#
# Usage:
#   run.sh --lib <path-to-cdylib> --input <ppm> --workdir <dir>
#          [--min-psnr 30.0] [--quality 75]
#
# On macOS Apple Silicon, `DYLD_INSERT_LIBRARIES` is restricted for
# system-protected binaries by System Integrity Protection (SIP). We only
# inject into Homebrew-installed `/opt/homebrew/**/magick` / `/usr/local/**`
# binaries, which are outside SIP scope. If the resolved binary lives under
# a SIP-protected prefix (/usr/bin, /bin, /System/...) we exit 8 so the
# calling harness can report a *real* skip-with-reason rather than a
# silent pass.

set -euo pipefail

MIN_PSNR="30.0"
QUALITY="75"
LIB=""
INPUT=""
WORKDIR=""

usage() {
  cat >&2 <<EOF
usage: $0 --lib <cdylib> --input <input.ppm> --workdir <dir>
          [--min-psnr $MIN_PSNR] [--quality $QUALITY]
EOF
  exit 10
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lib)      LIB="${2:-}";      shift 2 ;;
    --input)    INPUT="${2:-}";    shift 2 ;;
    --workdir)  WORKDIR="${2:-}";  shift 2 ;;
    --min-psnr) MIN_PSNR="${2:-}"; shift 2 ;;
    --quality)  QUALITY="${2:-}";  shift 2 ;;
    -h|--help)  usage ;;
    *) echo "unknown arg: $1" >&2; usage ;;
  esac
done

[[ -n "$LIB" && -n "$INPUT" && -n "$WORKDIR" ]] || usage
[[ -f "$LIB"   ]] || { echo "cdylib not found: $LIB"   >&2; exit 3; }
[[ -f "$INPUT" ]] || { echo "input not found: $INPUT"  >&2; exit 3; }
mkdir -p "$WORKDIR"

# ---- Locate ImageMagick ----------------------------------------------
# ImageMagick 7 ships `magick`; ImageMagick 6 ships `convert`. `magick
# convert ...` is the v7 invocation equivalent to v6 `convert ...`.
IM_BIN=""
IM_MODE=""
if command -v magick >/dev/null 2>&1; then
  IM_BIN="$(command -v magick)"
  IM_MODE="v7"
elif command -v convert >/dev/null 2>&1; then
  # v6 `convert`. Guard against the Windows/Linux filesystem `convert.exe`
  # that Windows ships — ImageMagick `convert` identifies with --version.
  if convert --version 2>/dev/null | grep -qi imagemagick; then
    IM_BIN="$(command -v convert)"
    IM_MODE="v6"
  fi
fi

if [[ -z "$IM_BIN" ]]; then
  echo "SKIP: ImageMagick (magick/convert) not on PATH" >&2
  exit 2
fi

# Resolve final real path so we can check SIP scope.
IM_REAL="$(/usr/bin/readlink -f "$IM_BIN" 2>/dev/null || readlink -f "$IM_BIN" 2>/dev/null || echo "$IM_BIN")"

# ---- Per-OS loader setup --------------------------------------------
OS="$(uname -s)"
LIB_DIR="$(dirname "$LIB")"
# Stage a symlink tree so the loader resolves by soname/install_name.
SYMLINK_DIR="$WORKDIR/symlinks"
mkdir -p "$SYMLINK_DIR"

case "$OS" in
  Linux)
    # Link our cdylib under the canonical soname so LD_PRELOAD picks it
    # in preference to the stock libjpeg.so.62 on the system.
    ln -sf "$LIB" "$SYMLINK_DIR/libjpeg.so.62"
    ln -sf "libjpeg.so.62" "$SYMLINK_DIR/libjpeg.so"
    export LD_PRELOAD="$SYMLINK_DIR/libjpeg.so.62${LD_PRELOAD:+:$LD_PRELOAD}"
    export LD_LIBRARY_PATH="$SYMLINK_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  Darwin)
    ln -sf "$LIB" "$SYMLINK_DIR/libjpeg.62.dylib"
    ln -sf "libjpeg.62.dylib" "$SYMLINK_DIR/libjpeg.dylib"
    # SIP scope: macOS strips DYLD_* when launching binaries under
    # /usr, /bin, /sbin, /System, /Applications. Homebrew at
    # /opt/homebrew and /usr/local is *not* SIP-protected, so the
    # injection works for brew-installed ImageMagick.
    case "$IM_REAL" in
      /usr/*|/bin/*|/sbin/*|/System/*|/Applications/*)
        echo "SKIP: macOS SIP blocks DYLD_INSERT_LIBRARIES for $IM_REAL" >&2
        echo "SKIP: install ImageMagick via Homebrew to enable this test" >&2
        exit 8
        ;;
    esac
    export DYLD_INSERT_LIBRARIES="$SYMLINK_DIR/libjpeg.62.dylib${DYLD_INSERT_LIBRARIES:+:$DYLD_INSERT_LIBRARIES}"
    # force_flat_namespace makes DYLD_INSERT_LIBRARIES win even when
    # the target was linked with two-level namespace (default on mac).
    export DYLD_FORCE_FLAT_NAMESPACE=1
    export DYLD_LIBRARY_PATH="$SYMLINK_DIR${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    ;;
  *)
    echo "SKIP: unsupported OS for cdylib injection: $OS" >&2
    exit 2
    ;;
esac

ENCODED="$WORKDIR/encoded.jpg"
DECODED="$WORKDIR/decoded.ppm"
rm -f "$ENCODED" "$DECODED"

im_invoke() {
  # Wrapper so v7 (`magick convert ...`) and v6 (`convert ...`) share
  # a single call site.
  if [[ "$IM_MODE" == "v7" ]]; then
    "$IM_BIN" convert "$@"
  else
    "$IM_BIN" "$@"
  fi
}

if ! im_invoke "$INPUT" -quality "$QUALITY" "$ENCODED" 2>"$WORKDIR/encode.err"; then
  echo "convert encode step failed:" >&2
  sed 's/^/  /' "$WORKDIR/encode.err" >&2 || true
  exit 4
fi

if [[ ! -s "$ENCODED" ]]; then
  echo "convert produced empty JPEG" >&2
  exit 4
fi

# Verify SOI/EOI — quick sanity that the encoder actually ran.
HDR=$(od -An -N2 -tx1 "$ENCODED" | tr -d ' \n')
if [[ "$HDR" != "ffd8" ]]; then
  echo "encoded output missing SOI marker (got 0x$HDR)" >&2
  exit 4
fi

if ! im_invoke "$ENCODED" "$DECODED" 2>"$WORKDIR/decode.err"; then
  echo "convert decode step failed:" >&2
  sed 's/^/  /' "$WORKDIR/decode.err" >&2 || true
  exit 5
fi

# ---- PSNR check ------------------------------------------------------
# Portable pure-POSIX PSNR against the original PPM: we parse the two
# PPM P6 headers, assert dims/maxval match, then compute MSE across
# every byte of raw pixel data. awk gives us the floating-point log10.
python3 - "$INPUT" "$DECODED" "$MIN_PSNR" <<'PYEOF'
import math
import sys

def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    # Tokenise the ASCII header (magic, width, height, maxval). PPM allows
    # comments with `#` and arbitrary whitespace between tokens.
    idx = 0
    def next_token():
        nonlocal idx
        while idx < len(data) and data[idx:idx+1] in (b" ", b"\t", b"\n", b"\r"):
            idx += 1
        if idx < len(data) and data[idx:idx+1] == b"#":
            while idx < len(data) and data[idx:idx+1] != b"\n":
                idx += 1
            return next_token()
        start = idx
        while idx < len(data) and data[idx:idx+1] not in (b" ", b"\t", b"\n", b"\r"):
            idx += 1
        return data[start:idx]
    magic = next_token()
    if magic != b"P6":
        raise SystemExit(f"{path}: not a P6 PPM (got {magic!r})")
    w = int(next_token())
    h = int(next_token())
    maxv = int(next_token())
    # Exactly one whitespace byte separates the header from the pixel block.
    idx += 1
    pixels = data[idx:]
    expected = w * h * 3 * (1 if maxv < 256 else 2)
    if len(pixels) < expected:
        raise SystemExit(f"{path}: short pixel data {len(pixels)} < {expected}")
    return w, h, maxv, pixels[:expected]

orig_path, dec_path, min_psnr_str = sys.argv[1], sys.argv[2], sys.argv[3]
min_psnr = float(min_psnr_str)

ow, oh, omax, opix = read_ppm(orig_path)
dw, dh, dmax, dpix = read_ppm(dec_path)
if (ow, oh) != (dw, dh):
    raise SystemExit(f"dimension mismatch: orig={ow}x{oh} decoded={dw}x{dh}")
if omax != dmax:
    raise SystemExit(f"maxval mismatch: orig={omax} decoded={dmax}")

# Byte-wise MSE is valid because both samples use identical maxval,
# and 8-bit PPMs are the exclusive path for JPEG quality=75 round-trips.
if omax >= 256:
    raise SystemExit("only 8-bit PPMs are supported by this smoke")

sse = 0
for a, b in zip(opix, dpix):
    d = a - b
    sse += d * d
count = len(opix)
mse = sse / count if count else 0.0
if mse == 0.0:
    psnr = float("inf")
else:
    psnr = 10.0 * math.log10((255.0 ** 2) / mse)

print(f"PSNR={psnr:.3f} dB (min={min_psnr:.3f}) mse={mse:.6f} bytes={count}")
if psnr < min_psnr:
    raise SystemExit(7)
PYEOF

echo "OK: ImageMagick round-trip via $IM_BIN ($IM_MODE) against $LIB"
