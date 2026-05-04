#!/usr/bin/env bash
# P2-10: FFmpeg round-trip smoke against our libjpeg-turbo-rs cdylib.
#
# FFmpeg's `mjpeg` codec ships in two flavours:
#   - **internal**: avcodec's own MJPEG encoder/decoder, never touches
#     libjpeg. This is the default for most distro builds (Homebrew,
#     Debian's `ffmpeg` package without `--enable-libjpeg`, etc.).
#   - **libjpeg-backed**: when ffmpeg is configured with
#     `--enable-libjpeg`, the `mjpeg` decoder routes through libjpeg's
#     `jpeg_create_decompress` / `jpeg_read_scanlines` directly.
#
# This harness exercises the libjpeg-backed path. We probe `ffmpeg
# -version` for `--enable-libjpeg`; if absent we exit 9 (skip-with-reason
# — the consumer is installed but does not actually use libjpeg) so the
# Rust wrapper can report it loudly. Real failures (encode/decode error,
# PSNR regression) panic.
#
# Exit codes:
#   0   success
#   2   ffmpeg binary not on PATH
#   3   --lib argument missing/invalid
#   4   ffmpeg encode step failed
#   5   ffmpeg decode step failed
#   7   PSNR below threshold
#   8   macOS SIP blocks DYLD_INSERT_LIBRARIES
#   9   this ffmpeg build is not linked against libjpeg
#   10  usage error
#
# Usage:
#   run.sh --lib <cdylib> --input <input.ppm> --workdir <dir>
#          [--min-psnr 30.0] [--quality 5]   # ffmpeg uses -q:v 1..31

set -euo pipefail

MIN_PSNR="30.0"
QUALITY="5"      # ffmpeg -q:v lower = higher quality; 5 ~= libjpeg q=80
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

# ---- Locate ffmpeg ---------------------------------------------------
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "SKIP: ffmpeg not on PATH" >&2
  exit 2
fi
FFMPEG_BIN="$(command -v ffmpeg)"
FFMPEG_REAL="$(/usr/bin/readlink -f "$FFMPEG_BIN" 2>/dev/null || readlink -f "$FFMPEG_BIN" 2>/dev/null || echo "$FFMPEG_BIN")"

# ---- Verify libjpeg is in the build ----------------------------------
# Two probes: (1) `--enable-libjpeg` flag in the configuration banner,
# (2) actual dynamic linkage to libjpeg.so / libjpeg.dylib. Either one
# satisfies the requirement; both signals together avoid false skips on
# distro builds that link libjpeg statically and don't list it in
# `--enable-libjpeg`.
HAS_LIBJPEG_FLAG=""
if "$FFMPEG_BIN" -version 2>/dev/null | grep -q -- "--enable-libjpeg"; then
  HAS_LIBJPEG_FLAG="yes"
fi
HAS_LIBJPEG_LINK=""
case "$(uname -s)" in
  Linux)
    if ldd "$FFMPEG_REAL" 2>/dev/null | grep -qE 'libjpeg\.so'; then
      HAS_LIBJPEG_LINK="yes"
    fi
    ;;
  Darwin)
    if otool -L "$FFMPEG_REAL" 2>/dev/null | grep -qE 'libjpeg\.[0-9.]+\.dylib'; then
      HAS_LIBJPEG_LINK="yes"
    fi
    ;;
esac

if [[ -z "$HAS_LIBJPEG_FLAG" && -z "$HAS_LIBJPEG_LINK" ]]; then
  echo "SKIP: this ffmpeg build does not use libjpeg ($FFMPEG_REAL)" >&2
  echo "      built-in MJPEG codec is internal to avcodec; rebuild ffmpeg" >&2
  echo "      with --enable-libjpeg to exercise this drop-in path" >&2
  exit 9
fi

# mozjpeg detection (rare for ffmpeg but defensive). See
# examples/libvips_smoke/run.sh for the rationale.
case "$(uname -s)" in
  Linux)
    LINKED_JPEG="$(ldd "$FFMPEG_REAL" 2>/dev/null | grep -E 'libjpeg\.so' || true)"
    ;;
  Darwin)
    LINKED_JPEG="$(otool -L "$FFMPEG_REAL" 2>/dev/null | grep -E 'libjpeg\.[0-9.]+\.dylib' || true)"
    ;;
esac
if [[ -n "$LINKED_JPEG" ]] && echo "$LINKED_JPEG" | grep -qi "mozjpeg"; then
  echo "SKIP: ffmpeg on this host is bound to mozjpeg (libjpeg-turbo fork" >&2
  echo "      with extended jpeg_compress_struct layout). dyld loads but" >&2
  echo "      runtime layout would diverge — Linux CI exercises the real" >&2
  echo "      libjpeg-turbo path." >&2
  exit 11
fi

# ---- Per-OS loader setup --------------------------------------------
OS="$(uname -s)"
SYMLINK_DIR="$WORKDIR/symlinks"
mkdir -p "$SYMLINK_DIR"

case "$OS" in
  Linux)
    ln -sf "$LIB" "$SYMLINK_DIR/libjpeg.so.62"
    ln -sf "libjpeg.so.62" "$SYMLINK_DIR/libjpeg.so"
    export LD_PRELOAD="$SYMLINK_DIR/libjpeg.so.62${LD_PRELOAD:+:$LD_PRELOAD}"
    export LD_LIBRARY_PATH="$SYMLINK_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    ;;
  Darwin)
    ln -sf "$LIB" "$SYMLINK_DIR/libjpeg.62.dylib"
    ln -sf "libjpeg.62.dylib" "$SYMLINK_DIR/libjpeg.dylib"
    case "$FFMPEG_REAL" in
      /usr/*|/bin/*|/sbin/*|/System/*|/Applications/*)
        echo "SKIP: macOS SIP blocks DYLD_INSERT_LIBRARIES for $FFMPEG_REAL" >&2
        exit 8
        ;;
    esac
    export DYLD_INSERT_LIBRARIES="$SYMLINK_DIR/libjpeg.62.dylib${DYLD_INSERT_LIBRARIES:+:$DYLD_INSERT_LIBRARIES}"
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

# Encode: PPM → JPEG via mjpeg codec. `-an` drops audio (none here, but
# defensive). `-y` overwrites without prompt. `-loglevel error` suppresses
# the noisy banner so PSNR output stays grep-able.
if ! "$FFMPEG_BIN" -y -loglevel error -i "$INPUT" -c:v mjpeg -q:v "$QUALITY" -an "$ENCODED" 2>"$WORKDIR/encode.err"; then
  echo "ffmpeg encode step failed:" >&2
  sed 's/^/  /' "$WORKDIR/encode.err" >&2 || true
  exit 4
fi

[[ -s "$ENCODED" ]] || { echo "ffmpeg produced empty JPEG" >&2; exit 4; }
HDR=$(od -An -N2 -tx1 "$ENCODED" | tr -d ' \n')
[[ "$HDR" == "ffd8" ]] || { echo "encoded output missing SOI marker (got 0x$HDR)" >&2; exit 4; }

# Decode: JPEG → PPM. ffmpeg auto-detects PPM from the .ppm extension.
if ! "$FFMPEG_BIN" -y -loglevel error -i "$ENCODED" "$DECODED" 2>"$WORKDIR/decode.err"; then
  echo "ffmpeg decode step failed:" >&2
  sed 's/^/  /' "$WORKDIR/decode.err" >&2 || true
  exit 5
fi

python3 - "$INPUT" "$DECODED" "$MIN_PSNR" <<'PYEOF'
import math
import sys

def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
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
if omax >= 256:
    raise SystemExit("only 8-bit PPMs are supported by this smoke")

sse = 0
for a, b in zip(opix, dpix):
    d = a - b
    sse += d * d
count = len(opix)
mse = sse / count if count else 0.0
psnr = float("inf") if mse == 0.0 else 10.0 * math.log10((255.0 ** 2) / mse)

print(f"PSNR={psnr:.3f} dB (min={min_psnr:.3f}) mse={mse:.6f} bytes={count}")
if psnr < min_psnr:
    raise SystemExit(7)
PYEOF
