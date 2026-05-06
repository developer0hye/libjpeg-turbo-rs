#!/usr/bin/env bash
# P2-10: libvips round-trip smoke against our libjpeg-turbo-rs cdylib.
#
# Round-trips a fixture PPM through `vips copy in.ppm out.jpg[Q=75]`
# (encode) then `vips copy out.jpg decoded.ppm` (decode), forcing libvips
# to load our cdylib (libjpeg.so.62 / libjpeg.62.dylib) via LD_PRELOAD
# (Linux) or DYLD_INSERT_LIBRARIES (macOS).
#
# libvips routes JPEG through `vips_foreign_save_jpeg_*` /
# `vips_foreign_load_jpeg_*`, both of which call into the libjpeg C ABI
# directly (`jpeg_create_compress`, `jpeg_write_scanlines`, `jpeg_finish_compress`,
# and the matching decompress entry points). Forcing libvips to bind those
# symbols against our cdylib exercises the same drop-in path the existing
# ImageMagick/Pillow harnesses cover but through libvips's `VipsImage`
# pipeline rather than `MagickWand` / `PIL.Image`.
#
# Exit codes (mirror examples/imagemagick_smoke/run.sh):
#   0   success (PSNR above threshold)
#   2   vips binary not on PATH
#   3   --lib argument missing/invalid
#   4   vips encode step failed
#   5   vips decode step failed
#   6   PPM parse error (handled by the Python PSNR check below)
#   7   PSNR below threshold
#   8   macOS SIP blocks DYLD_INSERT_LIBRARIES for the resolved binary
#   9   libvips not actually linked against libjpeg in this build
#   10  usage error
#  11   libvips is bound to mozjpeg (libjpeg-turbo *fork* with extra struct
#       fields) — our cdylib provides the dyld stubs (mozjpeg_compat.rs)
#       but cannot satisfy mozjpeg's wider `jpeg_compress_struct` layout
#       at runtime, so the in-process round-trip would segfault. The
#       Rust shim is ABI-compatible with stock libjpeg-turbo, which is
#       what every mainstream Linux distro (Debian, Ubuntu, Fedora)
#       packages — that's where this test exercises the real path.
#
# Usage:
#   run.sh --lib <cdylib> --input <input.ppm> --workdir <dir>
#          [--min-psnr 30.0] [--quality 75]

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

# ---- Locate vips -----------------------------------------------------
if ! command -v vips >/dev/null 2>&1; then
  echo "SKIP: vips not on PATH" >&2
  exit 2
fi
VIPS_BIN="$(command -v vips)"
VIPS_REAL="$(/usr/bin/readlink -f "$VIPS_BIN" 2>/dev/null || readlink -f "$VIPS_BIN" 2>/dev/null || echo "$VIPS_BIN")"

# ---- Verify libvips actually links libjpeg ---------------------------
# A vips build without `--with-jpeg` would silently fall through to a
# different decoder (e.g. libheif, libpng for PPM-only inputs). The whole
# point of this harness is to exercise our cdylib through libvips, so a
# vips that doesn't even link libjpeg is a real skip-with-reason — not a
# failure of our shim.
LINKED=""
case "$(uname -s)" in
  Linux)
    LINKED="$(ldd "$VIPS_REAL" 2>/dev/null | grep -E 'libjpeg\.so' || true)"
    ;;
  Darwin)
    LINKED="$(otool -L "$VIPS_REAL" 2>/dev/null | grep -E 'libjpeg\.[0-9.]+\.dylib' || true)"
    ;;
esac
# vips often loads libjpeg indirectly through libvips.so itself rather
# than from the CLI binary, so probe the libvips shared object too.
if [[ -z "$LINKED" ]]; then
  case "$(uname -s)" in
    Linux)
      LIBVIPS_SO="$(ldd "$VIPS_REAL" 2>/dev/null | awk '/libvips\.so/ {print $3}' | head -n1)"
      [[ -n "$LIBVIPS_SO" && -f "$LIBVIPS_SO" ]] && \
        LINKED="$(ldd "$LIBVIPS_SO" 2>/dev/null | grep -E 'libjpeg\.so' || true)"
      ;;
    Darwin)
      LIBVIPS_DYLIB="$(otool -L "$VIPS_REAL" 2>/dev/null | awk '/libvips\.[0-9.]+\.dylib/ {print $1}' | head -n1)"
      if [[ -n "$LIBVIPS_DYLIB" && -f "$LIBVIPS_DYLIB" ]]; then
        LINKED="$(otool -L "$LIBVIPS_DYLIB" 2>/dev/null | grep -E 'libjpeg\.[0-9.]+\.dylib' || true)"
      fi
      ;;
  esac
fi
if [[ -z "$LINKED" ]]; then
  echo "SKIP: this libvips build is not linked against libjpeg ($VIPS_REAL)" >&2
  echo "      install libvips with --with-jpeg / brew install vips to enable" >&2
  exit 9
fi

# mozjpeg detection. mozjpeg ships its own libjpeg.62.dylib that adds
# extra fields *inside* `jpeg_compress_struct` for trellis quantization,
# scan optimization, etc. A consumer compiled against mozjpeg's headers
# uses those fields directly (offsets past the libjpeg-turbo v8 layout)
# — there is no way our cdylib can satisfy that at runtime. Detect it
# by checking the dependency path resolution.
if [[ -n "$LINKED" ]] && echo "$LINKED" | grep -qi "mozjpeg"; then
  echo "SKIP: libvips on this host is bound to mozjpeg (a libjpeg-turbo fork" >&2
  echo "      with extra jpeg_compress_struct fields). The dyld load step" >&2
  echo "      succeeds via mozjpeg_compat.rs stubs but runtime calls into" >&2
  echo "      mozjpeg-specific struct layout would segfault. Test skipped" >&2
  echo "      on this host; Linux CI exercises the real libjpeg-turbo path." >&2
  echo "      Linked libjpeg: $(echo $LINKED | tr -s ' ' | head -c 200)" >&2
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
    case "$VIPS_REAL" in
      /usr/*|/bin/*|/sbin/*|/System/*|/Applications/*)
        echo "SKIP: macOS SIP blocks DYLD_INSERT_LIBRARIES for $VIPS_REAL" >&2
        echo "SKIP: install libvips via Homebrew to enable this test" >&2
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

# `vips copy <src>.ppm <dst>.jpg[Q=N,strip]` triggers
# vips_foreign_save_jpeg_file. The `Q=` option is libvips's quality knob;
# `strip` drops EXIF/XMP for a deterministic byte-stable output.
if ! "$VIPS_BIN" copy "$INPUT" "$ENCODED[Q=$QUALITY,strip]" 2>"$WORKDIR/encode.err"; then
  echo "vips encode step failed:" >&2
  sed 's/^/  /' "$WORKDIR/encode.err" >&2 || true
  exit 4
fi

[[ -s "$ENCODED" ]] || { echo "vips produced empty JPEG" >&2; exit 4; }
HDR=$(od -An -N2 -tx1 "$ENCODED" | tr -d ' \n')
[[ "$HDR" == "ffd8" ]] || { echo "encoded output missing SOI marker (got 0x$HDR)" >&2; exit 4; }

if ! "$VIPS_BIN" copy "$ENCODED" "$DECODED" 2>"$WORKDIR/decode.err"; then
  echo "vips decode step failed:" >&2
  sed 's/^/  /' "$WORKDIR/decode.err" >&2 || true
  exit 5
fi

# ---- PSNR check (same Python helper as imagemagick_smoke) ------------
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
