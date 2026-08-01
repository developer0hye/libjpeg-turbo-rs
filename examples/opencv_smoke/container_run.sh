#!/usr/bin/env bash
set -euo pipefail

EXPECTED_OPENCV_VERSION="${EXPECTED_OPENCV_VERSION:-4.6.0+dfsg-13.1ubuntu1}"
ACTUAL_OPENCV_VERSION="$(dpkg-query -W -f='${Version}' libopencv-imgcodecs-dev)"
if [[ "$ACTUAL_OPENCV_VERSION" != "$EXPECTED_OPENCV_VERSION" ]]; then
  echo "OpenCV version drift: expected $EXPECTED_OPENCV_VERSION, got $ACTUAL_OPENCV_VERSION" >&2
  exit 10
fi

OPENCV_IMGCODECS="$(ldconfig -p | awk '/libopencv_imgcodecs\.so\.406 / { print $NF; exit }')"
if [[ -z "$OPENCV_IMGCODECS" || ! -f "$OPENCV_IMGCODECS" ]]; then
  echo "could not resolve libopencv_imgcodecs.so.406" >&2
  exit 11
fi
if ! ldd "$OPENCV_IMGCODECS" | grep -Eq 'libjpeg\.so\.8 .*=>'; then
  echo "OpenCV imgcodecs is not dynamically linked to libjpeg.so.8" >&2
  ldd "$OPENCV_IMGCODECS" >&2
  exit 12
fi

CANDIDATE_SYMBOLS="$(nm -D --defined-only /input/liblibjpeg_turbo_rs_capi.so)"
if ! grep -Eq '[[:space:]]tj3InitVersion(@@LIBJPEGTURBO_8\.0)?$' \
  <<<"$CANDIDATE_SYMBOLS"; then
  echo "candidate does not export the Rust shim's tj3InitVersion surface" >&2
  exit 13
fi
CANDIDATE_DYNAMIC="$(readelf -d /input/liblibjpeg_turbo_rs_capi.so)"
if ! grep -Eq '\(SONAME\).*\[libjpeg\.so\.8\]$' <<<"$CANDIDATE_DYNAMIC"; then
  echo "candidate does not advertise DT_SONAME=libjpeg.so.8" >&2
  exit 14
fi

mkdir -p /tmp/libjpeg-rs
ln -sf /input/liblibjpeg_turbo_rs_capi.so /tmp/libjpeg-rs/libjpeg.so.8

g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Werror \
  /harness/main.cpp -o /tmp/opencv_smoke \
  -I/usr/include/opencv4 -lopencv_imgcodecs -lopencv_core

rm -f \
  /work/system.jpg /work/rust.jpg \
  /work/system.jpg.color.raw /work/system.jpg.gray.raw \
  /work/rust.jpg.color.raw /work/rust.jpg.gray.raw \
  /work/rust.jpg.cross-color.raw /work/rust.jpg.cross-gray.raw \
  /work/system.txt /work/rust.txt /work/system-cross.txt \
  /work/jpeg-sha256.txt /work/lddebug.*
/tmp/opencv_smoke /work/system.jpg | tee /work/system.txt

LD_LIBRARY_PATH=/tmp/libjpeg-rs \
LD_DEBUG=bindings \
LD_DEBUG_OUTPUT=/work/lddebug \
  /tmp/opencv_smoke /work/rust.jpg /work/system.jpg | tee /work/rust.txt

/tmp/opencv_smoke /tmp/system-cross.jpg /work/rust.jpg \
  | tee /work/system-cross.txt

for artifact in \
  /work/system.jpg /work/rust.jpg \
  /work/system.jpg.color.raw /work/system.jpg.gray.raw \
  /work/rust.jpg.color.raw /work/rust.jpg.gray.raw \
  /work/rust.jpg.cross-color.raw /work/rust.jpg.cross-gray.raw; do
  if [[ ! -f "$artifact" || -L "$artifact" ]]; then
    echo "missing or non-regular output artifact: $artifact" >&2
    exit 15
  fi
done
if [[ /work/system.jpg -ef /work/rust.jpg ]]; then
  echo "system and Rust JPEG outputs alias the same file" >&2
  exit 16
fi

cmp /work/system.jpg.color.raw /work/rust.jpg.cross-color.raw
cmp /work/system.jpg.gray.raw /work/rust.jpg.cross-gray.raw
cmp /work/rust.jpg.color.raw /tmp/system-cross.jpg.cross-color.raw
cmp /work/rust.jpg.gray.raw /tmp/system-cross.jpg.cross-gray.raw
cmp /work/system.jpg /work/rust.jpg

if ! LD_LIBRARY_PATH=/tmp/libjpeg-rs ldd /tmp/opencv_smoke \
  | grep -Eq 'libjpeg\.so\.8 => /tmp/libjpeg-rs/libjpeg\.so\.8'; then
  echo "dynamic loader did not resolve libjpeg.so.8 to the Rust shim" >&2
  LD_LIBRARY_PATH=/tmp/libjpeg-rs ldd /tmp/opencv_smoke >&2
  exit 17
fi

if ! grep -Eh 'binding file .*libopencv_imgcodecs.* to /tmp/libjpeg-rs/libjpeg\.so\.8.*jpeg_CreateCompress' \
  /work/lddebug.* >/dev/null; then
  echo "OpenCV jpeg_CreateCompress did not bind to the Rust shim" >&2
  exit 18
fi
if ! grep -Eh 'binding file .*libopencv_imgcodecs.* to /tmp/libjpeg-rs/libjpeg\.so\.8.*jpeg_CreateDecompress' \
  /work/lddebug.* >/dev/null; then
  echo "OpenCV jpeg_CreateDecompress did not bind to the Rust shim" >&2
  exit 19
fi

sha256sum /work/system.jpg /work/rust.jpg | tee /work/jpeg-sha256.txt
echo "OpenCV downstream replacement: PASS"
