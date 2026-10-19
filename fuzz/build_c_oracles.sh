#!/bin/sh
# Build the C oracles used by fuzz_ycbcr_diff_c, fuzz_raw_diff_c and
# fuzz_lossless_diff_c, and print the environment those targets read.
#
# usage: fuzz/build_c_oracles.sh <libjpeg-turbo 3.x prefix> [output dir]
#
# The prefix needs include/jpeglib.h and lib/ (or lib64/) libjpeg; the
# lossless oracle uses the 12- and 16-bit API, which only 3.x has. Run from
# the repository root. Output defaults to fuzz/oracles (git-ignored).
set -eu
prefix=${1:?usage: fuzz/build_c_oracles.sh <libjpeg-turbo prefix> [output dir]}
out=${2:-fuzz/oracles}
lib="$prefix/lib"
[ -d "$prefix/lib64" ] && lib="$prefix/lib64"
mkdir -p "$out"
for stem in ycbcr_decode_c_oracle raw_decode_c_oracle lossless_decode_c_oracle; do
  ${CC:-cc} -O2 -Wall -Wextra -o "$out/$stem" "examples/$stem.c" \
    -I"$prefix/include" -L"$lib" -ljpeg -Wl,-rpath,"$lib"
done
out=$(cd "$out" && pwd)
echo "export YCBCR_C_ORACLE=$out/ycbcr_decode_c_oracle"
echo "export RAW_C_ORACLE=$out/raw_decode_c_oracle"
echo "export LOSSLESS_C_ORACLE=$out/lossless_decode_c_oracle"
