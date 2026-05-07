# LAST_MILE Reference Commands

> **Index:** [docs/LAST_MILE.md](../LAST_MILE.md). Common commands referenced from the phase files.

## Workspace gates

```bash
cargo test --workspace --no-fail-fast
cargo build -p libjpeg-turbo-rs-capi --release
```

## Phase 1 / drop-in gates

```bash
# P0-1 — native transform cross-product
cargo test -p libjpeg-turbo-rs --test cross_product_transform tjtrantest_full_cross_product -- --exact

# P0-2 / P0-4 — stock tools build + run
bash examples/stock_djpeg_cjpeg/build.sh
bash examples/stock_djpeg_cjpeg/run.sh

# P0-3 — Pillow round-trip
cargo test --test capi_pillow_compat -- --nocapture

# Stock tool link + classic API surface
cargo test --test capi_stock_tool_link -- --include-ignored

# tjunittest harness
cargo test -p libjpeg-turbo-rs-capi --test tjunittest_link
```

## Phase 2 / system-library gates

```bash
# ABI cross-check (P3-1 / P2-4)
cargo test -p libjpeg-turbo-rs-capi --test abi_offsets --release

# Symbol inventory (P2-5 / P3-3)
cargo test -p libjpeg-turbo-rs-capi --test symbol_inventory --release

# Install-staging layout (P2-8)
cargo test -p libjpeg-turbo-rs-capi --test install_layout --release

# Distro-consumer harnesses (P2-10)
cargo test -p libjpeg-turbo-rs-capi --test capi_libvips_compat
cargo test -p libjpeg-turbo-rs-capi --test capi_ffmpeg_compat
cargo test -p libjpeg-turbo-rs-capi --test capi_gd_compat
cargo test -p libjpeg-turbo-rs-capi --test capi_sdl_image_compat

# format_message printf expansion (P2-2)
cargo test -p libjpeg-turbo-rs-capi --test format_message --release
```

## Differential fuzzing (P2-7)

```bash
cargo +nightly fuzz run fuzz_decode_diff_c     -- -max_total_time=600
cargo +nightly fuzz run fuzz_encode_diff_c     -- -max_total_time=600
cargo +nightly fuzz run fuzz_transform_diff_c  -- -max_total_time=600
```

Each must run 10 min in CI without finding a divergence.

## Encode SIMD perf (P1-Encode)

```bash
# Rust side
cargo bench --bench encode

# C baseline — source ships in examples/, no pre-built binary checked in.
# Source-file selection is platform-specific (different timing primitives):
#   * macOS → examples/bench_c_encode_matrix.c (mach_absolute_time)
#   * Linux → examples/bench_c_encode_linux.c (clock_gettime)
case "$(uname)" in
  Darwin) BENCH_SRC=examples/bench_c_encode_matrix.c ;;
  Linux)  BENCH_SRC=examples/bench_c_encode_linux.c ;;
  *)      echo "unsupported platform $(uname)"; exit 1 ;;
esac
if command -v pkg-config >/dev/null && pkg-config --exists libjpeg; then
  cc -O2 "$BENCH_SRC" -o /tmp/bench_c_encode_matrix \
     $(pkg-config --cflags --libs libjpeg) \
     -Wl,-rpath,$(pkg-config --variable=libdir libjpeg)
else
  PREFIX=${LIBJPEG_PREFIX:-${CONDA_PREFIX:-/opt/homebrew/opt/jpeg-turbo}}
  cc -O2 "$BENCH_SRC" -o /tmp/bench_c_encode_matrix \
     -I"$PREFIX/include" -L"$PREFIX/lib" -ljpeg \
     -Wl,-rpath,"$PREFIX/lib"
fi
/tmp/bench_c_encode_matrix
```

Acceptance: every encode benchmark `Rust/C ≤ 1.05×` with `RUSTFLAGS="-C target-cpu=native"`. Record the run in `experiments/encode.tsv` per the keep/discard/crash protocol in `experiments/README.md`.

## SONAME / install-name gate (P2-9)

```bash
# Default → loud cargo:warning lands on the build line.
cargo build -p libjpeg-turbo-rs-capi --release 2>&1 | grep -F "v6b"

# Production-safe build → silent.
CAPI_SONAME=libjpeg.so.8 CAPI_INSTALL_NAME=@rpath/libjpeg.8.dylib \
  cargo build -p libjpeg-turbo-rs-capi --release 2>&1 | grep -F "v6b" || echo "silent ok"
```

## tjbench harness (P2 Phase-1)

```bash
$OUT/tjbench testimages/testorig.jpg 95
```

Numbers within ±10 % of upstream `tjbench` on the same hardware.
