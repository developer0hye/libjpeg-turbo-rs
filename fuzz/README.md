# Fuzzing

Fuzz testing for libjpeg-turbo-rs using [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz) (libFuzzer).

## Setup

```bash
cargo install cargo-fuzz
rustup toolchain install nightly
```

## Generate seed corpus

Before running the fuzzer for the first time, generate seed JPEG files from existing test fixtures:

```bash
cargo test --test generate_fuzz_seeds
```

This populates `fuzz/corpus/<target>/` with a large variety of structurally valid JPEG files
so libFuzzer starts from meaningful inputs. The generator covers:

- Full matrix: 3 content types × 7 subsampling modes × 3 quality levels × 6 entropy modes
  (~320 seeds per decoder target)
- Real-world fixtures from `tests/fixtures/` (7 files)
- Wide-aspect (8×64) and tall-aspect (64×8) images exercising dimension-extreme paths
- Restart-marker JPEGs (DRI segment injected after SOI, interval=2)
- 16 structural edge-case byte sequences (bare SOI, truncated SOF0, multi-COM, APP1/APP2/APP14,
  zero-dimension SOF0, all-0xFF, non-JPEG bytes, etc.)
- `fuzz_transform_options`-specific seeds: 6 source JPEGs × 15 option combos + edge cases
  (117 seeds total for that target)

## JPEG marker dictionary

`fuzz/jpeg.dict` is a libFuzzer dictionary covering all standard JPEG markers (SOI/EOI, SOF0–SOF15,
DHT, DAC, DQT, DRI, RST0–RST7, SOS, APP0–APP15, COM, DNL, DHP, EXP), common segment-length
prefixes, precision/component-count bytes, JFIF/Exif/Adobe APP identifiers, and byte-stuffing.

Use it with `-dict=fuzz/jpeg.dict` to bias mutation towards structurally valid JPEG sequences:

```bash
cargo +nightly fuzz run fuzz_decompress -- -dict=fuzz/jpeg.dict -max_total_time=3600
```

## Fuzz targets

| Target | Description | Corpus seeds |
|--------|-------------|-------------|
| `fuzz_decompress` | Main decoder — highest priority target | ~347 |
| `fuzz_decompress_lenient` | Lenient-mode decoder (tolerates partial corruption) | ~347 |
| `fuzz_decompress_precision` | 12/16-bit and arbitrary-precision decode entry points (`api/precision.rs`) | 3 |
| `fuzz_roundtrip` | Compress then decompress — checks encoder/decoder consistency | ~320 |
| `fuzz_read_coefficients` | DCT coefficient reader | ~347 |
| `fuzz_transform` | Read coefficients then write them back | ~331 |
| `fuzz_progressive_decoder` | Progressive scan-by-scan decoder | ~350 |
| `fuzz_encode_roundtrip` | Structured header + raw pixels → encode → decode assertion | ~294 |
| `fuzz_transform_options` | `transform_jpeg_with_options` with all TransformOp × option combos | ~117 |

## Run

Requires nightly Rust. Each command runs until interrupted or the time limit is reached.

```bash
# Run a single target (60-second quick smoke test)
cargo +nightly fuzz run fuzz_decompress -- -max_total_time=60

# Run with dictionary for structure-aware mutation
cargo +nightly fuzz run fuzz_decompress -- -dict=fuzz/jpeg.dict -max_total_time=3600

# Run the new transform-options target
cargo +nightly fuzz run fuzz_transform_options -- -dict=fuzz/jpeg.dict -max_total_time=1800

# Run all targets sequentially (60 seconds each)
for target in $(cargo +nightly fuzz list); do
    echo "=== Fuzzing $target ==="
    cargo +nightly fuzz run "$target" -- -max_total_time=60
done
```

## List targets

```bash
cargo +nightly fuzz list
```

## Reproduce a crash

```bash
cargo +nightly fuzz run fuzz_decompress fuzz/artifacts/fuzz_decompress/<crash-file>
```

## CI smoke schedule and throughput

The Fuzz Smoke workflow runs every 6 hours with, per target:

- `-fork=4` — 4 parallel workers (the differential targets spend most wall time blocked
  on djpeg/cjpeg/jpegtran subprocesses, so parallelism overlaps that wait);
- `-dict=fuzz/jpeg.dict` — marker-structure mutations;
- `-timeout=30` — per-input hangs become findings instead of eating the time budget;
- an **accumulated corpus** carried across runs via `actions/cache` (key prefix
  `fuzz-corpus-v1-<target>-`), so each run continues from all previously discovered
  coverage instead of restarting from the deterministic seeds. Green runs `cmin` the
  corpus before saving; failed runs save unminimized so the crash neighborhood survives.

Regression seeds named `regression_p4_*.jpg` in `corpus/fuzz_decode_diff_c/` are past CI
crash artifacts (P4-22/23/27/28/29 class bugs) committed permanently so mutation always
starts near historically bug-prone structures (exotic sampling geometries, non-conformant
progressive scan scripts).

## CI smoke failure artifacts

The Fuzz Smoke workflow uploads two artifact groups when a target fails:

- `fuzz-artifacts-<target>`: libFuzzer crash inputs from `fuzz/artifacts/<target>/`.
- `fuzz-repro-<target>`: `repro.txt` with the exact rerun commands and `versions.txt` with
  Rust, cargo-fuzz, and C tool versions from the failing runner.

## Coverage

```bash
cargo +nightly fuzz coverage fuzz_decompress
```

## Directory structure

```
fuzz/
  Cargo.toml              # Fuzz crate manifest
  jpeg.dict               # JPEG marker dictionary for structure-aware mutation
  fuzz_targets/           # One .rs file per fuzz target
  corpus/<target>/        # Seed corpus per target (populated by generate_fuzz_seeds test)
  artifacts/<target>/     # Crash-reproducing inputs (gitignored, created by fuzzer)
  repro/<target>/         # CI-only repro metadata artifacts (gitignored, created on failure)
```
