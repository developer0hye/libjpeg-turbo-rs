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
- 15 structural edge-case byte sequences (bare SOI, truncated SOF0, multi-COM, APP1/APP2/APP14,
  zero-dimension SOF0, all-0xFF, non-JPEG bytes, etc.)
- `fuzz_transform_options`-specific seeds: 6 source JPEGs × 15 option combos + edge cases
  (117 seeds total for that target)
- `fuzz_api_sequence`-specific seeds: 9 operation programs × (6 source JPEGs + 15 structural
  edge-case bodies) = 189 seeds, each an `encode_program` byte prefix followed by one of those bodies

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
| `fuzz_decompress_precision` | 12/16-bit and arbitrary-precision decode entry points (`api/precision.rs`) | 353 |
| `fuzz_roundtrip` | Compress then decompress — checks encoder/decoder consistency | ~320 |
| `fuzz_read_coefficients` | DCT coefficient reader | ~347 |
| `fuzz_transform` | Read coefficients then write them back | ~331 |
| `fuzz_progressive_decoder` | Progressive scan-by-scan decoder | ~350 |
| `fuzz_encode_roundtrip` | Structured header + raw pixels → encode → decode assertion | ~294 |
| `fuzz_transform_options` | `transform_jpeg_with_options` with all TransformOp × option combos | ~117 |
| `fuzz_api_sequence` | Ordered `TjHandle` lifecycles — configure/probe/decode/reset/transform on one handle, each result compared against a handle built fresh from the same configuration (P4-141 criterion 3) | 189 |
| `fuzz_decode_diff_c` | Differential decode against a subprocess `djpeg` (P2-7) | 330 |
| `fuzz_encode_diff_c` | Differential encode round-trip against subprocess `cjpeg`/`djpeg` (P2-7) | 299 |
| `fuzz_transform_diff_c` | Differential transform against a subprocess `jpegtran` (P2-7) | 10 |
| `fuzz_ycbcr_diff_c` | YCbCr output (`set_output_colorspace(YCbCr)`) at every scale, box/fancy upsampling and crops, byte-exact against libjpeg's `JCS_YCbCr` (P4-244) | 198 |
| `fuzz_raw_diff_c` | `Decoder::decode_raw` component planes, byte-exact against `jpeg_read_raw_data` | 36 |
| `fuzz_lossless_diff_c` | `precision::decompress_lossless_arbitrary` at precision 2–16, byte-exact against libjpeg's lossless decoder (P4-247) | 25 |

### Targets with C oracle programs

`fuzz_ycbcr_diff_c`, `fuzz_raw_diff_c` and `fuzz_lossless_diff_c` compare
against libjpeg calls no command-line tool exposes, through small programs in
`examples/*_c_oracle.c`. Build them against a libjpeg-turbo 3.x development
install (headers and library; the lossless oracle needs the 12- and 16-bit
API), export the paths the script prints, then run as usual:

```bash
eval "$(fuzz/build_c_oracles.sh /opt/homebrew/opt/jpeg-turbo)"
cargo +nightly fuzz run fuzz_lossless_diff_c -- -dict=fuzz/jpeg.dict -max_total_time=600
```

Without its oracle such a target exits with an error rather than comparing
nothing, and an oracle that cannot be started, or that exits with anything but
0 or libjpeg's error status 1, panics the run. Each target asserts only on
inputs libjpeg decodes without a warning. Known divergences are handled in two
ways, each naming its LAST_MILE item; `LTRS_FUZZ_INCLUDE_KNOWN=1` turns both
off, to confirm the target still finds them or that a fix closed them:

- **Skipped before decoding**, because Rust refuses or cannot express the
  input: arithmetic-coded streams, YCbCr output of 12-bit and lossless frames
  (P4-246), lossless frames whose first scan codes fewer components than the
  frame or whose components are subsampled (P4-247), and lossless Tq/Ta above
  3 (P4-250).
- **Excused after the comparison**, so the input still exercises the code: a
  scaled decode of coefficients no 8-bit image produces (the scaled IDCTs
  saturate by design), 0xFF fill before a stuffed byte in a sequential
  Huffman frame (P4-248), raw output that equals libjpeg's with smoothing off
  (P4-249), and a lossless mismatch whose every differing sample equals
  libjpeg's masked to the precision (P4-66). P4-66 is only partly excused:
  once a wrapped sample spreads through later predictions the mismatch still
  fails, so lossless findings below 16 bits need replaying against a build
  that wraps modulo 2^16.

The lossless target also hands libjpeg a
one-component frame's sampling factors as 1×1, which T.81 makes equivalent:
for a 1×v component libjpeg applies a restart's prediction reset at the
start of the v-row iMCU row, so an interval that is not a multiple of v rows
decodes wrong in libjpeg-turbo 3.2.0 and right in Rust.

These targets run by hand only: they are not in `fuzz-smoke.yml`, and no PR
job builds them, so a green PR run does not show they still compile. The raw
target covers 8-bit frames only; 12-bit raw planes
(`raw_data_12::decompress_raw_12` against `jpeg12_read_raw_data`) have no
differential target.

`fuzz/known_divergences/<target>/` keeps one input per class, outside the
corpora so a run does not stop on them: replay with
`LTRS_FUZZ_INCLUDE_KNOWN=1` to confirm each still diverges, or that a fix
closed it. When an item closes, move its input into the target's corpus and
delete the skip or excuse that named it. The first campaign (2026-10-17, one
hour on a 20-core aarch64 host, about 44 million executions across the three
targets) found the classes above. A later two-minute run on this tree's base
found two more, which have no excuse yet, so their inputs fail with or without
the switch: a scaled, fancy-upsampled, cropped YCbCr decode with two Cb
samples wrong (P4-265), and an 8-bit RGB lossless frame with 18 samples wrong
(P4-266). Their seeds come from
`fuzz/scripts/make_c_oracle_diff_seeds.py <cjpeg 3.x>`, which includes a small
T.81 lossless encoder for layouts `cjpeg` cannot write; the
`diffseed_lossless_*` files it writes into `fuzz_decompress_precision/` too.

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
