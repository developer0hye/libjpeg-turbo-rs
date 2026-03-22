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
cargo test generate_seeds -- --ignored
```

This populates `fuzz/corpus/<target>/` with small valid JPEGs so the fuzzer starts from structurally meaningful inputs.

## Fuzz targets

| Target | Description |
|--------|-------------|
| `fuzz_decompress` | Main decoder — highest priority target |
| `fuzz_decompress_lenient` | Lenient-mode decoder (tolerates partial corruption) |
| `fuzz_roundtrip` | Compress then decompress — checks encoder/decoder consistency |
| `fuzz_read_coefficients` | DCT coefficient reader |
| `fuzz_transform` | Read coefficients then write them back |
| `fuzz_progressive_decoder` | Progressive scan-by-scan decoder |

## Run

Requires nightly Rust. Each command runs until interrupted or the time limit is reached.

```bash
# Run a single target (60-second quick smoke test)
cargo +nightly fuzz run fuzz_decompress -- -max_total_time=60

# Run with longer duration for thorough testing
cargo +nightly fuzz run fuzz_decompress -- -max_total_time=3600

# Run all targets sequentially (60 seconds each)
for target in $(cargo fuzz list); do
    echo "=== Fuzzing $target ==="
    cargo +nightly fuzz run "$target" -- -max_total_time=60
done
```

## List targets

```bash
cargo fuzz list
```

## Reproduce a crash

```bash
cargo +nightly fuzz run fuzz_decompress fuzz/artifacts/fuzz_decompress/<crash-file>
```

## Coverage

```bash
cargo +nightly fuzz coverage fuzz_decompress
```

## Directory structure

```
fuzz/
  Cargo.toml              # Fuzz crate manifest
  fuzz_targets/            # One .rs file per fuzz target
  corpus/<target>/         # Seed corpus per target (populated by generate_seeds test)
  artifacts/<target>/      # Crash-reproducing inputs (gitignored, created by fuzzer)
```
