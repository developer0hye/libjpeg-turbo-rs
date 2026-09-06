# libjpeg-turbo-rs

[![crates.io](https://img.shields.io/crates/v/libjpeg-turbo-rs.svg)](https://crates.io/crates/libjpeg-turbo-rs)
[![docs.rs](https://img.shields.io/docsrs/libjpeg-turbo-rs)](https://docs.rs/libjpeg-turbo-rs)
[![CI](https://github.com/developer0hye/libjpeg-turbo-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/developer0hye/libjpeg-turbo-rs/actions/workflows/ci.yml)
![MSRV](https://img.shields.io/badge/MSRV-1.87-blue)
![license](https://img.shields.io/crates/l/libjpeg-turbo-rs)

A high-performance, pure-Rust JPEG codec inspired by
[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo), with
NEON/AVX2/SSE2/WebAssembly SIMD128 acceleration, `no_std + alloc` support,
Rust-native streaming and buffer-reuse Application Programming Interfaces
(APIs), and extensive differential validation against pinned C
libjpeg-turbo oracles.

The Rust codec does **not** call a C JPEG implementation through a Foreign
Function Interface (FFI). "Pure Rust" does not mean "no `unsafe`":
architecture-specific Single Instruction, Multiple Data (SIMD) kernels and the
optional C Application Binary Interface (ABI) shim use narrowly scoped unsafe
code.

```bash
cargo add libjpeg-turbo-rs
```

```rust
use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

let image = decompress_to(&jpeg_bytes, PixelFormat::Rgb)?;
let encoded = compress(
    &image.data,
    image.width,
    image.height,
    PixelFormat::Rgb,
    85,
    Subsampling::S420,
)?;
```

## Should I use it?

Choose the interface you actually need. The readiness of one interface does
not promote the others.

| Your workload | Path | Current status |
| --- | --- | --- |
| Rust application or library | [`libjpeg-turbo-rs`](https://crates.io/crates/libjpeg-turbo-rs) | **T1:** feature-rich and appropriate for evaluation or production use after reviewing the live limitations below |
| Rust code using `image` traits | [`libjpeg-turbo-rs-image`](crates/libjpeg-turbo-rs-image) | Maintained bridge; validate color mapping and adapter overhead for your workload |
| Browser or WebAssembly System Interface (WASI) | [`libjpeg-turbo-rs-wasm`](crates/libjpeg-turbo-rs-wasm) | Supported; explicit `+simd128` compiler configuration is required outside this repository for the hand-written SIMD path |
| C/C++ using TurboJPEG 3 (`tj3*`) | [`libjpeg-turbo-rs-capi`](crates/libjpeg-turbo-rs-capi) | **T2:** primary C ABI target; use the symbol matrix and test the exact packaged artifact |
| Prebuilt C/C++ using classic libjpeg v8 (`libjpeg.so.8`) | Experimental v8 shim | **T3:** controlled pilots only; not a general system-library replacement |
| Prebuilt C/C++ using libjpeg v6b or v7 | None | **T4:** unsupported as a drop-in replacement because the exposed C structure layouts differ |

Read [`docs/ADOPTION_GUIDE.md`](docs/ADOPTION_GUIDE.md) for the decision tree,
evaluation checklist, migration steps, production rollout, stop conditions,
and rollback plan.

## Current readiness

The canonical, continuously maintained readiness source is
[`docs/LAST_MILE.md`](docs/LAST_MILE.md). This summary deliberately avoids an
unqualified "drop-in replacement" claim.

### Rust API

The major safe-API Undefined Behavior (UB) defects found during the 2026-08
audit are recorded as closed, and the live gate reports no known remaining UB
reachable through the safe Rust API. A formal memory-safety guarantee still
requires the remaining checked-layout centralization and automated unsafe-path
regression detection tracked by P4-139 and P4-141.

The full release-mode workspace gate is currently **red** because of P4-170:
two classic source-manager differential tests pass in debug mode and fail in
release mode. That does not imply that every Rust-native decode or encode path
fails, but it does mean the complete release gate must not be described as
green until the issue closes and the matrix is re-measured.

### C compatibility

- **TurboJPEG 3 is the primary C ABI target.** Its opaque handles avoid the
  version-dependent public-structure risk of classic libjpeg. Legacy
  TurboJPEG 1.x/2.x aliases are only partially covered; use the migration
  matrix in [`docs/ABI_COMPATIBILITY.md`](docs/ABI_COMPATIBILITY.md).
- **Classic libjpeg targets the v8 identity only** (`libjpeg.so.8`) and remains
  experimental and partial. Open lifecycle, option, error, threading,
  artifact-validation, and downstream-coverage gaps are tracked in the live
  gate.
- **v6b (`libjpeg.so.62`) and v7 (`libjpeg.so.7`) are not drop-in targets.**
  Renaming a v8-layout library does not make it binary-compatible with a
  consumer compiled against a different structure layout.

## Why choose it over a C binding?

For workloads that can use a Rust-native API, the project is designed to
remove adoption costs that codec benchmarks alone do not capture:

- one Cargo dependency rather than a C compiler, CMake/NASM setup, system
  package discovery, bindings, and platform-specific link configuration;
- Rust ownership, result types, builders, scanline/streaming APIs, and
  caller-owned reusable output buffers;
- one codec codebase for native, `no_std + alloc`, and WebAssembly (Wasm)
  targets;
- coefficient-domain transforms, metadata access, high-precision and lossless
  paths without crossing an FFI boundary;
- reproducible C-oracle differential tests, fuzzing, sanitizers, corpus tests,
  and explicit compatibility tiers;
- competitive measured performance on the strongest current targets: x86_64
  and aarch64.

The C implementation remains the safer choice when you require its installed
base, a v6b/v7 ABI, official platform packaging this project does not ship, an
uncovered architecture SIMD backend, or a classic libjpeg behavior that has
not passed this project's live replacement gate.

## Quick start

### Decode

```rust
use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};

let rgb = decompress(&jpeg_bytes)?;
println!("{}x{}", rgb.width, rgb.height);

let rgba = decompress_to(&jpeg_bytes, PixelFormat::Rgba)?;
```

### Decode into reusable memory

```rust
use libjpeg_turbo_rs::{decompress_into, output_buffer_size, PixelFormat};

let required = output_buffer_size(&jpeg_bytes, PixelFormat::Rgb)?;
let mut output = vec![0_u8; required];
let info = decompress_into(&jpeg_bytes, PixelFormat::Rgb, &mut output)?;
let pixels = &output[..info.bytes_written];
```

Keep and reuse the allocation in frame loops or services instead of measuring
only the one-shot convenience path.

### Encode

```rust
use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

let jpeg = compress(
    &rgb_pixels,
    width,
    height,
    PixelFormat::Rgb,
    85,
    Subsampling::S420,
)?;
```

### Configure advanced encoding

```rust
use libjpeg_turbo_rs::{Encoder, PixelFormat, Subsampling};

let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
    .quality(85)
    .subsampling(Subsampling::S420)
    .progressive(true)
    .optimize_huffman(true)
    .icc_profile(&icc_profile)
    .xmp_data(&xmp_packet)
    .iptc_data(&iptc_iim)
    .encode()?;
```

### Lossless coefficient transform

```rust
use libjpeg_turbo_rs::{transform, TransformOp};

let rotated = transform(&jpeg_bytes, TransformOp::Rot90)?;
```

Transforms rotate, flip, transpose, or crop Discrete Cosine Transform (DCT)
coefficients without decoding pixels. Metadata is preserved by default; use
`transform_jpeg_with_options` when you need an explicit marker-copy policy.

### Scanline processing

```rust
use libjpeg_turbo_rs::ScanlineDecoder;

let mut decoder = ScanlineDecoder::new(&jpeg_bytes)?;
let width = decoder.header().width as usize;
let height = decoder.header().height as usize;
let mut row = vec![0_u8; width * 3];

while decoder.output_scanline() < height {
    decoder.read_scanline(&mut row)?;
    // Process this RGB row.
}

let image = decoder.finish()?;
```

Runnable examples live in [`examples/`](examples). The crate-level examples
are compile-checked with `cargo test --doc`.

## Features

### JPEG modes

| Capability | Encode | Decode |
| --- | --- | --- |
| Baseline DCT with Huffman coding | yes | yes |
| Progressive DCT | yes | yes |
| Arithmetic coding | yes | yes |
| Lossless JPEG | yes | yes |
| 8/12/16-bit precision | yes | yes |
| Optimized Huffman tables | yes | not applicable |

The detailed feature inventory, including classic C API gaps, is maintained in
[`docs/FEATURE_PARITY.md`](docs/FEATURE_PARITY.md).

### Pixel formats and subsampling

Supported packed formats include grayscale, RGB, BGR, RGBA, BGRA, ARGB, ABGR,
RGBX, BGRX, XRGB, XBGR, CMYK, and RGB565 where the relevant operation supports
it.

Supported chroma layouts include 4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, 4:4:1,
4:1:0, and 2:4, plus grayscale and unusual/custom subsampling detection.

### Additional capabilities

- all 16 libjpeg scaled inverse DCT factors;
- coefficient access and lossless spatial transforms;
- Exchangeable Image File Format (EXIF) orientation inspection and application;
- International Color Consortium (ICC) profiles, XMP, IPTC, JFIF, Adobe APP14,
  and comment markers;
- packed and planar luminance/chrominance (YUV) paths;
- scanline and `std::io` streaming APIs;
- caller-owned reusable output buffers;
- MCU-aligned crop decoding;
- color quantization and dithering;
- recovery mode, custom Huffman/quantization tables, restart markers, and
  progress callbacks;
- `no_std + alloc` core codec;
- `image` trait bridge and Wasm/npm wrapper;
- TurboJPEG 3 and experimental classic libjpeg v8 C ABI shims.

## Feature flags and platform support

The root crate defaults to `std` and `simd`.

| Cargo feature | Default | Effect |
| --- | --- | --- |
| `std` | yes | `std::io` streaming, file helpers, runtime CPU-feature detection, standard error interoperability, and standard-library-dependent helpers |
| `simd` | yes | architecture intrinsics for supported targets |
| `png` | no | PNG image input/output for the relevant TurboJPEG image-loading helpers; implies `std` |

Build the core codec for `no_std + alloc` with:

```bash
cargo build --release --no-default-features
```

`alloc` is still required because decode, encode, and coefficient operations
own dynamically sized buffers.

| Target | Current acceleration | Status |
| --- | --- | --- |
| x86_64 Linux/macOS/Windows | AVX2 and SSE2 with runtime dispatch under `std` | First-class Rust target; Continuous Integration (CI) includes no-AVX2 coverage |
| aarch64 Linux/macOS | NEON | First-class and CI-tested |
| `wasm32` browser/WASI | SIMD128 when compiled with `+simd128` | Supported; compiler target feature is explicit outside this repository |
| `thumbv7em` bare metal | scalar | `no_std + alloc` build is CI-tested |
| armv7 / AArch32 | scalar by default | Functional CI coverage, but no production AArch32 NEON backend and no hardware performance claim |
| RISC-V, POWER, s390x | scalar | Functional support; compare on target hardware before adoption |

The root and C ABI crates currently declare Minimum Supported Rust Version
(MSRV) 1.87. The `image` bridge follows the higher requirement inherited from
its `image` dependency. MSRV changes are minor-version changes and are recorded
in [`CHANGELOG.md`](CHANGELOG.md).

## Performance

Performance claims are evidence, not universal properties. The tables below
are dated measurements from the repository harnesses. **A ratio below 1.00
means Rust used less time than C.**

### Representative C libjpeg-turbo comparison

| Platform and build | Operation | Workload | Rust / C time |
| --- | --- | --- | --- |
| Intel Core i5-10400, portable release decode, C 3.1.2 | decode | 1920x1080 4:2:0 | **0.90x** |
| Intel Core i5-10400, portable release decode, C 3.1.2 | decode | 3840x2160 4:2:0 | **0.88x** |
| Intel Core i5-10400, `target-cpu=native` encode, C 3.1.2 | encode | 1920x1080 4:2:0 | **0.98x** |
| Apple M1 Pro, portable release, C 3.1.0 | decode | 1920x1080 4:2:0 | 1.07x |
| Apple M1 Pro, portable release, C 3.1.0 | decode | 1920x1080 4:2:2 / 4:4:4 | **0.99x / 0.99x** |
| Apple M1 Pro, portable release, C 3.1.0 | encode | 1920x1080 4:2:0 / 4:2:2 / 4:4:4 | 1.07x / 1.06x / 1.03x |

Against `zune-jpeg` on the recorded aarch64 matrix, the decoder won 31 of 34
scored cases with a ±2% threshold; the losses were small fixed-cost or unusual
multi-scan cases. See
[`experiments/zune_matrix_aarch64_2026-07-28.md`](experiments/zune_matrix_aarch64_2026-07-28.md).

### Portable versus native x86_64 builds

Do not quote the native encode table as portable performance:

- **Portable:** plain `cargo build --release`. This is the relevant default for
  distributed binaries. Runtime dispatch still reaches SSE2/AVX2 kernels.
- **Native:** `RUSTFLAGS="-C target-cpu=native" cargo build --release`. This may
  enable additional BMI/FMA code generation, but the binary is tied to the
  build host's CPU capabilities.

Use a native build only for a controlled fleet whose CPUs match the build
target. Measure both paths with the same corpus before choosing one.

Full methodology and dated evidence live in
[`docs/ENCODING_PERFORMANCE.md`](docs/ENCODING_PERFORMANCE.md),
[`docs/CORPUS_TEST_REPORT.md`](docs/CORPUS_TEST_REPORT.md), and
[`experiments/`](experiments). The Product Requirements Document (PRD) requires
future benchmark reports to record CPU, operating system, compiler/toolchain,
C oracle version, build flags, corpus, warmup, sample count, and noise
threshold.

## Correctness, safety, and validation

The repository uses multiple complementary forms of evidence:

- differential tests against pinned `djpeg`, `cjpeg`, `jpegtran`, TurboJPEG,
  and classic libjpeg oracles;
- full feature-cross-product tests where exhaustive or bounded enumeration is
  practical;
- real and generated corpus tests;
- fuzz smoke jobs and longer-running fuzz targets;
- AddressSanitizer, UB checks, and C-boundary harnesses;
- cross-architecture and no-SIMD CI legs;
- WebAssembly and `no_std` builds;
- downstream OpenCV, libtiff, stock-tool, package, and loader harnesses where
  documented;
- code-level unsafe-boundary reviews and checked sizing/layout utilities.

No individual test, fuzzer, sanitizer, or review proves memory safety. The live
gate records both closed defects and the evidence still required before the
project offers a stronger guarantee.

Start with:

- [`docs/LAST_MILE.md`](docs/LAST_MILE.md) — canonical readiness and blockers;
- [`docs/TEST_PARITY.md`](docs/TEST_PARITY.md) — upstream behavior/test mapping;
- [`docs/CORPUS_TEST_REPORT.md`](docs/CORPUS_TEST_REPORT.md) — corpus results;
- [`docs/oracle_versions.tsv`](docs/oracle_versions.tsv) — pinned oracle
  identities.

## C ABI packages and release artifacts

The workspace includes `libjpeg-turbo-rs-capi`, which can produce the
TurboJPEG 3 library and an experimental classic v8 library. Tagged releases
publish checksummed native bundles for x86_64/aarch64 Linux and macOS,
including headers, package configuration files, CMake configuration, and the
SONAME/install-name chains.

Current distribution gaps:

- no Windows native C ABI bundle;
- no artifact signature or build attestation;
- no Software Bill of Materials (SBOM);
- no first-party deb/rpm package;
- classic v8 downstream harnesses do not yet prove every open lifecycle and
  compatibility contract against the exact shipped artifact.

Read [`docs/RELEASE_ARTIFACTS.md`](docs/RELEASE_ARTIFACTS.md) before installing
a bundle and [`docs/ABI_COMPATIBILITY.md`](docs/ABI_COMPATIBILITY.md) before
changing a dynamic-library search path.

## Documentation

[`docs/README.md`](docs/README.md) maps documents by audience and records their
source-of-truth order.

| Document | Purpose |
| --- | --- |
| [`PRD.md`](PRD.md) | Adoption strategy, requirements, milestones, priorities, and `1.0` definition of done |
| [`docs/ADOPTION_GUIDE.md`](docs/ADOPTION_GUIDE.md) | Integration choice, migration, evaluation, rollout, and rollback |
| [`docs/LAST_MILE.md`](docs/LAST_MILE.md) | Canonical T1-T4 release gate |
| [`docs/FEATURE_PARITY.md`](docs/FEATURE_PARITY.md) | Implemented feature inventory |
| [`docs/ABI_COMPATIBILITY.md`](docs/ABI_COMPATIBILITY.md) | C ABI, SONAME, layout, lifecycle, and threading policy |
| [`docs/C_API_REFERENCE.md`](docs/C_API_REFERENCE.md) | Per-function C API status |
| [`docs/TEST_PARITY.md`](docs/TEST_PARITY.md) | C reference and behavior validation map |
| [`docs/RELEASE_ARTIFACTS.md`](docs/RELEASE_ARTIFACTS.md) | Bundle contents, verification, installation, and known gaps |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Development workflow and required evidence |

## Contributing

The highest-priority work is not the largest unchecked feature list. It is the
work that makes adoption claims safer and easier to verify: release-mode gate
health, checked layout arithmetic, unsafe-boundary regression detection,
packaged-artifact validation, portable performance, supply-chain evidence,
and maintained ecosystem integrations.

Read [`CONTRIBUTING.md`](CONTRIBUTING.md), [`PRD.md`](PRD.md), and the relevant
live-gate item before starting. Pull requests that change feature, safety,
performance, API, ABI, platform, or readiness claims must include the evidence
and documentation update for that claim.

## License

Licensed under either of:

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE)); or
- MIT License ([`LICENSE-MIT`](LICENSE-MIT));

at your option.

## Acknowledgments

This software is based in part on the work of the Independent JPEG Group.
Algorithms and implementation techniques are informed by
[libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) and
[zune-jpeg](https://github.com/etemesi254/zune-image). Their licenses and
attribution requirements remain distinct from this project's dual license.
