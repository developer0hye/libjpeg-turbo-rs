# libjpeg-turbo-rs

[![crates.io](https://img.shields.io/crates/v/libjpeg-turbo-rs.svg)](https://crates.io/crates/libjpeg-turbo-rs)
[![docs.rs](https://img.shields.io/docsrs/libjpeg-turbo-rs)](https://docs.rs/libjpeg-turbo-rs)
[![CI](https://github.com/developer0hye/libjpeg-turbo-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/developer0hye/libjpeg-turbo-rs/actions/workflows/ci.yml)
![MSRV](https://img.shields.io/badge/MSRV-1.87-blue)
![license](https://img.shields.io/crates/l/libjpeg-turbo-rs)

Pure-Rust reimplementation of [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) with NEON/AVX2/SSE2/WASM-SIMD128 acceleration. No C dependencies, no unsafe FFI, `no_std`-capable — and byte-for-byte cross-validated against C libjpeg-turbo in CI.

```sh
cargo add libjpeg-turbo-rs
```

```rust
use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

let image = decompress_to(&jpeg_bytes, PixelFormat::Rgb)?; // decode any JPEG to RGB
let jpeg = compress(&image.data, image.width, image.height,
                    image.pixel_format, 85, Subsampling::S420)?; // re-encode
```

These snippets are compile-checked: the crate-level doctests mirror them
(`cargo test --doc`). Runnable examples live in
[`examples/`](examples/) (the directory README fronts the user-facing set once #388 lands; GitHub renders it in the directory view).

## How it compares

Measured with the in-repo harnesses (methodology: [#361](https://github.com/developer0hye/libjpeg-turbo-rs/issues/361), [#392](https://github.com/developer0hye/libjpeg-turbo-rs/issues/392); `examples/bench_zune_matrix.rs`, `experiments/image_bridge.md`):

| vs | Result (decode) |
| --- | --- |
| **zune-jpeg** (the `image` crate's default) | **31 wins / 3 losses** of 34 scored cases (±2% threshold) across subsampling × progressive × 8×8→8K, quiet aarch64, 2026-07-28; e.g. 4K progressive **0.65×**, 4K 4:2:0 **0.74×** of zune's time. Through the `image`-crate bridge: **1.31× faster** at 1080p. Losses are three tiny-image fixed-cost cases. |
| **C libjpeg-turbo** | Matches or beats C on most decode benchmarks on x86_64/AVX2 (i5-10400) and within a few % on aarch64/NEON (M1 Pro) — the dated per-platform tables are below. |

## Performance

### x86_64 (AVX2)

Intel Core i5-10400 @ 2.90GHz (turbo off, `performance` governor), C libjpeg-turbo 3.1.2, quality 75:

#### Decoding

| Image | Subsampling | Rust (us) | C (us) | Ratio |
|-------|-------------|-----------|--------|-------|
| 64x64 | 4:2:0 | 60 | 49 | 1.23x |
| 320x240 | 4:2:0 | 769 | 996 | **0.77x** |
| 640x480 | 4:2:0 | 929 | 880 | 1.05x |
| 640x480 | 4:2:2 | 3,267 | 3,480 | **0.94x** |
| 640x480 | 4:4:4 | 4,794 | 5,525 | **0.87x** |
| 1280x720 | 4:2:0 | 8,707 | 9,997 | **0.87x** |
| 1920x1080 | 4:2:0 | 19,736 | 22,031 | **0.90x** |
| 1920x1080 | 4:2:2 | 25,382 | 26,227 | **0.97x** |
| 1920x1080 | 4:4:4 | 37,585 | 40,026 | **0.94x** |
| 2560x1440 | 4:2:0 | 35,137 | 37,918 | **0.93x** |
| 3840x2160 | 4:2:0 | 78,868 | 89,325 | **0.88x** |

#### Encoding (built with `RUSTFLAGS="-C target-cpu=native"`)

| Image | Subsampling | Rust (µs) | C (µs) | Ratio |
|-------|-------------|-----------|--------|-------|
| 320x240 | 4:2:0 | 381 | 403 | **0.94x** |
| 320x240 | 4:2:2 | 474 | 508 | **0.93x** |
| 320x240 | 4:4:4 | 709 | 764 | **0.93x** |
| 640x480 | 4:2:2 | 1,653 | 1,731 | **0.96x** |
| 640x480 | 4:4:4 | 2,397 | 2,558 | **0.94x** |
| 1920x1080 | 4:2:0 | 10,273 | 10,474 | **0.98x** |
| 1920x1080 | 4:2:2 | 12,783 | 13,082 | **0.98x** |
| 1920x1080 | 4:4:4 | 19,057 | 19,873 | **0.96x** |

### aarch64 (NEON)

Apple M1 Pro, C libjpeg-turbo 3.1.0, quality 75:

#### Decoding (1920x1080)

| Subsampling | Rust (µs) | C (µs) | Ratio |
|-------------|-----------|--------|-------|
| 4:2:0 | 12,159 | 11,333 | 1.07x |
| 4:2:2 | 15,246 | 15,329 | **0.99x** |
| 4:4:4 | 22,972 | 23,130 | **0.99x** |

#### Encoding (1920x1080)

| Subsampling | Rust (µs) | C (µs) | Ratio |
|-------------|-----------|--------|-------|
| 4:2:0 | 5,724 | 5,332 | 1.07x |
| 4:2:2 | 7,148 | 6,766 | 1.06x |
| 4:4:4 | 10,596 | 10,272 | 1.03x |

**aarch64**: Decoding matches or beats C for 4:2:2 and 4:4:4; 4:2:0 has a 7% gap. Encoding matches or beats C in 7 of 8 configurations (see [`docs/ENCODING_PERFORMANCE.md`](docs/ENCODING_PERFORMANCE.md)); the remaining 1080p 4:2:0 gap (~4%) is structural function-call overhead.

**x86_64**: Decoding beats C across most resolutions. Encoding (with `target-cpu=native`) beats C in every benchmark above by 2–7 %; the encoder runs SSE2 Huffman + AVX2 FDCT/quantize/color/downsample. The Huffman bitmap-iteration hot path uses runtime `is_x86_feature_detected!("bmi1") && is_x86_feature_detected!("lzcnt")` dispatch (P4-8, see `src/encode/huffman_encode.rs:508,580,703`) so a stock `cargo build --release` automatically lights up TZCNT/BLSR/LZCNT on any CPU that supports them — no `RUSTFLAGS` needed for the AC-encoding inner loop. `target-cpu=native` still wins because it unlocks BMI2 PEXT/PDEP and FMA in code paths the runtime dispatch does not yet cover (FDCT scalar fallback, scalar quantization tail), so the recommendation stands for the last few percent: `RUSTFLAGS="-C target-cpu=native"` (best) or `-C target-feature=+bmi1,+lzcnt,+bmi2,+fma`. Pre-P4-8 the stock baseline trailed C by 5–10 pp at 1080p; that gap is now < 2 pp on a Haswell-class CPU.

## Quick Start

```toml
[dependencies]
libjpeg-turbo-rs = "0.8"

# Optional: enable PNG support for tj3LoadImage8 / tj3SaveImage8
# libjpeg-turbo-rs = { version = "0.8", features = ["png"] }
```

### Build flags (x86_64 only)

For x86_64 production builds, set:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
# or, for a portable v3 baseline:
RUSTFLAGS="-C target-feature=+bmi1,+lzcnt,+bmi2,+fma" cargo build --release
```

This unlocks BMI1 / LZCNT / BMI2 / FMA in the encoder's scalar
bitmap-iteration hot path, which the C reference's NASM SIMD already
embeds. Without these flags `cargo build --release` defaults to the
SSE2-only `x86_64-v1` baseline and the encoder trails C by 5–10 pp at
1080p; with them, Rust beats C in every encode benchmark in the
Performance section above. aarch64 / NEON builds are unaffected.

## Feature flags, MSRV, platforms

**MSRV: 1.87** (declared as `rust-version`; older compilers are not
tested).

| Target | SIMD | Notes |
| --- | --- | --- |
| `aarch64` (Linux/macOS) | NEON | compile-time selection, CI-tested |
| `x86_64` (Linux/macOS/Windows) | AVX2/SSE2 | runtime CPUID dispatch (`std`), CI-tested incl. no-AVX2 emulation |
| `wasm32` (browser/WASI) | SIMD128 | compile-time `target_feature` — see the wasm crate README |
| RISC-V / POWER / s390x / 32-bit ARM | scalar | works, unoptimized ([#359](https://github.com/developer0hye/libjpeg-turbo-rs/issues/359)) |
| `thumbv7em` (bare metal) | scalar | `no_std + alloc`, CI-built; no NEON backend is registered for thumb targets |


| flag | default | effect |
|---|---|---|
| `std` | ✅ | `std::io` streaming API (`decompress_from_reader`, `compress_to_writer`), file-path helpers, PNG image I/O, runtime CPU-feature detection, and `std::io::Error` interop. |
| `simd` | ✅ | Architecture intrinsics: NEON (aarch64), SSE2/AVX2 (x86_64), SIMD128 (wasm32). |
| `png` | ❌ | PNG support for `tj3LoadImage8` / `tj3SaveImage8` (implies `std`). |

**`no_std` + `alloc`**: build with `--no-default-features` for the core
codec — headers, entropy decode, IDCT, upsample, colour convert, and
encode all work. `alloc` is required (the decoder allocates pixel and
coefficient buffers). Without `std` there is no CPUID probe, so SIMD
dispatches on compile-time `target_feature` only; pass `-C
target-feature=+neon` (or equivalent) to vectorise a bare-metal build.
CI builds the crate for `thumbv7em-none-eabihf` on every PR.

### Decompress

```rust
use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};

// Decode to RGB
let img = decompress(&jpeg_bytes)?;
println!("{}x{}", img.width, img.height);

// Decode to specific format
let img = decompress_to(&jpeg_bytes, PixelFormat::Rgba)?;

// Decode into a caller-owned, reusable buffer (no per-frame output allocation)
use libjpeg_turbo_rs::{decompress_into, output_buffer_size};
let size = output_buffer_size(&jpeg_bytes, PixelFormat::Rgb)?;
let mut out = vec![0u8; size];
let info = decompress_into(&jpeg_bytes, PixelFormat::Rgb, &mut out)?;
println!("{}x{} ({} bytes)", info.width, info.height, info.bytes_written);
```

### Compress

```rust
use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

let jpeg = compress(&rgb_pixels, width, height, PixelFormat::Rgb, 85, Subsampling::S420)?;
```

### Builder API

```rust
use libjpeg_turbo_rs::Encoder;

let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
    .quality(85)
    .subsampling(Subsampling::S420)
    .progressive(true)
    .optimize_huffman(true)
    .icc_profile(&icc_data)
    .xmp_data(&xmp_packet)   // APP1 XMP
    .iptc_data(&iptc_iim)    // APP13 Photoshop IRB
    .encode()?;
```

Every builder option composes with every other, on every colorspace and in
every mode — including CMYK, and `colorspace(Rgb)` with `progressive`,
`arithmetic` or `lossless`
([#313](https://github.com/developer0hye/libjpeg-turbo-rs/issues/313),
[#322](https://github.com/developer0hye/libjpeg-turbo-rs/issues/322),
[#343](https://github.com/developer0hye/libjpeg-turbo-rs/issues/343),
[#345](https://github.com/developer0hye/libjpeg-turbo-rs/issues/345)).

### Composing baseline options

`Encoder` covers the common cases; `CompressParams` is the baseline core
underneath it, and takes every option at once — on every pixel format, CMYK
included.

```rust
use libjpeg_turbo_rs::encode::pipeline::{compress_with_params, CompressParams};

let jpeg = compress_with_params(
    &CompressParams::new(&rgb_pixels, width, height, PixelFormat::Rgb, 85, Subsampling::S420)
        .dct_method(DctMethod::IsFast)
        .restart_interval(8)
        .custom_quant(&quant_tables)
        .custom_huffman(&dc_tables, &ac_tables),
)?;
```

### Lossless Transform

```rust
use libjpeg_turbo_rs::{transform, TransformOp};

let rotated = transform(&jpeg_bytes, TransformOp::Rot90)?;
```

`transform` preserves metadata (EXIF/ICC/COM markers) by default, matching
C TurboJPEG's `tjTransform`; use `transform_jpeg_with_options` with
`MarkerCopyMode::None` to strip markers.

Lossless transforms are coefficient-domain: they entropy-decode to DCT
coefficients, permute blocks, and entropy-encode — no pixels are ever
produced, so the pixel-path SIMD (IDCT / color convert / upsample) is not
involved and transform throughput rides on scalar codegen. Build with the
default release profile (`opt-level = 3`); a size-optimized profile
(`opt-level = "z"`) roughly halves transform throughput. `-C
target-cpu=native` buys only a further ~3% (measured on a 24 MP rot90,
Zen 4; issue #308 has the full numbers).

### EXIF Orientation (load a phone photo the right way up)

Nearly every camera JPEG carries an EXIF orientation tag. Read it from
the header alone — no pixel decode — and apply it in whichever domain
fits (issue #391):

```rust
use libjpeg_turbo_rs::{decompress, Decoder, TransformOp};

// Probe without decoding pixels (None when the JPEG carries no EXIF):
let orientation: Option<u8> = Decoder::new(&jpeg_bytes)?.exif_orientation();

// Pixel domain — decode, then reorient in one call:
let upright = decompress(&jpeg_bytes)?.apply_orientation();

// DCT domain — rewrite the JPEG losslessly instead (skip the no-op
// re-encode for upright/untagged images). Strip the markers: transforms
// copy them by default, and a stale orientation tag on already-rotated
// pixels would make EXIF-aware viewers rotate twice. Note lossless
// transforms cannot fully reorient partial edge blocks when dimensions
// are not iMCU-aligned (see TransformOp::from_exif_orientation docs) —
// the pixel-domain path above is exact at any size.
use libjpeg_turbo_rs::{MarkerCopyMode, TransformOptions};
if let Some(op) = orientation.and_then(TransformOp::from_exif_orientation) {
    if op != TransformOp::None {
        let upright_jpeg = libjpeg_turbo_rs::transform_jpeg_with_options(
            &jpeg_bytes,
            &TransformOptions { op, copy_markers: MarkerCopyMode::None, ..Default::default() },
        )?;
    }
}
```

### Scanline-Level I/O

```rust
use libjpeg_turbo_rs::ScanlineDecoder;

let mut decoder = ScanlineDecoder::new(&jpeg_bytes)?;
let height = decoder.header().height as usize;
let width = decoder.header().width as usize;
let mut buf = vec![0u8; width * 3]; // RGB row buffer
while decoder.output_scanline() < height {
    decoder.read_scanline(&mut buf)?;
    // process buf...
}
let img = decoder.finish()?;
```

## Features

### Codec Support

| Feature | Encode | Decode |
|---------|--------|--------|
| Baseline DCT (Huffman) | yes | yes |
| Progressive DCT | yes | yes |
| Arithmetic coding | yes | yes |
| Lossless JPEG | yes | yes |
| 8/12/16-bit precision | yes | yes |
| Optimized Huffman tables | yes | - |

### Pixel Formats

Grayscale, RGB, BGR, RGBA, BGRA, ARGB, ABGR, RGBX, BGRX, XRGB, XBGR, CMYK, RGB565

### Chroma Subsampling

4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, 4:4:1

### SIMD

| Platform | Backend | Decode | Encode |
|----------|---------|--------|--------|
| aarch64 | NEON | IDCT, color convert, upsample, dequantize | FDCT, color convert, quantize+zigzag, downsample, Huffman |
| x86_64 | SSE2 | IDCT, color convert, upsample | Huffman bitmap+sign-correction |
| x86_64 | AVX2 | IDCT, color convert, upsample, merged upsample+color | FDCT, color convert, quantize+zigzag, downsample (fused H2V1/H2V2) |

aarch64 has comprehensive SIMD across the full pipeline. x86_64 decode and encode are both fully accelerated; encode pairs SSE2 Huffman bitmap construction with AVX2 fused FDCT/quantize/color/downsample.

All SIMD routines have scalar fallbacks. SIMD is enabled by default via the `simd` feature flag.

### Additional Features

- Scaled IDCT (all 16 libjpeg factors: 2/1, 15/8, 7/4, ..., 1/2, 1/4, 1/8)
- Lossless spatial transforms (rotate, flip, transpose)
- DCT coefficient access (`read_coefficients` / `write_coefficients`)
- Metadata: JFIF, EXIF, ICC profile, XMP (read incl. Extended XMP reassembly; write is single-segment), IPTC (APP13 Photoshop IRB), Adobe APP14, comments
- YUV plane encode/decode (raw component data)
- Scanline-level streaming API
- Crop decoding (MCU-aligned)
- Color quantization with dithering
- Error recovery mode
- Custom Huffman/quantization tables
- Restart markers (DRI)
- Progress callbacks

## C ABI replacement tiers

Beyond the Rust crate, the workspace ships C ABI shims: a TurboJPEG 3
cdylib (`libturbojpeg.so.0`, ready for TJ3 consumers) and a classic
libjpeg v8 cdylib (`libjpeg.so.8`, ready for v8 consumers) — with
byte-exact stock `djpeg`/`cjpeg`/`jpegtran` parity gates. The full
T1–T4 tier framing, legacy-alias matrix, SONAME opt-ins, and threading
contract live in [`docs/ABI_COMPATIBILITY.md`](docs/ABI_COMPATIBILITY.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Acknowledgments

This software is based in part on the work of the Independent JPEG Group.

Algorithms and implementation techniques referenced from [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) (IJG License / Modified BSD License) and [zune-jpeg](https://github.com/etemesi254/zune-image).
