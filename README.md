# libjpeg-turbo-rs

Pure Rust reimplementation of [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) with NEON/AVX2 SIMD acceleration. No C dependencies, no unsafe FFI — just `cargo add`.

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

**x86_64**: Decoding beats C across most resolutions. Encoding (with `target-cpu=native`) beats C in every benchmark above by 2–7 %; the encoder runs SSE2 Huffman + AVX2 FDCT/quantize/color/downsample. Without `target-cpu=native` (i.e. SSE2-only baseline), the same encode matrix trails C by 5–10 pp at 1080p because LLVM cannot emit `TZCNT`/`LZCNT`/BMI2 for the scalar bitmap-iteration code without an explicit target feature; the C reference's NASM-authored hot paths embed those instructions directly into `libjpeg.so` regardless of consumer build flags. Recommendation: production builds set `RUSTFLAGS="-C target-cpu=native"` (best) or at minimum `-C target-feature=+bmi1,+lzcnt,+bmi2,+fma`.

## Quick Start

```toml
[dependencies]
libjpeg-turbo-rs = "0.6"

# Optional: enable PNG support for tj3LoadImage8 / tj3SaveImage8
# libjpeg-turbo-rs = { version = "0.6", features = ["png"] }
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

### Decompress

```rust
use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};

// Decode to RGB
let img = decompress(&jpeg_bytes)?;
println!("{}x{}", img.width, img.height);

// Decode to specific format
let img = decompress_to(&jpeg_bytes, PixelFormat::Rgba)?;
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
    .encode()?;
```

### Lossless Transform

```rust
use libjpeg_turbo_rs::{transform, TransformOp};

let rotated = transform(&jpeg_bytes, TransformOp::Rot90)?;
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
- Metadata: JFIF, EXIF, ICC profile, Adobe APP14, comments
- YUV plane encode/decode (raw component data)
- Scanline-level streaming API
- Crop decoding (MCU-aligned)
- Color quantization with dithering
- Error recovery mode
- Custom Huffman/quantization tables
- Restart markers (DRI)
- Progress callbacks

## Running sanitizers locally

Requires a nightly toolchain (`rustup install nightly`) and the `rust-src` component:

```bash
rustup component add rust-src --toolchain nightly
```

**AddressSanitizer** (detects heap overflows, use-after-free, stack overflows):

```bash
RUSTFLAGS="-Z sanitizer=address" \
LSAN_OPTIONS="suppressions=$(pwd)/lsan_suppressions.txt:detect_leaks=1" \
cargo +nightly test --workspace --lib \
  --target x86_64-unknown-linux-gnu \
  --no-fail-fast -- --test-threads=1
```

**UB checks** (detects signed integer overflow, invalid enum discriminant, misaligned pointer dereference):

```bash
RUSTFLAGS="-Z ub-checks=yes" \
cargo +nightly test --workspace --lib \
  --no-fail-fast -- --test-threads=1
```

Note: `rustc` does not implement `sanitizer=undefined`; `-Z ub-checks=yes` is the correct nightly knob for runtime UB detection.

Both jobs run on every PR via `.github/workflows/sanitizers.yml`. macOS is excluded because the NEON SIMD paths produce spurious cross-thread ASan shadow-map false positives under parallel test execution.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Acknowledgments

This software is based in part on the work of the Independent JPEG Group.

Algorithms and implementation techniques referenced from [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) (IJG License / Modified BSD License) and [zune-jpeg](https://github.com/etemesi254/zune-image).
