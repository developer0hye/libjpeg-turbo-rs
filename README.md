# libjpeg-turbo-rs

Pure Rust reimplementation of [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) with NEON/AVX2 SIMD acceleration. No C dependencies, no unsafe FFI — just `cargo add`.

## Performance

Benchmarked against C libjpeg-turbo 3.1.0 on Apple Silicon (aarch64 NEON), quality 75:

### Encoding (1920x1080)

| Subsampling | Rust (us) | C (us) | Ratio |
|-------------|-----------|--------|-------|
| 4:2:0 | 5274 | 5076 | 1.04x |
| 4:2:2 | 6472 | 6441 | 1.00x |
| 4:4:4 | 9633 | 9714 | **0.99x** |

### Decoding (1920x1080)

| Subsampling | Rust (us) | C (us) | Ratio |
|-------------|-----------|--------|-------|
| 4:2:0 | 2559 | 2592 | **0.99x** |
| 4:2:2 | 2916 | 3020 | **0.97x** |
| 4:4:4 | 3750 | 3833 | **0.98x** |

Rust matches or beats C in most configurations. See [`docs/ENCODING_PERFORMANCE.md`](docs/ENCODING_PERFORMANCE.md) for full results.

## Quick Start

```toml
[dependencies]
libjpeg-turbo-rs = "0.1"
```

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
    .finish()?;
```

### Lossless Transform

```rust
use libjpeg_turbo_rs::{transform_jpeg, TransformOp, TransformOptions};

let rotated = transform_jpeg(&jpeg_bytes, TransformOp::Rot90, &TransformOptions::default())?;
```

### Scanline-Level I/O

```rust
use libjpeg_turbo_rs::ScanlineDecoder;

let mut decoder = ScanlineDecoder::new(&jpeg_bytes)?;
while decoder.output_scanline() < decoder.output_height() {
    let row = decoder.read_scanlines(1)?;
    // process row...
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

| Platform | Backend | Status |
|----------|---------|--------|
| aarch64 | NEON | IDCT, FDCT, color convert, (de)quantize, up/downsample, zigzag, Huffman |
| x86_64 | SSE2/AVX2 | IDCT, color convert (more coming) |

aarch64 NEON is fully optimized across the entire encode/decode pipeline. x86_64 AVX2/SSE2 optimization is planned to bring the same level of performance to Intel/AMD platforms.

All SIMD routines have scalar fallbacks. SIMD is enabled by default via the `simd` feature flag.

### Additional Features

- Scaled IDCT (1/2, 1/4, 1/8)
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

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Acknowledgments

This software is based in part on the work of the Independent JPEG Group.

Algorithms and implementation techniques referenced from [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) (IJG License / Modified BSD License) and [zune-jpeg](https://github.com/etemesi254/zune-image).
