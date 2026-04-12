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

#### Encoding

| Image | Subsampling | Rust (us) | C (us) | Ratio |
|-------|-------------|-----------|--------|-------|
| 320x240 | 4:2:0 | 436 | 401 | 1.09x |
| 320x240 | 4:2:2 | 537 | 527 | 1.02x |
| 320x240 | 4:4:4 | 800 | 787 | 1.02x |
| 640x480 | 4:2:2 | 1,818 | 1,711 | 1.06x |
| 640x480 | 4:4:4 | 2,622 | 2,524 | 1.04x |
| 1920x1080 | 4:2:0 | 11,836 | 10,442 | 1.13x |
| 1920x1080 | 4:2:2 | 14,573 | 13,123 | 1.11x |
| 1920x1080 | 4:4:4 | 21,839 | 20,076 | 1.09x |

### aarch64 (NEON)

Apple M1 Pro, C libjpeg-turbo 3.1.0, quality 75:

#### Decoding (1920x1080)

| Subsampling | Rust (us) | C (us) | Ratio |
|-------------|-----------|--------|-------|
| 4:2:0 | 2,559 | 2,592 | **0.99x** |
| 4:2:2 | 2,916 | 3,020 | **0.97x** |
| 4:4:4 | 3,750 | 3,833 | **0.98x** |

#### Encoding (1920x1080)

| Subsampling | Rust (us) | C (us) | Ratio |
|-------------|-----------|--------|-------|
| 4:2:0 | 5,274 | 5,076 | 1.04x |
| 4:2:2 | 6,472 | 6,441 | 1.00x |
| 4:4:4 | 9,633 | 9,714 | **0.99x** |

Decoding beats C on both platforms at most resolutions. Encoding is near-parity on aarch64; x86_64 encoding has room for further SIMD optimization (Huffman coding). See [`docs/ENCODING_PERFORMANCE.md`](docs/ENCODING_PERFORMANCE.md) for full results.

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
| x86_64 | SSE2/AVX2 | IDCT, FDCT, color convert (all pixel formats), quantize+zigzag, upsample, merged upsample+color |

Both platforms have comprehensive SIMD coverage across the encode/decode pipeline.

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
