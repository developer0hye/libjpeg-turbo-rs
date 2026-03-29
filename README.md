# libjpeg-turbo-rs

Pure Rust reimplementation of [libjpeg-turbo](https://github.com/libjpeg-turbo/libjpeg-turbo) with NEON/AVX2 SIMD acceleration. No C dependencies, no unsafe FFI — just `cargo add`.

## Performance

### x86_64 (AVX2)

Benchmarked against C libjpeg-turbo 3.1.0 on x86_64 (AVX2), quality 75:

#### Decoding

| Image | Subsampling | Rust (us) | C (us) | Ratio |
|-------|-------------|-----------|--------|-------|
| 64x64 | 4:2:0 | 60 | 49 | 1.22x |
| 320x240 | 4:2:0 | 797 | 774 | 1.03x |
| 640x480 | 4:2:0 | 943 | 889 | 1.06x |
| 640x480 | 4:2:2 | 3,409 | 3,393 | **1.00x** |
| 640x480 | 4:4:4 | 4,969 | 5,165 | **0.96x** |
| 1280x720 | 4:2:0 | 9,150 | 9,153 | **1.00x** |
| 1920x1080 | 4:2:0 | 20,679 | 21,058 | **0.98x** |
| 1920x1080 | 4:2:2 | 26,281 | 26,334 | **1.00x** |
| 1920x1080 | 4:4:4 | 38,813 | 39,308 | **0.99x** |
| 2560x1440 | 4:2:0 | 36,178 | 37,373 | **0.97x** |
| 3840x2160 | 4:2:0 | 81,636 | 82,030 | **1.00x** |

#### Encoding

| Image | Subsampling | Rust (us) | C (us) | Ratio |
|-------|-------------|-----------|--------|-------|
| 320x240 | 4:2:0 | 458 | 426 | 1.07x |
| 320x240 | 4:2:2 | 593 | 545 | 1.09x |
| 320x240 | 4:4:4 | 822 | 796 | 1.03x |
| 640x480 | 4:2:2 | 2,009 | 1,832 | 1.10x |
| 640x480 | 4:4:4 | 2,580 | 2,740 | **0.94x** |
| 1920x1080 | 4:2:0 | 12,707 | 11,203 | 1.13x |
| 1920x1080 | 4:2:2 | 15,985 | 14,210 | 1.12x |
| 1920x1080 | 4:4:4 | 22,305 | 21,067 | 1.06x |

### aarch64 (NEON)

Benchmarked against C libjpeg-turbo 3.1.0 on Apple Silicon (aarch64 NEON), quality 75:

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

Decoding matches or beats C on both platforms at larger resolutions. Encoding is near-parity on aarch64; x86_64 encoding has room for further SIMD optimization (Huffman coding). See [`docs/ENCODING_PERFORMANCE.md`](docs/ENCODING_PERFORMANCE.md) for full results.

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
