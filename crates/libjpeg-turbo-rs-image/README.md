# libjpeg-turbo-rs-image

[`image`](https://crates.io/crates/image) crate backend powered by [`libjpeg-turbo-rs`](https://crates.io/crates/libjpeg-turbo-rs) — a fast pure-Rust JPEG codec with NEON/AVX2 SIMD acceleration.

## Usage

```toml
[dependencies]
libjpeg-turbo-rs-image = "0.1"
image = "0.25"
```

### Decoding

```rust
use libjpeg_turbo_rs_image::JpegDecoder;
use image::ImageDecoder;
use std::fs;

let data = fs::read("photo.jpg").unwrap();
let mut decoder = JpegDecoder::new(&data).unwrap();
let (width, height) = decoder.dimensions();
let mut buf = vec![0u8; decoder.total_bytes() as usize];
decoder.read_image(&mut buf).unwrap();
// buf now contains RGB or L8 pixels depending on the JPEG color space
```

### Encoding

```rust
use libjpeg_turbo_rs_image::JpegEncoder;
use image::{ExtendedColorType, ImageEncoder};

let pixels: Vec<u8> = vec![/* RGB pixels */];
let mut output: Vec<u8> = Vec::new();
JpegEncoder::new_with_quality(&mut output, 85)
    .write_image(&pixels, 640, 480, ExtendedColorType::Rgb8)
    .unwrap();
// output contains the compressed JPEG bytes
```

## Color type mapping

| JPEG source    | `image::ColorType` |
|----------------|--------------------|
| Grayscale (1c) | `L8`               |
| YCbCr / RGB    | `Rgb8`             |

For other pixel formats (BGR, BGRA, CMYK, etc.) use `JpegDecoder::new_with_format()`.

## License

MIT OR Apache-2.0
