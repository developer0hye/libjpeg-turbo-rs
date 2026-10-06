# libjpeg-turbo-rs-image

[`image`](https://crates.io/crates/image) crate backend powered by [`libjpeg-turbo-rs`](https://crates.io/crates/libjpeg-turbo-rs) — a fast pure-Rust JPEG codec with NEON/AVX2 SIMD acceleration.

## Usage

```toml
[dependencies]
libjpeg-turbo-rs-image = "0.1"
# default-features = false keeps image's other format codecs (and the
# AVIF encoder's advisory-carrying rav1e chain) out of your build graph.
# Add the formats you actually use, e.g. features = ["png"].
image = { version = "0.25", default-features = false }
```

### Which `image` entry points use this backend

Only the ones you construct explicitly. This crate does **not** register
itself with `image`: `image::open`, `ImageReader::decode`, `load_from_memory`
and `DynamicImage::save` keep using `image`'s built-in JPEG codec. Build a
`JpegDecoder` and pass it to `DynamicImage::from_decoder`, and pass a
`JpegEncoder` to `DynamicImage::write_with_encoder`.

### Decoding

```rust,ignore
// Inside a function returning Result<_, Box<dyn std::error::Error>>;
// examples/thumbnail_pipeline.rs is the compiled version.
use image::{DynamicImage, ImageDecoder, Limits};
use libjpeg_turbo_rs_image::JpegDecoder;

let data: Vec<u8> = std::fs::read("photo.jpg")?;
// was: image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&data))?
let mut decoder = JpegDecoder::from_vec(data)?; // headers only; no pixels decoded yet
decoder.set_limits(Limits::default())?;          // refused here, before any pixel buffer
let orientation = decoder.orientation()?;        // reported, never applied
let mut image = DynamicImage::from_decoder(decoder)?;
image.apply_orientation(orientation);            // rotates exactly once
```

`examples/thumbnail_pipeline.rs` is a complete decode → orient → resize →
encode migration with the changed lines marked.

### Encoding

```rust
use image::{ExtendedColorType, ImageEncoder};
use libjpeg_turbo_rs_image::JpegEncoder;

let pixels: Vec<u8> = vec![/* RGB pixels */];
let mut output: Vec<u8> = Vec::new();
JpegEncoder::new_with_quality(&mut output, 85)
    .write_image(&pixels, 640, 480, ExtendedColorType::Rgb8)
    .unwrap();
// output contains the compressed JPEG bytes
```

## Behaviour

| | |
|---|---|
| Construction | `JpegDecoder::new` copies the compressed stream once and parses headers; `from_vec` takes ownership and copies nothing. No pixel is decoded until `read_image`. |
| `read_image` | Decodes into your buffer, which must be exactly `total_bytes()` long. For 8-bit grayscale and YCbCr/RGB streams no second decoded image exists; CMYK/YCCK, 12-bit and lossless streams are still staged in a full-size buffer and copied. Working memory remains either way (component planes, and coefficients for progressive streams). |
| `set_limits` | `max_image_width`, `max_image_height` and `max_alloc` are checked against the header before any pixel allocation. `max_alloc` bounds the core's decode-memory *estimate*, which counts the output buffer; it is non-strict for the staged paths above. Refusals are `ImageError::Limits`. |
| Metadata | `icc_profile`, `exif_metadata`, `xmp_metadata`, `iptc_metadata` and `orientation` return what `image 0.25`'s built-in JPEG decoder returns for the same file — the *last* segment when one repeats, the standard XMP packet without Extended XMP, IPTC as the Photoshop resource block. `original_color_type()` reports `Cmyk8` for four-component streams, where the built-in decoder reports `Rgb8`. |
| Errors | Limit and allocation refusals → `Limits`; unsupported features → `Unsupported`; wrong buffer sizes → `Parameter`; I/O → `IoError`; everything else → `Decoding` / `Encoding`. |
| Corrupt data | An error by default (as C libjpeg-turbo with `-strict`), where `image`'s built-in decoder fills what it cannot decode. `JpegDecoder::set_lenient(true)` opts into filling. |

## Color type mapping

| JPEG source              | `color_type()` | `original_color_type()` |
|--------------------------|----------------|-------------------------|
| Grayscale (1 component)  | `L8`           | `L8`                    |
| YCbCr / RGB (3)          | `Rgb8`         | `Rgb8`                  |
| CMYK / YCCK (4)          | `Rgb8`         | `Cmyk8`                 |

`JpegDecoder::new_with_format` selects `Rgba8` (or forces `L8` / `Rgb8`).
Formats `image` has no color type for (BGR, BGRA, CMYK, ...) are refused with
`ImageError::Unsupported`; decode those with `libjpeg_turbo_rs::Decoder`.

## License

MIT OR Apache-2.0
