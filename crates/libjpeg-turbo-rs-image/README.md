# libjpeg-turbo-rs-image

An [`image`](https://crates.io/crates/image) crate adapter powered by
[`libjpeg-turbo-rs`](https://crates.io/crates/libjpeg-turbo-rs), exposing JPEG
decode and encode through the `image` crate's `ImageDecoder` and
`ImageEncoder` traits.

Use this crate when your application or library already depends on `image`
traits but wants to select `libjpeg-turbo-rs` explicitly. Use the root codec
crate directly when you need its full pixel-format, metadata, transform,
high-precision, luminance/chrominance (YUV), scanline, or caller-owned-buffer
Application Programming Interface (API).

For the project-wide readiness and evaluation process, read
[`docs/ADOPTION_GUIDE.md`](../../docs/ADOPTION_GUIDE.md) and
[`docs/LAST_MILE.md`](../../docs/LAST_MILE.md).

## Why an explicit adapter?

The adapter keeps codec choice visible in your dependency graph and avoids
assuming that changing a transitive `image` feature automatically replaces its
JPEG implementation. It also provides a stable place to test color mapping,
trait behavior, and bridge overhead separately from the core codec.

A core-codec benchmark is not an end-to-end adapter benchmark. Measure the
actual trait path and surrounding image representation your application uses.

## Installation

```toml
[dependencies]
libjpeg-turbo-rs-image = "0.1"

# Keep unrelated image formats out of the dependency graph, then add only the
# formats your application uses.
image = { version = "0.25", default-features = false }
```

This bridge follows the Minimum Supported Rust Version (MSRV) required by its
`image` dependency, which may be higher than the root codec's MSRV.

## Decoding

```rust
use image::ImageDecoder;
use libjpeg_turbo_rs_image::JpegDecoder;

let mut decoder = JpegDecoder::new(&jpeg_bytes)?;
let (width, height) = decoder.dimensions();
let mut pixels = vec![0_u8; decoder.total_bytes() as usize];
decoder.read_image(&mut pixels)?;

println!("decoded {width}x{height}");
```

The default output is RGB for color JPEGs and L8 for grayscale JPEGs.
Use `JpegDecoder::new_with_format()` when you need one of the additional packed
formats exposed by the bridge.

## Encoding

```rust
use image::{ExtendedColorType, ImageEncoder};
use libjpeg_turbo_rs_image::JpegEncoder;

let pixels: Vec<u8> = vec![/* RGB pixels */];
let mut output = Vec::new();

JpegEncoder::new_with_quality(&mut output, 85)
    .write_image(&pixels, 640, 480, ExtendedColorType::Rgb8)?;
```

Set quality explicitly so codec behavior remains visible in application code.
For advanced subsampling, progressive, arithmetic, lossless, metadata, custom
tables, or transform controls, use the root `libjpeg-turbo-rs::Encoder` and
related APIs.

## Color mapping

| JPEG source or requested output | `image::ColorType` / behavior |
| --- | --- |
| Grayscale, one component | `L8` |
| YCbCr or RGB color source | `Rgb8` by default |
| Other packed formats such as BGR/BGRA/CMYK | use `JpegDecoder::new_with_format()` and the bridge-specific API |

Do not assume that a four-byte packed format has alpha semantics merely because
it has four bytes per pixel. Verify the selected format's alpha/padding and
channel order.

## When to use the root crate instead

Prefer `libjpeg-turbo-rs` directly when you need:

- reusable caller-owned output buffers with no per-frame output allocation;
- scanline or `std::io` streaming beyond the adapter contract;
- coefficient-domain rotate/flip/transpose/crop;
- Exchangeable Image File Format (EXIF), International Color Consortium (ICC),
  XMP, IPTC, or marker-level control;
- CMYK/YCCK policy beyond the adapter's color mapping;
- 12/16-bit precision, lossless JPEG, arithmetic coding, or custom scan scripts;
- raw YUV planes, custom Huffman/quantization tables, or advanced recovery;
- `no_std + alloc` or direct WebAssembly integration.

## Evaluation checklist

Before replacing an existing `image`-based JPEG path:

- [ ] Run the same representative corpus through both adapters.
- [ ] Compare dimensions, `ColorType`, row layout, channel order, and decoded
      pixels.
- [ ] Include grayscale, progressive, CMYK/YCCK, embedded color profiles,
      metadata, very small images, and malformed inputs relevant to your data.
- [ ] Benchmark the `ImageDecoder`/`ImageEncoder` calls end to end rather than
      quoting only a root-codec benchmark.
- [ ] Measure allocation and conversion overhead in the surrounding
      `DynamicImage` or application buffer path.
- [ ] Verify encoder quality and output color type explicitly.
- [ ] Keep the previous adapter/version available until rollback has been
      exercised.

The repository's dated adapter evidence is recorded under
[`experiments/`](../../experiments), while core feature and release readiness
remain in [`docs/FEATURE_PARITY.md`](../../docs/FEATURE_PARITY.md) and
[`docs/LAST_MILE.md`](../../docs/LAST_MILE.md).

## Scope and limitations

This crate is an adapter, not a promise to expose every root-codec capability
through `image` traits. Trait contracts intentionally have a smaller option
surface than the root builder APIs. New bridge behavior should be added only
with trait-level compatibility tests and a clear downstream use case.

The bridge inherits the root codec's safety and correctness status. It does not
promote the experimental classic C ABI and does not depend on a C JPEG codec.

## License

MIT OR Apache-2.0, matching the root codec. See
[`LICENSE-MIT`](../../LICENSE-MIT) and
[`LICENSE-APACHE`](../../LICENSE-APACHE).
