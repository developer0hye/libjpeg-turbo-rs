//! Pure-Rust reimplementation of [libjpeg-turbo] with NEON / SSE2 /
//! AVX2 / WASM-SIMD128 acceleration — no C dependencies, no unsafe FFI.
//!
//! The one-shot functions cover most uses: [`decompress`] /
//! [`decompress_to`] / [`decompress_into`] for decoding, [`compress`]
//! and the [`Encoder`] builder for encoding, [`transform()`] for
//! lossless (DCT-domain) rotation and flips. The configurable
//! [`Decoder`] adds scaling, cropping, output formats, and resource
//! limits.
//!
//! [libjpeg-turbo]: https://github.com/libjpeg-turbo/libjpeg-turbo
//!
//! # Quickstart
//!
//! Decode a JPEG to RGB pixels:
//!
//! ```
//! use libjpeg_turbo_rs::{decompress, PixelFormat, Subsampling};
//! # let jpeg_bytes = libjpeg_turbo_rs::compress(
//! #     &[128; 16 * 16 * 3], 16, 16, PixelFormat::Rgb, 90, Subsampling::S420,
//! # ).unwrap();
//! let image = decompress(&jpeg_bytes)?;
//! assert_eq!(image.pixel_format, PixelFormat::Rgb);
//! assert_eq!(image.data.len(), image.width * image.height * 3);
//! # Ok::<(), libjpeg_turbo_rs::JpegError>(())
//! ```
//!
//! Encode RGB pixels to a JPEG:
//!
//! ```
//! use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};
//!
//! let (width, height) = (32, 24);
//! let rgb = vec![200u8; width * height * 3];
//! let jpeg = compress(&rgb, width, height, PixelFormat::Rgb, 85, Subsampling::S420)?;
//! assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "SOI marker");
//! # Ok::<(), libjpeg_turbo_rs::JpegError>(())
//! ```
//!
//! Probe a header without decoding any pixels — [`probe`] parses
//! markers only (multi-scan streams are walked byte-wise to find scan
//! boundaries, so worst-case cost is linear in the compressed size,
//! far below a decode):
//!
//! ```
//! # let jpeg_bytes = libjpeg_turbo_rs::compress(
//! #     &[0; 64 * 48 * 3], 64, 48,
//! #     libjpeg_turbo_rs::PixelFormat::Rgb, 90,
//! #     libjpeg_turbo_rs::Subsampling::S444,
//! # ).unwrap();
//! let info = libjpeg_turbo_rs::probe(&jpeg_bytes)?;
//! assert_eq!((info.width, info.height), (64, 48));
//! assert_eq!(info.exif_orientation, None); // no EXIF in this fixture
//! # Ok::<(), libjpeg_turbo_rs::JpegError>(())
//! ```
//!
//! # Canonical API map
//!
//! Rustdoc renders every export flat, so here is which entry point to
//! reach for first (issue #386):
//!
//! | Task | Canonical path |
//! |---|---|
//! | Decode, defaults fine | [`decompress`] (or [`decompress_to`] / [`decompress_into`]) |
//! | Decode, configured | [`Decoder`] — chainable `with_*`, or imperative `set_*` |
//! | Probe header/metadata | [`probe`] (one call), [`Decoder::new`] + accessors for more |
//! | Encode, defaults fine | [`compress`] |
//! | Encode, configured | [`Encoder`] builder — quality, subsampling, progressive, markers |
//! | Lossless transform | [`transform()`] — DCT-domain rotate/flip/crop, no re-encode |
//! | Row streaming | [`ScanlineDecoder`] / [`ScanlineEncoder`], `stream::*` for readers |
//! | 12/16-bit & lossless | [`precision`] module |
//!
//! **Specialised `compress_*` entry points.** The remaining
//! free-function encode variants ([`compress_optimized`],
//! [`compress_progressive`], [`compress_arithmetic`],
//! [`compress_arithmetic_progressive`], the `compress_lossless*`
//! family, [`compress_with_metadata`], …) predate the [`Encoder`]
//! builder, which can express every one of them —
//! e.g. `Encoder::new(&px, w, h, fmt).progressive(true)` replaces
//! [`compress_progressive`]. [`compress_into`] is the exception: it
//! fills a caller-owned buffer, and [`Encoder::encode`] always returns
//! a fresh `Vec`. They stay for source compatibility and
//! one-line convenience; new code should prefer [`Encoder`] once more
//! than one knob is involved. **Deprecation intent:** these variants
//! are candidates for `#[deprecated]` at the next semver-major once
//! [`Encoder`] has shipped a full release cycle without gaps; none are
//! removed before that.
//!
//! # Feature flags
//!
//! - **`std`** *(default)*: `std::io` streaming API, file-path helpers,
//!   runtime CPU-feature detection, and std-backed error types.
//! - **`no_std` + `alloc`**: disable default features for the core
//!   codec (headers, entropy decode, IDCT, upsample, colour convert,
//!   encode). SIMD then dispatches on compile-time `target_feature`
//!   only — there is no CPUID probe without `std` (issue #356).
//! - **`simd`** *(default)*: architecture intrinsics (NEON / SSE2 /
//!   AVX2 / WASM SIMD128).
//! - **`png`**: PNG helpers for `tj3LoadImage8` / `tj3SaveImage8`
//!   (implies `std`).
//!
//! Performance against C libjpeg-turbo and zune-jpeg, replacement-tier
//! status for the C ABI shims, and build-flag guidance live in the
//! [repository README](https://github.com/developer0hye/libjpeg-turbo-rs).
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Safety posture (issue #389): every unsafe operation inside an unsafe fn
// must be acknowledged with its own block + SAFETY reasoning, matching
// the capi crate's existing policy.
#![deny(unsafe_op_in_unsafe_fn)]

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::vec::Vec;
extern crate alloc;

pub mod api;
pub mod common;
pub mod decode;
pub mod encode;
// The SIMD backends predate the crate-level lint and carry ~780 bare
// unsafe operations inside their unsafe fns; wrapping each with its own
// block + SAFETY note is a mechanical-but-careful sweep tracked in
// LAST_MILE (filed with issue #389). Non-SIMD code gets the full deny
// immediately.
#[allow(unsafe_op_in_unsafe_fn)]
pub mod simd;
pub mod transform;

pub use api::abbreviated::{read_header, HeaderResult, TablesOnlyState};
#[doc(inline)]
pub use api::coefficient::{
    copy_critical_parameters, read_coefficients, transform_jpeg as transform,
    transform_jpeg_with_options, write_coefficients, write_coefficients_arithmetic,
    write_coefficients_optimized, write_coefficients_progressive,
    write_coefficients_progressive_arithmetic, ComponentCoefficients, EncoderComponentInfo,
    EncoderConfig, JpegCoefficients,
};
#[doc(inline)]
pub use api::encoder::{Encoder, HuffmanTableDef};

/// Produce a tables-only abbreviated JPEG datastream for the given encoder configuration.
///
/// Returns `SOI + DQT(s) + DHT(s) + EOI` with no image data. Equivalent to
/// libjpeg-turbo's `jpeg_write_tables()`. The stream can be parsed by `read_header()`
/// to preload tables for subsequent decoding of body-only streams.
pub fn jpeg_write_tables(encoder: &Encoder<'_>) -> Vec<u8> {
    encoder.write_tables()
}
pub use api::high_level::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_into,
    compress_lossless, compress_lossless_arithmetic, compress_lossless_extended,
    compress_optimized, compress_progressive, compress_with_metadata, decompress,
    decompress_cropped, decompress_into, decompress_lenient, decompress_to, output_buffer_size,
};
#[cfg(feature = "png")]
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "png")))]
pub use api::image_io::load_png_from_bytes;
#[cfg(all(feature = "png", any(not(target_arch = "wasm32"), target_os = "wasi")))]
#[cfg(feature = "std")]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "png", any(not(target_arch = "wasm32"), target_os = "wasi"))))
)]
pub use api::image_io::save_png;
#[cfg(any(not(target_arch = "wasm32"), target_os = "wasi"))]
#[cfg(feature = "std")]
#[cfg_attr(
    docsrs,
    doc(cfg(all(feature = "std", any(not(target_arch = "wasm32"), target_os = "wasi"))))
)]
pub use api::image_io::{
    load_image, load_ppm_12bit, load_ppm_16bit, save_bmp, save_ppm, save_ppm_12bit, save_ppm_16bit,
};
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use api::image_io::{
    load_image_from_bytes, load_ppm_12bit_from_bytes, load_ppm_16bit_from_bytes, LoadedImage,
    LoadedImage12, LoadedImage16,
};
#[doc(inline)]
#[cfg(feature = "std")]
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
pub use api::incremental::{
    decompress_from_reader_incremental, decompress_from_reader_incremental_instrumented,
};
pub use api::precision::{
    compress_12bit, compress_16bit, decompress_12bit, decompress_16bit, read_scanlines_12,
    read_scanlines_16, write_scanlines_12, write_scanlines_16,
};
pub use api::quality::quality_scaling;
pub use api::quantize::requantize;
pub use api::raw_data::{compress_raw, decompress_raw, RawImage};
/// 12-bit raw planar encode/decode (YCbCr component planes at native resolution).
pub mod raw_data_12 {
    pub use crate::api::raw_data_12::{compress_raw_12, decompress_raw_12, RawImage12};
}
pub use api::raw_thumbnail::extract_embedded_jpeg;
pub use encode::marker_writer::MarkerStreamWriter;
/// Color quantization for 8-bit indexed/palette output.
pub mod quantize {
    pub use crate::api::quantize::{
        dequantize, quantize, requantize, DitherMode, QuantizeOptions, QuantizedImage,
    };
}
#[doc(inline)]
pub use api::progressive_output::ProgressiveDecoder;
#[doc(inline)]
pub use api::scanline::{ScanlineDecoder, ScanlineEncoder};
/// Streaming I/O functions for reading/writing JPEG via `std::io` traits and file paths.
#[cfg_attr(docsrs, doc(cfg(feature = "std")))]
#[cfg(feature = "std")]
pub use api::stream;
pub use common::bufsize::{
    calc_jpeg_dimensions, calc_output_dimensions, jpeg_buf_size, transform_buf_size, yuv_buf_size,
    yuv_plane_height, yuv_plane_size, yuv_plane_width,
};
#[doc(inline)]
pub use common::error::{DecodeWarning, JpegError, Result};
pub use common::jfif::extract_jfif_thumbnail;
pub use common::sample::Sample;
pub use common::traits::{DefaultErrorHandler, ErrorHandler, ProgressInfo, ProgressListener};
pub use common::types::*;
#[doc(inline)]
pub use decode::pipeline::{probe, Decoder, Image, ImageInfo, JpegInfo};
pub use decode::resync::{DefaultResyncStrategy, RestartResyncStrategy, ResyncAction};
#[doc(inline)]
pub use transform::{MarkerCopyMode, TransformOp, TransformOptions};

/// Whether this build of the codec has both the `simd` and `std` cargo
/// features — the feature wiring a repackaging crate must preserve.
///
/// This is a feature-flag predicate, not a claim about the current CPU
/// or target. What the two features buy is target-dependent: `simd`
/// compiles the vector kernels (NEON / SSE2 / AVX2 / WASM SIMD128 where
/// the target has them; other architectures run scalar either way), and
/// `std` additionally lets `x86_64` builds pick AVX2 vs SSE2 through
/// `std::arch::is_x86_feature_detected!` at runtime — without it a
/// stock x86-64-baseline build never takes the AVX2 paths.
///
/// Downstream crates that repackage this codec (e.g. the `image`
/// bridge) assert on this to catch a feature-wiring regression — issue
/// #381 shipped exactly that: the bridge depended on the codec with
/// `default-features = false`, silently turning its performance
/// headline off.
pub const fn simd_and_std_features_enabled() -> bool {
    cfg!(all(feature = "std", feature = "simd"))
}
/// 12-bit and 16-bit sample precision support.
pub mod precision {
    pub use crate::api::precision::{
        compress_12bit, compress_16bit, compress_lossless_arbitrary, decompress_12bit,
        decompress_16bit, decompress_lossless_arbitrary, read_scanlines_12, read_scanlines_16,
        write_scanlines_12, write_scanlines_16, Image12, Image16,
    };
}
/// TJ3-compatible handle/parameter API.
pub mod tj3 {
    pub use crate::api::tj3::{TjHandle, TjParam};
}
