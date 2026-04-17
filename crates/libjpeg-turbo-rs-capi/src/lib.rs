//! C ABI shim for `libjpeg-turbo-rs`.
//!
//! Exposes libjpeg-turbo-compatible `extern "C"` symbols so that existing
//! C consumers (`djpeg`, `cjpeg`, `jpegtran`, Pillow, ImageMagick, …) can
//! link against this crate in place of the stock `libjpeg.so.62` /
//! `libturbojpeg.so.0`.
//!
//! This initial A1-1 scaffold is deliberately empty; subsequent subtasks
//! fill in the TurboJPEG 3 handle lifecycle, compress/decompress, transform,
//! YUV family, and classic `jpeglib.h` entry points.

#![deny(unsafe_op_in_unsafe_fn)]

// Public re-exports for downstream Rust consumers that want direct access
// to the wrapped library. The C ABI surface is all `extern "C" fn` and
// lives in sibling modules that will be added incrementally.
pub use libjpeg_turbo_rs as inner;
