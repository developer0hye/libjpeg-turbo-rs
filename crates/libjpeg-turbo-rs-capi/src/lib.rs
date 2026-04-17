//! C ABI shim for `libjpeg-turbo-rs`.
//!
//! Exposes libjpeg-turbo-compatible `extern "C"` symbols so that existing
//! C consumers (`djpeg`, `cjpeg`, `jpegtran`, Pillow, ImageMagick, …) can
//! link against this crate in place of the stock `libjpeg.so.62` /
//! `libturbojpeg.so.0`.
//!
//! The public Rust surface re-exports the underlying pure-Rust
//! `libjpeg_turbo_rs` crate under `inner` so that downstream Rust
//! consumers can mix-and-match the C and Rust APIs.

#![deny(unsafe_op_in_unsafe_fn)]
// `extern "C"` shim functions intentionally accept raw pointers from C and
// dereference them after validating against NULL. Marking each function
// `unsafe fn` would change the ABI-visible symbol name and break the
// drop-in contract, so we silence clippy at the crate level here.
#![allow(clippy::not_unsafe_ptr_arg_deref)]
// Exported symbols must match the C case (`tj3Init`, `tj3Destroy`, ...),
// so we disable the snake_case lint at the crate root.
#![allow(non_snake_case)]

pub use libjpeg_turbo_rs as inner;

pub mod alloc;
pub mod compress;
pub mod convert;
pub mod decompress;
pub mod header;
pub mod tj3;

// Re-export the `extern "C"` symbols at the crate root for discoverability.
pub use alloc::{tj3Alloc, tj3Free};
pub use compress::tj3Compress8;
pub use decompress::tj3Decompress8;
pub use header::{
    tj3DecompressHeader, tj3SetCroppingRegion, tj3SetScalingFactor, TjRegion, TjScalingFactor,
};
pub use tj3::{tj3Destroy, tj3Get, tj3GetErrorCode, tj3GetErrorStr, tj3Init, tj3Set};
