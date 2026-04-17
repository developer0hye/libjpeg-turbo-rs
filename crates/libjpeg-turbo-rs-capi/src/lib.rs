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

pub mod tj3;

// Re-export the `extern "C"` symbols at the crate root so that tools
// linking the staticlib pull them in directly without having to name the
// sub-module. The `#[no_mangle]` attributes on the definitions themselves
// keep the exported symbol names intact.
pub use tj3::{tj3Destroy, tj3Get, tj3GetErrorCode, tj3GetErrorStr, tj3Init, tj3Set};
