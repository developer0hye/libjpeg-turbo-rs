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
pub mod bufsize;
pub mod compress;
pub mod convert;
pub mod decompress;
pub mod header;
pub mod imageio;
pub mod legacy;
pub mod precision;
pub mod tj3;
pub mod transform;
pub mod yuv;

// Re-export the `extern "C"` symbols at the crate root for discoverability.
pub use alloc::{tj3Alloc, tj3Free};
pub use bufsize::{
    tj3GetScalingFactors, tj3JPEGBufSize, tj3YUVBufSize, tj3YUVPlaneHeight, tj3YUVPlaneSize,
    tj3YUVPlaneWidth,
};
pub use compress::tj3Compress8;
pub use decompress::tj3Decompress8;
pub use header::{
    tj3DecompressHeader, tj3SetCroppingRegion, tj3SetScalingFactor, TjRegion, TjScalingFactor,
};
pub use imageio::{
    tj3LoadImage12, tj3LoadImage16, tj3LoadImage8, tj3SaveImage12, tj3SaveImage16, tj3SaveImage8,
};
pub use precision::{tj3Compress12, tj3Compress16, tj3Decompress12, tj3Decompress16};
pub use tj3::{
    tj3Destroy, tj3Get, tj3GetErrorCode, tj3GetErrorStr, tj3Init, tj3InitVersion, tj3Set,
};
pub use transform::{tj3Transform, TjTransform};
pub use yuv::{
    tj3CompressFromYUV8, tj3CompressFromYUVPlanes8, tj3DecodeYUV8, tj3DecodeYUVPlanes8,
    tj3DecompressToYUV8, tj3DecompressToYUVPlanes8, tj3EncodeYUV8, tj3EncodeYUVPlanes8,
};

// Legacy TJ1/TJ2 aliases — thin wrappers around the TJ3 surface above.
pub use legacy::{
    tjBufSize, tjBufSizeYUV, tjBufSizeYUV2, tjCompress2, tjDecodeYUV, tjDecompress2,
    tjDecompressHeader3, tjDestroy, tjEncodeYUV3, tjGetErrorStr2, tjInitCompress, tjInitDecompress,
    tjInitTransform, tjLoadImage, tjPlaneHeight, tjPlaneSizeYUV, tjPlaneWidth, tjSaveImage,
    tjTransform, TJBUFSIZE, TJBUFSIZEYUV,
};
