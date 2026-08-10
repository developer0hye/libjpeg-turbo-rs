//! Behavioural canaries for the P4-139 span guards (issue #478).
//!
//! The error-code table in `capi_classic_error_codes.rs` proves
//! `JERR_WIDTH_OVERFLOW`'s *number and message* match upstream. It does not
//! prove any code path reaches it. These tests exercise the guards themselves,
//! so wrong routing — the easy mistake, since three different upstream codes
//! are in play — fails here rather than shipping.
//!
//! **Coverage, stated honestly.** On a 64-bit host only the `JDIMENSION` guard
//! is reachable through the public API: `image_width` is a `u32` and
//! `input_components` a small `c_int`, so their product cannot overflow
//! `usize`. The `usize`-overflow and `isize::MAX` arms of the same guards —
//! and `tj3Compress8`'s source-span guard — are 32-bit-only and **are not
//! exercised anywhere today**.
//!
//! An earlier version of this comment claimed the `armv7` and `wasm32-wasip1`
//! legs covered them. They do not: `armv7.yml` runs the root crate's `--lib`
//! and `no_std_dispatch` only, and the WASI leg selects the root workspace
//! member. Neither builds this crate's C-ABI tests. Getting a 32-bit C-ABI leg
//! is real infrastructure work and is recorded under P4-139 criterion 5 rather
//! than asserted here.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::{
    jpeg_CreateCompress, jpeg_destroy_compress, jpeg_set_defaults, jpeg_start_compress,
    jpeg_std_error, JpegCompressPublic, JpegErrorMgr,
};

/// The version and struct size the create guard expects (P4-110).
const JPEG_LIB_VERSION: c_int = 80;

/// `jcmaster.c:190-194` — "Image too wide for this implementation".
const JERR_WIDTH_OVERFLOW: c_int = 72;

/// Sentinel written into `msg_code` before each call, so "no error raised" is
/// distinguishable from "raised code 0".
const NO_ERROR: c_int = -1;

/// A no-op `error_exit`.
///
/// It deliberately *returns*, where a real consumer would `longjmp`. Panicking
/// would cross an `extern "C"` frame and Rust aborts on unwind through one —
/// which is how the first version of this test died. Returning is safe for what
/// is observed here: every guard under test calls `error_exit` and then
/// immediately returns, so nothing runs on the rejected geometry.
///
/// The raised code is read from the error manager afterwards rather than
/// captured into a global. That is not a style preference: the first version
/// used a `static mut`, and because `cargo test` runs these two tests in
/// parallel in one process, the second read the first's value and failed
/// claiming 64x64 RGB tripped the width guard. The state belongs per-instance.
unsafe extern "C" fn ignore_error_exit(_cinfo: *mut c_void) {}

/// `image_width * input_components` must be rejected when it cannot be
/// represented as `JDIMENSION`, with upstream's `JERR_WIDTH_OVERFLOW`.
///
/// 65500 x 65573 is the shape from `jcmaster.c`: both factors are individually
/// legal, and the product (4,295,031,500) exceeds `u32::MAX` while fitting
/// `usize` on a 64-bit host. A guard written as a bare `checked_mul` — which
/// is what this test caught during review — passes it straight through.
#[test]
fn scanline_width_beyond_jdimension_raises_width_overflow() {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });

    // SAFETY: both are live, correctly-aligned allocations owned here, and the
    // declared struct size matches the type passed.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = err_ptr;
        jpeg_CreateCompress(
            &mut *cinfo as *mut JpegCompressPublic as *mut c_void,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegCompressPublic>(),
        );
        cinfo.err = err_ptr;
        (*err_ptr).error_exit = Some(ignore_error_exit);
        (*err_ptr).msg_code = NO_ERROR;

        jpeg_set_defaults(&mut *cinfo as *mut JpegCompressPublic as *mut c_void);
        cinfo.image_width = 65_500;
        cinfo.image_height = 8;
        cinfo.input_components = 65_573;

        jpeg_start_compress(&mut *cinfo as *mut JpegCompressPublic as *mut c_void, 1);

        let raised: c_int = (*err_ptr).msg_code;
        assert_eq!(
            raised, JERR_WIDTH_OVERFLOW,
            "a row width past JDIMENSION must raise JERR_WIDTH_OVERFLOW, not \
             JERR_IMAGE_TOO_BIG and not silence"
        );

        jpeg_destroy_compress(&mut *cinfo as *mut JpegCompressPublic as *mut c_void);
    }
}

/// The companion: ordinary geometry must still start cleanly. Without this,
/// the guard above could be "satisfied" by rejecting everything.
#[test]
fn ordinary_geometry_still_starts_compress() {
    let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
    let mut cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });

    // SAFETY: as above.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
        cinfo.err = err_ptr;
        jpeg_CreateCompress(
            &mut *cinfo as *mut JpegCompressPublic as *mut c_void,
            JPEG_LIB_VERSION,
            std::mem::size_of::<JpegCompressPublic>(),
        );
        cinfo.err = err_ptr;
        (*err_ptr).error_exit = Some(ignore_error_exit);
        (*err_ptr).msg_code = NO_ERROR;

        jpeg_set_defaults(&mut *cinfo as *mut JpegCompressPublic as *mut c_void);
        cinfo.image_width = 64;
        cinfo.image_height = 64;
        cinfo.input_components = 3;

        jpeg_start_compress(&mut *cinfo as *mut JpegCompressPublic as *mut c_void, 1);

        let raised: c_int = (*err_ptr).msg_code;
        assert_eq!(
            raised, NO_ERROR,
            "64x64 RGB must not trip any span guard (raised {raised})"
        );

        jpeg_destroy_compress(&mut *cinfo as *mut JpegCompressPublic as *mut c_void);
    }
}
