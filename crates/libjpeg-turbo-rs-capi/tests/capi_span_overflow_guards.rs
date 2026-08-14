//! Behavioural canaries for the P4-139 span guards (issue #478).
//!
//! The error-code table in `capi_classic_error_codes.rs` proves
//! `JERR_WIDTH_OVERFLOW`'s *number and message* match upstream. It does not
//! prove any code path reaches it. These tests exercise the guards themselves,
//! so wrong routing — the easy mistake, since three different upstream codes
//! are in play — fails here rather than shipping.
//!
//! **Coverage, stated honestly.** Of the *classic-ABI* guards, only the
//! `JDIMENSION` one is reachable on a 64-bit host: `image_width` is a `u32` and
//! `input_components` a small `c_int`, so their product cannot overflow
//! `usize`. Those guards' `usize`-overflow and `isize::MAX` arms are
//! 32-bit-only, and since 2026-08-14 exactly one of the two is exercised.
//! `armv7.yml` runs this file under qemu-arm, where `65500 * 65573` is
//! 4,295,031,500 and so leaves a 32-bit `usize`: the same test below that pins
//! the `JDIMENSION` bound on a 64-bit host takes `checked_samples_per_row`'s
//! *overflow* arm there. `checked_staging_span`'s `isize::MAX` arm is still
//! **not exercised anywhere** — the width test returns before reaching it and
//! the companion is 64x64.
//!
//! The *TJ3* source-span guards are a different case, which this comment used
//! to lump in with the above. Their width and height are both
//! caller-supplied `c_int`s, so their `isize::MAX` arm is reachable on 64-bit:
//! `capi_layout_adoption.rs`'s `source_span_bounds` module exercises it for
//! `tj3Compress12`/`tj3Compress16`, where the two-byte element puts the byte
//! span past the bound, and for `tj3Compress8` — `c_int::MAX` square at
//! `TJPF_RGBA` spans 18446744056529682436 bytes, which fits `usize` and
//! exceeds `isize::MAX`. `tj3Compress8`'s guard predates that file (P4-137
//! wrote it); what it lacked was a test, because this comment said the arm
//! was unreachable.
//!
//! How the 32-bit half got covered, since an earlier version of this comment
//! claimed it already was. Until P4-139 chunk 1 neither 32-bit leg built this
//! crate's C-ABI tests: `armv7.yml` ran the root crate's `--lib` and
//! `no_std_dispatch` only, and the WASI leg selects the root workspace member.
//! Chunk 1 made it possible — gating the encode ABI-offset block on
//! `target_pointer_width = "64"` removed the assertion that made this crate
//! uncompilable on ILP32 — and criterion 5's CI leg landed on 2026-08-14:
//! `armv7.yml` gained a second qemu-arm step running `-p
//! libjpeg-turbo-rs-capi --lib --test capi_layout_adoption --test
//! capi_span_overflow_guards`. The WASI leg still selects the root workspace
//! member and still does not build this file.

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
