//! P4-137 criterion 5 (issue #476): the packed-YUV length that sizes a
//! `slice::from_raw_parts` must be computed with checked arithmetic.
//!
//! `tj3CompressFromYUV8` and `tj3DecodeYUV8` used to size that slice with
//! `total.saturating_add(stride * ph)` — an *unchecked* multiply inside a
//! saturating add. Saturation is the worst available failure mode here,
//! because the result went straight into `from_raw_parts`: an overflowing
//! geometry yielded `usize::MAX`, so the decoder received a slice claiming the
//! entire address space rather than a short one. That is immediate undefined
//! behaviour, not a truncated read.
//!
//! Both entry points now go through a checked helper bounded by `isize::MAX`
//! (`from_raw_parts`'s own documented precondition) and report the overflow as
//! the documented `-1`.
//!
//! Only `NULL`/non-positive dimension checks run before the guard, so these
//! calls do reach it. The source buffer is deliberately tiny: if the guard
//! ever regresses, the call constructs an enormous slice over 16 bytes and the
//! sanitizer legs report it. A passing run means the rejection happened
//! *before* any dereference.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::{tj3CompressFromYUV8, tj3DecodeYUV8, tj3Destroy, tj3Init, tj3Set};

const TJINIT_COMPRESS: c_int = 0;
const TJINIT_DECOMPRESS: c_int = 2;
const TJPARAM_SUBSAMP: c_int = 4;
/// `TJSAMP_444` — full-resolution planes, the layout whose packed size grows
/// fastest and so the one that overflows first.
const TJSAMP_444: c_int = 0;
const TJPF_RGB: c_int = 0;

/// A buffer far too small for the declared geometry. Its only job is to be a
/// valid, non-NULL address: the guard must reject before reading it.
const TINY: [u8; 16] = [0u8; 16];

/// `c_int::MAX` square at 4:4:4 needs three planes of ~2^62 bytes, so the
/// total exceeds `isize::MAX` on 64-bit and overflows `usize` outright on
/// 32-bit. Either way the helper must refuse.
fn overflowing_dimension() -> c_int {
    c_int::MAX
}

#[test]
fn tj3_compress_from_yuv8_rejects_packed_length_overflow() {
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    // SAFETY: `handle` is a live instance from `tj3Init` above.
    assert_eq!(
        unsafe { tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444) },
        0,
        "tj3Set(TJPARAM_SUBSAMP, TJSAMP_444)"
    );

    let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
    let mut jpeg_size: usize = 0;
    let dim: c_int = overflowing_dimension();

    // SAFETY: `handle` is live; `TINY` is a valid readable allocation for its
    // own 16 bytes. The declared dimensions are the point of the test — the
    // guard must reject them before treating `TINY` as that large.
    let rc: c_int = unsafe {
        tj3CompressFromYUV8(
            handle,
            TINY.as_ptr(),
            dim,
            4,
            dim,
            &mut jpeg_buf as *mut *mut u8,
            &mut jpeg_size as *mut usize,
        )
    };

    assert_eq!(rc, -1, "overflowing packed-YUV geometry must return -1");
    assert!(
        jpeg_buf.is_null(),
        "no output buffer should be allocated on the rejection path"
    );

    // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
    unsafe { tj3Destroy(handle) };
}

#[test]
fn tj3_decode_yuv8_rejects_packed_length_overflow() {
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");

    // SAFETY: `handle` is a live instance from `tj3Init` above.
    assert_eq!(
        unsafe { tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444) },
        0,
        "tj3Set(TJPARAM_SUBSAMP, TJSAMP_444)"
    );

    let mut dst: [u8; 16] = [0u8; 16];
    let dim: c_int = overflowing_dimension();

    // SAFETY: as above — both buffers are valid for their real 16 bytes, and
    // the declared geometry must be rejected before either is indexed.
    let rc: c_int = unsafe {
        tj3DecodeYUV8(
            handle,
            TINY.as_ptr(),
            4,
            dst.as_mut_ptr(),
            dim,
            0,
            dim,
            TJPF_RGB,
        )
    };

    assert_eq!(rc, -1, "overflowing packed-YUV geometry must return -1");

    // SAFETY: `handle` came from `tj3Init` and is destroyed exactly once.
    unsafe { tj3Destroy(handle) };
}
