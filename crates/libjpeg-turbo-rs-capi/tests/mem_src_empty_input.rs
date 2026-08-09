//! Issue #469 (P4-109): `jpeg_mem_src` must reject NULL / zero-length input
//! the way upstream does, rather than installing a source manager over it.
//!
//! Upstream `jdatasrc.c`:
//!
//! ```c
//! if (inbuffer == NULL || insize == 0)   /* Treat empty input as fatal error */
//!   ERREXIT(cinfo, JERR_INPUT_EMPTY);
//! ```
//!
//! This shim accepted both and installed a manager pointing at the empty
//! buffer, so the failure surfaced later as a decode error — or not at all —
//! instead of at the call C rejects.
//!
//! The assertion is on `error_exit` firing with `msg_code == JERR_INPUT_EMPTY`,
//! not merely on "something failed": the point of the contract is *which*
//! error a C consumer's handler sees.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::{
    jpeg_CreateDecompress, jpeg_destroy_decompress, jpeg_mem_src, jpeg_std_error, JpegErrorMgr,
};
use std::sync::atomic::{AtomicI32, Ordering};

/// `jerror.h` JMESSAGE ordinal for `JERR_INPUT_EMPTY`.
const JERR_INPUT_EMPTY: c_int = 43;

static LAST_MSG_CODE: AtomicI32 = AtomicI32::new(-1);

/// Replacement `error_exit` that records the code and returns.
///
/// libjpeg's contract is that `error_exit` never returns, and a real consumer
/// would `longjmp` out. A Rust test cannot: panicking across an `extern "C"`
/// boundary aborts the process rather than unwinding, which is exactly what
/// the first version of this test did. Returning is safe here because
/// `jpeg_mem_src` returns immediately after raising — the contract this test
/// exercises is *which code is raised*, not what a handler does next.
unsafe extern "C" fn recording_error_exit(cinfo: *mut c_void) {
    // Read through the real `JpegErrorMgr` rather than a hand-computed offset:
    // `cinfo`'s first field is `err`, and the struct is the crate's own
    // ABI-pinned mirror, so this cannot drift from the layout under test.
    let err: *mut JpegErrorMgr = unsafe { *(cinfo as *const *mut JpegErrorMgr) };
    LAST_MSG_CODE.store(unsafe { (*err).msg_code }, Ordering::SeqCst);
}

fn drive(buf: *const u8, size: std::os::raw::c_ulong) -> c_int {
    LAST_MSG_CODE.store(-1, Ordering::SeqCst);
    let mut cinfo: Vec<u8> = vec![0u8; 1024];
    let mut jerr: JpegErrorMgr = unsafe { std::mem::zeroed() };
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut jerr as *mut JpegErrorMgr);
        assert!(!errp.is_null(), "jpeg_std_error");
        (*errp).error_exit = Some(recording_error_exit);
        *(cinfo.as_mut_ptr() as *mut *mut JpegErrorMgr) = errp;
        jpeg_CreateDecompress(cinfo.as_mut_ptr() as *mut c_void, 80, cinfo.len());
        jpeg_mem_src(cinfo.as_mut_ptr() as *mut c_void, buf, size);
        jpeg_destroy_decompress(cinfo.as_mut_ptr() as *mut c_void);
    }
    LAST_MSG_CODE.load(Ordering::SeqCst)
}

#[test]
fn jpeg_mem_src_rejects_null_buffer_with_input_empty() {
    assert_eq!(
        drive(std::ptr::null(), 16),
        JERR_INPUT_EMPTY,
        "a NULL buffer must raise JERR_INPUT_EMPTY, as jdatasrc.c does"
    );
}

#[test]
fn jpeg_mem_src_rejects_zero_length_with_input_empty() {
    let data: [u8; 4] = [0xFF, 0xD8, 0xFF, 0xD9];
    assert_eq!(
        drive(data.as_ptr(), 0),
        JERR_INPUT_EMPTY,
        "a zero-length input must raise JERR_INPUT_EMPTY even with a valid pointer"
    );
}
