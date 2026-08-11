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
    jpeg_CreateDecompress, jpeg_destroy_decompress, jpeg_mem_src, jpeg_std_error,
    JpegDecompressPublic, JpegErrorMgr,
};
/// `jerror.h` JMESSAGE ordinal for `JERR_INPUT_EMPTY`.
const JERR_INPUT_EMPTY: c_int = 43;

/// Written into `msg_code` before each run, so "nothing raised" is
/// distinguishable from "raised code 0".
const NO_ERROR: c_int = -1;

/// A no-op `error_exit`.
///
/// libjpeg's contract is that `error_exit` never returns, and a real consumer
/// would `longjmp` out. A Rust test cannot: panicking across an `extern "C"`
/// boundary aborts the process rather than unwinding, which is exactly what
/// the first version of this test did. Returning is safe here because
/// `jpeg_mem_src` returns immediately after raising — the contract this test
/// exercises is *which code is raised*, not what a handler does next.
///
/// The code is read afterwards from this run's own error manager. It used to
/// be captured into a process-global `AtomicI32` that `drive()` reset on entry
/// — and because `cargo test` runs this file's two tests in parallel, one
/// could reset the global between the other's raise and its read, so the
/// second saw `-1` and failed claiming no error was raised. Observed once in a
/// full-workspace run; the tests pass in isolation, which is what makes that
/// shape expensive to diagnose. Per-instance state removes the race entirely.
unsafe extern "C" fn ignore_error_exit(_cinfo: *mut c_void) {}

fn drive(buf: *const u8, size: std::os::raw::c_ulong) -> c_int {
    // P4-110: `structsize` is validated now, so this must be the real struct.
    // A `Vec<u8>` was also only byte-aligned, which the C ABI does not allow
    // for a `j_decompress_ptr`.
    let mut cinfo: std::mem::MaybeUninit<JpegDecompressPublic> = std::mem::MaybeUninit::zeroed();
    let mut jerr: JpegErrorMgr = unsafe { std::mem::zeroed() };
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut jerr as *mut JpegErrorMgr);
        assert!(!errp.is_null(), "jpeg_std_error");
        (*errp).error_exit = Some(ignore_error_exit);
        (*errp).msg_code = NO_ERROR;
        *(cinfo.as_mut_ptr() as *mut *mut JpegErrorMgr) = errp;
        jpeg_CreateDecompress(
            cinfo.as_mut_ptr() as *mut c_void,
            80,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        jpeg_mem_src(cinfo.as_mut_ptr() as *mut c_void, buf, size);
        // Read before destroy: the error manager is this run's own `jerr`, so
        // no other test can have touched it.
        let raised: c_int = (*errp).msg_code;
        jpeg_destroy_decompress(cinfo.as_mut_ptr() as *mut c_void);
        raised
    }
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
