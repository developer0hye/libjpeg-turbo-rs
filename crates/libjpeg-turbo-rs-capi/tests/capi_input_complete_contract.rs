//! P4-104 (#468): `jpeg_input_complete` must answer from the same place
//! upstream answers from, and reject a bad state the way upstream rejects it.
//!
//! Upstream (`jdapimin.c`) is three lines:
//!
//! ```c
//! if (cinfo->global_state < DSTATE_START || cinfo->global_state > DSTATE_STOPPING)
//!   ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);
//! return cinfo->inputctl->eoi_reached;
//! ```
//!
//! Note what it does **not** do: derive the answer from `global_state`. This
//! shim used to return `state >= DSTATE_SCANNING && !body_incomplete`, which
//! worked only because `jpeg_consume_input` jumped the state to `SCANNING` on
//! header completion. That hack is what blocked modelling `DSTATE_READY`, so
//! untangling it was a prerequisite for the rest of P4-104's transition work —
//! not a cosmetic change.

//! The P4-104 (#468) restructure landed: `jpeg_input_complete` answers from
//! `eoi_seen` — which now means exactly what upstream's
//! `inputctl->eoi_reached` means — and `jpeg_consume_input` follows
//! upstream's state dispatch (pre-start polls report `JPEG_REACHED_SOS`
//! without consuming). The three tests that spent their `#[ignore]`d lives
//! as executable specification for that contract are enabled below:
//! completion survives `jpeg_finish_decompress`, a reused handle clears it
//! at the next parse, and a baseline startup does not imply it.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_consume_input, jpeg_destroy_decompress, jpeg_input_complete,
    jpeg_mem_src, jpeg_read_header, jpeg_std_error, JpegDecompressPublic, JpegErrorMgr,
};

const JPEG_LIB_VERSION: c_int = 80;
/// `jerror.h` — "Improper call to JPEG library in state %d".
const JERR_BAD_STATE: c_int = 21;
const NO_ERROR: c_int = -1;

/// A minimal baseline JPEG, so the test does not depend on fixture files.
fn tiny_jpeg() -> Vec<u8> {
    use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 251) as u8).collect();
    compress(&pixels, 16, 16, PixelFormat::Rgb, 80, Subsampling::S420).expect("compress fixture")
}

struct Harness {
    cinfo: Box<JpegDecompressPublic>,
    _err: Box<JpegErrorMgr>,
}

impl Harness {
    fn new() -> Self {
        let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
        let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
        // SAFETY: both are live, correctly-aligned allocations owned here, and
        // the declared struct size matches the type passed.
        unsafe {
            let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err as *mut JpegErrorMgr);
            cinfo.err = errp;
            jpeg_CreateDecompress(
                &mut *cinfo as *mut JpegDecompressPublic as *mut c_void,
                JPEG_LIB_VERSION,
                std::mem::size_of::<JpegDecompressPublic>(),
            );
            cinfo.err = errp;
            // `error_exit` must return rather than `longjmp`; panicking would
            // cross an `extern "C"` frame and abort.
            unsafe extern "C" fn ignore_error_exit(_cinfo: *mut c_void) {}
            (*errp).error_exit = Some(ignore_error_exit);
            (*errp).msg_code = NO_ERROR;
        }
        Self { cinfo, _err: err }
    }

    fn ptr(&mut self) -> *mut c_void {
        &mut *self.cinfo as *mut JpegDecompressPublic as *mut c_void
    }

    fn raised(&self) -> c_int {
        // SAFETY: `err` was installed in `new` and outlives this borrow.
        unsafe { (*self.cinfo.err).msg_code }
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        // SAFETY: created by `jpeg_CreateDecompress`, destroyed exactly once.
        unsafe {
            jpeg_destroy_decompress(&mut *self.cinfo as *mut JpegDecompressPublic as *mut c_void)
        };
    }
}

/// Read every output row, so a following `jpeg_finish_decompress` is the
/// successful kind rather than the "too little data" kind upstream rejects.
fn drain_all_scanlines(p: *mut c_void, h: &mut Harness) {
    use libjpeg_turbo_rs_capi::jpeglib::jpeg_read_scanlines;

    let width: usize = h.cinfo.output_width as usize;
    let components: usize = h.cinfo.output_components as usize;
    let height: usize = h.cinfo.output_height as usize;
    let mut row: Vec<u8> = vec![0u8; width * components];
    for _ in 0..height {
        let mut rowptr: *mut u8 = row.as_mut_ptr();
        // SAFETY: `p` is a live decompress struct mid-output; `rowptr` names a
        // buffer of exactly `output_width * output_components` bytes, which is
        // what one scanline requires.
        let got: u32 = unsafe { jpeg_read_scanlines(p, &mut rowptr as *mut *mut u8, 1) };
        assert_eq!(got, 1, "expected one scanline per call");
    }
}

/// A fresh decompressor has reached no EOI, and saying so is not an error:
/// `DSTATE_START` is inside upstream's accepted range.
#[test]
fn fresh_decompressor_reports_incomplete_without_erroring() {
    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();
    // SAFETY: `p` is a live decompress struct from the harness.
    assert_eq!(unsafe { jpeg_input_complete(p) }, 0);
    assert_eq!(
        h.raised(),
        NO_ERROR,
        "DSTATE_START is in range, so this must not raise"
    );
}

/// A pre-start poll loop does NOT complete — upstream's contract, verbatim:
/// `jpeg_consume_input` "can't advance past first SOS until start_decompress
/// is called" (`jdapimin.c`), so from `DSTATE_READY` every poll reports
/// `JPEG_REACHED_SOS` without consuming and `jpeg_input_complete` stays
/// FALSE. This test used to assert the opposite — a shim-only promotion
/// that made the bare drain loop terminate — which was retired with the
/// P4-104 (#468) restructure; the working buffered idiom starts
/// decompression first (see the tests above and below).
#[test]
fn pre_start_polls_stay_incomplete_reporting_sos() {
    let jpeg: Vec<u8> = tiny_jpeg();
    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();

    // SAFETY: `p` is live; `jpeg` outlives every call below, which is what
    // `jpeg_mem_src`'s retained-pointer contract requires.
    unsafe {
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);
        assert_eq!(jpeg_read_header(p, 1), 1, "JPEG_HEADER_OK");

        const JPEG_REACHED_SOS: c_int = 1;
        for poll in 0..8 {
            assert_eq!(
                jpeg_input_complete(p),
                0,
                "poll {poll}: a parsed header is not a consumed datastream"
            );
            assert_eq!(
                jpeg_consume_input(p),
                JPEG_REACHED_SOS,
                "poll {poll}: pre-start consume reports SOS without advancing"
            );
        }
        assert_eq!(h.raised(), NO_ERROR, "idempotent polls must not raise");
    }
}

/// A state outside `DSTATE_START..=DSTATE_STOPPING` is `JERR_BAD_STATE`, not a
/// quiet `FALSE`. Returning 0 for a corrupt `cinfo` tells the caller "keep
/// polling", which is the opposite of what it needs to hear.
#[test]
fn out_of_range_state_raises_bad_state() {
    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();

    // Below the range: a compressor state handed to a decompressor entry point.
    h.cinfo.global_state = 100;
    // SAFETY: `p` is a live struct; only `global_state` was altered.
    assert_eq!(unsafe { jpeg_input_complete(p) }, 0);
    assert_eq!(
        h.raised(),
        JERR_BAD_STATE,
        "a compressor state must raise JERR_BAD_STATE"
    );

    // Above the range.
    // SAFETY: as above.
    unsafe { (*h.cinfo.err).msg_code = NO_ERROR };
    h.cinfo.global_state = 999;
    assert_eq!(unsafe { jpeg_input_complete(p) }, 0);
    assert_eq!(h.raised(), JERR_BAD_STATE, "999 is past DSTATE_STOPPING");
}

/// After `jpeg_start_decompress` on a non-buffered multi-scan image, the input
/// *is* complete: this shim decodes the whole datastream eagerly there, and
/// upstream has likewise consumed every scan by the time startup returns.
///
/// This is the regression the `eoi_seen` change introduced and review caught.
/// Answering from `global_state >= DSTATE_SCANNING` happened to report true
/// here; answering from `eoi_seen` reported false, because the eager path
/// never set it. A caller deciding "is this the final pass?" or "must I drain
/// more input?" would have been told to keep polling a stream that was already
/// finished.
#[test]
fn progressive_start_decompress_leaves_input_complete() {
    use libjpeg_turbo_rs_capi::inner::{Encoder, PixelFormat};
    use libjpeg_turbo_rs_capi::jpeglib::jpeg_start_decompress;

    let pixels: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 251) as u8).collect();
    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(80)
        .progressive(true)
        .encode()
        .expect("progressive encode");

    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();

    // SAFETY: `p` is live and `jpeg` outlives every call, as `jpeg_mem_src`'s
    // retained-pointer contract requires.
    unsafe {
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);
        assert_eq!(jpeg_read_header(p, 1), 1, "JPEG_HEADER_OK");
        assert_eq!(jpeg_start_decompress(p), 1, "start_decompress");
        assert_eq!(
            jpeg_input_complete(p),
            1,
            "a non-buffered multi-scan start_decompress consumes the whole \
             datastream, so the input is complete without further polling"
        );
        assert_eq!(h.raised(), NO_ERROR);
    }
}

/// After a successful `jpeg_finish_decompress`, the input is still complete.
///
/// Upstream's finish reaches EOI and then calls `jpeg_abort`, which rewrites
/// `global_state` and leaves `inputctl->eoi_reached` alone (`jcomapi.c:35-74`).
/// `DSTATE_START` is inside the accepted range, so the query stays valid and
/// still answers TRUE.
///
/// This shim used to clear its `eoi_seen` flag in finish and abort, with no
/// clear-on-fresh-parse. The P4-104 (#468) restructure moved the clear to
/// where upstream clears it — when the next datastream read begins
/// (`reset_input_controller`) — and left finish and abort alone, which is what
/// this test and `reusing_the_handle_clears_completion_for_the_new_image`
/// between them pin.
#[test]
fn input_stays_complete_after_finish_decompress() {
    use libjpeg_turbo_rs_capi::jpeglib::{jpeg_finish_decompress, jpeg_start_decompress};

    let jpeg: Vec<u8> = tiny_jpeg();
    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();

    // SAFETY: `p` is live and `jpeg` outlives every call.
    unsafe {
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);
        assert_eq!(jpeg_read_header(p, 1), 1);
        assert_eq!(jpeg_start_decompress(p), 1);
        // Read every row before finishing. Upstream raises
        // `JERR_TOO_LITTLE_DATA` when `output_scanline < output_height`
        // (`jdapimin.c:404-408`), and since P4-104 (#468) so does this shim,
        // so calling finish straight after startup would fail the assertion
        // below rather than assert a successful finish it never performed.
        drain_all_scanlines(p, &mut h);
        assert_eq!(jpeg_finish_decompress(p), 1, "finish");
        assert_eq!(
            jpeg_input_complete(p),
            1,
            "upstream's finish leaves eoi_reached set, so this stays TRUE"
        );
        assert_eq!(h.raised(), NO_ERROR);
    }
}

/// Reusing the handle for a second image clears it again — the flag tracks the
/// *current* datastream, so a fresh parse must not inherit the previous one's
/// completion. Without this, a reused handle would report "complete" before
/// reading anything.
#[test]
fn reusing_the_handle_clears_completion_for_the_new_image() {
    use libjpeg_turbo_rs_capi::jpeglib::{jpeg_finish_decompress, jpeg_start_decompress};

    let jpeg: Vec<u8> = tiny_jpeg();
    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();

    // SAFETY: `p` is live; `jpeg` outlives both decodes.
    unsafe {
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);
        assert_eq!(jpeg_read_header(p, 1), 1);
        assert_eq!(jpeg_start_decompress(p), 1);
        // Drain first, for the same reason the previous test does: upstream
        // rejects finish with `JERR_TOO_LITTLE_DATA` while
        // `output_scanline < output_height`, so skipping this would assert a
        // successful finish that never happened.
        drain_all_scanlines(p, &mut h);
        assert_eq!(jpeg_finish_decompress(p), 1);
        assert_eq!(jpeg_input_complete(p), 1, "first image complete");

        // Second image on the same handle. Installing a source does not
        // itself begin the read, so the previous datastream's completion
        // still stands at this point — as it does upstream.
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);
        assert_eq!(jpeg_input_complete(p), 1, "before the new parse begins");

        // **This is the assertion that matters**, and the first version of
        // this test stopped just short of it. Parsing the new header must
        // clear the previous image's completion: reporting the new datastream
        // complete before its body has been read would tell a buffered-image
        // caller to skip draining entirely.
        //
        // It failed until `jpeg_finish_decompress` was corrected to end at
        // `DSTATE_START` the way upstream's finish does — the clear keys on
        // that state, and a shim leaving `STOPPING` never reached it.
        assert_eq!(jpeg_read_header(p, 1), 1, "second header");
        assert_eq!(
            jpeg_input_complete(p),
            0,
            "a freshly parsed header must not inherit the previous image's EOI"
        );
        assert_eq!(h.raised(), NO_ERROR);
    }
}

/// A baseline single-scan image is **not** input-complete merely because
/// `jpeg_start_decompress` returned.
///
/// Upstream absorbs the whole datastream during startup for one shape only
/// (`jdapistd.c:55-70`): non-buffered *and* multi-scan. For baseline, EOI is
/// not reached until the rows are read. This shim decodes eagerly in every
/// case, and an earlier version of this work let that internal eagerness
/// change the advertised contract — reporting complete here where upstream
/// reports incomplete. Internal eagerness is not a licence to answer a public
/// query differently.
#[test]
fn baseline_startup_alone_is_not_completion() {
    use libjpeg_turbo_rs_capi::jpeglib::jpeg_start_decompress;

    let jpeg: Vec<u8> = tiny_jpeg();
    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();

    // SAFETY: `p` is live and `jpeg` outlives every call.
    unsafe {
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);
        assert_eq!(jpeg_read_header(p, 1), 1);
        assert_eq!(jpeg_start_decompress(p), 1);
        assert_eq!(
            jpeg_input_complete(p),
            0,
            "baseline startup does not absorb the datastream upstream, so it \
             must not report complete here"
        );
        assert_eq!(h.raised(), NO_ERROR);
    }
}

/// A successful `jpeg_finish_decompress` leaves `global_state` at
/// `DSTATE_START`.
///
/// Upstream's finish ends with `jpeg_abort` — *"We can use jpeg_abort to
/// release memory and reset global_state"* (`jdapimin.c`) — so `DSTATE_START`
/// is what a caller observes. `DSTATE_STOPPING` is the state finish passes
/// *through* while draining to EOI, and this shim used to stop there.
///
/// `global_state` is a public field consumers switch on, so this is an
/// observable contract, not an internal detail.
///
/// Enabled deliberately when it landed: the two other tests that touch finish
/// were `#[ignore]`d pending the `consume_input` rework, which would have left
/// the `DSTATE_START` change with no gate at all — reverting it would have
/// kept every enabled test green. Those two run alongside this one since the
/// P4-104 (#468) closure.
#[test]
fn finish_decompress_resets_state_to_start() {
    use libjpeg_turbo_rs_capi::jpeglib::{jpeg_finish_decompress, jpeg_start_decompress};

    /// `jpegint.h` — cross-checked against upstream by
    /// `every_state_constant_matches_upstream_jpegint_h`.
    const DSTATE_START: c_int = 200;

    let jpeg: Vec<u8> = tiny_jpeg();
    let mut h: Harness = Harness::new();
    let p: *mut c_void = h.ptr();

    // SAFETY: `p` is live and `jpeg` outlives every call.
    unsafe {
        jpeg_mem_src(p, jpeg.as_ptr(), jpeg.len() as std::os::raw::c_ulong);
        assert_eq!(jpeg_read_header(p, 1), 1);
        assert_eq!(jpeg_start_decompress(p), 1);
        // Upstream rejects finish with JERR_TOO_LITTLE_DATA while
        // `output_scanline < output_height`, so drain first — otherwise this
        // asserts the final state of a finish that should not have succeeded.
        drain_all_scanlines(p, &mut h);
        assert_eq!(jpeg_finish_decompress(p), 1, "finish");
    }

    assert_eq!(
        h.cinfo.global_state, DSTATE_START,
        "upstream's finish ends with jpeg_abort, leaving DSTATE_START"
    );
    assert_eq!(h.raised(), NO_ERROR);
}
