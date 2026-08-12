//! P4-154 (#538): classic `jpeg_start_compress` / `jpeg_write_scanlines`
//! must honour `data_precision` the way upstream does.
//!
//! `data_precision` is a public struct field a caller sets directly, with no
//! setter to funnel validation through — and this shim ignored it entirely on
//! the 8-bit scanline path: precision 8, 9, 12 and 16 all produced one and
//! the same 8-bit stream with no error — 629 identical bytes on the 16x16 RGB
//! frame P4-154 was filed on, 337 on this suite's 8x8 grayscale one. Upstream
//! has two gates in two different calls, and they disagree about 12:
//!
//!   - `jpeg_start_compress` → jcmaster.c `initial_setup`: lossy admits
//!     {8, 12}, lossless 2..=16 (`jcmaster.c:196-208`);
//!   - the 8-bit `jpeg_write_scanlines` entry (`jcapistd.c:92-105`): lossy
//!     requires exactly `BITS_IN_JSAMPLE` (8), lossless 2..=8.
//!
//! So which error a caller sees, per (precision, lossless) pair, is decided
//! by a **C oracle** (`examples/classic_precision_oracle.c`), not by reading:
//! the whole trace — stage, `msg_code`, and ERREXIT1's `msg_parm.i[0]` — is
//! compared line for line. The Rust side mirrors the C control flow (a stage
//! that raises ends the case, as `longjmp` does in the oracle).

use std::cell::Cell;
use std::ffi::{c_int, c_void};

mod helpers;

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_destroy_compress, jpeg_enable_lossless, jpeg_finish_compress, jpeg_mem_dest,
    jpeg_set_defaults, jpeg_start_compress, jpeg_std_error, jpeg_write_scanlines,
    JpegCompressPublic, JpegErrorMgr,
};

const WIDTH: usize = 8;
const HEIGHT: usize = 8;
/// `jpeglib.h`: `JCS_GRAYSCALE`.
const JCS_GRAYSCALE: c_int = 1;

thread_local! {
    static FIRED: Cell<bool> = const { Cell::new(false) };
    static MSG_CODE: Cell<c_int> = const { Cell::new(0) };
    static PARM0: Cell<c_int> = const { Cell::new(0) };
}

/// Records the first failure. Our shim returns after `error_exit` instead of
/// requiring a `longjmp`, so the harness checks `FIRED` after every stage and
/// stops — the same control flow the C oracle's `setjmp` gives it.
unsafe extern "C" fn recording_error_exit(cinfo: *mut c_void) {
    // SAFETY: the error path reads only the leading `err` pointer.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = (cinfo as *const *mut JpegErrorMgr).read();
        if FIRED.with(|f| f.get()) {
            return; // only the *first* failure is the trace
        }
        FIRED.with(|f| f.set(true));
        MSG_CODE.with(|c| c.set(std::ptr::addr_of!((*err_ptr).msg_code).read()));
        let parm: *const u8 = std::ptr::addr_of!((*err_ptr).msg_parm) as *const u8;
        let mut bytes: [u8; std::mem::size_of::<c_int>()] = Default::default();
        std::ptr::copy_nonoverlapping(parm, bytes.as_mut_ptr(), bytes.len());
        PARM0.with(|p| p.set(c_int::from_ne_bytes(bytes)));
    }
}

/// One (precision, lossless) case through the shim, in the oracle's exact
/// line format: `case stage code parm0`.
fn run_case(label: &str, precision: c_int, lossless: bool) -> String {
    FIRED.with(|f| f.set(false));
    MSG_CODE.with(|c| c.set(0));
    PARM0.with(|p| p.set(0));

    let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: *mut c_void = &mut *cinfo as *mut JpegCompressPublic as *mut c_void;
    let mut out_buf: *mut u8 = std::ptr::null_mut();
    let mut out_size: std::ffi::c_ulong = 0;

    let fired = || FIRED.with(|f| f.get());
    let line = |stage: &str| {
        format!(
            "{label} {stage} {} {}\n",
            MSG_CODE.with(|c| c.get()),
            PARM0.with(|p| p.get())
        )
    };

    // SAFETY: `cinfo` is a zeroed compress mirror; every call below follows
    // the classic API's documented sequence on it.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut err);
        (*errp).error_exit = Some(recording_error_exit);
        cinfo.err = errp;
        libjpeg_turbo_rs_capi::jpeglib::jpeg_create_compress(cinfo_ptr);
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);
        cinfo.image_width = WIDTH as u32;
        cinfo.image_height = HEIGHT as u32;
        cinfo.input_components = 1;
        cinfo.in_color_space = JCS_GRAYSCALE;
        jpeg_set_defaults(cinfo_ptr);
        cinfo.data_precision = precision;
        if lossless {
            jpeg_enable_lossless(cinfo_ptr, 1, 0);
        }

        jpeg_start_compress(cinfo_ptr, 1);
        if fired() {
            let out: String = line("start");
            jpeg_destroy_compress(cinfo_ptr);
            return out;
        }

        let mut row: [u8; WIDTH] = [0; WIDTH];
        for r in 0..HEIGHT {
            for (i, b) in row.iter_mut().enumerate() {
                *b = (r * WIDTH + i) as u8;
            }
            let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
            let _ = jpeg_write_scanlines(cinfo_ptr, rows.as_mut_ptr(), 1);
            if fired() {
                let out: String = line("write");
                jpeg_destroy_compress(cinfo_ptr);
                return out;
            }
        }

        jpeg_finish_compress(cinfo_ptr);
        let out: String = if fired() {
            line("finish")
        } else {
            // Exact size and byte checksum, as the oracle prints: an accepted
            // precision must also be the *encoded* precision — SOF3's
            // precision byte and the predictor arithmetic both feed the sum
            // (#538 review: the first version printed only "ok" and an
            // accepted 2-bit lossless request that encoded 8-bit passed).
            let bytes: &[u8] = std::slice::from_raw_parts(out_buf, out_size as usize);
            let sum: u64 = bytes
                .iter()
                .fold(0u64, |acc, b| (acc + u64::from(*b)) & 0xFFFF_FFFF);
            format!("{label} ok {out_size} {sum}\n")
        };
        jpeg_destroy_compress(cinfo_ptr);
        if !out_buf.is_null() {
            libc_free(out_buf as *mut c_void);
        }
        out
    }
}

/// `jpeg_mem_dest` hands back `malloc`-family storage the caller frees.
unsafe fn libc_free(p: *mut c_void) {
    extern "C" {
        fn free(p: *mut c_void);
    }
    // SAFETY: `p` came from the shim's mem-dest allocation, freed once.
    unsafe { free(p) };
}

/// Issue #538 (P4-154): the whole (precision, lossless) matrix, stage +
/// `msg_code` + `msg_parm.i[0]`, compared verbatim against stock libjpeg.
#[test]
fn classic_precision_gates_match_stock_libjpeg() {
    let Some(oracle) = helpers::build_classic_oracle("classic_precision_oracle") else {
        eprintln!(
            "SKIP: no classic libjpeg development install found; set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };
    let c_trace: String = helpers::run_oracle(&oracle, &[]);

    let mut ours: String = String::new();
    for (label, precision, lossless) in [
        ("lossy_2", 2, false),
        ("lossy_8", 8, false),
        ("lossy_9", 9, false),
        ("lossy_12", 12, false),
        ("lossy_16", 16, false),
        ("lossless_2", 2, true),
        ("lossless_8", 8, true),
        ("lossless_9", 9, true),
        ("lossless_12", 12, true),
        ("lossless_16", 16, true),
    ] {
        ours.push_str(&run_case(label, precision, lossless));
    }

    ours.push_str(&run_width_overflow_case("mixed_width_overflow_precision_9"));

    assert_eq!(
        ours, c_trace,
        "classic data_precision acceptance diverges from stock libjpeg \
         (P4-154, #538)"
    );
}

/// Error precedence for a doubly-invalid setup: a row too wide for
/// `JDIMENSION` (65500 × 65573 samples) *and* precision 9. Upstream raises
/// `JERR_WIDTH_OVERFLOW` first (`jcmaster.c:190-208`); a port that checks
/// precision first reports the wrong error (#538 review).
fn run_width_overflow_case(label: &str) -> String {
    FIRED.with(|f| f.set(false));
    MSG_CODE.with(|c| c.set(0));
    PARM0.with(|p| p.set(0));

    let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: *mut c_void = &mut *cinfo as *mut JpegCompressPublic as *mut c_void;
    let mut out_buf: *mut u8 = std::ptr::null_mut();
    let mut out_size: std::ffi::c_ulong = 0;

    // SAFETY: as `run_case` — the classic sequence over a zeroed mirror.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut err);
        (*errp).error_exit = Some(recording_error_exit);
        cinfo.err = errp;
        libjpeg_turbo_rs_capi::jpeglib::jpeg_create_compress(cinfo_ptr);
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);
        cinfo.image_width = 65500;
        cinfo.image_height = 1;
        cinfo.input_components = 1;
        cinfo.in_color_space = JCS_GRAYSCALE;
        jpeg_set_defaults(cinfo_ptr);
        cinfo.input_components = 65573; // after set_defaults, as a caller can
        cinfo.data_precision = 9;

        jpeg_start_compress(cinfo_ptr, 1);
        let out: String = if FIRED.with(|f| f.get()) {
            format!(
                "{label} start {} {}\n",
                MSG_CODE.with(|c| c.get()),
                PARM0.with(|p| p.get())
            )
        } else {
            format!("{label} accepted 0 0\n")
        };
        jpeg_destroy_compress(cinfo_ptr);
        if !out_buf.is_null() {
            libc_free(out_buf as *mut c_void);
        }
        out
    }
}
