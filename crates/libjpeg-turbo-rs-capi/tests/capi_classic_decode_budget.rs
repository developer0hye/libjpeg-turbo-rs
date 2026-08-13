//! P4-14 (#467): the classic decode sequence honours
//! `cinfo->mem->max_memory_to_use`.
//!
//! The memory-manager vtable has enforced the field since #519, but the
//! documented consumer sequence — `jpeg_read_header` →
//! `jpeg_start_decompress` → `jpeg_read_scanlines` — ran the native decoder
//! with default limits and was unbounded. That sequence is exactly what
//! issue #467 names.
//!
//! What upstream actually bounds is measured, not assumed
//! (`examples/classic_budget_oracle.c` vs stock 3.1.4.1): the field is
//! consulted by `realize_virt_arrays`, which exists only where whole-image
//! coefficient arrays do — so a 1000-byte budget passes a *baseline* 64×64
//! decode untouched and fails a *progressive* one at
//! `jpeg_start_decompress` with `JERR_NO_BACKING_STORE` (51). The exact byte
//! threshold is upstream's own accounting and is not compared — this port's
//! estimation model is documented as coarser (P4-14 PARTIAL note) — the
//! budgets sit far on either side of both models.

use std::cell::Cell;
use std::ffi::{c_int, c_long, c_void};

mod helpers;

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_destroy_compress, jpeg_destroy_decompress, jpeg_finish_compress, jpeg_finish_decompress,
    jpeg_mem_dest, jpeg_mem_src, jpeg_read_header, jpeg_read_scanlines, jpeg_set_defaults,
    jpeg_simple_progression, jpeg_start_compress, jpeg_start_decompress, jpeg_std_error,
    jpeg_write_scanlines, JpegCompressPublic, JpegDecompressPublic, JpegErrorMgr,
};
use libjpeg_turbo_rs_capi::memmgr::JpegMemoryMgr;

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
/// `jpeglib.h`: `JCS_GRAYSCALE`.
const JCS_GRAYSCALE: c_int = 1;
/// `jerror.h` at v8: `JERR_NO_BACKING_STORE`.
const JERR_NO_BACKING_STORE: c_int = 51;

thread_local! {
    static FIRED: Cell<bool> = const { Cell::new(false) };
    static MSG_CODE: Cell<c_int> = const { Cell::new(0) };
}

/// Records the first failure; the harness checks after every stage and
/// stops, the control flow the C oracle's `setjmp` gives it.
unsafe extern "C" fn recording_error_exit(cinfo: *mut c_void) {
    // SAFETY: the error path reads only the leading `err` pointer.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = (cinfo as *const *mut JpegErrorMgr).read();
        if FIRED.with(|f| f.get()) {
            return;
        }
        FIRED.with(|f| f.set(true));
        MSG_CODE.with(|c| c.set(std::ptr::addr_of!((*err_ptr).msg_code).read()));
    }
}

/// An in-memory grayscale JPEG through the shim's own classic compress,
/// mirroring the oracle's `make_source` shape for shape.
fn make_source(progressive: bool) -> Vec<u8> {
    let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: *mut c_void = &mut *cinfo as *mut JpegCompressPublic as *mut c_void;
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: std::ffi::c_ulong = 0;

    // SAFETY: the classic sequence over a zeroed mirror; buffers are locals.
    unsafe {
        cinfo.err = jpeg_std_error(&mut err);
        libjpeg_turbo_rs_capi::jpeglib::jpeg_create_compress(cinfo_ptr);
        jpeg_mem_dest(cinfo_ptr, &mut buf, &mut size);
        cinfo.image_width = WIDTH as u32;
        cinfo.image_height = HEIGHT as u32;
        cinfo.input_components = 1;
        cinfo.in_color_space = JCS_GRAYSCALE;
        jpeg_set_defaults(cinfo_ptr);
        if progressive {
            jpeg_simple_progression(cinfo_ptr);
        }
        jpeg_start_compress(cinfo_ptr, 1);
        let mut row: [u8; WIDTH] = [0; WIDTH];
        for r in 0..HEIGHT {
            for (i, b) in row.iter_mut().enumerate() {
                *b = ((r * 7 + i * 13) & 0xFF) as u8;
            }
            let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
            let _ = jpeg_write_scanlines(cinfo_ptr, rows.as_mut_ptr(), 1);
        }
        jpeg_finish_compress(cinfo_ptr);
        let out: Vec<u8> = std::slice::from_raw_parts(buf, size as usize).to_vec();
        jpeg_destroy_compress(cinfo_ptr);
        extern "C" {
            fn free(p: *mut c_void);
        }
        free(buf as *mut c_void);
        out
    }
}

/// One decode case in the oracle's exact line format: `case stage code`.
fn run_case(label: &str, progressive: bool, budget: c_long) -> String {
    FIRED.with(|f| f.set(false));
    MSG_CODE.with(|c| c.set(0));

    let src: Vec<u8> = make_source(progressive);
    let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;

    let fired = || FIRED.with(|f| f.get());
    let line = |stage: &str| format!("{label} {stage} {}\n", MSG_CODE.with(|c| c.get()));

    // SAFETY: the documented classic decode sequence over a zeroed mirror.
    unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut err);
        (*errp).error_exit = Some(recording_error_exit);
        cinfo.err = errp;
        libjpeg_turbo_rs_capi::jpeglib::jpeg_CreateDecompress(
            cinfo_ptr,
            80,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        jpeg_mem_src(cinfo_ptr, src.as_ptr(), src.len() as std::ffi::c_ulong);

        let _ = jpeg_read_header(cinfo_ptr, 1);
        if fired() {
            let out: String = line("header");
            jpeg_destroy_decompress(cinfo_ptr);
            return out;
        }

        // The caller sets the field after create/header, as libjpeg.txt
        // documents; this is the sequence #467 names.
        std::ptr::addr_of_mut!((*(cinfo.mem as *mut JpegMemoryMgr)).max_memory_to_use)
            .write(budget);

        let _ = jpeg_start_decompress(cinfo_ptr);
        if fired() {
            let out: String = line("start");
            jpeg_destroy_decompress(cinfo_ptr);
            return out;
        }

        let mut row: Vec<u8> = vec![0; WIDTH * 4];
        while cinfo.output_scanline < cinfo.output_height {
            let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
            let got: u32 = jpeg_read_scanlines(cinfo_ptr, rows.as_mut_ptr(), 1);
            if fired() {
                let out: String = line("scan");
                jpeg_destroy_decompress(cinfo_ptr);
                return out;
            }
            if got == 0 {
                break;
            }
        }
        let _ = jpeg_finish_decompress(cinfo_ptr);
        let out: String = if fired() {
            line("scan")
        } else {
            format!("{label} ok 0\n")
        };
        jpeg_destroy_decompress(cinfo_ptr);
        out
    }
}

/// Issue #467 (P4-14): the six-case matrix, stage + `msg_code`, compared
/// verbatim against stock libjpeg.
#[test]
fn classic_decode_budget_matches_stock_libjpeg() {
    let Some(oracle) = helpers::build_classic_oracle("classic_budget_oracle") else {
        eprintln!(
            "SKIP: no classic libjpeg development install found; set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };
    let c_trace: String = helpers::run_oracle(&oracle, &[]);

    let mut ours: String = String::new();
    ours.push_str(&run_case("baseline_unlimited", false, 0));
    ours.push_str(&run_case("baseline_tiny", false, 1000));
    ours.push_str(&run_case("baseline_generous", false, 100 * 1024 * 1024));
    ours.push_str(&run_case("progressive_unlimited", true, 0));
    ours.push_str(&run_case("progressive_tiny", true, 1000));
    ours.push_str(&run_case("progressive_generous", true, 100 * 1024 * 1024));

    assert_eq!(
        ours, c_trace,
        "classic max_memory_to_use enforcement diverges from stock libjpeg \
         (P4-14, #467)"
    );
}

/// The budget refusal itself, independent of any C oracle: a tiny budget on
/// a progressive stream raises `JERR_NO_BACKING_STORE` at
/// `jpeg_start_decompress`; baseline is untouched (upstream bounds only the
/// whole-image coefficient path) and a generous budget passes everything.
#[test]
fn budget_refusal_fires_standalone() {
    let progressive_tiny: String = run_case("p", true, 1000);
    assert_eq!(
        progressive_tiny,
        format!("p start {JERR_NO_BACKING_STORE}\n"),
        "a 1000-byte budget must refuse a progressive decode at start"
    );
    let baseline_tiny: String = run_case("b", false, 1000);
    assert_eq!(
        baseline_tiny, "b ok 0\n",
        "baseline needs no whole-image coefficient arrays — stock accepts it \
         under the same budget, and so must we"
    );
    let progressive_generous: String = run_case("g", true, 100 * 1024 * 1024);
    assert_eq!(progressive_generous, "g ok 0\n");
}
