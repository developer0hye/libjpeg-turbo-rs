//! P4-104 / P4-106 (#468): classic state transitions and the finish/abort
//! lifecycle, trace-compared verbatim against stock libjpeg.
//!
//! The upstream contracts under test (`jdapimin.c` / `jcapimin.c`, measured
//! by `examples/classic_lifecycle_state_oracle.c` against stock libjpeg —
//! 3.1.4.1 locally, and in CI the pinned 3.1.90 submodule build the step
//! points `LIBJPEG_TURBO_PREFIX` at; see `docs/oracle_versions.tsv`):
//!
//! * a direct `jpeg_read_header` leaves `DSTATE_READY` (202), matching the
//!   consume-driven path;
//! * `jpeg_finish_decompress` raises `JERR_TOO_LITTLE_DATA` (69) on unread
//!   rows, `JERR_BAD_STATE` (21, parm = state) from a non-started state or
//!   on a second finish, and on success drains to EOI, calls `term_source`
//!   exactly once, and `jpeg_abort`s back to `DSTATE_START` (200);
//! * `jpeg_finish_compress` raises 69 when `next_scanline < image_height`,
//!   21 from a non-started state or after a completed finish, and an
//!   error-trapped object is reusable after `jpeg_abort`.
//!
//! Stock's harness stops each case at the first error via `setjmp`; this
//! mirror stops by checking the recording `error_exit` after every call.

use std::cell::Cell;
use std::ffi::{c_int, c_long, c_void};

mod helpers;

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_abort_compress, jpeg_abort_decompress, jpeg_destroy_compress, jpeg_destroy_decompress,
    jpeg_finish_compress, jpeg_finish_decompress, jpeg_finish_output, jpeg_input_complete,
    jpeg_mem_dest, jpeg_mem_src, jpeg_read_header, jpeg_read_scanlines, jpeg_set_defaults,
    jpeg_skip_scanlines, jpeg_start_compress, jpeg_start_decompress, jpeg_start_output,
    jpeg_std_error, jpeg_write_scanlines, JpegCompressPublic, JpegDecompressPublic, JpegErrorMgr,
    JpegSourceMgr,
};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
/// `jpeglib.h`: `JCS_GRAYSCALE`.
const JCS_GRAYSCALE: c_int = 1;

thread_local! {
    static FIRED: Cell<bool> = const { Cell::new(false) };
    static MSG_CODE: Cell<c_int> = const { Cell::new(0) };
    static PARM0: Cell<c_int> = const { Cell::new(0) };
    static TERM_CALLS: Cell<i32> = const { Cell::new(0) };
}

unsafe extern "C" fn recording_error_exit(cinfo: *mut c_void) {
    // SAFETY: the error path reads only the leading `err` pointer.
    unsafe {
        let err_ptr: *mut JpegErrorMgr = (cinfo as *const *mut JpegErrorMgr).read();
        if FIRED.with(|f| f.get()) {
            return;
        }
        FIRED.with(|f| f.set(true));
        MSG_CODE.with(|c| c.set(std::ptr::addr_of!((*err_ptr).msg_code).read()));
        let parm: *const u8 = std::ptr::addr_of!((*err_ptr).msg_parm) as *const u8;
        let mut bytes: [u8; std::mem::size_of::<c_int>()] = Default::default();
        std::ptr::copy_nonoverlapping(parm, bytes.as_mut_ptr(), bytes.len());
        PARM0.with(|p| p.set(c_int::from_ne_bytes(bytes)));
    }
}

fn reset_recorder() {
    FIRED.with(|f| f.set(false));
    MSG_CODE.with(|c| c.set(0));
    PARM0.with(|p| p.set(0));
}

fn fired() -> bool {
    FIRED.with(|f| f.get())
}

fn code() -> c_int {
    MSG_CODE.with(|c| c.get())
}

fn parm0() -> c_int {
    PARM0.with(|p| p.get())
}

// --------------- counting source manager (d4) ---------------

/// Counting source: the whole buffer at init, fabricated EOI on refill,
/// `term_source` bumps a thread-local. Mirrors the oracle's
/// `counting_src_mgr` so `d4`'s `term` token is comparable.
#[repr(C)]
struct CountingSrc {
    pub_mgr: JpegSourceMgr,
}

unsafe extern "C" fn csrc_init(_cinfo: *mut c_void) {}

unsafe extern "C" fn csrc_fill(cinfo: *mut c_void) -> c_int {
    static EOI: [u8; 2] = [0xFF, 0xD9];
    // SAFETY: `cinfo->src` is the manager this test installed.
    unsafe {
        let src: *mut JpegSourceMgr = decompress_src(cinfo);
        (*src).next_input_byte = EOI.as_ptr();
        (*src).bytes_in_buffer = 2;
    }
    1
}

unsafe extern "C" fn csrc_skip(cinfo: *mut c_void, num_bytes: c_long) {
    if num_bytes <= 0 {
        return;
    }
    // SAFETY: as in `csrc_fill`.
    unsafe {
        let src: *mut JpegSourceMgr = decompress_src(cinfo);
        let mut remaining: usize = num_bytes as usize;
        while remaining > (*src).bytes_in_buffer {
            remaining -= (*src).bytes_in_buffer;
            let _ = csrc_fill(cinfo);
        }
        (*src).next_input_byte = (*src).next_input_byte.add(remaining);
        (*src).bytes_in_buffer -= remaining;
    }
}

unsafe extern "C" fn csrc_term(_cinfo: *mut c_void) {
    TERM_CALLS.with(|t| t.set(t.get() + 1));
}

/// `cinfo->src` as a raw manager pointer, via the public struct field.
unsafe fn decompress_src(cinfo: *mut c_void) -> *mut JpegSourceMgr {
    // SAFETY: caller passes a valid decompress cinfo.
    unsafe { (*(cinfo as *mut JpegDecompressPublic)).src }
}

// --------------- shared scaffolding ---------------

/// An in-memory grayscale JPEG through the shim's classic compress.
fn make_source() -> Vec<u8> {
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

struct Decomp {
    cinfo: Box<JpegDecompressPublic>,
    _err: Box<JpegErrorMgr>,
}

impl Decomp {
    fn new() -> Self {
        reset_recorder();
        let mut err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
        let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
        // SAFETY: zeroed mirror + std_error + recording handler.
        unsafe {
            let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *err);
            (*errp).error_exit = Some(recording_error_exit);
            cinfo.err = errp;
            let p: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;
            libjpeg_turbo_rs_capi::jpeglib::jpeg_CreateDecompress(
                p,
                80,
                std::mem::size_of::<JpegDecompressPublic>(),
            );
        }
        Decomp { cinfo, _err: err }
    }

    fn ptr(&mut self) -> *mut c_void {
        &mut *self.cinfo as *mut JpegDecompressPublic as *mut c_void
    }
}

impl Drop for Decomp {
    fn drop(&mut self) {
        // SAFETY: created in `new`; destroy is idempotent for this mirror.
        unsafe { jpeg_destroy_decompress(self.ptr()) };
    }
}

struct Comp {
    cinfo: Box<JpegCompressPublic>,
    _err: Box<JpegErrorMgr>,
    /// The `jpeg_mem_dest` out-params. Boxed because the manager retains
    /// their *addresses* and writes through them at `term_destination` —
    /// registering fields of a by-value-returned struct handed the shim
    /// pointers into a dead stack frame, which release+LTO turned into a
    /// SIGSEGV at destroy (#468 review, measured; debug builds only
    /// scribbled). The heap slot survives the move.
    dest: Box<(*mut u8, std::ffi::c_ulong)>,
}

impl Comp {
    fn new() -> Self {
        reset_recorder();
        let err: Box<JpegErrorMgr> = Box::new(unsafe { std::mem::zeroed() });
        let cinfo: Box<JpegCompressPublic> = Box::new(unsafe { std::mem::zeroed() });
        let mut me = Comp {
            cinfo,
            _err: err,
            dest: Box::new((std::ptr::null_mut(), 0)),
        };
        // SAFETY: zeroed mirror + std_error + recording handler + mem dest.
        unsafe {
            let errp: *mut JpegErrorMgr = jpeg_std_error(&mut *me._err);
            (*errp).error_exit = Some(recording_error_exit);
            me.cinfo.err = errp;
            let p: *mut c_void = me.ptr();
            libjpeg_turbo_rs_capi::jpeglib::jpeg_create_compress(p);
            jpeg_mem_dest(p, &mut me.dest.0, &mut me.dest.1);
            me.cinfo.image_width = WIDTH as u32;
            me.cinfo.image_height = HEIGHT as u32;
            me.cinfo.input_components = 1;
            me.cinfo.in_color_space = JCS_GRAYSCALE;
            jpeg_set_defaults(p);
        }
        me
    }

    fn ptr(&mut self) -> *mut c_void {
        &mut *self.cinfo as *mut JpegCompressPublic as *mut c_void
    }

    fn write_rows(&mut self, n: usize) {
        let mut row: [u8; WIDTH] = [0x40; WIDTH];
        let p = self.ptr();
        for _ in 0..n {
            let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
            // SAFETY: started compress; row outlives the call.
            unsafe {
                let _ = jpeg_write_scanlines(p, rows.as_mut_ptr(), 1);
            }
            if fired() {
                return;
            }
        }
    }
}

impl Drop for Comp {
    fn drop(&mut self) {
        // SAFETY: created in `new`; the mem-dest buffer is freed after.
        unsafe {
            jpeg_destroy_compress(self.ptr());
            extern "C" {
                fn free(p: *mut c_void);
            }
            free(self.dest.0 as *mut c_void);
        }
    }
}

// --------------- the trace ---------------

fn our_trace() -> String {
    let src: Vec<u8> = make_source();
    let mut out = String::new();

    // d1: state after a direct read_header.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
        }
        if fired() {
            out.push_str(&format!("d1 err {}\n", code()));
        } else {
            out.push_str(&format!("d1 state {}\n", d.cinfo.global_state));
        }
    }

    // d2: finish with half the rows unread.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        let mut row: Vec<u8> = vec![0; WIDTH * 4];
        // SAFETY: classic sequence.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
            let _ = jpeg_start_decompress(p);
            for _ in 0..HEIGHT / 2 {
                let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
                let _ = jpeg_read_scanlines(p, rows.as_mut_ptr(), 1);
                if fired() {
                    break;
                }
            }
            if !fired() {
                let _ = jpeg_finish_decompress(p);
            }
        }
        if fired() {
            out.push_str(&format!("d2 err {}\n", code()));
        } else {
            out.push_str(&format!("d2 ok state {}\n", d.cinfo.global_state));
        }
    }

    // d3: finish straight after read_header.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
            if !fired() {
                let _ = jpeg_finish_decompress(p);
            }
        }
        if fired() {
            out.push_str(&format!("d3 err {} parm {}\n", code(), parm0()));
        } else {
            out.push_str("d3 ok\n");
        }
    }

    // d4: full decode + finish with the counting source.
    {
        let mut d = Decomp::new();
        TERM_CALLS.with(|t| t.set(0));
        let mut csrc = CountingSrc {
            pub_mgr: JpegSourceMgr {
                next_input_byte: src.as_ptr(),
                bytes_in_buffer: src.len(),
                init_source: Some(csrc_init),
                fill_input_buffer: Some(csrc_fill),
                skip_input_data: Some(csrc_skip),
                resync_to_restart: None,
                term_source: Some(csrc_term),
            },
        };
        d.cinfo.src = &mut csrc.pub_mgr as *mut JpegSourceMgr;
        let p = d.ptr();
        let mut row: Vec<u8> = vec![0; WIDTH * 4];
        let mut ret: c_int = 0;
        // SAFETY: classic sequence over the caller-installed source.
        unsafe {
            let _ = jpeg_read_header(p, 1);
            let _ = jpeg_start_decompress(p);
            while !fired() && d.cinfo.output_scanline < d.cinfo.output_height {
                let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
                let got: u32 = jpeg_read_scanlines(p, rows.as_mut_ptr(), 1);
                if got == 0 {
                    break;
                }
            }
            if !fired() {
                ret = jpeg_finish_decompress(p);
            }
        }
        if fired() {
            out.push_str(&format!("d4 err {}\n", code()));
        } else {
            out.push_str(&format!(
                "d4 ret {} state {} term {}\n",
                ret,
                d.cinfo.global_state,
                TERM_CALLS.with(|t| t.get())
            ));
        }
    }

    // d5: double finish, with the counting source so "exactly once" for
    // term_source is pinned by the trace (a refused second finish must not
    // call it again).
    {
        let mut d = Decomp::new();
        TERM_CALLS.with(|t| t.set(0));
        let mut csrc = CountingSrc {
            pub_mgr: JpegSourceMgr {
                next_input_byte: src.as_ptr(),
                bytes_in_buffer: src.len(),
                init_source: Some(csrc_init),
                fill_input_buffer: Some(csrc_fill),
                skip_input_data: Some(csrc_skip),
                resync_to_restart: None,
                term_source: Some(csrc_term),
            },
        };
        d.cinfo.src = &mut csrc.pub_mgr as *mut JpegSourceMgr;
        let p = d.ptr();
        let mut row: Vec<u8> = vec![0; WIDTH * 4];
        // SAFETY: classic sequence.
        unsafe {
            let _ = jpeg_read_header(p, 1);
            let _ = jpeg_start_decompress(p);
            while !fired() && d.cinfo.output_scanline < d.cinfo.output_height {
                let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
                let got: u32 = jpeg_read_scanlines(p, rows.as_mut_ptr(), 1);
                if got == 0 {
                    break;
                }
            }
            if !fired() {
                let _ = jpeg_finish_decompress(p);
            }
            if !fired() {
                let _ = jpeg_finish_decompress(p);
            }
        }
        if fired() {
            out.push_str(&format!(
                "d5 err {} parm {} term {}\n",
                code(),
                parm0(),
                TERM_CALLS.with(|t| t.get())
            ));
        } else {
            out.push_str(&format!("d5 ok term {}\n", TERM_CALLS.with(|t| t.get())));
        }
    }

    // d6: abort mid-decode, reuse the same object.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        let mut row: Vec<u8> = vec![0; WIDTH * 4];
        // SAFETY: classic sequence, aborted and restarted.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
            let _ = jpeg_start_decompress(p);
            for _ in 0..HEIGHT / 2 {
                let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
                let _ = jpeg_read_scanlines(p, rows.as_mut_ptr(), 1);
                if fired() {
                    break;
                }
            }
            if fired() {
                out.push_str(&format!("d6 err {}\n", code()));
            } else {
                jpeg_abort_decompress(p);
                out.push_str(&format!("d6 aborted state {}\n", d.cinfo.global_state));

                jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
                let _ = jpeg_read_header(p, 1);
                let _ = jpeg_start_decompress(p);
                while !fired() && d.cinfo.output_scanline < d.cinfo.output_height {
                    let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
                    let got: u32 = jpeg_read_scanlines(p, rows.as_mut_ptr(), 1);
                    if got == 0 {
                        break;
                    }
                }
                let rows_read: u32 = d.cinfo.output_scanline;
                if !fired() {
                    let _ = jpeg_finish_decompress(p);
                }
                if fired() {
                    out.push_str(&format!("d6 err {}\n", code()));
                } else {
                    out.push_str(&format!(
                        "d6 reused state {} rows {}\n",
                        d.cinfo.global_state, rows_read
                    ));
                }
            }
        }
    }

    // d7: finish_output / start_output from READY are refused.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
            if !fired() {
                let _ = jpeg_finish_output(p);
            }
        }
        if fired() {
            out.push_str(&format!("d7a err {} parm {}\n", code(), parm0()));
        } else {
            out.push_str("d7a ok\n");
        }
    }
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
            if !fired() {
                let _ = jpeg_start_output(p, 1);
            }
        }
        if fired() {
            out.push_str(&format!("d7b err {} parm {}\n", code(), parm0()));
        } else {
            out.push_str("d7b ok\n");
        }
    }

    // d8: a second jpeg_read_header without finish/abort is refused.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
            if !fired() {
                let _ = jpeg_read_header(p, 1);
            }
        }
        if fired() {
            out.push_str(&format!("d8 err {} parm {}\n", code(), parm0()));
        } else {
            out.push_str("d8 ok\n");
        }
    }

    // d9: skipping every row reports input complete.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = jpeg_read_header(p, 1);
            let _ = jpeg_start_decompress(p);
            let h = d.cinfo.output_height;
            let _ = jpeg_skip_scanlines(p, h);
        }
        if fired() {
            out.push_str(&format!("d9 err {}\n", code()));
        } else {
            // SAFETY: `p` is live.
            let complete: c_int = unsafe { jpeg_input_complete(p) };
            out.push_str(&format!(
                "d9 skipped {} complete {}\n",
                d.cinfo.output_scanline, complete
            ));
            // SAFETY: as above; the drain-to-EOI finish.
            unsafe {
                let _ = jpeg_finish_decompress(p);
            }
        }
    }

    // c1: finish with half the rows written.
    {
        let mut c = Comp::new();
        let p = c.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_start_compress(p, 1);
        }
        c.write_rows(HEIGHT / 2);
        // SAFETY: as above.
        unsafe {
            if !fired() {
                jpeg_finish_compress(p);
            }
        }
        if fired() {
            out.push_str(&format!("c1 err {}\n", code()));
        } else {
            out.push_str(&format!("c1 ok size {}\n", c.dest.1));
        }
    }

    // c2: finish without start.
    {
        let mut c = Comp::new();
        let p = c.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_finish_compress(p);
        }
        if fired() {
            out.push_str(&format!("c2 err {} parm {}\n", code(), parm0()));
        } else {
            out.push_str("c2 ok\n");
        }
    }

    // c3: double finish.
    {
        let mut c = Comp::new();
        let p = c.ptr();
        // SAFETY: classic sequence.
        unsafe {
            jpeg_start_compress(p, 1);
        }
        c.write_rows(HEIGHT);
        // SAFETY: as above.
        unsafe {
            if !fired() {
                jpeg_finish_compress(p);
            }
            if !fired() {
                jpeg_finish_compress(p);
            }
        }
        if fired() {
            out.push_str(&format!("c3 err {} parm {}\n", code(), parm0()));
        } else {
            out.push_str("c3 ok\n");
        }
    }

    // c5: error (missing rows), abort, reuse the same object fully.
    {
        let mut c = Comp::new();
        let p = c.ptr();
        // SAFETY: classic sequence with an abort-recover in the middle.
        unsafe {
            jpeg_start_compress(p, 1);
        }
        c.write_rows(HEIGHT / 2);
        // SAFETY: as above.
        unsafe {
            if !fired() {
                jpeg_finish_compress(p);
            }
        }
        if !fired() {
            out.push_str("c5 unexpected ok\n");
        } else {
            out.push_str(&format!("c5 err {}\n", code()));
            let p = c.ptr();
            // SAFETY: abort + reinstall destination + full compress, the
            // oracle's recovery sequence.
            unsafe {
                jpeg_abort_compress(p);
                out.push_str(&format!("c5 aborted state {}\n", c.cinfo.global_state));
                reset_recorder();
                extern "C" {
                    fn free(q: *mut c_void);
                }
                free(c.dest.0 as *mut c_void);
                c.dest.0 = std::ptr::null_mut();
                c.dest.1 = 0;
                let (buf_ptr, size_ptr) = (&mut c.dest.0 as *mut *mut u8, &mut c.dest.1 as *mut _);
                jpeg_mem_dest(p, buf_ptr, size_ptr);
                jpeg_start_compress(p, 1);
            }
            c.write_rows(HEIGHT);
            // SAFETY: as above.
            unsafe {
                if !fired() {
                    jpeg_finish_compress(p);
                }
            }
            if fired() {
                out.push_str(&format!("c5 err2 {}\n", code()));
            } else {
                out.push_str(&format!(
                    "c5 reused ok {} state {}\n",
                    if c.dest.1 > 100 { 1 } else { 0 },
                    c.cinfo.global_state
                ));
            }
        }
    }

    out
}

/// Issue #468 (P4-104/P4-106): the lifecycle matrix, compared verbatim
/// against stock libjpeg.
#[test]
fn classic_lifecycle_states_match_stock_libjpeg() {
    let Some(oracle) = helpers::build_classic_oracle("classic_lifecycle_state_oracle") else {
        eprintln!(
            "SKIP: no classic libjpeg development install found; set \
             LIBJPEG_TURBO_PREFIX to make this a hard failure."
        );
        return;
    };
    let c_trace: String = helpers::run_oracle(&oracle, &[]);
    let ours: String = our_trace();
    assert_eq!(
        ours, c_trace,
        "classic lifecycle/state behaviour diverges from stock libjpeg \
         (P4-104/P4-106, #468)"
    );
}
