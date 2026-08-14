//! P4-109 (#469): classic source-manager setup and stdio semantics,
//! trace-compared verbatim against stock libjpeg.
//!
//! The upstream contracts under test (`jdatasrc.c`, measured by
//! `examples/classic_source_mgr_oracle.c` against stock 3.1.4.1):
//!
//! * `jpeg_mem_src` rejects NULL/empty input (`JERR_INPUT_EMPTY`, 43) and
//!   refuses to overwrite a source manager it did not create
//!   (`JERR_BUFFER_SIZE`, 24 — `jdatasrc.c:270-279`, the identity check the
//!   destination side already carries);
//! * `jpeg_stdio_src` reads through the caller's `FILE *`: a pre-positioned
//!   stream decodes from the current position whether it was moved with
//!   `fseek` (fd offset) or `fgetc` (stdio buffer), and after a completed
//!   decode the resting position is strictly before EOF when trailing bytes
//!   exist beyond the read-ahead — djpeg-style concatenated streams stay
//!   readable;
//! * the installed `fill_input_buffer`, driven directly through the ABI,
//!   serves one chunk and then warns `JWRN_JPEG_EOF` per dry read while
//!   handing back a fabricated `FF D9` (f4).
//!
//! `stdio_fill_after_decode_serves_trailing_bytes_first` is the one test here
//! with no oracle row: stock has no drain, so only this port can lose the
//! bytes it read past the EOI, and the pin is internal-consistency by
//! necessity.

use std::cell::{Cell, RefCell};
use std::ffi::{c_int, c_long, c_void, CString};

mod helpers;

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_destroy_compress, jpeg_destroy_decompress, jpeg_finish_compress, jpeg_finish_decompress,
    jpeg_mem_dest, jpeg_mem_src, jpeg_read_header, jpeg_read_scanlines, jpeg_set_defaults,
    jpeg_start_compress, jpeg_start_decompress, jpeg_std_error, jpeg_stdio_src,
    jpeg_write_scanlines, JpegCompressPublic, JpegDecompressPublic, JpegErrorMgr, JpegSourceMgr,
};

const WIDTH: usize = 64;
const HEIGHT: usize = 64;
/// `jpeglib.h`: `JCS_GRAYSCALE`.
const JCS_GRAYSCALE: c_int = 1;

thread_local! {
    static FIRED: Cell<bool> = const { Cell::new(false) };
    static MSG_CODE: Cell<c_int> = const { Cell::new(0) };
    static WARNS: RefCell<Vec<c_int>> = const { RefCell::new(Vec::new()) };
}

/// Records warning-level (`msg_level < 0`) `emit_message` codes, mirroring
/// the oracle's `record_emit` for the f4 fill-sequence row.
unsafe extern "C" fn recording_emit_message(cinfo: *mut c_void, msg_level: c_int) {
    if msg_level < 0 {
        // SAFETY: the error path reads only the leading `err` pointer.
        unsafe {
            let err_ptr: *mut JpegErrorMgr = (cinfo as *const *mut JpegErrorMgr).read();
            let code: c_int = std::ptr::addr_of!((*err_ptr).msg_code).read();
            WARNS.with(|w| w.borrow_mut().push(code));
        }
    }
}

fn drain_warns() -> Vec<c_int> {
    WARNS.with(|w| std::mem::take(&mut *w.borrow_mut()))
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
    }
}

fn fired() -> bool {
    FIRED.with(|f| f.get())
}

fn code() -> c_int {
    MSG_CODE.with(|c| c.get())
}

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
        FIRED.with(|f| f.set(false));
        MSG_CODE.with(|c| c.set(0));
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
        // SAFETY: created in `new`.
        unsafe { jpeg_destroy_decompress(self.ptr()) };
    }
}

// Foreign source manager callbacks for m3 (never actually driven).
unsafe extern "C" fn fake_init(_c: *mut c_void) {}
unsafe extern "C" fn fake_fill(_c: *mut c_void) -> c_int {
    0
}
unsafe extern "C" fn fake_skip(_c: *mut c_void, _n: c_long) {}
unsafe extern "C" fn fake_term(_c: *mut c_void) {}

/// Drain all rows then finish; append the oracle-format tokens for a
/// full-decode case.
unsafe fn decode_fully(p: *mut c_void, c: &mut JpegDecompressPublic) -> u32 {
    let mut row: Vec<u8> = vec![0; WIDTH * 4];
    // SAFETY: caller established the classic sequence.
    unsafe {
        let _ = jpeg_read_header(p, 1);
        let _ = jpeg_start_decompress(p);
        while !fired() && c.output_scanline < c.output_height {
            let mut rows: [*mut u8; 1] = [row.as_mut_ptr()];
            let got: u32 = jpeg_read_scanlines(p, rows.as_mut_ptr(), 1);
            if got == 0 {
                break;
            }
        }
        let rows_read: u32 = c.output_scanline;
        if !fired() {
            let _ = jpeg_finish_decompress(p);
        }
        rows_read
    }
}

fn our_trace() -> String {
    let src: Vec<u8> = make_source();
    let mut out = String::new();

    // m1/m2: NULL / empty input.
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: NULL buffer is the case under test.
        unsafe { jpeg_mem_src(p, std::ptr::null(), src.len() as std::ffi::c_ulong) };
        out.push_str(&if fired() {
            format!("m1 err {}\n", code())
        } else {
            "m1 ok\n".into()
        });
    }
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: zero size is the case under test.
        unsafe { jpeg_mem_src(p, src.as_ptr(), 0) };
        out.push_str(&if fired() {
            format!("m2 err {}\n", code())
        } else {
            "m2 ok\n".into()
        });
    }

    // m3: foreign manager already installed.
    {
        let mut d = Decomp::new();
        let mut foreign: JpegSourceMgr = JpegSourceMgr {
            next_input_byte: std::ptr::null(),
            bytes_in_buffer: 0,
            init_source: Some(fake_init),
            fill_input_buffer: Some(fake_fill),
            skip_input_data: Some(fake_skip),
            resync_to_restart: None,
            term_source: Some(fake_term),
        };
        d.cinfo.src = &mut foreign as *mut JpegSourceMgr;
        let p = d.ptr();
        // SAFETY: the overwrite attempt is the case under test.
        unsafe { jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong) };
        out.push_str(&if fired() {
            format!("m3 err {}\n", code())
        } else {
            "m3 ok\n".into()
        });
    }

    // m4: mem_src twice (legal reuse).
    {
        let mut d = Decomp::new();
        let p = d.ptr();
        // SAFETY: classic sequence, twice.
        unsafe {
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let _ = decode_fully(p, &mut d.cinfo);
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            let rows = decode_fully(p, &mut d.cinfo);
            out.push_str(&if fired() {
                format!("m4 err {}\n", code())
            } else {
                format!("m4 ok rows {rows}\n")
            });
        }
    }

    extern "C" {
        fn fopen(path: *const std::ffi::c_char, mode: *const std::ffi::c_char) -> *mut c_void;
        fn fclose(f: *mut c_void) -> c_int;
        fn fwrite(ptr: *const c_void, size: usize, n: usize, f: *mut c_void) -> usize;
        fn fseek(f: *mut c_void, off: c_long, whence: c_int) -> c_int;
        fn ftell(f: *mut c_void) -> c_long;
        fn fgetc(f: *mut c_void) -> c_int;
    }

    // m5/m6: cross-installing stdio over mem and mem over stdio — both
    // foreign to each other's identity check, as on stock.
    let cross_path = CString::new(format!(
        "{}/p4109_mirror_cross.bin",
        std::env::temp_dir().display()
    ))
    .expect("path");
    // SAFETY: plain C stdio over a temp path.
    unsafe {
        let mode_w = CString::new("wb").expect("mode");
        let mode_r = CString::new("rb").expect("mode");
        let w = fopen(cross_path.as_ptr(), mode_w.as_ptr());
        assert!(!w.is_null(), "fopen(cross w)");
        fwrite(src.as_ptr() as *const c_void, 1, src.len(), w);
        fclose(w);

        {
            let mut d = Decomp::new();
            let p = d.ptr();
            let cf = fopen(cross_path.as_ptr(), mode_r.as_ptr());
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            jpeg_stdio_src(p, cf);
            out.push_str(&if fired() {
                format!("m5 err {}\n", code())
            } else {
                "m5 ok\n".into()
            });
            fclose(cf);
        }
        {
            let mut d = Decomp::new();
            let p = d.ptr();
            let cf = fopen(cross_path.as_ptr(), mode_r.as_ptr());
            jpeg_stdio_src(p, cf);
            jpeg_mem_src(p, src.as_ptr(), src.len() as std::ffi::c_ulong);
            out.push_str(&if fired() {
                format!("m6 err {}\n", code())
            } else {
                "m6 ok\n".into()
            });
            fclose(cf);
        }
        let _ = std::fs::remove_file(std::path::PathBuf::from(
            cross_path.to_string_lossy().into_owned(),
        ));
    }

    // f1/f2: stdio cases over a prefix + JPEG + 8 KiB trailer file.
    let path = CString::new(format!(
        "{}/p4109_mirror_input.bin",
        std::env::temp_dir().display()
    ))
    .expect("path");
    let mode_w = CString::new("wb").expect("mode");
    let mode_r = CString::new("rb").expect("mode");
    let prefix: [u8; 16] = [0x55; 16];
    let trailer: Vec<u8> = (0..8192u32).map(|i| (i & 0x7F) as u8).collect();
    let fsize: c_long = (16 + src.len() + 8192) as c_long;
    // SAFETY: plain C stdio over a temp path; the file outlives each case.
    unsafe {
        let w = fopen(path.as_ptr(), mode_w.as_ptr());
        assert!(!w.is_null(), "fopen(w)");
        fwrite(prefix.as_ptr() as *const c_void, 1, prefix.len(), w);
        fwrite(src.as_ptr() as *const c_void, 1, src.len(), w);
        fwrite(trailer.as_ptr() as *const c_void, 1, trailer.len(), w);
        fclose(w);

        // f1: fseek past the prefix.
        {
            let mut d = Decomp::new();
            let p = d.ptr();
            let f = fopen(path.as_ptr(), mode_r.as_ptr());
            assert!(!f.is_null(), "fopen(f1)");
            fseek(f, 16, 0 /* SEEK_SET */);
            jpeg_stdio_src(p, f);
            let rows = decode_fully(p, &mut d.cinfo);
            if fired() {
                out.push_str(&format!("f1 err {}\n", code()));
            } else {
                let pos: c_long = ftell(f);
                out.push_str(&format!(
                    "f1 rows {rows} pos_class {}\n",
                    if pos < 0 {
                        "err"
                    } else if pos >= fsize {
                        "eof"
                    } else {
                        "before_eof"
                    }
                ));
            }
            fclose(f);
        }

        // f2: consume the prefix through the stdio buffer with fgetc.
        {
            let mut d = Decomp::new();
            let p = d.ptr();
            let f = fopen(path.as_ptr(), mode_r.as_ptr());
            assert!(!f.is_null(), "fopen(f2)");
            for _ in 0..16 {
                let _ = fgetc(f);
            }
            jpeg_stdio_src(p, f);
            let rows = decode_fully(p, &mut d.cinfo);
            if fired() {
                out.push_str(&format!("f2 err {}\n", code()));
            } else {
                out.push_str(&format!("f2 rows {rows}\n"));
            }
            fclose(f);
        }

        let _ = std::fs::remove_file(std::path::PathBuf::from(
            path.to_string_lossy().into_owned(),
        ));

        // f3: I/O failure — an empty FILE yields zero bytes on the very
        // first chunked read; stock ERREXITs with JERR_INPUT_EMPTY from
        // fill_input_buffer's start_of_file branch when jpeg_read_header
        // pulls the first marker.
        {
            let empty_path =
                std::env::temp_dir().join(format!("p4109_mirror_empty_{}.bin", std::process::id()));
            std::fs::write(&empty_path, b"").expect("create empty file");
            let empty_c = CString::new(empty_path.to_str().expect("utf8 path")).expect("no NUL");
            let mut d = Decomp::new();
            let p = d.ptr();
            let f = fopen(empty_c.as_ptr(), mode_r.as_ptr());
            assert!(!f.is_null(), "fopen(f3)");
            jpeg_stdio_src(p, f);
            let _ = jpeg_read_header(p, 1);
            if fired() {
                out.push_str(&format!("f3 err {}\n", code()));
            } else {
                out.push_str("f3 ok\n");
            }
            fclose(f);
            let _ = std::fs::remove_file(&empty_path);
        }

        // f4: the installed fill_input_buffer driven directly through the
        // ABI. One chunked read serves the whole 6-byte file; each dry
        // read after it warns JWRN_JPEG_EOF and fabricates FF D9
        // (jdatasrc.c:106-118).
        {
            let fill_path =
                std::env::temp_dir().join(format!("p4109_mirror_fill_{}.bin", std::process::id()));
            std::fs::write(&fill_path, [0x10u8, 0x20, 0x30, 0x40, 0x50, 0x60])
                .expect("create fill payload");
            let fill_c = CString::new(fill_path.to_str().expect("utf8 path")).expect("no NUL");
            let mut d = Decomp::new();
            let p = d.ptr();
            (*d.cinfo.err).emit_message = Some(recording_emit_message);
            let _ = drain_warns();
            let f = fopen(fill_c.as_ptr(), mode_r.as_ptr());
            assert!(!f.is_null(), "fopen(f4)");
            jpeg_stdio_src(p, f);
            for i in 1..=3 {
                let fill = (*d.cinfo.src).fill_input_buffer.expect("fill installed");
                let r: c_int = fill(p);
                for w in drain_warns() {
                    out.push_str(&format!("f4 warn {w}\n"));
                }
                let bib: usize = (*d.cinfo.src).bytes_in_buffer;
                let first: u8 = *(*d.cinfo.src).next_input_byte;
                out.push_str(&format!(
                    "f4 fill {i} ret {r} bib {bib} first 0x{first:02X}\n"
                ));
            }
            fclose(f);
            let _ = std::fs::remove_file(&fill_path);
        }
    }

    out
}

/// Issue #469 (P4-109): the setup/stdio matrix, compared verbatim against
/// stock libjpeg.
#[test]
fn classic_source_mgr_matches_stock_libjpeg() {
    let Some(oracle) = helpers::build_classic_oracle("classic_source_mgr_oracle") else {
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
        "classic source-manager behaviour diverges from stock libjpeg \
         (P4-109, #469)"
    );
}

/// `jpeglib.h`: `JWRN_JPEG_EOF` (cross-checked against stock by the f4
/// oracle row's `warn 123` line).
const JWRN_JPEG_EOF: c_int = 123;

/// After a drained stdio decode with trailing bytes, a direct ABI
/// `fill_input_buffer` call must serve the retained post-EOI remainder
/// before touching the `FILE` again — the drain already read those bytes,
/// so a fresh `fread` would silently skip them — and the dry read after
/// that warns `JWRN_JPEG_EOF` with the fake `FF D9`. Internal-consistency
/// pin for the shim's remainder bookkeeping (stock has no drain; its one
/// continuous reader cannot lose bytes this way).
#[test]
fn stdio_fill_after_decode_serves_trailing_bytes_first() {
    extern "C" {
        fn fopen(path: *const std::ffi::c_char, mode: *const std::ffi::c_char) -> *mut c_void;
        fn fclose(f: *mut c_void) -> c_int;
    }

    let src: Vec<u8> = make_source();
    let trailer: Vec<u8> = (0..64u8).map(|i| 0xA0u8.wrapping_add(i)).collect();
    let path =
        std::env::temp_dir().join(format!("p4109_mirror_trailing_{}.bin", std::process::id()));
    let mut file_bytes: Vec<u8> = src.clone();
    file_bytes.extend_from_slice(&trailer);
    std::fs::write(&path, &file_bytes).expect("create trailing-bytes file");
    let path_c = CString::new(path.to_str().expect("utf8 path")).expect("no NUL");
    let mode_r = CString::new("rb").expect("mode");

    let mut d = Decomp::new();
    let p = d.ptr();
    // SAFETY: classic sequence over the shim's own manager; the direct
    // fill calls are the ABI use under test.
    unsafe {
        (*d.cinfo.err).emit_message = Some(recording_emit_message);
        let _ = drain_warns();
        let f = fopen(path_c.as_ptr(), mode_r.as_ptr());
        assert!(!f.is_null(), "fopen(trailing)");
        jpeg_stdio_src(p, f);
        let rows: u32 = decode_fully(p, &mut d.cinfo);
        assert!(!fired(), "decode failed with code {}", code());
        assert_eq!(rows, HEIGHT as u32, "full decode before the fill probe");

        let fill = (*d.cinfo.src).fill_input_buffer.expect("fill installed");

        // First fill: the 64 trailing bytes the drain read past the EOI,
        // in order, not a skipped-ahead fread window.
        assert_eq!(fill(p), 1, "remainder-serving fill returns TRUE");
        assert!(drain_warns().is_empty(), "no warning while bytes remain");
        let bib: usize = (*d.cinfo.src).bytes_in_buffer;
        assert_eq!(bib, trailer.len(), "whole remainder served");
        let served: &[u8] = std::slice::from_raw_parts((*d.cinfo.src).next_input_byte, bib);
        assert_eq!(served, &trailer[..], "remainder bytes served verbatim");

        // Second fill: the stream is dry — stock's warn-and-fake-EOI path.
        assert_eq!(fill(p), 1, "dry fill returns TRUE with the fake EOI");
        assert_eq!(
            drain_warns(),
            vec![JWRN_JPEG_EOF],
            "dry read warns JWRN_JPEG_EOF exactly once"
        );
        assert_eq!((*d.cinfo.src).bytes_in_buffer, 2, "fake EOI window");
        assert_eq!(*(*d.cinfo.src).next_input_byte, 0xFF, "fake EOI first byte");

        fclose(f);
    }
    let _ = std::fs::remove_file(&path);
}
