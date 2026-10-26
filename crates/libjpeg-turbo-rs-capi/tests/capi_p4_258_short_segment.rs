//! P4-258: the classic C ABI refuses a short entropy-coded segment.
//!
//! Since P4-254 the strict `Decoder` behind `jpeg_start_decompress` refuses
//! entropy-coded data that ends before the last MCU. Stock libjpeg-turbo
//! decodes the rest of the segment grey and warns `JWRN_HIT_MARKER` through
//! `emit_message` instead (`jdhuff.c` `jpeg_fill_bit_buffer`), which
//! `tests/cross_check_p4_254_short_entropy_segment.rs` shows against djpeg.
//!
//! This pins what the shim does today — `JERR_INPUT_EOF` from
//! `jpeg_start_decompress` — so the C ABI half of P4-258 is measured, not
//! inferred by reading. Closing P4-258 flips this test: it should then
//! decode every scanline and report the warning.

use std::cell::Cell;
use std::ffi::{c_int, c_void};
use std::path::PathBuf;

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_destroy_decompress, jpeg_mem_src, jpeg_read_header, jpeg_start_decompress, jpeg_std_error,
    JpegDecompressPublic, JpegErrorMgr,
};

/// `jerror.h`: `JERR_INPUT_EOF`, "Premature end of input file".
const JERR_INPUT_EOF: c_int = 44;
/// `jpeg_read_header`'s `JPEG_HEADER_OK`.
const JPEG_HEADER_OK: c_int = 1;

thread_local! {
    static FIRED: Cell<bool> = const { Cell::new(false) };
    static MSG_CODE: Cell<c_int> = const { Cell::new(0) };
}

/// Records the first failure. The shim returns after `error_exit` instead of
/// requiring a `longjmp`, so the test checks `FIRED` after each call.
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

/// `testorig.jpg` cut at 60% of its entropy-coded data, with an EOI after the
/// cut: the measurement P4-258 records for `TjHandle::decompress`.
fn short_segment_stream() -> Option<Vec<u8>> {
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../references/libjpeg-turbo/testimages/testorig.jpg");
    let jpeg: Vec<u8> = std::fs::read(&path).ok()?;
    let sos: usize = jpeg
        .windows(2)
        .position(|marker| marker == [0xFF, 0xDA])
        .expect("testorig.jpg has an SOS");
    let header_end: usize =
        sos + 2 + usize::from(u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]));
    let entropy_len: usize = jpeg.len() - 2 - header_end;
    let mut cut: usize = header_end + entropy_len * 60 / 100;
    // Never cut between a 0xFF and its stuffed 0x00.
    if jpeg[cut - 1] == 0xFF {
        cut -= 1;
    }
    let mut stream: Vec<u8> = jpeg[..cut].to_vec();
    stream.extend_from_slice(&[0xFF, 0xD9]);
    Some(stream)
}

/// P4-258: `jpeg_start_decompress` raises `JERR_INPUT_EOF` on a short
/// segment that stock libjpeg-turbo decodes with a warning.
#[test]
fn classic_abi_refuses_a_short_segment_with_input_eof() {
    let Some(stream) = short_segment_stream() else {
        eprintln!("SKIP: testorig.jpg not found — submodule not initialised");
        return;
    };
    FIRED.with(|f| f.set(false));
    MSG_CODE.with(|c| c.set(0));
    let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let cinfo_ptr: *mut c_void = &mut *cinfo as *mut JpegDecompressPublic as *mut c_void;

    // SAFETY: the documented classic decode sequence over a zeroed mirror;
    // `stream` outlives every call that reads it.
    let (header, header_fired, start_fired, code): (c_int, bool, bool, c_int) = unsafe {
        let errp: *mut JpegErrorMgr = jpeg_std_error(&mut err);
        (*errp).error_exit = Some(recording_error_exit);
        cinfo.err = errp;
        libjpeg_turbo_rs_capi::jpeglib::jpeg_CreateDecompress(
            cinfo_ptr,
            80,
            std::mem::size_of::<JpegDecompressPublic>(),
        );
        jpeg_mem_src(
            cinfo_ptr,
            stream.as_ptr(),
            stream.len() as std::ffi::c_ulong,
        );
        let header: c_int = jpeg_read_header(cinfo_ptr, 1);
        let header_fired: bool = FIRED.with(|f| f.get());
        let _ = jpeg_start_decompress(cinfo_ptr);
        let start_fired: bool = FIRED.with(|f| f.get());
        let code: c_int = MSG_CODE.with(|c| c.get());
        jpeg_destroy_decompress(cinfo_ptr);
        (header, header_fired, start_fired, code)
    };
    assert_eq!(header, JPEG_HEADER_OK, "the header is intact");
    assert!(
        !header_fired,
        "jpeg_read_header must not fail (code {code})"
    );
    assert!(
        start_fired,
        "P4-258 closed? jpeg_start_decompress accepted the short segment; \
         flip this test to assert the decoded image and JWRN_HIT_MARKER"
    );
    assert_eq!(code, JERR_INPUT_EOF, "jpeg_start_decompress error code");
}
