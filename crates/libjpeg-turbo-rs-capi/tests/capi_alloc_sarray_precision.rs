//! Regression: `alloc_sarray` must size rows by the consumer's
//! `data_precision`, mirroring upstream `jmemmgr.c::alloc_sarray`
//! (J12SAMPLE/J16SAMPLE rows are 2 bytes per sample).
//!
//! Stock 12-bit consumers (djpeg's `j12init_write_ppm` → wrppm) write
//! `samplesperrow * sizeof(J12SAMPLE)` bytes into each returned row.
//! Sizing rows at 1 byte/sample under-allocates by half — a silent heap
//! overflow that the stock-tool link gate caught on `monkey12.jpg`
//! (12-bit lossless) once #351's allocation changes shifted the heap
//! layout; valgrind shows the same invalid writes on earlier commits.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::jpeglib::{
    jpeg_CreateDecompress, jpeg_destroy_decompress, jpeg_std_error, JpegDecompressPublic,
    JpegErrorMgr,
};
use libjpeg_turbo_rs_capi::memmgr::JpegMemoryMgr;

const JPOOL_IMAGE: c_int = 1;

/// Measured stride between two consecutive rows of a fresh sarray for
/// the given `data_precision`.
fn row_stride_for_precision(precision: c_int, samplesperrow: u32) -> usize {
    let mut err: JpegErrorMgr = unsafe { std::mem::zeroed() };
    let mut cinfo: JpegDecompressPublic = unsafe { std::mem::zeroed() };
    cinfo.err = unsafe { jpeg_std_error(&mut err) };
    let cinfo_ptr: *mut c_void = &mut cinfo as *mut JpegDecompressPublic as *mut c_void;
    unsafe { jpeg_CreateDecompress(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>()) };
    cinfo.data_precision = precision;

    let mgr = unsafe { &*(cinfo.mem as *const JpegMemoryMgr) };
    let alloc_sarray = mgr.alloc_sarray.expect("alloc_sarray populated");
    let rows = unsafe { alloc_sarray(cinfo_ptr, JPOOL_IMAGE, samplesperrow, 2) };
    assert!(!rows.is_null(), "alloc_sarray returned NULL");
    let row0 = unsafe { *rows } as usize;
    let row1 = unsafe { *rows.add(1) } as usize;
    unsafe { jpeg_destroy_decompress(cinfo_ptr) };
    row1 - row0
}

#[test]
fn sarray_rows_are_two_bytes_per_sample_at_12_bit() {
    // monkey12.jpg's geometry: 149 px * 3 components.
    let samplesperrow: u32 = 149 * 3;
    let stride = row_stride_for_precision(12, samplesperrow);
    assert!(
        stride >= samplesperrow as usize * 2,
        "12-bit sarray row stride {stride} < {} bytes — J12SAMPLE rows \
         need 2 bytes/sample (jmemmgr.c alloc_sarray)",
        samplesperrow * 2
    );
}

#[test]
fn sarray_rows_are_two_bytes_per_sample_at_16_bit() {
    let samplesperrow: u32 = 301;
    let stride = row_stride_for_precision(16, samplesperrow);
    assert!(
        stride >= samplesperrow as usize * 2,
        "16-bit sarray row stride {stride} < {} bytes",
        samplesperrow * 2
    );
}

#[test]
fn sarray_rows_stay_compact_at_8_bit() {
    // 8-bit rows keep the historical 1 byte/sample sizing (plus the
    // 64-byte SIMD alignment) — the fix must not double 8-bit memory.
    let samplesperrow: u32 = 640 * 3;
    let stride = row_stride_for_precision(8, samplesperrow);
    assert!(
        stride < samplesperrow as usize * 2,
        "8-bit sarray row stride {stride} unexpectedly >= 2 bytes/sample"
    );
}
