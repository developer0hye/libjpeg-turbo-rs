//! SA1 verification: `JpegDecompressPublic` must be a byte-exact mirror
//! of libjpeg's `struct jpeg_decompress_struct` (JPEG_LIB_VERSION = 80).
//!
//! This test drives the libjpeg-style API via `dlopen` and then inspects
//! the caller-allocated `cinfo` buffer at the exact byte offsets real
//! libjpeg consumers read. It pins down the fix for the previously-
//! observed `JERR_BAD_PRECISION` abort in stock `djpeg`: djpeg reads
//! `cinfo.data_precision` at offset 296 on LP64, and before this ABI
//! mirror that offset fell outside our subset and returned garbage.
//!
//! The test is gated on `target_pointer_width = "64"` and not-Windows,
//! matching the compile-time offset assertions in `jpeglib.rs`.

#![cfg(all(target_pointer_width = "64", not(windows)))]

use libjpeg_turbo_rs_capi::jpeglib::JpegDecompressPublic;
use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJINIT_COMPRESS: c_int = 1;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;
const JPEG_HEADER_OK: c_int = 1;

fn dlext() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}
fn lib_prefix() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    }
}
fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf =
            dir.join(format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix(), dlext()));
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("could not locate cdylib near {}", exe.display());
}

/// Build a small 8x8 baseline JPEG fixture via the already-validated
/// `tj3Compress8` entry point.
fn build_fixture(lib: &libloading::Library) -> Vec<u8> {
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").expect("tj3Set");
        let tj3_compress: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress8").expect("tj3Compress8");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        let h_enc: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_QUALITY, 80), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_SUBSAMP, TJSAMP_444), 0);

        let w: usize = 8;
        let h_px: usize = 8;
        let src: Vec<u8> = vec![128u8; w * h_px * 3];
        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc: c_int = tj3_compress(
            h_enc,
            src.as_ptr(),
            w as c_int,
            0,
            h_px as c_int,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
        );
        assert_eq!(rc, 0);
        let jpeg: Vec<u8> = std::slice::from_raw_parts(jpeg_buf, jpeg_size).to_vec();
        tj3_free(jpeg_buf as *mut c_void);
        tj3_destroy(h_enc);
        jpeg
    }
}

/// After `jpeg_read_header`, `cinfo.data_precision` sits at byte offset
/// 296 on LP64 (libjpeg 3.1.2 layout for `JPEG_LIB_VERSION = 80`). Real
/// `djpeg` reads this offset and aborts with `JERR_BAD_PRECISION` if it
/// doesn't match 8 / 12 / 16. This test asserts we deliver the correct
/// value at the correct byte.
#[test]
fn data_precision_is_at_libjpeg_offset_296_after_read_header() {
    const EXPECTED_DATA_PRECISION_OFFSET: usize = 296;

    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    let jpeg: Vec<u8> = build_fixture(&lib);

    unsafe {
        let mut cinfo: MaybeUninit<JpegDecompressPublic> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;

        const ERR_BYTES: usize = 512;
        let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        jpeg_std_error(err_ptr);
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);

        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        jpeg_create_decompress(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>());

        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg.as_ptr(), jpeg.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        // Read `data_precision` at the raw byte offset, as real libjpeg
        // consumers do. The field is a C `int` (4 bytes).
        let data_precision_at_offset: c_int = std::ptr::read_unaligned(
            (cinfo_ptr as *mut u8).add(EXPECTED_DATA_PRECISION_OFFSET) as *const c_int,
        );
        assert_eq!(
            data_precision_at_offset, 8,
            "data_precision at offset {EXPECTED_DATA_PRECISION_OFFSET} must be 8 \
             (read returned {data_precision_at_offset}); stock djpeg would \
             abort with JERR_BAD_PRECISION otherwise"
        );

        // `progressive_mode` follows at offset 316, `arith_code` at 320.
        let progressive_mode_at_offset: c_int =
            std::ptr::read_unaligned((cinfo_ptr as *mut u8).add(316) as *const c_int);
        assert_eq!(
            progressive_mode_at_offset, 0,
            "baseline JPEG => not progressive"
        );

        let arith_code_at_offset: c_int =
            std::ptr::read_unaligned((cinfo_ptr as *mut u8).add(320) as *const c_int);
        assert_eq!(
            arith_code_at_offset, 0,
            "Huffman-coded JPEG => arith_code is FALSE"
        );

        // `num_components` is at offset 56; verify the 3-component value
        // lands at the real libjpeg slot.
        let num_components_at_offset: c_int =
            std::ptr::read_unaligned((cinfo_ptr as *mut u8).add(56) as *const c_int);
        assert_eq!(num_components_at_offset, 3);

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}
