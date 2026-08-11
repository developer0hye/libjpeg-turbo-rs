//! A1-12: `cinfo.arith_code` is correctly populated by `jpeg_read_header`.
//!
//! Classic-API callers that branch on `cinfo.arith_code` (e.g. to pick a
//! codec path or to print stream metadata) must see `1` for arithmetic-coded
//! JPEGs and `0` for Huffman-coded JPEGs.
//!
//! Per ISO 10918-1 Table B.1, the entropy-coding family is determined by bit 3
//! of `(SOF_marker & 0x0F)`: arithmetic markers are SOF9/SOF10/SOF11
//! (0xC9–0xCB) and the differential variants SOF13/SOF14/SOF15 (0xCD–0xCF).
//! All others (SOF0–SOF3, SOF5–SOF7) are Huffman-coded.
//!
//! The shim previously hardcoded `arith_code = 0` regardless of the actual
//! SOF marker; these tests pin the correct behaviour.

use libjpeg_turbo_rs_capi::jpeglib::JpegDecompressPublic;
use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

// libjpeg `jpeg_read_header` return codes.
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

/// Run `jpeg_create_decompress` → `jpeg_mem_src` → `jpeg_read_header` →
/// `jpeg_capi_test_arith_code` → `jpeg_destroy_decompress` on `jpeg_bytes`,
/// and return the `arith_code` value the shim put in `cinfo`.
fn read_arith_code_flag(lib: &libloading::Library, jpeg_bytes: &[u8]) -> c_int {
    unsafe {
        let mut cinfo: MaybeUninit<JpegDecompressPublic> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;

        const ERR_BYTES: usize = 512;
        let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

        // Set up the error manager (`err` is the first field of cinfo).
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let err_ret: *mut c_void = jpeg_std_error(err_ptr);
        assert_eq!(err_ret, err_ptr, "jpeg_std_error must return its argument");
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);

        // `jpeg_create_decompress` expands to
        // `jpeg_CreateDecompress(cinfo, JPEG_LIB_VERSION, sizeof_struct)`.
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        let jpeg_lib_version: c_int = 80; // JPEG_LIB_VERSION for libjpeg-turbo 3.x
        jpeg_create_decompress(
            cinfo_ptr,
            jpeg_lib_version,
            std::mem::size_of::<JpegDecompressPublic>(),
        );

        // Point the decoder at the in-memory JPEG bytes.
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg_bytes.as_ptr(), jpeg_bytes.len() as c_ulong);

        // Parse the header.
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        let rc: c_int = jpeg_read_header(cinfo_ptr, 1);
        assert_eq!(rc, JPEG_HEADER_OK, "jpeg_read_header must succeed");

        // Read back the arith_code field via the test accessor.
        let get_arith_code: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> = lib
            .get(b"jpeg_capi_test_arith_code")
            .expect("jpeg_capi_test_arith_code");
        let arith_code: c_int = get_arith_code(cinfo_ptr);

        // Tear down cleanly.
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);

        arith_code
    }
}

/// `testimgari.jpg` is an arithmetic-coded JPEG (SOF9). After
/// `jpeg_read_header`, `cinfo.arith_code` must be 1.
#[test]
fn arith_code_is_1_for_arithmetic_coded_jpeg() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    // testimgari.jpg — arithmetic-coded (SOF9).
    // Locate via CARGO_MANIFEST_DIR (crates/libjpeg-turbo-rs-capi/) → ../../references/
    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let testimgari: PathBuf =
        manifest_dir.join("../../references/libjpeg-turbo/testimages/testimgari.jpg");
    if !testimgari.exists() {
        eprintln!(
            "SKIP: testimgari.jpg not found at {} — submodule not initialised",
            testimgari.display()
        );
        return;
    }
    let jpeg_bytes: Vec<u8> = std::fs::read(&testimgari)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", testimgari.display()));

    let arith_code: c_int = read_arith_code_flag(&lib, &jpeg_bytes);
    assert_eq!(
        arith_code, 1,
        "arith_code must be 1 for arithmetic-coded JPEG (got {arith_code})"
    );
}

/// `testimgint.jpg` is a progressive Huffman-coded JPEG (SOF2). After
/// `jpeg_read_header`, `cinfo.arith_code` must be 0.
#[test]
fn arith_code_is_0_for_huffman_coded_jpeg() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    // testimgint.jpg — progressive Huffman-coded (SOF2).
    // Locate via CARGO_MANIFEST_DIR (crates/libjpeg-turbo-rs-capi/) → ../../references/
    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let testimgint: PathBuf =
        manifest_dir.join("../../references/libjpeg-turbo/testimages/testimgint.jpg");
    if !testimgint.exists() {
        eprintln!(
            "SKIP: testimgint.jpg not found at {} — submodule not initialised",
            testimgint.display()
        );
        return;
    }
    let jpeg_bytes: Vec<u8> = std::fs::read(&testimgint)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", testimgint.display()));

    let arith_code: c_int = read_arith_code_flag(&lib, &jpeg_bytes);
    assert_eq!(
        arith_code, 0,
        "arith_code must be 0 for Huffman-coded JPEG (got {arith_code})"
    );
}

/// `testorig.jpg` is a baseline Huffman-coded JPEG (SOF0). After
/// `jpeg_read_header`, `cinfo.arith_code` must be 0.  This pins the baseline
/// (SOF0) case explicitly, separate from the progressive (SOF2) case above.
#[test]
fn arith_code_is_0_for_baseline_sof0() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture: PathBuf =
        manifest_dir.join("../../references/libjpeg-turbo/testimages/testorig.jpg");
    if !fixture.exists() {
        eprintln!(
            "SKIP: testorig.jpg not found at {} — submodule not initialised",
            fixture.display()
        );
        return;
    }
    let jpeg_bytes: Vec<u8> = std::fs::read(&fixture)
        .unwrap_or_else(|e| panic!("could not read {}: {e}", fixture.display()));

    let arith_code: c_int = read_arith_code_flag(&lib, &jpeg_bytes);
    assert_eq!(
        arith_code, 0,
        "arith_code must be 0 for baseline SOF0 JPEG (got {arith_code})"
    );
}
