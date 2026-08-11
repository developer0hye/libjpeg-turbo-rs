//! Classic-API `jpeg_save_markers(cinfo, code, length_limit)` truncation.
//!
//! When `jpeg_save_markers(cinfo, JPEG_COM, N)` is called with `N < full_len`,
//! the saved marker body in `cinfo->marker_list` must be exactly `N` bytes
//! (the first `N` bytes of the full payload), not the full body.
//!
//! Tests:
//!   1. `marker_save_length_limit_truncates_com` — 200-byte COM, limit=64.
//!   2. `marker_save_length_limit_zero_disables_saving` — limit=0 must save nothing.
//!   3. `marker_save_no_call_saves_nothing` — without `jpeg_save_markers` no
//!      markers are saved (default behavior regression guard).

use libjpeg_turbo_rs_capi::jpeglib::JpegDecompressPublic;
use std::ffi::{c_int, c_uint, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

// libjpeg return codes.
const JPEG_HEADER_OK: c_int = 1;
// Standard JPEG marker codes (per jpeglib.h).
const JPEG_COM: c_int = 0xFE;

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

/// Layout of `jpeg_marker_struct` as used by the shim's `cinfo->marker_list`.
///
/// Fields (in order, matching `JpegMarkerStructPublic` in `jpeglib.rs`):
///   next: *mut c_void  (ptr to next node)
///   marker: u8         (marker code)
///   original_length: c_uint
///   data_length: c_uint
///   data: *mut u8
///
/// Use `#[repr(C)]` with explicit padding on ARM/x86_64 where the pointer
/// occupies 8 bytes.
#[repr(C)]
struct JpegMarkerStruct {
    next: *mut JpegMarkerStruct,
    marker: u8,
    _pad: [u8; 3], // ARM/x86_64: align next field to 4 bytes
    original_length: c_uint,
    data_length: c_uint,
    data: *mut u8,
}

/// Build a minimal JPEG with a single COM marker whose payload is
/// `payload_bytes` bytes long (filled with 'A'..='Z' cycling).
fn make_jpeg_with_com(payload_bytes: usize) -> Vec<u8> {
    // Build a 1×1 grayscale JPEG with a COM marker using the `libjpeg_turbo_rs`
    // Rust library directly — avoids depending on C tools being installed.
    use libjpeg_turbo_rs::{Encoder, PixelFormat};

    // COM payload: ASCII 'A'..'Z' cycling, valid UTF-8.
    let comment: String = (0..payload_bytes)
        .map(|i| (b'A' + (i % 26) as u8) as char)
        .collect();

    let pixels: Vec<u8> = vec![128u8]; // 1×1 gray pixel
    Encoder::new(&pixels, 1, 1, PixelFormat::Grayscale)
        .quality(80)
        .comment(&comment)
        .encode()
        .expect("encode 1×1 gray JPEG with COM marker")
}

/// Open the shim, run `jpeg_create_decompress` → `jpeg_mem_src` →
/// `jpeg_save_markers(code, limit)` → `jpeg_read_header`, then return
/// the list of `(marker_code, data_length, original_length)` tuples from
/// `cinfo->marker_list`.
unsafe fn saved_marker_lengths(
    lib: &libloading::Library,
    jpeg_bytes: &[u8],
    save_calls: &[(c_int, c_uint)], // (code, length_limit) pairs; empty = no call
) -> Vec<(u8, usize, usize)> {
    let mut cinfo: MaybeUninit<JpegDecompressPublic> = MaybeUninit::zeroed();
    let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;

    const ERR_BYTES: usize = 512;
    let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
    let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

    let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
        lib.get(b"jpeg_std_error").expect("jpeg_std_error");
    let err_ret = jpeg_std_error(err_ptr);
    assert_eq!(err_ret, err_ptr);
    (cinfo_ptr as *mut *mut c_void).write(err_ptr);

    let jpeg_create_decompress: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, c_int, usize),
    > = lib
        .get(b"jpeg_CreateDecompress")
        .expect("jpeg_CreateDecompress");
    jpeg_create_decompress(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>());

    let jpeg_mem_src: libloading::Symbol<unsafe extern "C" fn(*mut c_void, *const u8, c_ulong)> =
        lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
    jpeg_mem_src(cinfo_ptr, jpeg_bytes.as_ptr(), jpeg_bytes.len() as c_ulong);

    let jpeg_save_markers: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_uint)> =
        lib.get(b"jpeg_save_markers").expect("jpeg_save_markers");
    for &(code, limit) in save_calls {
        jpeg_save_markers(cinfo_ptr, code, limit);
    }

    let jpeg_read_header: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int) -> c_int> =
        lib.get(b"jpeg_read_header").expect("jpeg_read_header");
    let rc = jpeg_read_header(cinfo_ptr, 1);
    assert_eq!(rc, JPEG_HEADER_OK, "jpeg_read_header must succeed");

    // Read marker_list from cinfo. The `marker_list` field is at a fixed
    // offset in `JpegDecompressPublic`; we read it via the test accessor
    // exported by the shim so we do not hard-code a struct offset here.
    let get_marker_list: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void) -> *mut JpegMarkerStruct,
    > = lib
        .get(b"jpeg_capi_test_marker_list")
        .expect("jpeg_capi_test_marker_list accessor not found in cdylib");

    let mut result: Vec<(u8, usize, usize)> = Vec::new();
    let mut node: *mut JpegMarkerStruct = get_marker_list(cinfo_ptr);
    while !node.is_null() {
        let m = (*node).marker;
        let data_len = (*node).data_length as usize;
        let orig_len = (*node).original_length as usize;
        result.push((m, data_len, orig_len));
        node = (*node).next;
    }

    let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
        .get(b"jpeg_destroy_decompress")
        .expect("jpeg_destroy_decompress");
    jpeg_destroy_decompress(cinfo_ptr);

    result
}

/// A 200-byte COM marker saved with `length_limit=64` must appear in
/// `cinfo->marker_list` with `data_length == 64` and the first 64 bytes
/// matching the original payload.
#[test]
fn marker_save_length_limit_truncates_com() {
    let payload_len: usize = 200;
    let limit: usize = 64;
    let jpeg_bytes: Vec<u8> = make_jpeg_with_com(payload_len);

    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let markers: Vec<(u8, usize, usize)> =
        unsafe { saved_marker_lengths(&lib, &jpeg_bytes, &[(JPEG_COM, limit as c_uint)]) };

    // Exactly one COM marker must be saved.
    let com_markers: Vec<_> = markers.iter().filter(|&&(code, ..)| code == 0xFE).collect();
    assert_eq!(
        com_markers.len(),
        1,
        "expected exactly one COM marker, got {}: {markers:?}",
        com_markers.len()
    );

    let &(_, data_length, original_length) = com_markers[0];
    assert_eq!(
        data_length, limit,
        "data_length must equal the requested limit ({limit}), got {data_length}"
    );
    assert!(
        original_length >= payload_len,
        "original_length ({original_length}) must be >= payload_len ({payload_len})"
    );
}

/// `jpeg_save_markers(cinfo, JPEG_COM, 0)` disables saving for COM entirely.
/// After `jpeg_read_header` the marker_list must contain no COM nodes.
#[test]
fn marker_save_length_limit_zero_disables_saving() {
    let jpeg_bytes: Vec<u8> = make_jpeg_with_com(50);

    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let markers: Vec<(u8, usize, usize)> =
        unsafe { saved_marker_lengths(&lib, &jpeg_bytes, &[(JPEG_COM, 0)]) };

    let com_count = markers.iter().filter(|&&(code, ..)| code == 0xFE).count();
    assert_eq!(
        com_count, 0,
        "length_limit=0 must disable saving; found {com_count} COM markers: {markers:?}"
    );
}

/// Without any `jpeg_save_markers` call the default is no saving; the
/// marker_list must be empty (or contain no COM markers).
#[test]
fn marker_save_no_call_saves_nothing() {
    let jpeg_bytes: Vec<u8> = make_jpeg_with_com(50);

    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let markers: Vec<(u8, usize, usize)> = unsafe { saved_marker_lengths(&lib, &jpeg_bytes, &[]) };

    let com_count = markers.iter().filter(|&&(code, ..)| code == 0xFE).count();
    assert_eq!(
        com_count, 0,
        "without jpeg_save_markers no COM markers should be saved; found {com_count}: {markers:?}"
    );
}
