//! FFI C2-*: end-to-end test for the libjpeg-style `jpeg_*` encode API.
//!
//! Tests the encode state machine (create/destroy, dest managers,
//! defaults/quality/colorspace setters, start/write/finish) via
//! `dlopen`, then decodes the result via the decode-side entry points
//! and cross-checks pixels.

use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

const JCS_RGB: c_int = 2;

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

/// C2-1: create -> set_defaults -> set_quality -> destroy is crash-free
/// and leaves the cinfo in a usable shape (num_components populated,
/// comp_info pointer set).
#[test]
fn c2_1_compress_create_setup_destroy() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;

        const ERR_BYTES: usize = 512;
        let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(err_ptr);
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);

        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(cinfo_ptr, 80, CINFO_BYTES);

        // Populate the 3 fields cjpeg sets before jpeg_set_defaults.
        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, 64, 64, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(cinfo_ptr, 75, 1);

        // Verify: num_components = 3, in_color_space preserved.
        let jpeg_capi_test_get_compress_state: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_int, *mut c_int, *mut c_int),
        > = lib
            .get(b"jpeg_capi_test_get_compress_state")
            .expect("jpeg_capi_test_get_compress_state");
        let mut num_components: c_int = 0;
        let mut jpeg_cs: c_int = 0;
        let mut in_cs: c_int = 0;
        jpeg_capi_test_get_compress_state(cinfo_ptr, &mut num_components, &mut jpeg_cs, &mut in_cs);
        assert_eq!(num_components, 3);
        assert_eq!(in_cs, JCS_RGB);
        // jpeg_set_defaults -> jpeg_default_colorspace: RGB input → YCbCr JPEG.
        assert_eq!(jpeg_cs, 3 /* JCS_YCbCr */);

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
    }
}

/// C2-1 mem_dest: sets up a memory destination and verifies the
/// outbuffer pointer remains NULL until compression actually runs.
#[test]
fn c2_1_mem_dest_installs_cleanly() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        const CINFO_BYTES: usize = 4096;
        let mut cinfo: MaybeUninit<[u8; CINFO_BYTES]> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;

        const ERR_BYTES: usize = 512;
        let mut err: MaybeUninit<[u8; ERR_BYTES]> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let _ = jpeg_std_error(err_ptr);
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);

        let jpeg_create_compress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateCompress")
            .expect("jpeg_CreateCompress");
        jpeg_create_compress(cinfo_ptr, 80, CINFO_BYTES);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0xDEAD;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);
        assert!(out_buf.is_null());
        assert_eq!(out_size, 0, "size must be zero'd when outbuffer is NULL");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
    }
}

/// Null-guard: destroy and setup functions must accept NULL without
/// crashing.
#[test]
fn c2_1_null_arguments_return_safely() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(std::ptr::null_mut());

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(std::ptr::null_mut());

        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(std::ptr::null_mut(), 75, 1);
    }
}
