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

/// C2-2: the start/write/finish pipeline produces a JPEG that our own
/// decode side can read back pixel-identically to the Rust-native
/// compress function.
#[test]
fn c2_2_write_scanlines_roundtrip_pixel_matches_rust_native() {
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

        // Fill a 64x64 RGB gradient, identical to the decode-side fixture.
        let w: usize = 64;
        let h_px: usize = 64;
        let mut src: Vec<u8> = Vec::with_capacity(w * h_px * 3);
        for y in 0..h_px {
            for x in 0..w {
                src.push(x as u8);
                src.push(y as u8);
                src.push(((x + y) / 2) as u8);
            }
        }

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_set_quality: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int)> =
            lib.get(b"jpeg_set_quality").expect("jpeg_set_quality");
        jpeg_set_quality(cinfo_ptr, 75, 1);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");

        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_start_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> = lib
            .get(b"jpeg_start_compress")
            .expect("jpeg_start_compress");
        jpeg_start_compress(cinfo_ptr, 1);

        let jpeg_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_write_scanlines")
            .expect("jpeg_write_scanlines");

        let mut written: usize = 0;
        while written < h_px {
            let row_ptr: *mut u8 = src[written * w * 3..].as_ptr() as *mut u8;
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_write_scanlines(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(
                got >= 1,
                "jpeg_write_scanlines returned 0 with rows remaining"
            );
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);

        // After finish, the mem-dest outbuffer should hold a valid JPEG.
        assert!(!out_buf.is_null(), "mem_dest outbuffer was not populated");
        assert!(out_size > 100, "output too small ({out_size} bytes)");
        // SOI / EOI sanity.
        let encoded: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        assert_eq!(&encoded[..2], &[0xFF, 0xD8], "missing SOI");
        assert_eq!(&encoded[encoded.len() - 2..], &[0xFF, 0xD9], "missing EOI");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        // Cross-check: the byte stream must decode via the Rust-native
        // decoder to a pixel grid that closely matches the source gradient.
        let decoded = libjpeg_turbo_rs::decompress(&encoded).expect("decompress roundtrip");
        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h_px);

        // Quality=75 4:2:0 roundtrip: max per-channel diff ≤ 20 for
        // smooth gradients. We measure observed diff to detect regression
        // without overfitting to a single quality factor.
        let mut max_diff: u8 = 0;
        for (&a, &b) in src.iter().zip(decoded.data.iter()) {
            let d: u8 = a.abs_diff(b);
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(max_diff <= 20, "roundtrip max diff {max_diff} exceeds 20");

        // The C-path JPEG and a Rust-path JPEG from the same pixels and
        // parameters should also round-trip to the same decoded output,
        // i.e. both paths call into the same encode pipeline.
        let native_jpeg = libjpeg_turbo_rs::compress(
            &src,
            w,
            h_px,
            libjpeg_turbo_rs::PixelFormat::Rgb,
            75,
            libjpeg_turbo_rs::Subsampling::S420,
        )
        .expect("native compress");
        assert_eq!(
            encoded, native_jpeg,
            "classic jpeg_* bytes diverge from Rust-native bytes"
        );

        // Release the libc-malloc'd buffer.
        let libc_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        libc_free(out_buf as *mut c_void);
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
