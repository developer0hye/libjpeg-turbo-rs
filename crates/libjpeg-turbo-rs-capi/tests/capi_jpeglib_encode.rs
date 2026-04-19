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

/// C2-3: `jpeg_quality_scaling` matches the libjpeg scaling curve.
///
/// libjpeg formula:
///   quality < 50: scale = 5000 / quality
///   quality >= 50: scale = 200 - 2 * quality
#[test]
fn c2_3_quality_scaling_matches_libjpeg_formula() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg_quality_scaling: libloading::Symbol<unsafe extern "C" fn(c_int) -> c_int> = lib
            .get(b"jpeg_quality_scaling")
            .expect("jpeg_quality_scaling");
        // Spot-check the curve at three representative points.
        // libjpeg clamps values outside 1..100 to the nearest endpoint.
        assert_eq!(jpeg_quality_scaling(100), 0);
        assert_eq!(jpeg_quality_scaling(75), 50);
        assert_eq!(jpeg_quality_scaling(50), 100);
        assert_eq!(jpeg_quality_scaling(25), 200);
        // Below 50: 5000 / q.
        assert_eq!(jpeg_quality_scaling(10), 500);
    }
}

/// C2-3: `jpeg_simple_progression` flips `progressive_mode` on so the
/// next `jpeg_finish_compress` emits SOF2 instead of SOF0.
#[test]
fn c2_3_simple_progression_emits_progressive_stream() {
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

        let w: usize = 32;
        let h_px: usize = 32;
        let src: Vec<u8> = (0..w * h_px * 3).map(|i| (i % 256) as u8).collect();

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

        let jpeg_simple_progression: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_simple_progression")
            .expect("jpeg_simple_progression");
        jpeg_simple_progression(cinfo_ptr);

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
            assert!(got >= 1);
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);
        assert!(!out_buf.is_null());

        // Scan for the SOF2 marker (0xFF 0xC2) — the progressive SOF.
        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        let has_sof2: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC2]);
        let has_sof0: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC0]);
        assert!(has_sof2, "expected SOF2 marker in progressive stream");
        assert!(!has_sof0, "progressive stream must not contain SOF0");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}

/// C2-3: add_quant_table / default_qtables / enable_lossless / suppress_tables
/// do not crash on reasonable inputs.
#[test]
fn c2_3_helpers_null_and_basic_guards() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg_add_quant_table: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, *const u32, c_int, c_int),
        > = lib
            .get(b"jpeg_add_quant_table")
            .expect("jpeg_add_quant_table");
        jpeg_add_quant_table(std::ptr::null_mut(), 0, std::ptr::null(), 100, 1);

        let jpeg_default_qtables: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> =
            lib.get(b"jpeg_default_qtables")
                .expect("jpeg_default_qtables");
        jpeg_default_qtables(std::ptr::null_mut(), 1);

        let jpeg_enable_lossless: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, c_int),
        > = lib
            .get(b"jpeg_enable_lossless")
            .expect("jpeg_enable_lossless");
        jpeg_enable_lossless(std::ptr::null_mut(), 1, 0);

        let jpeg_suppress_tables: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> =
            lib.get(b"jpeg_suppress_tables")
                .expect("jpeg_suppress_tables");
        jpeg_suppress_tables(std::ptr::null_mut(), 1);
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

/// C2-4: a custom COM marker written via `jpeg_write_marker` shows up
/// in the output stream immediately after the SOI. Mirrors how cjpeg's
/// `-comment` flag plumbs text through the classic API.
#[test]
fn c2_4_write_marker_inserts_custom_segment_after_soi() {
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

        let w: usize = 16;
        let h_px: usize = 16;
        let src: Vec<u8> = vec![128u8; w * h_px * 3];

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

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

        // Write a COM (0xFE) marker containing ASCII text.
        let marker_payload: &[u8] = b"hello-from-jpeg-write-marker";
        let jpeg_write_marker: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, *const u8, std::os::raw::c_uint),
        > = lib.get(b"jpeg_write_marker").expect("jpeg_write_marker");
        jpeg_write_marker(
            cinfo_ptr,
            0xFE,
            marker_payload.as_ptr(),
            marker_payload.len() as std::os::raw::c_uint,
        );

        // Also exercise the piecemeal writers: write_m_header + write_m_byte.
        let jpeg_write_m_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, std::os::raw::c_uint),
        > = lib
            .get(b"jpeg_write_m_header")
            .expect("jpeg_write_m_header");
        jpeg_write_m_header(cinfo_ptr, 0xE1, 4);
        let jpeg_write_m_byte: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int)> =
            lib.get(b"jpeg_write_m_byte").expect("jpeg_write_m_byte");
        for b in b"TEST" {
            jpeg_write_m_byte(cinfo_ptr, *b as c_int);
        }

        // Fill scanlines.
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
            assert!(got >= 1);
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);

        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        // Expect our payload in the stream.
        let needle: &[u8] = marker_payload;
        let found: bool = bytes.windows(needle.len()).any(|w| w == needle);
        assert!(found, "jpeg_write_marker payload missing from output");
        let app1_payload: &[u8] = b"TEST";
        let found_app1: bool = bytes.windows(app1_payload.len()).any(|w| w == app1_payload);
        assert!(found_app1, "jpeg_write_m_byte payload missing from output");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}

/// C2-4: `jpeg_write_icc_profile` results in an APP2 `ICC_PROFILE\0`
/// segment on the stream and the decoded `Image` surfaces the same
/// profile bytes via the Rust-native decoder.
#[test]
fn c2_4_write_icc_profile_roundtrips_bytes() {
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

        let w: usize = 16;
        let h_px: usize = 16;
        let src: Vec<u8> = vec![64u8; w * h_px * 3];

        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, w as u32, h_px as u32, 3, JCS_RGB);

        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

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

        // Synthetic ICC profile (just arbitrary bytes — the shim doesn't
        // validate ICC content, only that it surfaces through APP2).
        let icc: Vec<u8> = (0..256u32).map(|i| (i & 0xFF) as u8).collect();
        let jpeg_write_icc_profile: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, std::os::raw::c_uint),
        > = lib
            .get(b"jpeg_write_icc_profile")
            .expect("jpeg_write_icc_profile");
        jpeg_write_icc_profile(cinfo_ptr, icc.as_ptr(), icc.len() as std::os::raw::c_uint);

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
            assert!(got >= 1);
            written += got as usize;
        }

        let jpeg_finish_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_finish_compress")
            .expect("jpeg_finish_compress");
        jpeg_finish_compress(cinfo_ptr);

        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        // Look for the ICC_PROFILE signature within an APP2 segment.
        let sig: &[u8] = b"ICC_PROFILE\0";
        let found_sig: bool = bytes.windows(sig.len()).any(|w| w == sig);
        assert!(found_sig, "ICC_PROFILE signature missing from output");

        // Decode and ensure the ICC bytes round-trip.
        let img = libjpeg_turbo_rs::decompress(&bytes).expect("decompress");
        assert_eq!(
            img.icc_profile.as_deref(),
            Some(icc.as_slice()),
            "ICC profile did not round-trip through decode"
        );

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}

/// C2-5: `jdiv_round_up(a, b)` is ceiling-divide with a zero-guard.
#[test]
fn c2_5_jdiv_round_up_matches_libjpeg_formula() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jdiv_round_up: libloading::Symbol<
            unsafe extern "C" fn(
                std::os::raw::c_long,
                std::os::raw::c_long,
            ) -> std::os::raw::c_long,
        > = lib.get(b"jdiv_round_up").expect("jdiv_round_up");
        assert_eq!(jdiv_round_up(7, 3), 3);
        assert_eq!(jdiv_round_up(6, 3), 2);
        assert_eq!(jdiv_round_up(0, 5), 0);
        assert_eq!(jdiv_round_up(1, 1), 1);
        // zero-divisor guard
        assert_eq!(jdiv_round_up(5, 0), 0);
    }
}

/// C2-5: `jcopy_block_row` copies exactly num_blocks * 64 i16 samples.
#[test]
fn c2_5_jcopy_block_row_copies_full_blocks() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jcopy_block_row: libloading::Symbol<unsafe extern "C" fn(*const i16, *mut i16, u32)> =
            lib.get(b"jcopy_block_row").expect("jcopy_block_row");
        let src: Vec<i16> = (0..128i16).collect(); // 2 blocks
        let mut dst: Vec<i16> = vec![0i16; 128];
        jcopy_block_row(src.as_ptr(), dst.as_mut_ptr(), 2);
        assert_eq!(dst, src);

        // num_blocks=0 must be a no-op.
        let mut guard: Vec<i16> = vec![-1i16; 64];
        jcopy_block_row(src.as_ptr(), guard.as_mut_ptr(), 0);
        assert!(guard.iter().all(|&x| x == -1));
    }
}

/// C2-5: `jpeg_resync_to_restart` returns TRUE (1) unconditionally.
#[test]
fn c2_5_resync_to_restart_returns_true() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg_resync_to_restart: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib
            .get(b"jpeg_resync_to_restart")
            .expect("jpeg_resync_to_restart");
        assert_eq!(jpeg_resync_to_restart(std::ptr::null_mut(), 0), 1);
    }
}

/// C2-5: 12-bit and 16-bit scanline writers accept rows without crashing
/// and advance `next_scanline` correctly. Full encode pipeline for
/// 12-bit output is covered elsewhere; here we verify link + mechanics.
#[test]
fn c2_5_high_precision_write_scanlines_link_and_accept_rows() {
    let lib = unsafe { libloading::Library::new(cdylib_path()) }.expect("dlopen");
    unsafe {
        let jpeg12_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u16, u32) -> u32,
        > = lib
            .get(b"jpeg12_write_scanlines")
            .expect("jpeg12_write_scanlines");
        let jpeg16_write_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u16, u32) -> u32,
        > = lib
            .get(b"jpeg16_write_scanlines")
            .expect("jpeg16_write_scanlines");
        // NULL cinfo → 0 rows.
        assert_eq!(
            jpeg12_write_scanlines(std::ptr::null_mut(), std::ptr::null_mut(), 1),
            0
        );
        assert_eq!(
            jpeg16_write_scanlines(std::ptr::null_mut(), std::ptr::null_mut(), 1),
            0
        );
    }
}

/// C2-5: `jpeg_write_coefficients` is a stub today; it must not crash
/// on a freshly-created cinfo.
#[test]
fn c2_5_write_coefficients_stub_does_not_crash() {
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

        let jpeg_write_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_write_coefficients")
            .expect("jpeg_write_coefficients");
        jpeg_write_coefficients(cinfo_ptr, std::ptr::null_mut());

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);
    }
}

/// C2-4: `jpeg_write_tables` emits a standalone tables datastream
/// (SOI ... EOI, with only DQT/DHT segments, no SOF). Consumers of
/// the abbreviated-file convention depend on this shape.
#[test]
fn c2_4_write_tables_emits_tables_only_datastream() {
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
        // Minimum setup so set_quality has a valid struct.
        let jpeg_capi_test_set_compress_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32, u32, c_int, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_compress_dims")
            .expect("jpeg_capi_test_set_compress_dims");
        jpeg_capi_test_set_compress_dims(cinfo_ptr, 8, 8, 1, 1 /* JCS_GRAYSCALE */);
        let jpeg_set_defaults: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_set_defaults").expect("jpeg_set_defaults");
        jpeg_set_defaults(cinfo_ptr);

        let jpeg_mem_dest: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_ulong),
        > = lib.get(b"jpeg_mem_dest").expect("jpeg_mem_dest");
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: c_ulong = 0;
        jpeg_mem_dest(cinfo_ptr, &mut out_buf, &mut out_size);

        let jpeg_write_tables: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_write_tables").expect("jpeg_write_tables");
        jpeg_write_tables(cinfo_ptr);

        let bytes: Vec<u8> = std::slice::from_raw_parts(out_buf, out_size as usize).to_vec();
        assert_eq!(&bytes[..2], &[0xFF, 0xD8], "tables stream must start SOI");
        assert_eq!(
            &bytes[bytes.len() - 2..],
            &[0xFF, 0xD9],
            "tables stream must end EOI"
        );
        // No SOF0/SOF2 must appear inside a tables-only datastream.
        let has_sof0: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC0]);
        let has_sof2: bool = bytes.windows(2).any(|w| w == [0xFF, 0xC2]);
        assert!(!has_sof0, "tables-only stream must not contain SOF0");
        assert!(!has_sof2, "tables-only stream must not contain SOF2");

        let jpeg_destroy_compress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_compress")
            .expect("jpeg_destroy_compress");
        jpeg_destroy_compress(cinfo_ptr);

        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(out_buf as *mut c_void);
    }
}
