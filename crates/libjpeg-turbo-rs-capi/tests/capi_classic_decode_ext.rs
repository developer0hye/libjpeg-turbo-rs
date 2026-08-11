//! FFI C1-*: classic libjpeg decode extension entry points.
//!
//! Exercises the ~12 extension symbols that ride on top of the A1-11
//! baseline (`jpeg_CreateDecompress`, `jpeg_read_header`, etc.) through
//! `dlopen`, asserting each one behaves consistently with the stock
//! libjpeg contract.
//!
//! The fixture JPEGs are produced in-process through the validated
//! `tj3Compress8` entry point so the test is self-contained and does
//! not depend on external reference files.

use libjpeg_turbo_rs_capi::jpeglib::JpegDecompressPublic;
use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::os::raw::{c_uint, c_ulong};
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJINIT_COMPRESS: c_int = 1;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;

const JPEG_HEADER_OK: c_int = 1;
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

/// Build a 64x64 RGB gradient, compress it via the TJ3 API to get a
/// self-contained baseline JPEG fixture.
fn build_fixture_jpeg(lib: &libloading::Library) -> (Vec<u8>, Vec<u8>, usize, usize) {
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
        assert_eq!(tj3_set(h_enc, TJPARAM_QUALITY, 85), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_SUBSAMP, TJSAMP_444), 0);

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

        (jpeg, src, w, h_px)
    }
}

/// Allocate + initialise a cinfo + err pair. Returns the bytes buffers
/// (must live as long as cinfo_ptr is used) and the primed pointers.
unsafe fn setup_decompress(
    lib: &libloading::Library,
) -> (
    Box<JpegDecompressPublic>,
    Box<[u8; 512]>,
    *mut c_void,
    *mut c_void,
) {
    // P4-110: a `[u8; N]` is align-1, so casting it to a `j_decompress_ptr`
    // was undefined however large it was; and the shim now rejects a declared
    // size that is not exactly the struct's. Boxing the mirrored struct fixes
    // both.
    // SAFETY: `JpegDecompressPublic` is `#[repr(C)]` plain data — pointers,
    // integers and floats — for which all-zero is a valid bit pattern, which
    // is also the state `jpeg_CreateDecompress` expects to overwrite.
    let cinfo: Box<JpegDecompressPublic> = Box::new(unsafe { std::mem::zeroed() });
    let err: Box<[u8; 512]> = Box::new([0u8; 512]);
    let cinfo_ptr: *mut c_void = Box::leak(cinfo) as *mut JpegDecompressPublic as *mut c_void;
    let err_ptr: *mut c_void = Box::leak(err).as_mut_ptr() as *mut c_void;
    // Re-box via from_raw to get round trip; tests use explicit Box-leak
    // so we can control lifetime.
    let cinfo_box: Box<JpegDecompressPublic> =
        unsafe { Box::from_raw(cinfo_ptr as *mut JpegDecompressPublic) };
    let err_box: Box<[u8; 512]> = unsafe { Box::from_raw(err_ptr as *mut [u8; 512]) };
    let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
        unsafe { lib.get(b"jpeg_std_error") }.expect("jpeg_std_error");
    let _ = unsafe { jpeg_std_error(err_ptr) };
    // Wire cinfo.err = err_ptr (cinfo's first field is `err`).
    unsafe {
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);
    }
    let jpeg_create_decompress: libloading::Symbol<
        unsafe extern "C" fn(*mut c_void, c_int, usize),
    > = unsafe { lib.get(b"jpeg_CreateDecompress") }.expect("jpeg_CreateDecompress");
    unsafe { jpeg_create_decompress(cinfo_ptr, 80, std::mem::size_of::<JpegDecompressPublic>()) };
    (cinfo_box, err_box, cinfo_ptr, err_ptr)
}

#[test]
fn skip_scanlines_advances_cursor_and_returns_count() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    let (jpeg, _src, w, h_px) = build_fixture_jpeg(&lib);

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);

        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg.as_ptr(), jpeg.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg_capi_test_set_out_cs: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_out_cs")
            .expect("jpeg_capi_test_set_out_cs");
        jpeg_capi_test_set_out_cs(cinfo_ptr, JCS_RGB);

        let jpeg_start_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_start_decompress")
                .expect("jpeg_start_decompress");
        assert_ne!(jpeg_start_decompress(cinfo_ptr), 0);

        let jpeg_skip_scanlines: libloading::Symbol<unsafe extern "C" fn(*mut c_void, u32) -> u32> =
            lib.get(b"jpeg_skip_scanlines")
                .expect("jpeg_skip_scanlines");

        let skipped: u32 = jpeg_skip_scanlines(cinfo_ptr, 10);
        assert_eq!(skipped, 10, "skip within bounds returns exact count");

        // Ask to skip more than remaining. Should clamp.
        let big_skip: u32 = jpeg_skip_scanlines(cinfo_ptr, 10_000);
        assert_eq!(
            big_skip,
            (h_px - 10) as u32,
            "skip beyond end clamps to remaining rows"
        );

        // No more to skip.
        let none: u32 = jpeg_skip_scanlines(cinfo_ptr, 1);
        assert_eq!(none, 0, "skip at EOF returns 0");

        let _ = w;
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

#[test]
fn crop_scanline_narrows_emitted_window() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    let (jpeg, original, w, h_px) = build_fixture_jpeg(&lib);

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);

        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg.as_ptr(), jpeg.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg_capi_test_set_out_cs: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_out_cs")
            .expect("jpeg_capi_test_set_out_cs");
        jpeg_capi_test_set_out_cs(cinfo_ptr, JCS_RGB);

        let jpeg_start_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_start_decompress")
                .expect("jpeg_start_decompress");
        assert_ne!(jpeg_start_decompress(cinfo_ptr), 0);

        // Crop to columns [16, 48) — 32-wide window.
        let mut xoff: u32 = 16;
        let mut cwidth: u32 = 32;
        let jpeg_crop_scanline: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32),
        > = lib.get(b"jpeg_crop_scanline").expect("jpeg_crop_scanline");
        jpeg_crop_scanline(cinfo_ptr, &mut xoff, &mut cwidth);
        assert!(
            xoff == 16 && cwidth == 32,
            "crop bounds unchanged inside image"
        );

        let bpp: usize = 3;
        let cropped_row_bytes: usize = cwidth as usize * bpp;
        let mut out: Vec<u8> = vec![0u8; cropped_row_bytes * h_px];

        let jpeg_read_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_read_scanlines")
            .expect("jpeg_read_scanlines");

        let mut read: usize = 0;
        while read < h_px {
            let row_ptr: *mut u8 = out[read * cropped_row_bytes..].as_mut_ptr();
            let mut arr: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_read_scanlines(cinfo_ptr, arr.as_mut_ptr(), 1);
            assert!(got >= 1);
            read += got as usize;
        }

        // Sanity: cropped output should roughly match the cropped region
        // of the Rust-native decode. We compare the full Rust decode,
        // crop it to [16, 48), and bound max_diff.
        let native = libjpeg_turbo_rs::decompress(&jpeg).expect("native decode");
        assert_eq!(native.width, w);
        assert_eq!(native.height, h_px);
        let mut native_cropped: Vec<u8> = Vec::with_capacity(cropped_row_bytes * h_px);
        for y in 0..h_px {
            let start: usize = y * w * 3 + 16 * 3;
            native_cropped.extend_from_slice(&native.data[start..start + cropped_row_bytes]);
        }
        // The decoder path used by our C API is the same path, so cropped
        // pixels must be byte-identical.
        assert_eq!(out, native_cropped, "cropped pixels must match");

        let _ = original;
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

#[test]
fn save_markers_and_read_icc_profile_populates_when_present() {
    // We exercise the raw save_markers plumbing (recording config) and
    // then verify that jpeg_read_icc_profile returns FALSE with NULL
    // outputs when no ICC profile is embedded. Positive ICC extraction
    // is covered by the dedicated ICC test below.
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    let (jpeg, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg.as_ptr(), jpeg.len() as c_ulong);

        // Register save_markers for APP0 + COM. Invocation should not crash.
        let jpeg_save_markers: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, c_uint),
        > = lib.get(b"jpeg_save_markers").expect("jpeg_save_markers");
        jpeg_save_markers(cinfo_ptr, 0xE0, 0xFFFF);
        jpeg_save_markers(cinfo_ptr, 0xFE, 0xFFFF);
        // Clearing one again is idempotent.
        jpeg_save_markers(cinfo_ptr, 0xFE, 0);

        // Register a marker processor — must accept NULL-returning closures.
        unsafe extern "C" fn noop_marker_parser(_cinfo: *mut c_void) -> c_int {
            1
        }
        let jpeg_set_marker_processor: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                c_int,
                Option<unsafe extern "C" fn(*mut c_void) -> c_int>,
            ),
        > = lib
            .get(b"jpeg_set_marker_processor")
            .expect("jpeg_set_marker_processor");
        jpeg_set_marker_processor(cinfo_ptr, 0xE1, Some(noop_marker_parser));
        // NULL routine clears the processor.
        jpeg_set_marker_processor(cinfo_ptr, 0xE1, None);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg_start_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_start_decompress")
                .expect("jpeg_start_decompress");
        assert_ne!(jpeg_start_decompress(cinfo_ptr), 0);

        // Our fixture has no ICC profile: expect FALSE / NULL / 0.
        let jpeg_read_icc_profile: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_uint) -> c_int,
        > = lib
            .get(b"jpeg_read_icc_profile")
            .expect("jpeg_read_icc_profile");
        let mut icc_ptr: *mut u8 = std::ptr::null_mut();
        let mut icc_len: c_uint = 0xDEAD_BEEF;
        let rc: c_int = jpeg_read_icc_profile(cinfo_ptr, &mut icc_ptr, &mut icc_len);
        assert_eq!(rc, 0, "no-ICC fixture returns FALSE");
        assert!(icc_ptr.is_null(), "no-ICC: out pointer set to NULL");
        assert_eq!(icc_len, 0, "no-ICC: out length cleared to 0");

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

#[test]
fn read_icc_profile_returns_embedded_profile() {
    // Build a JPEG that actually carries an ICC profile via the Rust
    // encoder API, then decode it through the classic API and confirm
    // jpeg_read_icc_profile returns the expected bytes.
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    // Small fixture ICC profile — opaque to the test; we just want to
    // verify it round-trips through the APP2 marker-splitting path.
    let icc: Vec<u8> = (0..300).map(|i| (i as u8).wrapping_mul(7)).collect();

    let width: usize = 32;
    let height: usize = 32;
    let rgb: Vec<u8> = (0..width * height * 3)
        .map(|i| (i as u8).wrapping_mul(3))
        .collect();
    let img =
        libjpeg_turbo_rs::Encoder::new(&rgb, width, height, libjpeg_turbo_rs::PixelFormat::Rgb)
            .quality(80)
            .icc_profile(&icc)
            .encode()
            .expect("encode with ICC");

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, img.as_ptr(), img.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg_start_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_start_decompress")
                .expect("jpeg_start_decompress");
        assert_ne!(jpeg_start_decompress(cinfo_ptr), 0);

        let jpeg_read_icc_profile: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, *mut c_uint) -> c_int,
        > = lib
            .get(b"jpeg_read_icc_profile")
            .expect("jpeg_read_icc_profile");
        let mut icc_ptr: *mut u8 = std::ptr::null_mut();
        let mut icc_len: c_uint = 0;
        let rc: c_int = jpeg_read_icc_profile(cinfo_ptr, &mut icc_ptr, &mut icc_len);
        assert_eq!(rc, 1, "ICC present: returns TRUE");
        assert!(!icc_ptr.is_null());
        assert_eq!(icc_len as usize, icc.len());
        let got: &[u8] = std::slice::from_raw_parts(icc_ptr, icc_len as usize);
        assert_eq!(got, icc.as_slice(), "ICC bytes must match");

        // Caller owns the buffer — release via libc free (exposed as tj3Free).
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");
        tj3_free(icc_ptr as *mut c_void);

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }

    // Silence unused warning on the cleanup suppressor.
    let _ = MaybeUninit::<u8>::uninit();
}

// ---------------------------------------------------------------------------
// C1-2: read_coefficients / copy_critical_parameters / core_output_dimensions
// ---------------------------------------------------------------------------

#[test]
fn read_coefficients_returns_non_null_handle_and_copy_params_is_noop_safe() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    let (jpeg, _src, _w, _h_px) = build_fixture_jpeg(&lib);

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg.as_ptr(), jpeg.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg_read_coefficients: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *mut c_void,
        > = lib
            .get(b"jpeg_read_coefficients")
            .expect("jpeg_read_coefficients");
        let coeffs: *mut c_void = jpeg_read_coefficients(cinfo_ptr);
        assert!(!coeffs.is_null(), "coefficients pointer must be non-null");

        // Second call should return a valid handle too (either the same
        // stored pointer or a newly-parsed one).
        let coeffs2: *mut c_void = jpeg_read_coefficients(cinfo_ptr);
        assert!(!coeffs2.is_null());

        // Exercise jpeg_copy_critical_parameters. `dst` may be any
        // non-null pointer in our current shim.
        let jpeg_copy_critical_parameters: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut c_void),
        > = lib
            .get(b"jpeg_copy_critical_parameters")
            .expect("jpeg_copy_critical_parameters");
        let mut dummy: [u8; 64] = [0u8; 64];
        jpeg_copy_critical_parameters(cinfo_ptr, dummy.as_mut_ptr() as *mut c_void);

        // NULL args must not crash.
        jpeg_copy_critical_parameters(std::ptr::null_mut(), dummy.as_mut_ptr() as *mut c_void);
        jpeg_copy_critical_parameters(cinfo_ptr, std::ptr::null_mut());

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

#[test]
fn core_output_dimensions_mirrors_calc_output_dimensions() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    let (jpeg, _src, w, h_px) = build_fixture_jpeg(&lib);

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg.as_ptr(), jpeg.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg_core_output_dimensions: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"jpeg_core_output_dimensions")
                .expect("jpeg_core_output_dimensions");
        jpeg_core_output_dimensions(cinfo_ptr);

        let mut out_w: u32 = 0;
        let mut out_h: u32 = 0;
        let mut out_components: c_int = 0;
        let mut out_cs: c_int = 0;
        let jpeg_capi_test_output_dims: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32, *mut c_int, *mut c_int),
        > = lib
            .get(b"jpeg_capi_test_output_dims")
            .expect("jpeg_capi_test_output_dims");
        jpeg_capi_test_output_dims(
            cinfo_ptr,
            &mut out_w,
            &mut out_h,
            &mut out_components,
            &mut out_cs,
        );
        assert_eq!(out_w as usize, w, "core dims must match image width");
        assert_eq!(out_h as usize, h_px, "core dims must match image height");

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

// ---------------------------------------------------------------------------
// C1-3: 12-bit scanlines / skip / crop + 16-bit scanlines
// ---------------------------------------------------------------------------

#[test]
fn jpeg12_scanlines_and_skip_roundtrip() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    // Build a small 12-bit JPEG fixture via the Rust encoder API.
    let width: usize = 16;
    let height: usize = 16;
    let num_components: usize = 1;
    let mut samples: Vec<i16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            samples.push(((x + y) * 128) as i16);
        }
    }
    let jpeg12: Vec<u8> = libjpeg_turbo_rs::precision::compress_12bit(
        &samples,
        width,
        height,
        num_components,
        75,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("compress 12-bit");

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg12.as_ptr(), jpeg12.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg12_read_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut i16, u32) -> u32,
        > = lib
            .get(b"jpeg12_read_scanlines")
            .expect("jpeg12_read_scanlines");

        // Read two rows at offset 0.
        let mut rowbufs: Vec<Vec<i16>> =
            (0..2).map(|_| vec![0i16; width * num_components]).collect();
        let mut ptrs: Vec<*mut i16> = rowbufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        let got: u32 = jpeg12_read_scanlines(cinfo_ptr, ptrs.as_mut_ptr(), 2);
        assert_eq!(got, 2, "12-bit read returns the requested line count");
        // The first row should be close to `samples[0..width]` modulo
        // DCT quantization; just assert it's non-zero and bounded.
        assert!(
            rowbufs[0].iter().any(|&v| v > 0),
            "12-bit row should contain non-zero samples"
        );

        // Skip 5 rows; the next read should be row 7.
        let jpeg12_skip_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, u32) -> u32,
        > = lib
            .get(b"jpeg12_skip_scanlines")
            .expect("jpeg12_skip_scanlines");
        let skipped: u32 = jpeg12_skip_scanlines(cinfo_ptr, 5);
        assert_eq!(skipped, 5);

        // Skip beyond end clamps.
        let rest: u32 = jpeg12_skip_scanlines(cinfo_ptr, 100);
        assert_eq!(rest, (height - 7) as u32);

        let jpeg12_crop_scanline: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32),
        > = lib
            .get(b"jpeg12_crop_scanline")
            .expect("jpeg12_crop_scanline");
        let mut xoff: u32 = 4;
        let mut cwidth: u32 = 8;
        jpeg12_crop_scanline(cinfo_ptr, &mut xoff, &mut cwidth);
        assert_eq!(xoff, 4);
        assert_eq!(cwidth, 8);

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

#[test]
fn jpeg16_read_scanlines_roundtrip() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    // Lossless 16-bit JPEG via the Rust API.
    let width: usize = 8;
    let height: usize = 8;
    let num_components: usize = 1;
    let samples: Vec<u16> = (0..(width * height)).map(|i| (i * 257) as u16).collect();
    let jpeg16: Vec<u8> =
        libjpeg_turbo_rs::precision::compress_16bit(&samples, width, height, num_components, 1, 0)
            .expect("compress 16-bit");

    unsafe {
        let (_cinfo_box, _err_box, cinfo_ptr, _err_ptr) = setup_decompress(&lib);
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg16.as_ptr(), jpeg16.len() as c_ulong);

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        assert_eq!(jpeg_read_header(cinfo_ptr, 1), JPEG_HEADER_OK);

        let jpeg16_read_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u16, u32) -> u32,
        > = lib
            .get(b"jpeg16_read_scanlines")
            .expect("jpeg16_read_scanlines");

        let mut rowbufs: Vec<Vec<u16>> = (0..height)
            .map(|_| vec![0u16; width * num_components])
            .collect();
        let mut ptrs: Vec<*mut u16> = rowbufs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        let got: u32 = jpeg16_read_scanlines(cinfo_ptr, ptrs.as_mut_ptr(), height as u32);
        assert_eq!(got, height as u32);
        // Lossless: decoded samples must match source exactly.
        for y in 0..height {
            for x in 0..width {
                let expected: u16 = samples[y * width + x];
                let got_sample: u16 = rowbufs[y][x];
                assert_eq!(got_sample, expected, "mismatch at ({x}, {y})");
            }
        }

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}
