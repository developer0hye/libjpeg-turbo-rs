//! FFI A1-11: end-to-end test for the libjpeg-style `jpeg_*` decode API.
//!
//! Exercises the full create/stdio-or-mem-src/read-header/start-decompress/
//! read-scanlines/finish-decompress/destroy sequence via `dlopen`, and
//! cross-checks the resulting pixels against a Rust-native `decompress()`
//! call. This pins the ABI-compatible path to the same correctness bar
//! as the Rust-native path.
//!
//! The fixture JPEG is produced up-front via the already-validated
//! `tj3Compress8` entry point so this test stays self-contained (no
//! external fixture files needed).

use libjpeg_turbo_rs_capi::jpeglib::{JpegDecompressPublic, JpegErrorMgr};
use std::ffi::{c_char, c_int, c_void, CString};
use std::mem::MaybeUninit;
use std::os::raw::c_ulong;
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJINIT_COMPRESS: c_int = 1;
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;

// libjpeg `jpeg_read_header` return codes.
const JPEG_HEADER_OK: c_int = 1;

// `J_COLOR_SPACE` enumerators used by the test.
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
        assert_eq!(tj3_set(h_enc, TJPARAM_QUALITY, 80), 0);
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

#[test]
fn jpeg_lib_decode_roundtrip_matches_rust_native() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let (jpeg, original_rgb, w, h_px): (Vec<u8>, Vec<u8>, usize, usize) = build_fixture_jpeg(&lib);

    unsafe {
        let mut cinfo: MaybeUninit<JpegDecompressPublic> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;

        let mut err: MaybeUninit<JpegErrorMgr> = MaybeUninit::zeroed();
        let err_ptr: *mut c_void = err.as_mut_ptr() as *mut c_void;

        // `jpeg_std_error(err)` populates callbacks + returns `err`.
        let jpeg_std_error: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *mut c_void> =
            lib.get(b"jpeg_std_error").expect("jpeg_std_error");
        let err_ret: *mut c_void = jpeg_std_error(err_ptr);
        assert_eq!(err_ret, err_ptr, "jpeg_std_error must return its argument");

        // Wire the error manager into the cinfo. Because `err` is the
        // first field of `jpeg_decompress_struct`, writing a pointer at
        // offset 0 is a valid ABI-level setup — the real libjpeg client
        // would write `cinfo.err = jpeg_std_error(&err);` which compiles
        // to exactly that.
        (cinfo_ptr as *mut *mut c_void).write(err_ptr);

        // `jpeg_create_decompress` expands to
        // `jpeg_CreateDecompress(cinfo, JPEG_LIB_VERSION, sizeof(struct))`,
        // so we call the expanded form directly.
        let jpeg_create_decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int, usize),
        > = lib
            .get(b"jpeg_CreateDecompress")
            .expect("jpeg_CreateDecompress");
        // JPEG_LIB_VERSION constant used by libjpeg-turbo 3.x.
        let jpeg_lib_version: c_int = 80;
        jpeg_create_decompress(
            cinfo_ptr,
            jpeg_lib_version,
            std::mem::size_of::<JpegDecompressPublic>(),
        );

        // `jpeg_mem_src(cinfo, buf, size)` points the decoder at an
        // in-memory JPEG datastream.
        let jpeg_mem_src: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, c_ulong),
        > = lib.get(b"jpeg_mem_src").expect("jpeg_mem_src");
        jpeg_mem_src(cinfo_ptr, jpeg.as_ptr(), jpeg.len() as c_ulong);

        // `jpeg_read_header(cinfo, require_image) -> int`.
        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        let header_rc: c_int = jpeg_read_header(cinfo_ptr, 1);
        assert_eq!(header_rc, JPEG_HEADER_OK);

        // After header, `image_width` / `image_height` / `num_components`
        // must reflect the JPEG. These sit at known offsets inside the
        // struct; the library exposes dedicated accessors for tests so we
        // don't have to lock in an offset that could churn.
        let jpeg_capi_test_dimensions: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut u32, *mut u32, *mut c_int, *mut c_int),
        > = lib
            .get(b"jpeg_capi_test_dimensions")
            .expect("jpeg_capi_test_dimensions");
        let mut got_w: u32 = 0;
        let mut got_h: u32 = 0;
        let mut got_nc: c_int = 0;
        let mut got_cs: c_int = 0;
        jpeg_capi_test_dimensions(cinfo_ptr, &mut got_w, &mut got_h, &mut got_nc, &mut got_cs);
        assert_eq!(got_w as usize, w);
        assert_eq!(got_h as usize, h_px);
        assert_eq!(got_nc, 3);

        // Force RGB output.
        let jpeg_capi_test_set_out_cs: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int),
        > = lib
            .get(b"jpeg_capi_test_set_out_cs")
            .expect("jpeg_capi_test_set_out_cs");
        jpeg_capi_test_set_out_cs(cinfo_ptr, JCS_RGB);

        let jpeg_start_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_start_decompress")
                .expect("jpeg_start_decompress");
        let start_rc: c_int = jpeg_start_decompress(cinfo_ptr);
        assert_ne!(start_rc, 0, "jpeg_start_decompress must return non-zero");

        // After start_decompress we know output dims and bpp.
        let mut out_w: u32 = 0;
        let mut out_h: u32 = 0;
        let mut out_components: c_int = 0;
        let mut _out_cs: c_int = 0;
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
            &mut _out_cs,
        );
        assert_eq!(out_w as usize, w);
        assert_eq!(out_h as usize, h_px);
        assert_eq!(out_components, 3);

        // Allocate the output buffer and read scanlines one row at a time.
        let row_bytes: usize = w * 3;
        let mut output: Vec<u8> = vec![0u8; row_bytes * h_px];

        let jpeg_read_scanlines: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *mut *mut u8, u32) -> u32,
        > = lib
            .get(b"jpeg_read_scanlines")
            .expect("jpeg_read_scanlines");

        let mut rows_read: usize = 0;
        while rows_read < h_px {
            let row_ptr: *mut u8 = output[rows_read * row_bytes..].as_mut_ptr();
            let mut row_array: [*mut u8; 1] = [row_ptr];
            let got: u32 = jpeg_read_scanlines(cinfo_ptr, row_array.as_mut_ptr(), 1);
            assert!(
                got >= 1,
                "jpeg_read_scanlines returned 0 with rows remaining"
            );
            rows_read += got as usize;
        }

        let jpeg_finish_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_finish_decompress")
                .expect("jpeg_finish_decompress");
        assert_ne!(jpeg_finish_decompress(cinfo_ptr), 0);

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);

        // Cross-validate: same JPEG decoded via pure Rust API.
        let rust_img = libjpeg_turbo_rs::decompress(&jpeg).expect("native decompress");
        assert_eq!(rust_img.width, w);
        assert_eq!(rust_img.height, h_px);
        assert_eq!(rust_img.data.len(), output.len());
        // Pixel-identical: both paths must delegate to the same decoder.
        assert_eq!(output, rust_img.data, "FFI pixels must match Rust-native");

        // Lossy round-trip sanity vs original gradient.
        let mut max_diff: u8 = 0;
        for (&a, &b) in original_rgb.iter().zip(output.iter()) {
            let d: u8 = a.abs_diff(b);
            if d > max_diff {
                max_diff = d;
            }
        }
        // Measured for this fixture: q=80 4:4:4 stays <= 8/255.
        assert!(max_diff <= 8, "max diff {max_diff} exceeds bound 8");
    }
    // Avoid "unused" on CString if we add error-path tests later.
    let _ = CString::new("").ok();
    let _: c_char = 0;
}

/// Regression: the buffered/progressive idiom
///   `while (!jpeg_input_complete(cinfo)) jpeg_consume_input(cinfo);`
/// must terminate. Earlier we returned `JPEG_REACHED_EOI` from
/// `jpeg_consume_input` once the header was parsed but left
/// `global_state` at `DSTATE_INHEADER`, so `jpeg_input_complete`
/// (which keys off `global_state >= DSTATE_SCANNING`) reported FALSE
/// forever.
#[test]
fn jpeg_consume_input_loop_terminates_after_header_parsed() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    let (jpeg, _src, _w, _h_px): (Vec<u8>, Vec<u8>, usize, usize) = build_fixture_jpeg(&lib);

    unsafe {
        let mut cinfo: MaybeUninit<JpegDecompressPublic> = MaybeUninit::zeroed();
        let cinfo_ptr: *mut c_void = cinfo.as_mut_ptr() as *mut c_void;

        let mut err: MaybeUninit<JpegErrorMgr> = MaybeUninit::zeroed();
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

        let jpeg_input_complete: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_input_complete")
                .expect("jpeg_input_complete");
        let jpeg_consume_input: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"jpeg_consume_input").expect("jpeg_consume_input");

        // Drive the buffered/progressive polling loop. Cap iterations
        // so a regression manifests as a deterministic test failure
        // instead of a hang.
        let mut iterations: u32 = 0;
        while jpeg_input_complete(cinfo_ptr) == 0 {
            let _ = jpeg_consume_input(cinfo_ptr);
            iterations += 1;
            assert!(
                iterations < 16,
                "jpeg_input_complete never reported TRUE after {iterations} \
                 jpeg_consume_input calls — buffered/progressive loop would hang"
            );
        }

        // Once the loop exits, a follow-up consume_input must keep
        // reporting `JPEG_REACHED_EOI` (=2) since our shim buffers the
        // entire datastream.
        const JPEG_REACHED_EOI: c_int = 2;
        assert_eq!(jpeg_consume_input(cinfo_ptr), JPEG_REACHED_EOI);

        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(cinfo_ptr);
    }
}

#[test]
fn jpeg_lib_decode_null_arguments_return_safely() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    unsafe {
        // NULL-handle guards must not crash.
        let jpeg_destroy_decompress: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> = lib
            .get(b"jpeg_destroy_decompress")
            .expect("jpeg_destroy_decompress");
        jpeg_destroy_decompress(std::ptr::null_mut());

        let jpeg_read_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, c_int) -> c_int,
        > = lib.get(b"jpeg_read_header").expect("jpeg_read_header");
        // Expect a negative code (or 0) — crucially, not a crash.
        let rc: c_int = jpeg_read_header(std::ptr::null_mut(), 1);
        assert!(rc <= 0, "NULL cinfo must not succeed");
    }
}
