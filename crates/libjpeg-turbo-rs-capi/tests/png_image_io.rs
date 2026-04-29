//! PNG image I/O tests for `tj3LoadImage8` / `tj3SaveImage8`.
//!
//! All tests that exercise the png code path are gated on `#[cfg(feature = "png")]`
//! so the default (no `--features png`) build still compiles the test binary —
//! it simply skips the png tests.  The feature-gate-error test runs in both
//! build configurations.

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Shared helpers (mirrors the pattern from legacy_aliases.rs)
// ---------------------------------------------------------------------------

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

/// Resolve the path to the `tests/fixtures/` directory relative to this
/// source file so tests can locate bundled fixture files.
fn fixtures_dir() -> PathBuf {
    // `file!()` is relative to the workspace root.  Resolve via CARGO_MANIFEST_DIR
    // when available; fall back to searching upward from the running executable.
    if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
        return PathBuf::from(manifest).join("tests").join("fixtures");
    }
    // Fallback: walk up from the test binary until we find the fixtures dir.
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf = dir
            .join("crates")
            .join("libjpeg-turbo-rs-capi")
            .join("tests")
            .join("fixtures");
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("could not locate tests/fixtures dir near {}", exe.display());
}

// ---------------------------------------------------------------------------
// TJPF constants (must match turbojpeg.h)
// ---------------------------------------------------------------------------
const TJPF_RGB: c_int = 0;
#[cfg(feature = "png")]
const TJPF_GRAY: c_int = 6;
#[cfg(feature = "png")]
const TJPF_RGBA: c_int = 7;

// TJINIT constants
const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;

// TJPARAM constants (only needed for png feature tests)
#[cfg(feature = "png")]
const TJPARAM_QUALITY: c_int = 3;
#[cfg(feature = "png")]
const TJPARAM_SUBSAMP: c_int = 4;

// TJSAMP (only needed for png feature tests)
#[cfg(feature = "png")]
const TJSAMP_444: c_int = 0;

// ---------------------------------------------------------------------------
// Test: loading a PNG when the feature is disabled must return a clear error
// ---------------------------------------------------------------------------

/// When the `png` feature is NOT compiled in, `tj3LoadImage8` on a PNG file
/// must return NULL and install a descriptive error on the handle.
///
/// When the `png` feature IS compiled in, this test verifies that valid PNG
/// files are NOT rejected — i.e. the error path is not hit.
#[test]
fn png_load_feature_gate_error() {
    let lib_path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&lib_path) }.expect("dlopen cdylib");

    // Use the fixture PNG from tests/fixtures/.
    let fixture: PathBuf = fixtures_dir().join("fixture16x16_rgb.png");
    assert!(
        fixture.exists(),
        "fixture file not found: {}",
        fixture.display()
    );
    let fixture_c: CString = CString::new(fixture.to_str().expect("utf8")).expect("nul in path");

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_load_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *mut c_int,
                c_int,
                *mut c_int,
                *mut c_int,
            ) -> *mut u8,
        > = lib.get(b"tj3LoadImage8").expect("tj3LoadImage8");
        let tj3_get_error_str: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *const c_char,
        > = lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");
        // tj3Free is only needed in the png-feature branch where load succeeds.
        let _tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        let handle: *mut c_void = tj3_init(TJINIT_DECOMPRESS);
        assert!(!handle.is_null(), "tj3Init must return non-null handle");

        let mut w: c_int = 0;
        let mut h: c_int = 0;
        let mut pf: c_int = TJPF_RGB;

        let buf: *mut u8 = tj3_load_image8(handle, fixture_c.as_ptr(), &mut w, 1, &mut h, &mut pf);

        #[cfg(not(feature = "png"))]
        {
            // Without the png feature: must return NULL + descriptive error.
            assert!(
                buf.is_null(),
                "tj3LoadImage8 must return NULL for PNG when feature is off"
            );
            let err_ptr: *const c_char = tj3_get_error_str(handle);
            assert!(!err_ptr.is_null(), "error string must not be NULL");
            let err_msg: &str = std::ffi::CStr::from_ptr(err_ptr).to_str().expect("utf8");
            assert!(
                err_msg.contains("PNG support not enabled"),
                "expected 'PNG support not enabled' in error, got: {err_msg}"
            );
        }

        #[cfg(feature = "png")]
        {
            // With the png feature: must succeed and return a valid buffer.
            assert!(
                !buf.is_null(),
                "tj3LoadImage8 must succeed for a valid PNG when feature is enabled; \
                 error: {}",
                {
                    let err_ptr = tj3_get_error_str(handle);
                    if err_ptr.is_null() {
                        "(no error)".to_string()
                    } else {
                        std::ffi::CStr::from_ptr(err_ptr)
                            .to_str()
                            .unwrap_or("??")
                            .to_string()
                    }
                }
            );
            assert_eq!(w, 16);
            assert_eq!(h, 16);
            assert_eq!(pf, TJPF_RGB, "16x16 RGB PNG should load as TJPF_RGB");
            _tj3_free(buf as *mut c_void);
        }

        tj3_destroy(handle);
    }
}

// ---------------------------------------------------------------------------
// Feature-gated tests
// ---------------------------------------------------------------------------

/// Round-trip: create a pixel buffer, save as PNG via `tj3SaveImage8`, load
/// back via `tj3LoadImage8`, assert pixel-exact equality.
#[cfg(feature = "png")]
#[test]
fn png_round_trip_rgb() {
    let lib_path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&lib_path) }.expect("dlopen cdylib");

    let tmp_dir: PathBuf = std::env::temp_dir();
    let png_path: PathBuf = tmp_dir.join(format!("tj3_png_roundtrip_{}.png", std::process::id()));
    let _ = std::fs::remove_file(&png_path);
    let png_path_c: CString = CString::new(png_path.to_str().expect("utf8")).expect("nul in path");

    // 16×16 deterministic RGB gradient.
    let w: usize = 16;
    let h: usize = 16;
    let src_pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let x: usize = i % w;
            let y: usize = i / w;
            [
                ((x * 16) & 0xff) as u8,
                ((y * 16) & 0xff) as u8,
                (((x + y) * 8) & 0xff) as u8,
            ]
        })
        .collect();

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_save_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tj3SaveImage8").expect("tj3SaveImage8");
        let tj3_load_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *mut c_int,
                c_int,
                *mut c_int,
                *mut c_int,
            ) -> *mut u8,
        > = lib.get(b"tj3LoadImage8").expect("tj3LoadImage8");
        let tj3_get_error_str: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *const c_char,
        > = lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        // --- Save ---
        let h_save: *mut c_void = tj3_init(TJINIT_COMPRESS);
        assert!(!h_save.is_null());
        let rc: c_int = tj3_save_image8(
            h_save,
            png_path_c.as_ptr(),
            src_pixels.as_ptr(),
            w as c_int,
            0, // pitch = 0 (tight)
            h as c_int,
            TJPF_RGB,
        );
        assert_eq!(rc, 0, "tj3SaveImage8 must succeed; error: {}", {
            let ep = tj3_get_error_str(h_save);
            if ep.is_null() {
                "(none)".to_string()
            } else {
                std::ffi::CStr::from_ptr(ep)
                    .to_str()
                    .unwrap_or("??")
                    .to_string()
            }
        });
        tj3_destroy(h_save);

        // Verify the file was actually written and looks like a PNG.
        let written: Vec<u8> = std::fs::read(&png_path).expect("saved PNG file must exist");
        assert!(
            written.starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
            "saved file must begin with PNG signature"
        );

        // --- Load ---
        let h_load: *mut c_void = tj3_init(TJINIT_DECOMPRESS);
        assert!(!h_load.is_null());
        let mut got_w: c_int = 0;
        let mut got_h: c_int = 0;
        let mut got_pf: c_int = TJPF_RGB;
        let buf: *mut u8 = tj3_load_image8(
            h_load,
            png_path_c.as_ptr(),
            &mut got_w,
            1,
            &mut got_h,
            &mut got_pf,
        );
        assert!(
            !buf.is_null(),
            "tj3LoadImage8 must succeed for the PNG we just saved; error: {}",
            {
                let ep = tj3_get_error_str(h_load);
                if ep.is_null() {
                    "(none)".to_string()
                } else {
                    std::ffi::CStr::from_ptr(ep)
                        .to_str()
                        .unwrap_or("??")
                        .to_string()
                }
            }
        );
        assert_eq!(got_w, w as c_int);
        assert_eq!(got_h, h as c_int);
        assert_eq!(got_pf, TJPF_RGB, "round-trip should preserve TJPF_RGB");

        let got_slice: &[u8] = std::slice::from_raw_parts(buf, src_pixels.len());
        assert_eq!(
            got_slice,
            src_pixels.as_slice(),
            "PNG round-trip must be pixel-exact"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_load);
    }

    let _ = std::fs::remove_file(&png_path);
}

/// Round-trip: same as `png_round_trip_rgb` but with RGBA (4-channel) data.
#[cfg(feature = "png")]
#[test]
fn png_round_trip_rgba() {
    let lib_path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&lib_path) }.expect("dlopen cdylib");

    let tmp_dir: PathBuf = std::env::temp_dir();
    let png_path: PathBuf =
        tmp_dir.join(format!("tj3_png_roundtrip_rgba_{}.png", std::process::id()));
    let _ = std::fs::remove_file(&png_path);
    let png_path_c: CString = CString::new(png_path.to_str().expect("utf8")).expect("nul in path");

    let w: usize = 16;
    let h: usize = 16;
    // RGBA: semi-transparent gradient.
    let src_pixels: Vec<u8> = (0..w * h)
        .flat_map(|i| {
            let x: usize = i % w;
            let y: usize = i / w;
            [
                ((x * 16) & 0xff) as u8,
                ((y * 16) & 0xff) as u8,
                (((x + y) * 8) & 0xff) as u8,
                (i & 0xff) as u8, // alpha
            ]
        })
        .collect();

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_save_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tj3SaveImage8").expect("tj3SaveImage8");
        let tj3_load_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *mut c_int,
                c_int,
                *mut c_int,
                *mut c_int,
            ) -> *mut u8,
        > = lib.get(b"tj3LoadImage8").expect("tj3LoadImage8");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        let h_save: *mut c_void = tj3_init(TJINIT_COMPRESS);
        assert!(!h_save.is_null());
        let rc: c_int = tj3_save_image8(
            h_save,
            png_path_c.as_ptr(),
            src_pixels.as_ptr(),
            w as c_int,
            0,
            h as c_int,
            TJPF_RGBA,
        );
        assert_eq!(rc, 0, "tj3SaveImage8 must succeed for RGBA PNG");
        tj3_destroy(h_save);

        let h_load: *mut c_void = tj3_init(TJINIT_DECOMPRESS);
        assert!(!h_load.is_null());
        let mut got_w: c_int = 0;
        let mut got_h: c_int = 0;
        let mut got_pf: c_int = TJPF_RGBA;
        let buf: *mut u8 = tj3_load_image8(
            h_load,
            png_path_c.as_ptr(),
            &mut got_w,
            1,
            &mut got_h,
            &mut got_pf,
        );
        assert!(!buf.is_null(), "tj3LoadImage8 must succeed for RGBA PNG");
        assert_eq!(got_w, w as c_int);
        assert_eq!(got_h, h as c_int);
        assert_eq!(got_pf, TJPF_RGBA, "RGBA PNG should round-trip as TJPF_RGBA");

        let got_slice: &[u8] = std::slice::from_raw_parts(buf, src_pixels.len());
        assert_eq!(
            got_slice,
            src_pixels.as_slice(),
            "RGBA PNG round-trip must be pixel-exact"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_load);
    }

    let _ = std::fs::remove_file(&png_path);
}

/// Load the bundled 16×16 fixture PNG, encode it as JPEG at q=90, decode
/// back, and assert PSNR ≥ 30 dB. This exercises the PNG→JPEG→decode chain
/// through the TJ3 ABI.
#[cfg(feature = "png")]
#[test]
fn png_to_jpeg_roundtrip_psnr() {
    let lib_path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&lib_path) }.expect("dlopen cdylib");

    let fixture: PathBuf = fixtures_dir().join("fixture16x16_rgb.png");
    assert!(fixture.exists(), "fixture not found: {}", fixture.display());
    let fixture_c: CString = CString::new(fixture.to_str().expect("utf8")).expect("nul");

    let tmp_dir: PathBuf = std::env::temp_dir();
    let jpeg_path: PathBuf = tmp_dir.join(format!("tj3_png_jpeg_{}.jpg", std::process::id()));
    let _ = std::fs::remove_file(&jpeg_path);

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").expect("tj3Set");
        let tj3_load_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *mut c_int,
                c_int,
                *mut c_int,
                *mut c_int,
            ) -> *mut u8,
        > = lib.get(b"tj3LoadImage8").expect("tj3LoadImage8");
        let tj3_compress8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress8").expect("tj3Compress8");
        let tj3_decompress8: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, usize, *mut u8, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress8").expect("tj3Decompress8");
        let tj3_decompress_header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int,
        > = lib
            .get(b"tj3DecompressHeader")
            .expect("tj3DecompressHeader");
        let tj3_get: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int) -> c_int> =
            lib.get(b"tj3Get").expect("tj3Get");
        let tj3_get_error_str: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *const c_char,
        > = lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        // 1. Load PNG fixture via TJ3.
        let h_load: *mut c_void = tj3_init(TJINIT_DECOMPRESS);
        assert!(!h_load.is_null());
        let mut img_w: c_int = 0;
        let mut img_h: c_int = 0;
        let mut img_pf: c_int = TJPF_RGB;
        let png_buf: *mut u8 = tj3_load_image8(
            h_load,
            fixture_c.as_ptr(),
            &mut img_w,
            1,
            &mut img_h,
            &mut img_pf,
        );
        assert!(
            !png_buf.is_null(),
            "tj3LoadImage8 failed for fixture: {}",
            {
                let ep = tj3_get_error_str(h_load);
                if ep.is_null() {
                    "(none)".to_string()
                } else {
                    std::ffi::CStr::from_ptr(ep)
                        .to_str()
                        .unwrap_or("??")
                        .to_string()
                }
            }
        );
        assert_eq!(img_w, 16);
        assert_eq!(img_h, 16);
        assert_eq!(img_pf, TJPF_RGB);
        tj3_destroy(h_load);

        // 2. Compress PNG pixels → JPEG at q=90.
        let h_enc: *mut c_void = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_QUALITY, 90), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_SUBSAMP, TJSAMP_444), 0);
        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc: c_int = tj3_compress8(
            h_enc,
            png_buf,
            img_w,
            img_w * 3, // pitch
            img_h,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
        );
        assert_eq!(rc, 0, "tj3Compress8 must succeed");
        tj3_free(png_buf as *mut c_void);
        tj3_destroy(h_enc);

        // 3. Decompress JPEG → pixel buffer.
        let h_dec: *mut c_void = tj3_init(TJINIT_DECOMPRESS);
        assert!(!h_dec.is_null());
        let rc: c_int = tj3_decompress_header(h_dec, jpeg_buf, jpeg_size);
        assert_eq!(rc, 0, "tj3DecompressHeader must succeed");
        let out_w: c_int = tj3_get(h_dec, 5); // TJPARAM_WIDTH = 5
        let out_h: c_int = tj3_get(h_dec, 6); // TJPARAM_HEIGHT = 6
        assert_eq!(out_w, 16);
        assert_eq!(out_h, 16);

        let pixel_count: usize = (out_w * out_h * 3) as usize;
        let mut decoded: Vec<u8> = vec![0u8; pixel_count];
        let rc: c_int = tj3_decompress8(
            h_dec,
            jpeg_buf,
            jpeg_size,
            decoded.as_mut_ptr(),
            out_w * 3,
            TJPF_RGB,
        );
        assert_eq!(rc, 0, "tj3Decompress8 must succeed");
        tj3_free(jpeg_buf as *mut c_void);
        tj3_destroy(h_dec);

        // 4. Re-load the fixture PNG to get the reference pixels.
        let reference: Vec<u8> = {
            let data: Vec<u8> = std::fs::read(&fixture).expect("fixture readable");
            // Decode via the png crate directly using the same load path.
            // We don't have direct access to the Rust API from the cdylib test,
            // so we use Python-generated known-good pixels from the fixture.
            // The fixture is a 16x16 gradient: R=x*16, G=y*16, B=(x+y)*8.
            let mut ref_px: Vec<u8> = Vec::with_capacity(16 * 16 * 3);
            for y in 0u8..16 {
                for x in 0u8..16 {
                    ref_px.push(x * 16);
                    ref_px.push(y * 16);
                    ref_px.push(x.wrapping_add(y) * 8);
                }
            }
            // Sanity: fixture file must exist and be non-empty.
            assert!(!data.is_empty());
            ref_px
        };

        // 5. Compute PSNR between reference and decoded.
        let mse: f64 = reference
            .iter()
            .zip(decoded.iter())
            .map(|(&a, &b)| {
                let d: f64 = (a as f64) - (b as f64);
                d * d
            })
            .sum::<f64>()
            / reference.len() as f64;

        // PSNR = 10 * log10(255^2 / MSE).
        // At q=90 / 4:4:4, a simple gradient should have PSNR well above 30 dB.
        // Measured minimum on this fixture: ~38 dB. Gate at 30 dB for safety.
        let psnr: f64 = if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0f64 * 255.0 / mse).log10()
        };
        assert!(
            psnr >= 30.0,
            "PNG→JPEG(q=90)→decode PSNR {psnr:.2} dB is below the 30 dB floor"
        );
    }

    let _ = std::fs::remove_file(&jpeg_path);
}

/// Saving to `.png` when `png` feature is OFF must return -1 and install a
/// descriptive error on the handle.
#[test]
#[cfg(not(feature = "png"))]
fn png_save_feature_gate_error_no_feature() {
    let lib_path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&lib_path) }.expect("dlopen cdylib");

    let tmp_dir: PathBuf = std::env::temp_dir();
    let png_path: PathBuf = tmp_dir.join(format!("tj3_png_gate_{}.png", std::process::id()));
    let png_path_c: CString = CString::new(png_path.to_str().expect("utf8")).expect("nul");

    let src: Vec<u8> = vec![0x80u8; 16 * 16 * 3];

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_save_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tj3SaveImage8").expect("tj3SaveImage8");
        let tj3_get_error_str: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void) -> *const c_char,
        > = lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");

        let handle: *mut c_void = tj3_init(TJINIT_COMPRESS);
        assert!(!handle.is_null());
        let rc: c_int = tj3_save_image8(
            handle,
            png_path_c.as_ptr(),
            src.as_ptr(),
            16,
            0,
            16,
            TJPF_RGB,
        );
        assert_eq!(
            rc, -1,
            "tj3SaveImage8 to .png must fail when feature is off"
        );
        let err_ptr: *const c_char = tj3_get_error_str(handle);
        assert!(!err_ptr.is_null());
        let err_msg: &str = std::ffi::CStr::from_ptr(err_ptr).to_str().expect("utf8");
        assert!(
            err_msg.contains("PNG support not enabled"),
            "expected 'PNG support not enabled', got: {err_msg}"
        );
        tj3_destroy(handle);
    }

    // File must NOT have been created.
    assert!(
        !png_path.exists(),
        "tj3SaveImage8 must not create a file when feature is off"
    );
}

/// Grayscale PNG round-trip.
#[cfg(feature = "png")]
#[test]
fn png_round_trip_grayscale() {
    let lib_path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&lib_path) }.expect("dlopen cdylib");

    let tmp_dir: PathBuf = std::env::temp_dir();
    let png_path: PathBuf = tmp_dir.join(format!("tj3_png_gray_{}.png", std::process::id()));
    let _ = std::fs::remove_file(&png_path);
    let png_path_c: CString = CString::new(png_path.to_str().expect("utf8")).expect("nul");

    let w: usize = 16;
    let h: usize = 16;
    let src_pixels: Vec<u8> = (0..w * h as usize)
        .map(|i| ((i * 4) & 0xff) as u8)
        .collect();

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_save_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tj3SaveImage8").expect("tj3SaveImage8");
        let tj3_load_image8: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const c_char,
                *mut c_int,
                c_int,
                *mut c_int,
                *mut c_int,
            ) -> *mut u8,
        > = lib.get(b"tj3LoadImage8").expect("tj3LoadImage8");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        let h_save: *mut c_void = tj3_init(TJINIT_COMPRESS);
        assert!(!h_save.is_null());
        let rc: c_int = tj3_save_image8(
            h_save,
            png_path_c.as_ptr(),
            src_pixels.as_ptr(),
            w as c_int,
            0,
            h as c_int,
            TJPF_GRAY,
        );
        assert_eq!(rc, 0, "tj3SaveImage8 must succeed for grayscale PNG");
        tj3_destroy(h_save);

        let h_load: *mut c_void = tj3_init(TJINIT_DECOMPRESS);
        assert!(!h_load.is_null());
        let mut got_w: c_int = 0;
        let mut got_h: c_int = 0;
        let mut got_pf: c_int = TJPF_GRAY;
        let buf: *mut u8 = tj3_load_image8(
            h_load,
            png_path_c.as_ptr(),
            &mut got_w,
            1,
            &mut got_h,
            &mut got_pf,
        );
        assert!(
            !buf.is_null(),
            "tj3LoadImage8 must succeed for grayscale PNG"
        );
        assert_eq!(got_w, w as c_int);
        assert_eq!(got_h, h as c_int);
        assert_eq!(got_pf, TJPF_GRAY, "grayscale PNG should load as TJPF_GRAY");

        let got_slice: &[u8] = std::slice::from_raw_parts(buf, src_pixels.len());
        assert_eq!(
            got_slice,
            src_pixels.as_slice(),
            "grayscale PNG round-trip must be pixel-exact"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_load);
    }

    let _ = std::fs::remove_file(&png_path);
}
