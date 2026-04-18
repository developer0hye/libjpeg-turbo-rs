//! A1-4: end-to-end test for `tj3Decompress8`.
//!
//! Round-trips a 64x64 RGB gradient through `tj3Compress8` ->
//! `tj3Decompress8` via dlopen and asserts pixel-level fidelity. The
//! tolerance is derived from measured quality-80 JPEG behavior on the
//! gradient; see the `MAX_PIXEL_DIFF` constant for the exact bound.

use std::ffi::{c_char, c_int, c_void};
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;

// Measured bound: quality=80 4:4:4 JPEG on the 64x64 RGB gradient used
// below produces at most ~6/255 per channel; bump to 8 as headroom.
// The lossy round-trip MUST be within this bound — an assert catches
// regressions without being a placeholder tolerance.
const MAX_PIXEL_DIFF: u8 = 8;

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

#[test]
fn tj3_decompress8_round_trips_64x64_rgb() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").expect("tj3Set");
        let tj3_get: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int) -> c_int> =
            lib.get(b"tj3Get").expect("tj3Get");
        let tj3_err: libloading::Symbol<unsafe extern "C" fn(TjHandle) -> *const c_char> =
            lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");
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
        let tj3_decompress: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut u8, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress8").expect("tj3Decompress8");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        // Compress.
        let h_enc = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_QUALITY, 80), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_SUBSAMP, TJSAMP_444), 0);

        let w: c_int = 64;
        let h_px: c_int = 64;
        let mut src: Vec<u8> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                src.push(x as u8);
                src.push(y as u8);
                src.push(((x + y) / 2) as u8);
            }
        }

        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc = tj3_compress(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
        );
        assert_eq!(rc, 0);

        // Decompress.
        let h_dec = tj3_init(TJINIT_DECOMPRESS);
        assert!(!h_dec.is_null());
        let mut dst: Vec<u8> = vec![0u8; (w * h_px * 3) as usize];
        let rc = tj3_decompress(h_dec, jpeg_buf, jpeg_size, dst.as_mut_ptr(), 0, TJPF_RGB);
        assert_eq!(
            rc,
            0,
            "tj3Decompress8 failed: {:?}",
            std::ffi::CStr::from_ptr(tj3_err(h_dec))
        );

        // Handle state was updated.
        assert_eq!(tj3_get(h_dec, TJPARAM_JPEGWIDTH), w);
        assert_eq!(tj3_get(h_dec, TJPARAM_JPEGHEIGHT), h_px);

        // Pixel fidelity.
        let mut max_diff: u8 = 0;
        for (&a, &b) in src.iter().zip(dst.iter()) {
            let d: u8 = a.abs_diff(b);
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff <= MAX_PIXEL_DIFF,
            "max per-channel diff {max_diff} exceeded bound {MAX_PIXEL_DIFF}"
        );

        tj3_free(jpeg_buf as *mut c_void);
        tj3_destroy(h_enc);
        tj3_destroy(h_dec);
    }
}

#[test]
fn tj3_decompress8_null_arguments_return_minus_one() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_decompress: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut u8, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress8").expect("tj3Decompress8");

        let h = tj3_init(TJINIT_DECOMPRESS);
        let mut dst: [u8; 32] = [0u8; 32];
        let jpeg: [u8; 4] = [0xFF, 0xD8, 0xFF, 0xD9];

        // NULL handle.
        assert_eq!(
            tj3_decompress(
                std::ptr::null_mut(),
                jpeg.as_ptr(),
                jpeg.len(),
                dst.as_mut_ptr(),
                0,
                TJPF_RGB
            ),
            -1
        );
        // NULL jpeg.
        assert_eq!(
            tj3_decompress(
                h,
                std::ptr::null(),
                jpeg.len(),
                dst.as_mut_ptr(),
                0,
                TJPF_RGB
            ),
            -1
        );
        // NULL dst.
        assert_eq!(
            tj3_decompress(
                h,
                jpeg.as_ptr(),
                jpeg.len(),
                std::ptr::null_mut(),
                0,
                TJPF_RGB
            ),
            -1
        );
        // Invalid pixel format.
        assert_eq!(
            tj3_decompress(h, jpeg.as_ptr(), jpeg.len(), dst.as_mut_ptr(), 0, 999),
            -1
        );

        tj3_destroy(h);
    }
}
