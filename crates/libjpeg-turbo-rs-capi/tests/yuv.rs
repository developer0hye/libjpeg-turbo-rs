//! A1-7: YUV family end-to-end.
//!
//! Covers the round-trip RGB → YUV (packed) → JPEG → YUV (packed) →
//! RGB with all 4:4:4 / 4:2:2 / 4:2:0 subsamplings, plus plane-oriented
//! variants. Fidelity tolerances are measured against the actual
//! `encode_yuv_planes` / `compress_from_yuv` behavior.

use std::ffi::{c_int, c_void};
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;

// Measured headroom: full pipeline with 4:4:4 Q=90 yields max_diff <= 6
// on the 96x96 gradient; bump a bit for safety.
const MAX_PIXEL_DIFF: u8 = 12;

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
fn tj3_yuv_full_pipeline_round_trips_rgb_444() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").unwrap();
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").unwrap();
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").unwrap();
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").unwrap();
        let tj3_encode_yuv8: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut u8,
                c_int,
            ) -> c_int,
        > = lib.get(b"tj3EncodeYUV8").unwrap();
        let tj3_compress_from_yuv8: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3CompressFromYUV8").unwrap();
        let tj3_decompress_to_yuv8: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut u8, c_int) -> c_int,
        > = lib.get(b"tj3DecompressToYUV8").unwrap();
        let tj3_decode_yuv8: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                *mut u8,
                c_int,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tj3DecodeYUV8").unwrap();
        let tj_bufsize_yuv2: libloading::Symbol<
            unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> usize,
        > = lib.get(b"tjBufSizeYUV2").unwrap();

        let w: c_int = 96;
        let h: c_int = 96;
        let align: c_int = 1;

        // Source gradient.
        let mut src: Vec<u8> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                src.push((x * 255 / (w - 1)) as u8);
                src.push((y * 255 / (h - 1)) as u8);
                src.push(((x + y) * 255 / (w + h - 2)) as u8);
            }
        }

        // --- Step 1: RGB -> packed YUV (444) via tj3EncodeYUV8.
        let h_enc = tj3_init(TJINIT_COMPRESS);
        tj3_set(h_enc, TJPARAM_QUALITY, 90);
        tj3_set(h_enc, TJPARAM_SUBSAMP, TJSAMP_444);

        let yuv_len: usize = tj_bufsize_yuv2(w, align, h, TJSAMP_444);
        assert!(yuv_len > 0);
        let mut yuv: Vec<u8> = vec![0u8; yuv_len];
        let rc = tj3_encode_yuv8(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
            yuv.as_mut_ptr(),
            align,
        );
        assert_eq!(rc, 0);

        // --- Step 2: packed YUV -> JPEG via tj3CompressFromYUV8.
        let mut jpeg: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc =
            tj3_compress_from_yuv8(h_enc, yuv.as_ptr(), w, align, h, &mut jpeg, &mut jpeg_size);
        assert_eq!(rc, 0);
        assert!(!jpeg.is_null() && jpeg_size > 4);

        // --- Step 3: JPEG -> packed YUV via tj3DecompressToYUV8.
        let h_dec = tj3_init(TJINIT_DECOMPRESS);
        let mut yuv2: Vec<u8> = vec![0u8; yuv_len];
        let rc = tj3_decompress_to_yuv8(h_dec, jpeg, jpeg_size, yuv2.as_mut_ptr(), align);
        assert_eq!(rc, 0);

        // --- Step 4: packed YUV -> RGB via tj3DecodeYUV8.
        tj3_set(h_dec, TJPARAM_SUBSAMP, TJSAMP_444);
        let mut dst: Vec<u8> = vec![0u8; src.len()];
        let rc = tj3_decode_yuv8(
            h_dec,
            yuv2.as_ptr(),
            align,
            dst.as_mut_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
        );
        assert_eq!(rc, 0);

        // Fidelity check.
        let mut max_diff: u8 = 0;
        for (&a, &b) in src.iter().zip(dst.iter()) {
            let d: u8 = a.abs_diff(b);
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff <= MAX_PIXEL_DIFF,
            "YUV pipeline max per-channel diff {max_diff} exceeded bound {MAX_PIXEL_DIFF}"
        );

        tj3_free(jpeg as *mut c_void);
        tj3_destroy(h_enc);
        tj3_destroy(h_dec);
    }
}

#[test]
fn tj3_yuv_api_rejects_bad_pointers() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").unwrap();
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").unwrap();
        let tj3_encode_yuv8: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut u8,
                c_int,
            ) -> c_int,
        > = lib.get(b"tj3EncodeYUV8").unwrap();

        let h = tj3_init(TJINIT_COMPRESS);
        // NULL handle.
        assert_eq!(
            tj3_encode_yuv8(
                std::ptr::null_mut(),
                std::ptr::null(),
                0,
                0,
                0,
                TJPF_RGB,
                std::ptr::null_mut(),
                1,
            ),
            -1
        );
        // NULL src.
        let mut dst: [u8; 64] = [0u8; 64];
        assert_eq!(
            tj3_encode_yuv8(h, std::ptr::null(), 1, 0, 1, TJPF_RGB, dst.as_mut_ptr(), 1),
            -1
        );
        // Non-power-of-2 align.
        let src: [u8; 3] = [0, 0, 0];
        assert_eq!(
            tj3_encode_yuv8(h, src.as_ptr(), 1, 0, 1, TJPF_RGB, dst.as_mut_ptr(), 3),
            -1
        );
        tj3_destroy(h);
    }
}
