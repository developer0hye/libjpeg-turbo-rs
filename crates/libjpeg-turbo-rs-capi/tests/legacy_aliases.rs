//! A1-10: legacy TurboJPEG 1.x / 2.x alias coverage.
//!
//! These tests validate that the thin `tj*` wrappers forward correctly
//! to the TJ3 implementation. The goal is binary compatibility for
//! existing C clients, not re-testing the underlying engine.

#![allow(clippy::type_complexity)]

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::PathBuf;

const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;
const TJSAMP_420: c_int = 2;

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
fn tj_init_destroy_compress_decompress_legacy_path() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    unsafe {
        let tj_init_compress: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
            lib.get(b"tjInitCompress").expect("tjInitCompress");
        let tj_init_decompress: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
            lib.get(b"tjInitDecompress").expect("tjInitDecompress");
        let tj_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"tjDestroy").expect("tjDestroy");
        let tj_compress2: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tjCompress2").expect("tjCompress2");
        let tj_decompress2: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const u8,
                usize,
                *mut u8,
                c_int,
                c_int,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tjDecompress2").expect("tjDecompress2");
        let tj_header3: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const u8,
                usize,
                *mut c_int,
                *mut c_int,
                *mut c_int,
                *mut c_int,
            ) -> c_int,
        > = lib
            .get(b"tjDecompressHeader3")
            .expect("tjDecompressHeader3");
        let tj_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        // Compress a 64x64 gradient via the legacy API.
        let h_enc = tj_init_compress();
        assert!(!h_enc.is_null());
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
        let rc = tj_compress2(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
            TJSAMP_444,
            80,
            0,
        );
        assert_eq!(rc, 0);
        assert!(!jpeg_buf.is_null());

        // tjDecompressHeader3 pulls dimensions via the legacy out-pointers.
        let h_dec = tj_init_decompress();
        let mut out_w: c_int = 0;
        let mut out_h: c_int = 0;
        let mut out_subsamp: c_int = -1;
        let mut out_colorspace: c_int = -1;
        let rc = tj_header3(
            h_dec,
            jpeg_buf,
            jpeg_size,
            &mut out_w,
            &mut out_h,
            &mut out_subsamp,
            &mut out_colorspace,
        );
        assert_eq!(rc, 0);
        assert_eq!(out_w, w);
        assert_eq!(out_h, h_px);
        assert_eq!(out_subsamp, TJSAMP_444);

        // tjDecompress2 round-trips back to pixels.
        let mut dst: Vec<u8> = vec![0u8; (w * h_px * 3) as usize];
        let rc = tj_decompress2(
            h_dec,
            jpeg_buf,
            jpeg_size,
            dst.as_mut_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            0,
        );
        assert_eq!(rc, 0);

        tj_free(jpeg_buf as *mut c_void);
        assert_eq!(tj_destroy(h_enc), 0);
        assert_eq!(tj_destroy(h_dec), 0);
    }
}

#[test]
fn tj_bufsize_helpers_return_non_zero_for_valid_inputs() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj_bufsize: libloading::Symbol<unsafe extern "C" fn(c_int, c_int, c_int) -> usize> =
            lib.get(b"tjBufSize").expect("tjBufSize");
        let tj_bufsize_yuv2: libloading::Symbol<
            unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> usize,
        > = lib.get(b"tjBufSizeYUV2").expect("tjBufSizeYUV2");
        let tj_plane_w: libloading::Symbol<unsafe extern "C" fn(c_int, c_int, c_int) -> c_int> =
            lib.get(b"tjPlaneWidth").expect("tjPlaneWidth");
        let tj_plane_h: libloading::Symbol<unsafe extern "C" fn(c_int, c_int, c_int) -> c_int> =
            lib.get(b"tjPlaneHeight").expect("tjPlaneHeight");

        assert!(tj_bufsize(640, 480, TJSAMP_420) > 0);
        assert!(tj_bufsize_yuv2(640, 4, 480, TJSAMP_420) > 0);

        // 4:2:0 halves chroma both dimensions.
        let y_w: c_int = tj_plane_w(0, 640, TJSAMP_420);
        let cb_w: c_int = tj_plane_w(1, 640, TJSAMP_420);
        let cr_w: c_int = tj_plane_w(2, 640, TJSAMP_420);
        assert_eq!(y_w, 640);
        assert_eq!(cb_w, 320);
        assert_eq!(cr_w, 320);

        let y_h: c_int = tj_plane_h(0, 480, TJSAMP_420);
        let cb_h: c_int = tj_plane_h(1, 480, TJSAMP_420);
        assert_eq!(y_h, 480);
        assert_eq!(cb_h, 240);

        // Invalid inputs return 0 / -1.
        assert_eq!(tj_bufsize(-1, 480, TJSAMP_420), 0);
        assert_eq!(tj_bufsize_yuv2(640, 0, 480, TJSAMP_420), 0);
        assert_eq!(tj_plane_w(3, 640, TJSAMP_420), -1);
    }
}

#[test]
fn tj_encode_yuv_and_load_image_stubs_fail_loudly() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj_init_compress: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
            lib.get(b"tjInitCompress").expect("tjInitCompress");
        let tj_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"tjDestroy").expect("tjDestroy");
        let tj_encode_yuv3: libloading::Symbol<
            unsafe extern "C" fn(
                *mut c_void,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut u8,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tjEncodeYUV3").expect("tjEncodeYUV3");
        let tj_get_err2: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *const c_char> =
            lib.get(b"tjGetErrorStr2").expect("tjGetErrorStr2");

        let h = tj_init_compress();
        let rc = tj_encode_yuv3(
            h,
            std::ptr::null(),
            0,
            1,
            0,
            TJPF_RGB,
            std::ptr::null_mut(),
            1,
            TJSAMP_420,
            0,
        );
        assert_eq!(rc, -1, "YUV encode stub must signal failure until A1-7");
        let msg: &str = CStr::from_ptr(tj_get_err2(h)).to_str().expect("utf8");
        assert!(
            msg.contains("tjEncodeYUV3") || msg.contains("not yet"),
            "expected descriptive stub error, got: {msg}"
        );
        tj_destroy(h);
    }
}
