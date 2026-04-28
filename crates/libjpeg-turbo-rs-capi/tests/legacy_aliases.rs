//! A1-10: legacy TurboJPEG 1.x / 2.x alias coverage.
//!
//! These tests validate that the thin `tj*` wrappers forward correctly
//! to the TJ3 implementation. The goal is binary compatibility for
//! existing C clients, not re-testing the underlying engine.

#![allow(clippy::type_complexity)]

use std::ffi::{c_char, c_int, c_void};
// CStr is referenced via fully-qualified `std::ffi::CStr` in the
// no-handle error-recovery test; no top-level import needed.
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

        // Invalid inputs: C reference turbojpeg.c returns
        // `(unsigned long)-1` (usize::MAX) for the sizing wrappers when
        // the underlying tj3* helper returns 0. `tjPlaneWidth` retains
        // the pre-3.0 -1 sentinel for component-out-of-range.
        assert_eq!(tj_bufsize(-1, 480, TJSAMP_420), usize::MAX);
        assert_eq!(tj_bufsize_yuv2(640, 0, 480, TJSAMP_420), usize::MAX);
        assert_eq!(tj_plane_w(3, 640, TJSAMP_420), -1);
    }
}

#[test]
fn tj_load_image_reports_error_for_missing_file() {
    // `tjLoadImage` is handle-less per upstream `turbojpeg.h` —
    // first arg is `filename`, no `tjhandle`. Loading a
    // non-existent path returns NULL **and** the diagnostic must be
    // recoverable via `tjGetErrorStr2(NULL)` (the legacy
    // handle-less ABI uses the global no-handle error slot).
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj_load_image: libloading::Symbol<
            unsafe extern "C" fn(
                *const c_char,
                *mut c_int,
                c_int,
                *mut c_int,
                *mut c_int,
                c_int,
            ) -> *mut u8,
        > = lib.get(b"tjLoadImage").expect("tjLoadImage");
        let tj_get_err2: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *const c_char> =
            lib.get(b"tjGetErrorStr2").expect("tjGetErrorStr2");

        let mut w: c_int = 0;
        let mut hh: c_int = 0;
        let mut pf: c_int = -1; // TJPF_UNKNOWN — accept native format
        let ret = tj_load_image(c"/nonexistent".as_ptr(), &mut w, 1, &mut hh, &mut pf, 0);
        assert!(
            ret.is_null(),
            "tjLoadImage must return NULL for missing file"
        );

        // The handle-less ABI must surface the failure through the
        // global no-handle error slot so callers can diagnose it.
        let msg_ptr: *const c_char = tj_get_err2(std::ptr::null_mut());
        assert!(!msg_ptr.is_null(), "tjGetErrorStr2(NULL) must not be NULL");
        let msg: &str = std::ffi::CStr::from_ptr(msg_ptr).to_str().expect("utf8");
        assert!(
            msg.contains("/nonexistent") || msg.contains("cannot read"),
            "expected file-not-found error from tjGetErrorStr2(NULL), got: {msg}"
        );
    }
}

#[test]
fn tj_load_save_image_round_trip_ppm_through_legacy_alias() {
    // End-to-end: write a tiny PPM via the handle-less `tjSaveImage`,
    // load it back via the handle-less `tjLoadImage`, verify pixels
    // round-trip exactly. Exercises both legacy aliases through
    // their tj3 delegates and through the underlying Rust image IO
    // helpers.
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let tmp_dir: std::path::PathBuf = std::env::temp_dir();
    let ppm_path: std::path::PathBuf =
        tmp_dir.join(format!("tj_load_save_test_{}.ppm", std::process::id()));
    let _ = std::fs::remove_file(&ppm_path);
    let ppm_path_c: std::ffi::CString =
        std::ffi::CString::new(ppm_path.to_str().expect("utf8")).expect("nul");

    // 4x3 RGB gradient — small enough to read back trivially.
    let w: c_int = 4;
    let h: c_int = 3;
    let pf: c_int = 0; // TJPF_RGB
    let pixels: Vec<u8> = (0..w as usize * h as usize)
        .flat_map(|i| [(i * 11) as u8, (i * 23) as u8, (i * 47) as u8])
        .collect();

    unsafe {
        let tj_save_image: libloading::Symbol<
            unsafe extern "C" fn(
                *const c_char,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                c_int,
            ) -> c_int,
        > = lib.get(b"tjSaveImage").expect("tjSaveImage");
        let tj_load_image: libloading::Symbol<
            unsafe extern "C" fn(
                *const c_char,
                *mut c_int,
                c_int,
                *mut c_int,
                *mut c_int,
                c_int,
            ) -> *mut u8,
        > = lib.get(b"tjLoadImage").expect("tjLoadImage");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        let rc = tj_save_image(ppm_path_c.as_ptr(), pixels.as_ptr(), w, 0, h, pf, 0);
        assert_eq!(rc, 0, "tjSaveImage must succeed for a writable path");

        let mut got_w: c_int = 0;
        let mut got_h: c_int = 0;
        let mut got_pf: c_int = -1;
        let buf = tj_load_image(
            ppm_path_c.as_ptr(),
            &mut got_w,
            1,
            &mut got_h,
            &mut got_pf,
            0,
        );
        assert!(
            !buf.is_null(),
            "tjLoadImage must succeed for the file we just wrote"
        );
        assert_eq!(got_w, w);
        assert_eq!(got_h, h);
        assert_eq!(got_pf, 0, "PPM should round-trip as TJPF_RGB");
        let got: &[u8] = std::slice::from_raw_parts(buf, pixels.len());
        assert_eq!(got, pixels.as_slice(), "round-trip pixels must match");
        tj3_free(buf as *mut c_void);
    }
    let _ = std::fs::remove_file(&ppm_path);
}

// Type alias matching upstream `tjEncodeYUV3` ABI:
//   int tjEncodeYUV3(tjhandle, const unsigned char *srcBuf,
//                    int width, int pitch, int height,
//                    int pixelFormat, unsigned char *dstBuf,
//                    int align, int subsamp, int flags);
type TjEncodeYUV3 = unsafe extern "C" fn(
    *mut c_void, // handle
    *const u8,   // srcBuf
    c_int,       // width
    c_int,       // pitch (input RGB row stride; 0 = tight w*bpp)
    c_int,       // height
    c_int,       // pixelFormat
    *mut u8,     // dstBuf
    c_int,       // align (YUV plane row alignment)
    c_int,       // subsamp
    c_int,       // flags
) -> c_int;

// Type alias matching upstream `tjDecodeYUV` ABI:
//   int tjDecodeYUV(tjhandle, const unsigned char *srcBuf, int align,
//                   int subsamp, unsigned char *dstBuf, int width,
//                   int pitch, int height, int pixelFormat, int flags);
type TjDecodeYUV = unsafe extern "C" fn(
    *mut c_void, // handle
    *const u8,   // srcBuf (packed YUV)
    c_int,       // align (YUV plane row alignment)
    c_int,       // subsamp
    *mut u8,     // dstBuf (output)
    c_int,       // width
    c_int,       // pitch (output row stride; 0 = tight w*bpp)
    c_int,       // height
    c_int,       // pixelFormat
    c_int,       // flags
) -> c_int;

#[test]
fn tj_encode_decode_yuv_legacy_aliases_roundtrip_444() {
    // Verify the legacy `tjEncodeYUV3` and `tjDecodeYUV` aliases
    // forward correctly to the TJ3 YUV family with the **upstream**
    // ABI: 4th arg is `pitch` (RGB row stride in bytes; `0` = tight
    // `width * bpp`), 8th arg is YUV `align`. With 4:4:4 (no chroma
    // subsampling) the RGB → YUV → RGB round-trip should recover
    // the source within a few units per channel — only the 8-bit
    // BT.601 conversion rounding contributes diff.
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
        let tj_buf_size_yuv2: libloading::Symbol<
            unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> usize,
        > = lib.get(b"tjBufSizeYUV2").expect("tjBufSizeYUV2");
        let tj_encode_yuv3: libloading::Symbol<TjEncodeYUV3> =
            lib.get(b"tjEncodeYUV3").expect("tjEncodeYUV3");
        let tj_decode_yuv: libloading::Symbol<TjDecodeYUV> =
            lib.get(b"tjDecodeYUV").expect("tjDecodeYUV");

        let w: c_int = 64;
        let h: c_int = 64;
        let yuv_align: c_int = 1;

        // Source: deterministic gradient. Tight pitch (= 0 to the
        // ABI; the wrapper interprets that as `w * bpp`).
        let mut src: Vec<u8> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                src.push((x * 4) as u8);
                src.push((y * 4) as u8);
                src.push(((x + y) * 2) as u8);
            }
        }

        // Encode RGB → packed YUV via the legacy alias.
        let yuv_len: usize = tj_buf_size_yuv2(w, yuv_align, h, TJSAMP_444);
        assert!(yuv_len > 0, "tjBufSizeYUV2 must accept (w,h,444)");
        let mut yuv: Vec<u8> = vec![0u8; yuv_len];

        let h_enc: *mut c_void = tj_init_compress();
        assert!(!h_enc.is_null(), "tjInitCompress");
        let rc = tj_encode_yuv3(
            h_enc,
            src.as_ptr(),
            w,
            0, /* pitch = 0 → tight w * 3 */
            h,
            TJPF_RGB,
            yuv.as_mut_ptr(),
            yuv_align,
            TJSAMP_444,
            0,
        );
        assert_eq!(rc, 0, "tjEncodeYUV3 must succeed");
        tj_destroy(h_enc);

        // Decode packed YUV → RGB via the legacy alias.
        let h_dec: *mut c_void = tj_init_decompress();
        assert!(!h_dec.is_null(), "tjInitDecompress");
        let mut dst: Vec<u8> = vec![0u8; src.len()];
        let rc = tj_decode_yuv(
            h_dec,
            yuv.as_ptr(),
            yuv_align,
            TJSAMP_444,
            dst.as_mut_ptr(),
            w,
            0, /* tight output pitch */
            h,
            TJPF_RGB,
            0,
        );
        assert_eq!(rc, 0, "tjDecodeYUV must succeed");
        tj_destroy(h_dec);

        // Fidelity bound: 4:4:4 colorspace round-trip rounding only.
        // Measured headroom is well under 8; allow 8 for safety on
        // extreme gradients across architectures.
        let mut max_diff: u8 = 0;
        for (&a, &b) in src.iter().zip(dst.iter()) {
            let d: u8 = a.abs_diff(b);
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff <= 8,
            "RGB→YUV(444)→RGB round-trip max per-channel diff {max_diff} exceeded 8"
        );
    }
    // Touch the unused TJSAMP_420 const to keep it in scope; future
    // 4:2:0 round-trip test can reuse it without re-adding.
    let _ = TJSAMP_420;
}

#[test]
fn tj_yuv_legacy_aliases_propagate_bottomup_flag() {
    // Upstream `tjEncodeYUV3` / `tjDecodeYUV` map legacy
    // `TJFLAG_BOTTOMUP` (= 2) onto `TJPARAM_BOTTOMUP` on the
    // caller's handle before delegating. This test verifies the
    // propagation via tj3Get(handle, TJPARAM_BOTTOMUP) after each
    // call returns.
    const TJFLAG_BOTTOMUP: c_int = 2;
    const TJPARAM_BOTTOMUP: c_int = 1;

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
        let tj3_get: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int) -> c_int> =
            lib.get(b"tj3Get").expect("tj3Get");
        let tj_buf_size_yuv2: libloading::Symbol<
            unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> usize,
        > = lib.get(b"tjBufSizeYUV2").expect("tjBufSizeYUV2");
        let tj_encode_yuv3: libloading::Symbol<TjEncodeYUV3> =
            lib.get(b"tjEncodeYUV3").expect("tjEncodeYUV3");
        let tj_decode_yuv: libloading::Symbol<TjDecodeYUV> =
            lib.get(b"tjDecodeYUV").expect("tjDecodeYUV");

        let w: c_int = 16;
        let h: c_int = 16;
        let yuv_align: c_int = 1;
        let yuv_len: usize = tj_buf_size_yuv2(w, yuv_align, h, TJSAMP_444);
        let mut yuv: Vec<u8> = vec![0u8; yuv_len];
        let src: Vec<u8> = vec![0x80u8; (w * h * 3) as usize];

        // Compress side: TJFLAG_BOTTOMUP must set TJPARAM_BOTTOMUP=1.
        let h_enc: *mut c_void = tj_init_compress();
        assert!(!h_enc.is_null());
        assert_eq!(
            tj3_get(h_enc, TJPARAM_BOTTOMUP),
            0,
            "TJPARAM_BOTTOMUP must default to 0"
        );
        let rc = tj_encode_yuv3(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
            yuv.as_mut_ptr(),
            yuv_align,
            TJSAMP_444,
            TJFLAG_BOTTOMUP,
        );
        assert_eq!(rc, 0, "tjEncodeYUV3 with TJFLAG_BOTTOMUP must succeed");
        assert_eq!(
            tj3_get(h_enc, TJPARAM_BOTTOMUP),
            1,
            "tjEncodeYUV3 must propagate TJFLAG_BOTTOMUP → TJPARAM_BOTTOMUP=1"
        );
        tj_destroy(h_enc);

        // Decompress side: same propagation requirement.
        let h_dec: *mut c_void = tj_init_decompress();
        assert!(!h_dec.is_null());
        assert_eq!(
            tj3_get(h_dec, TJPARAM_BOTTOMUP),
            0,
            "TJPARAM_BOTTOMUP must default to 0"
        );
        let mut dst: Vec<u8> = vec![0u8; src.len()];
        let rc = tj_decode_yuv(
            h_dec,
            yuv.as_ptr(),
            yuv_align,
            TJSAMP_444,
            dst.as_mut_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
            TJFLAG_BOTTOMUP,
        );
        assert_eq!(rc, 0, "tjDecodeYUV with TJFLAG_BOTTOMUP must succeed");
        assert_eq!(
            tj3_get(h_dec, TJPARAM_BOTTOMUP),
            1,
            "tjDecodeYUV must propagate TJFLAG_BOTTOMUP → TJPARAM_BOTTOMUP=1"
        );
        tj_destroy(h_dec);
    }
}

#[test]
fn tj_yuv_legacy_aliases_actually_flip_rows_under_bottomup() {
    // Beyond just propagating the flag onto `TJPARAM_BOTTOMUP`, the
    // YUV pipeline must actually honour bottom-up row order:
    //
    // - Encoding a row-asymmetric RGB buffer with `TJFLAG_BOTTOMUP`
    //   must yield the same YUV as encoding the row-reversed
    //   buffer without the flag (because flag flips input rows).
    // - Decoding the same YUV with `TJFLAG_BOTTOMUP` must yield the
    //   row-reversed pixels of decoding without the flag.
    //
    // This regression-tests the actual flip behaviour in
    // `tj3EncodeYUV8` / `tj3DecodeYUV8`, not just the parameter
    // propagation.
    const TJFLAG_BOTTOMUP: c_int = 2;

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
        let tj_buf_size_yuv2: libloading::Symbol<
            unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> usize,
        > = lib.get(b"tjBufSizeYUV2").expect("tjBufSizeYUV2");
        let tj_encode_yuv3: libloading::Symbol<TjEncodeYUV3> =
            lib.get(b"tjEncodeYUV3").expect("tjEncodeYUV3");
        let tj_decode_yuv: libloading::Symbol<TjDecodeYUV> =
            lib.get(b"tjDecodeYUV").expect("tjDecodeYUV");

        let w: c_int = 16;
        let h: c_int = 16;
        let row_bytes: usize = (w * 3) as usize;
        let yuv_align: c_int = 1;
        let yuv_len: usize = tj_buf_size_yuv2(w, yuv_align, h, TJSAMP_444);

        // Row-asymmetric source: row index 0 is dark, row index
        // h-1 is bright. Without bottom-up, encode reads dark
        // first; with bottom-up, encode reads bright first.
        let mut src_top_down: Vec<u8> = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            let v: u8 = (y * 16) as u8;
            for _x in 0..w {
                src_top_down.push(v);
                src_top_down.push(v);
                src_top_down.push(v);
            }
        }

        // Reverse row order to produce a buffer that, when read
        // bottom-up, equals `src_top_down` read top-down.
        let mut src_row_reversed: Vec<u8> = vec![0u8; src_top_down.len()];
        for y in 0..(h as usize) {
            let src_off = (h as usize - 1 - y) * row_bytes;
            let dst_off = y * row_bytes;
            src_row_reversed[dst_off..dst_off + row_bytes]
                .copy_from_slice(&src_top_down[src_off..src_off + row_bytes]);
        }

        // Encode top-down without bottom-up flag → reference YUV.
        let mut yuv_ref: Vec<u8> = vec![0u8; yuv_len];
        let h_enc1 = tj_init_compress();
        let rc = tj_encode_yuv3(
            h_enc1,
            src_top_down.as_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
            yuv_ref.as_mut_ptr(),
            yuv_align,
            TJSAMP_444,
            0,
        );
        assert_eq!(rc, 0);
        tj_destroy(h_enc1);

        // Encode the row-reversed buffer WITH bottom-up flag → must
        // equal the reference YUV byte-for-byte.
        let mut yuv_bup: Vec<u8> = vec![0u8; yuv_len];
        let h_enc2 = tj_init_compress();
        let rc = tj_encode_yuv3(
            h_enc2,
            src_row_reversed.as_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
            yuv_bup.as_mut_ptr(),
            yuv_align,
            TJSAMP_444,
            TJFLAG_BOTTOMUP,
        );
        assert_eq!(rc, 0);
        tj_destroy(h_enc2);

        assert_eq!(
            yuv_bup, yuv_ref,
            "tjEncodeYUV3 must read rows bottom-up under TJFLAG_BOTTOMUP"
        );

        // Decode top-down → reference RGB.
        let mut dst_ref: Vec<u8> = vec![0u8; src_top_down.len()];
        let h_dec1 = tj_init_decompress();
        let rc = tj_decode_yuv(
            h_dec1,
            yuv_ref.as_ptr(),
            yuv_align,
            TJSAMP_444,
            dst_ref.as_mut_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
            0,
        );
        assert_eq!(rc, 0);
        tj_destroy(h_dec1);

        // Decode WITH bottom-up flag → output rows must be the
        // row-reversed `dst_ref` (decode wrote bottom-up).
        let mut dst_bup: Vec<u8> = vec![0u8; src_top_down.len()];
        let h_dec2 = tj_init_decompress();
        let rc = tj_decode_yuv(
            h_dec2,
            yuv_ref.as_ptr(),
            yuv_align,
            TJSAMP_444,
            dst_bup.as_mut_ptr(),
            w,
            0,
            h,
            TJPF_RGB,
            TJFLAG_BOTTOMUP,
        );
        assert_eq!(rc, 0);
        tj_destroy(h_dec2);

        for y in 0..(h as usize) {
            let bup_row = &dst_bup[y * row_bytes..(y + 1) * row_bytes];
            let mirrored = &dst_ref
                [(h as usize - 1 - y) * row_bytes..(h as usize - 1 - y) * row_bytes + row_bytes];
            assert_eq!(
                bup_row, mirrored,
                "tjDecodeYUV must write rows bottom-up under TJFLAG_BOTTOMUP at y={y}"
            );
        }
    }
}

#[test]
fn tj_encode_yuv3_does_not_over_read_padded_input_buffer() {
    // Regression test: when the caller passes `pitch > width * bpp`
    // (i.e. the input has trailing per-row padding), the wrapper
    // must NOT read past the last row's `width * bpp` valid bytes.
    // libjpeg-turbo's contract is `srcBuf` size =
    // `(height - 1) * pitch + width * bpp`, leaving the trailing
    // padding of the last row unreadable. Earlier this crate's
    // `densify_pitched_bytes` constructed a `&[u8]` of length
    // `pitch * height`, which is undefined behavior under Rust's
    // aliasing rules even if the post-padding bytes were never
    // dereferenced.
    //
    // The fix reads each row as a separate `&[u8]` of length
    // `width * bpp`. This test exercises the boundary by
    // allocating exactly `(h-1) * pitch + w * 3` bytes and asking
    // tjEncodeYUV3 to encode it. Under Miri / address sanitizer
    // the old over-read would trip; under release builds it would
    // be silent UB.
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj_init_compress: libloading::Symbol<unsafe extern "C" fn() -> *mut c_void> =
            lib.get(b"tjInitCompress").expect("tjInitCompress");
        let tj_destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> c_int> =
            lib.get(b"tjDestroy").expect("tjDestroy");
        let tj_buf_size_yuv2: libloading::Symbol<
            unsafe extern "C" fn(c_int, c_int, c_int, c_int) -> usize,
        > = lib.get(b"tjBufSizeYUV2").expect("tjBufSizeYUV2");
        let tj_encode_yuv3: libloading::Symbol<TjEncodeYUV3> =
            lib.get(b"tjEncodeYUV3").expect("tjEncodeYUV3");

        let w: c_int = 32;
        let h: c_int = 32;
        let row_bytes: usize = (w * 3) as usize;
        let pitch: c_int = (row_bytes as c_int) + 7; // arbitrary trailing pad
        let total: usize = (h as usize - 1) * pitch as usize + row_bytes;
        // Allocate EXACTLY `total` bytes. Reading byte
        // `pitch * h - 1` (the old over-read) would walk past this
        // allocation.
        let src: Vec<u8> = (0..total).map(|i| (i & 0xff) as u8).collect();

        let yuv_align: c_int = 1;
        let yuv_len: usize = tj_buf_size_yuv2(w, yuv_align, h, TJSAMP_444);
        let mut yuv: Vec<u8> = vec![0u8; yuv_len];

        let h_enc: *mut c_void = tj_init_compress();
        assert!(!h_enc.is_null());
        let rc = tj_encode_yuv3(
            h_enc,
            src.as_ptr(),
            w,
            pitch,
            h,
            TJPF_RGB,
            yuv.as_mut_ptr(),
            yuv_align,
            TJSAMP_444,
            0,
        );
        assert_eq!(
            rc, 0,
            "tjEncodeYUV3 with pitch > w*bpp must succeed without over-reading"
        );
        tj_destroy(h_enc);
    }
}
