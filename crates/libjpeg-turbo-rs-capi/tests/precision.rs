//! A1-9: `tj3Compress12` / `tj3Decompress12` / `tj3Compress16` /
//! `tj3Decompress16`.
//!
//! 12-bit is lossy (SOF1) so we assert against a measured per-sample
//! tolerance (worst-case ~64/4095 at the handle's default Q=75).
//! 16-bit is lossless, so diff must be exactly zero.

use std::ffi::{c_int, c_short, c_void};
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_PRECISION: c_int = 7;
const TJPARAM_LOSSLESS: c_int = 15;
const TJPARAM_LOSSLESSPSV: c_int = 16;
const TJPARAM_LOSSLESSPT: c_int = 17;
const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;
const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;

/// Locate the `0xFF 0xC3` (SOF3 — lossless Huffman) marker in `jpeg` and
/// return the precision byte stored at offset 4 of the marker payload.
///
/// SOF marker layout (after the `0xFF Cn` start bytes):
/// `[Lf_hi, Lf_lo, P, Y_hi, Y_lo, X_hi, X_lo, Nf, ...]`
/// — so `P` (the sample precision) sits 2 bytes after `0xFFC3`.
fn sof_precision_byte(jpeg: &[u8], marker_lo: u8) -> Option<u8> {
    let mut i: usize = 0;
    while i + 4 < jpeg.len() {
        if jpeg[i] == 0xFF && jpeg[i + 1] == marker_lo {
            // jpeg[i+2..i+4] is the segment length (Lf), then jpeg[i+4] is P.
            return Some(jpeg[i + 4]);
        }
        i += 1;
    }
    None
}

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
fn tj3_compress12_decompress12_round_trips_gray_ramp() {
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
        let tj3_compress12: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const c_short,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress12").unwrap();
        let tj3_decompress12: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut c_short, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress12").unwrap();

        let h_enc = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_QUALITY, 95), 0);

        // 64x64 12-bit gray ramp: values 0..4095 wrapping.
        let w: c_int = 64;
        let h_px: c_int = 64;
        let mut src: Vec<i16> = Vec::with_capacity((w * h_px) as usize);
        for y in 0..h_px {
            for x in 0..w {
                let v: i32 = y * w + x;
                src.push((v & 0x0FFF) as i16);
            }
        }

        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc = tj3_compress12(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_GRAY,
            &mut buf,
            &mut size,
        );
        assert_eq!(rc, 0);
        assert!(!buf.is_null() && size > 4);

        let h_dec = tj3_init(TJINIT_DECOMPRESS);
        let mut dst: Vec<i16> = vec![0i16; (w * h_px) as usize];
        let rc = tj3_decompress12(h_dec, buf, size, dst.as_mut_ptr(), 0, TJPF_GRAY);
        assert_eq!(rc, 0);

        // Q=95 on a 12-bit gray ramp: measured max |diff| ~ 64; use 96
        // as a safe upper bound.
        let mut max_diff: i32 = 0;
        for (&a, &b) in src.iter().zip(dst.iter()) {
            let d: i32 = (a as i32 - b as i32).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff <= 96,
            "12-bit round-trip max diff {max_diff} exceeded 96"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_enc);
        tj3_destroy(h_dec);
    }
}

#[test]
fn tj3_compress16_decompress16_is_lossless() {
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
        let tj3_compress16: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u16,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress16").unwrap();
        let tj3_decompress16: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut u16, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress16").unwrap();

        let h_enc = tj3_init(TJINIT_COMPRESS);
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESSPSV, 1), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESSPT, 0), 0);

        let w: c_int = 32;
        let h_px: c_int = 32;
        let mut src: Vec<u16> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                // Fit within 16 bits. Mix bits so adjacent pixels differ.
                src.push(((y as u32 * 1234 + x as u32 * 7919) & 0xFFFF) as u16);
                src.push(((x as u32 * 2718 + y as u32 * 4099) & 0xFFFF) as u16);
                src.push(((x as u32 * y as u32 * 31) & 0xFFFF) as u16);
            }
        }

        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc = tj3_compress16(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut buf,
            &mut size,
        );
        assert_eq!(rc, 0);

        let h_dec = tj3_init(TJINIT_DECOMPRESS);
        let mut dst: Vec<u16> = vec![0u16; src.len()];
        let rc = tj3_decompress16(h_dec, buf, size, dst.as_mut_ptr(), 0, TJPF_RGB);
        assert_eq!(rc, 0);

        assert_eq!(
            src, dst,
            "16-bit SOF3 round-trip MUST be bit-exact (lossless)"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_enc);
        tj3_destroy(h_dec);
    }
}

#[test]
fn precision_fns_reject_null_and_unsupported_tjpf() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").unwrap();
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").unwrap();
        let tj3_compress12: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const c_short,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress12").unwrap();

        let h = tj3_init(TJINIT_COMPRESS);
        let src: [i16; 4] = [0, 0, 0, 0];
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;

        // NULL handle.
        assert_eq!(
            tj3_compress12(
                std::ptr::null_mut(),
                src.as_ptr(),
                2,
                0,
                2,
                TJPF_GRAY,
                &mut buf,
                &mut size,
            ),
            -1
        );
        // NULL src.
        assert_eq!(
            tj3_compress12(h, std::ptr::null(), 2, 0, 2, TJPF_GRAY, &mut buf, &mut size,),
            -1
        );
        // Alpha-containing TJPF (unsupported in 12-bit).
        let tjpf_rgba: c_int = 7;
        assert_eq!(
            tj3_compress12(h, src.as_ptr(), 1, 0, 1, tjpf_rgba, &mut buf, &mut size,),
            -1
        );

        tj3_destroy(h);
    }
}

/// `tj3Compress8` with `TJPARAM_LOSSLESS=1` and `TJPARAM_PRECISION=4` must
/// emit a SOF3 marker whose sample-precision byte equals 4. This validates
/// that the C-API wires the handle's `TJPARAM_PRECISION` through to the
/// precision-aware lossless encoder path.
#[test]
fn tj3_compress8_lossless_precision4_writes_sof_byte_4() {
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
        let tj3_compress8: libloading::Symbol<
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
        > = lib.get(b"tj3Compress8").unwrap();

        let h_enc = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESS, 1), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_PRECISION, 4), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESSPSV, 1), 0);

        // 16x16 RGB image; samples must fit in 4 bits (0..=15).
        let w: c_int = 16;
        let h_px: c_int = 16;
        let mut src: Vec<u8> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                src.push(((x as u32 + y as u32) & 0x0F) as u8);
                src.push(((x as u32 * 3 + y as u32) & 0x0F) as u8);
                src.push(((x as u32 + y as u32 * 5) & 0x0F) as u8);
            }
        }

        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc = tj3_compress8(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut buf,
            &mut size,
        );
        assert_eq!(rc, 0, "tj3Compress8 with precision=4 must succeed");
        assert!(!buf.is_null() && size > 4);

        let jpeg: &[u8] = std::slice::from_raw_parts(buf, size);
        let p: u8 = sof_precision_byte(jpeg, 0xC3)
            .expect("SOF3 (FFC3) marker must be present in lossless output");
        assert_eq!(
            p, 4,
            "SOF3 sample-precision byte must reflect TJPARAM_PRECISION=4"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_enc);
    }
}

/// `tj3Compress12` with `TJPARAM_LOSSLESS=1` and `TJPARAM_PRECISION=10`
/// must emit a SOF3 marker whose sample-precision byte equals 10.
#[test]
fn tj3_compress12_lossless_precision10_writes_sof_byte_10() {
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
        let tj3_compress12: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const c_short,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress12").unwrap();

        let h_enc = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESS, 1), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_PRECISION, 10), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESSPSV, 1), 0);

        // 32x32 RGB image; samples must fit in 10 bits (0..=1023).
        let w: c_int = 32;
        let h_px: c_int = 32;
        let mut src: Vec<i16> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                src.push(((x as i32 * 17 + y as i32 * 31) & 0x03FF) as i16);
                src.push(((x as i32 * 23 + y as i32 * 11) & 0x03FF) as i16);
                src.push(((x as i32 * 41 + y as i32 * 7) & 0x03FF) as i16);
            }
        }

        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc = tj3_compress12(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut buf,
            &mut size,
        );
        assert_eq!(rc, 0, "tj3Compress12 with precision=10 must succeed");
        assert!(!buf.is_null() && size > 4);

        let jpeg: &[u8] = std::slice::from_raw_parts(buf, size);
        let p: u8 = sof_precision_byte(jpeg, 0xC3)
            .expect("SOF3 (FFC3) marker must be present in lossless output");
        assert_eq!(
            p, 10,
            "SOF3 sample-precision byte must reflect TJPARAM_PRECISION=10"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_enc);
    }
}

/// `tj3Compress16` with `TJPARAM_LOSSLESS=1` and `TJPARAM_PRECISION=14`
/// must emit a SOF3 marker whose sample-precision byte equals 14.
#[test]
fn tj3_compress16_lossless_precision14_writes_sof_byte_14() {
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
        let tj3_compress16: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u16,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress16").unwrap();

        let h_enc = tj3_init(TJINIT_COMPRESS);
        assert!(!h_enc.is_null());
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESS, 1), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_PRECISION, 14), 0);
        assert_eq!(tj3_set(h_enc, TJPARAM_LOSSLESSPSV, 1), 0);

        // 32x32 RGB image; samples must fit in 14 bits (0..=16383).
        let w: c_int = 32;
        let h_px: c_int = 32;
        let mut src: Vec<u16> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                src.push(((x as u32 * 113 + y as u32 * 257) & 0x3FFF) as u16);
                src.push(((x as u32 * 199 + y as u32 * 89) & 0x3FFF) as u16);
                src.push(((x as u32 * 311 + y as u32 * 53) & 0x3FFF) as u16);
            }
        }

        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc = tj3_compress16(
            h_enc,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut buf,
            &mut size,
        );
        assert_eq!(rc, 0, "tj3Compress16 with precision=14 must succeed");
        assert!(!buf.is_null() && size > 4);

        let jpeg: &[u8] = std::slice::from_raw_parts(buf, size);
        let p: u8 = sof_precision_byte(jpeg, 0xC3)
            .expect("SOF3 (FFC3) marker must be present in lossless output");
        assert_eq!(
            p, 14,
            "SOF3 sample-precision byte must reflect TJPARAM_PRECISION=14"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h_enc);
    }
}

/// ITU-T T.81 Annex H requires the lossless point transform Pt to be
/// strictly less than the sample precision P (Pt shifts the lower Pt
/// bits off each sample, so Pt == P would zero every sample). Mirror
/// upstream `references/libjpeg-turbo/src/jclossls.c::start_pass_lossls`.
///
/// This guards against a regression where the precision-override path
/// silently accepted an inconsistent Pt and emitted a SOF3 stream that
/// upstream would reject. The 8-bit / 12-bit / 16-bit entry points
/// each get an arm so any future entry-point that forgets the check
/// fails this test.
#[test]
fn tj3_compress_rejects_lossless_pt_ge_precision() {
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
        let tj3_compress8: libloading::Symbol<
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
        > = lib.get(b"tj3Compress8").unwrap();
        let tj3_compress12: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const i16,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress12").unwrap();
        let tj3_compress16: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u16,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress16").unwrap();

        // 8-bit arm: precision=4, Pt=4 (==P) must reject.
        let h: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null());
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESS, 1), 0);
        assert_eq!(tj3_set(h, TJPARAM_PRECISION, 4), 0);
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESSPSV, 1), 0);
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESSPT, 4), 0);
        let src8: Vec<u8> = vec![0u8; 16 * 16 * 3];
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc8: c_int = tj3_compress8(h, src8.as_ptr(), 16, 0, 16, TJPF_RGB, &mut buf, &mut size);
        assert_eq!(
            rc8, -1,
            "tj3Compress8 must reject TJPARAM_LOSSLESSPT >= TJPARAM_PRECISION"
        );
        assert!(
            buf.is_null(),
            "no output buffer should be allocated on Pt-vs-P rejection"
        );
        tj3_destroy(h);

        // 12-bit arm: precision=10, Pt=10 (==P) must reject.
        let h: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null());
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESS, 1), 0);
        assert_eq!(tj3_set(h, TJPARAM_PRECISION, 10), 0);
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESSPSV, 1), 0);
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESSPT, 10), 0);
        let src12: Vec<i16> = vec![0i16; 16 * 16 * 3];
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc12: c_int =
            tj3_compress12(h, src12.as_ptr(), 16, 0, 16, TJPF_RGB, &mut buf, &mut size);
        assert_eq!(
            rc12, -1,
            "tj3Compress12 must reject TJPARAM_LOSSLESSPT >= TJPARAM_PRECISION"
        );
        assert!(
            buf.is_null(),
            "no output buffer should be allocated on Pt-vs-P rejection"
        );
        tj3_destroy(h);

        // 16-bit arm: precision=14, Pt=14 (==P) must reject.
        let h: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null());
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESS, 1), 0);
        assert_eq!(tj3_set(h, TJPARAM_PRECISION, 14), 0);
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESSPSV, 1), 0);
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESSPT, 14), 0);
        let src16: Vec<u16> = vec![0u16; 16 * 16 * 3];
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc16: c_int =
            tj3_compress16(h, src16.as_ptr(), 16, 0, 16, TJPF_RGB, &mut buf, &mut size);
        assert_eq!(
            rc16, -1,
            "tj3Compress16 must reject TJPARAM_LOSSLESSPT >= TJPARAM_PRECISION"
        );
        assert!(
            buf.is_null(),
            "no output buffer should be allocated on Pt-vs-P rejection"
        );
        tj3_destroy(h);
    }
}

/// `tj3Set(handle, TJPARAM_PRECISION, value)` must reject globally
/// invalid values (outside 2..=16) immediately, mirroring upstream
/// `references/libjpeg-turbo/src/turbojpeg.c:769` (`SET_PARAM(precision,
/// 2, 16)`). Otherwise an absurd value like 0 or 100 would survive in
/// the param store and silently encode at the entry-point default,
/// hiding caller bugs.
#[test]
fn tj3_set_rejects_globally_invalid_precision() {
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

        let h: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null());

        // Below the global lower bound (2): 0, 1, -1 must reject.
        assert_eq!(
            tj3_set(h, TJPARAM_PRECISION, 0),
            -1,
            "TJPARAM_PRECISION=0 must reject (outside 2..=16)"
        );
        assert_eq!(
            tj3_set(h, TJPARAM_PRECISION, 1),
            -1,
            "TJPARAM_PRECISION=1 must reject (outside 2..=16)"
        );
        assert_eq!(
            tj3_set(h, TJPARAM_PRECISION, -1),
            -1,
            "TJPARAM_PRECISION=-1 must reject (outside 2..=16)"
        );
        // Above the global upper bound (16): 17, 100 must reject.
        assert_eq!(
            tj3_set(h, TJPARAM_PRECISION, 17),
            -1,
            "TJPARAM_PRECISION=17 must reject (outside 2..=16)"
        );
        assert_eq!(
            tj3_set(h, TJPARAM_PRECISION, 100),
            -1,
            "TJPARAM_PRECISION=100 must reject (outside 2..=16)"
        );
        // Inside the global range: must accept (per-entry-point
        // narrowing happens at encode time, not at set time).
        for legal in [2, 8, 12, 16] {
            assert_eq!(
                tj3_set(h, TJPARAM_PRECISION, legal),
                0,
                "TJPARAM_PRECISION={legal} must succeed (inside 2..=16)"
            );
        }
        tj3_destroy(h);
    }
}

/// Out-of-entry-point-range `TJPARAM_PRECISION` (still inside 2..=16)
/// must silently fall back to the entry-point's natural precision
/// (BITS_IN_JSAMPLE), matching upstream
/// `references/libjpeg-turbo/src/turbojpeg-mp.c::tj3Compress*` lines
/// 109-117. The previous implementation raised `TJERR_FATAL` on
/// out-of-range precision; this regression guards against that
/// divergence by setting precision values that fall outside each
/// entry-point's lossless range and asserting the encode succeeds with
/// the SOF byte at the entry-point default (8 / 12 / 16).
#[test]
fn tj3_compress_silently_falls_back_on_out_of_range_precision() {
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
        let tj3_compress8: libloading::Symbol<
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
        > = lib.get(b"tj3Compress8").unwrap();

        // Set TJPARAM_PRECISION=15 (way outside tj3Compress8's lossless
        // range of 2..=8). Upstream falls back silently to BITS_IN_JSAMPLE
        // = 8 and emits a regular 8-bit SOF stream.
        let h: TjHandle = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null());
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESS, 1), 0);
        assert_eq!(tj3_set(h, TJPARAM_PRECISION, 15), 0);
        assert_eq!(tj3_set(h, TJPARAM_LOSSLESSPSV, 1), 0);

        let w: c_int = 16;
        let h_px: c_int = 16;
        let src: Vec<u8> = vec![0u8; (w * h_px * 3) as usize];
        let mut buf: *mut u8 = std::ptr::null_mut();
        let mut size: usize = 0;
        let rc: c_int = tj3_compress8(h, src.as_ptr(), w, 0, h_px, TJPF_RGB, &mut buf, &mut size);
        assert_eq!(
            rc, 0,
            "tj3Compress8 must silently fall back when TJPARAM_PRECISION is out of range \
             (matches upstream turbojpeg-mp.c lines 109-117); got rc={rc}"
        );
        assert!(!buf.is_null() && size > 4);

        let jpeg: &[u8] = std::slice::from_raw_parts(buf, size);
        // Upstream falls back to BITS_IN_JSAMPLE = 8 → SOF3 byte should be 8.
        let p: u8 =
            sof_precision_byte(jpeg, 0xC3).expect("SOF3 marker must be present in lossless output");
        assert_eq!(
            p, 8,
            "out-of-range TJPARAM_PRECISION=15 on tj3Compress8 must fall back to 8, got {p}"
        );

        tj3_free(buf as *mut c_void);
        tj3_destroy(h);
    }
}
