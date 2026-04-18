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
const TJPARAM_LOSSLESSPSV: c_int = 16;
const TJPARAM_LOSSLESSPT: c_int = 17;
const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;
const TJPF_RGB: c_int = 0;
const TJPF_GRAY: c_int = 6;

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
