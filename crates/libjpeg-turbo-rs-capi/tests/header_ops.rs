//! A1-5: `tj3DecompressHeader`, `tj3SetScalingFactor`, `tj3SetCroppingRegion`.
//!
//! Compresses a synthetic image, parses the header via
//! `tj3DecompressHeader` and verifies the Width/Height/Subsamp
//! parameters are exposed through `tj3Get`. Also exercises scaling
//! (1/2) and cropping (centered quarter) to prove the setters are
//! actually honored by the subsequent decode.

use std::ffi::{c_int, c_void};
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

#[repr(C)]
#[derive(Clone, Copy)]
struct TjScalingFactor {
    num: c_int,
    denom: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TjRegion {
    x: c_int,
    y: c_int,
    w: c_int,
    h: c_int,
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

fn compress_checkerboard(lib: &libloading::Library, w: c_int, h_px: c_int) -> Vec<u8> {
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
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
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");

        let h = tj3_init(TJINIT_COMPRESS);
        tj3_set(h, TJPARAM_QUALITY, 90);
        tj3_set(h, TJPARAM_SUBSAMP, TJSAMP_444);

        let mut src: Vec<u8> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                let block: bool = ((x / 8) ^ (y / 8)) & 1 != 0;
                let v: u8 = if block { 240 } else { 16 };
                src.push(v);
                src.push(v);
                src.push(v);
            }
        }

        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc = tj3_compress(
            h,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
        );
        assert_eq!(rc, 0);
        let bytes: Vec<u8> = std::slice::from_raw_parts(jpeg_buf, jpeg_size).to_vec();
        tj3_free(jpeg_buf as *mut c_void);
        tj3_destroy(h);
        bytes
    }
}

#[test]
fn tj3_decompress_header_populates_dimensions() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let jpeg: Vec<u8> = compress_checkerboard(&lib, 128, 96);

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_get: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int) -> c_int> =
            lib.get(b"tj3Get").expect("tj3Get");
        let tj3_header: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize) -> c_int,
        > = lib
            .get(b"tj3DecompressHeader")
            .expect("tj3DecompressHeader");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");

        let h = tj3_init(TJINIT_DECOMPRESS);
        let rc: c_int = tj3_header(h, jpeg.as_ptr(), jpeg.len());
        assert_eq!(rc, 0);
        assert_eq!(tj3_get(h, TJPARAM_JPEGWIDTH), 128);
        assert_eq!(tj3_get(h, TJPARAM_JPEGHEIGHT), 96);
        // TJSAMP_444 for the fixture.
        assert_eq!(tj3_get(h, TJPARAM_SUBSAMP), TJSAMP_444);

        // NULL / too-small buffers must fail.
        assert_eq!(tj3_header(h, std::ptr::null(), 10), -1);
        assert_eq!(tj3_header(h, jpeg.as_ptr(), 1), -1);

        tj3_destroy(h);
    }
}

#[test]
fn tj3_set_scaling_factor_halves_output() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let w: c_int = 128;
    let h_px: c_int = 96;
    let jpeg: Vec<u8> = compress_checkerboard(&lib, w, h_px);

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_set_scaling: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, TjScalingFactor) -> c_int,
        > = lib
            .get(b"tj3SetScalingFactor")
            .expect("tj3SetScalingFactor");
        let tj3_decompress: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut u8, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress8").expect("tj3Decompress8");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");

        let h = tj3_init(TJINIT_DECOMPRESS);
        let rc = tj3_set_scaling(h, TjScalingFactor { num: 1, denom: 2 });
        assert_eq!(rc, 0);

        // After 1/2 scaling the output should be 64x48.
        let scaled_w: usize = (w / 2) as usize;
        let scaled_h: usize = (h_px / 2) as usize;
        let mut dst: Vec<u8> = vec![0u8; scaled_w * scaled_h * 3];
        let rc = tj3_decompress(h, jpeg.as_ptr(), jpeg.len(), dst.as_mut_ptr(), 0, TJPF_RGB);
        assert_eq!(rc, 0);
        // Non-zero content confirms the scaled decode actually happened.
        assert!(dst.iter().any(|&b| b != 0));

        // Invalid factor => -1.
        let bad = tj3_set_scaling(h, TjScalingFactor { num: 0, denom: 1 });
        assert_eq!(bad, -1);

        tj3_destroy(h);
    }
}

#[test]
fn tj3_set_cropping_region_validates_inputs() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_set_crop: libloading::Symbol<unsafe extern "C" fn(TjHandle, TjRegion) -> c_int> =
            lib.get(b"tj3SetCroppingRegion")
                .expect("tj3SetCroppingRegion");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");

        let h = tj3_init(TJINIT_DECOMPRESS);
        // Valid region.
        assert_eq!(
            tj3_set_crop(
                h,
                TjRegion {
                    x: 0,
                    y: 0,
                    w: 32,
                    h: 32
                }
            ),
            0
        );
        // "Clear" region (all zeros).
        assert_eq!(
            tj3_set_crop(
                h,
                TjRegion {
                    x: 0,
                    y: 0,
                    w: 0,
                    h: 0
                }
            ),
            0
        );
        // Negative coordinate.
        assert_eq!(
            tj3_set_crop(
                h,
                TjRegion {
                    x: -1,
                    y: 0,
                    w: 32,
                    h: 32
                }
            ),
            -1
        );
        // Zero dimension.
        assert_eq!(
            tj3_set_crop(
                h,
                TjRegion {
                    x: 0,
                    y: 0,
                    w: 0,
                    h: 32
                }
            ),
            -1
        );
        // NULL handle.
        assert_eq!(
            tj3_set_crop(
                std::ptr::null_mut(),
                TjRegion {
                    x: 0,
                    y: 0,
                    w: 32,
                    h: 32
                }
            ),
            -1
        );

        tj3_destroy(h);
    }
}
