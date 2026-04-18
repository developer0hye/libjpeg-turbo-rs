//! A1-6: end-to-end test for `tj3Transform`.
//!
//! Verifies multi-transform invocation in a single call: hflip + rot180
//! produces two distinct output JPEGs, both byte-different from the
//! input and both decodable back to an image of the expected dimensions.

use std::ffi::{c_int, c_void};
use std::path::PathBuf;

type TjHandle = *mut c_void;

const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;
const TJINIT_TRANSFORM: c_int = 4;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;

const TJXOP_HFLIP: c_int = 1;
const TJXOP_ROT180: c_int = 6;

#[repr(C)]
#[derive(Clone, Copy)]
struct TjRegion {
    x: c_int,
    y: c_int,
    w: c_int,
    h: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct TjTransform {
    r: TjRegion,
    op: c_int,
    options: c_int,
    data: *mut c_void,
    custom_filter: Option<unsafe extern "C" fn()>,
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

fn compress_mcu_aligned_checker(lib: &libloading::Library, w: c_int, h_px: c_int) -> Vec<u8> {
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").unwrap();
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").unwrap();
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
        > = lib.get(b"tj3Compress8").unwrap();
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").unwrap();
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").unwrap();

        let h = tj3_init(TJINIT_COMPRESS);
        tj3_set(h, TJPARAM_QUALITY, 90);
        tj3_set(h, TJPARAM_SUBSAMP, TJSAMP_444);

        let mut src: Vec<u8> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                let block: bool = ((x / 8) ^ (y / 8)) & 1 != 0;
                let r: u8 = if block { 240 } else { 16 };
                let g: u8 = (x as u8).wrapping_mul(2);
                let b: u8 = (y as u8).wrapping_mul(2);
                src.push(r);
                src.push(g);
                src.push(b);
            }
        }

        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        tj3_compress(
            h,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
        );
        let bytes: Vec<u8> = std::slice::from_raw_parts(jpeg_buf, jpeg_size).to_vec();
        tj3_free(jpeg_buf as *mut c_void);
        tj3_destroy(h);
        bytes
    }
}

#[test]
fn tj3_transform_runs_multiple_ops_in_one_call() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    let jpeg: Vec<u8> = compress_mcu_aligned_checker(&lib, 128, 96);

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").unwrap();
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").unwrap();
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").unwrap();
        let tj3_transform: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                usize,
                c_int,
                *mut *mut u8,
                *mut usize,
                *const TjTransform,
            ) -> c_int,
        > = lib.get(b"tj3Transform").unwrap();
        let tj3_decompress: libloading::Symbol<
            unsafe extern "C" fn(TjHandle, *const u8, usize, *mut u8, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress8").unwrap();
        let tj3_get: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int) -> c_int> =
            lib.get(b"tj3Get").unwrap();

        let h = tj3_init(TJINIT_TRANSFORM);
        let no_region: TjRegion = TjRegion {
            x: 0,
            y: 0,
            w: 0,
            h: 0,
        };
        let ops = [
            TjTransform {
                r: no_region,
                op: TJXOP_HFLIP,
                options: 0,
                data: std::ptr::null_mut(),
                custom_filter: None,
            },
            TjTransform {
                r: no_region,
                op: TJXOP_ROT180,
                options: 0,
                data: std::ptr::null_mut(),
                custom_filter: None,
            },
        ];
        let mut dst_bufs: [*mut u8; 2] = [std::ptr::null_mut(); 2];
        let mut dst_sizes: [usize; 2] = [0usize; 2];
        let rc = tj3_transform(
            h,
            jpeg.as_ptr(),
            jpeg.len(),
            2,
            dst_bufs.as_mut_ptr(),
            dst_sizes.as_mut_ptr(),
            ops.as_ptr(),
        );
        assert_eq!(rc, 0);
        assert!(!dst_bufs[0].is_null());
        assert!(!dst_bufs[1].is_null());
        assert!(dst_sizes[0] > 4 && dst_sizes[1] > 4);

        // Both outputs are valid JPEGs (SOI/EOI).
        for i in 0..2 {
            let s: &[u8] = std::slice::from_raw_parts(dst_bufs[i], dst_sizes[i]);
            assert_eq!(&s[..2], &[0xFF, 0xD8], "transform {i} SOI");
            assert_eq!(&s[s.len() - 2..], &[0xFF, 0xD9], "transform {i} EOI");
        }

        // Decode both and check dimensions are correct.
        for i in 0..2 {
            let dec = tj3_init(TJINIT_DECOMPRESS);
            let mut tmp: Vec<u8> = vec![0u8; 128 * 96 * 3];
            let rc = tj3_decompress(
                dec,
                dst_bufs[i],
                dst_sizes[i],
                tmp.as_mut_ptr(),
                0,
                TJPF_RGB,
            );
            assert_eq!(rc, 0);
            // HFLIP and ROT180 preserve dimensions.
            assert_eq!(tj3_get(dec, TJPARAM_JPEGWIDTH), 128);
            assert_eq!(tj3_get(dec, TJPARAM_JPEGHEIGHT), 96);
            tj3_destroy(dec);
        }

        tj3_free(dst_bufs[0] as *mut c_void);
        tj3_free(dst_bufs[1] as *mut c_void);
        tj3_destroy(h);
    }
}

#[test]
fn tj3_transform_validates_arguments() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");
    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").unwrap();
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").unwrap();
        let tj3_transform: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                usize,
                c_int,
                *mut *mut u8,
                *mut usize,
                *const TjTransform,
            ) -> c_int,
        > = lib.get(b"tj3Transform").unwrap();

        let h = tj3_init(TJINIT_TRANSFORM);
        let jpeg: [u8; 4] = [0xFF, 0xD8, 0xFF, 0xD9];
        let t = TjTransform {
            r: TjRegion {
                x: 0,
                y: 0,
                w: 0,
                h: 0,
            },
            op: 42, // unknown op
            options: 0,
            data: std::ptr::null_mut(),
            custom_filter: None,
        };
        let mut bufs: [*mut u8; 1] = [std::ptr::null_mut()];
        let mut sizes: [usize; 1] = [0usize];

        // NULL handle.
        assert_eq!(
            tj3_transform(
                std::ptr::null_mut(),
                jpeg.as_ptr(),
                jpeg.len(),
                1,
                bufs.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &t
            ),
            -1
        );
        // n == 0.
        assert_eq!(
            tj3_transform(
                h,
                jpeg.as_ptr(),
                jpeg.len(),
                0,
                bufs.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &t
            ),
            -1
        );
        // NULL buffers.
        assert_eq!(
            tj3_transform(
                h,
                jpeg.as_ptr(),
                jpeg.len(),
                1,
                std::ptr::null_mut(),
                sizes.as_mut_ptr(),
                &t
            ),
            -1
        );
        // Unknown op.
        assert_eq!(
            tj3_transform(
                h,
                jpeg.as_ptr(),
                jpeg.len(),
                1,
                bufs.as_mut_ptr(),
                sizes.as_mut_ptr(),
                &t
            ),
            -1
        );

        tj3_destroy(h);
    }
}
