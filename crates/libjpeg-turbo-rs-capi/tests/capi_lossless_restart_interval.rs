//! P4-121: the lossless restart interval through `tj3Compress8`.
//!
//! For lossless precisions 2..=7, `tj3Compress8` calls the precision-aware
//! lossless encoder directly, and it passed `TJPARAM_RESTARTROWS` on as an
//! MCU count. Upstream converts rows to MCUs (`jcmaster.c` per_scan_setup:
//! `restart_in_rows * MCUs_per_row`, capped at 65535), and a lossless MCU is
//! one pixel, so one restart row on a 33-pixel-wide image is a 33-MCU
//! interval, not a 1-MCU one. The 1-MCU interval made a stream that T.81
//! H.1.1 forbids and `djpeg` refuses ("Invalid restart interval 1; must be an
//! integer multiple of the number of MCUs in an MCU row (33)"); once the
//! encoder refused such intervals, the same call failed outright.
//!
//! An explicit `TJPARAM_RESTARTBLOCKS` that is not a whole number of rows is
//! refused, as `jclossls.c` start_pass refuses it with `JERR_BAD_RESTART`.

use std::ffi::{c_char, c_int, c_void, CStr};

use libjpeg_turbo_rs::precision::decompress_lossless_arbitrary;
use libjpeg_turbo_rs_capi::{tj3Compress8, tj3Destroy, tj3Free, tj3GetErrorStr, tj3Init, tj3Set};

/// `turbojpeg.h`: `TJINIT_COMPRESS`.
const TJINIT_COMPRESS: c_int = 0;
/// `turbojpeg.h`: `TJPARAM_PRECISION`.
const TJPARAM_PRECISION: c_int = 7;
/// `turbojpeg.h`: `TJPARAM_LOSSLESS`.
const TJPARAM_LOSSLESS: c_int = 15;
/// `turbojpeg.h`: `TJPARAM_RESTARTBLOCKS`.
const TJPARAM_RESTARTBLOCKS: c_int = 18;
/// `turbojpeg.h`: `TJPARAM_RESTARTROWS`.
const TJPARAM_RESTARTROWS: c_int = 19;
/// `turbojpeg.h`: `TJPF_GRAY`.
const TJPF_GRAY: c_int = 6;

const WIDTH: usize = 33;
const HEIGHT: usize = 17;
const PRECISION: c_int = 7;

/// 7-bit greyscale samples: a gradient plus a little texture.
fn source() -> Vec<u8> {
    (0..WIDTH * HEIGHT)
        .map(|i| ((i % WIDTH) * 3 + (i / WIDTH) * 5 + (i * 7) % 11) as u8 & 0x7F)
        .collect()
}

/// A lossless 7-bit `tj3Compress8` of [`source`] with one restart parameter
/// set, returning the JPEG or the handle's error string.
fn compress(restart_param: c_int, value: c_int) -> Result<Vec<u8>, String> {
    let src: Vec<u8> = source();
    let handle: *mut c_void = tj3Init(TJINIT_COMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_COMPRESS)");
    let mut buf: *mut u8 = std::ptr::null_mut();
    let mut size: usize = 0;
    // SAFETY: `handle` is a live compressor used only here; `src` holds
    // WIDTH * HEIGHT greyscale samples (pitch 0 = packed), and `buf`/`size`
    // are valid out-pointers that the library fills with its own allocation.
    let rc: c_int = unsafe {
        assert_eq!(tj3Set(handle, TJPARAM_LOSSLESS, 1), 0, "set lossless");
        assert_eq!(
            tj3Set(handle, TJPARAM_PRECISION, PRECISION),
            0,
            "set precision"
        );
        assert_eq!(tj3Set(handle, restart_param, value), 0, "set restart");
        tj3Compress8(
            handle,
            src.as_ptr(),
            WIDTH as c_int,
            0,
            HEIGHT as c_int,
            TJPF_GRAY,
            &mut buf,
            &mut size,
        )
    };
    let result: Result<Vec<u8>, String> = if rc == 0 {
        // SAFETY: on success `buf` holds `size` bytes allocated by the library.
        Ok(unsafe { std::slice::from_raw_parts(buf, size) }.to_vec())
    } else {
        // SAFETY: `tj3GetErrorStr` returns a NUL-terminated string owned by
        // the live handle.
        let raw: *const c_char = unsafe { tj3GetErrorStr(handle) };
        Err(unsafe { CStr::from_ptr(raw) }
            .to_string_lossy()
            .into_owned())
    };
    if !buf.is_null() {
        // SAFETY: allocated by the library, freed once.
        unsafe { tj3Free(buf as *mut c_void) };
    }
    // SAFETY: destroyed once.
    unsafe { tj3Destroy(handle) };
    result
}

/// The DRI interval a JPEG declares.
fn dri(jpeg: &[u8]) -> u16 {
    let at: usize = jpeg
        .windows(4)
        .position(|w| w == [0xFF, 0xDD, 0x00, 0x04])
        .expect("the stream has a DRI segment");
    u16::from_be_bytes([jpeg[at + 4], jpeg[at + 5]])
}

#[test]
fn lossless_restart_rows_count_rows_of_pixels() {
    for rows in [1, 2] {
        let jpeg: Vec<u8> =
            compress(TJPARAM_RESTARTROWS, rows).unwrap_or_else(|e| panic!("{rows} rows: {e}"));
        assert_eq!(
            usize::from(dri(&jpeg)),
            rows as usize * WIDTH,
            "{rows} rows"
        );
        let image = decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| panic!("{e}"));
        assert!(
            image.data.iter().map(|&v| v as u8).eq(source()),
            "{rows} rows: the stream does not decode to the source"
        );
    }
}

#[test]
fn lossless_partial_row_restart_blocks_are_refused() {
    let err: String = compress(TJPARAM_RESTARTBLOCKS, 10)
        .expect_err("a 10-MCU interval on a 33-MCU row must be refused");
    assert!(err.contains("restart interval"), "{err}");
    let jpeg: Vec<u8> = compress(TJPARAM_RESTARTBLOCKS, 66).expect("two whole rows");
    assert_eq!(dri(&jpeg), 66);
}
