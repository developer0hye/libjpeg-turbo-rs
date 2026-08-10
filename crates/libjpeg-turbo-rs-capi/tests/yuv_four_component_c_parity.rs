//! C cross-validation for the P4-125 component-count guard.
//!
//! `yuv_four_component_guard.rs` and `yuv_decompress_planes_component_guard.rs`
//! pin *our* rejection of 4-component (CMYK/YCCK) frames on the TurboJPEG YUV
//! decompress entry points. This file proves the rejection is parity with stock
//! libjpeg-turbo rather than a local invention: it drives
//! `examples/yuv_component_count_c_oracle.c`, linked against a real
//! `libturbojpeg`, over the same fixtures and requires the two implementations
//! to agree on accept-vs-reject for both entry points.
//!
//! Upstream's guard lives in `tj3DecompressToYUVPlanes8`
//! (`references/libjpeg-turbo/src/turbojpeg.c:2229-2230`); its
//! `tj3DecompressToYUV8` inherits it by delegating. This port does not
//! delegate, so both of our entry points carry the check.

mod helpers;

use std::ffi::{c_int, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::{
    tj3DecompressToYUV8, tj3DecompressToYUVPlanes8, tj3Destroy, tj3Init, tj3YUVBufSize,
};

const TJINIT_DECOMPRESS: c_int = 2;
const TJSAMP_444: c_int = 0;
const ALIGN: c_int = 1;
const WIDTH: usize = 16;
const HEIGHT: usize = 16;

/// `(yuv8_rc, planes8_rc)` as reported by stock libjpeg-turbo.
fn c_return_codes(oracle: &Path, jpeg: &[u8], label: &str) -> (i32, i32) {
    let jpeg_path: PathBuf = std::env::temp_dir().join(format!(
        "libjpeg_turbo_rs_yuv_oracle_{}_{label}.jpg",
        std::process::id()
    ));
    std::fs::write(&jpeg_path, jpeg).expect("write oracle fixture");

    let output = Command::new(oracle)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run the C oracle: {e}"));
    let _ = std::fs::remove_file(&jpeg_path);
    assert!(
        output.status.success(),
        "C oracle harness failed for {label}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout: String = String::from_utf8_lossy(&output.stdout).into_owned();
    let parse = |prefix: &str| -> i32 {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(prefix)?.split_whitespace().next())
            .unwrap_or_else(|| panic!("no `{prefix}` line in oracle output:\n{stdout}"))
            .strip_prefix("rc=")
            .unwrap_or_else(|| panic!("malformed rc field in oracle output:\n{stdout}"))
            .parse()
            .unwrap_or_else(|e| panic!("unparseable rc in oracle output ({e}):\n{stdout}"))
    };
    (parse("yuv8 "), parse("planes8 "))
}

/// `(yuv8_rc, planes8_rc)` from this crate, with destinations sized the way the
/// TurboJPEG contract documents: `tj3YUVBufSize` bytes packed, 3 plane pointers
/// planar.
fn rust_return_codes(jpeg: &[u8]) -> (c_int, c_int) {
    let mut packed: Vec<u8> =
        vec![0; tj3YUVBufSize(WIDTH as c_int, ALIGN, HEIGHT as c_int, TJSAMP_444)];
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let rc_yuv8: c_int = unsafe {
        tj3DecompressToYUV8(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            packed.as_mut_ptr(),
            ALIGN,
        )
    };
    // SAFETY: `handle` is a live handle this test created and has not
    // destroyed; nothing else can reach it.
    unsafe { tj3Destroy(handle) };
    let mut planes: Vec<Vec<u8>> = (0..3).map(|_| vec![0u8; WIDTH * HEIGHT]).collect();
    let mut plane_ptrs: Vec<*mut u8> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let rc_planes8: c_int = unsafe {
        tj3DecompressToYUVPlanes8(
            handle,
            jpeg.as_ptr(),
            jpeg.len(),
            plane_ptrs.as_mut_ptr(),
            std::ptr::null(),
        )
    };
    // SAFETY: `handle` is a live handle this test created and has not
    // destroyed; nothing else can reach it.
    unsafe { tj3Destroy(handle) };
    (rc_yuv8, rc_planes8)
}

fn jpeg_fixture(format: PixelFormat, channels: usize) -> Vec<u8> {
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * channels)
        .map(|i| (i % 251) as u8)
        .collect();
    compress(&pixels, WIDTH, HEIGHT, format, 90, Subsampling::S444)
        .unwrap_or_else(|e| panic!("compress {channels}-channel fixture: {e}"))
}

/// P4-125: our accept/reject decision on both YUV decompress entry points must
/// match stock libjpeg-turbo — rejecting the 4-component frame that used to
/// overrun the caller's buffers, and still accepting a 3-component one.
#[test]
fn yuv_decompress_component_count_matches_c() {
    let Some(oracle) = helpers::build_oracle("yuv_component_count_c_oracle") else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found (need a turbojpeg.h declaring tj3* \
             plus a linkable libturbojpeg); set LIBJPEG_TURBO_PREFIX to require it"
        );
        return;
    };

    let cmyk: Vec<u8> = jpeg_fixture(PixelFormat::Cmyk, 4);
    let (c_yuv8, c_planes8) = c_return_codes(&oracle, &cmyk, "cmyk");
    let (rust_yuv8, rust_planes8) = rust_return_codes(&cmyk);
    assert_eq!(
        c_yuv8, -1,
        "upstream tj3DecompressToYUV8 is expected to reject a 4-component frame"
    );
    assert_eq!(
        c_planes8, -1,
        "upstream tj3DecompressToYUVPlanes8 is expected to reject a 4-component frame"
    );
    assert_eq!(
        (rust_yuv8, rust_planes8),
        (c_yuv8, c_planes8),
        "4-component frame: our (yuv8, planes8) return codes must match C"
    );

    let rgb: Vec<u8> = jpeg_fixture(PixelFormat::Rgb, 3);
    let (c_yuv8, c_planes8) = c_return_codes(&oracle, &rgb, "rgb");
    let (rust_yuv8, rust_planes8) = rust_return_codes(&rgb);
    assert_eq!(
        (c_yuv8, c_planes8),
        (0, 0),
        "upstream is expected to accept a 3-component frame"
    );
    assert_eq!(
        (rust_yuv8, rust_planes8),
        (c_yuv8, c_planes8),
        "3-component frame: the guard must not narrow what we accept"
    );
}
