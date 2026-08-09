//! Security finding F1: a 4-component (CMYK/YCCK) JPEG must not make the
//! TurboJPEG YUV decompress entry points emit a 4th plane.
//!
//! `tj3YUVBufSize` sizes packed buffers for 1 (gray) or 3 planes, and
//! `tj3DecompressToYUVPlanes8`'s `dstPlanes` is documented as a 3-element
//! array. The plane count, however, comes from the attacker's SOF marker,
//! so a CMYK frame used to make both sinks write a 4th (luma-sized) plane
//! past what the caller allocated. Both entry points must reject such
//! frames with -1, the way upstream libjpeg-turbo does in
//! `tj3DecompressToYUVPlanes8` (turbojpeg.c).

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::{
    tj3DecompressToYUV8, tj3DecompressToYUVPlanes8, tj3Destroy, tj3Init, tj3YUVBufSize,
};

const TJINIT_DECOMPRESS: c_int = 2;
const TJSAMP_444: c_int = 0;
const ALIGN: c_int = 1;
const WIDTH: usize = 16;
const HEIGHT: usize = 16;

/// Byte pattern used to prove nothing was written past the 3-plane region.
const SENTINEL: u8 = 0xA5;

/// A 4:4:4 CMYK JPEG — 4 SOF components, all planes `WIDTH * HEIGHT`.
fn cmyk_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * 4).map(|i| (i % 251) as u8).collect();
    compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        90,
        Subsampling::S444,
    )
    .expect("compress CMYK fixture")
}

/// Finding F1: `tj3DecompressToYUV8` used to pack every SOF component into
/// the caller's buffer, overflowing it by a full luma plane for a
/// 4-component JPEG, because `tj3YUVBufSize` only ever sizes 3 planes.
#[test]
fn tj3_decompress_to_yuv8_rejects_four_component_jpeg() {
    let jpeg: Vec<u8> = cmyk_jpeg();
    let sized_len: usize = tj3YUVBufSize(WIDTH as c_int, ALIGN, HEIGHT as c_int, TJSAMP_444);
    assert_eq!(sized_len, WIDTH * HEIGHT * 3, "3-plane 4:4:4 sizing");

    // Deliberately over-allocate so that the pre-fix overflow lands in
    // test-owned memory: the failure is then an assertion, not corruption.
    let mut dst_buf: Vec<u8> = vec![SENTINEL; sized_len * 2];

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let rc: c_int = tj3DecompressToYUV8(
        handle,
        jpeg.as_ptr(),
        jpeg.len(),
        dst_buf.as_mut_ptr(),
        ALIGN,
    );
    // SAFETY: `handle` is a live handle this test created and has not
    // destroyed; nothing else can reach it.
    unsafe { tj3Destroy(handle) };
    assert_eq!(rc, -1, "4-component JPEG must be rejected, not packed");
    assert!(
        dst_buf[sized_len..].iter().all(|&b| b == SENTINEL),
        "wrote past the tj3YUVBufSize-sized region"
    );
}

/// Finding F1: `tj3DecompressToYUVPlanes8` used to loop over every SOF
/// component, so a 4-component JPEG read a 4th pointer past the caller's
/// documented 3-element `dstPlanes` array and wrote a luma plane through
/// whatever it found there.
#[test]
fn tj3_decompress_to_yuv_planes8_rejects_four_component_jpeg() {
    let jpeg: Vec<u8> = cmyk_jpeg();

    // A real caller only allocates 3 plane buffers; the 4th here exists
    // solely so the pre-fix write is captured instead of corrupting the
    // test process. It must stay untouched.
    let mut planes: Vec<Vec<u8>> = (0..4).map(|_| vec![SENTINEL; WIDTH * HEIGHT]).collect();
    let mut plane_ptrs: Vec<*mut u8> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let rc: c_int = tj3DecompressToYUVPlanes8(
        handle,
        jpeg.as_ptr(),
        jpeg.len(),
        plane_ptrs.as_mut_ptr(),
        std::ptr::null(),
    );
    // SAFETY: `handle` is a live handle this test created and has not
    // destroyed; nothing else can reach it.
    unsafe { tj3Destroy(handle) };
    assert_eq!(rc, -1, "4-component JPEG must be rejected, not written out");
    assert!(
        planes[3].iter().all(|&b| b == SENTINEL),
        "wrote a 4th plane past the caller's 3-element dstPlanes array"
    );
}
