//! Security finding F2: `tj3DecompressToYUVPlanes8` must not index past the
//! three-entry `dstPlanes` array the TurboJPEG contract defines.
//!
//! The plane loop is bounded by the component count of the JPEG's own SOF
//! marker, so a 4-component (CMYK/YCCK) stream used to read `dstPlanes[3]`
//! — one pointer past the caller's Y/Cb/Cr array — and then write a full
//! plane through whatever stale word it found. Upstream libjpeg-turbo
//! rejects these frames in the same entry point
//! (`src/turbojpeg.c`: "JPEG image must have 3 or fewer components").

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::{tj3DecompressToYUVPlanes8, tj3Destroy, tj3Init};

const TJINIT_DECOMPRESS: c_int = 2;
const WIDTH: usize = 16;
const HEIGHT: usize = 16;

/// Byte pattern proving no plane was written past the Y/Cb/Cr contract.
const CANARY: u8 = 0xA5;

/// Finding F2: a 4:4:4 CMYK JPEG carries 4 SOF components, so the plane
/// loop ran a fourth time and wrote a `WIDTH * HEIGHT` plane through
/// `dstPlanes[3]`, which a conforming caller never allocated.
///
/// The array here has a fourth entry pointing at a canary the test owns, so
/// the pre-fix out-of-bounds write lands in test memory and shows up as an
/// assertion failure instead of heap corruption.
#[test]
fn tj3_decompress_to_yuv_planes8_rejects_four_component_jpeg() {
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT)
        .flat_map(|i| [(i % 251) as u8, (i % 241) as u8, (i % 239) as u8, 0x10])
        .collect();
    let jpeg: Vec<u8> = compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Cmyk,
        90,
        Subsampling::S444,
    )
    .expect("compress CMYK fixture");

    // 4:4:4 leaves every plane at full resolution, so one `WIDTH * HEIGHT`
    // buffer per entry is exactly what the decoder writes — including the
    // fourth plane the pre-fix code emitted.
    let mut planes: Vec<Vec<u8>> = (0..4).map(|_| vec![CANARY; WIDTH * HEIGHT]).collect();
    let mut plane_ptrs: Vec<*mut u8> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    // NULL strides keeps every write at `plane_width` bytes per row, so a
    // pre-fix fourth-plane write stays inside the canary buffer.
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
    assert!(
        planes[3].iter().all(|&b| b == CANARY),
        "wrote a 4th plane through dstPlanes[3], past the caller's 3-entry array"
    );
    assert_eq!(rc, -1, "4-component JPEG must be rejected, not written out");
}
