//! P4-234 (#667) and P4-225 (#652): what `tj3DecompressToYUV8` /
//! `tj3DecompressToYUVPlanes8` may write, and the handle limits they apply.
//!
//! Upstream sizes YUV output from the *scaled* frame (`TJSCALED` in the packed
//! wrapper, `jpeg_calc_output_dimensions` in the planar body), so a caller
//! following `turbojpeg.h` allocates `tj3YUVBufSize(TJSCALED(w), align,
//! TJSCALED(h), subsamp)`. The port decoded at full size whatever the scaling
//! factor and wrote the unscaled planes into that buffer.
//!
//! Each overrun check fills the caller's buffer and an 8 KiB guard band past
//! it with a sentinel, so a write past the documented size fails the
//! assertion on any build. The sanitizers workflow also runs this file under
//! AddressSanitizer, which reports a write past the guard band as well.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::yuv::{tj3DecompressToYUV8, tj3DecompressToYUVPlanes8};
use libjpeg_turbo_rs_capi::{
    tj3Destroy, tj3Get, tj3GetScalingFactors, tj3Init, tj3Set, tj3SetScalingFactor, tj3YUVBufSize,
    TjScalingFactor,
};

const TJINIT_DECOMPRESS: c_int = 1;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_MAXMEMORY: c_int = 23;
const TJSAMP_420: c_int = 2;

const PHOTO_420: &[u8] = include_bytes!("../../../tests/fixtures/photo_64x64_420.jpg");

/// Bytes past a correctly sized buffer the guard band can see.
const GUARD: usize = 8192;
const SENTINEL: u8 = 0xA5;

fn instance() -> *mut c_void {
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init(TJINIT_DECOMPRESS)");
    handle
}

fn destroy(handle: *mut c_void) {
    // SAFETY: `handle` came from `tj3Init` and is not used again.
    unsafe { tj3Destroy(handle) };
}

/// Issue #667: at scaling factor 1/2 the caller's buffer holds the 32x32
/// planes, `tj3YUVBufSize(32, 1, 32, TJSAMP_420)` = 1536 bytes. Stock writes
/// exactly those; the port wrote the 64x64 planes, 6144 bytes. Until scaled
/// YUV output exists the decode is refused, and nothing is written at all.
#[test]
fn a_scaled_yuv_decompress_writes_nothing_past_the_scaled_buffer() {
    const SCALED_SIZE: usize = 32 * 32 + 2 * 16 * 16;
    for planar in [false, true] {
        let handle: *mut c_void = instance();
        let mut buffer: Vec<u8> = vec![SENTINEL; SCALED_SIZE + GUARD];
        // SAFETY: live handle; `buffer` holds the scaled planes plus a guard
        // band, so an overrun lands in memory this test owns and inspects.
        let rc: c_int = unsafe {
            assert_eq!(
                tj3SetScalingFactor(handle, TjScalingFactor { num: 1, denom: 2 }),
                0
            );
            if planar {
                let base: *mut u8 = buffer.as_mut_ptr();
                let mut planes: [*mut u8; 3] = [base, base.add(1024), base.add(1024 + 256)];
                tj3DecompressToYUVPlanes8(
                    handle,
                    PHOTO_420.as_ptr(),
                    PHOTO_420.len(),
                    planes.as_mut_ptr(),
                    std::ptr::null(),
                )
            } else {
                tj3DecompressToYUV8(
                    handle,
                    PHOTO_420.as_ptr(),
                    PHOTO_420.len(),
                    buffer.as_mut_ptr(),
                    1,
                )
            }
        };
        assert_eq!(
            rc, -1,
            "planar={planar}: refused until P4-234's scaled output lands"
        );
        assert!(
            buffer[SCALED_SIZE..].iter().all(|&byte| byte == SENTINEL),
            "planar={planar}: wrote past the {SCALED_SIZE}-byte scaled buffer"
        );
        // The header read before the refusal published, as upstream's does.
        // SAFETY: live handle.
        assert_eq!(unsafe { tj3Get(handle, TJPARAM_JPEGWIDTH) }, 64);
        destroy(handle);
    }
}

/// Issue #667: a stride shorter than its plane's width is refused, as
/// upstream refuses it (`turbojpeg.c:2253-2254`), and so is a negative one.
/// Honouring 16 for the 64-wide luma plane wrote its last row 48 bytes past a
/// `16 * 64`-byte buffer.
#[test]
fn a_stride_shorter_than_the_plane_is_refused() {
    for luma_stride in [16, -1] {
        let handle: *mut c_void = instance();
        let mut luma: Vec<u8> = vec![SENTINEL; 16 * 64 + GUARD];
        let mut chroma: Vec<u8> = vec![0; 2 * 32 * 32];
        let strides: [c_int; 3] = [luma_stride, 0, 0];
        let base: *mut u8 = chroma.as_mut_ptr();
        // SAFETY: live handle; every plane pointer is a live allocation, the
        // luma one with a guard band past `16 * 64` bytes.
        let rc: c_int = unsafe {
            let mut planes: [*mut u8; 3] = [luma.as_mut_ptr(), base, base.add(32 * 32)];
            tj3DecompressToYUVPlanes8(
                handle,
                PHOTO_420.as_ptr(),
                PHOTO_420.len(),
                planes.as_mut_ptr(),
                strides.as_ptr(),
            )
        };
        destroy(handle);
        assert_eq!(rc, -1, "stride {luma_stride}");
        assert!(
            luma.iter().all(|&byte| byte == SENTINEL),
            "stride {luma_stride}: wrote into the plane before refusing"
        );
    }
}

/// Issue #652: the YUV decompressors apply the handle's `TJPARAM_MAXMEMORY`,
/// which they ignored. A 1024x1024 4:4:4 frame's raw-decode estimate is
/// 6 MiB (`check_header_limits`: three component planes plus the packed
/// output term it counts on every path), so 5 MiB refuses and 6 MiB decodes —
/// measured. Stock's budget reaches only its whole-image arrays and refuses
/// a different set of frames, which is why this is not in the C trace.
#[test]
fn the_yuv_decompressors_honour_maxmemory() {
    let (width, height): (usize, usize) = (1024, 1024);
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();
    let jpeg: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        50,
        Subsampling::S444,
    )
    .expect("encode");
    let mut yuv: Vec<u8> = vec![0; width * height * 3];
    for (megabytes, expected) in [(5, -1), (6, 0)] {
        for planar in [false, true] {
            let handle: *mut c_void = instance();
            // SAFETY: live handle; `yuv` holds three 1024x1024 planes.
            let rc: c_int = unsafe {
                assert_eq!(tj3Set(handle, TJPARAM_MAXMEMORY, megabytes), 0);
                if planar {
                    let base: *mut u8 = yuv.as_mut_ptr();
                    let mut planes: [*mut u8; 3] =
                        [base, base.add(width * height), base.add(2 * width * height)];
                    tj3DecompressToYUVPlanes8(
                        handle,
                        jpeg.as_ptr(),
                        jpeg.len(),
                        planes.as_mut_ptr(),
                        std::ptr::null(),
                    )
                } else {
                    tj3DecompressToYUV8(handle, jpeg.as_ptr(), jpeg.len(), yuv.as_mut_ptr(), 1)
                }
            };
            destroy(handle);
            assert_eq!(rc, expected, "MAXMEMORY={megabytes} planar={planar}");
        }
    }
}

/// `TJSCALED` (`turbojpeg.h`): `(dim * num + denom - 1) / denom`.
fn tj_scaled(dimension: c_int, factor: TjScalingFactor) -> c_int {
    (dimension * factor.num + factor.denom - 1) / factor.denom
}

/// Issue #667, legacy path: the 1.x/2.x `tjDecompressToYUV2` is not exported
/// here; `docs/ABI_COMPATIBILITY.md` tells its callers to shim it onto
/// `tj3DecompressToYUV8`. Upstream's own body (`turbojpeg.c:2453-2483`) is
/// that shim: pick the first scaling factor whose output fits the requested
/// `width` x `height`, `tj3SetScalingFactor`, then the TJ3 call — and its
/// caller sized the buffer with `tjBufSizeYUV2(width, align, height, …)`.
/// Asking for 32x32 from the 64x64 frame therefore selects 1/2, which used to
/// write the 64x64 planes into a buffer sized for 32x32. Asking for the full
/// size selects 1/1 and still decodes.
#[test]
fn the_legacy_decompress_to_yuv2_shim_cannot_overrun() {
    let mut count: c_int = 0;
    // SAFETY: `count` is a live out-parameter.
    let table: *mut TjScalingFactor = unsafe { tj3GetScalingFactors(&mut count) };
    assert!(!table.is_null() && count > 0);
    // SAFETY: `tj3GetScalingFactors` returns `count` entries.
    let factors: &[TjScalingFactor] = unsafe { std::slice::from_raw_parts(table, count as usize) };

    for (requested, expected) in [(32, -1), (64, 0)] {
        let factor: TjScalingFactor = *factors
            .iter()
            .find(|&&factor| {
                // The frame is square, so one test covers width and height.
                tj_scaled(64, factor) <= requested
            })
            .expect("a factor fits");
        let size: usize = tj3YUVBufSize(requested, 4, requested, TJSAMP_420);
        assert!(size > 0);
        let mut buffer: Vec<u8> = vec![SENTINEL; size + GUARD];
        let handle: *mut c_void = instance();
        // SAFETY: live handle; `buffer` is the size the legacy caller
        // allocated plus a guard band.
        let rc: c_int = unsafe {
            assert_eq!(tj3SetScalingFactor(handle, factor), 0);
            tj3DecompressToYUV8(
                handle,
                PHOTO_420.as_ptr(),
                PHOTO_420.len(),
                buffer.as_mut_ptr(),
                4,
            )
        };
        destroy(handle);
        assert_eq!(rc, expected, "requested {requested}x{requested}");
        assert!(
            buffer[size..].iter().all(|&byte| byte == SENTINEL),
            "requested {requested}x{requested}: wrote past the {size}-byte buffer"
        );
    }
}
