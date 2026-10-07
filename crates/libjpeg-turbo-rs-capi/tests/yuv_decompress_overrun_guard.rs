//! P4-234 (#667) and P4-225 (#652): what `tj3DecompressToYUV8` /
//! `tj3DecompressToYUVPlanes8` may write, and the handle limits they apply.
//!
//! Upstream sizes YUV output from the *scaled* frame (`TJSCALED` in the packed
//! wrapper, `jpeg_calc_output_dimensions` in the planar body), so a caller
//! following `turbojpeg.h` allocates `tj3YUVBufSize(TJSCALED(w), align,
//! TJSCALED(h), subsamp)`. The port decoded at full size whatever the scaling
//! factor and wrote the unscaled planes into that buffer; it now decodes the
//! scaled planes, byte-identical to stock 3.2.0.
//!
//! Each overrun check fills the caller's buffer and an 8 KiB guard band past
//! it with a sentinel, so a write past the documented size fails the
//! assertion on any build. The sanitizers workflow also runs this file under
//! AddressSanitizer, which reports a write past the guard band as well.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::yuv::{tj3DecompressToYUV8, tj3DecompressToYUVPlanes8};
use libjpeg_turbo_rs_capi::{
    tj3DecompressHeader, tj3Destroy, tj3Get, tj3GetScalingFactors, tj3Init, tj3Set,
    tj3SetScalingFactor, tj3YUVBufSize, tj3YUVPlaneSize, TjScalingFactor,
};

const TJINIT_DECOMPRESS: c_int = 1;
const TJPARAM_JPEGWIDTH: c_int = 5;
const TJPARAM_JPEGHEIGHT: c_int = 6;
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

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(14_695_981_039_346_656_037, |hash: u64, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
        })
}

/// Issue #667: at a scaling factor other than 1/1 the planes are the scaled
/// frame's, written into exactly the `tj3YUVBufSize(TJSCALED(w), …)` bytes a
/// caller allocates and not one byte further. The port used to write the
/// unscaled planes — 6144 bytes into the 1536 of `photo_64x64_420.jpg` at
/// 1/2 — and then, as the stopgap, refused the call. Each digest is stock
/// 3.2.0's for the packed planes at align 1, so the bytes are pinned too; the
/// planar entry point writes the same planes through three pointers.
#[test]
fn a_scaled_yuv_decompress_writes_the_scaled_planes_and_nothing_past_them() {
    const PHOTO_227: &[u8] =
        include_bytes!("../../../tests/fixtures/real_world/libjpeg_testorig_227x149_baseline.jpg");
    for (jpeg, (num, denom), size, digest) in [
        (PHOTO_420, (1, 2), 1536, 0x1b8d_29b6_b6c7_3607_u64),
        (PHOTO_420, (3, 8), 864, 0xeae2_e8a9_7aa3_6b20),
        (PHOTO_420, (15, 8), 21600, 0xf603_47ed_aa0c_d20a),
        (PHOTO_227, (1, 2), 12996, 0xa858_1ec3_1932_a9ce),
        (PHOTO_227, (3, 8), 7224, 0xb858_9bbf_9123_3d6e),
        (PHOTO_227, (15, 8), 178920, 0x7a76_84df_7380_e570),
    ] {
        let (scaled_width, scaled_height): (c_int, c_int) = scaled(jpeg, num, denom);
        assert_eq!(
            tj3YUVBufSize(scaled_width, 1, scaled_height, TJSAMP_420),
            size,
            "{num}/{denom}: the documented size"
        );
        for planar in [false, true] {
            let handle: *mut c_void = instance();
            let mut buffer: Vec<u8> = vec![SENTINEL; size + GUARD];
            // SAFETY: live handle; `buffer` holds the scaled planes plus a
            // guard band, so an overrun lands in memory this test owns.
            let rc: c_int = unsafe {
                assert_eq!(
                    tj3SetScalingFactor(handle, TjScalingFactor { num, denom }),
                    0
                );
                if planar {
                    let base: *mut u8 = buffer.as_mut_ptr();
                    // The packed layout at align 1, split at its plane edges.
                    let luma: usize =
                        tj3YUVPlaneSize(0, scaled_width, 0, scaled_height, TJSAMP_420);
                    let chroma: usize =
                        tj3YUVPlaneSize(1, scaled_width, 0, scaled_height, TJSAMP_420);
                    let mut planes: [*mut u8; 3] = [base, base.add(luma), base.add(luma + chroma)];
                    tj3DecompressToYUVPlanes8(
                        handle,
                        jpeg.as_ptr(),
                        jpeg.len(),
                        planes.as_mut_ptr(),
                        std::ptr::null(),
                    )
                } else {
                    tj3DecompressToYUV8(handle, jpeg.as_ptr(), jpeg.len(), buffer.as_mut_ptr(), 1)
                }
            };
            destroy(handle);
            let case: String = format!("{num}/{denom} planar={planar}");
            assert_eq!(rc, 0, "{case}");
            assert!(
                buffer[size..].iter().all(|&byte| byte == SENTINEL),
                "{case}: wrote past the {size}-byte scaled buffer"
            );
            assert_eq!(
                fnv1a(&buffer[..size]),
                digest,
                "{case}: planes differ from stock"
            );
        }
    }
}

/// `TJSCALED(width)`, `TJSCALED(height)` of `jpeg`'s frame.
fn scaled(jpeg: &[u8], num: c_int, denom: c_int) -> (c_int, c_int) {
    let handle: *mut c_void = instance();
    // SAFETY: live handle; `jpeg` is a live slice.
    let (width, height): (c_int, c_int) = unsafe {
        assert_eq!(tj3DecompressHeader(handle, jpeg.as_ptr(), jpeg.len()), 0);
        (
            tj3Get(handle, TJPARAM_JPEGWIDTH),
            tj3Get(handle, TJPARAM_JPEGHEIGHT),
        )
    };
    destroy(handle);
    (
        (width * num + denom - 1) / denom,
        (height * num + denom - 1) / denom,
    )
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
/// write the 64x64 planes into a buffer sized for 32x32; both requests now
/// decode, each into exactly the buffer the legacy caller allocated.
#[test]
fn the_legacy_decompress_to_yuv2_shim_cannot_overrun() {
    let mut count: c_int = 0;
    // SAFETY: `count` is a live out-parameter.
    let table: *mut TjScalingFactor = unsafe { tj3GetScalingFactors(&mut count) };
    assert!(!table.is_null() && count > 0);
    // SAFETY: `tj3GetScalingFactors` returns `count` entries.
    let factors: &[TjScalingFactor] = unsafe { std::slice::from_raw_parts(table, count as usize) };

    for requested in [32, 64] {
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
        assert_eq!(rc, 0, "requested {requested}x{requested}");
        assert!(
            buffer[size..].iter().all(|&byte| byte == SENTINEL),
            "requested {requested}x{requested}: wrote past the {size}-byte buffer"
        );
    }
}
