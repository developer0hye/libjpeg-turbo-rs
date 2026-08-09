//! P4-127: the TurboJPEG YUV decompress entry points must validate from the
//! *header*, before decoding, the way upstream does.
//!
//! Upstream reads the header, then applies every check — maxPixels, subsampling,
//! the `dstPlanes` NULL test, and `num_components > 3` — before any
//! decompression (`references/libjpeg-turbo/src/turbojpeg.c:2214-2230`). This
//! port decoded the whole frame first and only then applied P4-125's component
//! guard, so an attacker-supplied frame still cost a full decode and every plane
//! allocation before being thrown away, and `TJPARAM_MAXPIXELS` never reached
//! the path at all: it routes through the root-crate `decompress_to_yuv_planes`,
//! which takes no handle and runs `Decoder::new` with `Limits::default`.
//!
//! "Did not decode" is asserted without timing. Each fixture keeps a **valid
//! header** and a **corrupt entropy segment**: if the implementation decodes
//! before checking, it fails with a decode error, and if it checks from the
//! header it reports the specific rejection. The reported message therefore
//! distinguishes the two orders deterministically.

use std::ffi::{c_int, c_void, CStr};

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::{
    tj3DecompressToYUV8, tj3DecompressToYUVPlanes8, tj3Destroy, tj3GetErrorStr, tj3Init, tj3Set,
    tj3YUVBufSize,
};

const TJINIT_DECOMPRESS: c_int = 2;
const TJPARAM_MAXPIXELS: c_int = 24;
const TJSAMP_444: c_int = 0;
const ALIGN: c_int = 1;
const WIDTH: usize = 64;
const HEIGHT: usize = 64;
const SENTINEL: u8 = 0xA5;

/// Truncate the entropy-coded segment while leaving every header marker intact:
/// SOI/SOF/DHT/SOS all parse, the scan data does not. A decode-first
/// implementation cannot get past this; a header-first one never looks at it.
fn corrupt_scan(mut jpeg: Vec<u8>) -> Vec<u8> {
    // Find SOS (0xFFDA), keep its segment header, drop the entropy data after it.
    let sos: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("fixture has an SOS marker");
    let seg_len: usize = u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]) as usize;
    let keep: usize = sos + 2 + seg_len + 4; // a few entropy bytes, then nothing
    assert!(keep < jpeg.len(), "fixture is too small to truncate");
    jpeg.truncate(keep);
    jpeg
}

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

fn rgb_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * 3).map(|i| (i % 251) as u8).collect();
    compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .expect("compress RGB fixture")
}

fn error_of(handle: *mut c_void) -> String {
    let ptr = tj3GetErrorStr(handle);
    assert!(!ptr.is_null(), "tj3GetErrorStr returned NULL");
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// P4-127 criterion 1: a 4-component frame is rejected from the header, so the
/// corrupt entropy segment is never reached. Decoding first would surface a
/// decode error instead of the component message.
#[test]
fn four_component_frame_is_rejected_without_decoding() {
    let jpeg: Vec<u8> = corrupt_scan(cmyk_jpeg());
    let sized: usize = tj3YUVBufSize(WIDTH as c_int, ALIGN, HEIGHT as c_int, TJSAMP_444);
    let mut dst: Vec<u8> = vec![SENTINEL; sized];

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null());
    let rc: c_int = tj3DecompressToYUV8(handle, jpeg.as_ptr(), jpeg.len(), dst.as_mut_ptr(), ALIGN);
    let err: String = error_of(handle);
    tj3Destroy(handle);

    assert_eq!(rc, -1, "4-component frame must be rejected");
    assert!(
        err.contains("3 or fewer components"),
        "expected the header-time component rejection, got {err:?} — a decode error here means \
         the frame was decoded before the check"
    );
    assert!(
        dst.iter().all(|&b| b == SENTINEL),
        "destination was written despite rejection"
    );
}

/// P4-127 criterion 3: `TJPARAM_MAXPIXELS` must bound these entry points.
/// The fixture's entropy segment is corrupt, so reaching a decode error instead
/// of the limit message also proves the limit was applied too late.
#[test]
fn max_pixels_bounds_the_packed_entry_point() {
    let jpeg: Vec<u8> = corrupt_scan(rgb_jpeg());
    let sized: usize = tj3YUVBufSize(WIDTH as c_int, ALIGN, HEIGHT as c_int, TJSAMP_444);
    let mut dst: Vec<u8> = vec![SENTINEL; sized];

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null());
    // One pixel below the frame's own size.
    let limit: c_int = (WIDTH * HEIGHT - 1) as c_int;
    assert_eq!(tj3Set(handle, TJPARAM_MAXPIXELS, limit), 0, "tj3Set");

    let rc: c_int = tj3DecompressToYUV8(handle, jpeg.as_ptr(), jpeg.len(), dst.as_mut_ptr(), ALIGN);
    let err: String = error_of(handle);
    tj3Destroy(handle);

    assert_eq!(rc, -1, "a frame over TJPARAM_MAXPIXELS must be rejected");
    assert!(
        err.to_lowercase().contains("pixel"),
        "expected a pixel-limit rejection, got {err:?}"
    );
    assert!(
        dst.iter().all(|&b| b == SENTINEL),
        "destination was written despite rejection"
    );
}

/// P4-127 criterion 4: a rejected planar call leaves every caller buffer
/// byte-for-byte unmodified. Upstream checks `dstPlanes[1]`/`[2]` up front and
/// writes nothing; decoding first and checking inside the copy loop wrote
/// planes 0 and 1 before failing on a NULL plane 2.
#[test]
fn null_plane_pointer_leaves_all_caller_buffers_untouched() {
    let jpeg: Vec<u8> = rgb_jpeg();
    let mut planes: Vec<Vec<u8>> = (0..3).map(|_| vec![SENTINEL; WIDTH * HEIGHT]).collect();
    let mut ptrs: Vec<*mut u8> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();
    ptrs[2] = std::ptr::null_mut();

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null());
    let rc: c_int = tj3DecompressToYUVPlanes8(
        handle,
        jpeg.as_ptr(),
        jpeg.len(),
        ptrs.as_mut_ptr(),
        std::ptr::null(),
    );
    tj3Destroy(handle);

    assert_eq!(rc, -1, "a NULL plane pointer must be rejected");
    for (i, plane) in planes.iter().enumerate().take(2) {
        assert!(
            plane.iter().all(|&b| b == SENTINEL),
            "plane {i} was written before the NULL plane 2 was noticed"
        );
    }
}

/// P4-127 criterion 2: when two rules are violated at once, the *precedence*
/// must match C. Upstream validates `align` at function entry
/// (`turbojpeg.c:2395-2397`, `"Invalid argument"`) — before the header is even
/// read — so a CMYK frame with `align = 0` is an argument error, not a
/// component error. Checking align inside `pack_yuv_planes` put it after
/// P4-125's guard and flipped the order.
#[test]
fn bad_align_outranks_the_component_check_as_in_c() {
    let jpeg: Vec<u8> = cmyk_jpeg();
    let mut dst: Vec<u8> = vec![SENTINEL; WIDTH * HEIGHT * 4];

    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null());
    let rc: c_int = tj3DecompressToYUV8(handle, jpeg.as_ptr(), jpeg.len(), dst.as_mut_ptr(), 0);
    let err: String = error_of(handle);
    tj3Destroy(handle);

    assert_eq!(rc, -1, "align = 0 must be rejected");
    assert!(
        !err.contains("3 or fewer components"),
        "align is validated at entry in C, so it must outrank the component check; got {err:?}"
    );
    assert!(
        err.to_lowercase().contains("align") || err.to_lowercase().contains("invalid argument"),
        "expected an argument error naming align, got {err:?}"
    );
}
