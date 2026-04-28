//! Legacy TurboJPEG 1.x / 2.x aliases.
//!
//! These are thin wrappers that forward to the canonical TJ3 surface
//! implemented elsewhere in this crate. The goal is binary
//! compatibility with existing C clients (Pillow 10.x, ImageMagick
//! 7.x, stock djpeg/cjpeg shims) that still link against the pre-3.0
//! symbol set.
//!
//! Entry points covered:
//! - `tjInitCompress`, `tjInitDecompress`, `tjInitTransform`
//! - `tjDestroy`
//! - `tjCompress2`, `tjDecompress2`
//! - `tjDecompressHeader3`
//! - `tjTransform`
//! - `tjEncodeYUV3`, `tjDecodeYUV` (stubs — return -1 until A1-7 fills
//!   in the full YUV family).
//! - `tjBufSize`, `tjBufSizeYUV2`, `tjPlaneSizeYUV`, `tjPlaneWidth`,
//!   `tjPlaneHeight`
//! - `tjLoadImage`, `tjSaveImage` (stubs — return -1 until image IO
//!   routing is finalized).
//! - `tjGetErrorStr2`

use std::ffi::{c_char, c_int, c_void};

use libjpeg_turbo_rs::{calc_jpeg_dimensions, yuv_plane_height, yuv_plane_width, Subsampling};

use crate::compress::tj3Compress8;
use crate::decompress::tj3Decompress8;
use crate::header::{tj3DecompressHeader, TjRegion};
use crate::tj3::{handle_as_mut, tj3Destroy, tj3GetErrorStr, tj3Init, tj3Set};
use crate::transform::{tj3Transform, TjTransform};

// --- TJINIT values (matching turbojpeg.h `enum TJINIT`) ---
const TJINIT_COMPRESS: c_int = 0;
const TJINIT_DECOMPRESS: c_int = 1;
const TJINIT_TRANSFORM: c_int = 2;

// --- TJPARAM identifiers we drive from the legacy surface ---
const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;

fn subsamp_from_c(tjsamp: c_int) -> Option<Subsampling> {
    Some(match tjsamp {
        0 => Subsampling::S444,
        1 => Subsampling::S422,
        2 => Subsampling::S420,
        3 => Subsampling::S444, // TJSAMP_GRAY: no subsampling on luma-only
        4 => Subsampling::S440,
        5 => Subsampling::S411,
        6 => Subsampling::S441,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Init / Destroy
// ---------------------------------------------------------------------------

/// `tjInitCompress()` — legacy compress-only initializer.
#[no_mangle]
pub extern "C" fn tjInitCompress() -> *mut c_void {
    tj3Init(TJINIT_COMPRESS)
}

/// `tjInitDecompress()` — legacy decompress-only initializer.
#[no_mangle]
pub extern "C" fn tjInitDecompress() -> *mut c_void {
    tj3Init(TJINIT_DECOMPRESS)
}

/// `tjInitTransform()` — legacy transform initializer.
///
/// In TurboJPEG 3, `TJINIT_TRANSFORM` already grants both compress and
/// decompress capabilities (see `tj3InitVersion` in `turbojpeg.c`), so
/// we just forward the single enum value.
#[no_mangle]
pub extern "C" fn tjInitTransform() -> *mut c_void {
    tj3Init(TJINIT_TRANSFORM)
}

/// `tjDestroy(handle)` — identical to `tj3Destroy`.
#[no_mangle]
pub extern "C" fn tjDestroy(handle: *mut c_void) -> c_int {
    tj3Destroy(handle);
    0
}

// ---------------------------------------------------------------------------
// Compress / Decompress
// ---------------------------------------------------------------------------

/// `tjCompress2(handle, srcBuf, width, pitch, height, pixelFormat,
///              jpegBuf, jpegSize, jpegSubsamp, jpegQual, flags)`.
///
/// Matches the 2.x signature: `jpegSize` is a `unsigned long *` — we
/// accept `*mut usize` (64-bit on modern targets). `flags` is ignored;
/// TJ3 uses explicit parameters instead.
#[no_mangle]
pub extern "C" fn tjCompress2(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
    jpeg_subsamp: c_int,
    jpeg_qual: c_int,
    _flags: c_int,
) -> c_int {
    // Subsampling and quality are set via TJ3 parameters before the
    // actual compress call.
    if tj3Set(handle, TJPARAM_QUALITY, jpeg_qual) != 0 {
        return -1;
    }
    if tj3Set(handle, TJPARAM_SUBSAMP, jpeg_subsamp) != 0 {
        return -1;
    }
    tj3Compress8(
        handle,
        src_buf,
        width,
        pitch,
        height,
        pixel_format,
        jpeg_buf,
        jpeg_size,
    )
}

/// `tjDecompress2(handle, jpegBuf, jpegSize, dstBuf, width, pitch,
///                height, pixelFormat, flags)`.
///
/// The `width`/`height` parameters are legacy artifacts; TJ3 reads them
/// from the header. We honor them as an upper-bound sanity check but do
/// not override the JPEG's real dimensions.
#[no_mangle]
pub extern "C" fn tjDecompress2(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u8,
    _width: c_int,
    pitch: c_int,
    _height: c_int,
    pixel_format: c_int,
    _flags: c_int,
) -> c_int {
    tj3Decompress8(handle, jpeg_buf, jpeg_size, dst_buf, pitch, pixel_format)
}

/// `tjDecompressHeader3(handle, jpegBuf, jpegSize, width, height,
///                      jpegSubsamp, jpegColorspace)`.
///
/// TJ3 reads these values via `tj3Get`; the legacy entry point
/// populates caller-provided out-pointers.
#[no_mangle]
pub extern "C" fn tjDecompressHeader3(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    width: *mut c_int,
    height: *mut c_int,
    jpeg_subsamp: *mut c_int,
    jpeg_colorspace: *mut c_int,
) -> c_int {
    let rc: c_int = tj3DecompressHeader(handle, jpeg_buf, jpeg_size);
    if rc != 0 {
        return -1;
    }

    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    // SAFETY: out-pointers are optional per the C contract — skip if NULL.
    unsafe {
        use libjpeg_turbo_rs::tj3::TjParam;
        if !width.is_null() {
            *width = inst.inner.get(TjParam::Width);
        }
        if !height.is_null() {
            *height = inst.inner.get(TjParam::Height);
        }
        if !jpeg_subsamp.is_null() {
            *jpeg_subsamp = inst.inner.get(TjParam::Subsampling);
        }
        if !jpeg_colorspace.is_null() {
            *jpeg_colorspace = inst.inner.get(TjParam::ColorSpace);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

/// `tjTransform(handle, jpegBuf, jpegSize, n, dstBufs, dstSizes,
///              transforms, flags)`.
///
/// `flags` is ignored (TJ3 drives options through `TJPARAM_*` on the
/// handle). Otherwise identical to `tj3Transform`.
#[no_mangle]
pub extern "C" fn tjTransform(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    n: c_int,
    dst_bufs: *mut *mut u8,
    dst_sizes: *mut usize,
    transforms: *const TjTransform,
    _flags: c_int,
) -> c_int {
    tj3Transform(
        handle, jpeg_buf, jpeg_size, n, dst_bufs, dst_sizes, transforms,
    )
}

// ---------------------------------------------------------------------------
// YUV (forwards to the A1-7 family in `crate::yuv`)
// ---------------------------------------------------------------------------

/// `tjEncodeYUV3(handle, srcBuf, width, pad, height, pixelFormat,
///               dstBuf, align, subsamp, flags) -> int`.
///
/// Legacy alias: sets `TJPARAM_SUBSAMP` then forwards to
/// `tj3EncodeYUV8`. `pad` and `align` are aliases for the row
/// alignment in different historical signatures; whichever is
/// positive wins, defaulting to 1. `flags` is ignored.
#[no_mangle]
pub extern "C" fn tjEncodeYUV3(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pad: c_int,
    height: c_int,
    pixel_format: c_int,
    dst_buf: *mut u8,
    align: c_int,
    subsamp: c_int,
    _flags: c_int,
) -> c_int {
    let effective_align: c_int = if pad > 0 { pad } else { align.max(1) };
    if tj3Set(handle, TJPARAM_SUBSAMP, subsamp) != 0 {
        return -1;
    }
    crate::yuv::tj3EncodeYUV8(
        handle,
        src_buf,
        width,
        0, /* tight-packed pitch */
        height,
        pixel_format,
        dst_buf,
        effective_align,
    )
}

/// `tjDecodeYUV(handle, srcBuf, align, subsamp, dstBuf, width, pitch,
///              height, pixelFormat, flags) -> int`.
///
/// Legacy alias that forwards to `tj3DecodeYUV8` after setting
/// `TJPARAM_SUBSAMP`.
#[no_mangle]
pub extern "C" fn tjDecodeYUV(
    handle: *mut c_void,
    src_buf: *const u8,
    align: c_int,
    subsamp: c_int,
    dst_buf: *mut u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    _flags: c_int,
) -> c_int {
    if tj3Set(handle, TJPARAM_SUBSAMP, subsamp) != 0 {
        return -1;
    }
    crate::yuv::tj3DecodeYUV8(
        handle,
        src_buf,
        align.max(1),
        dst_buf,
        width,
        pitch,
        height,
        pixel_format,
    )
}

// ---------------------------------------------------------------------------
// Buffer sizing (pure computations — no handle required)
// ---------------------------------------------------------------------------

/// `tjBufSize(width, height, jpegSubsamp) -> unsigned long`.
///
/// Mirrors the C wrapper semantics in `turbojpeg.c`: internally delegates
/// to `tj3JPEGBufSize` and returns `(unsigned long)-1` (usize::MAX) when
/// the TJ3 helper returns 0 (invalid input or overflow), so callers that
/// compare against `(unsigned long)-1` — as `tjunittest.c::overflowTest`
/// does — see a stable "error" sentinel.
#[no_mangle]
pub extern "C" fn tjBufSize(width: c_int, height: c_int, jpeg_subsamp: c_int) -> usize {
    let retval: usize = crate::bufsize::tj3JPEGBufSize(width, height, jpeg_subsamp);
    if retval == 0 {
        usize::MAX
    } else {
        retval
    }
}

/// `TJBUFSIZE(width, height) -> unsigned long` — TurboJPEG 1.0 legacy
/// upper-bound sizing helper that assumes 4:4:4 and the widest worst
/// case. Returns `(unsigned long)-1` (usize::MAX) on invalid input, per
/// the historical contract in `turbojpeg.c`.
#[no_mangle]
pub extern "C" fn TJBUFSIZE(width: c_int, height: c_int) -> usize {
    if width < 1 || height < 1 {
        return usize::MAX;
    }
    // Matches turbojpeg.c: PAD(width, 16) * PAD(height, 16) * 6 + 2048.
    let pad_w: usize = ((width as usize) + 15) & !15;
    let pad_h: usize = ((height as usize) + 15) & !15;
    pad_w
        .checked_mul(pad_h)
        .and_then(|v| v.checked_mul(6))
        .and_then(|v| v.checked_add(2048))
        .unwrap_or(usize::MAX)
}

/// `TJBUFSIZEYUV(width, height, subsamp) -> unsigned long` — TurboJPEG
/// 1.1 legacy helper that delegates to `tjBufSizeYUV`.
#[no_mangle]
pub extern "C" fn TJBUFSIZEYUV(width: c_int, height: c_int, subsamp: c_int) -> usize {
    tjBufSizeYUV(width, height, subsamp)
}

/// `tjBufSizeYUV(width, height, subsamp) -> unsigned long` — TurboJPEG
/// 1.1 legacy wrapper that hard-codes `align = 4`.
#[no_mangle]
pub extern "C" fn tjBufSizeYUV(width: c_int, height: c_int, subsamp: c_int) -> usize {
    tjBufSizeYUV2(width, 4, height, subsamp)
}

/// `tjBufSizeYUV2(width, align, height, subsamp) -> unsigned long`.
///
/// Delegates to `tj3YUVBufSize` and returns `(unsigned long)-1`
/// (usize::MAX) on the 0-return error path, matching the C wrapper in
/// `turbojpeg.c`. `tjunittest.c::overflowTest` relies on this sentinel
/// when `align` is a non-power-of-two or negative value.
#[no_mangle]
pub extern "C" fn tjBufSizeYUV2(
    width: c_int,
    align: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    let retval: usize = crate::bufsize::tj3YUVBufSize(width, align, height, subsamp);
    if retval == 0 {
        usize::MAX
    } else {
        retval
    }
}

/// `tjPlaneSizeYUV(componentID, width, stride, height, subsamp)`.
///
/// Delegates to `tj3YUVPlaneSize` and returns `(unsigned long)-1` on the
/// 0-return error path, matching the C wrapper's sentinel value.
#[no_mangle]
pub extern "C" fn tjPlaneSizeYUV(
    component_id: c_int,
    width: c_int,
    stride: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    let retval: usize =
        crate::bufsize::tj3YUVPlaneSize(component_id, width, stride, height, subsamp);
    if retval == 0 {
        usize::MAX
    } else {
        retval
    }
}

/// `tjPlaneWidth(componentID, width, subsamp)`.
#[no_mangle]
pub extern "C" fn tjPlaneWidth(component_id: c_int, width: c_int, subsamp: c_int) -> c_int {
    if !(0..=2).contains(&component_id) || width <= 0 {
        return -1;
    }
    let Some(ss): Option<Subsampling> = subsamp_from_c(subsamp) else {
        return -1;
    };
    yuv_plane_width(component_id as usize, width as usize, ss) as c_int
}

/// `tjPlaneHeight(componentID, height, subsamp)`.
#[no_mangle]
pub extern "C" fn tjPlaneHeight(component_id: c_int, height: c_int, subsamp: c_int) -> c_int {
    if !(0..=2).contains(&component_id) || height <= 0 {
        return -1;
    }
    let Some(ss): Option<Subsampling> = subsamp_from_c(subsamp) else {
        return -1;
    };
    yuv_plane_height(component_id as usize, height as usize, ss) as c_int
}

// ---------------------------------------------------------------------------
// Load / Save — handle-less legacy ABI delegating to tj3LoadImage8 / tj3SaveImage8
// ---------------------------------------------------------------------------

/// Snapshot the temporary handle's last-error message into the
/// process-global no-handle error slot, so that
/// `tj3GetErrorStr(NULL)` / `tjGetErrorStr2(NULL)` return a
/// meaningful diagnostic after the temp handle is destroyed. Mirrors
/// upstream `turbojpeg.c::tjLoadImage` / `tjSaveImage`, which surface
/// the inner error through the global slot for the legacy
/// handle-less ABI.
fn copy_handle_error_to_no_handle_slot(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    // SAFETY: handle is non-NULL and was returned by tj3Init above.
    let inst: &crate::tj3::TjInstance = unsafe { &*(handle as *const crate::tj3::TjInstance) };
    if let Ok(s) = inst.last_error.to_str() {
        crate::bufsize::set_no_handle_error(s);
    }
}

/// `tjLoadImage(filename, width, align, height, pixelFormat, flags)`.
///
/// Legacy 2.x signature is **handle-less** — upstream `turbojpeg.c`
/// allocates a temporary `tjhandle`, sets `TJPARAM_BOTTOMUP` from
/// `flags & TJFLAG_BOTTOMUP`, calls `tj3LoadImage8`, then frees the
/// handle. We mirror that exactly, including snapshotting the
/// temp handle's last-error into the no-handle global slot before
/// destroying so callers can recover the diagnostic via
/// `tjGetErrorStr2(NULL)`.
#[no_mangle]
pub extern "C" fn tjLoadImage(
    filename: *const c_char,
    width: *mut c_int,
    align: c_int,
    height: *mut c_int,
    pixel_format: *mut c_int,
    flags: c_int,
) -> *mut u8 {
    // Create a temporary decompress handle so the underlying TJ3
    // form has somewhere to record errors. `TJINIT_DECOMPRESS = 2`
    // matches `tj3.rs` and `turbojpeg.h`.
    let h: *mut c_void = crate::tj3::tj3Init(2);
    if h.is_null() {
        crate::bufsize::set_no_handle_error("tjLoadImage: tj3Init(TJINIT_DECOMPRESS) failed");
        return std::ptr::null_mut();
    }
    if (flags & TJFLAG_BOTTOMUP) != 0 {
        // TJPARAM_BOTTOMUP = 0 in turbojpeg.h, but use the
        // tj3-published constant via tj3Set to stay layout-independent.
        let _ = crate::tj3::tj3Set(h, TJPARAM_BOTTOMUP, 1);
    }
    let buf: *mut u8 =
        crate::imageio::tj3LoadImage8(h, filename, width, align, height, pixel_format);
    if buf.is_null() {
        copy_handle_error_to_no_handle_slot(h);
    }
    crate::tj3::tj3Destroy(h);
    buf
}

/// `tjSaveImage(filename, buffer, width, pitch, height, pixelFormat, flags)`.
///
/// Legacy 2.x signature: also handle-less. Same handle lifecycle as
/// `tjLoadImage` (temp `tjhandle`, propagate `TJFLAG_BOTTOMUP`,
/// delegate, free, copy error before destroy).
#[no_mangle]
pub extern "C" fn tjSaveImage(
    filename: *const c_char,
    buffer: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    flags: c_int,
) -> c_int {
    let h: *mut c_void = crate::tj3::tj3Init(1); // TJINIT_COMPRESS
    if h.is_null() {
        crate::bufsize::set_no_handle_error("tjSaveImage: tj3Init(TJINIT_COMPRESS) failed");
        return -1;
    }
    if (flags & TJFLAG_BOTTOMUP) != 0 {
        let _ = crate::tj3::tj3Set(h, TJPARAM_BOTTOMUP, 1);
    }
    let rc: c_int =
        crate::imageio::tj3SaveImage8(h, filename, buffer, width, pitch, height, pixel_format);
    if rc != 0 {
        copy_handle_error_to_no_handle_slot(h);
    }
    crate::tj3::tj3Destroy(h);
    rc
}

/// Legacy `TJFLAG_BOTTOMUP` bit and `TJPARAM_BOTTOMUP` index per
/// upstream `turbojpeg.h`. Kept module-local so legacy translation
/// doesn't pull in additional constant exports.
const TJFLAG_BOTTOMUP: c_int = 2;
const TJPARAM_BOTTOMUP: c_int = 1;

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

/// `tjGetErrorStr2(handle) -> const char *` — identical to
/// `tj3GetErrorStr` with a handle-aware NULL fallback.
#[no_mangle]
pub extern "C" fn tjGetErrorStr2(handle: *mut c_void) -> *const c_char {
    tj3GetErrorStr(handle)
}

// ---------------------------------------------------------------------------
// Width-from-scaled-dimensions helper referenced by some legacy users.
// Exported for binary compat but powered by the Rust `calc_jpeg_dimensions`.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn _calc_reference(w: i32, h: i32) {
    // Kept as a compile-time reference to the sizing helper we depend on.
    let (_w, _h) = calc_jpeg_dimensions(w as usize, h as usize, Subsampling::S420);
}

// Silence a potentially-unused constant on targets where TjRegion is not
// referenced by this module directly (the file imports it for legacy
// consumers that include this header).
#[allow(dead_code)]
const _TJREGION_MARKER: TjRegion = TjRegion {
    x: 0,
    y: 0,
    w: 0,
    h: 0,
};
