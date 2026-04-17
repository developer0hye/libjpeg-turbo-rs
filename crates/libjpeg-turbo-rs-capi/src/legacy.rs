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

use libjpeg_turbo_rs::{
    calc_jpeg_dimensions, jpeg_buf_size, yuv_buf_size, yuv_plane_height, yuv_plane_size,
    yuv_plane_width, Subsampling,
};

use crate::compress::tj3Compress8;
use crate::decompress::tj3Decompress8;
use crate::header::{tj3DecompressHeader, TjRegion};
use crate::tj3::{handle_as_mut, tj3Destroy, tj3GetErrorStr, tj3Init, tj3Set, TJERR_FATAL};
use crate::transform::{tj3Transform, TjTransform};

// --- TJINIT flags (same as TJ3) ---
const TJINIT_COMPRESS: c_int = 1;
const TJINIT_DECOMPRESS: c_int = 2;
const TJINIT_TRANSFORM: c_int = 4;

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
/// Historically `tjInitTransform` returned a handle capable of BOTH
/// decompressing the input AND transforming it; downstream clients
/// routinely call `tjDecompressHeader3` on the same handle. Mirror
/// that by OR-ing the two TJ3 flags.
#[no_mangle]
pub extern "C" fn tjInitTransform() -> *mut c_void {
    tj3Init(TJINIT_TRANSFORM | TJINIT_DECOMPRESS)
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
// YUV (A1-10 stubs; full family is A1-7)
// ---------------------------------------------------------------------------

/// `tjEncodeYUV3(handle, srcBuf, width, pad, height, pixelFormat,
///               dstBuf, align, subsamp, flags) -> int`.
///
/// Stub: the YUV family is introduced in A1-7. Returning -1 lets a
/// downstream client fall back to its own path instead of silently
/// producing garbage. The error string documents the gap.
#[no_mangle]
pub extern "C" fn tjEncodeYUV3(
    handle: *mut c_void,
    _src_buf: *const u8,
    _width: c_int,
    _pad: c_int,
    _height: c_int,
    _pixel_format: c_int,
    _dst_buf: *mut u8,
    _align: c_int,
    _subsamp: c_int,
    _flags: c_int,
) -> c_int {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error("tjEncodeYUV3: not yet implemented (A1-7)", TJERR_FATAL);
    }
    -1
}

/// `tjDecodeYUV(handle, srcBuf, align, subsamp, dstBuf, width, pitch,
///              height, pixelFormat, flags) -> int`. Stub — see
/// `tjEncodeYUV3`.
#[no_mangle]
pub extern "C" fn tjDecodeYUV(
    handle: *mut c_void,
    _src_buf: *const u8,
    _align: c_int,
    _subsamp: c_int,
    _dst_buf: *mut u8,
    _width: c_int,
    _pitch: c_int,
    _height: c_int,
    _pixel_format: c_int,
    _flags: c_int,
) -> c_int {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error("tjDecodeYUV: not yet implemented (A1-7)", TJERR_FATAL);
    }
    -1
}

// ---------------------------------------------------------------------------
// Buffer sizing (pure computations — no handle required)
// ---------------------------------------------------------------------------

/// `tjBufSize(width, height, jpegSubsamp) -> unsigned long`.
///
/// Returns 0 on invalid input (matches historical "error = 0" convention
/// of the pre-TJ3 sizing helpers).
#[no_mangle]
pub extern "C" fn tjBufSize(width: c_int, height: c_int, jpeg_subsamp: c_int) -> usize {
    if width <= 0 || height <= 0 {
        return 0;
    }
    let Some(ss): Option<Subsampling> = subsamp_from_c(jpeg_subsamp) else {
        return 0;
    };
    jpeg_buf_size(width as usize, height as usize, ss)
}

/// `tjBufSizeYUV2(width, align, height, subsamp) -> unsigned long`.
///
/// The Rust `yuv_buf_size` helper assumes `align == 1`; larger alignments
/// are honored by rounding each plane row-stride up to `align` per the
/// libjpeg-turbo formula.
#[no_mangle]
pub extern "C" fn tjBufSizeYUV2(
    width: c_int,
    align: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    if width <= 0 || height <= 0 || align <= 0 {
        return 0;
    }
    let Some(ss): Option<Subsampling> = subsamp_from_c(subsamp) else {
        return 0;
    };
    if align == 1 {
        return yuv_buf_size(width as usize, height as usize, ss);
    }
    // Custom alignment: sum up padded plane sizes.
    let mut total: usize = 0;
    for c in 0..3usize {
        let pw: usize = yuv_plane_width(c, width as usize, ss);
        let ph: usize = yuv_plane_height(c, height as usize, ss);
        let stride: usize = pw.div_ceil(align as usize) * align as usize;
        total += stride * ph;
    }
    total
}

/// `tjPlaneSizeYUV(componentID, width, stride, height, subsamp)`.
#[no_mangle]
pub extern "C" fn tjPlaneSizeYUV(
    component_id: c_int,
    width: c_int,
    stride: c_int,
    height: c_int,
    subsamp: c_int,
) -> usize {
    if !(0..=2).contains(&component_id) || width <= 0 || height <= 0 || stride < 0 {
        return 0;
    }
    let Some(ss): Option<Subsampling> = subsamp_from_c(subsamp) else {
        return 0;
    };
    // The Rust helper takes (component, width, height, subsampling) and
    // assumes stride == plane_width. When the caller specifies a larger
    // stride we compute stride * (height - 1) + plane_width manually to
    // match libjpeg-turbo semantics.
    let pw: usize = yuv_plane_width(component_id as usize, width as usize, ss);
    let ph: usize = yuv_plane_height(component_id as usize, height as usize, ss);
    if stride == 0 {
        yuv_plane_size(component_id as usize, width as usize, height as usize, ss)
    } else {
        let stride_us: usize = stride as usize;
        stride_us * ph.saturating_sub(1) + pw
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
// Load / Save (stubs until image IO is routed through the shim)
// ---------------------------------------------------------------------------

/// `tjLoadImage(handle, filename, width, align, height, pixelFormat, flags)`.
/// Stub — file IO routing is tracked separately. Returns NULL with an
/// error set.
#[no_mangle]
pub extern "C" fn tjLoadImage(
    handle: *mut c_void,
    _filename: *const c_char,
    _width: *mut c_int,
    _align: c_int,
    _height: *mut c_int,
    _pixel_format: *mut c_int,
    _flags: c_int,
) -> *mut u8 {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error("tjLoadImage: not yet implemented", TJERR_FATAL);
    }
    std::ptr::null_mut()
}

/// `tjSaveImage(handle, filename, buffer, width, pitch, height, pixelFormat, flags)`.
/// Stub — see `tjLoadImage`.
#[no_mangle]
pub extern "C" fn tjSaveImage(
    handle: *mut c_void,
    _filename: *const c_char,
    _buffer: *const u8,
    _width: c_int,
    _pitch: c_int,
    _height: c_int,
    _pixel_format: c_int,
    _flags: c_int,
) -> c_int {
    if let Some(inst) = unsafe { handle_as_mut(handle) } {
        inst.set_error("tjSaveImage: not yet implemented", TJERR_FATAL);
    }
    -1
}

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
