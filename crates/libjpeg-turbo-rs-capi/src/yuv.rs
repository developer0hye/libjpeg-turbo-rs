//! TJ3 YUV family (8 entry points) — A1-7.
//!
//! All YUV entry points use libjpeg-turbo's canonical plane-order:
//! Y first, then Cb, then Cr. Packed buffers concatenate the planes
//! contiguously with per-plane alignment controlled by the `align`
//! parameter (power of two, 1 = no padding).
//!
//! How many planes there are comes from the *subsampling*, not from the
//! geometry: `TJSAMP_GRAY` has one (luma alone) and every other mode has
//! three, which is the rule `tj3YUVBufSize` sizes by. `Subsampling` cannot
//! carry it — GRAY and 4:4:4 share one geometry — so the count travels
//! separately, as `plane_count_from_tj`. Losing it was P4-165.
//!
//! Signatures mirror `turbojpeg.h`:
//! ```c
//! int tj3EncodeYUV8(tjhandle handle, const unsigned char *srcBuf,
//!                   int width, int pitch, int height, int pixelFormat,
//!                   unsigned char *dstBuf, int align);
//! int tj3EncodeYUVPlanes8(tjhandle handle, const unsigned char *srcBuf,
//!                         int width, int pitch, int height, int pixelFormat,
//!                         unsigned char **dstPlanes, int *strides);
//! int tj3CompressFromYUV8(tjhandle handle, const unsigned char *srcBuf,
//!                         int width, int align, int height,
//!                         unsigned char **jpegBuf, size_t *jpegSize);
//! int tj3CompressFromYUVPlanes8(tjhandle handle,
//!                               const unsigned char * const *srcPlanes,
//!                               int width, const int *strides, int height,
//!                               unsigned char **jpegBuf, size_t *jpegSize);
//! int tj3DecompressToYUV8(tjhandle handle, const unsigned char *jpegBuf,
//!                         size_t jpegSize, unsigned char *dstBuf, int align);
//! int tj3DecompressToYUVPlanes8(tjhandle handle, const unsigned char *jpegBuf,
//!                               size_t jpegSize, unsigned char **dstPlanes,
//!                               int *strides);
//! int tj3DecodeYUV8(tjhandle handle, const unsigned char *srcBuf, int align,
//!                   unsigned char *dstBuf, int width, int pitch, int height,
//!                   int pixelFormat);
//! int tj3DecodeYUVPlanes8(tjhandle handle,
//!                         const unsigned char * const *srcPlanes,
//!                         const int *strides, unsigned char *dstBuf,
//!                         int width, int pitch, int height, int pixelFormat);
//! ```

use std::ffi::{c_int, c_void};

// The packed entry points below split the caller's buffer themselves and then
// go through the *planes* functions, rather than through `compress_from_yuv` /
// `decode_yuv`. Those two infer the plane count from the buffer *length*, and
// a GRAY buffer at 4:4:4 geometry is exactly as long as one plane of a
// three-plane image — the inference that makes them unsafe to hand a length
// this layer already knows (P4-165).
use libjpeg_turbo_rs::api::yuv::{
    compress_from_yuv_planes, decode_yuv_planes, decompress_to_yuv_planes, encode_yuv,
    encode_yuv_planes,
};
use libjpeg_turbo_rs::common::layout::checked_span;
use libjpeg_turbo_rs::tj3::FrameInfo;
use libjpeg_turbo_rs::{yuv_plane_height, yuv_plane_width, PixelFormat, Subsampling};

use crate::alloc::{deliver_compressed_output, OutputDelivery};

/// Byte length of a packed YUV buffer, or `None` if the geometry cannot
/// describe one.
///
/// `plane_count` is the caller's, not this function's, because [`Subsampling`]
/// cannot supply it: `TJSAMP_GRAY` and `TJSAMP_444` share one geometry and
/// differ only in how many planes exist (P4-165). Passing 3 for a GRAY buffer
/// produces a length roughly three times what `tj3YUVBufSize` promised the
/// caller — which then went straight into `slice::from_raw_parts` over that
/// caller's buffer.
///
/// Every factor is caller-supplied, so the product is checked rather than
/// saturated (P4-137 criterion 5). Saturation is the wrong failure here in the
/// worst possible way: the result went straight into `slice::from_raw_parts`,
/// so an overflow produced a slice of `usize::MAX` bytes — immediate UB, not a
/// short read. The `isize::MAX` bound is `from_raw_parts`'s own documented
/// precondition, and it now comes from [`checked_span`] rather than being
/// respelled here, so a plane's span and every other span in the port refuse
/// at the same boundary (P4-139).
fn packed_yuv_len(
    width: usize,
    height: usize,
    align: i32,
    ss: Subsampling,
    plane_count: usize,
) -> Option<usize> {
    // Defensive only: both callers validate `align >= 1` and power-of-two in
    // argument validation before reaching this (P4-155 #539 re-review) — this
    // clamp cannot mask a refusal the way the legacy wrappers' `.max(1)` did.
    let align: usize = (align.max(1)) as usize;
    let mut total: usize = 0;
    for c in 0..plane_count {
        let pw: usize = yuv_plane_width(c, width, ss);
        let ph: usize = yuv_plane_height(c, height, ss);
        let stride: usize = pw.div_ceil(align).checked_mul(align)?;
        let plane: usize = checked_span(&[stride, ph], "packed YUV plane").ok()?;
        total = total.checked_add(plane)?;
    }
    (total <= isize::MAX as usize).then_some(total)
}
use crate::convert::pixel_format_from_tj;
use crate::tj3::{with_handle, TjInstance, TJERR_FATAL};

/// The plane *geometry* for a TJ subsampling index.
///
/// `TJSAMP_GRAY` maps to `S444` because a grayscale image's one plane is
/// full-resolution luma, which is exactly `S444`'s plane-0 rule. It says
/// nothing about how many planes exist — see [`plane_count_from_tj`], which
/// carries the other half. Conflating the two was P4-165: every consumer
/// trusted `Subsampling` alone to imply the plane count, so a GRAY call
/// handled three planes' worth of a one-plane buffer.
fn subsamp_from_tj(tjsamp: c_int) -> Option<Subsampling> {
    Some(match tjsamp {
        0 => Subsampling::S444,
        1 => Subsampling::S422,
        2 => Subsampling::S420,
        3 => Subsampling::S444, // TJSAMP_GRAY
        4 => Subsampling::S440,
        5 => Subsampling::S411,
        6 => Subsampling::S441,
        _ => return None,
    })
}

/// How many planes a TJ subsampling index describes — upstream's
/// `nc = (subsamp == TJSAMP_GRAY ? 1 : 3)`
/// (`references/libjpeg-turbo/src/turbojpeg.c:1038`).
///
/// This is the same rule `tj3YUVBufSize` sizes by, taken from the same
/// predicate so the two cannot drift.
fn plane_count_from_tj(tjsamp: c_int) -> usize {
    if crate::bufsize::is_gray(tjsamp) {
        1
    } else {
        3
    }
}

fn current_subsampling(inst: &TjInstance) -> Option<Subsampling> {
    subsamp_from_tj(inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::Subsampling))
}

/// The plane count implied by the handle's `TJPARAM_SUBSAMP`.
///
/// Only meaningful once [`current_subsampling`] has accepted the same value;
/// every caller here reads it after that gate.
fn current_plane_count(inst: &TjInstance) -> usize {
    plane_count_from_tj(inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::Subsampling))
}

fn tjpf_from_handle(inst: &TjInstance) -> Option<PixelFormat> {
    pixel_format_from_tj(inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::ColorSpace))
}

/// Reverse the row order of a buffer in-place. `row_bytes` is the
/// stride in bytes; trailing bytes (if `bytes.len() % row_bytes !=
/// 0`) are left alone. Used to honour `TJPARAM_BOTTOMUP` in the
/// YUV entry points — mirrors upstream's `bottomUp` handling.
fn flip_rows_in_place(bytes: &mut [u8], row_bytes: usize) {
    if row_bytes == 0 {
        return;
    }
    let rows: usize = bytes.len() / row_bytes;
    if rows < 2 {
        return;
    }
    let mut top: usize = 0;
    let mut bot: usize = rows - 1;
    while top < bot {
        let (lo, hi) = bytes.split_at_mut(bot * row_bytes);
        lo[top * row_bytes..top * row_bytes + row_bytes].swap_with_slice(&mut hi[..row_bytes]);
        top += 1;
        bot -= 1;
    }
}

/// Densify a pitched RGB source into a contiguous `width * bpp * height`
/// buffer, validating `pitch` along the way.
///
/// Reads each row separately so we never construct a `&[u8]` that
/// extends past the caller's actual allocation. Upstream callers
/// (`tj3EncodeYUV8`, `tj3EncodeYUVPlanes8`, and through them
/// `tjEncodeYUV3`) accept `pitch > width * bpp`. With the per-row
/// access pattern, the maximum byte the caller must own is
/// `(height - 1) * pitch + width * bpp - 1` (the trailing pad of
/// the last row is never touched), matching libjpeg-turbo's
/// `srcBuf` size contract.
fn densify_pitched_bytes(
    src: *const u8,
    width: usize,
    height: usize,
    bpp: usize,
    pitch: c_int,
) -> Option<Vec<u8>> {
    let line: usize = if pitch == 0 {
        width * bpp
    } else if pitch < 0 {
        return None;
    } else {
        pitch as usize
    };
    if line < width * bpp {
        return None;
    }
    let row_bytes: usize = width * bpp;
    let mut out: Vec<u8> = Vec::with_capacity(row_bytes * height);
    for row in 0..height {
        // SAFETY: caller guarantees `src + row * line` is valid for
        // `row_bytes` bytes (libjpeg-turbo's contract: each row's
        // first `width * bpp` bytes are valid; the trailing pitch
        // padding may not be allocated and must not be read).
        let row_slice: &[u8] =
            unsafe { std::slice::from_raw_parts(src.add(row * line), row_bytes) };
        out.extend_from_slice(row_slice);
    }
    Some(out)
}

/// Split the packed YUV buffer into dense per-plane buffers, given the
/// subsampling, the `align` row-stride, and how many planes the buffer holds.
/// Returns `None` if `align` is non-positive, `align` is not a power of two,
/// or the buffer is shorter than the planes described.
///
/// `plane_count` is 1 for `TJSAMP_GRAY` (Y only) and 3 otherwise (Y|Cb|Cr) —
/// see [`plane_count_from_tj`]. It cannot be derived from `subsampling`, which
/// is `S444` for both GRAY and 4:4:4.
fn split_packed_yuv(
    yuv: &[u8],
    width: usize,
    height: usize,
    align: c_int,
    subsampling: Subsampling,
    plane_count: usize,
) -> Option<Vec<Vec<u8>>> {
    if align <= 0 || (align as usize).count_ones() != 1 {
        return None;
    }
    let align_us: usize = align as usize;

    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(plane_count);
    let mut offset: usize = 0;
    for component in 0..plane_count {
        let pw: usize = yuv_plane_width(component, width, subsampling);
        let ph: usize = yuv_plane_height(component, height, subsampling);
        let stride: usize = pw.div_ceil(align_us) * align_us;
        let plane_total: usize = stride * ph;
        if offset + plane_total > yuv.len() {
            return None;
        }
        let mut packed: Vec<u8> = Vec::with_capacity(pw * ph);
        for row in 0..ph {
            let row_start: usize = offset + row * stride;
            packed.extend_from_slice(&yuv[row_start..row_start + pw]);
        }
        planes.push(packed);
        offset += plane_total;
    }
    Some(planes)
}

/// Concatenate plane buffers into a packed output buffer with the
/// given `align` row-stride, and return it.
///
/// The plane count is `planes.len()`, so callers must hand over exactly the
/// planes the destination is sized for: one for `TJSAMP_GRAY`, three
/// otherwise. Packing a chroma plane a GRAY caller did not allocate is the
/// out-of-bounds write half of P4-165.
fn pack_yuv_planes(
    planes: &[Vec<u8>],
    width: usize,
    height: usize,
    align: c_int,
    subsampling: Subsampling,
) -> Option<Vec<u8>> {
    if align <= 0 || (align as usize).count_ones() != 1 {
        return None;
    }
    let align_us: usize = align as usize;
    let mut out: Vec<u8> = Vec::new();
    for (component, plane) in planes.iter().enumerate() {
        let pw: usize = yuv_plane_width(component, width, subsampling);
        let ph: usize = yuv_plane_height(component, height, subsampling);
        let stride: usize = pw.div_ceil(align_us) * align_us;
        if plane.len() < pw * ph {
            return None;
        }
        for row in 0..ph {
            out.extend_from_slice(&plane[row * pw..row * pw + pw]);
            // Zero-pad to stride. `repeat_n` is what clippy wants over
            // a hot loop of `out.push(0)`.
            out.extend(std::iter::repeat_n(0u8, stride.saturating_sub(pw)));
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Pixels (RGB/BGR/Gray/...) → YUV (no JPEG involvement)
// ---------------------------------------------------------------------------
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3EncodeYUV8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    dst_buf: *mut u8,
    align: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            // Upstream's packed wrapper validates dims, `align` (a power of
            // two) and the buffers before anything else
            // (`turbojpeg.c:1745-1750`); the pixel-format range check lives in
            // the `…Planes8` delegate, *downstream* of the subsampling gate.
            if src_buf.is_null()
                || dst_buf.is_null()
                || width <= 0
                || height <= 0
                || align < 1
                || (align & (align - 1)) != 0
            {
                inst.set_error("tj3EncodeYUV8: invalid pointer/dim/align", TJERR_FATAL);
                return -1;
            }
            let ss: Subsampling = match current_subsampling(inst) {
                Some(s) => s,
                None => {
                    // P4-155 (#539): *unset* is upstream's "must be specified",
                    // not "bad" — the mapping cannot tell -1 from garbage.
                    if !crate::tj3::require_specified(inst, "tj3EncodeYUV8", false, true) {
                        return -1;
                    }
                    inst.set_error("tj3EncodeYUV8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
                    return -1;
                }
            };
            let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
                Some(p) => p,
                None => {
                    inst.set_error("tj3EncodeYUV8: unsupported TJPF", TJERR_FATAL);
                    return -1;
                }
            };

            let w: usize = width as usize;
            let h: usize = height as usize;
            let bpp: usize = pf.bytes_per_pixel();
            let mut dense: Vec<u8> = match densify_pitched_bytes(src_buf, w, h, bpp, pitch) {
                Some(v) => v,
                None => {
                    inst.set_error("tj3EncodeYUV8: bad pitch", TJERR_FATAL);
                    return -1;
                }
            };

            // Bottom-up: caller's buffer is rows bottom-to-top. Flip rows in
            // place so the encoder reads them in canonical top-to-bottom
            // order. Mirrors upstream `turbojpeg.c::tjEncodeYUV3` which
            // honours `bottomUp` via `cinfo->next_scanline` ordering.
            if inst.bottom_up_flag() {
                flip_rows_in_place(&mut dense, w * bpp);
            }

            let planes: Vec<Vec<u8>> = match encode_yuv_planes(&dense, w, h, pf, ss) {
                Ok(p) => p,
                Err(e) => {
                    inst.set_error(format!("tj3EncodeYUV8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };
            // GRAY keeps only the luma plane (P4-165). `encode_yuv_planes`
            // works from the *pixel* format, so an RGB source yields three
            // planes whatever the subsampling; upstream instead sets
            // `jpeg_color_space = JCS_GRAYSCALE` and emits one
            // (`turbojpeg.c:1756-1759`, which passes NULL for planes 1 and 2).
            // Plane 0 is already right: RGB→Y does not depend on the chroma
            // sampling that is being dropped.
            //
            // The `min` also covers the reverse mismatch — a grayscale source
            // under a non-grayscale subsampling, one plane where three are
            // wanted. Upstream refuses that call outright
            // (`JERR_CONVERSION_NOTIMPL`, `jccolor.c:624-628`); we accept it
            // and fill only the luma third. Pre-existing, tracked as P4-166.
            let emitted: usize = current_plane_count(inst).min(planes.len());
            let packed: Vec<u8> = match pack_yuv_planes(&planes[..emitted], w, h, align, ss) {
                Some(p) => p,
                None => {
                    inst.set_error(
                        "tj3EncodeYUV8: align must be a positive power of 2",
                        TJERR_FATAL,
                    );
                    return -1;
                }
            };

            // SAFETY: caller guarantees `dst_buf` is sized per
            // `tjBufSizeYUV2(width, align, height, subsamp)`, which sizes
            // `current_plane_count(inst)` planes. `packed` holds `emitted`
            // planes and `emitted <= current_plane_count(inst)`, so the copy
            // never exceeds it. The two are equal except for a grayscale
            // *source* under a non-grayscale subsampling, where this
            // under-writes rather than over-writes — a divergence from
            // upstream, which refuses that combination (P4-166).
            unsafe {
                std::ptr::copy_nonoverlapping(packed.as_ptr(), dst_buf, packed.len());
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `dst_planes`, `strides` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3EncodeYUVPlanes8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    dst_planes: *mut *mut u8,
    strides: *const c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            if src_buf.is_null() || dst_planes.is_null() || width <= 0 || height <= 0 {
                inst.set_error("tj3EncodeYUVPlanes8: NULL / bad dim", TJERR_FATAL);
                return -1;
            }
            let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
                Some(p) => p,
                None => {
                    inst.set_error("tj3EncodeYUVPlanes8: unsupported TJPF", TJERR_FATAL);
                    return -1;
                }
            };
            let ss: Subsampling = match current_subsampling(inst) {
                Some(s) => s,
                None => {
                    // P4-155 (#539): *unset* is upstream's "must be specified",
                    // not "bad" — the mapping cannot tell -1 from garbage.
                    if !crate::tj3::require_specified(inst, "tj3EncodeYUVPlanes8", false, true) {
                        return -1;
                    }
                    inst.set_error("tj3EncodeYUVPlanes8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
                    return -1;
                }
            };
            let w: usize = width as usize;
            let h: usize = height as usize;
            let bpp: usize = pf.bytes_per_pixel();
            let mut dense: Vec<u8> = match densify_pitched_bytes(src_buf, w, h, bpp, pitch) {
                Some(v) => v,
                None => {
                    inst.set_error("tj3EncodeYUVPlanes8: bad pitch", TJERR_FATAL);
                    return -1;
                }
            };

            // Bottom-up: caller's buffer is rows bottom-to-top. Flip in
            // place so the encoder reads canonical top-to-bottom order.
            // Mirrors upstream's `bottomUp` handling in `tj3EncodeYUVPlanes8`.
            if inst.bottom_up_flag() {
                flip_rows_in_place(&mut dense, w * bpp);
            }

            let planes: Vec<Vec<u8>> = match encode_yuv_planes(&dense, w, h, pf, ss) {
                Ok(p) => p,
                Err(e) => {
                    inst.set_error(format!("tj3EncodeYUVPlanes8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };
            // GRAY writes plane 0 only (P4-165). Upstream never touches
            // `dstPlanes[1]`/`[2]` for it — its own packed wrapper passes NULL
            // there (`turbojpeg.c:1756-1759`) and its NULL check exempts GRAY
            // (`:1589-1590`) — so a caller may legally supply a one-element
            // array, and walking three slots reads past it.
            //
            // The `min` also covers the reverse mismatch — a grayscale source
            // under a non-grayscale subsampling, which upstream refuses
            // (`JERR_CONVERSION_NOTIMPL`, `jccolor.c:624-628`) and we accept,
            // leaving the caller's chroma planes untouched. Pre-existing,
            // tracked as P4-166.
            let emitted: usize = current_plane_count(inst).min(planes.len());

            // SAFETY: caller guarantees `dst_planes[i]` / `strides[i]` are valid
            // for `emitted` entries and each buffer is sized to hold
            // `stride * plane_height` samples.
            unsafe {
                for (i, plane) in planes[..emitted].iter().enumerate() {
                    let pw: usize = yuv_plane_width(i, w, ss);
                    let ph: usize = yuv_plane_height(i, h, ss);
                    let dst: *mut u8 = *dst_planes.add(i);
                    if dst.is_null() {
                        inst.set_error("tj3EncodeYUVPlanes8: NULL destination plane", TJERR_FATAL);
                        return -1;
                    }
                    let stride: usize = if strides.is_null() {
                        pw
                    } else {
                        let s: c_int = *strides.add(i);
                        if s <= 0 {
                            pw
                        } else {
                            s as usize
                        }
                    };
                    for row in 0..ph {
                        let src_row = plane[row * pw..row * pw + pw].as_ptr();
                        let dst_row = dst.add(row * stride);
                        std::ptr::copy_nonoverlapping(src_row, dst_row, pw);
                    }
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

// ---------------------------------------------------------------------------
// Pixels → YUV → JPEG (compress path)
// ---------------------------------------------------------------------------
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `jpeg_buf`, `jpeg_size` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
///
/// A non-null `*jpeg_buf` is **freed by this function only when
/// `TJPARAM_NOREALLOC` is unset** — see
/// [Ownership transfer](crate#pointer-contract). With the flag set the output
/// is written in place, your pointer comes back unchanged, and `*jpeg_size` is
/// an *input* carrying the buffer's capacity: output that does not fit is an
/// error rather than a resize (P4-145).
#[no_mangle]
pub unsafe extern "C" fn tj3CompressFromYUV8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    align: c_int,
    height: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            // Upstream validates dims and `align` first (`turbojpeg.c:1493-1496`).
            if src_buf.is_null()
                || jpeg_buf.is_null()
                || jpeg_size.is_null()
                || width <= 0
                || height <= 0
                || align < 1
                || (align & (align - 1)) != 0
            {
                inst.set_error("tj3CompressFromYUV8: NULL / bad dim/align", TJERR_FATAL);
                return -1;
            }
            // P4-155 (#539): this entry gates the *subsampling* itself — it
            // needs it to size the packed planes (`turbojpeg.c:1497-1498`) —
            // and the quality gate is reached only through the `…Planes8`
            // delegate (`:1347-1350`), so with both unset upstream reports
            // TJPARAM_SUBSAMP first. Order pinned by the oracle's
            // `p4155_fromyuv8_unset` line (#548-review round).
            let ss: Subsampling = match current_subsampling(inst) {
                Some(s) => s,
                None => {
                    // *Unset* is upstream's "must be specified", not "bad" —
                    // the mapping cannot tell -1 from garbage.
                    if !crate::tj3::require_specified(inst, "tj3CompressFromYUV8", false, true) {
                        return -1;
                    }
                    inst.set_error("tj3CompressFromYUV8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
                    return -1;
                }
            };
            // The delegate's unconditional quality gate — YUV compression is
            // inherently lossy (`turbojpeg.c:1347-1350`).
            if !crate::tj3::require_specified(inst, "tj3CompressFromYUV8", true, false) {
                return -1;
            }
            let quality: i32 = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::Quality);

            // Compute the raw packed YUV length from align+dims. GRAY is one
            // plane, which is what `tj3YUVBufSize` promised this caller
            // (P4-165); reading three from a buffer sized for one ran the
            // slice below past the allocation.
            let plane_count: usize = current_plane_count(inst);
            let total: usize =
                match packed_yuv_len(width as usize, height as usize, align, ss, plane_count) {
                    Some(t) => t,
                    None => {
                        inst.set_error(
                            "tj3CompressFromYUV8: image dimensions overflow the buffer size",
                            TJERR_FATAL,
                        );
                        return -1;
                    }
                };
            // SAFETY: caller asserted `src_buf` valid for `total` bytes per
            // `tjBufSizeYUV2(width, align, height, subsamp)`, and `total` is
            // checked arithmetic bounded by `isize::MAX`.
            let packed: &[u8] = unsafe { std::slice::from_raw_parts(src_buf, total) };

            // Unpack into dense planes for the Rust-side compressor, which
            // reads the plane count from the slice it is handed — one plane
            // means a grayscale (1-component) JPEG, matching upstream's
            // `setCompDefaults` under TJSAMP_GRAY.
            let planes: Vec<Vec<u8>> = match split_packed_yuv(
                packed,
                width as usize,
                height as usize,
                align,
                ss,
                plane_count,
            ) {
                Some(p) => p,
                None => {
                    inst.set_error("tj3CompressFromYUV8: bad packed YUV layout", TJERR_FATAL);
                    return -1;
                }
            };
            let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();

            let jpeg: Vec<u8> = match compress_from_yuv_planes(
                &plane_refs,
                width as usize,
                height as usize,
                ss,
                quality as u8,
            ) {
                Ok(v) => v,
                Err(e) => {
                    inst.set_error(format!("tj3CompressFromYUV8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };

            // P4-145: honour `TJPARAM_NOREALLOC`. This used to allocate a fresh
            // buffer and `free()` the previous pointee unconditionally — including
            // when the caller had set the flag, which is exactly when the buffer is
            // not `malloc`-owned. A stack array handed here was freed with the
            // wrong allocator.
            let norealloc: bool = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) != 0;
            // SAFETY: both out-pointers were validated non-NULL above; the caller's
            // contract covers the buffer's validity and non-aliasing with `jpeg`,
            // which this function owns.
            match unsafe { deliver_compressed_output(&jpeg, jpeg_buf, jpeg_size, norealloc) } {
                OutputDelivery::Delivered => {}
                OutputDelivery::BufferTooSmall { needed, capacity } => {
                    inst.set_error(
                        format!(
                            "tj3CompressFromYUV8: TJPARAM_NOREALLOC is set and the JPEG buffer is too small \
                             ({needed} bytes needed, {capacity} available)"
                        ),
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::NoBufferSupplied => {
                    inst.set_error(
                        "tj3CompressFromYUV8: TJPARAM_NOREALLOC is set but no output buffer was supplied",
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::OutOfMemory => {
                    inst.set_error("tj3CompressFromYUV8: OOM", TJERR_FATAL);
                    return -1;
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}
/// # Safety
///
/// C ABI entry point. `handle`, `src_planes`, `strides`, `jpeg_buf`, `jpeg_size` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
///
/// A non-null `*jpeg_buf` is **freed by this function only when
/// `TJPARAM_NOREALLOC` is unset** — see
/// [Ownership transfer](crate#pointer-contract). With the flag set the output
/// is written in place, your pointer comes back unchanged, and `*jpeg_size` is
/// an *input* carrying the buffer's capacity: output that does not fit is an
/// error rather than a resize (P4-145).
#[no_mangle]
pub unsafe extern "C" fn tj3CompressFromYUVPlanes8(
    handle: *mut c_void,
    src_planes: *const *const u8,
    width: c_int,
    strides: *const c_int,
    height: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            if src_planes.is_null()
                || jpeg_buf.is_null()
                || jpeg_size.is_null()
                || width <= 0
                || height <= 0
            {
                inst.set_error("tj3CompressFromYUVPlanes8: NULL / bad dim", TJERR_FATAL);
                return -1;
            }
            // P4-155 (#539): YUV compression is inherently lossy, so upstream
            // gates TJPARAM_QUALITY unconditionally, before the subsampling
            // (`turbojpeg.c:1347-1350`).
            if !crate::tj3::require_specified(inst, "tj3CompressFromYUVPlanes8", true, false) {
                return -1;
            }
            let ss: Subsampling = match current_subsampling(inst) {
                Some(s) => s,
                None => {
                    // P4-155 (#539): *unset* is upstream's "must be specified",
                    // not "bad" — the mapping cannot tell -1 from garbage.
                    if !crate::tj3::require_specified(
                        inst,
                        "tj3CompressFromYUVPlanes8",
                        false,
                        true,
                    ) {
                        return -1;
                    }
                    inst.set_error(
                        "tj3CompressFromYUVPlanes8: bad TJPARAM_SUBSAMP",
                        TJERR_FATAL,
                    );
                    return -1;
                }
            };
            let quality: i32 = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::Quality);

            // Collect dense per-plane slices, respecting caller strides. GRAY
            // reads `srcPlanes[0]` alone (P4-165): upstream's own packed
            // wrapper passes NULL for slots 1 and 2 (`turbojpeg.c:1504-1507`)
            // and its NULL check exempts GRAY (`:1344-1345`), so a caller may
            // legally supply a one-element array.
            let plane_count: usize = current_plane_count(inst);
            // SAFETY: caller guarantees `plane_count` plane pointers + strides valid.
            let mut owned_planes: Vec<Vec<u8>> = Vec::with_capacity(plane_count);
            for c in 0..plane_count {
                let pw: usize = yuv_plane_width(c, width as usize, ss);
                let ph: usize = yuv_plane_height(c, height as usize, ss);
                unsafe {
                    let plane_ptr: *const u8 = *src_planes.add(c);
                    if plane_ptr.is_null() {
                        inst.set_error("tj3CompressFromYUVPlanes8: NULL plane", TJERR_FATAL);
                        return -1;
                    }
                    let stride: usize = if strides.is_null() {
                        pw
                    } else {
                        let s: c_int = *strides.add(c);
                        if s <= 0 {
                            pw
                        } else {
                            s as usize
                        }
                    };
                    let mut dense: Vec<u8> = Vec::with_capacity(pw * ph);
                    for row in 0..ph {
                        let row_ptr = plane_ptr.add(row * stride);
                        let row_slice: &[u8] = std::slice::from_raw_parts(row_ptr, pw);
                        dense.extend_from_slice(row_slice);
                    }
                    owned_planes.push(dense);
                }
            }
            let slice_refs: Vec<&[u8]> = owned_planes.iter().map(|v| v.as_slice()).collect();

            let jpeg: Vec<u8> = match compress_from_yuv_planes(
                &slice_refs,
                width as usize,
                height as usize,
                ss,
                quality as u8,
            ) {
                Ok(v) => v,
                Err(e) => {
                    inst.set_error(format!("tj3CompressFromYUVPlanes8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };
            // P4-145: honour `TJPARAM_NOREALLOC`. This used to allocate a fresh
            // buffer and `free()` the previous pointee unconditionally — including
            // when the caller had set the flag, which is exactly when the buffer is
            // not `malloc`-owned. A stack array handed here was freed with the
            // wrong allocator.
            let norealloc: bool = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::NoRealloc) != 0;
            // SAFETY: both out-pointers were validated non-NULL above; the caller's
            // contract covers the buffer's validity and non-aliasing with `jpeg`,
            // which this function owns.
            match unsafe { deliver_compressed_output(&jpeg, jpeg_buf, jpeg_size, norealloc) } {
                OutputDelivery::Delivered => {}
                OutputDelivery::BufferTooSmall { needed, capacity } => {
                    inst.set_error(
                        format!(
                            "tj3CompressFromYUVPlanes8: TJPARAM_NOREALLOC is set and the JPEG buffer is too small \
                             ({needed} bytes needed, {capacity} available)"
                        ),
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::NoBufferSupplied => {
                    inst.set_error(
                        "tj3CompressFromYUVPlanes8: TJPARAM_NOREALLOC is set but no output buffer was supplied",
                        TJERR_FATAL,
                    );
                    return -1;
                }
                OutputDelivery::OutOfMemory => {
                    inst.set_error("tj3CompressFromYUVPlanes8: OOM", TJERR_FATAL);
                    return -1;
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

// ---------------------------------------------------------------------------
// JPEG → YUV (decompress path, no color conversion)
// ---------------------------------------------------------------------------

/// Upper bound on the planes the TurboJPEG YUV ABI can describe.
///
/// `tj3YUVBufSize` only ever sizes 1 (grayscale) or 3 (Y/Cb/Cr) planes,
/// and `dstPlanes` is documented as a 3-element array. The plane count of
/// a decompressed frame, in contrast, comes from the JPEG's own SOF
/// marker, so a 4-component (CMYK/YCCK) stream would make the decompress
/// sinks below emit a plane the caller never allocated. Both entry points
/// therefore reject those frames, mirroring the guard upstream applies in
/// `tj3DecompressToYUVPlanes8` (turbojpeg.c).
const MAX_YUV_PLANES: usize = 3;
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3DecompressToYUV8(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u8,
    align: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            // Upstream validates every argument at function entry, before the
            // header is read (turbojpeg.c:2395-2397, `"Invalid argument"`). `align`
            // used to be discovered inside `pack_yuv_planes`, i.e. after the
            // component guard, which flipped the precedence C reports (P4-127).
            if jpeg_buf.is_null() || dst_buf.is_null() || jpeg_size < 2 {
                inst.set_error("tj3DecompressToYUV8: NULL / size", TJERR_FATAL);
                return -1;
            }
            if align < 1 || !(align as u32).is_power_of_two() {
                inst.set_error(
                    "tj3DecompressToYUV8: Invalid argument (align must be a positive power of 2)",
                    TJERR_FATAL,
                );
                return -1;
            }
            let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };
            // Header first, decode second — the whole point of P4-127. This also
            // applies the handle's TJPARAM_MAXPIXELS, which the handle-less
            // `decompress_to_yuv_planes` below cannot see.
            let info: FrameInfo = match inst.inner.inspect_header(jpeg) {
                Ok(info) => info,
                Err(e) => {
                    inst.set_error(format!("tj3DecompressToYUV8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };
            if info.num_components > MAX_YUV_PLANES {
                inst.set_error(
                    "tj3DecompressToYUV8: JPEG image must have 3 or fewer components",
                    TJERR_FATAL,
                );
                return -1;
            }
            let (planes, w, h, ss) = match decompress_to_yuv_planes(jpeg) {
                Ok(v) => v,
                Err(e) => {
                    inst.set_error(format!("tj3DecompressToYUV8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };
            // Bounded by `MAX_YUV_PLANES`, not merely by the guard above: that
            // guard checked `inspect_header`'s component count, while `planes`
            // comes from a second, independent parse. The caller sized
            // `dst_buf` for at most three planes, so this is the count that
            // must bound what is packed into it — the two parses agreeing is
            // an invariant of this crate, not of the caller's contract.
            let packed: Vec<u8> =
                match pack_yuv_planes(&planes[..planes.len().min(MAX_YUV_PLANES)], w, h, align, ss)
                {
                    Some(p) => p,
                    None => {
                        inst.set_error("tj3DecompressToYUV8: bad align", TJERR_FATAL);
                        return -1;
                    }
                };
            // SAFETY: caller asserts dst_buf is large enough.
            unsafe {
                std::ptr::copy_nonoverlapping(packed.as_ptr(), dst_buf, packed.len());
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_planes`, `strides` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3DecompressToYUVPlanes8(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_planes: *mut *mut u8,
    strides: *const c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            if jpeg_buf.is_null() || dst_planes.is_null() || jpeg_size < 2 {
                inst.set_error("tj3DecompressToYUVPlanes8: NULL / size", TJERR_FATAL);
                return -1;
            }
            let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };
            // Header first (P4-127): this applies the handle's TJPARAM_MAXPIXELS and
            // settles the component count before a single MCU is decoded.
            let info: FrameInfo = match inst.inner.inspect_header(jpeg) {
                Ok(info) => info,
                Err(e) => {
                    inst.set_error(format!("tj3DecompressToYUVPlanes8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };
            if info.num_components > MAX_YUV_PLANES {
                inst.set_error(
                    "tj3DecompressToYUVPlanes8: JPEG image must have 3 or fewer components",
                    TJERR_FATAL,
                );
                return -1;
            }
            // Upstream rejects a NULL chroma plane up front, so nothing is written
            // when it fails (turbojpeg.c:2226-2227). Checking inside the copy loop
            // below meant planes 0 and 1 were already in caller memory by the time a
            // NULL plane 2 was noticed.
            let plane_count: usize = info.num_components.min(MAX_YUV_PLANES);
            for i in 0..plane_count {
                if unsafe { *dst_planes.add(i) }.is_null() {
                    inst.set_error(
                        "tj3DecompressToYUVPlanes8: Invalid argument (NULL plane pointer)",
                        TJERR_FATAL,
                    );
                    return -1;
                }
            }
            let (planes, w, h, ss) = match decompress_to_yuv_planes(jpeg) {
                Ok(v) => v,
                Err(e) => {
                    inst.set_error(format!("tj3DecompressToYUVPlanes8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };
            // `take(MAX_YUV_PLANES)` for the same reason the packed sibling
            // clamps: the `> MAX_YUV_PLANES` guard above tested
            // `inspect_header`'s count, while `planes` comes from a second,
            // independent parse, and `dstPlanes` is a three-element array in
            // `turbojpeg.h`. The two parses agreeing is an invariant of this
            // crate, not something the caller's array size can rely on.
            unsafe {
                for (i, plane) in planes.iter().take(MAX_YUV_PLANES).enumerate() {
                    let pw: usize = yuv_plane_width(i, w, ss);
                    let ph: usize = yuv_plane_height(i, h, ss);
                    let dst: *mut u8 = *dst_planes.add(i);
                    if dst.is_null() {
                        inst.set_error("tj3DecompressToYUVPlanes8: NULL plane dst", TJERR_FATAL);
                        return -1;
                    }
                    let stride: usize = if strides.is_null() {
                        pw
                    } else {
                        let s: c_int = *strides.add(i);
                        if s <= 0 {
                            pw
                        } else {
                            s as usize
                        }
                    };
                    for row in 0..ph {
                        let src_row = plane[row * pw..row * pw + pw].as_ptr();
                        let dst_row = dst.add(row * stride);
                        std::ptr::copy_nonoverlapping(src_row, dst_row, pw);
                    }
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

// ---------------------------------------------------------------------------
// YUV → Pixels (color conversion only, no JPEG)
// ---------------------------------------------------------------------------
/// # Safety
///
/// C ABI entry point. `handle`, `src_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3DecodeYUV8(
    handle: *mut c_void,
    src_buf: *const u8,
    align: c_int,
    dst_buf: *mut u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            // Upstream's packed wrapper validates dims, `align` and the
            // buffers first (`turbojpeg.c:2721-2726`); the pixel-format range
            // check lives in the `…Planes8` delegate, *downstream* of the
            // subsampling gate.
            if src_buf.is_null()
                || dst_buf.is_null()
                || width <= 0
                || height <= 0
                || align < 1
                || (align & (align - 1)) != 0
            {
                inst.set_error("tj3DecodeYUV8: NULL / bad dim/align", TJERR_FATAL);
                return -1;
            }
            let ss: Subsampling = match current_subsampling(inst) {
                Some(s) => s,
                None => {
                    // P4-155 (#539): *unset* is upstream's "must be specified",
                    // not "bad" — the mapping cannot tell -1 from garbage.
                    if !crate::tj3::require_specified(inst, "tj3DecodeYUV8", false, true) {
                        return -1;
                    }
                    inst.set_error("tj3DecodeYUV8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
                    return -1;
                }
            };
            let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
                Some(p) => p,
                None => {
                    inst.set_error("tj3DecodeYUV8: unsupported TJPF", TJERR_FATAL);
                    return -1;
                }
            };

            let w: usize = width as usize;
            let h: usize = height as usize;
            // Compute packed length. GRAY is one plane, which is what
            // `tj3YUVBufSize` promised this caller (P4-165); reading three
            // both ran the slice below past the allocation and folded whatever
            // followed it into the image as chroma.
            let plane_count: usize = current_plane_count(inst);
            let total: usize = match packed_yuv_len(w, h, align, ss, plane_count) {
                Some(t) => t,
                None => {
                    inst.set_error(
                        "tj3DecodeYUV8: image dimensions overflow the buffer size",
                        TJERR_FATAL,
                    );
                    return -1;
                }
            };
            // SAFETY: caller asserted `src_buf` valid for `total` bytes, and
            // `total` is checked arithmetic bounded by `isize::MAX`.
            let packed: &[u8] = unsafe { std::slice::from_raw_parts(src_buf, total) };
            let planes: Vec<Vec<u8>> = match split_packed_yuv(packed, w, h, align, ss, plane_count)
            {
                Some(p) => p,
                None => {
                    inst.set_error("tj3DecodeYUV8: bad packed YUV layout", TJERR_FATAL);
                    return -1;
                }
            };
            // One plane means grayscale: `decode_yuv_planes` then writes
            // R = G = B = Y, which is what YCbCr→RGB collapses to at
            // Cb = Cr = 128 and what upstream's `setDecodeDefaults` produces
            // under TJSAMP_GRAY (`turbojpeg.c:2502-2504`).
            let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
            let pixels: Vec<u8> = match decode_yuv_planes(&plane_refs, w, h, ss, pf) {
                Ok(v) => v,
                Err(e) => {
                    inst.set_error(format!("tj3DecodeYUV8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };

            let bpp: usize = pf.bytes_per_pixel();
            let dst_stride: usize = if pitch == 0 {
                w * bpp
            } else if pitch < 0 {
                inst.set_error("tj3DecodeYUV8: negative pitch", TJERR_FATAL);
                return -1;
            } else {
                pitch as usize
            };
            if dst_stride < w * bpp {
                inst.set_error("tj3DecodeYUV8: pitch too small", TJERR_FATAL);
                return -1;
            }
            let bottom_up: bool = inst.bottom_up_flag();
            // SAFETY: caller guarantees dst is `dst_stride * h` bytes.
            unsafe {
                for row in 0..h {
                    let src_row = pixels[row * w * bpp..row * w * bpp + w * bpp].as_ptr();
                    // Bottom-up: row `i` of the decoded image goes to
                    // `height - i - 1` in the caller's buffer, mirroring
                    // upstream's `bottomUp` write loop.
                    let dst_row_index: usize = if bottom_up { h - 1 - row } else { row };
                    let dst_row = dst_buf.add(dst_row_index * dst_stride);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, w * bpp);
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}
/// # Safety
///
/// C ABI entry point. `handle`, `src_planes`, `strides`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3DecodeYUVPlanes8(
    handle: *mut c_void,
    src_planes: *const *const u8,
    strides: *const c_int,
    dst_buf: *mut u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut TjInstance| -> c_int {
            if src_planes.is_null() || dst_buf.is_null() || width <= 0 || height <= 0 {
                inst.set_error("tj3DecodeYUVPlanes8: NULL / bad dim", TJERR_FATAL);
                return -1;
            }
            let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
                Some(p) => p,
                None => {
                    inst.set_error("tj3DecodeYUVPlanes8: unsupported TJPF", TJERR_FATAL);
                    return -1;
                }
            };
            let ss: Subsampling = match current_subsampling(inst) {
                Some(s) => s,
                None => {
                    // P4-155 (#539): *unset* is upstream's "must be specified",
                    // not "bad" — the mapping cannot tell -1 from garbage.
                    if !crate::tj3::require_specified(inst, "tj3DecodeYUVPlanes8", false, true) {
                        return -1;
                    }
                    inst.set_error("tj3DecodeYUVPlanes8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
                    return -1;
                }
            };
            let w: usize = width as usize;
            let h: usize = height as usize;

            // Gather dense per-plane slices. GRAY reads `srcPlanes[0]` alone
            // (P4-165): upstream's packed wrapper passes NULL for slots 1 and
            // 2 (`turbojpeg.c:2732-2735`) and its NULL check exempts GRAY
            // (`:2575-2576`), so a caller may legally supply a one-element
            // array. One plane also selects `decode_yuv_planes`' grayscale
            // path, which writes R = G = B = Y.
            let plane_count: usize = current_plane_count(inst);
            let mut owned: Vec<Vec<u8>> = Vec::with_capacity(plane_count);
            for c in 0..plane_count {
                let pw: usize = yuv_plane_width(c, w, ss);
                let ph: usize = yuv_plane_height(c, h, ss);
                unsafe {
                    let plane_ptr: *const u8 = *src_planes.add(c);
                    if plane_ptr.is_null() {
                        inst.set_error("tj3DecodeYUVPlanes8: NULL plane", TJERR_FATAL);
                        return -1;
                    }
                    let stride: usize = if strides.is_null() {
                        pw
                    } else {
                        let s: c_int = *strides.add(c);
                        if s <= 0 {
                            pw
                        } else {
                            s as usize
                        }
                    };
                    let mut dense: Vec<u8> = Vec::with_capacity(pw * ph);
                    for row in 0..ph {
                        let row_ptr = plane_ptr.add(row * stride);
                        let row_slice: &[u8] = std::slice::from_raw_parts(row_ptr, pw);
                        dense.extend_from_slice(row_slice);
                    }
                    owned.push(dense);
                }
            }
            let plane_refs: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();
            let pixels: Vec<u8> = match decode_yuv_planes(&plane_refs, w, h, ss, pf) {
                Ok(v) => v,
                Err(e) => {
                    inst.set_error(format!("tj3DecodeYUVPlanes8: {e}"), TJERR_FATAL);
                    return -1;
                }
            };

            let bpp: usize = pf.bytes_per_pixel();
            let dst_stride: usize = if pitch == 0 {
                w * bpp
            } else if pitch < 0 {
                inst.set_error("tj3DecodeYUVPlanes8: negative pitch", TJERR_FATAL);
                return -1;
            } else {
                pitch as usize
            };
            if dst_stride < w * bpp {
                inst.set_error("tj3DecodeYUVPlanes8: pitch too small", TJERR_FATAL);
                return -1;
            }
            let bottom_up: bool = inst.bottom_up_flag();
            unsafe {
                for row in 0..h {
                    let src_row = pixels[row * w * bpp..row * w * bpp + w * bpp].as_ptr();
                    // Bottom-up: row `i` of the decoded image goes to
                    // `height - i - 1` in the caller's buffer, mirroring
                    // upstream's `bottomUp` write loop in
                    // `tj3DecodeYUVPlanes8`.
                    let dst_row_index: usize = if bottom_up { h - 1 - row } else { row };
                    let dst_row = dst_buf.add(dst_row_index * dst_stride);
                    std::ptr::copy_nonoverlapping(src_row, dst_row, w * bpp);
                }
            }
            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
}

// Keep `encode_yuv` / `tjpf_from_handle` referenced so the module
// remains easy to extend with packed-YUV encodes into the handle's
// ColorSpace without needing to re-add the imports.
#[allow(dead_code)]
const _YUV_HELPER_MARKER: fn() = || {
    let _: fn(&[u8], usize, usize, PixelFormat, Subsampling) -> _ = encode_yuv;
    let _: fn(&TjInstance) -> Option<PixelFormat> = tjpf_from_handle;
};
