//! TJ3 YUV family (8 entry points) — A1-7.
//!
//! All YUV entry points use libjpeg-turbo's canonical plane-order:
//! Y first, then Cb, then Cr. Packed buffers concatenate the three
//! planes contiguously with per-plane alignment controlled by the
//! `align` parameter (power of two, 1 = no padding).
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

use libjpeg_turbo_rs::api::yuv::{
    compress_from_yuv, compress_from_yuv_planes, decode_yuv, decode_yuv_planes,
    decompress_to_yuv_planes, encode_yuv, encode_yuv_planes,
};
use libjpeg_turbo_rs::{yuv_plane_height, yuv_plane_width, PixelFormat, Subsampling};

use crate::alloc::{libc_free, libc_from_slice};
use crate::convert::pixel_format_from_tj;
use crate::tj3::{handle_as_mut, TjInstance, TJERR_FATAL};

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

fn current_subsampling(inst: &TjInstance) -> Option<Subsampling> {
    subsamp_from_tj(inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::Subsampling))
}

fn tjpf_from_handle(inst: &TjInstance) -> Option<PixelFormat> {
    pixel_format_from_tj(inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::ColorSpace))
}

/// Densify a pitched RGB source into a contiguous `width * bpp * height`
/// buffer, validating `pitch` along the way.
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
    let total: usize = line * height;
    // SAFETY: caller guarantees `src` is valid for `total` bytes.
    let raw: &[u8] = unsafe { std::slice::from_raw_parts(src, total) };
    if line == width * bpp {
        return Some(raw.to_vec());
    }
    let mut out: Vec<u8> = Vec::with_capacity(width * bpp * height);
    for row in 0..height {
        let start: usize = row * line;
        out.extend_from_slice(&raw[start..start + width * bpp]);
    }
    Some(out)
}

/// Split the packed YUV buffer (Y|Cb|Cr) into `(plane_ptr, plane_len)`
/// slices given the subsampling and `align` row-stride. Returns `None`
/// if `align` is non-positive or `align` is not a power of two.
fn split_packed_yuv(
    yuv: &[u8],
    width: usize,
    height: usize,
    align: c_int,
    subsampling: Subsampling,
) -> Option<Vec<Vec<u8>>> {
    if align <= 0 || (align as usize).count_ones() != 1 {
        return None;
    }
    let align_us: usize = align as usize;

    let mut planes: Vec<Vec<u8>> = Vec::with_capacity(3);
    let mut offset: usize = 0;
    // All subsampling modes produce 3 planes (Y/Cb/Cr); grayscale JPEG
    // paths use a separate single-plane code path in
    // `decompress_to_yuv_planes`, so we always split 3 here.
    let planes_count: usize = 3;
    for component in 0..planes_count {
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

#[no_mangle]
pub extern "C" fn tj3EncodeYUV8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    dst_buf: *mut u8,
    align: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if src_buf.is_null() || dst_buf.is_null() || width <= 0 || height <= 0 {
        inst.set_error("tj3EncodeYUV8: invalid pointer/dim", TJERR_FATAL);
        return -1;
    }
    let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
        Some(p) => p,
        None => {
            inst.set_error("tj3EncodeYUV8: unsupported TJPF", TJERR_FATAL);
            return -1;
        }
    };
    let ss: Subsampling = match current_subsampling(inst) {
        Some(s) => s,
        None => {
            inst.set_error("tj3EncodeYUV8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
            return -1;
        }
    };

    let w: usize = width as usize;
    let h: usize = height as usize;
    let bpp: usize = pf.bytes_per_pixel();
    let dense: Vec<u8> = match densify_pitched_bytes(src_buf, w, h, bpp, pitch) {
        Some(v) => v,
        None => {
            inst.set_error("tj3EncodeYUV8: bad pitch", TJERR_FATAL);
            return -1;
        }
    };

    let planes: Vec<Vec<u8>> = match encode_yuv_planes(&dense, w, h, pf, ss) {
        Ok(p) => p,
        Err(e) => {
            inst.set_error(format!("tj3EncodeYUV8: {e}"), TJERR_FATAL);
            return -1;
        }
    };
    let packed: Vec<u8> = match pack_yuv_planes(&planes, w, h, align, ss) {
        Some(p) => p,
        None => {
            inst.set_error(
                "tj3EncodeYUV8: align must be a positive power of 2",
                TJERR_FATAL,
            );
            return -1;
        }
    };

    // SAFETY: caller guarantees `dst_buf` is sized large enough per
    // `tjBufSizeYUV2(width, align, height, subsamp)`.
    unsafe {
        std::ptr::copy_nonoverlapping(packed.as_ptr(), dst_buf, packed.len());
    }
    inst.clear_error();
    0
}

#[no_mangle]
pub extern "C" fn tj3EncodeYUVPlanes8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
    dst_planes: *mut *mut u8,
    strides: *const c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
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
            inst.set_error("tj3EncodeYUVPlanes8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
            return -1;
        }
    };
    let w: usize = width as usize;
    let h: usize = height as usize;
    let bpp: usize = pf.bytes_per_pixel();
    let dense: Vec<u8> = match densify_pitched_bytes(src_buf, w, h, bpp, pitch) {
        Some(v) => v,
        None => {
            inst.set_error("tj3EncodeYUVPlanes8: bad pitch", TJERR_FATAL);
            return -1;
        }
    };
    let planes: Vec<Vec<u8>> = match encode_yuv_planes(&dense, w, h, pf, ss) {
        Ok(p) => p,
        Err(e) => {
            inst.set_error(format!("tj3EncodeYUVPlanes8: {e}"), TJERR_FATAL);
            return -1;
        }
    };

    // SAFETY: caller guarantees `dst_planes[i]` / `strides[i]` are valid
    // for `planes.len()` entries and each buffer is sized to hold
    // `stride * plane_height` samples.
    unsafe {
        for (i, plane) in planes.iter().enumerate() {
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
}

// ---------------------------------------------------------------------------
// Pixels → YUV → JPEG (compress path)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn tj3CompressFromYUV8(
    handle: *mut c_void,
    src_buf: *const u8,
    width: c_int,
    align: c_int,
    height: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if src_buf.is_null() || jpeg_buf.is_null() || jpeg_size.is_null() || width <= 0 || height <= 0 {
        inst.set_error("tj3CompressFromYUV8: NULL / bad dim", TJERR_FATAL);
        return -1;
    }
    let ss: Subsampling = match current_subsampling(inst) {
        Some(s) => s,
        None => {
            inst.set_error("tj3CompressFromYUV8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
            return -1;
        }
    };
    let quality: i32 = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::Quality);

    // Compute the raw packed YUV length from align+dims.
    let mut total: usize = 0;
    for c in 0..3 {
        let pw: usize = yuv_plane_width(c, width as usize, ss);
        let ph: usize = yuv_plane_height(c, height as usize, ss);
        let stride: usize = pw.div_ceil(align.max(1) as usize) * align.max(1) as usize;
        total = total.saturating_add(stride * ph);
    }
    // SAFETY: caller asserted `src_buf` valid for `total` bytes per
    // `tjBufSizeYUV2(width, align, height, subsamp)`.
    let packed: &[u8] = unsafe { std::slice::from_raw_parts(src_buf, total) };

    // Unpack into (Y, Cb, Cr) dense planes, then re-pack as Y|Cb|Cr
    // with align=1 for the Rust-side `compress_from_yuv`.
    let planes: Vec<Vec<u8>> =
        match split_packed_yuv(packed, width as usize, height as usize, align, ss) {
            Some(p) => p,
            None => {
                inst.set_error("tj3CompressFromYUV8: bad packed YUV layout", TJERR_FATAL);
                return -1;
            }
        };
    let mut dense: Vec<u8> = Vec::new();
    for p in &planes {
        dense.extend_from_slice(p);
    }

    let jpeg: Vec<u8> =
        match compress_from_yuv(&dense, width as usize, height as usize, ss, quality as u8) {
            Ok(v) => v,
            Err(e) => {
                inst.set_error(format!("tj3CompressFromYUV8: {e}"), TJERR_FATAL);
                return -1;
            }
        };

    let ptr: *mut u8 = libc_from_slice(&jpeg);
    if ptr.is_null() && !jpeg.is_empty() {
        inst.set_error("tj3CompressFromYUV8: OOM", TJERR_FATAL);
        return -1;
    }
    // SAFETY: jpeg_buf / jpeg_size validated non-NULL above.
    unsafe {
        let prior: *mut u8 = *jpeg_buf;
        if !prior.is_null() {
            libc_free(prior);
        }
        *jpeg_buf = ptr;
        *jpeg_size = jpeg.len();
    }
    inst.clear_error();
    0
}

#[no_mangle]
pub extern "C" fn tj3CompressFromYUVPlanes8(
    handle: *mut c_void,
    src_planes: *const *const u8,
    width: c_int,
    strides: *const c_int,
    height: c_int,
    jpeg_buf: *mut *mut u8,
    jpeg_size: *mut usize,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if src_planes.is_null()
        || jpeg_buf.is_null()
        || jpeg_size.is_null()
        || width <= 0
        || height <= 0
    {
        inst.set_error("tj3CompressFromYUVPlanes8: NULL / bad dim", TJERR_FATAL);
        return -1;
    }
    let ss: Subsampling = match current_subsampling(inst) {
        Some(s) => s,
        None => {
            inst.set_error(
                "tj3CompressFromYUVPlanes8: bad TJPARAM_SUBSAMP",
                TJERR_FATAL,
            );
            return -1;
        }
    };
    let quality: i32 = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::Quality);

    // Collect dense per-plane slices, respecting caller strides.
    // SAFETY: caller guarantees 3 plane pointers + 3 strides valid.
    let mut owned_planes: Vec<Vec<u8>> = Vec::with_capacity(3);
    for c in 0..3usize {
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
    let ptr: *mut u8 = libc_from_slice(&jpeg);
    if ptr.is_null() && !jpeg.is_empty() {
        inst.set_error("tj3CompressFromYUVPlanes8: OOM", TJERR_FATAL);
        return -1;
    }
    unsafe {
        let prior: *mut u8 = *jpeg_buf;
        if !prior.is_null() {
            libc_free(prior);
        }
        *jpeg_buf = ptr;
        *jpeg_size = jpeg.len();
    }
    inst.clear_error();
    0
}

// ---------------------------------------------------------------------------
// JPEG → YUV (decompress path, no color conversion)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn tj3DecompressToYUV8(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u8,
    align: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if jpeg_buf.is_null() || dst_buf.is_null() || jpeg_size < 2 {
        inst.set_error("tj3DecompressToYUV8: NULL / size", TJERR_FATAL);
        return -1;
    }
    let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };
    let (planes, w, h, ss) = match decompress_to_yuv_planes(jpeg) {
        Ok(v) => v,
        Err(e) => {
            inst.set_error(format!("tj3DecompressToYUV8: {e}"), TJERR_FATAL);
            return -1;
        }
    };
    let packed: Vec<u8> = match pack_yuv_planes(&planes, w, h, align, ss) {
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
}

#[no_mangle]
pub extern "C" fn tj3DecompressToYUVPlanes8(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_planes: *mut *mut u8,
    strides: *const c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if jpeg_buf.is_null() || dst_planes.is_null() || jpeg_size < 2 {
        inst.set_error("tj3DecompressToYUVPlanes8: NULL / size", TJERR_FATAL);
        return -1;
    }
    let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };
    let (planes, w, h, ss) = match decompress_to_yuv_planes(jpeg) {
        Ok(v) => v,
        Err(e) => {
            inst.set_error(format!("tj3DecompressToYUVPlanes8: {e}"), TJERR_FATAL);
            return -1;
        }
    };
    unsafe {
        for (i, plane) in planes.iter().enumerate() {
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
}

// ---------------------------------------------------------------------------
// YUV → Pixels (color conversion only, no JPEG)
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn tj3DecodeYUV8(
    handle: *mut c_void,
    src_buf: *const u8,
    align: c_int,
    dst_buf: *mut u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
    if src_buf.is_null() || dst_buf.is_null() || width <= 0 || height <= 0 {
        inst.set_error("tj3DecodeYUV8: NULL / bad dim", TJERR_FATAL);
        return -1;
    }
    let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
        Some(p) => p,
        None => {
            inst.set_error("tj3DecodeYUV8: unsupported TJPF", TJERR_FATAL);
            return -1;
        }
    };
    let ss: Subsampling = match current_subsampling(inst) {
        Some(s) => s,
        None => {
            inst.set_error("tj3DecodeYUV8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
            return -1;
        }
    };

    let w: usize = width as usize;
    let h: usize = height as usize;
    // Compute packed length.
    let mut total: usize = 0;
    for c in 0..3 {
        let pw: usize = yuv_plane_width(c, w, ss);
        let ph: usize = yuv_plane_height(c, h, ss);
        let stride: usize = pw.div_ceil(align.max(1) as usize) * align.max(1) as usize;
        total = total.saturating_add(stride * ph);
    }
    let packed: &[u8] = unsafe { std::slice::from_raw_parts(src_buf, total) };
    let planes: Vec<Vec<u8>> = match split_packed_yuv(packed, w, h, align, ss) {
        Some(p) => p,
        None => {
            inst.set_error("tj3DecodeYUV8: bad packed YUV layout", TJERR_FATAL);
            return -1;
        }
    };
    let dense: Vec<u8> = {
        let mut v: Vec<u8> = Vec::new();
        for p in &planes {
            v.extend_from_slice(p);
        }
        v
    };
    let pixels: Vec<u8> = match decode_yuv(&dense, w, h, ss, pf) {
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
    // SAFETY: caller guarantees dst is `dst_stride * h` bytes.
    unsafe {
        for row in 0..h {
            let src_row = pixels[row * w * bpp..row * w * bpp + w * bpp].as_ptr();
            let dst_row = dst_buf.add(row * dst_stride);
            std::ptr::copy_nonoverlapping(src_row, dst_row, w * bpp);
        }
    }
    inst.clear_error();
    0
}

#[no_mangle]
pub extern "C" fn tj3DecodeYUVPlanes8(
    handle: *mut c_void,
    src_planes: *const *const u8,
    strides: *const c_int,
    dst_buf: *mut u8,
    width: c_int,
    pitch: c_int,
    height: c_int,
    pixel_format: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };
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
            inst.set_error("tj3DecodeYUVPlanes8: bad TJPARAM_SUBSAMP", TJERR_FATAL);
            return -1;
        }
    };
    let w: usize = width as usize;
    let h: usize = height as usize;

    // Gather dense per-plane slices.
    let mut owned: Vec<Vec<u8>> = Vec::with_capacity(3);
    for c in 0..3usize {
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
    unsafe {
        for row in 0..h {
            let src_row = pixels[row * w * bpp..row * w * bpp + w * bpp].as_ptr();
            let dst_row = dst_buf.add(row * dst_stride);
            std::ptr::copy_nonoverlapping(src_row, dst_row, w * bpp);
        }
    }
    inst.clear_error();
    0
}

// Keep `encode_yuv` / `tjpf_from_handle` referenced so the module
// remains easy to extend with packed-YUV encodes into the handle's
// ColorSpace without needing to re-add the imports.
#[allow(dead_code)]
const _YUV_HELPER_MARKER: fn() = || {
    let _: fn(&[u8], usize, usize, PixelFormat, Subsampling) -> _ = encode_yuv;
    let _: fn(&TjInstance) -> Option<PixelFormat> = tjpf_from_handle;
};
