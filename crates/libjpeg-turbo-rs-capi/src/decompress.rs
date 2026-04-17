//! `tj3Decompress8` — 8-bit JPEG decompression entry point.
//!
//! Signature (from `turbojpeg.h`):
//! ```c
//! int tj3Decompress8(tjhandle handle, const unsigned char *jpegBuf,
//!                    size_t jpegSize, unsigned char *dstBuf,
//!                    int pitch, int pixelFormat);
//! ```
//!
//! Unlike `tj3Compress8`, the destination buffer here is **owned by the
//! caller** — we write into it. Callers typically invoke
//! `tj3DecompressHeader` first to learn the image dimensions, then
//! allocate `width * height * bpp(pf)` bytes for `dstBuf`.
//!
//! `pitch == 0` means "tightly packed" (`width * bpp`).
//!
//! Behavior contract:
//! - Returns 0 on success, -1 on failure.
//! - On success the handle's `Width`/`Height`/`Precision`/`ColorSpace`/
//!   `Subsampling` get/set read values are updated to reflect the just-
//!   decoded image.

use std::ffi::{c_int, c_void};

use libjpeg_turbo_rs::PixelFormat;

use crate::convert::pixel_format_from_tj;
use crate::tj3::{handle_as_mut, TJERR_FATAL};

/// `tj3Decompress8(handle, jpegBuf, jpegSize, dstBuf, pitch, pixelFormat)
///   -> int`.
#[no_mangle]
pub extern "C" fn tj3Decompress8(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u8,
    pitch: c_int,
    pixel_format: c_int,
) -> c_int {
    let inst = match unsafe { handle_as_mut(handle) } {
        Some(i) => i,
        None => return -1,
    };

    if jpeg_buf.is_null() {
        inst.set_error("tj3Decompress8: jpegBuf is NULL", TJERR_FATAL);
        return -1;
    }
    if dst_buf.is_null() {
        inst.set_error("tj3Decompress8: dstBuf is NULL", TJERR_FATAL);
        return -1;
    }
    if jpeg_size < 2 {
        inst.set_error(
            format!("tj3Decompress8: jpegSize {jpeg_size} too small"),
            TJERR_FATAL,
        );
        return -1;
    }
    if pitch < 0 {
        inst.set_error(
            format!("tj3Decompress8: pitch must be non-negative (got {pitch})"),
            TJERR_FATAL,
        );
        return -1;
    }

    let pf: PixelFormat = match pixel_format_from_tj(pixel_format) {
        Some(p) => p,
        None => {
            inst.set_error(
                format!("tj3Decompress8: unsupported TJPF {pixel_format}"),
                TJERR_FATAL,
            );
            return -1;
        }
    };

    // SAFETY: caller asserts `jpeg_buf` is valid for `jpeg_size` bytes.
    let jpeg: &[u8] = unsafe { std::slice::from_raw_parts(jpeg_buf, jpeg_size) };

    // The Rust-side `decompress` returns an `Image` with a dense row-
    // major data buffer in the caller-requested pixel format. It also
    // updates the handle's `Width`/`Height`/etc. as `tj3Decompress*` must.
    //
    // We mirror the C contract by overriding the handle's ColorSpace so
    // the Rust decoder yields the requested `PixelFormat`.
    let requested_cs: i32 = match pf {
        PixelFormat::Grayscale => 2, // TJCS_GRAY
        PixelFormat::Cmyk => 3,
        _ => 0, // TJCS_RGB for all RGB/BGR/... variants
    };
    // Preserve the caller's existing ColorSpace setting to restore later.
    let saved_cs: i32 = inst.inner.get(libjpeg_turbo_rs::tj3::TjParam::ColorSpace);
    let _ = inst
        .inner
        .set(libjpeg_turbo_rs::tj3::TjParam::ColorSpace, requested_cs);

    let img = match inst.inner.decompress(jpeg) {
        Ok(i) => i,
        Err(e) => {
            // Restore color space before returning.
            let _ = inst
                .inner
                .set(libjpeg_turbo_rs::tj3::TjParam::ColorSpace, saved_cs);
            inst.set_error(format!("tj3Decompress8: {e}"), TJERR_FATAL);
            return -1;
        }
    };
    let _ = inst
        .inner
        .set(libjpeg_turbo_rs::tj3::TjParam::ColorSpace, saved_cs);

    // Reconcile the decoder's output pixel format with the caller's
    // request. The Rust `decompress()` selects based on ColorSpace, but
    // only returns a limited subset of formats. We repack into the
    // caller's format when the two don't already match.
    let out_format: PixelFormat = img.pixel_format;
    let dst_bpp: usize = pf.bytes_per_pixel();
    let w: usize = img.width;
    let h: usize = img.height;

    let effective_pitch: usize = if pitch == 0 {
        w * dst_bpp
    } else {
        pitch as usize
    };
    if effective_pitch < w * dst_bpp {
        inst.set_error(
            format!(
                "tj3Decompress8: pitch {effective_pitch} smaller than width*bpp ({})",
                w * dst_bpp
            ),
            TJERR_FATAL,
        );
        return -1;
    }

    // Pixel format adaptation. We implement the format pairs common in
    // real-world C clients (Pillow, ImageMagick): request any RGB/BGR
    // variant and receive RGB/Gray from the decoder. The repack loop
    // rearranges channels row-by-row directly into `dst_buf`.
    //
    // SAFETY: `dst_buf` is guaranteed by the caller to be at least
    // `effective_pitch * h` bytes; we never write beyond that.
    let dst_total: usize = effective_pitch.saturating_mul(h);
    let dst: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(dst_buf, dst_total) };

    if let Err(e) = repack_into_pitched(&img.data, out_format, w, h, pf, dst, effective_pitch) {
        inst.set_error(format!("tj3Decompress8: {e}"), TJERR_FATAL);
        return -1;
    }

    inst.clear_error();
    0
}

/// Repack an `Image` from `src_fmt` into the caller's `dst_fmt` with the
/// specified destination pitch. Only the channel permutations our decoder
/// actually emits plus Gray/CMYK passthrough are supported; everything
/// else returns a descriptive error so upstream tooling can surface the
/// gap instead of writing garbage bytes.
fn repack_into_pitched(
    src: &[u8],
    src_fmt: PixelFormat,
    w: usize,
    h: usize,
    dst_fmt: PixelFormat,
    dst: &mut [u8],
    dst_pitch: usize,
) -> Result<(), String> {
    let src_bpp: usize = src_fmt.bytes_per_pixel();
    let dst_bpp: usize = dst_fmt.bytes_per_pixel();
    let row_bytes: usize = w * dst_bpp;

    // Fast path: identical formats and dense pitch.
    if src_fmt == dst_fmt && dst_pitch == row_bytes && src.len() == w * src_bpp * h {
        dst[..src.len()].copy_from_slice(src);
        return Ok(());
    }

    // Grayscale-out with grayscale source: per-row memcpy.
    if src_fmt == PixelFormat::Grayscale && dst_fmt == PixelFormat::Grayscale {
        for row in 0..h {
            let s: &[u8] = &src[row * w..(row + 1) * w];
            let d: &mut [u8] = &mut dst[row * dst_pitch..row * dst_pitch + w];
            d.copy_from_slice(s);
        }
        return Ok(());
    }

    // CMYK passthrough.
    if src_fmt == PixelFormat::Cmyk && dst_fmt == PixelFormat::Cmyk {
        for row in 0..h {
            let s: &[u8] = &src[row * w * 4..(row + 1) * w * 4];
            let d: &mut [u8] = &mut dst[row * dst_pitch..row * dst_pitch + w * 4];
            d.copy_from_slice(s);
        }
        return Ok(());
    }

    // Extract RGB from the source pixel using its channel offsets, then
    // write to the destination in the requested order. This handles every
    // RGB/BGR/alpha/padding permutation in `PixelFormat`.
    let (src_r, src_g, src_b) = match (
        src_fmt.red_offset(),
        src_fmt.green_offset(),
        src_fmt.blue_offset(),
    ) {
        (Some(r), Some(g), Some(b)) => (r, g, b),
        _ => {
            return Err(format!(
                "unsupported source pixel format {src_fmt:?} for repack"
            ));
        }
    };

    let (dst_r, dst_g, dst_b) = match (
        dst_fmt.red_offset(),
        dst_fmt.green_offset(),
        dst_fmt.blue_offset(),
    ) {
        (Some(r), Some(g), Some(b)) => (r, g, b),
        _ => {
            return Err(format!(
                "unsupported destination pixel format {dst_fmt:?} for repack"
            ));
        }
    };

    for row in 0..h {
        let src_row_start: usize = row * w * src_bpp;
        let dst_row_start: usize = row * dst_pitch;
        for x in 0..w {
            let sp: usize = src_row_start + x * src_bpp;
            let dp: usize = dst_row_start + x * dst_bpp;
            // Zero any padding/alpha byte in the destination.
            for k in 0..dst_bpp {
                dst[dp + k] = 0xFF; // alpha defaults to fully opaque
            }
            dst[dp + dst_r] = src[sp + src_r];
            dst[dp + dst_g] = src[sp + src_g];
            dst[dp + dst_b] = src[sp + src_b];
        }
    }

    Ok(())
}
