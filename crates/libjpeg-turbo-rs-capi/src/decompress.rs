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
use crate::tj3::{with_handle, TJERR_FATAL};

/// `tj3Decompress8(handle, jpegBuf, jpegSize, dstBuf, pitch, pixelFormat)
///   -> int`.
///
/// # Safety
///
/// C ABI entry point. `handle`, `jpeg_buf`, `dst_buf` must satisfy the crate-level
/// [pointer contract](crate#pointer-contract): valid for the whole call,
/// correctly aligned, large enough for the accesses described above, and
/// not aliased by another live reference. A pointer this function documents as
/// optional may be null; any other null is reported through the documented
/// error value rather than dereferenced.
#[no_mangle]
pub unsafe extern "C" fn tj3Decompress8(
    handle: *mut c_void,
    jpeg_buf: *const u8,
    jpeg_size: usize,
    dst_buf: *mut u8,
    pitch: c_int,
    pixel_format: c_int,
) -> c_int {
    crate::unwind_guard!(-1, {
        // Defined outside the `unsafe` block below so the body's own `unsafe`
        // blocks stay meaningful rather than nesting inside a blanket one.
        let body = |inst: &mut crate::tj3::TjInstance| -> c_int {
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
            // `saturating_mul` yielded usize::MAX on overflow -- precisely the
            // wrong value here, since it becomes the length of a raw slice and
            // claims the entire address space (P4-139).
            let dst_total: usize = match effective_pitch.checked_mul(h) {
                Some(v) => v,
                None => {
                    inst.set_error("tj3Decompress8: pitch * height overflows", TJERR_FATAL);
                    return -1;
                }
            };
            let dst: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(dst_buf, dst_total) };

            if let Err(e) =
                repack_into_pitched(&img.data, out_format, w, h, pf, dst, effective_pitch)
            {
                inst.set_error(format!("tj3Decompress8: {e}"), TJERR_FATAL);
                return -1;
            }

            inst.clear_error();
            0
        };

        // SAFETY: `with_handle` NULL-checks; the caller owns handle validity
        // and exclusivity per its contract.
        unsafe { with_handle(handle, body) }.unwrap_or(-1)
    })
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

    // Grayscale source -> RGB-family destination: replicate the gray sample
    // into R, G, and B channels. Matches libjpeg-turbo's automatic
    // grayscale-to-RGB expansion (`jdcolor.c` `gray_rgb_convert`) and is the
    // path tjunittest exercises when decoding a gray JPEG into TJPF_RGB et al.
    if src_fmt == PixelFormat::Grayscale {
        if let (Some(dst_r), Some(dst_g), Some(dst_b)) = (
            dst_fmt.red_offset(),
            dst_fmt.green_offset(),
            dst_fmt.blue_offset(),
        ) {
            for row in 0..h {
                let s_row: &[u8] = &src[row * w..(row + 1) * w];
                let d_row_start: usize = row * dst_pitch;
                for (x, &g_val) in s_row.iter().enumerate() {
                    let dp: usize = d_row_start + x * dst_bpp;
                    // Initialize all destination bytes to 0xFF so any X/A
                    // padding byte defaults to opaque, matching the
                    // RGB->RGB-family path below.
                    for k in 0..dst_bpp {
                        dst[dp + k] = 0xFF;
                    }
                    dst[dp + dst_r] = g_val;
                    dst[dp + dst_g] = g_val;
                    dst[dp + dst_b] = g_val;
                }
            }
            return Ok(());
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_src(samples: &[u8]) -> Vec<u8> {
        samples.to_vec()
    }

    #[test]
    fn gray_to_rgb_replicates_luma_into_three_channels() {
        // 4x1 gray ramp; each pixel must land in R,G,B.
        let src: Vec<u8> = gray_src(&[10, 80, 150, 220]);
        let mut dst: Vec<u8> = vec![0; 4 * 3];
        repack_into_pitched(
            &src,
            PixelFormat::Grayscale,
            4,
            1,
            PixelFormat::Rgb,
            &mut dst,
            12,
        )
        .expect("gray->rgb must succeed");
        // Each gray sample replicated into all three RGB bytes.
        assert_eq!(
            dst,
            vec![10, 10, 10, 80, 80, 80, 150, 150, 150, 220, 220, 220]
        );
    }

    #[test]
    fn gray_to_bgr_replicates_luma_into_three_channels() {
        // BGR has the same layout for monochrome input — order doesn't matter
        // when R==G==B. Confirms the branch fires for both color orderings.
        let src: Vec<u8> = gray_src(&[5, 250]);
        let mut dst: Vec<u8> = vec![0; 2 * 3];
        repack_into_pitched(
            &src,
            PixelFormat::Grayscale,
            2,
            1,
            PixelFormat::Bgr,
            &mut dst,
            6,
        )
        .expect("gray->bgr must succeed");
        assert_eq!(dst, vec![5, 5, 5, 250, 250, 250]);
    }

    #[test]
    fn gray_to_rgba_sets_alpha_opaque() {
        // 4-bpp destinations get 0xFF in the unused (X/A) lane to match the
        // RGB->RGB-family path's "alpha defaults to fully opaque" convention.
        let src: Vec<u8> = gray_src(&[42]);
        let mut dst: Vec<u8> = vec![0; 4];
        repack_into_pitched(
            &src,
            PixelFormat::Grayscale,
            1,
            1,
            PixelFormat::Rgba,
            &mut dst,
            4,
        )
        .expect("gray->rgba must succeed");
        // R, G, B = 42, A = 0xFF.
        assert_eq!(dst, vec![42, 42, 42, 0xFF]);
    }

    #[test]
    fn gray_to_xrgb_sets_padding_byte_opaque() {
        // X is the leading byte for XRGB; verify our 0xFF init covers it.
        let src: Vec<u8> = gray_src(&[99]);
        let mut dst: Vec<u8> = vec![0; 4];
        repack_into_pitched(
            &src,
            PixelFormat::Grayscale,
            1,
            1,
            PixelFormat::Xrgb,
            &mut dst,
            4,
        )
        .expect("gray->xrgb must succeed");
        // X = 0xFF, then R, G, B = 99.
        assert_eq!(dst, vec![0xFF, 99, 99, 99]);
    }

    #[test]
    fn gray_to_abgr_handles_alpha_first_layout() {
        let src: Vec<u8> = gray_src(&[7, 200]);
        let mut dst: Vec<u8> = vec![0; 2 * 4];
        repack_into_pitched(
            &src,
            PixelFormat::Grayscale,
            2,
            1,
            PixelFormat::Abgr,
            &mut dst,
            8,
        )
        .expect("gray->abgr must succeed");
        // ABGR layout: A, B, G, R.
        assert_eq!(dst, vec![0xFF, 7, 7, 7, 0xFF, 200, 200, 200]);
    }

    #[test]
    fn gray_to_rgb_with_padded_pitch_writes_only_within_row_bytes() {
        // Pitch larger than w*bpp: the trailing pad bytes must remain whatever
        // was there before (we never touch them). Initialize dst with a known
        // sentinel and verify only the active row region got rewritten.
        // 2x2 gray source: row0=[1,2], row1=[3,4].
        let src: Vec<u8> = gray_src(&[1, 2, 3, 4]);
        let mut dst: Vec<u8> = vec![0xAA; 2 * 8]; // pitch=8, row only needs 6
        repack_into_pitched(
            &src,
            PixelFormat::Grayscale,
            2,
            2,
            PixelFormat::Rgb,
            &mut dst,
            8,
        )
        .expect("gray->rgb with padded pitch must succeed");
        // First row: pixels 1,2 -> offsets 0..6, padding bytes 6..8 stay 0xAA.
        // Second row: pixels 3,4 -> offsets 8..14, padding 14..16 stay 0xAA.
        assert_eq!(&dst[0..6], &[1, 1, 1, 2, 2, 2]);
        assert_eq!(&dst[6..8], &[0xAA, 0xAA]);
        assert_eq!(&dst[8..14], &[3, 3, 3, 4, 4, 4]);
        assert_eq!(&dst[14..16], &[0xAA, 0xAA]);
    }
}
