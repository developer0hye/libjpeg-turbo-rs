//! Buffer size calculation helpers matching libjpeg-turbo's TurboJPEG API.
//!
//! These functions compute worst-case buffer sizes for JPEG and YUV data,
//! following the exact formulas from `tj3JPEGBufSize`, `tj3YUVBufSize`,
//! `tj3YUVPlaneSize`, `tj3YUVPlaneWidth`, and `tj3YUVPlaneHeight`.

//! **Unrepresentable geometry reports 0, never a wrapped or saturated size.**
//! Every public function here answers with a plain `usize` (or a pair of
//! them), a signature that predates the crate's typed span errors, so 0 is the
//! refusal. It is the same answer the C entry points give (`tj3JPEGBufSize`
//! returns 0 and records "Image is too large"; `tj3YUVPlaneWidth` returns 0
//! for an argument it rejects), and one a caller cannot mistake for a usable
//! capacity — unlike `usize::MAX`, which is the worst value a size can take
//! and which `tests/sizing_arithmetic_gate.rs` bars repo-wide (P4-139).
//!
//! The one thing that is still not a value here is
//! [`calc_output_dimensions`]' `assert!(scale_denom != 0)`, a panic on public
//! input. It is P4-139 criterion 4's, not this module's: the fix is making
//! `ScalingFactor`'s fields private behind a fallible constructor, which is a
//! breaking change sequenced for 0.9.0.

use super::types::Subsampling;
use crate::common::layout::checked_span;
use crate::transform::TransformOp;

/// Rounds `value` up to the nearest multiple of `alignment`, or `None` when
/// the rounded value is not representable.
///
/// `alignment` must be a power of two. The `None` arm is what makes the sizes
/// above it honest: `value + alignment - 1` wraps for a `value` within
/// `alignment` of `usize::MAX`, and a wrapped padding yields a buffer size
/// *smaller* than the unpadded geometry — the one direction a buffer size may
/// never be wrong in.
#[inline]
fn checked_pad(value: usize, alignment: usize) -> Option<usize> {
    debug_assert!(
        alignment.is_power_of_two(),
        "alignment must be a power of two"
    );
    Some(value.checked_add(alignment - 1)? & !(alignment - 1))
}

/// Worst-case JPEG output buffer size for given dimensions and subsampling.
///
/// Matches `tj3JPEGBufSize()` from libjpeg-turbo. The result accounts for rare
/// corner cases where compressed JPEG data can exceed uncompressed pixel data.
///
/// Returns 0 for geometry whose size is not representable — see the module
/// note.
pub fn jpeg_buf_size(width: usize, height: usize, subsampling: Subsampling) -> usize {
    let mcu_width: usize = subsampling.mcu_width_blocks() * 8;
    let mcu_height: usize = subsampling.mcu_height_blocks() * 8;

    // Chroma scaling factor: ratio of chroma samples to total MCU pixels.
    // For color images: 4 * 64 / (mcu_width * mcu_height).
    let chroma_scale_factor: usize = 4 * 64 / (mcu_width * mcu_height);

    let (Some(padded_width), Some(padded_height)) = (
        checked_pad(width, mcu_width),
        checked_pad(height, mcu_height),
    ) else {
        return 0;
    };
    checked_span(
        &[padded_width, padded_height, 2 + chroma_scale_factor],
        "JPEG output buffer",
    )
    .ok()
    // The `+ 2048` header allowance is inside the ceiling, not outside it:
    // `checked_span` alone would let a span of exactly `isize::MAX` come back
    // as `isize::MAX + 2048`, a capacity no allocation can satisfy.
    .and_then(|span| span.checked_add(2048))
    .filter(|total| *total <= isize::MAX as usize)
    .unwrap_or(0)
}

/// Number of planes the YUV model describes for a colour subsampling.
///
/// Grayscale is deliberately absent: [`Subsampling`] has no variant for it, so
/// a helper taking one cannot apply C's `nc = (subsamp == TJSAMP_GRAY ? 1 : 3)`.
/// The grayscale half of that bound lives in the C-ABI layer, which still has
/// the raw `subsamp` (P4-126).
const YUV_PLANE_COUNT: usize = 3;

/// Width (in pixels/samples) of a single YUV plane.
///
/// Matches `tj3YUVPlaneWidth()`. Component 0 is luma (Y); components 1 and 2
/// are chroma (Cb, Cr). The width is padded to the subsampling factor boundary.
///
/// Returns 0 for a component index the model has no plane for, which is how
/// `tj3YUVPlaneWidth` reports an invalid argument
/// (`references/libjpeg-turbo/src/turbojpeg.c:1123-1125`). A 4-component
/// CMYK/YCCK frame's fourth component is **not** a chroma plane, so callers
/// must not size it through here — see `decompress_to_yuv_planes`, which sizes
/// it at full resolution.
pub fn yuv_plane_width(component: usize, width: usize, subsampling: Subsampling) -> usize {
    if component >= YUV_PLANE_COUNT {
        return 0;
    }
    let h_factor: usize = subsampling.mcu_width_blocks(); // mcuw / 8
    let Some(padded_width) = checked_pad(width, h_factor) else {
        return 0;
    };

    if component == 0 {
        padded_width
    } else {
        // Chroma plane: divide by the horizontal subsampling ratio. Written
        // `padded_width * 8 / (h_factor * 8)` until P4-139 chunk 2 — an
        // algebraic no-op whose only effect was that the `* 8` could wrap, and
        // wrap *downward*: at 4:2:0 a width of 2^62 + 2 reported a chroma
        // width of 1 instead of 2^60 + 1.
        padded_width / h_factor
    }
}

/// Height (in pixels/samples) of a single YUV plane.
///
/// Matches `tj3YUVPlaneHeight()`. Component 0 is luma (Y); components 1 and 2
/// are chroma (Cb, Cr). The height is padded to the subsampling factor boundary.
///
/// Returns 0 for an out-of-range component index — see [`yuv_plane_width`].
pub fn yuv_plane_height(component: usize, height: usize, subsampling: Subsampling) -> usize {
    if component >= YUV_PLANE_COUNT {
        return 0;
    }
    let v_factor: usize = subsampling.mcu_height_blocks(); // mcuh / 8
    let Some(padded_height) = checked_pad(height, v_factor) else {
        return 0;
    };

    if component == 0 {
        padded_height
    } else {
        // Chroma plane: divide by the vertical subsampling ratio — see
        // [`yuv_plane_width`] for why the `* 8 / (v_factor * 8)` form went.
        padded_height / v_factor
    }
}

/// Buffer size for a single YUV plane (stride = plane width, no padding).
///
/// Matches `tj3YUVPlaneSize()` with `stride = 0`. The formula is
/// `stride * (height - 1) + width`, which for stride == width simplifies to
/// `width * height`.
///
/// Returns 0 for geometry whose size is not representable — see the module
/// note. This is the checked twin of the C-ABI crate's `tj3YUVPlaneSize`,
/// which reaches the same conclusion through `set_no_handle_error` and a 0
/// return; before P4-139 chunk 2 the two disagreed, and this one wrapped.
pub fn yuv_plane_size(
    component: usize,
    width: usize,
    height: usize,
    subsampling: Subsampling,
) -> usize {
    let plane_width: usize = yuv_plane_width(component, width, subsampling);
    let plane_height: usize = yuv_plane_height(component, height, subsampling);

    // stride * (ph - 1) + pw, where stride == pw
    checked_span(&[plane_width, plane_height], "YUV plane").unwrap_or(0)
}

/// Total packed YUV buffer size (3 components, no row padding).
///
/// Matches `tj3YUVBufSize()` with `align = 1`. Returns the sum of all three
/// plane sizes (Y + Cb + Cr).
///
/// Returns 0 for geometry whose size is not representable — see the module
/// note. Both halves need care: the sum itself is checked, *and* a plane that
/// refused (reporting 0) must not simply contribute 0 to it, since that would
/// under-report the total instead of refusing it.
pub fn yuv_buf_size(width: usize, height: usize, subsampling: Subsampling) -> usize {
    if width == 0 || height == 0 {
        return 0;
    }
    let mut total: usize = 0;
    for component in 0..YUV_PLANE_COUNT {
        let plane: usize = yuv_plane_size(component, width, height, subsampling);
        // A nonzero width and height always pad up to a nonzero plane, so a
        // zero here is `yuv_plane_size` refusing, not an empty plane. Summing
        // it anyway would under-report the total — the one direction a buffer
        // size may never be wrong in.
        if plane == 0 {
            return 0;
        }
        total = match total
            .checked_add(plane)
            .filter(|sum| *sum <= isize::MAX as usize)
        {
            Some(sum) => sum,
            None => return 0,
        };
    }
    total
}

/// Worst-case transform output buffer size, accounting for dimension swaps.
///
/// Matches `tj3TransformBufSize()`. For transforms that swap width and height
/// (Transpose, Transverse, Rot90, Rot270), dimensions are swapped before
/// computing the buffer size via `jpeg_buf_size`.
///
/// Returns 0 for geometry whose size is not representable, inherited from
/// [`jpeg_buf_size`] — see the module note.
pub fn transform_buf_size(
    width: usize,
    height: usize,
    subsampling: Subsampling,
    op: TransformOp,
) -> usize {
    let swaps_dimensions: bool = matches!(
        op,
        TransformOp::Transpose | TransformOp::Transverse | TransformOp::Rot90 | TransformOp::Rot270
    );

    if swaps_dimensions {
        jpeg_buf_size(height, width, subsampling)
    } else {
        jpeg_buf_size(width, height, subsampling)
    }
}

/// Compute scaled output dimensions for decompression.
///
/// Matches `jpeg_calc_output_dimensions()`. Applies the scaling factor
/// `scale_num / scale_denom` to both width and height, rounding up.
///
/// An axis whose scaled value is not representable reports 0 — see the module
/// note. The `scale_denom == 0` **panic** is the exception the note names: it
/// predates this work and belongs to P4-139 criterion 4, which makes
/// `ScalingFactor`'s fields private behind a fallible constructor in 0.9.0.
pub fn calc_output_dimensions(
    width: usize,
    height: usize,
    scale_num: u32,
    scale_denom: u32,
) -> (usize, usize) {
    assert!(scale_denom != 0, "scale denominator must not be zero");
    let scale = |dim: usize| -> usize {
        dim.checked_mul(scale_num as usize)
            .map_or(0, |scaled| scaled.div_ceil(scale_denom as usize))
    };
    (scale(width), scale(height))
}

/// Compute MCU-padded JPEG dimensions for encoding.
///
/// Matches `jpeg_calc_jpeg_dimensions()`. Rounds width and height up to the
/// nearest MCU boundary determined by the subsampling mode.
///
/// An axis that cannot be padded without wrapping reports 0 for that axis —
/// see the module note. A zero dimension is refused by every encode path, so
/// the refusal is visible rather than silently smaller than the input.
pub fn calc_jpeg_dimensions(
    width: usize,
    height: usize,
    subsampling: Subsampling,
) -> (usize, usize) {
    let mcu_width: usize = subsampling.mcu_width_blocks() * 8;
    let mcu_height: usize = subsampling.mcu_height_blocks() * 8;
    (
        checked_pad(width, mcu_width).unwrap_or(0),
        checked_pad(height, mcu_height).unwrap_or(0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_rounds_up_to_power_of_two() {
        assert_eq!(checked_pad(0, 8), Some(0));
        assert_eq!(checked_pad(1, 8), Some(8));
        assert_eq!(checked_pad(7, 8), Some(8));
        assert_eq!(checked_pad(8, 8), Some(8));
        assert_eq!(checked_pad(9, 8), Some(16));
        assert_eq!(checked_pad(640, 16), Some(640));
        assert_eq!(checked_pad(641, 16), Some(656));
    }

    #[test]
    fn pad_alignment_one() {
        assert_eq!(checked_pad(641, 1), Some(641));
        assert_eq!(checked_pad(0, 1), Some(0));
    }

    /// The arm that made `pad` fallible: rounding `usize::MAX` up to the next
    /// multiple of 16 wrapped to 0 before P4-139 chunk 2, so every size built
    /// on it came out *smaller* than the geometry asked for.
    #[test]
    fn padding_past_the_top_of_the_address_space_refuses() {
        assert_eq!(checked_pad(usize::MAX, 16), None);
        assert_eq!(checked_pad(usize::MAX - 14, 16), None);
        // One below that boundary is exactly representable.
        assert_eq!(checked_pad(usize::MAX - 15, 16), Some(usize::MAX - 15));
    }

    /// A chroma plane is exactly the padded luma plane divided by the
    /// subsampling ratio, at every magnitude.
    ///
    /// The arm this pins was written `padded * 8 / (factor * 8)`, whose only
    /// effect was that the `* 8` could wrap: at a width near `isize::MAX` the
    /// old form reported a chroma width *smaller* than the true one, which is
    /// the direction that under-sizes a buffer. The multiply is gone, so the
    /// identity now holds for every input.
    #[test]
    fn chroma_plane_dimensions_are_exact_at_every_magnitude() {
        // Even, so padding to any power-of-two factor is the identity, and
        // large enough that `* 8` would have wrapped on either pointer width.
        let huge: usize = (isize::MAX as usize) & !7;
        assert_eq!(yuv_plane_width(0, huge, Subsampling::S420), huge);
        assert_eq!(yuv_plane_width(1, huge, Subsampling::S420), huge / 2);
        assert_eq!(yuv_plane_height(1, huge, Subsampling::S420), huge / 2);
        assert_eq!(yuv_plane_width(1, huge, Subsampling::S411), huge / 4);
        assert_eq!(yuv_plane_height(1, huge, Subsampling::S441), huge / 4);
        // Ordinary geometry is unchanged by the rewrite.
        assert_eq!(yuv_plane_width(1, 641, Subsampling::S420), 321);
        assert_eq!(yuv_plane_height(1, 481, Subsampling::S420), 241);
        assert_eq!(yuv_plane_width(1, 640, Subsampling::S444), 640);
    }

    /// Public sizes report 0 rather than a wrapped product, matching the
    /// C-ABI twin's `tj3JPEGBufSize` / `tj3YUVPlaneSize` refusal.
    #[test]
    fn unrepresentable_geometry_reports_zero_not_a_wrapped_size() {
        assert_eq!(jpeg_buf_size(usize::MAX, usize::MAX, Subsampling::S444), 0);
        assert_eq!(jpeg_buf_size(usize::MAX / 4, 64, Subsampling::S420), 0);
        assert_eq!(
            yuv_plane_size(0, usize::MAX, usize::MAX, Subsampling::S444),
            0
        );
        assert_eq!(yuv_buf_size(usize::MAX, usize::MAX, Subsampling::S420), 0);
        assert_eq!(yuv_plane_width(0, usize::MAX, Subsampling::S420), 0);
        assert_eq!(yuv_plane_height(0, usize::MAX, Subsampling::S420), 0);
        assert_eq!(
            calc_jpeg_dimensions(usize::MAX, usize::MAX, Subsampling::S420),
            (0, 0)
        );
    }

    /// The checked rewrite must be value-identical on the geometry callers
    /// actually pass. These are the plain products the formulas had before.
    #[test]
    fn ordinary_geometry_is_unchanged_by_the_checked_rewrite() {
        // 4:2:0 pads to 16x16 MCUs; chroma scale factor is 4*64/(16*16) = 1.
        assert_eq!(
            jpeg_buf_size(640, 480, Subsampling::S420),
            640 * 480 * 3 + 2048
        );
        // 4:4:4 pads to 8x8; chroma scale factor is 4*64/(8*8) = 4.
        assert_eq!(
            jpeg_buf_size(640, 480, Subsampling::S444),
            640 * 480 * 6 + 2048
        );
        assert_eq!(yuv_plane_size(0, 640, 480, Subsampling::S420), 640 * 480);
        assert_eq!(
            yuv_plane_size(1, 640, 480, Subsampling::S420),
            320 * 240,
            "chroma is half in each axis at 4:2:0"
        );
        assert_eq!(
            yuv_buf_size(640, 480, Subsampling::S420),
            640 * 480 + 2 * (320 * 240)
        );
        assert_eq!(
            calc_jpeg_dimensions(641, 481, Subsampling::S420),
            (656, 496)
        );
        // A zero dimension keeps reporting an empty buffer, not a refusal.
        assert_eq!(yuv_buf_size(0, 480, Subsampling::S420), 0);
    }
}
