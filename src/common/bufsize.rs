//! Buffer size calculation helpers matching libjpeg-turbo's TurboJPEG API.
//!
//! These functions compute worst-case buffer sizes for JPEG and YUV data,
//! following the exact formulas from `tj3JPEGBufSize`, `tj3YUVBufSize`,
//! `tj3YUVPlaneSize`, `tj3YUVPlaneWidth`, and `tj3YUVPlaneHeight`.

use super::types::Subsampling;

/// Rounds `value` up to the nearest multiple of `alignment`.
///
/// `alignment` must be a power of two.
#[inline]
fn pad(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Worst-case JPEG output buffer size for given dimensions and subsampling.
///
/// Matches `tj3JPEGBufSize()` from libjpeg-turbo. The result accounts for rare
/// corner cases where compressed JPEG data can exceed uncompressed pixel data.
pub fn jpeg_buf_size(width: usize, height: usize, subsampling: Subsampling) -> usize {
    let mcu_width: usize = subsampling.mcu_width_blocks() * 8;
    let mcu_height: usize = subsampling.mcu_height_blocks() * 8;

    // Chroma scaling factor: ratio of chroma samples to total MCU pixels.
    // For color images: 4 * 64 / (mcu_width * mcu_height).
    let chroma_scale_factor: usize = 4 * 64 / (mcu_width * mcu_height);

    pad(width, mcu_width) * pad(height, mcu_height) * (2 + chroma_scale_factor) + 2048
}

/// Width (in pixels/samples) of a single YUV plane.
///
/// Matches `tj3YUVPlaneWidth()`. Component 0 is luma (Y); components 1 and 2
/// are chroma (Cb, Cr). The width is padded to the subsampling factor boundary.
pub fn yuv_plane_width(component: usize, width: usize, subsampling: Subsampling) -> usize {
    let h_factor: usize = subsampling.mcu_width_blocks(); // mcuw / 8
    let padded_width: usize = pad(width, h_factor);

    if component == 0 {
        padded_width
    } else {
        // Chroma plane: divide by horizontal subsampling ratio
        padded_width * 8 / (h_factor * 8)
    }
}

/// Height (in pixels/samples) of a single YUV plane.
///
/// Matches `tj3YUVPlaneHeight()`. Component 0 is luma (Y); components 1 and 2
/// are chroma (Cb, Cr). The height is padded to the subsampling factor boundary.
pub fn yuv_plane_height(component: usize, height: usize, subsampling: Subsampling) -> usize {
    let v_factor: usize = subsampling.mcu_height_blocks(); // mcuh / 8
    let padded_height: usize = pad(height, v_factor);

    if component == 0 {
        padded_height
    } else {
        // Chroma plane: divide by vertical subsampling ratio
        padded_height * 8 / (v_factor * 8)
    }
}

/// Buffer size for a single YUV plane (stride = plane width, no padding).
///
/// Matches `tj3YUVPlaneSize()` with `stride = 0`. The formula is
/// `stride * (height - 1) + width`, which for stride == width simplifies to
/// `width * height`.
pub fn yuv_plane_size(
    component: usize,
    width: usize,
    height: usize,
    subsampling: Subsampling,
) -> usize {
    let plane_width: usize = yuv_plane_width(component, width, subsampling);
    let plane_height: usize = yuv_plane_height(component, height, subsampling);

    // stride * (ph - 1) + pw, where stride == pw
    plane_width * plane_height
}

/// Total packed YUV buffer size (3 components, no row padding).
///
/// Matches `tj3YUVBufSize()` with `align = 1`. Returns the sum of all three
/// plane sizes (Y + Cb + Cr).
pub fn yuv_buf_size(width: usize, height: usize, subsampling: Subsampling) -> usize {
    let mut total: usize = 0;
    for component in 0..3 {
        total += yuv_plane_size(component, width, height, subsampling);
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_rounds_up_to_power_of_two() {
        assert_eq!(pad(0, 8), 0);
        assert_eq!(pad(1, 8), 8);
        assert_eq!(pad(7, 8), 8);
        assert_eq!(pad(8, 8), 8);
        assert_eq!(pad(9, 8), 16);
        assert_eq!(pad(640, 16), 640);
        assert_eq!(pad(641, 16), 656);
    }

    #[test]
    fn pad_alignment_one() {
        assert_eq!(pad(641, 1), 641);
        assert_eq!(pad(0, 1), 0);
    }
}
