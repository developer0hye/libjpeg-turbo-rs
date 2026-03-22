//! Merged upsampling: combine chroma upsample + YCbCr->RGB color conversion in one pass.
//!
//! This avoids writing upsampled chroma to a temporary buffer, reducing memory
//! traffic and improving cache behavior. Equivalent to libjpeg-turbo's jdmerge.c.
//!
//! Supports H2V1 (4:2:2) and H2V2 (4:2:0) subsampling modes.
//! Uses the same BT.601 fixed-point coefficients as `color.rs`.

/// Fixed-point scale: 16-bit shift (matches libjpeg-turbo SCALEBITS=16).
const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);

/// FIX(1.40200) = Cr->R coefficient
const CR_R: i32 = 91881;
/// FIX(0.34414) = Cb->G coefficient (negative)
const CB_G: i32 = -22554;
/// FIX(0.71414) = Cr->G coefficient (negative)
const CR_G: i32 = -46802;
/// FIX(1.77200) = Cb->B coefficient
const CB_B: i32 = 116130;

#[inline(always)]
fn clamp(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

/// Compute the chroma offsets for a Cb/Cr pair. Returns (cred, cgreen, cblue)
/// which are integer deltas to be added to Y values.
#[inline(always)]
fn chroma_offsets(cb: u8, cr: u8) -> (i32, i32, i32) {
    let cb_i: i32 = cb as i32 - 128;
    let cr_i: i32 = cr as i32 - 128;
    let cred: i32 = (CR_R * cr_i + ONE_HALF) >> SCALEBITS;
    let cgreen: i32 = (CB_G * cb_i + CR_G * cr_i + ONE_HALF) >> SCALEBITS;
    let cblue: i32 = (CB_B * cb_i + ONE_HALF) >> SCALEBITS;
    (cred, cgreen, cblue)
}

/// Write one RGB pixel into the output buffer at the given base offset.
#[inline(always)]
fn emit_rgb(out: &mut [u8], base: usize, y_val: u8, cred: i32, cgreen: i32, cblue: i32) {
    let y_i: i32 = y_val as i32;
    out[base] = clamp(y_i + cred);
    out[base + 1] = clamp(y_i + cgreen);
    out[base + 2] = clamp(y_i + cblue);
}

/// Merged H2V1 upsample + YCbCr->RGB conversion.
///
/// Processes one row: Y is full width, Cb/Cr are half width.
/// Output is interleaved RGB at full width.
///
/// For each pair of horizontal pixels, one Cb/Cr sample is shared (box filter
/// replication). This avoids the intermediate upsampled chroma buffer entirely.
pub fn merged_h2v1_ycbcr_to_rgb(
    y_row: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out: &mut [u8],
    width: usize,
) {
    let pairs: usize = width / 2;
    for col in 0..pairs {
        let (cred, cgreen, cblue) = chroma_offsets(cb_row[col], cr_row[col]);
        // Left pixel
        emit_rgb(rgb_out, (col * 2) * 3, y_row[col * 2], cred, cgreen, cblue);
        // Right pixel
        emit_rgb(
            rgb_out,
            (col * 2 + 1) * 3,
            y_row[col * 2 + 1],
            cred,
            cgreen,
            cblue,
        );
    }
    // Handle odd width: last pixel uses last Cb/Cr sample
    if width & 1 != 0 {
        let last_x: usize = width - 1;
        let chroma_col: usize = last_x / 2;
        let (cred, cgreen, cblue) = chroma_offsets(cb_row[chroma_col], cr_row[chroma_col]);
        emit_rgb(rgb_out, last_x * 3, y_row[last_x], cred, cgreen, cblue);
    }
}

/// Merged H2V2 upsample + YCbCr->RGB conversion.
///
/// Processes two output rows at once. Y has 2 rows, Cb/Cr have 1 row (shared
/// between the two Y rows). Output is interleaved RGB at full width.
///
/// This is the 4:2:0 merged path: each Cb/Cr sample covers a 2x2 block of
/// luma pixels. By computing the chroma contribution once per 2x2 block, we
/// eliminate 3/4 of the chroma multiplications compared to the separate path.
pub fn merged_h2v2_ycbcr_to_rgb(
    y_row0: &[u8],
    y_row1: &[u8],
    cb_row: &[u8],
    cr_row: &[u8],
    rgb_out0: &mut [u8],
    rgb_out1: &mut [u8],
    width: usize,
) {
    let pairs: usize = width / 2;
    for col in 0..pairs {
        let (cred, cgreen, cblue) = chroma_offsets(cb_row[col], cr_row[col]);
        // Top-left pixel
        emit_rgb(
            rgb_out0,
            (col * 2) * 3,
            y_row0[col * 2],
            cred,
            cgreen,
            cblue,
        );
        // Top-right pixel
        emit_rgb(
            rgb_out0,
            (col * 2 + 1) * 3,
            y_row0[col * 2 + 1],
            cred,
            cgreen,
            cblue,
        );
        // Bottom-left pixel
        emit_rgb(
            rgb_out1,
            (col * 2) * 3,
            y_row1[col * 2],
            cred,
            cgreen,
            cblue,
        );
        // Bottom-right pixel
        emit_rgb(
            rgb_out1,
            (col * 2 + 1) * 3,
            y_row1[col * 2 + 1],
            cred,
            cgreen,
            cblue,
        );
    }
    // Handle odd width
    if width & 1 != 0 {
        let last_x: usize = width - 1;
        let chroma_col: usize = last_x / 2;
        let (cred, cgreen, cblue) = chroma_offsets(cb_row[chroma_col], cr_row[chroma_col]);
        emit_rgb(rgb_out0, last_x * 3, y_row0[last_x], cred, cgreen, cblue);
        emit_rgb(rgb_out1, last_x * 3, y_row1[last_x], cred, cgreen, cblue);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h2v1_pure_gray_produces_gray_rgb() {
        // Y=128, Cb=128, Cr=128 -> RGB should be (128, 128, 128)
        let y_row: Vec<u8> = vec![128; 8];
        let cb_row: Vec<u8> = vec![128; 4];
        let cr_row: Vec<u8> = vec![128; 4];
        let mut rgb: Vec<u8> = vec![0u8; 8 * 3];
        merged_h2v1_ycbcr_to_rgb(&y_row, &cb_row, &cr_row, &mut rgb, 8);
        for i in 0..8 {
            assert_eq!(rgb[i * 3], 128, "R at pixel {}", i);
            assert_eq!(rgb[i * 3 + 1], 128, "G at pixel {}", i);
            assert_eq!(rgb[i * 3 + 2], 128, "B at pixel {}", i);
        }
    }

    #[test]
    fn h2v1_odd_width() {
        let y_row: Vec<u8> = vec![200; 7];
        let cb_row: Vec<u8> = vec![128; 4]; // ceil(7/2) = 4
        let cr_row: Vec<u8> = vec![128; 4];
        let mut rgb: Vec<u8> = vec![0u8; 7 * 3];
        merged_h2v1_ycbcr_to_rgb(&y_row, &cb_row, &cr_row, &mut rgb, 7);
        // All pixels should be (200, 200, 200) since Cb=Cr=128
        for i in 0..7 {
            assert_eq!(rgb[i * 3], 200, "R at pixel {}", i);
            assert_eq!(rgb[i * 3 + 1], 200, "G at pixel {}", i);
            assert_eq!(rgb[i * 3 + 2], 200, "B at pixel {}", i);
        }
    }

    #[test]
    fn h2v2_pure_gray() {
        let y0: Vec<u8> = vec![100; 6];
        let y1: Vec<u8> = vec![150; 6];
        let cb: Vec<u8> = vec![128; 3];
        let cr: Vec<u8> = vec![128; 3];
        let mut rgb0: Vec<u8> = vec![0u8; 6 * 3];
        let mut rgb1: Vec<u8> = vec![0u8; 6 * 3];
        merged_h2v2_ycbcr_to_rgb(&y0, &y1, &cb, &cr, &mut rgb0, &mut rgb1, 6);
        for i in 0..6 {
            assert_eq!(rgb0[i * 3], 100);
            assert_eq!(rgb0[i * 3 + 1], 100);
            assert_eq!(rgb0[i * 3 + 2], 100);
            assert_eq!(rgb1[i * 3], 150);
            assert_eq!(rgb1[i * 3 + 1], 150);
            assert_eq!(rgb1[i * 3 + 2], 150);
        }
    }

    #[test]
    fn h2v1_matches_separate_upsample_color_convert() {
        // Verify that merged produces identical output to separate box upsample + convert
        let width: usize = 10;
        let y_row: Vec<u8> = (0..10).map(|i| (i * 25) as u8).collect();
        let cb_row: Vec<u8> = vec![100, 120, 140, 160, 180];
        let cr_row: Vec<u8> = vec![80, 100, 120, 140, 160];

        // Merged path
        let mut merged_rgb: Vec<u8> = vec![0u8; width * 3];
        merged_h2v1_ycbcr_to_rgb(&y_row, &cb_row, &cr_row, &mut merged_rgb, width);

        // Separate: box upsample then color convert
        let mut cb_full: Vec<u8> = vec![0u8; width];
        let mut cr_full: Vec<u8> = vec![0u8; width];
        for i in 0..5 {
            cb_full[i * 2] = cb_row[i];
            cb_full[i * 2 + 1] = cb_row[i];
            cr_full[i * 2] = cr_row[i];
            cr_full[i * 2 + 1] = cr_row[i];
        }
        let mut separate_rgb: Vec<u8> = vec![0u8; width * 3];
        for x in 0..width {
            let (r, g, b) =
                crate::decode::color::ycbcr_to_rgb_pixel(y_row[x], cb_full[x], cr_full[x]);
            separate_rgb[x * 3] = r;
            separate_rgb[x * 3 + 1] = g;
            separate_rgb[x * 3 + 2] = b;
        }

        assert_eq!(merged_rgb, separate_rgb);
    }

    #[test]
    fn h2v2_matches_separate_box_upsample_color_convert() {
        let width: usize = 8;
        let y0: Vec<u8> = (0..8).map(|i| (i * 30 + 10) as u8).collect();
        let y1: Vec<u8> = (0..8).map(|i| (i * 20 + 50) as u8).collect();
        let cb: Vec<u8> = vec![90, 110, 130, 150];
        let cr: Vec<u8> = vec![70, 90, 110, 130];

        // Merged
        let mut m_rgb0: Vec<u8> = vec![0u8; width * 3];
        let mut m_rgb1: Vec<u8> = vec![0u8; width * 3];
        merged_h2v2_ycbcr_to_rgb(&y0, &y1, &cb, &cr, &mut m_rgb0, &mut m_rgb1, width);

        // Separate: box upsample (both rows use same chroma)
        let mut cb_full: Vec<u8> = vec![0u8; width];
        let mut cr_full: Vec<u8> = vec![0u8; width];
        for i in 0..4 {
            cb_full[i * 2] = cb[i];
            cb_full[i * 2 + 1] = cb[i];
            cr_full[i * 2] = cr[i];
            cr_full[i * 2 + 1] = cr[i];
        }
        let mut s_rgb0: Vec<u8> = vec![0u8; width * 3];
        let mut s_rgb1: Vec<u8> = vec![0u8; width * 3];
        for x in 0..width {
            let (r, g, b) = crate::decode::color::ycbcr_to_rgb_pixel(y0[x], cb_full[x], cr_full[x]);
            s_rgb0[x * 3] = r;
            s_rgb0[x * 3 + 1] = g;
            s_rgb0[x * 3 + 2] = b;
            let (r2, g2, b2) =
                crate::decode::color::ycbcr_to_rgb_pixel(y1[x], cb_full[x], cr_full[x]);
            s_rgb1[x * 3] = r2;
            s_rgb1[x * 3 + 1] = g2;
            s_rgb1[x * 3 + 2] = b2;
        }

        assert_eq!(m_rgb0, s_rgb0);
        assert_eq!(m_rgb1, s_rgb1);
    }
}
