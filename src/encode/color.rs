/// Forward color conversion: RGB/RGBA to YCbCr.
///
/// Uses ITU-R BT.601 coefficients in fixed-point arithmetic,
/// matching libjpeg-turbo's color conversion (jccolor.c / jccolext.c).
///
/// The Cb/Cr center offset (128) and rounding are incorporated into the
/// fixed-point computation *before* the right shift, exactly as libjpeg-turbo
/// does, to prevent overflow when the shift result would be 128 (producing 256
/// after a post-shift add).
const SCALEBITS: i32 = 16;
const ONE_HALF: i32 = 1 << (SCALEBITS - 1);
/// Center offset for Cb/Cr, pre-shifted into fixed-point domain.
const CBCR_OFFSET: i32 = 128 << SCALEBITS;

// Fixed-point constants: FIX(x) = (x * 65536.0 + 0.5) as i32
const FIX_0_299: i32 = 19595; // FIX(0.29900)
const FIX_0_587: i32 = 38470; // FIX(0.58700)
const FIX_0_114: i32 = 7471; // FIX(0.11400)
const FIX_0_16874: i32 = 11059; // FIX(0.16874)
const FIX_0_33126: i32 = 21709; // FIX(0.33126)
const FIX_0_500: i32 = 32768; // FIX(0.50000)
const FIX_0_41869: i32 = 27439; // FIX(0.41869)
const FIX_0_08131: i32 = 5329; // FIX(0.08131)

/// Compute Y, Cb, Cr from R, G, B values.
///
/// Uses the same rounding strategy as libjpeg-turbo: the Cb/Cr rounding
/// fudge factor is `ONE_HALF - 1` to ensure max output is 255, not 256.
#[inline(always)]
fn rgb_to_ycbcr(r: i32, g: i32, b: i32) -> (u8, u8, u8) {
    let y_val = ((FIX_0_299 * r + FIX_0_587 * g + FIX_0_114 * b + ONE_HALF) >> SCALEBITS) as u8;

    // Incorporate CBCR_OFFSET and rounding fudge before the shift.
    // The "- 1" prevents overflow to 256 at max input values.
    let cb_val = ((-FIX_0_16874 * r - FIX_0_33126 * g + FIX_0_500 * b + CBCR_OFFSET + ONE_HALF - 1)
        >> SCALEBITS) as u8;

    let cr_val = ((FIX_0_500 * r - FIX_0_41869 * g - FIX_0_08131 * b + CBCR_OFFSET + ONE_HALF - 1)
        >> SCALEBITS) as u8;

    (y_val, cb_val, cr_val)
}

/// Convert one row of RGB pixels to Y, Cb, Cr planes.
///
/// `rgb` must contain at least `width * 3` bytes (R, G, B per pixel).
/// `y`, `cb`, `cr` must each have at least `width` bytes.
pub fn rgb_to_ycbcr_row(rgb: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
    for i in 0..width {
        let r = rgb[i * 3] as i32;
        let g = rgb[i * 3 + 1] as i32;
        let b = rgb[i * 3 + 2] as i32;
        let (y_val, cb_val, cr_val) = rgb_to_ycbcr(r, g, b);
        y[i] = y_val;
        cb[i] = cb_val;
        cr[i] = cr_val;
    }
}

/// Convert one row of RGBA pixels to Y, Cb, Cr planes (alpha channel is ignored).
///
/// `rgba` must contain at least `width * 4` bytes (R, G, B, A per pixel).
/// `y`, `cb`, `cr` must each have at least `width` bytes.
pub fn rgba_to_ycbcr_row(rgba: &[u8], y: &mut [u8], cb: &mut [u8], cr: &mut [u8], width: usize) {
    for i in 0..width {
        let r = rgba[i * 4] as i32;
        let g = rgba[i * 4 + 1] as i32;
        let b = rgba[i * 4 + 2] as i32;
        let (y_val, cb_val, cr_val) = rgb_to_ycbcr(r, g, b);
        y[i] = y_val;
        cb[i] = cb_val;
        cr[i] = cr_val;
    }
}

/// Convert a row of pixels to Y, Cb, Cr planes using explicit channel offsets.
///
/// Supports any pixel format where R, G, B channels are at known byte offsets
/// within each `bpp`-byte pixel (Rgbx, Bgrx, Xrgb, Xbgr, Argb, Abgr, etc.).
#[allow(clippy::too_many_arguments)]
pub fn generic_to_ycbcr_row(
    pixels: &[u8],
    y: &mut [u8],
    cb: &mut [u8],
    cr: &mut [u8],
    width: usize,
    bpp: usize,
    r_off: usize,
    g_off: usize,
    b_off: usize,
) {
    for i in 0..width {
        let base: usize = i * bpp;
        let r = pixels[base + r_off] as i32;
        let g = pixels[base + g_off] as i32;
        let b = pixels[base + b_off] as i32;
        let (y_val, cb_val, cr_val) = rgb_to_ycbcr(r, g, b);
        y[i] = y_val;
        cb[i] = cb_val;
        cr[i] = cr_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_white_rgb() {
        let rgb = [255u8, 255, 255];
        let mut y = [0u8; 1];
        let mut cb = [0u8; 1];
        let mut cr = [0u8; 1];
        rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, 1);
        // White should map to Y=255, Cb=128, Cr=128
        assert_eq!(y[0], 255);
        assert_eq!(cb[0], 128);
        assert_eq!(cr[0], 128);
    }

    #[test]
    fn pure_black_rgb() {
        let rgb = [0u8, 0, 0];
        let mut y = [0u8; 1];
        let mut cb = [0u8; 1];
        let mut cr = [0u8; 1];
        rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, 1);
        assert_eq!(y[0], 0);
        assert_eq!(cb[0], 128);
        assert_eq!(cr[0], 128);
    }

    #[test]
    fn pure_red_rgb() {
        let rgb = [255u8, 0, 0];
        let mut y = [0u8; 1];
        let mut cb = [0u8; 1];
        let mut cr = [0u8; 1];
        rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, 1);
        // Y = 0.299 * 255 = 76.245 -> 76
        // Cb = 128 - 0.168736 * 255 = 128 - 43.03 -> 84 or 85
        // Cr = 128 + 0.5 * 255 = 128 + 127.5 -> 255
        assert_eq!(y[0], 76);
        assert!(cb[0] == 84 || cb[0] == 85);
        assert_eq!(cr[0], 255);
    }

    #[test]
    fn pure_green_rgb() {
        let rgb = [0u8, 255, 0];
        let mut y = [0u8; 1];
        let mut cb = [0u8; 1];
        let mut cr = [0u8; 1];
        rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, 1);
        // Y = 0.587 * 255 = 149.685 -> 149 or 150
        assert!(y[0] == 149 || y[0] == 150);
    }

    #[test]
    fn rgba_ignores_alpha() {
        let rgba = [100u8, 150, 200, 255];
        let mut y1 = [0u8; 1];
        let mut cb1 = [0u8; 1];
        let mut cr1 = [0u8; 1];
        rgba_to_ycbcr_row(&rgba, &mut y1, &mut cb1, &mut cr1, 1);

        let rgba2 = [100u8, 150, 200, 0];
        let mut y2 = [0u8; 1];
        let mut cb2 = [0u8; 1];
        let mut cr2 = [0u8; 1];
        rgba_to_ycbcr_row(&rgba2, &mut y2, &mut cb2, &mut cr2, 1);

        assert_eq!(y1[0], y2[0]);
        assert_eq!(cb1[0], cb2[0]);
        assert_eq!(cr1[0], cr2[0]);
    }

    #[test]
    fn rgb_multiple_pixels() {
        let rgb = [255u8, 0, 0, 0, 255, 0, 0, 0, 255];
        let mut y = [0u8; 3];
        let mut cb = [0u8; 3];
        let mut cr = [0u8; 3];
        rgb_to_ycbcr_row(&rgb, &mut y, &mut cb, &mut cr, 3);
        // Just verify all three pixels are converted (non-trivial values)
        assert_eq!(y[0], 76); // Red
        assert!(y[1] == 149 || y[1] == 150); // Green
        assert_eq!(y[2], 29); // Blue: Y = 0.114 * 255 ≈ 29
    }
}
