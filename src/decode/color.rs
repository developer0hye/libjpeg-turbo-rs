#[inline(always)]
fn clamp(val: i32) -> u8 {
    val.clamp(0, 255) as u8
}

/// Convert a single YCbCr pixel to RGB (ITU-R BT.601, fixed-point).
pub fn ycbcr_to_rgb_pixel(y: u8, cb: u8, cr: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let cb = cb as i32 - 128;
    let cr = cr as i32 - 128;

    let r = y + ((91881 * cr + 32768) >> 16);
    let g = y - ((22554 * cb + 46802 * cr + 32768) >> 16);
    let b = y + ((116130 * cb + 32768) >> 16);

    (clamp(r), clamp(g), clamp(b))
}

/// Convert a row of YCbCr pixels to interleaved RGB.
pub fn ycbcr_to_rgb_row(y: &[u8], cb: &[u8], cr: &[u8], rgb: &mut [u8], width: usize) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        rgb[x * 3] = r;
        rgb[x * 3 + 1] = g;
        rgb[x * 3 + 2] = b;
    }
}

/// Convert a row of YCbCr pixels to interleaved RGBA (alpha = 255).
pub fn ycbcr_to_rgba_row(y: &[u8], cb: &[u8], cr: &[u8], rgba: &mut [u8], width: usize) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        rgba[x * 4] = r;
        rgba[x * 4 + 1] = g;
        rgba[x * 4 + 2] = b;
        rgba[x * 4 + 3] = 255;
    }
}

/// Convert a row of YCbCr pixels to interleaved BGR.
pub fn ycbcr_to_bgr_row(y: &[u8], cb: &[u8], cr: &[u8], bgr: &mut [u8], width: usize) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        bgr[x * 3] = b;
        bgr[x * 3 + 1] = g;
        bgr[x * 3 + 2] = r;
    }
}

/// Convert a row of YCbCr pixels to interleaved BGRA (alpha = 255).
pub fn ycbcr_to_bgra_row(y: &[u8], cb: &[u8], cr: &[u8], bgra: &mut [u8], width: usize) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        bgra[x * 4] = b;
        bgra[x * 4 + 1] = g;
        bgra[x * 4 + 2] = r;
        bgra[x * 4 + 3] = 255;
    }
}

/// Convert a row of YCbCr pixels to an output format using explicit channel offsets.
///
/// Places R, G, B at `r_off`, `g_off`, `b_off` within each `bpp`-byte pixel.
/// The remaining byte (padding/alpha) is set to 255.
#[allow(clippy::too_many_arguments)]
pub fn ycbcr_to_generic_4bpp_row(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    out: &mut [u8],
    width: usize,
    r_off: usize,
    g_off: usize,
    b_off: usize,
    pad_off: usize,
) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        let base: usize = x * 4;
        out[base + r_off] = r;
        out[base + g_off] = g;
        out[base + b_off] = b;
        out[base + pad_off] = 255;
    }
}

/// Convert a row of YCbCr pixels to Rgb565 (5-6-5 packed, native endian).
pub fn ycbcr_to_rgb565_row(y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        let packed: u16 = ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3);
        let bytes: [u8; 2] = packed.to_ne_bytes();
        out[x * 2] = bytes[0];
        out[x * 2 + 1] = bytes[1];
    }
}

/// 4x4 ordered dither matrix for RGB565 output, matching libjpeg-turbo's jdcolor.c.
///
/// Each row is packed into a u32 with 4 dither values in the bytes.
/// The matrix reduces quantization banding when truncating 8-bit RGB to 5-6-5.
const DITHER_MATRIX: [u32; 4] = [0x0008_020A, 0x0C04_0E06, 0x030B_0109, 0x0F07_0D05];

/// Convert a row of YCbCr pixels to dithered Rgb565 (5-6-5 packed, native endian).
///
/// Uses the same 4x4 ordered dither pattern as libjpeg-turbo. The `row_index`
/// selects the dither row (typically `y & 3`).
pub fn ycbcr_to_rgb565_dithered_row(
    y: &[u8],
    cb: &[u8],
    cr: &[u8],
    out: &mut [u8],
    width: usize,
    row_index: usize,
) {
    let mut dither: u32 = DITHER_MATRIX[row_index & 0x3];
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        let d: u8 = (dither & 0xFF) as u8;
        // Add dither offset and clamp to 255 before truncation.
        // R and B get full dither value (5-bit target = 3 bits lost).
        // G gets half dither (6-bit target = 2 bits lost).
        let rd: u8 = r.saturating_add(d);
        let gd: u8 = g.saturating_add(d >> 1);
        let bd: u8 = b.saturating_add(d);
        let packed: u16 = ((rd as u16 >> 3) << 11) | ((gd as u16 >> 2) << 5) | (bd as u16 >> 3);
        let bytes: [u8; 2] = packed.to_ne_bytes();
        out[x * 2] = bytes[0];
        out[x * 2 + 1] = bytes[1];
        // Rotate dither: shift right by 8, wrap low byte to high.
        dither = ((dither & 0xFF) << 24) | ((dither >> 8) & 0x00FF_FFFF);
    }
}

/// Convert a grayscale value to dithered Rgb565 (5-6-5 packed, native endian).
///
/// For grayscale images decoded to RGB565 with dithering enabled.
pub fn gray_to_rgb565_dithered_row(y: &[u8], out: &mut [u8], width: usize, row_index: usize) {
    let mut dither: u32 = DITHER_MATRIX[row_index & 0x3];
    for x in 0..width {
        let v: u8 = y[x];
        let d: u8 = (dither & 0xFF) as u8;
        let vd_r: u8 = v.saturating_add(d);
        let vd_g: u8 = v.saturating_add(d >> 1);
        let vd_b: u8 = v.saturating_add(d);
        let packed: u16 =
            ((vd_r as u16 >> 3) << 11) | ((vd_g as u16 >> 2) << 5) | (vd_b as u16 >> 3);
        let bytes: [u8; 2] = packed.to_ne_bytes();
        out[x * 2] = bytes[0];
        out[x * 2 + 1] = bytes[1];
        dither = ((dither & 0xFF) << 24) | ((dither >> 8) & 0x00FF_FFFF);
    }
}

/// Copy grayscale values directly (no color conversion needed).
pub fn grayscale_row(y: &[u8], output: &mut [u8], width: usize) {
    output[..width].copy_from_slice(&y[..width]);
}

/// Convert a row of YCCK pixels to interleaved CMYK.
///
/// YCCK is Adobe's encoding: the first 3 channels are YCbCr, the 4th is K.
/// We convert YCbCr → RGB, then invert to get CMY: C = 255 - R, M = 255 - G, Y = 255 - B.
/// K passes through unchanged.
pub fn ycck_to_cmyk_row(y: &[u8], cb: &[u8], cr: &[u8], k: &[u8], cmyk: &mut [u8], width: usize) {
    for x in 0..width {
        let (r, g, b) = ycbcr_to_rgb_pixel(y[x], cb[x], cr[x]);
        cmyk[x * 4] = 255 - r;
        cmyk[x * 4 + 1] = 255 - g;
        cmyk[x * 4 + 2] = 255 - b;
        cmyk[x * 4 + 3] = k[x];
    }
}

/// Convert a row of CMYK pixels to interleaved RGB.
///
/// Uses the standard formula: R = (255 - C) * (255 - K) / 255, etc.
pub fn cmyk_to_rgb_row(c: &[u8], m: &[u8], y: &[u8], k: &[u8], rgb: &mut [u8], width: usize) {
    for x in 0..width {
        let ki = 255 - k[x] as u16;
        rgb[x * 3] = (((255 - c[x] as u16) * ki + 127) / 255) as u8;
        rgb[x * 3 + 1] = (((255 - m[x] as u16) * ki + 127) / 255) as u8;
        rgb[x * 3 + 2] = (((255 - y[x] as u16) * ki + 127) / 255) as u8;
    }
}

/// Convert a row of CMYK pixels to interleaved RGBA (alpha = 255).
pub fn cmyk_to_rgba_row(c: &[u8], m: &[u8], y: &[u8], k: &[u8], rgba: &mut [u8], width: usize) {
    for x in 0..width {
        let ki = 255 - k[x] as u16;
        rgba[x * 4] = (((255 - c[x] as u16) * ki + 127) / 255) as u8;
        rgba[x * 4 + 1] = (((255 - m[x] as u16) * ki + 127) / 255) as u8;
        rgba[x * 4 + 2] = (((255 - y[x] as u16) * ki + 127) / 255) as u8;
        rgba[x * 4 + 3] = 255;
    }
}

/// Convert a row of CMYK pixels to interleaved BGR.
pub fn cmyk_to_bgr_row(c: &[u8], m: &[u8], y: &[u8], k: &[u8], bgr: &mut [u8], width: usize) {
    for x in 0..width {
        let ki = 255 - k[x] as u16;
        let r = (((255 - c[x] as u16) * ki + 127) / 255) as u8;
        let g = (((255 - m[x] as u16) * ki + 127) / 255) as u8;
        let b = (((255 - y[x] as u16) * ki + 127) / 255) as u8;
        bgr[x * 3] = b;
        bgr[x * 3 + 1] = g;
        bgr[x * 3 + 2] = r;
    }
}

/// Convert a row of CMYK pixels to interleaved BGRA (alpha = 255).
pub fn cmyk_to_bgra_row(c: &[u8], m: &[u8], y: &[u8], k: &[u8], bgra: &mut [u8], width: usize) {
    for x in 0..width {
        let ki = 255 - k[x] as u16;
        let r = (((255 - c[x] as u16) * ki + 127) / 255) as u8;
        let g = (((255 - m[x] as u16) * ki + 127) / 255) as u8;
        let b = (((255 - y[x] as u16) * ki + 127) / 255) as u8;
        bgra[x * 4] = b;
        bgra[x * 4 + 1] = g;
        bgra[x * 4 + 2] = r;
        bgra[x * 4 + 3] = 255;
    }
}

/// Copy 4 component planes to interleaved CMYK (no conversion).
pub fn cmyk_passthrough_row(c: &[u8], m: &[u8], y: &[u8], k: &[u8], cmyk: &mut [u8], width: usize) {
    for x in 0..width {
        cmyk[x * 4] = c[x];
        cmyk[x * 4 + 1] = m[x];
        cmyk[x * 4 + 2] = y[x];
        cmyk[x * 4 + 3] = k[x];
    }
}
