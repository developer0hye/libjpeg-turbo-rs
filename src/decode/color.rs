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

/// Copy grayscale values directly (no color conversion needed).
pub fn grayscale_row(y: &[u8], output: &mut [u8], width: usize) {
    output[..width].copy_from_slice(&y[..width]);
}
