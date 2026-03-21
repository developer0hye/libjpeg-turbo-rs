use libjpeg_turbo_rs::decode::color;

#[test]
fn ycbcr_to_rgb_white() {
    let (r, g, b) = color::ycbcr_to_rgb_pixel(255, 128, 128);
    assert_eq!((r, g, b), (255, 255, 255));
}

#[test]
fn ycbcr_to_rgb_black() {
    let (r, g, b) = color::ycbcr_to_rgb_pixel(0, 128, 128);
    assert_eq!((r, g, b), (0, 0, 0));
}

#[test]
fn ycbcr_to_rgb_red() {
    let (r, g, b) = color::ycbcr_to_rgb_pixel(76, 84, 255);
    assert!(r >= 254, "red channel: {}", r);
    assert!(g <= 1, "green channel: {}", g);
    assert!(b <= 1, "blue channel: {}", b);
}

#[test]
fn ycbcr_to_rgb_bulk() {
    let y = [255u8, 0, 76, 149];
    let cb = [128u8, 128, 84, 43];
    let cr = [128u8, 128, 255, 21];

    let mut rgb = [0u8; 12];
    color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, 4);

    assert_eq!(&rgb[0..3], &[255, 255, 255]);
    assert_eq!(&rgb[3..6], &[0, 0, 0]);
}

// --- RGBA tests ---

#[test]
fn ycbcr_to_rgba_white() {
    let y = [255u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut rgba = [0u8; 4];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 1);
    assert_eq!(&rgba, &[255, 255, 255, 255]);
}

#[test]
fn ycbcr_to_rgba_black() {
    let y = [0u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut rgba = [0u8; 4];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 1);
    assert_eq!(&rgba, &[0, 0, 0, 255]);
}

#[test]
fn ycbcr_to_rgba_bulk() {
    let y = [255u8, 0, 76, 149];
    let cb = [128u8, 128, 84, 43];
    let cr = [128u8, 128, 255, 21];
    let mut rgba = [0u8; 16];
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 4);

    // White: R=255,G=255,B=255,A=255
    assert_eq!(&rgba[0..4], &[255, 255, 255, 255]);
    // Black: R=0,G=0,B=0,A=255
    assert_eq!(&rgba[4..8], &[0, 0, 0, 255]);
    // Alpha always 255
    assert_eq!(rgba[7], 255);
    assert_eq!(rgba[11], 255);
    assert_eq!(rgba[15], 255);
}

// --- BGR tests ---

#[test]
fn ycbcr_to_bgr_white() {
    let y = [255u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut bgr = [0u8; 3];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 1);
    assert_eq!(&bgr, &[255, 255, 255]);
}

#[test]
fn ycbcr_to_bgr_red() {
    // Pure red: should produce BGR = [~0, ~0, ~255]
    let y = [76u8];
    let cb = [84u8];
    let cr = [255u8];
    let mut bgr = [0u8; 3];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 1);
    assert!(bgr[0] <= 1, "B channel: {}", bgr[0]); // B first in BGR
    assert!(bgr[1] <= 1, "G channel: {}", bgr[1]);
    assert!(bgr[2] >= 254, "R channel: {}", bgr[2]); // R last in BGR
}

#[test]
fn ycbcr_to_bgr_bulk() {
    let y = [255u8, 0];
    let cb = [128u8, 128];
    let cr = [128u8, 128];
    let mut bgr = [0u8; 6];
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 2);
    assert_eq!(&bgr[0..3], &[255, 255, 255]); // White is same in BGR
    assert_eq!(&bgr[3..6], &[0, 0, 0]); // Black is same in BGR
}

// --- BGRA tests ---

#[test]
fn ycbcr_to_bgra_white() {
    let y = [255u8];
    let cb = [128u8];
    let cr = [128u8];
    let mut bgra = [0u8; 4];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 1);
    assert_eq!(&bgra, &[255, 255, 255, 255]);
}

#[test]
fn ycbcr_to_bgra_red() {
    let y = [76u8];
    let cb = [84u8];
    let cr = [255u8];
    let mut bgra = [0u8; 4];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 1);
    assert!(bgra[0] <= 1, "B channel: {}", bgra[0]);
    assert!(bgra[1] <= 1, "G channel: {}", bgra[1]);
    assert!(bgra[2] >= 254, "R channel: {}", bgra[2]);
    assert_eq!(bgra[3], 255, "A channel must be 255");
}

#[test]
fn ycbcr_to_bgra_bulk_alpha() {
    let y = [100u8, 200, 50];
    let cb = [128u8, 128, 128];
    let cr = [128u8, 128, 128];
    let mut bgra = [0u8; 12];
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 3);
    // All alpha channels must be 255
    assert_eq!(bgra[3], 255);
    assert_eq!(bgra[7], 255);
    assert_eq!(bgra[11], 255);
}

// --- Cross-format consistency ---

#[test]
fn rgb_rgba_bgr_bgra_consistency() {
    // Same YCbCr input should produce consistent R,G,B across all formats
    let y = [76u8, 149, 29];
    let cb = [84u8, 43, 255];
    let cr = [255u8, 21, 107];

    let mut rgb = [0u8; 9];
    let mut rgba = [0u8; 12];
    let mut bgr = [0u8; 9];
    let mut bgra = [0u8; 12];

    color::ycbcr_to_rgb_row(&y, &cb, &cr, &mut rgb, 3);
    color::ycbcr_to_rgba_row(&y, &cb, &cr, &mut rgba, 3);
    color::ycbcr_to_bgr_row(&y, &cb, &cr, &mut bgr, 3);
    color::ycbcr_to_bgra_row(&y, &cb, &cr, &mut bgra, 3);

    for i in 0..3 {
        let r_rgb = rgb[i * 3];
        let g_rgb = rgb[i * 3 + 1];
        let b_rgb = rgb[i * 3 + 2];

        // RGBA: same order + alpha
        assert_eq!(rgba[i * 4], r_rgb, "RGBA R mismatch at pixel {}", i);
        assert_eq!(rgba[i * 4 + 1], g_rgb, "RGBA G mismatch at pixel {}", i);
        assert_eq!(rgba[i * 4 + 2], b_rgb, "RGBA B mismatch at pixel {}", i);
        assert_eq!(rgba[i * 4 + 3], 255, "RGBA A must be 255");

        // BGR: reversed channel order
        assert_eq!(bgr[i * 3], b_rgb, "BGR B mismatch at pixel {}", i);
        assert_eq!(bgr[i * 3 + 1], g_rgb, "BGR G mismatch at pixel {}", i);
        assert_eq!(bgr[i * 3 + 2], r_rgb, "BGR R mismatch at pixel {}", i);

        // BGRA: reversed + alpha
        assert_eq!(bgra[i * 4], b_rgb, "BGRA B mismatch at pixel {}", i);
        assert_eq!(bgra[i * 4 + 1], g_rgb, "BGRA G mismatch at pixel {}", i);
        assert_eq!(bgra[i * 4 + 2], r_rgb, "BGRA R mismatch at pixel {}", i);
        assert_eq!(bgra[i * 4 + 3], 255, "BGRA A must be 255");
    }
}
