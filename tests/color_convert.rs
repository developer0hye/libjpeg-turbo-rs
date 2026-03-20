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
