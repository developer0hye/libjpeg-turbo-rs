use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};

#[test]
fn conformance_grayscale_8x8() {
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 8);
    assert_eq!(image.height, 8);

    for &pixel in &image.data {
        assert!(
            (pixel as i16 - 128).unsigned_abs() <= 2,
            "pixel {} too far from 128",
            pixel
        );
    }
}

#[test]
fn conformance_rgb_444() {
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    assert_eq!(image.data.len(), 16 * 16 * 3);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r > 240, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn conformance_rgb_422() {
    let data = include_bytes!("fixtures/green_16x16_422.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g > 240, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

#[test]
fn conformance_rgb_420() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");
    let image = decompress(data).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            assert!(r < 15, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b > 240, "pixel ({},{}) B={}", x, y, b);
        }
    }
}

// --- Output format tests ---

#[test]
fn decompress_to_rgba_444() {
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let image = decompress_to(data, PixelFormat::Rgba).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Rgba);
    assert_eq!(image.data.len(), 16 * 16 * 4);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 4;
            let r = image.data[idx];
            let g = image.data[idx + 1];
            let b = image.data[idx + 2];
            let a = image.data[idx + 3];
            assert!(r > 240, "pixel ({},{}) R={}", x, y, r);
            assert!(g < 15, "pixel ({},{}) G={}", x, y, g);
            assert!(b < 15, "pixel ({},{}) B={}", x, y, b);
            assert_eq!(a, 255, "pixel ({},{}) A={}", x, y, a);
        }
    }
}

#[test]
fn decompress_to_bgr_444() {
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let image = decompress_to(data, PixelFormat::Bgr).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Bgr);
    assert_eq!(image.data.len(), 16 * 16 * 3);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 3;
            // BGR order: B, G, R
            let b_val = image.data[idx];
            let g_val = image.data[idx + 1];
            let r_val = image.data[idx + 2];
            assert!(r_val > 240, "pixel ({},{}) R={}", x, y, r_val);
            assert!(g_val < 15, "pixel ({},{}) G={}", x, y, g_val);
            assert!(b_val < 15, "pixel ({},{}) B={}", x, y, b_val);
        }
    }
}

#[test]
fn decompress_to_bgra_420() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");
    let image = decompress_to(data, PixelFormat::Bgra).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Bgra);
    assert_eq!(image.data.len(), 16 * 16 * 4);

    for y in 0..16 {
        for x in 0..16 {
            let idx = (y * 16 + x) * 4;
            // BGRA order: B, G, R, A
            let b_val = image.data[idx];
            let g_val = image.data[idx + 1];
            let r_val = image.data[idx + 2];
            let a_val = image.data[idx + 3];
            assert!(r_val < 15, "pixel ({},{}) R={}", x, y, r_val);
            assert!(g_val < 15, "pixel ({},{}) G={}", x, y, g_val);
            assert!(b_val > 240, "pixel ({},{}) B={}", x, y, b_val);
            assert_eq!(a_val, 255, "pixel ({},{}) A={}", x, y, a_val);
        }
    }
}

#[test]
fn decompress_to_rgb_default() {
    // decompress_to with Rgb should match decompress
    let data = include_bytes!("fixtures/red_16x16_444.jpg");
    let img_default = decompress(data).unwrap();
    let img_explicit = decompress_to(data, PixelFormat::Rgb).unwrap();
    assert_eq!(img_default.data, img_explicit.data);
    assert_eq!(img_default.pixel_format, img_explicit.pixel_format);
}

#[test]
fn decompress_to_grayscale_stays_grayscale() {
    let data = include_bytes!("fixtures/gray_8x8.jpg");
    let image = decompress_to(data, PixelFormat::Grayscale).unwrap();
    assert_eq!(image.pixel_format, PixelFormat::Grayscale);
    assert_eq!(image.data.len(), 8 * 8);
}
