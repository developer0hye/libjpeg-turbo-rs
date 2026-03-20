use libjpeg_turbo_rs::decompress;

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
