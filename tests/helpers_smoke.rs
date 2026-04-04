//! Smoke test for shared test helpers module.

mod helpers;

#[test]
fn helpers_c_tool_discovery() {
    // djpeg should be findable on dev machines; graceful None on CI
    let djpeg = helpers::djpeg_path();
    if djpeg.is_none() {
        eprintln!("SKIP: djpeg not found");
    }
}

#[test]
fn helpers_temp_file_lifecycle() {
    let tf = helpers::TempFile::new("smoke_test.txt");
    tf.write_bytes(b"hello");
    assert!(tf.path().exists());
    let path = tf.path().to_owned();
    drop(tf);
    assert!(!path.exists(), "TempFile should auto-delete on drop");
}

#[test]
fn helpers_generate_gradient() {
    let pixels = helpers::generate_gradient(16, 16);
    assert_eq!(pixels.len(), 16 * 16 * 3);
    // Top-left pixel should be (0, 0, 0)
    assert_eq!(pixels[0], 0);
    assert_eq!(pixels[1], 0);
    assert_eq!(pixels[2], 0);
}

#[test]
fn helpers_parse_ppm_roundtrip() {
    let width: usize = 4;
    let height: usize = 3;
    let pixels: Vec<u8> = helpers::generate_gradient(width, height);
    let ppm: Vec<u8> = helpers::build_ppm(&pixels, width, height);
    let (w, h, data) = helpers::parse_ppm(&ppm).expect("parse_ppm should succeed");
    assert_eq!(w, width);
    assert_eq!(h, height);
    assert_eq!(data, pixels);
}

#[test]
fn helpers_parse_pgm_roundtrip() {
    let width: usize = 4;
    let height: usize = 3;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
    let pgm: Vec<u8> = helpers::build_pgm(&pixels, width, height);
    let (w, h, data) = helpers::parse_pgm(&pgm).expect("parse_pgm should succeed");
    assert_eq!(w, width);
    assert_eq!(h, height);
    assert_eq!(data, pixels);
}

#[test]
fn helpers_pixel_max_diff() {
    let a: Vec<u8> = vec![100, 200, 50];
    let b: Vec<u8> = vec![100, 203, 48];
    assert_eq!(helpers::pixel_max_diff(&a, &b), 3);

    let c: Vec<u8> = vec![100, 200, 50];
    assert_eq!(helpers::pixel_max_diff(&a, &c), 0);
}

#[test]
fn helpers_assert_pixels_identical_passes() {
    let pixels: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
    helpers::assert_pixels_identical(&pixels, &pixels, 2, 1, 3, "identical_test");
}

#[test]
fn helpers_build_ppm_format() {
    let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
    let ppm: Vec<u8> = helpers::build_ppm(&pixels, 3, 1);
    assert!(ppm.starts_with(b"P6\n3 1\n255\n"));
    assert_eq!(ppm.len(), "P6\n3 1\n255\n".len() + 9);
}
