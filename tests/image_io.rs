use libjpeg_turbo_rs::api::image_io::{
    load_image, load_image_from_bytes, save_bmp, save_ppm, LoadedImage,
};
use libjpeg_turbo_rs::PixelFormat;
use std::path::PathBuf;

/// Helper: create a temp file path with a unique name.
fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("ljt_test_{}", name))
}

/// Helper: generate a deterministic RGB pixel pattern for testing.
fn make_test_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 37 + y * 13) % 256) as u8;
            let g: u8 = ((x * 59 + y * 7) % 256) as u8;
            let b: u8 = ((x * 11 + y * 41) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Helper: generate a deterministic RGBA pixel pattern for testing.
fn make_test_rgba(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 37 + y * 13) % 256) as u8;
            let g: u8 = ((x * 59 + y * 7) % 256) as u8;
            let b: u8 = ((x * 11 + y * 41) % 256) as u8;
            let a: u8 = ((x * 23 + y * 31) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
            pixels.push(a);
        }
    }
    pixels
}

/// Helper: generate grayscale pixel data.
fn make_test_gray(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            let v: u8 = ((x * 37 + y * 13) % 256) as u8;
            pixels.push(v);
        }
    }
    pixels
}

#[test]
fn bmp_roundtrip_rgb() {
    let path: PathBuf = temp_path("bmp_rgb.bmp");
    let width: usize = 16;
    let height: usize = 12;
    let pixels: Vec<u8> = make_test_rgb(width, height);

    save_bmp(&path, &pixels, width, height, PixelFormat::Rgb).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    assert_eq!(loaded.width, width);
    assert_eq!(loaded.height, height);
    assert_eq!(loaded.pixel_format, PixelFormat::Rgb);
    assert_eq!(loaded.pixels, pixels);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn bmp_roundtrip_grayscale() {
    // Grayscale is not natively supported by BMP as 24-bit,
    // so save_bmp should convert grayscale to BGR and load_image returns RGB.
    // We test by saving grayscale as BMP (which internally writes as 24-bit BGR
    // with R=G=B=gray) and loading back as RGB, then verifying each component
    // matches the original gray value.
    let path: PathBuf = temp_path("bmp_gray.bmp");
    let width: usize = 8;
    let height: usize = 8;
    let gray: Vec<u8> = make_test_gray(width, height);

    save_bmp(&path, &gray, width, height, PixelFormat::Grayscale).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    assert_eq!(loaded.width, width);
    assert_eq!(loaded.height, height);
    // Loaded as RGB since BMP is 24-bit
    assert_eq!(loaded.pixel_format, PixelFormat::Rgb);
    // Each RGB triple should be (gray, gray, gray)
    for (i, &g) in gray.iter().enumerate() {
        assert_eq!(loaded.pixels[i * 3], g, "red mismatch at pixel {}", i);
        assert_eq!(loaded.pixels[i * 3 + 1], g, "green mismatch at pixel {}", i);
        assert_eq!(loaded.pixels[i * 3 + 2], g, "blue mismatch at pixel {}", i);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn ppm_roundtrip_rgb() {
    let path: PathBuf = temp_path("ppm_rgb.ppm");
    let width: usize = 16;
    let height: usize = 12;
    let pixels: Vec<u8> = make_test_rgb(width, height);

    save_ppm(&path, &pixels, width, height, PixelFormat::Rgb).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    assert_eq!(loaded.width, width);
    assert_eq!(loaded.height, height);
    assert_eq!(loaded.pixel_format, PixelFormat::Rgb);
    assert_eq!(loaded.pixels, pixels);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn ppm_roundtrip_grayscale() {
    let path: PathBuf = temp_path("pgm_gray.pgm");
    let width: usize = 10;
    let height: usize = 10;
    let gray: Vec<u8> = make_test_gray(width, height);

    save_ppm(&path, &gray, width, height, PixelFormat::Grayscale).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    assert_eq!(loaded.width, width);
    assert_eq!(loaded.height, height);
    assert_eq!(loaded.pixel_format, PixelFormat::Grayscale);
    assert_eq!(loaded.pixels, gray);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn auto_detect_format_from_content() {
    let width: usize = 4;
    let height: usize = 4;
    let pixels: Vec<u8> = make_test_rgb(width, height);

    // Save as BMP and PPM to different files
    let bmp_path: PathBuf = temp_path("detect.bmp");
    let ppm_path: PathBuf = temp_path("detect.ppm");

    save_bmp(&bmp_path, &pixels, width, height, PixelFormat::Rgb).unwrap();
    save_ppm(&ppm_path, &pixels, width, height, PixelFormat::Rgb).unwrap();

    // Read each file's bytes and load from bytes to verify auto-detection
    let bmp_bytes: Vec<u8> = std::fs::read(&bmp_path).unwrap();
    let ppm_bytes: Vec<u8> = std::fs::read(&ppm_path).unwrap();

    let bmp_loaded: LoadedImage = load_image_from_bytes(&bmp_bytes).unwrap();
    let ppm_loaded: LoadedImage = load_image_from_bytes(&ppm_bytes).unwrap();

    assert_eq!(bmp_loaded.pixels, pixels);
    assert_eq!(ppm_loaded.pixels, pixels);

    let _ = std::fs::remove_file(&bmp_path);
    let _ = std::fs::remove_file(&ppm_path);
}

#[test]
fn error_on_unsupported_format() {
    // Bytes that don't match BMP or PPM header
    let garbage: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0x04];
    let result = load_image_from_bytes(&garbage);
    assert!(result.is_err());
}

#[test]
fn bmp_row_padding_odd_width() {
    // Width=3 means row size = 3*3=9 bytes, padded to 12 bytes (next 4-byte boundary).
    // This tests that padding is handled correctly.
    let path: PathBuf = temp_path("bmp_odd.bmp");
    let width: usize = 3;
    let height: usize = 5;
    let pixels: Vec<u8> = make_test_rgb(width, height);

    save_bmp(&path, &pixels, width, height, PixelFormat::Rgb).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    assert_eq!(loaded.width, width);
    assert_eq!(loaded.height, height);
    assert_eq!(loaded.pixel_format, PixelFormat::Rgb);
    assert_eq!(loaded.pixels, pixels);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn bmp_roundtrip_bgra_32bit() {
    // 32-bit BMP with BGRA pixel format
    let path: PathBuf = temp_path("bmp_bgra.bmp");
    let width: usize = 8;
    let height: usize = 6;
    let rgba_pixels: Vec<u8> = make_test_rgba(width, height);

    // Save as RGBA — the save function should write a 32-bit BMP
    save_bmp(&path, &rgba_pixels, width, height, PixelFormat::Rgba).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    assert_eq!(loaded.width, width);
    assert_eq!(loaded.height, height);
    assert_eq!(loaded.pixel_format, PixelFormat::Rgba);
    assert_eq!(loaded.pixels, rgba_pixels);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn bmp_row_padding_width_1() {
    // Width=1 means row size = 1*3=3 bytes, padded to 4 bytes.
    let path: PathBuf = temp_path("bmp_w1.bmp");
    let width: usize = 1;
    let height: usize = 3;
    let pixels: Vec<u8> = make_test_rgb(width, height);

    save_bmp(&path, &pixels, width, height, PixelFormat::Rgb).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    assert_eq!(loaded.width, width);
    assert_eq!(loaded.height, height);
    assert_eq!(loaded.pixels, pixels);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn ppm_roundtrip_different_sizes() {
    // Triangulation: test with a different size to ensure no hardcoding
    for &(w, h) in &[(1, 1), (7, 3), (100, 50)] {
        let path: PathBuf = temp_path(&format!("ppm_{}x{}.ppm", w, h));
        let pixels: Vec<u8> = make_test_rgb(w, h);

        save_ppm(&path, &pixels, w, h, PixelFormat::Rgb).unwrap();
        let loaded: LoadedImage = load_image(&path).unwrap();

        assert_eq!(loaded.width, w);
        assert_eq!(loaded.height, h);
        assert_eq!(loaded.pixels, pixels);

        let _ = std::fs::remove_file(&path);
    }
}

#[test]
fn bmp_from_bgr_pixel_format() {
    // BMP natively stores BGR. When saving Bgr pixel format, it should
    // write directly without conversion, and load back as Rgb.
    let path: PathBuf = temp_path("bmp_bgr.bmp");
    let width: usize = 4;
    let height: usize = 4;

    // Create BGR pixels
    let mut bgr_pixels: Vec<u8> = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let b: u8 = ((x * 11 + y * 41) % 256) as u8;
            let g: u8 = ((x * 59 + y * 7) % 256) as u8;
            let r: u8 = ((x * 37 + y * 13) % 256) as u8;
            bgr_pixels.push(b);
            bgr_pixels.push(g);
            bgr_pixels.push(r);
        }
    }

    save_bmp(&path, &bgr_pixels, width, height, PixelFormat::Bgr).unwrap();
    let loaded: LoadedImage = load_image(&path).unwrap();

    // Loaded image is always in Rgb format for 24-bit BMP
    assert_eq!(loaded.pixel_format, PixelFormat::Rgb);
    // Verify pixel values match after BGR→RGB conversion
    for i in 0..width * height {
        let b: u8 = bgr_pixels[i * 3];
        let g: u8 = bgr_pixels[i * 3 + 1];
        let r: u8 = bgr_pixels[i * 3 + 2];
        assert_eq!(loaded.pixels[i * 3], r);
        assert_eq!(loaded.pixels[i * 3 + 1], g);
        assert_eq!(loaded.pixels[i * 3 + 2], b);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn save_bmp_invalid_pixel_count() {
    let path: PathBuf = temp_path("bmp_invalid.bmp");
    let pixels: Vec<u8> = vec![0; 10]; // Wrong size for 4x4 RGB
    let result = save_bmp(&path, &pixels, 4, 4, PixelFormat::Rgb);
    assert!(result.is_err());
}

#[test]
fn save_ppm_invalid_pixel_count() {
    let path: PathBuf = temp_path("ppm_invalid.ppm");
    let pixels: Vec<u8> = vec![0; 10]; // Wrong size for 4x4 RGB
    let result = save_ppm(&path, &pixels, 4, 4, PixelFormat::Rgb);
    assert!(result.is_err());
}
