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

// -----------------------------------------------------------------------
// C djpeg cross-validation for BMP output
// -----------------------------------------------------------------------

/// Path to C djpeg binary, or `None` if not installed.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    std::process::Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

struct IoTempFile {
    path: PathBuf,
}

impl IoTempFile {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!(
                "ljt_imgio_{}_{}_{name}",
                std::process::id(),
                id
            )),
        }
    }
}

impl Drop for IoTempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Parse BMP header and extract raw pixel data (bottom-up 24-bit BGR → top-down RGB).
fn parse_bmp_pixels(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() >= 54, "BMP too short for header");
    assert_eq!(&data[0..2], b"BM", "not a BMP file");

    let pixel_offset: u32 = u32::from_le_bytes([data[10], data[11], data[12], data[13]]);
    let width: u32 = u32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let height_raw: i32 = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);
    let bits_per_pixel: u16 = u16::from_le_bytes([data[28], data[29]]);
    let height: u32 = height_raw.unsigned_abs();
    let bottom_up: bool = height_raw > 0;

    let bpp: usize = bits_per_pixel as usize / 8;
    let row_stride: usize = ((width as usize * bpp + 3) / 4) * 4;

    let mut pixels: Vec<u8> = Vec::with_capacity(width as usize * height as usize * 3);
    for row in 0..height as usize {
        let bmp_row: usize = if bottom_up {
            height as usize - 1 - row
        } else {
            row
        };
        let start: usize = pixel_offset as usize + bmp_row * row_stride;
        for x in 0..width as usize {
            let offset: usize = start + x * bpp;
            if offset + 2 < data.len() {
                // BMP stores BGR; convert to RGB
                pixels.push(data[offset + 2]); // R
                pixels.push(data[offset + 1]); // G
                pixels.push(data[offset]); // B
            }
        }
    }

    (width as usize, height as usize, pixels)
}

#[test]
fn c_djpeg_cross_validation_bmp_output() {
    let djpeg = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 32;
    let source_pixels: Vec<u8> = make_test_rgb(width, height);

    // Encode a JPEG from source pixels
    let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &source_pixels,
        width,
        height,
        PixelFormat::Rgb,
        95,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("compress failed");

    let tmp_jpg = IoTempFile::new("bmp_xval.jpg");
    let tmp_rust_bmp = IoTempFile::new("bmp_xval_rust.bmp");
    let tmp_c_bmp = IoTempFile::new("bmp_xval_c.bmp");
    std::fs::write(&tmp_jpg.path, &jpeg).expect("write temp JPEG");

    // Decode with Rust and save as BMP
    let rust_image = libjpeg_turbo_rs::decompress(&jpeg).expect("Rust decompress failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);
    save_bmp(
        &tmp_rust_bmp.path,
        &rust_image.data,
        rust_image.width,
        rust_image.height,
        PixelFormat::Rgb,
    )
    .expect("save_bmp failed");

    // Decode with C djpeg -bmp
    let output = std::process::Command::new(&djpeg)
        .arg("-bmp")
        .arg("-outfile")
        .arg(&tmp_c_bmp.path)
        .arg(&tmp_jpg.path)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg -bmp failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Read both BMP files
    let rust_bmp_data: Vec<u8> = std::fs::read(&tmp_rust_bmp.path).expect("read Rust BMP");
    let c_bmp_data: Vec<u8> = std::fs::read(&tmp_c_bmp.path).expect("read C BMP");

    // Try byte-for-byte comparison first
    if rust_bmp_data == c_bmp_data {
        // Perfect match
        return;
    }

    // If BMP headers differ, compare extracted pixel data
    let (r_w, r_h, rust_pixels) = parse_bmp_pixels(&rust_bmp_data);
    let (c_w, c_h, c_pixels) = parse_bmp_pixels(&c_bmp_data);

    assert_eq!(r_w, width, "Rust BMP width mismatch");
    assert_eq!(r_h, height, "Rust BMP height mismatch");
    assert_eq!(c_w, width, "C BMP width mismatch");
    assert_eq!(c_h, height, "C BMP height mismatch");

    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "pixel data length mismatch"
    );

    // Compare pixel data: diff should be 0 since both decode the same JPEG
    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_pixels.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  bmp pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
    }
    assert_eq!(
        max_diff, 0,
        "bmp cross-validation: {} pixels differ, max_diff={}",
        mismatch_count, max_diff
    );
}
