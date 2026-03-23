use libjpeg_turbo_rs::{compress_progressive, decompress, PixelFormat, Subsampling};

#[test]
fn progressive_roundtrip_rgb_444() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn progressive_roundtrip_rgb_420() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn progressive_roundtrip_grayscale() {
    let pixels = vec![128u8; 64 * 64];
    let jpeg = compress_progressive(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn progressive_has_sof2_marker() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg =
        compress_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    let has_sof2 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "progressive JPEG should contain SOF2 marker");
}

fn gradient_pixels(width: usize, height: usize, channels: usize) -> Vec<u8> {
    let mut pixels = vec![0u8; width * height * channels];
    for y in 0..height {
        for x in 0..width {
            let offset: usize = (y * width + x) * channels;
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            if channels >= 3 {
                pixels[offset] = r;
                pixels[offset + 1] = g;
                pixels[offset + 2] = b;
            } else {
                pixels[offset] = r;
            }
        }
    }
    pixels
}

#[test]
fn ac_refine_roundtrip_gradient_rgb_444() {
    let pixels = gradient_pixels(64, 64, 3);
    let jpeg =
        compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 90, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
}

#[test]
fn ac_refine_roundtrip_gradient_rgb_420() {
    let pixels = gradient_pixels(64, 64, 3);
    let jpeg =
        compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 85, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
}

#[test]
fn ac_refine_roundtrip_gradient_grayscale() {
    let pixels = gradient_pixels(64, 64, 1);
    let jpeg = compress_progressive(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 64);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

#[test]
fn ac_refine_produces_14_scans_rgb() {
    let pixels = gradient_pixels(32, 32, 3);
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let sos_count: usize = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    assert_eq!(sos_count, 14, "3-comp progressive should have 14 scans");
}

#[test]
fn ac_refine_produces_6_scans_grayscale() {
    let pixels = gradient_pixels(32, 32, 1);
    let jpeg = compress_progressive(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let sos_count: usize = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    assert_eq!(sos_count, 6, "grayscale progressive should have 6 scans");
}

#[test]
fn ac_refine_roundtrip_noise_pattern() {
    let mut pixels = vec![0u8; 48 * 48 * 3];
    let mut rng: u32 = 42;
    for pixel in pixels.iter_mut() {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        *pixel = ((rng >> 16) & 0xFF) as u8;
    }
    let jpeg =
        compress_progressive(&pixels, 48, 48, PixelFormat::Rgb, 95, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 48);
    assert_eq!(img.height, 48);
}
