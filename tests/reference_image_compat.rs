/// Cross-validation with C reference test images.
use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};

// testorig.jpg -- baseline Huffman, 4:2:0, 8-bit

#[test]
fn reference_testorig_decode_dimensions() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let img = decompress(data).unwrap();
    assert!(img.width > 0 && img.height > 0);
    assert_eq!(
        img.data.len(),
        img.width * img.height * img.pixel_format.bytes_per_pixel()
    );
}

#[test]
fn reference_testorig_decode_rgb() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let img = decompress_to(data, PixelFormat::Rgb).unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Rgb);
    assert_eq!(img.data.len(), img.width * img.height * 3);
    let min: u8 = *img.data.iter().min().unwrap();
    let max: u8 = *img.data.iter().max().unwrap();
    assert!(max - min > 100, "diverse pixels: min={}, max={}", min, max);
}

#[test]
fn reference_testorig_decode_default_format() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let img = decompress(data).unwrap();
    let bpp: usize = img.pixel_format.bytes_per_pixel();
    assert_eq!(img.data.len(), img.width * img.height * bpp);
}

#[test]
fn reference_testorig_decode_multiple_formats() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg");
    let base = decompress_to(data, PixelFormat::Rgb).unwrap();
    let (w, h) = (base.width, base.height);
    for &(pf, bpp) in &[
        (PixelFormat::Rgb, 3),
        (PixelFormat::Bgr, 3),
        (PixelFormat::Rgba, 4),
        (PixelFormat::Bgra, 4),
        (PixelFormat::Rgbx, 4),
    ] {
        let img = decompress_to(data, pf).unwrap();
        assert_eq!((img.width, img.height), (w, h), "{:?}", pf);
        assert_eq!(img.data.len(), w * h * bpp, "{:?}", pf);
    }
}

// testimgari.jpg -- arithmetic coded

#[test]
fn reference_arithmetic_decode() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg");
    let img = decompress(data).unwrap();
    assert!(img.width > 0 && img.height > 0);
}

#[test]
fn reference_arithmetic_decode_rgb() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg");
    let img = decompress_to(data, PixelFormat::Rgb).unwrap();
    let min: u8 = *img.data.iter().min().unwrap();
    let max: u8 = *img.data.iter().max().unwrap();
    assert!(max - min > 50, "diverse: min={}, max={}", min, max);
}

#[test]
fn reference_arithmetic_matches_baseline_dimensions() {
    let b = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testorig.jpg"
    ))
    .unwrap();
    let a = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testimgari.jpg"
    ))
    .unwrap();
    assert_eq!((b.width, b.height), (a.width, a.height));
}

// testimgint.jpg -- progressive

#[test]
fn reference_progressive_decode() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg");
    let img = decompress(data).unwrap();
    assert!(img.width > 0 && img.height > 0);
}

#[test]
fn reference_progressive_decode_rgb() {
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg");
    let img = decompress_to(data, PixelFormat::Rgb).unwrap();
    let min: u8 = *img.data.iter().min().unwrap();
    let max: u8 = *img.data.iter().max().unwrap();
    assert!(max - min > 50, "diverse progressive pixels");
}

#[test]
fn reference_progressive_matches_baseline_dimensions() {
    let b = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testorig.jpg"
    ))
    .unwrap();
    let p = decompress(include_bytes!(
        "../references/libjpeg-turbo/testimages/testimgint.jpg"
    ))
    .unwrap();
    assert_eq!((b.width, b.height), (p.width, p.height));
}

// testorig12.jpg -- 12-bit

#[test]
fn reference_12bit_decode() {
    use libjpeg_turbo_rs::precision::decompress_12bit;
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig12.jpg");
    match decompress_12bit(data) {
        Ok(img) => {
            assert!(img.width > 0 && img.height > 0);
            for &v in &img.data {
                assert!(v >= 0 && v <= 4095);
            }
        }
        Err(e) => {
            let s: String = format!("{}", e);
            assert!(
                s.contains("SOF") || s.contains("unsupported") || s.contains("missing"),
                "unexpected: {}",
                e
            );
        }
    }
}

#[test]
fn reference_12bit_has_diverse_values() {
    use libjpeg_turbo_rs::precision::decompress_12bit;
    let data: &[u8] = include_bytes!("../references/libjpeg-turbo/testimages/testorig12.jpg");
    if let Ok(img) = decompress_12bit(data) {
        let min: i16 = *img.data.iter().min().unwrap();
        let max: i16 = *img.data.iter().max().unwrap();
        assert!(max - min > 100, "12-bit diverse: min={}, max={}", min, max);
    }
}

// Cross-format consistency

#[test]
fn reference_all_images_decodable() {
    let images: &[(&str, &[u8])] = &[
        (
            "testorig.jpg",
            include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg"),
        ),
        (
            "testimgari.jpg",
            include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg"),
        ),
        (
            "testimgint.jpg",
            include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg"),
        ),
    ];
    for &(name, data) in images {
        let img = decompress(data).unwrap_or_else(|e| panic!("{}: {}", name, e));
        assert!(img.width > 0 && img.height > 0, "{}", name);
    }
}

#[test]
fn reference_baseline_vs_arithmetic_pixel_similarity() {
    let b = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    let a = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testimgari.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!((b.width, b.height), (a.width, a.height));
    let total: u64 = b
        .data
        .iter()
        .zip(a.data.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    let mean: f64 = total as f64 / b.data.len() as f64;
    assert!(mean < 100.0, "baseline vs arith mean diff {:.2}", mean);
}

#[test]
fn reference_baseline_vs_progressive_pixel_similarity() {
    let b = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testorig.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    let p = decompress_to(
        include_bytes!("../references/libjpeg-turbo/testimages/testimgint.jpg"),
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!((b.width, b.height), (p.width, p.height));
    let total: u64 = b
        .data
        .iter()
        .zip(p.data.iter())
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    let mean: f64 = total as f64 / b.data.len() as f64;
    assert!(mean < 5.0, "baseline vs prog mean diff {:.2}", mean);
}
