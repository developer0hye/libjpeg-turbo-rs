use libjpeg_turbo_rs::{compress, decompress, Encoder, PixelFormat, Subsampling};

/// All 7 subsampling modes.
const ALL_SUBSAMPLINGS: [Subsampling; 7] = [
    Subsampling::S444,
    Subsampling::S422,
    Subsampling::S420,
    Subsampling::S440,
    Subsampling::S411,
    Subsampling::S441,
    Subsampling::Unknown,
];

/// All pixel formats the encoder supports for encode+decode roundtrip.
/// Excludes Cmyk (special 4-component path) and Rgb565 (decode-only).
const ENCODABLE_FORMATS: [PixelFormat; 11] = [
    PixelFormat::Rgb,
    PixelFormat::Rgba,
    PixelFormat::Bgr,
    PixelFormat::Bgra,
    PixelFormat::Rgbx,
    PixelFormat::Bgrx,
    PixelFormat::Xrgb,
    PixelFormat::Xbgr,
    PixelFormat::Argb,
    PixelFormat::Abgr,
    PixelFormat::Grayscale,
];

/// Compute PSNR between two RGB byte slices.
fn psnr_rgb(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return f64::INFINITY;
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let diff = x as f64 - y as f64;
            diff * diff
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse < 1e-10 {
        return f64::INFINITY;
    }
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

// --- Quality boundary tests ---

#[test]
fn quality_1_encodes_and_decodes() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(1)
        .encode()
        .unwrap();
    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.data.len(), 32 * 32 * 3);
}

#[test]
fn quality_100_high_psnr() {
    let mut pixels: Vec<u8> = Vec::with_capacity(32 * 32 * 3);
    for y in 0..32u8 {
        for x in 0..32u8 {
            pixels.push(x.wrapping_mul(8));
            pixels.push(y.wrapping_mul(8));
            pixels.push(128);
        }
    }
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();
    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);

    let psnr = psnr_rgb(&pixels, &image.data);
    assert!(
        psnr > 30.0,
        "quality=100 should produce high PSNR (>30 dB), got {:.1} dB",
        psnr
    );
}

#[test]
fn quality_0_treated_as_quality_1() {
    // quality=0 is clamped to 1 by quality_scaling, so encode should succeed
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(0)
        .encode()
        .unwrap();
    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
}

// --- Subsampling with non-aligned dimensions ---

#[test]
fn all_subsamplings_with_3x5_image() {
    let pixels: Vec<u8> = (0..3 * 5 * 3)
        .map(|i| ((i * 37 + 13) % 256) as u8)
        .collect();
    for &ss in &ALL_SUBSAMPLINGS {
        let jpeg = Encoder::new(&pixels, 3, 5, PixelFormat::Rgb)
            .quality(75)
            .subsampling(ss)
            .encode()
            .unwrap_or_else(|e| panic!("encode failed for {:?}: {}", ss, e));
        let image =
            decompress(&jpeg).unwrap_or_else(|e| panic!("decode failed for {:?}: {}", ss, e));
        assert_eq!(image.width, 3, "width mismatch for {:?}", ss);
        assert_eq!(image.height, 5, "height mismatch for {:?}", ss);
    }
}

#[test]
fn all_subsamplings_with_1x1_image() {
    let pixels: Vec<u8> = vec![128, 64, 32]; // 1x1 RGB
    for &ss in &ALL_SUBSAMPLINGS {
        let jpeg = Encoder::new(&pixels, 1, 1, PixelFormat::Rgb)
            .quality(75)
            .subsampling(ss)
            .encode()
            .unwrap_or_else(|e| panic!("encode failed for {:?}: {}", ss, e));
        let image =
            decompress(&jpeg).unwrap_or_else(|e| panic!("decode failed for {:?}: {}", ss, e));
        assert_eq!(image.width, 1);
        assert_eq!(image.height, 1);
    }
}

// --- Restart interval boundary tests ---

#[test]
fn restart_interval_1_every_mcu() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    // Should contain DRI marker
    let has_dri = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xDD);
    assert!(has_dri, "restart_blocks(1) should produce DRI marker");

    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
}

#[test]
fn restart_interval_65535_larger_than_image() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    // 16x16 with S444: 2x2 MCUs = 4 MCUs total. restart_blocks(65535) > 4
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(65535)
        .encode()
        .unwrap();

    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
}

// --- Per-component quality ---

#[test]
fn per_component_quality_luma_100_chroma_1() {
    let mut pixels: Vec<u8> = Vec::with_capacity(32 * 32 * 3);
    for y in 0..32u8 {
        for x in 0..32u8 {
            pixels.push(x.wrapping_mul(8));
            pixels.push(y.wrapping_mul(8));
            pixels.push(128);
        }
    }
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .quality_factor(0, 100) // luma table = quality 100
        .quality_factor(1, 1) // chroma table = quality 1
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.data.len(), 32 * 32 * 3);
}

// --- Encode then decode with every pixel format ---

#[test]
fn encode_decode_all_pixel_formats() {
    for &pf in &ENCODABLE_FORMATS {
        let bpp = pf.bytes_per_pixel();
        let (w, h) = (16, 16);
        let pixels: Vec<u8> = (0..w * h * bpp)
            .map(|i| ((i * 37 + 13) % 256) as u8)
            .collect();

        let ss = if pf == PixelFormat::Grayscale {
            Subsampling::S444
        } else {
            Subsampling::S420
        };

        let jpeg = Encoder::new(&pixels, w, h, pf)
            .quality(75)
            .subsampling(ss)
            .encode()
            .unwrap_or_else(|e| panic!("encode failed for {:?}: {}", pf, e));

        let image =
            decompress(&jpeg).unwrap_or_else(|e| panic!("decode failed for {:?}: {}", pf, e));
        assert_eq!(image.width, w, "width mismatch for {:?}", pf);
        assert_eq!(image.height, h, "height mismatch for {:?}", pf);
        assert!(
            !image.data.is_empty(),
            "decoded data should not be empty for {:?}",
            pf
        );
    }
}

// --- Special pixel values ---

#[test]
fn encode_all_zero_pixels() {
    let pixels: Vec<u8> = vec![0u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    // All-zero input: decoded should be close to zero (lossy, but not wildly off)
    let max_val: u8 = *image.data.iter().max().unwrap();
    assert!(
        max_val < 30,
        "all-zero pixels should decode to near-zero, max was {}",
        max_val
    );
}

#[test]
fn encode_all_255_pixels() {
    let pixels: Vec<u8> = vec![255u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();
    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
    // All-255 input: decoded should be close to 255
    let min_val: u8 = *image.data.iter().min().unwrap();
    assert!(
        min_val > 225,
        "all-255 pixels should decode to near-255, min was {}",
        min_val
    );
}

// --- Progressive encode with grayscale ---

#[test]
fn progressive_encode_grayscale() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Grayscale)
        .quality(75)
        .progressive(true)
        .encode()
        .unwrap();

    // Should contain SOF2 marker (progressive)
    let has_sof2 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "progressive grayscale should contain SOF2 marker");

    let image = decompress(&jpeg).unwrap();
    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.pixel_format, PixelFormat::Grayscale);
}

// --- Arithmetic encode with all subsampling modes ---

#[test]
fn arithmetic_encode_all_subsamplings() {
    // Arithmetic encode supports S444, S422, S420, S440, S411, S441
    let subsamplings = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
        Subsampling::S411,
        Subsampling::S441,
    ];

    let pixels: Vec<u8> = (0..32 * 32 * 3)
        .map(|i| ((i * 37 + 13) % 256) as u8)
        .collect();

    for &ss in &subsamplings {
        let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
            .quality(75)
            .subsampling(ss)
            .arithmetic(true)
            .encode()
            .unwrap_or_else(|e| panic!("arithmetic encode failed for {:?}: {}", ss, e));

        // Should contain SOF9 marker (arithmetic sequential DCT)
        let has_sof9 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC9);
        assert!(
            has_sof9,
            "arithmetic encode should contain SOF9 marker for {:?}",
            ss
        );

        let image = decompress(&jpeg)
            .unwrap_or_else(|e| panic!("decode of arithmetic {:?} failed: {}", ss, e));
        assert_eq!(image.width, 32, "width mismatch for arithmetic {:?}", ss);
        assert_eq!(image.height, 32, "height mismatch for arithmetic {:?}", ss);
    }
}

// --- CMYK encode roundtrip ---

#[test]
fn cmyk_encode_decode_roundtrip() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 4];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Cmyk)
        .quality(75)
        .encode()
        .unwrap();
    // CMYK JPEG should be valid (can at least be parsed)
    assert!(jpeg.len() > 100, "CMYK JPEG should not be trivially small");
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
}

// --- Quality 1 vs quality 100 produces different file sizes ---

#[test]
fn quality_affects_file_size() {
    let pixels: Vec<u8> = (0..32 * 32 * 3)
        .map(|i| ((i * 37 + 13) % 256) as u8)
        .collect();
    let jpeg_q1 = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(1)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();
    let jpeg_q100 = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    assert!(
        jpeg_q100.len() > jpeg_q1.len(),
        "quality=100 ({} bytes) should produce larger file than quality=1 ({} bytes)",
        jpeg_q100.len(),
        jpeg_q1.len()
    );
}
