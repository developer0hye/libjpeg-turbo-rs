//! Decompression cross-product test — port of C's tjdecomptest.in.
//!
//! Tests decoding with all parameter combinations:
//! - subsampling: [444, 422, 440, 420, 411, 441, gray]
//! - crop: [none, 14x14+23+23, 21x21+4+4, 18x18+13+13, 21x21+0+0, 24x26+20+18]
//! - scale: [1/1, 1/2, 1/4, 1/8]
//! - nosmooth (fast upsample): [off, on] (only for 422/420/440)
//! - dct: [default (islow), fast]
//! - output: [native, grayscale-from-color]
//!
//! Skip conditions match C behavior:
//! - crop + scale 1/4 or 1/8 → skip (C skips 3/8 and below)
//! - fast DCT + any scale → skip, unless scale=1/2 AND subsamp=420, or no scale
//! - nosmooth only for 422/420/440
//! - grayscale output only when nosmooth is off

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{
    compress, ColorSpace, CropRegion, DctMethod, Image, PixelFormat, ScalingFactor, Subsampling,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a deterministic 64x64 RGB test pattern.
fn generate_rgb_pattern(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 4 + y) % 256) as u8);
            pixels.push(((y * 4 + x) % 256) as u8);
            pixels.push(((x.wrapping_mul(y).wrapping_add(37)) % 256) as u8);
        }
    }
    pixels
}

/// Generate a deterministic 64x64 grayscale test pattern.
fn generate_gray_pattern(width: usize, height: usize) -> Vec<u8> {
    (0..width * height)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect()
}

/// Encode a 64x64 color JPEG with the given subsampling.
fn encode_color_jpeg(subsamp: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = generate_rgb_pattern(64, 64);
    compress(&pixels, 64, 64, PixelFormat::Rgb, 90, subsamp)
        .unwrap_or_else(|e| panic!("compress failed for {:?}: {:?}", subsamp, e))
}

/// Encode a 64x64 grayscale JPEG.
fn encode_gray_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = generate_gray_pattern(64, 64);
    compress(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
    )
    .unwrap_or_else(|e| panic!("compress grayscale failed: {:?}", e))
}

/// All crop regions from C's tjdecomptest.in, plus None for no crop.
fn crop_regions() -> Vec<Option<CropRegion>> {
    vec![
        None,
        Some(CropRegion {
            x: 23,
            y: 23,
            width: 14,
            height: 14,
        }),
        Some(CropRegion {
            x: 4,
            y: 4,
            width: 21,
            height: 21,
        }),
        Some(CropRegion {
            x: 13,
            y: 13,
            width: 18,
            height: 18,
        }),
        Some(CropRegion {
            x: 0,
            y: 0,
            width: 21,
            height: 21,
        }),
        Some(CropRegion {
            x: 20,
            y: 18,
            width: 24,
            height: 26,
        }),
    ]
}

/// Supported scaling factors: 1/1, 1/2, 1/4, 1/8.
fn scaling_factors() -> Vec<ScalingFactor> {
    vec![
        ScalingFactor::new(1, 1),
        ScalingFactor::new(1, 2),
        ScalingFactor::new(1, 4),
        ScalingFactor::new(1, 8),
    ]
}

/// Describes one test combination for diagnostic output.
fn combo_label(
    subsamp: &str,
    crop: &Option<CropRegion>,
    scale: &ScalingFactor,
    fast_upsample: bool,
    fast_dct: bool,
    output_mode: &str,
) -> String {
    let crop_str: String = match crop {
        None => "none".to_string(),
        Some(c) => format!("{}x{}+{}+{}", c.width, c.height, c.x, c.y),
    };
    format!(
        "subsamp={} crop={} scale={}/{} fast_up={} fast_dct={} output={}",
        subsamp, crop_str, scale.num, scale.denom, fast_upsample, fast_dct, output_mode
    )
}

/// Returns true if nosmooth (fast upsample) applies to this subsampling mode.
/// C only tests nosmooth for 422, 420, 440.
fn nosmooth_applicable(subsamp: Subsampling) -> bool {
    matches!(
        subsamp,
        Subsampling::S422 | Subsampling::S420 | Subsampling::S440
    )
}

/// Returns true if fast DCT should be skipped for this scale + subsampling.
///
/// C rule: fast DCT is only tested when:
/// - no scaling (scale = 1/1), OR
/// - scale = 1/2 AND subsampling is 420
fn should_skip_fast_dct(scale: &ScalingFactor, subsamp: Subsampling) -> bool {
    if scale.num == 1 && scale.denom == 1 {
        // No scaling — always test fast DCT
        return false;
    }
    if scale.num == 1 && scale.denom == 2 && subsamp == Subsampling::S420 {
        // scale 1/2 + 420 — test fast DCT
        return false;
    }
    // All other scale + subsamp combos — skip fast DCT
    true
}

/// Returns true if this crop + scale combination should be skipped.
/// C skips scales 1/8, 2/8 (1/4), 3/8 when crop is present.
/// Our supported scales in that range: 1/4 (denom=4) and 1/8 (denom=8).
fn should_skip_crop_scale(crop: &Option<CropRegion>, scale: &ScalingFactor) -> bool {
    crop.is_some() && scale.denom >= 4
}

// ---------------------------------------------------------------------------
// Main cross-product test
// ---------------------------------------------------------------------------

/// Decompression cross-product: subsampling x crop x scale x nosmooth x dct x output.
///
/// Port of C's tjdecomptest.in main loop. Verifies that every valid parameter
/// combination decodes without error and produces reasonable output.
#[test]
fn tjdecomptest_cross_product() {
    let color_subsampling_modes: Vec<Subsampling> = vec![
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];

    // Pre-encode test JPEGs (encode once, decode many times)
    let color_jpegs: Vec<(Subsampling, Vec<u8>)> = color_subsampling_modes
        .iter()
        .map(|&s| (s, encode_color_jpeg(s)))
        .collect();
    let gray_jpeg: Vec<u8> = encode_gray_jpeg();

    let crops: Vec<Option<CropRegion>> = crop_regions();
    let scales: Vec<ScalingFactor> = scaling_factors();

    let mut tested: u32 = 0;
    let mut skipped: u32 = 0;
    let mut failed: u32 = 0;
    let mut failures: Vec<String> = Vec::new();

    // ----- Color subsampling modes -----
    for (subsamp, jpeg) in &color_jpegs {
        let subsamp_name: &str = match subsamp {
            Subsampling::S444 => "444",
            Subsampling::S422 => "422",
            Subsampling::S440 => "440",
            Subsampling::S420 => "420",
            Subsampling::S411 => "411",
            Subsampling::S441 => "441",
            _ => "unknown",
        };

        for crop in &crops {
            for scale in &scales {
                // Skip small scale + crop (matching C)
                if should_skip_crop_scale(crop, scale) {
                    skipped += 1;
                    continue;
                }

                for use_fast_upsample in [false, true] {
                    // nosmooth only for 422/420/440
                    if use_fast_upsample && !nosmooth_applicable(*subsamp) {
                        continue;
                    }

                    for use_fast_dct in [false, true] {
                        // Fast DCT skip conditions (matching C)
                        if use_fast_dct && should_skip_fast_dct(scale, *subsamp) {
                            skipped += 1;
                            continue;
                        }

                        // --- Native output (RGB) ---
                        let label: String = combo_label(
                            subsamp_name,
                            crop,
                            scale,
                            use_fast_upsample,
                            use_fast_dct,
                            "rgb",
                        );
                        match try_decode(jpeg, *scale, crop, use_fast_upsample, use_fast_dct, None)
                        {
                            Ok(img) => {
                                verify_dimensions(&img, scale, crop, 64, 64, &label);
                            }
                            Err(e) => {
                                failed += 1;
                                failures.push(format!("FAIL [{}]: {:?}", label, e));
                            }
                        }
                        tested += 1;

                        // --- Grayscale output from color JPEG ---
                        // C only tests grayscale output when nosmooth is off
                        if !use_fast_upsample {
                            let label_gray: String = combo_label(
                                subsamp_name,
                                crop,
                                scale,
                                use_fast_upsample,
                                use_fast_dct,
                                "grayscale",
                            );
                            match try_decode(
                                jpeg,
                                *scale,
                                crop,
                                use_fast_upsample,
                                use_fast_dct,
                                Some(ColorSpace::Grayscale),
                            ) {
                                Ok(img) => {
                                    assert_eq!(
                                        img.pixel_format,
                                        PixelFormat::Grayscale,
                                        "expected grayscale output for {}",
                                        label_gray
                                    );
                                    verify_dimensions(&img, scale, crop, 64, 64, &label_gray);
                                }
                                Err(e) => {
                                    failed += 1;
                                    failures.push(format!("FAIL [{}]: {:?}", label_gray, e));
                                }
                            }
                            tested += 1;
                        }
                    }
                }
            }
        }
    }

    // ----- Grayscale JPEG -----
    {
        let subsamp_name: &str = "gray";
        for crop in &crops {
            for scale in &scales {
                if should_skip_crop_scale(crop, scale) {
                    skipped += 1;
                    continue;
                }

                // nosmooth does not apply to grayscale
                let use_fast_upsample: bool = false;

                for use_fast_dct in [false, true] {
                    // Fast DCT skip: for grayscale, apply same rule as non-420
                    // (only test fast DCT at no-scale)
                    if use_fast_dct && should_skip_fast_dct(scale, Subsampling::S444) {
                        skipped += 1;
                        continue;
                    }

                    // --- Native grayscale output ---
                    let label: String = combo_label(
                        subsamp_name,
                        crop,
                        scale,
                        use_fast_upsample,
                        use_fast_dct,
                        "native",
                    );
                    match try_decode(
                        &gray_jpeg,
                        *scale,
                        crop,
                        use_fast_upsample,
                        use_fast_dct,
                        None,
                    ) {
                        Ok(img) => {
                            assert_eq!(
                                img.pixel_format,
                                PixelFormat::Grayscale,
                                "gray JPEG should decode as grayscale for {}",
                                label
                            );
                            verify_dimensions(&img, scale, crop, 64, 64, &label);
                        }
                        Err(e) => {
                            failed += 1;
                            failures.push(format!("FAIL [{}]: {:?}", label, e));
                        }
                    }
                    tested += 1;

                    // --- RGB output from grayscale JPEG ---
                    // C tests -r flag (RGB output from gray).
                    // Uses set_output_format(Rgb) rather than set_output_colorspace,
                    // because the decoder does not support colorspace override to Rgb
                    // from a grayscale source.
                    let label_rgb: String = combo_label(
                        subsamp_name,
                        crop,
                        scale,
                        use_fast_upsample,
                        use_fast_dct,
                        "rgb-from-gray",
                    );
                    match try_decode_with_format(
                        &gray_jpeg,
                        *scale,
                        crop,
                        use_fast_upsample,
                        use_fast_dct,
                        Some(PixelFormat::Rgb),
                    ) {
                        Ok(img) => {
                            // When requesting RGB from grayscale, output should be RGB
                            assert_eq!(
                                img.pixel_format,
                                PixelFormat::Rgb,
                                "expected RGB output for {}",
                                label_rgb
                            );
                            verify_dimensions(&img, scale, crop, 64, 64, &label_rgb);
                        }
                        Err(e) => {
                            failed += 1;
                            failures.push(format!("FAIL [{}]: {:?}", label_rgb, e));
                        }
                    }
                    tested += 1;
                }
            }
        }
    }

    // Print summary
    eprintln!(
        "Decompress cross-product: {} tested, {} skipped, {} failed",
        tested, skipped, failed
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("  {}", f);
        }
    }
    assert_eq!(
        failed, 0,
        "{} of {} combinations failed (see stderr for details)",
        failed, tested
    );
}

/// Multi-format output cross-product.
///
/// For each color subsampling mode x scale, decode to every supported output pixel
/// format. Also tests grayscale JPEG to Grayscale and RGB output formats.
/// This is the widest format coverage: 6 subsampling x 4 scales x 10 formats = 240
/// plus grayscale JPEG variants.
#[test]
fn tjdecomptest_output_format_cross_product() {
    let color_subsampling_modes: Vec<Subsampling> = vec![
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];

    // Grayscale output from color JPEG requires set_output_colorspace, not
    // set_output_format, so it is tested separately in the grayscale cross-product.
    let output_formats: Vec<PixelFormat> = vec![
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Rgbx,
        PixelFormat::Bgrx,
        PixelFormat::Xrgb,
        PixelFormat::Xbgr,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];

    let scales: Vec<ScalingFactor> = scaling_factors();

    let mut tested: u32 = 0;
    let mut failed: u32 = 0;
    let mut failures: Vec<String> = Vec::new();

    for subsamp in &color_subsampling_modes {
        let jpeg: Vec<u8> = encode_color_jpeg(*subsamp);

        for scale in &scales {
            for format in &output_formats {
                let label: String = format!(
                    "subsamp={:?} scale={}/{} format={:?}",
                    subsamp, scale.num, scale.denom, format
                );

                match try_decode_with_format(&jpeg, *scale, &None, false, false, Some(*format)) {
                    Ok(img) => {
                        assert_eq!(
                            img.pixel_format, *format,
                            "pixel format mismatch for {}",
                            label
                        );
                        let expected_len: usize = img.width * img.height * format.bytes_per_pixel();
                        assert_eq!(
                            img.data.len(),
                            expected_len,
                            "data length mismatch for {}",
                            label
                        );
                        let expected_w: usize = scale.scale_dim(64);
                        let expected_h: usize = scale.scale_dim(64);
                        assert_eq!(img.width, expected_w, "width mismatch for {}", label);
                        assert_eq!(img.height, expected_h, "height mismatch for {}", label);
                    }
                    Err(e) => {
                        failed += 1;
                        failures.push(format!("FAIL [{}]: {:?}", label, e));
                    }
                }
                tested += 1;
            }
        }
    }

    // Also test grayscale JPEG to various formats at each scale
    let gray_jpeg: Vec<u8> = encode_gray_jpeg();
    let gray_formats: Vec<PixelFormat> = vec![PixelFormat::Grayscale, PixelFormat::Rgb];
    for scale in &scales {
        for format in &gray_formats {
            let label: String = format!(
                "subsamp=gray scale={}/{} format={:?}",
                scale.num, scale.denom, format
            );
            match try_decode_with_format(&gray_jpeg, *scale, &None, false, false, Some(*format)) {
                Ok(img) => {
                    assert_eq!(
                        img.pixel_format, *format,
                        "pixel format mismatch for {}",
                        label
                    );
                }
                Err(e) => {
                    failed += 1;
                    failures.push(format!("FAIL [{}]: {:?}", label, e));
                }
            }
            tested += 1;
        }
    }

    eprintln!(
        "Output format cross-product: {} tested, {} failed",
        tested, failed
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("  {}", f);
        }
    }
    assert_eq!(
        failed, 0,
        "{} of {} format combinations failed",
        failed, tested
    );
}

/// Grayscale output from color JPEG cross-product.
///
/// For each subsampling x crop x scale combination, decode a color JPEG to
/// grayscale and verify the output is single-channel with correct dimensions.
/// 6 subsampling x 6 crops x 4 scales (with crop+scale skips) = ~96 combinations.
#[test]
fn tjdecomptest_grayscale_output_cross_product() {
    let color_subsampling_modes: Vec<Subsampling> = vec![
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let scales: Vec<ScalingFactor> = scaling_factors();
    let crops: Vec<Option<CropRegion>> = crop_regions();

    let mut tested: u32 = 0;
    let mut skipped: u32 = 0;
    let mut failed: u32 = 0;
    let mut failures: Vec<String> = Vec::new();

    for subsamp in &color_subsampling_modes {
        let jpeg: Vec<u8> = encode_color_jpeg(*subsamp);

        for crop in &crops {
            for scale in &scales {
                // Apply same crop+scale skip as main test
                if should_skip_crop_scale(crop, scale) {
                    skipped += 1;
                    continue;
                }

                let crop_str: String = match crop {
                    None => "none".to_string(),
                    Some(c) => format!("{}x{}+{}+{}", c.width, c.height, c.x, c.y),
                };
                let label: String = format!(
                    "subsamp={:?} crop={} scale={}/{} -> grayscale",
                    subsamp, crop_str, scale.num, scale.denom
                );

                match try_decode(
                    &jpeg,
                    *scale,
                    crop,
                    false,
                    false,
                    Some(ColorSpace::Grayscale),
                ) {
                    Ok(img) => {
                        assert_eq!(
                            img.pixel_format,
                            PixelFormat::Grayscale,
                            "expected grayscale output for {}",
                            label
                        );
                        assert!(
                            img.width > 0 && img.height > 0,
                            "zero dimensions for {}",
                            label
                        );
                        assert_eq!(
                            img.data.len(),
                            img.width * img.height,
                            "data length mismatch for {}",
                            label
                        );
                        // Without crop, verify exact scaled dimensions
                        if crop.is_none() {
                            let expected_w: usize = scale.scale_dim(64);
                            let expected_h: usize = scale.scale_dim(64);
                            assert_eq!(img.width, expected_w, "width mismatch for {}", label);
                            assert_eq!(img.height, expected_h, "height mismatch for {}", label);
                        }
                    }
                    Err(e) => {
                        failed += 1;
                        failures.push(format!("FAIL [{}]: {:?}", label, e));
                    }
                }
                tested += 1;
            }
        }
    }

    eprintln!(
        "Grayscale output cross-product: {} tested, {} skipped, {} failed",
        tested, skipped, failed
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("  {}", f);
        }
    }
    assert_eq!(
        failed, 0,
        "{} of {} grayscale combinations failed",
        failed, tested
    );
}

/// Decoder toggles cross-product: block_smoothing x merged_upsample x subsampling x scale.
///
/// Tests additional decoder options that C's tjdecomptest.in exercises implicitly.
/// Block smoothing and merged upsample are independent decoder toggles that affect
/// the decode path. 6 subsamp x 4 scales x 4 toggle combos = 96, plus gray variants.
#[test]
fn tjdecomptest_decoder_toggles_cross_product() {
    let color_subsampling_modes: Vec<Subsampling> = vec![
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let scales: Vec<ScalingFactor> = scaling_factors();

    let mut tested: u32 = 0;
    let mut failed: u32 = 0;
    let mut failures: Vec<String> = Vec::new();

    for subsamp in &color_subsampling_modes {
        let jpeg: Vec<u8> = encode_color_jpeg(*subsamp);

        for scale in &scales {
            for block_smoothing in [false, true] {
                for merged_upsample in [false, true] {
                    // Merged upsample only applies to H2V1 (422) and H2V2 (420)
                    if merged_upsample && !matches!(*subsamp, Subsampling::S422 | Subsampling::S420)
                    {
                        continue;
                    }

                    let label: String = format!(
                        "subsamp={:?} scale={}/{} block_smooth={} merged_up={}",
                        subsamp, scale.num, scale.denom, block_smoothing, merged_upsample
                    );

                    match try_decode_toggles(&jpeg, *scale, block_smoothing, merged_upsample) {
                        Ok(img) => {
                            assert!(
                                img.width > 0 && img.height > 0,
                                "zero dimensions for {}",
                                label
                            );
                            let expected_w: usize = scale.scale_dim(64);
                            let expected_h: usize = scale.scale_dim(64);
                            assert_eq!(img.width, expected_w, "width mismatch for {}", label);
                            assert_eq!(img.height, expected_h, "height mismatch for {}", label);
                        }
                        Err(e) => {
                            failed += 1;
                            failures.push(format!("FAIL [{}]: {:?}", label, e));
                        }
                    }
                    tested += 1;
                }
            }
        }
    }

    // Also test grayscale JPEG with block smoothing toggle
    let gray_jpeg: Vec<u8> = encode_gray_jpeg();
    for scale in &scales {
        for block_smoothing in [false, true] {
            let label: String = format!(
                "subsamp=gray scale={}/{} block_smooth={}",
                scale.num, scale.denom, block_smoothing
            );

            match try_decode_toggles(&gray_jpeg, *scale, block_smoothing, false) {
                Ok(img) => {
                    assert!(
                        img.width > 0 && img.height > 0,
                        "zero dimensions for {}",
                        label
                    );
                }
                Err(e) => {
                    failed += 1;
                    failures.push(format!("FAIL [{}]: {:?}", label, e));
                }
            }
            tested += 1;
        }
    }

    eprintln!(
        "Decoder toggles cross-product: {} tested, {} failed",
        tested, failed
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("  {}", f);
        }
    }
    assert_eq!(
        failed, 0,
        "{} of {} decoder toggle combinations failed",
        failed, tested
    );
}

/// DCT method cross-product: islow x ifast x float x subsampling x scale.
///
/// Tests all three DCT methods across all subsampling modes and scales.
/// 6 subsamp x 4 scales x 3 dct_methods = 72, plus gray variants.
#[test]
fn tjdecomptest_dct_method_cross_product() {
    let color_subsampling_modes: Vec<Subsampling> = vec![
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];
    let scales: Vec<ScalingFactor> = scaling_factors();
    let dct_methods: Vec<DctMethod> = vec![DctMethod::IsLow, DctMethod::IsFast, DctMethod::Float];

    let mut tested: u32 = 0;
    let mut failed: u32 = 0;
    let mut failures: Vec<String> = Vec::new();

    for subsamp in &color_subsampling_modes {
        let jpeg: Vec<u8> = encode_color_jpeg(*subsamp);

        for scale in &scales {
            for dct in &dct_methods {
                let label: String = format!(
                    "subsamp={:?} scale={}/{} dct={:?}",
                    subsamp, scale.num, scale.denom, dct
                );

                match try_decode_dct_method(&jpeg, *scale, *dct) {
                    Ok(img) => {
                        assert!(
                            img.width > 0 && img.height > 0,
                            "zero dimensions for {}",
                            label
                        );
                        let expected_w: usize = scale.scale_dim(64);
                        let expected_h: usize = scale.scale_dim(64);
                        assert_eq!(img.width, expected_w, "width mismatch for {}", label);
                        assert_eq!(img.height, expected_h, "height mismatch for {}", label);
                    }
                    Err(e) => {
                        failed += 1;
                        failures.push(format!("FAIL [{}]: {:?}", label, e));
                    }
                }
                tested += 1;
            }
        }
    }

    // Grayscale JPEG with each DCT method
    let gray_jpeg: Vec<u8> = encode_gray_jpeg();
    for scale in &scales {
        for dct in &dct_methods {
            let label: String = format!(
                "subsamp=gray scale={}/{} dct={:?}",
                scale.num, scale.denom, dct
            );

            match try_decode_dct_method(&gray_jpeg, *scale, *dct) {
                Ok(img) => {
                    assert!(
                        img.width > 0 && img.height > 0,
                        "zero dimensions for {}",
                        label
                    );
                }
                Err(e) => {
                    failed += 1;
                    failures.push(format!("FAIL [{}]: {:?}", label, e));
                }
            }
            tested += 1;
        }
    }

    eprintln!(
        "DCT method cross-product: {} tested, {} failed",
        tested, failed
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("  {}", f);
        }
    }
    assert_eq!(
        failed, 0,
        "{} of {} DCT method combinations failed",
        failed, tested
    );
}

/// Crop x format cross-product.
///
/// For each subsampling x crop (with no-scale only), decode to multiple output
/// formats. Ensures crop interacts correctly with pixel format conversion.
/// 6 subsamp x 6 crops x 5 formats = 180 combinations.
#[test]
fn tjdecomptest_crop_format_cross_product() {
    let color_subsampling_modes: Vec<Subsampling> = vec![
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
        Subsampling::S411,
        Subsampling::S441,
    ];

    let output_formats: Vec<PixelFormat> = vec![
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Argb,
    ];

    let crops: Vec<Option<CropRegion>> = crop_regions();
    let scale_1x: ScalingFactor = ScalingFactor::new(1, 1);

    let mut tested: u32 = 0;
    let mut failed: u32 = 0;
    let mut failures: Vec<String> = Vec::new();

    for subsamp in &color_subsampling_modes {
        let jpeg: Vec<u8> = encode_color_jpeg(*subsamp);

        for crop in &crops {
            for format in &output_formats {
                let crop_str: String = match crop {
                    None => "none".to_string(),
                    Some(c) => format!("{}x{}+{}+{}", c.width, c.height, c.x, c.y),
                };
                let label: String = format!(
                    "subsamp={:?} crop={} format={:?}",
                    subsamp, crop_str, format
                );

                match try_decode_with_format(&jpeg, scale_1x, crop, false, false, Some(*format)) {
                    Ok(img) => {
                        assert_eq!(
                            img.pixel_format, *format,
                            "pixel format mismatch for {}",
                            label
                        );
                        assert!(
                            img.width > 0 && img.height > 0,
                            "zero dimensions for {}",
                            label
                        );
                        let expected_len: usize = img.width * img.height * format.bytes_per_pixel();
                        assert_eq!(
                            img.data.len(),
                            expected_len,
                            "data length mismatch for {}",
                            label
                        );
                    }
                    Err(e) => {
                        failed += 1;
                        failures.push(format!("FAIL [{}]: {:?}", label, e));
                    }
                }
                tested += 1;
            }
        }
    }

    eprintln!(
        "Crop x format cross-product: {} tested, {} failed",
        tested, failed
    );
    if !failures.is_empty() {
        for f in &failures {
            eprintln!("  {}", f);
        }
    }
    assert_eq!(
        failed, 0,
        "{} of {} crop x format combinations failed",
        failed, tested
    );
}

// ---------------------------------------------------------------------------
// Decode helpers
// ---------------------------------------------------------------------------

/// Attempt to decode a JPEG with the given parameter combination.
fn try_decode(
    jpeg: &[u8],
    scale: ScalingFactor,
    crop: &Option<CropRegion>,
    fast_upsample: bool,
    fast_dct: bool,
    output_colorspace: Option<ColorSpace>,
) -> libjpeg_turbo_rs::Result<Image> {
    let mut decoder: Decoder = Decoder::new(jpeg)?;

    if scale.num != 1 || scale.denom != 1 {
        decoder.set_scale(scale);
    }

    if let Some(c) = crop {
        decoder.set_crop_region(c.x, c.y, c.width, c.height);
    }

    if fast_upsample {
        decoder.set_fast_upsample(true);
    }

    if fast_dct {
        decoder.set_fast_dct(true);
        decoder.set_dct_method(DctMethod::IsFast);
    }

    if let Some(cs) = output_colorspace {
        decoder.set_output_colorspace(cs);
    }

    decoder.decode_image()
}

/// Attempt to decode a JPEG with the given parameters, using output pixel format
/// instead of output colorspace (needed for gray JPEG -> RGB output).
fn try_decode_with_format(
    jpeg: &[u8],
    scale: ScalingFactor,
    crop: &Option<CropRegion>,
    fast_upsample: bool,
    fast_dct: bool,
    output_format: Option<PixelFormat>,
) -> libjpeg_turbo_rs::Result<Image> {
    let mut decoder: Decoder = Decoder::new(jpeg)?;

    if scale.num != 1 || scale.denom != 1 {
        decoder.set_scale(scale);
    }

    if let Some(c) = crop {
        decoder.set_crop_region(c.x, c.y, c.width, c.height);
    }

    if fast_upsample {
        decoder.set_fast_upsample(true);
    }

    if fast_dct {
        decoder.set_fast_dct(true);
        decoder.set_dct_method(DctMethod::IsFast);
    }

    if let Some(fmt) = output_format {
        decoder.set_output_format(fmt);
    }

    decoder.decode_image()
}

/// Attempt to decode with block_smoothing and merged_upsample toggles.
fn try_decode_toggles(
    jpeg: &[u8],
    scale: ScalingFactor,
    block_smoothing: bool,
    merged_upsample: bool,
) -> libjpeg_turbo_rs::Result<Image> {
    let mut decoder: Decoder = Decoder::new(jpeg)?;

    if scale.num != 1 || scale.denom != 1 {
        decoder.set_scale(scale);
    }

    decoder.set_block_smoothing(block_smoothing);
    decoder.set_merged_upsample(merged_upsample);

    decoder.decode_image()
}

/// Attempt to decode with a specific DCT method.
fn try_decode_dct_method(
    jpeg: &[u8],
    scale: ScalingFactor,
    dct_method: DctMethod,
) -> libjpeg_turbo_rs::Result<Image> {
    let mut decoder: Decoder = Decoder::new(jpeg)?;

    if scale.num != 1 || scale.denom != 1 {
        decoder.set_scale(scale);
    }

    decoder.set_dct_method(dct_method);
    if dct_method == DctMethod::IsFast {
        decoder.set_fast_dct(true);
    }

    decoder.decode_image()
}

/// Verify decoded image dimensions are reasonable for the given parameters.
fn verify_dimensions(
    img: &Image,
    scale: &ScalingFactor,
    crop: &Option<CropRegion>,
    orig_w: usize,
    orig_h: usize,
    label: &str,
) {
    assert!(
        img.width > 0 && img.height > 0,
        "zero dimensions for {}",
        label
    );
    assert!(!img.data.is_empty(), "empty data for {}", label);

    let bpp: usize = img.pixel_format.bytes_per_pixel();
    let expected_data_len: usize = img.width * img.height * bpp;
    assert_eq!(
        img.data.len(),
        expected_data_len,
        "data length mismatch ({}x{}x{} != {}) for {}",
        img.width,
        img.height,
        bpp,
        img.data.len(),
        label
    );

    if crop.is_none() {
        // Without crop, dimensions should match scaled original
        let expected_w: usize = scale.scale_dim(orig_w);
        let expected_h: usize = scale.scale_dim(orig_h);
        assert_eq!(
            img.width, expected_w,
            "width mismatch (expected {}, got {}) for {}",
            expected_w, img.width, label
        );
        assert_eq!(
            img.height, expected_h,
            "height mismatch (expected {}, got {}) for {}",
            expected_h, img.height, label
        );
    } else {
        // With crop, output dimensions should be <= crop region
        // (may be smaller due to clamping to image bounds)
        let c: &CropRegion = crop.as_ref().unwrap();
        let scaled_w: usize = scale.scale_dim(orig_w);
        let scaled_h: usize = scale.scale_dim(orig_h);
        // Crop coordinates are in original pixel space; the decoder handles
        // the mapping to scaled space internally. Just verify output is bounded.
        assert!(
            img.width <= scaled_w.max(c.width),
            "crop output width {} exceeds bounds for {}",
            img.width,
            label
        );
        assert!(
            img.height <= scaled_h.max(c.height),
            "crop output height {} exceeds bounds for {}",
            img.height,
            label
        );
    }
}
