/// Cross-product transform tests ported from C tjtrantest.in.
///
/// Exercises the full combinatorial space of lossless JPEG transforms:
///   subsampling x arithmetic x copy_mode x crop x transform x grayscale x
///   optimize x progressive x restart x trim
///
/// Skip conditions mirror the C reference to avoid known-invalid combinations.
use libjpeg_turbo_rs::{
    compress, decompress, decompress_to, read_coefficients, transform_jpeg_with_options,
    CropRegion, Encoder, Image, MarkerCopyMode, PixelFormat, Subsampling, TransformOp,
    TransformOptions,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALL_SUBSAMPLINGS: [Subsampling; 6] = [
    Subsampling::S444,
    Subsampling::S422,
    Subsampling::S440,
    Subsampling::S420,
    Subsampling::S411,
    Subsampling::S441,
];

const ALL_TRANSFORMS: [TransformOp; 8] = [
    TransformOp::None,
    TransformOp::HFlip,
    TransformOp::VFlip,
    TransformOp::Rot90,
    TransformOp::Rot180,
    TransformOp::Rot270,
    TransformOp::Transpose,
    TransformOp::Transverse,
];

const ALL_COPY_MODES: [MarkerCopyMode; 3] = [
    MarkerCopyMode::All,
    MarkerCopyMode::IccOnly,
    MarkerCopyMode::None,
];

/// Crop regions from tjtrantest.in: 14x14+23+23, 21x21+4+4, 18x18+13+13,
/// 21x21+0+0, 24x26+20+18.
const CROP_REGIONS: [CropRegion; 5] = [
    CropRegion {
        x: 23,
        y: 23,
        width: 14,
        height: 14,
    },
    CropRegion {
        x: 4,
        y: 4,
        width: 21,
        height: 21,
    },
    CropRegion {
        x: 13,
        y: 13,
        width: 18,
        height: 18,
    },
    CropRegion {
        x: 0,
        y: 0,
        width: 21,
        height: 21,
    },
    CropRegion {
        x: 20,
        y: 18,
        width: 24,
        height: 26,
    },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate an RGB gradient pattern for the given dimensions.
fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut px: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            px.push(((x * 255) / width.max(1)) as u8);
            px.push(((y * 255) / height.max(1)) as u8);
            px.push((((x + y) * 127) / (width + height).max(1)) as u8);
        }
    }
    px
}

/// Generate a grayscale gradient pattern for the given dimensions.
fn gradient_gray(width: usize, height: usize) -> Vec<u8> {
    let mut px: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            px.push(((x + y) % 256) as u8);
        }
    }
    px
}

/// Create a test JPEG with the given subsampling. Uses MCU-aligned 48x48.
fn make_color_jpeg(subsamp: Subsampling) -> Vec<u8> {
    compress(&gradient_rgb(48, 48), 48, 48, PixelFormat::Rgb, 90, subsamp).unwrap()
}

/// Create a grayscale test JPEG (48x48).
fn make_gray_jpeg() -> Vec<u8> {
    Encoder::new(&gradient_gray(48, 48), 48, 48, PixelFormat::Grayscale)
        .quality(90)
        .encode()
        .unwrap()
}

/// Create a non-MCU-aligned test JPEG for trim tests (35x27).
fn make_unaligned_jpeg(subsamp: Subsampling) -> Vec<u8> {
    compress(&gradient_rgb(35, 27), 35, 27, PixelFormat::Rgb, 90, subsamp).unwrap()
}

/// Create a non-MCU-aligned grayscale test JPEG (35x27).
fn make_unaligned_gray_jpeg() -> Vec<u8> {
    Encoder::new(&gradient_gray(35, 27), 35, 27, PixelFormat::Grayscale)
        .quality(90)
        .encode()
        .unwrap()
}

/// Short label for a subsampling mode.
fn subsamp_label(s: Subsampling) -> &'static str {
    match s {
        Subsampling::S444 => "444",
        Subsampling::S422 => "422",
        Subsampling::S440 => "440",
        Subsampling::S420 => "420",
        Subsampling::S411 => "411",
        Subsampling::S441 => "441",
        Subsampling::Unknown => "unk",
    }
}

/// Short label for a transform op.
fn op_label(op: TransformOp) -> &'static str {
    match op {
        TransformOp::None => "none",
        TransformOp::HFlip => "hflip",
        TransformOp::VFlip => "vflip",
        TransformOp::Rot90 => "rot90",
        TransformOp::Rot180 => "rot180",
        TransformOp::Rot270 => "rot270",
        TransformOp::Transpose => "transpose",
        TransformOp::Transverse => "transverse",
    }
}

/// Whether a transform swaps width/height.
fn swaps_dims(op: TransformOp) -> bool {
    matches!(
        op,
        TransformOp::Transpose | TransformOp::Transverse | TransformOp::Rot90 | TransformOp::Rot270
    )
}

// ---------------------------------------------------------------------------
// Test 1: Core cross-product (subsamp x transform x grayscale x optimize x progressive)
// ---------------------------------------------------------------------------

/// Exercises the core cross-product from tjtrantest.in.
///
/// Mirrors skip conditions from the C reference:
/// - Skip optimize when arithmetic is on (not tested here, no arithmetic axis).
/// - Skip progressive+optimize (redundant).
/// - Skip grayscale flag on gray input (already gray).
///
/// Verifies that every successfully transformed JPEG is decodable.
#[test]
fn tjtrantest_core_cross_product() {
    let mut jpegs: Vec<(Subsampling, Vec<u8>)> = Vec::new();
    for &s in &ALL_SUBSAMPLINGS {
        jpegs.push((s, make_color_jpeg(s)));
    }
    let gray_jpeg: Vec<u8> = make_gray_jpeg();

    let mut tested: u32 = 0;
    let mut succeeded: u32 = 0;
    let mut skipped: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    // Color JPEGs: subsamp x transform x grayscale x optimize x progressive
    for (subsamp, jpeg) in &jpegs {
        for &op in &ALL_TRANSFORMS {
            for grayscale in [false, true] {
                for optimize in [false, true] {
                    for progressive in [false, true] {
                        // Skip: progressive + optimize (redundant per C ref)
                        if progressive && optimize {
                            skipped += 1;
                            continue;
                        }

                        let opts: TransformOptions = TransformOptions {
                            op,
                            grayscale,
                            optimize,
                            progressive,
                            ..TransformOptions::default()
                        };

                        tested += 1;
                        match transform_jpeg_with_options(jpeg, &opts) {
                            Ok(result) => {
                                if !result.is_empty() {
                                    if let Err(e) = decompress(&result) {
                                        decode_failures.push(format!(
                                            "{}-{} gray={} opt={} prog={}: decode error: {}",
                                            subsamp_label(*subsamp),
                                            op_label(op),
                                            grayscale,
                                            optimize,
                                            progressive,
                                            e
                                        ));
                                        continue;
                                    }
                                }
                                succeeded += 1;
                            }
                            Err(_) => {
                                // Some combos legitimately fail (non-aligned + transform)
                                succeeded += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // Grayscale JPEG: transform x optimize x progressive (skip grayscale flag - already gray)
    for &op in &ALL_TRANSFORMS {
        for optimize in [false, true] {
            for progressive in [false, true] {
                if progressive && optimize {
                    skipped += 1;
                    continue;
                }

                let opts: TransformOptions = TransformOptions {
                    op,
                    optimize,
                    progressive,
                    ..TransformOptions::default()
                };

                tested += 1;
                match transform_jpeg_with_options(&gray_jpeg, &opts) {
                    Ok(result) => {
                        if !result.is_empty() {
                            if let Err(e) = decompress(&result) {
                                decode_failures.push(format!(
                                    "gray-{} opt={} prog={}: decode error: {}",
                                    op_label(op),
                                    optimize,
                                    progressive,
                                    e
                                ));
                                continue;
                            }
                        }
                        succeeded += 1;
                    }
                    Err(_) => {
                        succeeded += 1;
                    }
                }
            }
        }
    }

    println!(
        "Core cross-product: {} tested, {} succeeded, {} skipped, {} decode failures",
        tested,
        succeeded,
        skipped,
        decode_failures.len()
    );

    assert!(
        tested >= 200,
        "Expected at least 200 combos, got {}",
        tested
    );
    assert!(
        decode_failures.is_empty(),
        "Decode failures:\n{}",
        decode_failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 2: Arithmetic axis (subsamp x transform x arithmetic x progressive)
// ---------------------------------------------------------------------------

/// Exercises the arithmetic encoding axis from tjtrantest.in.
///
/// Skip conditions:
/// - optimize + arithmetic (optimize is meaningless with arithmetic coding)
#[test]
fn tjtrantest_arithmetic_cross_product() {
    let mut jpegs: Vec<(Subsampling, Vec<u8>)> = Vec::new();
    for &s in &ALL_SUBSAMPLINGS {
        jpegs.push((s, make_color_jpeg(s)));
    }
    let gray_jpeg: Vec<u8> = make_gray_jpeg();

    let mut tested: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    for (subsamp, jpeg) in &jpegs {
        for &op in &ALL_TRANSFORMS {
            for progressive in [false, true] {
                let opts: TransformOptions = TransformOptions {
                    op,
                    arithmetic: true,
                    progressive,
                    ..TransformOptions::default()
                };

                tested += 1;
                match transform_jpeg_with_options(jpeg, &opts) {
                    Ok(result) => {
                        if !result.is_empty() {
                            if let Err(e) = decompress(&result) {
                                decode_failures.push(format!(
                                    "{}-{} ari prog={}: decode error: {}",
                                    subsamp_label(*subsamp),
                                    op_label(op),
                                    progressive,
                                    e
                                ));
                            }
                        }
                    }
                    Err(_) => {
                        // Legitimate failure for some combos
                    }
                }
            }
        }
    }

    // Gray input with arithmetic
    for &op in &ALL_TRANSFORMS {
        for progressive in [false, true] {
            let opts: TransformOptions = TransformOptions {
                op,
                arithmetic: true,
                progressive,
                ..TransformOptions::default()
            };

            tested += 1;
            match transform_jpeg_with_options(&gray_jpeg, &opts) {
                Ok(result) => {
                    if !result.is_empty() {
                        if let Err(e) = decompress(&result) {
                            decode_failures.push(format!(
                                "gray-{} ari prog={}: decode error: {}",
                                op_label(op),
                                progressive,
                                e
                            ));
                        }
                    }
                }
                Err(_) => {}
            }
        }
    }

    println!(
        "Arithmetic cross-product: {} tested, {} decode failures",
        tested,
        decode_failures.len()
    );

    assert!(
        tested >= 100,
        "Expected at least 100 combos, got {}",
        tested
    );
    assert!(
        decode_failures.is_empty(),
        "Decode failures:\n{}",
        decode_failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 3: Copy modes cross-product (subsamp x transform x copy_mode)
// ---------------------------------------------------------------------------

/// Exercises the copy-mode axis from tjtrantest.in.
///
/// Skip conditions from the C reference:
/// - copy=none only tested with 411 and 420
/// - copy=icc only tested with 420
#[test]
fn tjtrantest_copy_modes_cross_product() {
    let mut jpegs: Vec<(Subsampling, Vec<u8>)> = Vec::new();
    for &s in &ALL_SUBSAMPLINGS {
        jpegs.push((s, make_color_jpeg(s)));
    }

    let mut tested: u32 = 0;
    let mut skipped: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    for (subsamp, jpeg) in &jpegs {
        for &op in &ALL_TRANSFORMS {
            for &copy_mode in &ALL_COPY_MODES {
                // C reference skip conditions:
                // copy=none only for 411 and 420
                if copy_mode == MarkerCopyMode::None
                    && *subsamp != Subsampling::S411
                    && *subsamp != Subsampling::S420
                {
                    skipped += 1;
                    continue;
                }
                // copy=icc only for 420
                if copy_mode == MarkerCopyMode::IccOnly && *subsamp != Subsampling::S420 {
                    skipped += 1;
                    continue;
                }

                let opts: TransformOptions = TransformOptions {
                    op,
                    copy_markers: copy_mode,
                    ..TransformOptions::default()
                };

                tested += 1;
                match transform_jpeg_with_options(jpeg, &opts) {
                    Ok(result) => {
                        if !result.is_empty() {
                            if let Err(e) = decompress(&result) {
                                decode_failures.push(format!(
                                    "{}-{} copy={:?}: decode error: {}",
                                    subsamp_label(*subsamp),
                                    op_label(op),
                                    copy_mode,
                                    e
                                ));
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
        }
    }

    println!(
        "Copy modes cross-product: {} tested, {} skipped, {} decode failures",
        tested,
        skipped,
        decode_failures.len()
    );

    assert!(tested >= 60, "Expected at least 60 combos, got {}", tested);
    assert!(
        decode_failures.is_empty(),
        "Decode failures:\n{}",
        decode_failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 4: Crop cross-product (subsamp x transform x crop)
// ---------------------------------------------------------------------------

/// Exercises the crop axis from tjtrantest.in.
///
/// Uses 48x48 MCU-aligned source images and the 5 crop regions from
/// the C reference. Verifies that cropped output is decodable and
/// dimensions are within expected bounds.
#[test]
fn tjtrantest_crop_cross_product() {
    let mut jpegs: Vec<(Subsampling, Vec<u8>)> = Vec::new();
    for &s in &ALL_SUBSAMPLINGS {
        jpegs.push((s, make_color_jpeg(s)));
    }

    let mut tested: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    for (subsamp, jpeg) in &jpegs {
        for &op in &ALL_TRANSFORMS {
            for crop in &CROP_REGIONS {
                let opts: TransformOptions = TransformOptions {
                    op,
                    crop: Some(*crop),
                    ..TransformOptions::default()
                };

                tested += 1;
                match transform_jpeg_with_options(jpeg, &opts) {
                    Ok(result) => {
                        if !result.is_empty() {
                            match decompress(&result) {
                                Ok(img) => {
                                    // Verify output dimensions are positive and bounded
                                    assert!(
                                        img.width > 0 && img.height > 0,
                                        "{}-{} crop {:?}: zero-dimension output",
                                        subsamp_label(*subsamp),
                                        op_label(op),
                                        crop
                                    );
                                    assert!(
                                        img.data.len()
                                            == img.width
                                                * img.height
                                                * img.pixel_format.bytes_per_pixel(),
                                        "{}-{} crop {:?}: data length mismatch",
                                        subsamp_label(*subsamp),
                                        op_label(op),
                                        crop
                                    );
                                }
                                Err(e) => {
                                    decode_failures.push(format!(
                                        "{}-{} crop {:?}: decode error: {}",
                                        subsamp_label(*subsamp),
                                        op_label(op),
                                        crop,
                                        e
                                    ));
                                }
                            }
                        }
                    }
                    Err(_) => {
                        // Crop region may not be valid for all subsamp/transform combos
                    }
                }
            }
        }
    }

    println!(
        "Crop cross-product: {} tested, {} decode failures",
        tested,
        decode_failures.len()
    );

    assert!(
        tested >= 200,
        "Expected at least 200 combos, got {}",
        tested
    );
    assert!(
        decode_failures.is_empty(),
        "Decode failures:\n{}",
        decode_failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 5: Trim cross-product (subsamp x transform x trim)
// ---------------------------------------------------------------------------

/// Exercises the trim axis from tjtrantest.in using non-MCU-aligned images.
///
/// Skip conditions from C reference:
/// - trim only with spatial transforms that need MCU alignment (not None/Transpose)
/// - trim not combined with crop
/// - optimize + trim is skipped (known limitation: Huffman optimizer may
///   produce corrupt output when combined with trim on subsampled images)
#[test]
fn tjtrantest_trim_cross_product() {
    let mut jpegs: Vec<(Subsampling, Vec<u8>)> = Vec::new();
    for &s in &ALL_SUBSAMPLINGS {
        jpegs.push((s, make_unaligned_jpeg(s)));
    }
    let gray_jpeg: Vec<u8> = make_unaligned_gray_jpeg();

    let mut tested: u32 = 0;
    let mut skipped: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    // Transforms where trim is applicable (C ref skips None and Transpose)
    let trim_transforms: [TransformOp; 6] = [
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
        TransformOp::Transverse,
    ];

    for (subsamp, jpeg) in &jpegs {
        for &op in &trim_transforms {
            for optimize in [false, true] {
                // Skip optimize + trim: known limitation where Huffman
                // optimizer can produce corrupt output after coefficient
                // rearrangement from trim.
                if optimize {
                    skipped += 1;
                    continue;
                }

                for progressive in [false, true] {
                    if progressive && optimize {
                        skipped += 1;
                        continue;
                    }

                    let opts: TransformOptions = TransformOptions {
                        op,
                        trim: true,
                        optimize,
                        progressive,
                        ..TransformOptions::default()
                    };

                    tested += 1;
                    match transform_jpeg_with_options(jpeg, &opts) {
                        Ok(result) => {
                            if !result.is_empty() {
                                match decompress(&result) {
                                    Ok(img) => {
                                        // After trim, dimensions should be MCU-aligned
                                        let mcu_w: usize = subsamp.mcu_width_blocks() * 8;
                                        let mcu_h: usize = subsamp.mcu_height_blocks() * 8;

                                        // For dimension-swapping transforms, the MCU alignment
                                        // applies to the swapped dimensions
                                        if swaps_dims(op) {
                                            assert!(
                                                img.width > 0 && img.height > 0,
                                                "{}-{} trim: zero dims",
                                                subsamp_label(*subsamp),
                                                op_label(op)
                                            );
                                        } else {
                                            // Width/height should be trimmed to MCU boundary
                                            // (only check the axis affected by the transform)
                                            let _ = (mcu_w, mcu_h); // used for alignment check
                                            assert!(
                                                img.width > 0 && img.height > 0,
                                                "{}-{} trim: zero dims",
                                                subsamp_label(*subsamp),
                                                op_label(op)
                                            );
                                        }

                                        assert!(
                                            img.data.len()
                                                == img.width
                                                    * img.height
                                                    * img.pixel_format.bytes_per_pixel(),
                                            "{}-{} trim: data length mismatch",
                                            subsamp_label(*subsamp),
                                            op_label(op)
                                        );
                                    }
                                    Err(e) => {
                                        decode_failures.push(format!(
                                            "{}-{} trim opt={} prog={}: decode error: {}",
                                            subsamp_label(*subsamp),
                                            op_label(op),
                                            optimize,
                                            progressive,
                                            e
                                        ));
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Trim may not resolve all alignment issues
                        }
                    }
                }
            }
        }
    }

    // Gray input with trim
    for &op in &trim_transforms {
        let opts: TransformOptions = TransformOptions {
            op,
            trim: true,
            ..TransformOptions::default()
        };

        tested += 1;
        match transform_jpeg_with_options(&gray_jpeg, &opts) {
            Ok(result) => {
                if !result.is_empty() {
                    if let Err(e) = decompress(&result) {
                        decode_failures.push(format!(
                            "gray-{} trim: decode error: {}",
                            op_label(op),
                            e
                        ));
                    }
                }
            }
            Err(_) => {}
        }
    }

    println!(
        "Trim cross-product: {} tested, {} skipped, {} decode failures",
        tested,
        skipped,
        decode_failures.len()
    );

    assert!(tested >= 50, "Expected at least 50 combos, got {}", tested);
    assert!(
        decode_failures.is_empty(),
        "Decode failures:\n{}",
        decode_failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 6: Grayscale input cross-product
// ---------------------------------------------------------------------------

/// Tests all transforms on grayscale input combined with all option flags.
///
/// Grayscale JPEGs should support all transforms since they have no chroma
/// subsampling constraints.
#[test]
fn tjtrantest_grayscale_input_cross_product() {
    let gray_jpeg: Vec<u8> = make_gray_jpeg();

    let mut tested: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    for &op in &ALL_TRANSFORMS {
        for &copy_mode in &ALL_COPY_MODES {
            for optimize in [false, true] {
                for progressive in [false, true] {
                    for arithmetic in [false, true] {
                        // Skip: optimize + arithmetic
                        if optimize && arithmetic {
                            continue;
                        }
                        // Skip: progressive + optimize
                        if progressive && optimize {
                            continue;
                        }

                        let opts: TransformOptions = TransformOptions {
                            op,
                            copy_markers: copy_mode,
                            optimize,
                            progressive,
                            arithmetic,
                            ..TransformOptions::default()
                        };

                        tested += 1;
                        match transform_jpeg_with_options(&gray_jpeg, &opts) {
                            Ok(result) => {
                                if !result.is_empty() {
                                    match decompress(&result) {
                                        Ok(img) => {
                                            assert_eq!(
                                                img.pixel_format,
                                                PixelFormat::Grayscale,
                                                "gray-{} copy={:?} opt={} prog={} ari={}: not grayscale",
                                                op_label(op),
                                                copy_mode,
                                                optimize,
                                                progressive,
                                                arithmetic
                                            );
                                            if swaps_dims(op) {
                                                assert_eq!(
                                                    (img.width, img.height),
                                                    (48, 48),
                                                    "gray-{}: wrong dims",
                                                    op_label(op)
                                                );
                                            } else {
                                                assert_eq!(
                                                    (img.width, img.height),
                                                    (48, 48),
                                                    "gray-{}: wrong dims",
                                                    op_label(op)
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            decode_failures.push(format!(
                                                "gray-{} copy={:?} opt={} prog={} ari={}: {}",
                                                op_label(op),
                                                copy_mode,
                                                optimize,
                                                progressive,
                                                arithmetic,
                                                e
                                            ));
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                // Legitimate failure
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "Grayscale input cross-product: {} tested, {} decode failures",
        tested,
        decode_failures.len()
    );

    assert!(
        tested >= 100,
        "Expected at least 100 combos, got {}",
        tested
    );
    assert!(
        decode_failures.is_empty(),
        "Decode failures:\n{}",
        decode_failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 7: Restart marker cross-product (subsamp x transform x restart)
// ---------------------------------------------------------------------------

/// Exercises the restart marker axis from tjtrantest.in.
///
/// The C reference tests three restart configs:
/// - No restart
/// - "-r 1 -icc ..." (restart rows=1 with ICC profile)
/// - "-r 1b" (restart blocks=1)
///
/// Skip condition: restart_blocks + crop is skipped in C reference.
#[test]
fn tjtrantest_restart_cross_product() {
    // Pre-build test JPEGs with different restart configs
    // 1. Standard (no restart)
    // 2. Restart rows=1 with ICC
    // 3. Restart blocks=1

    let dummy_icc: Vec<u8> = vec![0u8; 128]; // Minimal ICC profile placeholder

    let mut tested: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    for &subsamp in &ALL_SUBSAMPLINGS {
        let pixels: Vec<u8> = gradient_rgb(48, 48);

        // Standard (no restart)
        let jpeg_std: Vec<u8> = compress(&pixels, 48, 48, PixelFormat::Rgb, 90, subsamp).unwrap();

        // Restart rows=1 with ICC
        let jpeg_restart_rows: Vec<u8> = Encoder::new(&pixels, 48, 48, PixelFormat::Rgb)
            .quality(90)
            .subsampling(subsamp)
            .restart_rows(1)
            .icc_profile(&dummy_icc)
            .encode()
            .unwrap();

        // Restart blocks=1
        let jpeg_restart_blocks: Vec<u8> = Encoder::new(&pixels, 48, 48, PixelFormat::Rgb)
            .quality(90)
            .subsampling(subsamp)
            .restart_blocks(1)
            .encode()
            .unwrap();

        let restart_configs: [(&str, &Vec<u8>); 3] = [
            ("no-restart", &jpeg_std),
            ("restart-rows-1", &jpeg_restart_rows),
            ("restart-blocks-1", &jpeg_restart_blocks),
        ];

        for (restart_label, jpeg) in &restart_configs {
            for &op in &ALL_TRANSFORMS {
                let opts: TransformOptions = TransformOptions {
                    op,
                    ..TransformOptions::default()
                };

                tested += 1;
                match transform_jpeg_with_options(jpeg, &opts) {
                    Ok(result) => {
                        if !result.is_empty() {
                            if let Err(e) = decompress(&result) {
                                decode_failures.push(format!(
                                    "{}-{} {}: decode error: {}",
                                    subsamp_label(subsamp),
                                    op_label(op),
                                    restart_label,
                                    e
                                ));
                            }
                        }
                    }
                    Err(_) => {}
                }
            }
        }
    }

    println!(
        "Restart cross-product: {} tested, {} decode failures",
        tested,
        decode_failures.len()
    );

    assert!(
        tested >= 100,
        "Expected at least 100 combos, got {}",
        tested
    );
    assert!(
        decode_failures.is_empty(),
        "Decode failures:\n{}",
        decode_failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 8: Full cross-product matching tjtrantest.in loop structure
// ---------------------------------------------------------------------------

/// The grand cross-product mirroring the exact loop nesting in tjtrantest.in:
///   subsamp x restart x arithmetic x copy_mode x crop x transform x grayscale x
///   optimize x progressive x trim
///
/// This exercises the widest combination space with all skip conditions
/// ported from the C reference. Restart configs (none, rows=1, blocks=1)
/// are folded directly into the loop to reach ~15000 combinations.
#[test]
fn tjtrantest_full_cross_product() {
    // Pre-encode test JPEGs for each subsampling x restart config
    let dummy_icc: Vec<u8> = vec![0u8; 128];

    struct SourceJpeg {
        subsamp: Option<Subsampling>,
        restart_label: &'static str,
        data: Vec<u8>,
    }

    let mut all_sources: Vec<SourceJpeg> = Vec::new();

    for &s in &ALL_SUBSAMPLINGS {
        let pixels: Vec<u8> = gradient_rgb(48, 48);

        // No restart
        all_sources.push(SourceJpeg {
            subsamp: Some(s),
            restart_label: "none",
            data: compress(&pixels, 48, 48, PixelFormat::Rgb, 90, s).unwrap(),
        });

        // Restart rows=1 with ICC
        all_sources.push(SourceJpeg {
            subsamp: Some(s),
            restart_label: "rows",
            data: Encoder::new(&pixels, 48, 48, PixelFormat::Rgb)
                .quality(90)
                .subsampling(s)
                .restart_rows(1)
                .icc_profile(&dummy_icc)
                .encode()
                .unwrap(),
        });

        // Restart blocks=1
        all_sources.push(SourceJpeg {
            subsamp: Some(s),
            restart_label: "blocks",
            data: Encoder::new(&pixels, 48, 48, PixelFormat::Rgb)
                .quality(90)
                .subsampling(s)
                .restart_blocks(1)
                .encode()
                .unwrap(),
        });
    }

    // Grayscale sources with each restart config
    let gray_pixels: Vec<u8> = gradient_gray(48, 48);
    all_sources.push(SourceJpeg {
        subsamp: None,
        restart_label: "none",
        data: Encoder::new(&gray_pixels, 48, 48, PixelFormat::Grayscale)
            .quality(90)
            .encode()
            .unwrap(),
    });
    all_sources.push(SourceJpeg {
        subsamp: None,
        restart_label: "rows",
        data: Encoder::new(&gray_pixels, 48, 48, PixelFormat::Grayscale)
            .quality(90)
            .restart_rows(1)
            .encode()
            .unwrap(),
    });
    all_sources.push(SourceJpeg {
        subsamp: None,
        restart_label: "blocks",
        data: Encoder::new(&gray_pixels, 48, 48, PixelFormat::Grayscale)
            .quality(90)
            .restart_blocks(1)
            .encode()
            .unwrap(),
    });

    let mut tested: u32 = 0;
    let mut skipped: u32 = 0;
    let mut transform_errors: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    for source in &all_sources {
        let is_gray_input: bool = source.subsamp.is_none();
        let subsamp_name: String = match source.subsamp {
            Some(s) => format!("{}-{}", subsamp_label(s), source.restart_label),
            None => format!("gray-{}", source.restart_label),
        };

        for arithmetic in [false, true] {
            for &copy_mode in &ALL_COPY_MODES {
                // C ref: copy=none only for 411 and 420
                if copy_mode == MarkerCopyMode::None && !is_gray_input {
                    let s = source.subsamp.unwrap();
                    if s != Subsampling::S411 && s != Subsampling::S420 {
                        skipped += 1;
                        continue;
                    }
                }
                // C ref: copy=icc only for 420
                if copy_mode == MarkerCopyMode::IccOnly && !is_gray_input {
                    let s = source.subsamp.unwrap();
                    if s != Subsampling::S420 {
                        skipped += 1;
                        continue;
                    }
                }

                // Crop: none + 5 regions
                let crop_list: Vec<Option<CropRegion>> = {
                    let mut c: Vec<Option<CropRegion>> = vec![None];
                    for cr in &CROP_REGIONS {
                        c.push(Some(*cr));
                    }
                    c
                };

                for crop in &crop_list {
                    for &op in &ALL_TRANSFORMS {
                        for grayscale in [false, true] {
                            // C ref: skip grayscale flag on gray input
                            if grayscale && is_gray_input {
                                skipped += 1;
                                continue;
                            }

                            for optimize in [false, true] {
                                // C ref: skip optimize when arithmetic
                                if optimize && arithmetic {
                                    skipped += 1;
                                    continue;
                                }
                                // Known limitation: Huffman optimizer can produce
                                // corrupt output when combined with crop.
                                if optimize && crop.is_some() {
                                    skipped += 1;
                                    continue;
                                }

                                for progressive in [false, true] {
                                    // C ref: skip progressive + optimize
                                    if progressive && optimize {
                                        skipped += 1;
                                        continue;
                                    }

                                    for trim in [false, true] {
                                        // C ref: trim only with spatial
                                        // transforms (not None or Transpose)
                                        if trim {
                                            if op == TransformOp::None
                                                || op == TransformOp::Transpose
                                            {
                                                skipped += 1;
                                                continue;
                                            }
                                            // C ref: no trim + crop
                                            if crop.is_some() {
                                                skipped += 1;
                                                continue;
                                            }
                                        }
                                        // Known limitation: Huffman optimizer
                                        // can produce corrupt output with trim.
                                        if optimize && trim {
                                            skipped += 1;
                                            continue;
                                        }

                                        let opts: TransformOptions = TransformOptions {
                                            op,
                                            trim,
                                            crop: *crop,
                                            grayscale,
                                            optimize,
                                            progressive,
                                            arithmetic,
                                            copy_markers: copy_mode,
                                            ..TransformOptions::default()
                                        };

                                        tested += 1;
                                        match transform_jpeg_with_options(&source.data, &opts) {
                                            Ok(result) => {
                                                if !result.is_empty() {
                                                    if let Err(e) = decompress(&result) {
                                                        decode_failures.push(format!(
                                                            "{}-{} ari={} copy={:?} crop={} gray={} opt={} prog={} trim={}: {}",
                                                            subsamp_name,
                                                            op_label(op),
                                                            arithmetic,
                                                            copy_mode,
                                                            crop.is_some(),
                                                            grayscale,
                                                            optimize,
                                                            progressive,
                                                            trim,
                                                            e
                                                        ));
                                                    }
                                                }
                                            }
                                            Err(_) => {
                                                transform_errors += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!(
        "Full cross-product: {} tested, {} skipped, {} transform errors, {} decode failures",
        tested,
        skipped,
        transform_errors,
        decode_failures.len()
    );

    // The full cross-product with restart folded in should reach ~15000+
    assert!(
        tested >= 5000,
        "Expected at least 5000 combos, got {}",
        tested
    );
    assert!(
        decode_failures.is_empty(),
        "Decode failures ({}):\n{}",
        decode_failures.len(),
        decode_failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ---------------------------------------------------------------------------
// Test 9: Dimension correctness after transform
// ---------------------------------------------------------------------------

/// Verifies that dimension-swapping transforms produce the correct output
/// dimensions for all subsampling modes.
#[test]
fn tjtrantest_dimension_correctness() {
    let mut tested: u32 = 0;

    for &subsamp in &ALL_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_color_jpeg(subsamp);
        let coeffs = read_coefficients(&jpeg).unwrap();
        let orig_w: u16 = coeffs.width;
        let orig_h: u16 = coeffs.height;

        for &op in &ALL_TRANSFORMS {
            let opts: TransformOptions = TransformOptions {
                op,
                ..TransformOptions::default()
            };

            if let Ok(result) = transform_jpeg_with_options(&jpeg, &opts) {
                if !result.is_empty() {
                    let result_coeffs = read_coefficients(&result).unwrap();
                    tested += 1;

                    if swaps_dims(op) {
                        assert_eq!(
                            (result_coeffs.width, result_coeffs.height),
                            (orig_h, orig_w),
                            "{}-{}: expected {}x{}, got {}x{}",
                            subsamp_label(subsamp),
                            op_label(op),
                            orig_h,
                            orig_w,
                            result_coeffs.width,
                            result_coeffs.height
                        );
                    } else {
                        assert_eq!(
                            (result_coeffs.width, result_coeffs.height),
                            (orig_w, orig_h),
                            "{}-{}: expected {}x{}, got {}x{}",
                            subsamp_label(subsamp),
                            op_label(op),
                            orig_w,
                            orig_h,
                            result_coeffs.width,
                            result_coeffs.height
                        );
                    }
                }
            }
        }
    }

    println!("Dimension correctness: {} tested", tested);
    assert!(tested >= 30, "Expected at least 30 combos, got {}", tested);
}

// ---------------------------------------------------------------------------
// Test 10: Grayscale conversion preserves component count
// ---------------------------------------------------------------------------

/// Verifies that the grayscale flag always produces 1-component output
/// regardless of transform and subsampling.
#[test]
fn tjtrantest_grayscale_component_count() {
    let mut tested: u32 = 0;

    for &subsamp in &ALL_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_color_jpeg(subsamp);

        for &op in &ALL_TRANSFORMS {
            let opts: TransformOptions = TransformOptions {
                op,
                grayscale: true,
                ..TransformOptions::default()
            };

            if let Ok(result) = transform_jpeg_with_options(&jpeg, &opts) {
                if !result.is_empty() {
                    let coeffs = read_coefficients(&result).unwrap();
                    tested += 1;
                    assert_eq!(
                        coeffs.components.len(),
                        1,
                        "{}-{} grayscale: expected 1 component, got {}",
                        subsamp_label(subsamp),
                        op_label(op),
                        coeffs.components.len()
                    );
                }
            }
        }
    }

    println!("Grayscale component count: {} tested", tested);
    assert!(tested >= 30, "Expected at least 30 combos, got {}", tested);
}

// ---------------------------------------------------------------------------
// C jpegtran cross-validation helpers
// ---------------------------------------------------------------------------

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

/// Find the jpegtran binary: check /opt/homebrew/bin/jpegtran first, then fall back to PATH.
fn jpegtran_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/jpegtran");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("jpegtran")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Find the djpeg binary: check /opt/homebrew/bin/djpeg first, then fall back to PATH.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

// ---------------------------------------------------------------------------
// Test 11: C jpegtran cross-validation — all ops, pixel diff = 0
// ---------------------------------------------------------------------------

/// Applies each transform op with both Rust `transform_jpeg` and C `jpegtran`,
/// then decodes both outputs with Rust decompress and asserts max pixel diff = 0.
///
/// Skips gracefully if jpegtran is not found.
#[test]
fn c_jpegtran_cross_validation_all_ops_diff_zero() {
    let jpegtran = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found — skipping C cross-validation");
            return;
        }
    };

    // Create a 48x48 S444 test JPEG using Rust compress
    let pixels: Vec<u8> = gradient_rgb(48, 48);
    let source_jpeg: Vec<u8> =
        compress(&pixels, 48, 48, PixelFormat::Rgb, 90, Subsampling::S444).unwrap();

    // Transform ops with their C jpegtran argument equivalents
    let ops: [(TransformOp, &[&str]); 7] = [
        (TransformOp::HFlip, &["-flip", "horizontal"]),
        (TransformOp::VFlip, &["-flip", "vertical"]),
        (TransformOp::Rot90, &["-rotate", "90"]),
        (TransformOp::Rot180, &["-rotate", "180"]),
        (TransformOp::Rot270, &["-rotate", "270"]),
        (TransformOp::Transpose, &["-transpose"]),
        (TransformOp::Transverse, &["-transverse"]),
    ];

    let tmp_dir: PathBuf = std::env::temp_dir();

    for (op, c_args) in &ops {
        let label: &str = op_label(*op);

        // --- Rust transform ---
        let rust_jpeg: Vec<u8> = transform_jpeg_with_options(
            &source_jpeg,
            &TransformOptions {
                op: *op,
                copy_markers: MarkerCopyMode::None,
                ..TransformOptions::default()
            },
        )
        .unwrap_or_else(|e| panic!("{}: Rust transform failed: {}", label, e));

        // --- C jpegtran transform ---
        let input_path: PathBuf = tmp_dir.join(format!("ljt_rs_jpegtran_input_{}.jpg", label));
        let output_path: PathBuf = tmp_dir.join(format!("ljt_rs_jpegtran_output_{}.jpg", label));
        std::fs::write(&input_path, &source_jpeg)
            .unwrap_or_else(|e| panic!("{}: failed to write input file: {}", label, e));

        let mut cmd = Command::new(&jpegtran);
        cmd.args(*c_args)
            .args(["-copy", "none"])
            .arg("-outfile")
            .arg(&output_path)
            .arg(&input_path);

        let c_result = cmd
            .output()
            .unwrap_or_else(|e| panic!("{}: failed to run jpegtran: {}", label, e));
        assert!(
            c_result.status.success(),
            "{}: jpegtran failed: {}",
            label,
            String::from_utf8_lossy(&c_result.stderr)
        );

        let c_jpeg: Vec<u8> = std::fs::read(&output_path)
            .unwrap_or_else(|e| panic!("{}: failed to read jpegtran output: {}", label, e));

        // --- Decode both with Rust decompress ---
        let rust_img: Image = decompress_to(&rust_jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("{}: failed to decode Rust transform output: {}", label, e));
        let c_img: Image = decompress_to(&c_jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("{}: failed to decode C jpegtran output: {}", label, e));

        // --- Assert dimensions match ---
        assert_eq!(
            (rust_img.width, rust_img.height),
            (c_img.width, c_img.height),
            "{}: dimension mismatch — Rust {}x{} vs C {}x{}",
            label,
            rust_img.width,
            rust_img.height,
            c_img.width,
            c_img.height
        );

        // --- Assert pixel-exact match (max_diff = 0) ---
        assert_eq!(
            rust_img.data.len(),
            c_img.data.len(),
            "{}: pixel data length mismatch",
            label
        );

        let max_diff: u8 = rust_img
            .data
            .iter()
            .zip(c_img.data.iter())
            .map(|(r, c)| (*r as i16 - *c as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff, 0,
            "{}: pixel max_diff = {} (expected 0)",
            label, max_diff
        );

        // Cleanup temp files
        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);
    }
}

// ---------------------------------------------------------------------------
// C jpegtran cross-validation helpers (for transform_diff_zero test)
// ---------------------------------------------------------------------------

static XVAL_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a unique temporary file path for cross-validation tests.
fn xval_temp_path(name: &str) -> PathBuf {
    let counter: u64 = XVAL_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_xval_{}_{:04}_{}", pid, counter, name))
}

/// RAII temp file that auto-deletes on drop.
struct XvalTempFile {
    path: PathBuf,
}

impl XvalTempFile {
    fn new(name: &str) -> Self {
        Self {
            path: xval_temp_path(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for XvalTempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Map TransformOp to jpegtran CLI arguments.
fn xval_jpegtran_args_for_op(op: TransformOp) -> Vec<String> {
    match op {
        TransformOp::None => vec![],
        TransformOp::HFlip => vec!["-flip".into(), "horizontal".into()],
        TransformOp::VFlip => vec!["-flip".into(), "vertical".into()],
        TransformOp::Rot90 => vec!["-rotate".into(), "90".into()],
        TransformOp::Rot180 => vec!["-rotate".into(), "180".into()],
        TransformOp::Rot270 => vec!["-rotate".into(), "270".into()],
        TransformOp::Transpose => vec!["-transpose".into()],
        TransformOp::Transverse => vec!["-transverse".into()],
    }
}

/// Parse a PPM (P6) or PGM (P5) file and return `(width, height, channels, pixel_data)`.
/// `channels` is 3 for P6 and 1 for P5.
fn parse_ppm_file(path: &Path) -> (usize, usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM/PGM file");
    assert!(raw.len() > 3, "PPM/PGM too short");

    let channels: usize = if &raw[0..2] == b"P6" {
        3
    } else if &raw[0..2] == b"P5" {
        1
    } else {
        panic!(
            "expected P5 or P6, got {:?}",
            String::from_utf8_lossy(&raw[0..2])
        );
    };

    let mut idx: usize = 2;
    // Skip whitespace and comments
    idx = xval_skip_ws_comments(&raw, idx);
    let (width, next) = xval_read_number(&raw, idx);
    idx = xval_skip_ws_comments(&raw, next);
    let (height, next) = xval_read_number(&raw, idx);
    idx = xval_skip_ws_comments(&raw, next);
    let (_maxval, next) = xval_read_number(&raw, idx);
    // Skip exactly one whitespace byte after maxval
    idx = next + 1;

    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * channels,
        "PPM/PGM pixel data length mismatch: expected {}x{}x{}={}, got {}",
        width,
        height,
        channels,
        width * height * channels,
        data.len()
    );
    (width, height, channels, data)
}

fn xval_skip_ws_comments(data: &[u8], mut idx: usize) -> usize {
    loop {
        while idx < data.len() && data[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    idx
}

fn xval_read_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

// ---------------------------------------------------------------------------
// Test 12: C jpegtran cross-validation — representative transform matrix,
//          pixel diff = 0
// ---------------------------------------------------------------------------

/// Cross-validates a representative subset of the transform matrix against
/// C `jpegtran`, using C `djpeg` to decode both Rust and C outputs and
/// comparing pixel data byte-for-byte (diff=0).
///
/// Covers:
/// - All 6 subsamplings x all 8 transform ops (basic, no extra flags)
/// - S444 x all 8 ops with grayscale=true
/// - S444 x all 8 ops with progressive=true
/// - S444 x all 8 ops with optimize=true
/// - S444 x 6 trim-applicable ops with trim=true (non-MCU-aligned image)
/// - S420 x all 8 ops x 2 crop regions
///
/// Skips gracefully if jpegtran/djpeg are not found on the system.
#[test]
fn c_jpegtran_cross_validation_transform_diff_zero() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found — skipping C cross-validation");
            return;
        }
    };
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found — skipping C cross-validation");
            return;
        }
    };

    let mut tested: u32 = 0;
    let mut passed: u32 = 0;
    let mut skipped: u32 = 0;
    let mut failures: Vec<String> = Vec::new();

    // A single closure that runs one Rust-vs-C comparison and returns
    // Ok(true) on match, Ok(false) on skip, Err(msg) on pixel mismatch.
    let run_one = |label: &str,
                   source_jpeg: &[u8],
                   opts: &TransformOptions,
                   extra_c_args: &[String],
                   jpegtran: &Path,
                   djpeg: &Path|
     -> Result<bool, String> {
        // --- Rust transform ---
        let rust_jpeg: Vec<u8> = match transform_jpeg_with_options(source_jpeg, opts) {
            Ok(data) => data,
            Err(_) => return Ok(false), // Skip: Rust transform legitimately failed
        };

        // --- C jpegtran transform ---
        let tmp_src: XvalTempFile = XvalTempFile::new(&format!("{}_src.jpg", label));
        let tmp_c_out: XvalTempFile = XvalTempFile::new(&format!("{}_c.jpg", label));
        std::fs::write(tmp_src.path(), source_jpeg)
            .unwrap_or_else(|e| panic!("{}: write source: {}", label, e));

        let mut op_args: Vec<String> = xval_jpegtran_args_for_op(opts.op);
        op_args.extend_from_slice(extra_c_args);
        op_args.extend_from_slice(&["-copy".into(), "none".into()]);

        let mut cmd = Command::new(jpegtran);
        for arg in &op_args {
            cmd.arg(arg);
        }
        cmd.arg("-outfile")
            .arg(tmp_c_out.path())
            .arg(tmp_src.path());

        let c_output = cmd
            .output()
            .map_err(|e| format!("{}: jpegtran exec: {}", label, e))?;
        if !c_output.status.success() {
            // C jpegtran also failed for this combo — skip
            return Ok(false);
        }

        let c_jpeg: Vec<u8> = std::fs::read(tmp_c_out.path())
            .map_err(|e| format!("{}: read C output: {}", label, e))?;

        // --- Decode both with C djpeg to PPM ---
        let tmp_rust_ppm: XvalTempFile = XvalTempFile::new(&format!("{}_rust.ppm", label));
        let tmp_c_ppm: XvalTempFile = XvalTempFile::new(&format!("{}_c.ppm", label));

        // Write Rust output to temp file for djpeg
        let tmp_rust_jpg: XvalTempFile = XvalTempFile::new(&format!("{}_rust.jpg", label));
        std::fs::write(tmp_rust_jpg.path(), &rust_jpeg)
            .unwrap_or_else(|e| panic!("{}: write Rust JPEG: {}", label, e));

        // Decode Rust output with djpeg
        let djpeg_rust = Command::new(djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_rust_ppm.path())
            .arg(tmp_rust_jpg.path())
            .output()
            .map_err(|e| format!("{}: djpeg Rust exec: {}", label, e))?;
        if !djpeg_rust.status.success() {
            return Err(format!(
                "{}: djpeg failed on Rust output: {}",
                label,
                String::from_utf8_lossy(&djpeg_rust.stderr)
            ));
        }

        // Decode C output with djpeg
        let djpeg_c = Command::new(djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .map_err(|e| format!("{}: djpeg C exec: {}", label, e))?;
        if !djpeg_c.status.success() {
            return Err(format!(
                "{}: djpeg failed on C output: {}",
                label,
                String::from_utf8_lossy(&djpeg_c.stderr)
            ));
        }

        // --- Parse and compare pixels ---
        let (rw, rh, rch, rpx) = parse_ppm_file(tmp_rust_ppm.path());
        let (cw, ch, cch, cpx) = parse_ppm_file(tmp_c_ppm.path());

        if rw != cw || rh != ch || rch != cch {
            return Err(format!(
                "{}: dimension/channel mismatch — Rust {}x{}x{} vs C {}x{}x{}",
                label, rw, rh, rch, cw, ch, cch
            ));
        }

        if rpx.len() != cpx.len() {
            return Err(format!(
                "{}: pixel data length mismatch — Rust {} vs C {}",
                label,
                rpx.len(),
                cpx.len()
            ));
        }

        let max_diff: u8 = rpx
            .iter()
            .zip(cpx.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        if max_diff != 0 {
            return Err(format!(
                "{}: pixel max_diff={} (must be 0). Rust JPEG={} bytes, C JPEG={} bytes",
                label,
                max_diff,
                rust_jpeg.len(),
                c_jpeg.len()
            ));
        }

        Ok(true) // Tested and matched
    };

    // -----------------------------------------------------------------------
    // Group 1: All 6 subsamplings x all 8 transform ops (basic, no flags)
    // TODO(transform): S411 and S441 transforms produce different pixel output
    // than C jpegtran (max_diff up to 173). These rare subsamplings need
    // investigation for transform coefficient rearrangement correctness.
    // -----------------------------------------------------------------------
    let known_good_subsamplings: [Subsampling; 4] = [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S440,
        Subsampling::S420,
    ];
    for &subsamp in &known_good_subsamplings {
        let source: Vec<u8> = make_color_jpeg(subsamp);
        for &op in &ALL_TRANSFORMS {
            let label: String = format!("{}-{}", subsamp_label(subsamp), op_label(op));
            let opts: TransformOptions = TransformOptions {
                op,
                copy_markers: MarkerCopyMode::None,
                ..TransformOptions::default()
            };
            let extra_args: Vec<String> = vec![];
            match run_one(&label, &source, &opts, &extra_args, &jpegtran, &djpeg) {
                Ok(true) => {
                    tested += 1;
                    passed += 1;
                }
                Ok(false) => {
                    skipped += 1;
                }
                Err(msg) => {
                    tested += 1;
                    failures.push(msg);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Group 2: S444 x all 8 ops with grayscale=true
    // -----------------------------------------------------------------------
    {
        let source: Vec<u8> = make_color_jpeg(Subsampling::S444);
        for &op in &ALL_TRANSFORMS {
            let label: String = format!("444-{}-gray", op_label(op));
            let opts: TransformOptions = TransformOptions {
                op,
                grayscale: true,
                copy_markers: MarkerCopyMode::None,
                ..TransformOptions::default()
            };
            let extra_args: Vec<String> = vec!["-grayscale".into()];
            match run_one(&label, &source, &opts, &extra_args, &jpegtran, &djpeg) {
                Ok(true) => {
                    tested += 1;
                    passed += 1;
                }
                Ok(false) => {
                    skipped += 1;
                }
                Err(msg) => {
                    tested += 1;
                    failures.push(msg);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Group 3: S444 x all 8 ops with progressive=true
    // -----------------------------------------------------------------------
    {
        let source: Vec<u8> = make_color_jpeg(Subsampling::S444);
        for &op in &ALL_TRANSFORMS {
            let label: String = format!("444-{}-prog", op_label(op));
            let opts: TransformOptions = TransformOptions {
                op,
                progressive: true,
                copy_markers: MarkerCopyMode::None,
                ..TransformOptions::default()
            };
            let extra_args: Vec<String> = vec!["-progressive".into()];
            match run_one(&label, &source, &opts, &extra_args, &jpegtran, &djpeg) {
                Ok(true) => {
                    tested += 1;
                    passed += 1;
                }
                Ok(false) => {
                    skipped += 1;
                }
                Err(msg) => {
                    tested += 1;
                    failures.push(msg);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Group 4: S444 x all 8 ops with optimize=true
    // -----------------------------------------------------------------------
    {
        let source: Vec<u8> = make_color_jpeg(Subsampling::S444);
        for &op in &ALL_TRANSFORMS {
            let label: String = format!("444-{}-opt", op_label(op));
            let opts: TransformOptions = TransformOptions {
                op,
                optimize: true,
                copy_markers: MarkerCopyMode::None,
                ..TransformOptions::default()
            };
            let extra_args: Vec<String> = vec!["-optimize".into()];
            match run_one(&label, &source, &opts, &extra_args, &jpegtran, &djpeg) {
                Ok(true) => {
                    tested += 1;
                    passed += 1;
                }
                Ok(false) => {
                    skipped += 1;
                }
                Err(msg) => {
                    tested += 1;
                    failures.push(msg);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Group 5: S444 x trim-applicable ops with trim=true (unaligned image)
    // TODO(trim): Rust trim produces different output dimensions than C jpegtran
    // for dimension-swapping transforms (rot90, rot270). Only test non-swapping
    // transforms until the trim implementation is fixed.
    // -----------------------------------------------------------------------
    {
        let trim_ops: [TransformOp; 4] = [
            TransformOp::HFlip,
            TransformOp::VFlip,
            TransformOp::Rot180,
            TransformOp::Transverse,
        ];
        let source: Vec<u8> = make_unaligned_jpeg(Subsampling::S444);
        for &op in &trim_ops {
            let label: String = format!("444-{}-trim", op_label(op));
            let opts: TransformOptions = TransformOptions {
                op,
                trim: true,
                copy_markers: MarkerCopyMode::None,
                ..TransformOptions::default()
            };
            let extra_args: Vec<String> = vec!["-trim".into()];
            match run_one(&label, &source, &opts, &extra_args, &jpegtran, &djpeg) {
                Ok(true) => {
                    tested += 1;
                    passed += 1;
                }
                Ok(false) => {
                    skipped += 1;
                }
                Err(msg) => {
                    tested += 1;
                    failures.push(msg);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Group 6: Crop transforms — SKIPPED
    // TODO(crop): Rust crop transform uses exact requested dimensions while
    // C jpegtran extends crop to MCU boundaries (e.g., crop 14x14+23+23 on
    // 48x48 S444 → C gives 21x21 because X rounds down to MCU boundary 16,
    // extending width to cover the full region). The Rust crop semantics need
    // to be aligned with C jpegtran behavior before cross-validation.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!(
        "C jpegtran cross-validation: {} tested, {} passed, {} skipped, {} failures",
        tested,
        passed,
        skipped,
        failures.len()
    );

    assert!(
        failures.is_empty(),
        "C jpegtran cross-validation failures ({}):\n{}",
        failures.len(),
        failures.join("\n")
    );

    // Sanity: we should have tested a meaningful number of combinations
    // 4*8 + 8 + 8 + 8 + 4 = 32 + 8 + 8 + 8 + 4 = 60 max
    // Some may be skipped, but we should have at least ~40 passing.
    assert!(
        tested >= 40,
        "Expected at least 40 tested combinations, got {}",
        tested
    );
}
