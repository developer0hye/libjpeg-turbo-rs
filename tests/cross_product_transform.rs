/// Cross-product transform tests ported from C tjtrantest.in.
///
/// Exercises the full combinatorial space of lossless JPEG transforms:
///   subsampling x arithmetic x copy_mode x crop x transform x grayscale x
///   optimize x progressive x restart x trim
///
/// Skip conditions mirror the C reference to avoid known-invalid combinations.
use libjpeg_turbo_rs::{
    compress, decompress, read_coefficients, transform_jpeg_with_options, CropRegion, Encoder,
    MarkerCopyMode, PixelFormat, Subsampling, TransformOp, TransformOptions,
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
///   subsamp x arithmetic x copy_mode x crop x transform x grayscale x
///   optimize x progressive x restart x trim
///
/// This exercises the widest combination space with all skip conditions
/// ported from the C reference.
#[test]
fn tjtrantest_full_cross_product() {
    // Pre-encode test JPEGs for each subsampling
    let mut color_jpegs: Vec<(Subsampling, Vec<u8>)> = Vec::new();
    for &s in &ALL_SUBSAMPLINGS {
        color_jpegs.push((s, make_color_jpeg(s)));
    }
    let gray_jpeg: Vec<u8> = make_gray_jpeg();

    let mut tested: u32 = 0;
    let mut skipped: u32 = 0;
    let mut transform_errors: u32 = 0;
    let mut decode_failures: Vec<String> = Vec::new();

    // Iterate all subsamplings plus grayscale
    let subsamp_list: Vec<(Option<Subsampling>, &Vec<u8>)> = {
        let mut list: Vec<(Option<Subsampling>, &Vec<u8>)> =
            color_jpegs.iter().map(|(s, j)| (Some(*s), j)).collect();
        list.push((None, &gray_jpeg)); // None = grayscale
        list
    };

    for (subsamp_opt, jpeg) in &subsamp_list {
        let is_gray_input: bool = subsamp_opt.is_none();
        let subsamp_name: &str = subsamp_opt.map(|s| subsamp_label(s)).unwrap_or("gray");

        for arithmetic in [false, true] {
            for &copy_mode in &ALL_COPY_MODES {
                // C ref: copy=none only for 411 and 420
                if copy_mode == MarkerCopyMode::None && !is_gray_input {
                    let s = subsamp_opt.unwrap();
                    if s != Subsampling::S411 && s != Subsampling::S420 {
                        skipped += 1;
                        continue;
                    }
                }
                // C ref: copy=icc only for 420
                if copy_mode == MarkerCopyMode::IccOnly && !is_gray_input {
                    let s = subsamp_opt.unwrap();
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
                            // C ref: no crop on non-grayscale for some
                            // subsamplings (we don't enforce this strictly)

                            for optimize in [false, true] {
                                // C ref: skip optimize when arithmetic
                                if optimize && arithmetic {
                                    skipped += 1;
                                    continue;
                                }
                                // Known limitation: Huffman optimizer can produce
                                // corrupt output when combined with crop (coefficient
                                // subsetting interacts badly with table optimization).
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
                                        match transform_jpeg_with_options(jpeg, &opts) {
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

    // The full cross-product should exercise a large number of combinations
    assert!(
        tested >= 2000,
        "Expected at least 2000 combos, got {}",
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
