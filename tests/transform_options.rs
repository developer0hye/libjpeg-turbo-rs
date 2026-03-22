use libjpeg_turbo_rs::{
    compress, decompress, read_coefficients, CropRegion, PixelFormat, Subsampling, TransformOp,
    TransformOptions,
};

/// Helper: create a small color JPEG for testing.
fn make_test_jpeg(width: usize, height: usize, subsampling: Subsampling) -> Vec<u8> {
    let bpp: usize = 3;
    let mut pixels: Vec<u8> = vec![0u8; width * height * bpp];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * bpp;
            pixels[idx] = (x * 255 / width.max(1)) as u8; // R gradient
            pixels[idx + 1] = (y * 255 / height.max(1)) as u8; // G gradient
            pixels[idx + 2] = 128; // B constant
        }
    }
    compress(&pixels, width, height, PixelFormat::Rgb, 90, subsampling).unwrap()
}

/// Helper: create a grayscale JPEG for testing.
fn make_gray_jpeg(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height];
    for y in 0..height {
        for x in 0..width {
            pixels[y * width + x] = ((x + y) % 256) as u8;
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
    )
    .unwrap()
}

// --- Default options ---

#[test]
fn default_options_has_sensible_defaults() {
    let opts: TransformOptions = TransformOptions::default();
    assert_eq!(opts.op, TransformOp::None);
    assert!(!opts.perfect);
    assert!(!opts.trim);
    assert!(opts.crop.is_none());
    assert!(!opts.grayscale);
    assert!(!opts.no_output);
    assert!(!opts.progressive);
    assert!(!opts.arithmetic);
    assert!(!opts.optimize);
    assert!(opts.copy_markers);
}

// --- Grayscale transform ---

#[test]
fn transform_grayscale_drops_chroma_components() {
    // Encode a color JPEG, transform with grayscale=true, verify 1-component output.
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        grayscale: true,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let coeffs = read_coefficients(&result).unwrap();

    // After grayscale conversion, should have exactly 1 component.
    assert_eq!(coeffs.components.len(), 1);

    // Output should be decompressible.
    let image = decompress(&result).unwrap();
    assert_eq!(image.width, 64);
    assert_eq!(image.height, 64);
}

#[test]
fn transform_grayscale_on_already_grayscale_is_noop() {
    let data: Vec<u8> = make_gray_jpeg(32, 32);
    let opts: TransformOptions = TransformOptions {
        grayscale: true,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let coeffs = read_coefficients(&result).unwrap();
    assert_eq!(coeffs.components.len(), 1);
}

// --- Perfect flag ---

#[test]
fn transform_perfect_fails_on_partial_mcu() {
    // Create image with dimensions not aligned to MCU boundaries.
    // For 4:2:0, MCU = 16x16, so 30x30 has partial MCUs.
    let data: Vec<u8> = make_test_jpeg(30, 30, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        op: TransformOp::HFlip,
        perfect: true,
        ..TransformOptions::default()
    };

    let result = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts);
    assert!(result.is_err());
}

#[test]
fn transform_perfect_succeeds_on_aligned_image() {
    // 64x64 with 4:2:0 (MCU=16x16) is perfectly aligned.
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        op: TransformOp::HFlip,
        perfect: true,
        ..TransformOptions::default()
    };

    let result = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts);
    assert!(result.is_ok());
}

// --- No output (dry run) ---

#[test]
fn transform_no_output_returns_empty() {
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        no_output: true,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    assert!(result.is_empty());
}

#[test]
fn transform_no_output_with_perfect_still_validates() {
    // Even with no_output, the perfect check should still run.
    let data: Vec<u8> = make_test_jpeg(30, 30, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        op: TransformOp::HFlip,
        perfect: true,
        no_output: true,
        ..TransformOptions::default()
    };

    let result = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts);
    assert!(result.is_err());
}

// --- Optimize flag ---

#[test]
fn transform_with_optimize_produces_valid_jpeg() {
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        optimize: true,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let image = decompress(&result).unwrap();
    assert_eq!(image.width, 64);
    assert_eq!(image.height, 64);
}

#[test]
fn transform_with_optimize_smaller_than_standard() {
    // Optimized Huffman tables should produce smaller or equal output.
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S444);

    let standard: Vec<u8> =
        libjpeg_turbo_rs::transform_jpeg_with_options(&data, &TransformOptions::default()).unwrap();

    let optimized: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(
        &data,
        &TransformOptions {
            optimize: true,
            ..TransformOptions::default()
        },
    )
    .unwrap();

    assert!(optimized.len() <= standard.len());
}

// --- Trim flag ---

#[test]
fn transform_trim_adjusts_partial_mcu_edges() {
    // 30x30 with 4:2:0 (MCU=16x16) has partial MCU edges.
    // Rot180 needs both width and height aligned, so trim removes both partial edges.
    let data: Vec<u8> = make_test_jpeg(30, 30, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        op: TransformOp::Rot180,
        trim: true,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let image = decompress(&result).unwrap();
    // Both dimensions should be trimmed to MCU boundary (16x16 for 4:2:0).
    assert_eq!(image.width % 16, 0);
    assert_eq!(image.height % 16, 0);
    // Should be 16x16 (floor of 30 to 16).
    assert_eq!(image.width, 16);
    assert_eq!(image.height, 16);
}

// --- Crop flag ---

#[test]
fn transform_crop_produces_correct_size() {
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        crop: Some(CropRegion {
            x: 0,
            y: 0,
            width: 32,
            height: 32,
        }),
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let coeffs = read_coefficients(&result).unwrap();
    // Width and height in pixels should match crop request (MCU-aligned).
    assert!(coeffs.width <= 32);
    assert!(coeffs.height <= 32);
}

// --- Combined transform + grayscale ---

#[test]
fn transform_rot90_with_grayscale() {
    let data: Vec<u8> = make_test_jpeg(64, 48, Subsampling::S420);
    let opts: TransformOptions = TransformOptions {
        op: TransformOp::Rot90,
        grayscale: true,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let image = decompress(&result).unwrap();
    // Rot90 swaps dimensions.
    assert_eq!(image.width, 48);
    assert_eq!(image.height, 64);
    // Should be grayscale (1 component).
    let coeffs = read_coefficients(&result).unwrap();
    assert_eq!(coeffs.components.len(), 1);
}

// --- transform_jpeg_with_options as identity ---

#[test]
fn transform_with_default_options_matches_original() {
    let data: Vec<u8> = make_test_jpeg(64, 64, Subsampling::S420);
    let original = decompress(&data).unwrap();

    let result: Vec<u8> =
        libjpeg_turbo_rs::transform_jpeg_with_options(&data, &TransformOptions::default()).unwrap();
    let transformed = decompress(&result).unwrap();

    assert_eq!(original.width, transformed.width);
    assert_eq!(original.height, transformed.height);
    assert_eq!(original.data, transformed.data);
}
