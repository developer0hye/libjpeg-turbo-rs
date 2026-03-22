//! Tests for pathological image dimensions.
//!
//! Encode then decode with extreme sizes, odd aspect ratios, non-MCU-aligned
//! dimensions, and quality extremes. Every test verifies the roundtrip
//! succeeds with correct output dimensions.

use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

// ---------------------------------------------------------------------------
// Helper: roundtrip encode → decode and assert dimensions match
// ---------------------------------------------------------------------------

fn roundtrip_rgb(width: usize, height: usize, quality: u8, subsampling: Subsampling) {
    let bpp: usize = 3;
    let pixels: Vec<u8> = (0..width * height * bpp)
        .map(|i| (i % 251) as u8) // pseudo-random but deterministic
        .collect();
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        quality,
        subsampling,
    )
    .unwrap_or_else(|e| {
        panic!(
            "compress {}x{} q{} {:?} failed: {}",
            width, height, quality, subsampling, e
        )
    });
    let img = decompress(&jpeg).unwrap_or_else(|e| {
        panic!(
            "decompress {}x{} q{} {:?} failed: {}",
            width, height, quality, subsampling, e
        )
    });
    assert_eq!(
        img.width, width,
        "width mismatch for {}x{} {:?}",
        width, height, subsampling
    );
    assert_eq!(
        img.height, height,
        "height mismatch for {}x{} {:?}",
        width, height, subsampling
    );
    assert_eq!(
        img.data.len(),
        width * height * bpp,
        "data length mismatch for {}x{} {:?}",
        width,
        height,
        subsampling,
    );
}

fn roundtrip_gray(width: usize, height: usize, quality: u8) {
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 251) as u8).collect();
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        quality,
        Subsampling::S444,
    )
    .unwrap_or_else(|e| {
        panic!(
            "compress gray {}x{} q{} failed: {}",
            width, height, quality, e
        )
    });
    let img = decompress(&jpeg).unwrap_or_else(|e| {
        panic!(
            "decompress gray {}x{} q{} failed: {}",
            width, height, quality, e
        )
    });
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.data.len(), width * height);
}

// ===========================================================================
// 1x1 pixel — smallest possible image
// ===========================================================================

#[test]
fn one_by_one_s444() {
    roundtrip_rgb(1, 1, 75, Subsampling::S444);
}

#[test]
fn one_by_one_s422() {
    roundtrip_rgb(1, 1, 75, Subsampling::S422);
}

#[test]
fn one_by_one_s420() {
    roundtrip_rgb(1, 1, 75, Subsampling::S420);
}

#[test]
fn one_by_one_s440() {
    roundtrip_rgb(1, 1, 75, Subsampling::S440);
}

#[test]
fn one_by_one_s411() {
    roundtrip_rgb(1, 1, 75, Subsampling::S411);
}

#[test]
fn one_by_one_s441() {
    roundtrip_rgb(1, 1, 75, Subsampling::S441);
}

#[test]
fn one_by_one_grayscale() {
    roundtrip_gray(1, 1, 75);
}

// ===========================================================================
// 1x2 and 2x1 pixels — minimal multi-pixel
// ===========================================================================

#[test]
fn one_by_two() {
    roundtrip_rgb(1, 2, 75, Subsampling::S444);
}

#[test]
fn two_by_one() {
    roundtrip_rgb(2, 1, 75, Subsampling::S444);
}

#[test]
fn one_by_two_s420() {
    roundtrip_rgb(1, 2, 75, Subsampling::S420);
}

#[test]
fn two_by_one_s420() {
    roundtrip_rgb(2, 1, 75, Subsampling::S420);
}

// ===========================================================================
// Extreme aspect ratios
// ===========================================================================

#[test]
fn one_by_hundred() {
    roundtrip_rgb(1, 100, 75, Subsampling::S444);
}

#[test]
fn hundred_by_one() {
    roundtrip_rgb(100, 1, 75, Subsampling::S444);
}

#[test]
fn one_by_hundred_s420() {
    roundtrip_rgb(1, 100, 75, Subsampling::S420);
}

#[test]
fn hundred_by_one_s420() {
    roundtrip_rgb(100, 1, 75, Subsampling::S420);
}

// ===========================================================================
// Non-MCU-aligned dimensions
// ===========================================================================

#[test]
fn seven_by_seven_s444() {
    roundtrip_rgb(7, 7, 75, Subsampling::S444);
}

#[test]
fn seven_by_seven_s422() {
    roundtrip_rgb(7, 7, 75, Subsampling::S422);
}

#[test]
fn seven_by_seven_s420() {
    roundtrip_rgb(7, 7, 75, Subsampling::S420);
}

#[test]
fn seven_by_seven_s440() {
    roundtrip_rgb(7, 7, 75, Subsampling::S440);
}

#[test]
fn seven_by_seven_s411() {
    roundtrip_rgb(7, 7, 75, Subsampling::S411);
}

#[test]
fn seven_by_seven_s441() {
    roundtrip_rgb(7, 7, 75, Subsampling::S441);
}

#[test]
fn fifteen_by_fifteen_s420() {
    // 4:2:0 MCU is 16x16, so 15x15 has partial MCU in both directions
    roundtrip_rgb(15, 15, 75, Subsampling::S420);
}

#[test]
fn thirty_one_by_seventeen_s411() {
    // 4:1:1 MCU width=32, so 31x17 has partial MCU in horizontal direction
    roundtrip_rgb(31, 17, 75, Subsampling::S411);
}

// ===========================================================================
// Prime dimensions (no divisibility tricks possible)
// ===========================================================================

#[test]
fn prime_dimensions_s444() {
    roundtrip_rgb(1009, 1013, 75, Subsampling::S444);
}

#[test]
fn prime_dimensions_s420() {
    roundtrip_rgb(1009, 1013, 75, Subsampling::S420);
}

#[test]
fn prime_dimensions_s411() {
    roundtrip_rgb(1009, 1013, 75, Subsampling::S411);
}

// ===========================================================================
// Quality extremes
// ===========================================================================

#[test]
fn quality_1_minimum() {
    roundtrip_rgb(16, 16, 1, Subsampling::S444);
}

#[test]
fn quality_100_maximum() {
    roundtrip_rgb(16, 16, 100, Subsampling::S444);
}

#[test]
fn quality_1_large_image() {
    roundtrip_rgb(64, 64, 1, Subsampling::S420);
}

#[test]
fn quality_100_large_image() {
    roundtrip_rgb(64, 64, 100, Subsampling::S420);
}

#[test]
fn quality_1_grayscale() {
    roundtrip_gray(16, 16, 1);
}

#[test]
fn quality_100_grayscale() {
    roundtrip_gray(16, 16, 100);
}

// ===========================================================================
// All subsampling modes x odd dimensions
// ===========================================================================

#[test]
fn odd_3x5_s444() {
    roundtrip_rgb(3, 5, 75, Subsampling::S444);
}

#[test]
fn odd_3x5_s422() {
    roundtrip_rgb(3, 5, 75, Subsampling::S422);
}

#[test]
fn odd_3x5_s420() {
    roundtrip_rgb(3, 5, 75, Subsampling::S420);
}

#[test]
fn odd_3x5_s440() {
    roundtrip_rgb(3, 5, 75, Subsampling::S440);
}

#[test]
fn odd_3x5_s411() {
    roundtrip_rgb(3, 5, 75, Subsampling::S411);
}

#[test]
fn odd_3x5_s441() {
    roundtrip_rgb(3, 5, 75, Subsampling::S441);
}

#[test]
fn odd_9x11_s444() {
    roundtrip_rgb(9, 11, 75, Subsampling::S444);
}

#[test]
fn odd_9x11_s422() {
    roundtrip_rgb(9, 11, 75, Subsampling::S422);
}

#[test]
fn odd_9x11_s420() {
    roundtrip_rgb(9, 11, 75, Subsampling::S420);
}

#[test]
fn odd_9x11_s440() {
    roundtrip_rgb(9, 11, 75, Subsampling::S440);
}

#[test]
fn odd_9x11_s411() {
    roundtrip_rgb(9, 11, 75, Subsampling::S411);
}

#[test]
fn odd_9x11_s441() {
    roundtrip_rgb(9, 11, 75, Subsampling::S441);
}

// ===========================================================================
// Additional edge combos: quality extremes × odd dimensions × subsampling
// ===========================================================================

#[test]
fn q1_odd_5x3_s420() {
    roundtrip_rgb(5, 3, 1, Subsampling::S420);
}

#[test]
fn q100_odd_5x3_s420() {
    roundtrip_rgb(5, 3, 100, Subsampling::S420);
}

#[test]
fn q1_odd_13x7_s411() {
    roundtrip_rgb(13, 7, 1, Subsampling::S411);
}

#[test]
fn q100_odd_13x7_s411() {
    roundtrip_rgb(13, 7, 100, Subsampling::S411);
}

#[test]
fn q1_odd_11x9_s441() {
    roundtrip_rgb(11, 9, 1, Subsampling::S441);
}

#[test]
fn q100_odd_11x9_s441() {
    roundtrip_rgb(11, 9, 100, Subsampling::S441);
}
