//! C-compatible crop region tests.
//!
//! The C libjpeg-turbo test suite tests these 5 exact crop regions on a
//! 64x64 image: 14x14+23+23, 21x21+4+4, 18x18+13+13, 21x21+0+0, 24x26+20+18
//!
//! This test file verifies our cropped decompress API produces correct results
//! for these exact coordinates, matching C behavior:
//! - Output dimensions match requested crop (clamped to image bounds)
//! - Pixel values match full decode at the same coordinates
//! - Works with both 4:4:4 and 4:2:0 subsampling

use libjpeg_turbo_rs::{
    compress, decompress, decompress_cropped, CropRegion, PixelFormat, Subsampling,
};

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Generate a 64x64 gradient test image, encode as JPEG, return bytes.
fn encode_64x64(subsampling: Subsampling) -> Vec<u8> {
    let (w, h) = (64, 64);
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            // Distinct per-pixel pattern so crop correctness is verifiable
            pixels.push(((x * 4 + y) % 256) as u8);
            pixels.push(((y * 4 + x) % 256) as u8);
            pixels.push(((x * y) % 256) as u8);
        }
    }
    compress(&pixels, w, h, PixelFormat::Rgb, 95, subsampling).unwrap()
}

/// Verify that cropped pixels match the corresponding region in a full decode.
fn verify_crop_matches_full(
    jpeg_data: &[u8],
    crop_x: usize,
    crop_y: usize,
    crop_w: usize,
    crop_h: usize,
) {
    let full = decompress(jpeg_data).unwrap();
    let bpp: usize = full.pixel_format.bytes_per_pixel();
    let region = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_w,
        height: crop_h,
    };
    let cropped = decompress_cropped(jpeg_data, region).unwrap();

    // The crop API clamps to image bounds
    let effective_w: usize = crop_w.min(full.width.saturating_sub(crop_x));
    let effective_h: usize = crop_h.min(full.height.saturating_sub(crop_y));

    assert_eq!(
        cropped.width, effective_w,
        "crop {}x{}+{}+{}: width mismatch (expected {}, got {})",
        crop_w, crop_h, crop_x, crop_y, effective_w, cropped.width
    );
    assert_eq!(
        cropped.height, effective_h,
        "crop {}x{}+{}+{}: height mismatch (expected {}, got {})",
        crop_w, crop_h, crop_x, crop_y, effective_h, cropped.height
    );
    assert_eq!(
        cropped.data.len(),
        effective_w * effective_h * bpp,
        "crop {}x{}+{}+{}: data length mismatch",
        crop_w,
        crop_h,
        crop_x,
        crop_y
    );

    // Pixel-by-pixel comparison with full decode
    for row in 0..effective_h {
        for col in 0..effective_w {
            let crop_idx: usize = (row * effective_w + col) * bpp;
            let full_idx: usize = ((crop_y + row) * full.width + (crop_x + col)) * bpp;
            for c in 0..bpp {
                assert_eq!(
                    cropped.data[crop_idx + c],
                    full.data[full_idx + c],
                    "crop {}x{}+{}+{}: pixel mismatch at row={}, col={}, channel={}",
                    crop_w,
                    crop_h,
                    crop_x,
                    crop_y,
                    row,
                    col,
                    c
                );
            }
        }
    }
}

// ===========================================================================
// C test coordinates on 4:4:4 subsampling
// ===========================================================================

#[test]
fn crop_14x14_at_23_23_444() {
    let jpeg = encode_64x64(Subsampling::S444);
    verify_crop_matches_full(&jpeg, 23, 23, 14, 14);
}

#[test]
fn crop_21x21_at_4_4_444() {
    let jpeg = encode_64x64(Subsampling::S444);
    verify_crop_matches_full(&jpeg, 4, 4, 21, 21);
}

#[test]
fn crop_18x18_at_13_13_444() {
    let jpeg = encode_64x64(Subsampling::S444);
    verify_crop_matches_full(&jpeg, 13, 13, 18, 18);
}

#[test]
fn crop_21x21_at_0_0_444() {
    let jpeg = encode_64x64(Subsampling::S444);
    verify_crop_matches_full(&jpeg, 0, 0, 21, 21);
}

#[test]
fn crop_24x26_at_20_18_444() {
    let jpeg = encode_64x64(Subsampling::S444);
    verify_crop_matches_full(&jpeg, 20, 18, 24, 26);
}

// ===========================================================================
// C test coordinates on 4:2:0 subsampling
// ===========================================================================

#[test]
fn crop_14x14_at_23_23_420() {
    let jpeg = encode_64x64(Subsampling::S420);
    verify_crop_matches_full(&jpeg, 23, 23, 14, 14);
}

#[test]
fn crop_21x21_at_4_4_420() {
    let jpeg = encode_64x64(Subsampling::S420);
    verify_crop_matches_full(&jpeg, 4, 4, 21, 21);
}

#[test]
fn crop_18x18_at_13_13_420() {
    let jpeg = encode_64x64(Subsampling::S420);
    verify_crop_matches_full(&jpeg, 13, 13, 18, 18);
}

#[test]
fn crop_21x21_at_0_0_420() {
    let jpeg = encode_64x64(Subsampling::S420);
    verify_crop_matches_full(&jpeg, 0, 0, 21, 21);
}

#[test]
fn crop_24x26_at_20_18_420() {
    let jpeg = encode_64x64(Subsampling::S420);
    verify_crop_matches_full(&jpeg, 20, 18, 24, 26);
}

// ===========================================================================
// C test coordinates with real photo fixture (photo_64x64_420.jpg)
// ===========================================================================

#[test]
fn crop_14x14_at_23_23_photo() {
    let jpeg = include_bytes!("fixtures/photo_64x64_420.jpg");
    verify_crop_matches_full(jpeg, 23, 23, 14, 14);
}

#[test]
fn crop_21x21_at_4_4_photo() {
    let jpeg = include_bytes!("fixtures/photo_64x64_420.jpg");
    verify_crop_matches_full(jpeg, 4, 4, 21, 21);
}

#[test]
fn crop_18x18_at_13_13_photo() {
    let jpeg = include_bytes!("fixtures/photo_64x64_420.jpg");
    verify_crop_matches_full(jpeg, 13, 13, 18, 18);
}

#[test]
fn crop_21x21_at_0_0_photo() {
    let jpeg = include_bytes!("fixtures/photo_64x64_420.jpg");
    verify_crop_matches_full(jpeg, 0, 0, 21, 21);
}

#[test]
fn crop_24x26_at_20_18_photo() {
    let jpeg = include_bytes!("fixtures/photo_64x64_420.jpg");
    verify_crop_matches_full(jpeg, 20, 18, 24, 26);
}

// ===========================================================================
// Edge cases derived from C coordinates
// ===========================================================================

#[test]
fn crop_extends_beyond_image_bounds() {
    // C coordinate 24x26+20+18 on 64x64: x+w=44, y+h=44, within bounds.
    // Test a crop that WOULD exceed: 24x26+50+50 on 64x64.
    let jpeg = encode_64x64(Subsampling::S444);
    let region = CropRegion {
        x: 50,
        y: 50,
        width: 24,
        height: 26,
    };
    let cropped = decompress_cropped(&jpeg, region).unwrap();
    // Clamped: effective_w = min(24, 64-50) = 14, effective_h = min(26, 64-50) = 14
    assert_eq!(cropped.width, 14);
    assert_eq!(cropped.height, 14);
}

#[test]
fn crop_all_five_c_regions_sequential_444() {
    // Verify all 5 C test crops produce valid output in sequence on same image
    let jpeg = encode_64x64(Subsampling::S444);
    let regions: Vec<(usize, usize, usize, usize)> = vec![
        (23, 23, 14, 14),
        (4, 4, 21, 21),
        (13, 13, 18, 18),
        (0, 0, 21, 21),
        (20, 18, 24, 26),
    ];
    for (x, y, w, h) in regions {
        let region = CropRegion {
            x,
            y,
            width: w,
            height: h,
        };
        let cropped = decompress_cropped(&jpeg, region).unwrap();
        assert_eq!(cropped.width, w, "crop {}x{}+{}+{} width", w, h, x, y);
        assert_eq!(cropped.height, h, "crop {}x{}+{}+{} height", w, h, x, y);
        assert!(
            !cropped.data.is_empty(),
            "crop {}x{}+{}+{} should produce non-empty data",
            w,
            h,
            x,
            y
        );
    }
}

#[test]
fn crop_non_mcu_aligned_offsets_420() {
    // 4:2:0 has 16x16 MCU blocks. The C test coordinates (23,23), (4,4),
    // (13,13) are intentionally non-MCU-aligned to test sub-MCU extraction.
    let jpeg = encode_64x64(Subsampling::S420);
    let non_aligned_offsets: Vec<(usize, usize)> = vec![(23, 23), (4, 4), (13, 13)];
    for (ox, oy) in non_aligned_offsets {
        let region = CropRegion {
            x: ox,
            y: oy,
            width: 16,
            height: 16,
        };
        let cropped = decompress_cropped(&jpeg, region).unwrap();
        assert_eq!(
            cropped.width, 16,
            "non-aligned crop at ({},{}) width",
            ox, oy
        );
        assert_eq!(
            cropped.height, 16,
            "non-aligned crop at ({},{}) height",
            ox, oy
        );
    }
}

#[test]
fn crop_full_image_64x64() {
    // Crop the entire image should match full decode
    let jpeg = encode_64x64(Subsampling::S444);
    verify_crop_matches_full(&jpeg, 0, 0, 64, 64);
}
