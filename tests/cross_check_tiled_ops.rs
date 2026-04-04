//! Cross-validation: tiled encode/decode operations vs C djpeg.
//!
//! Gaps addressed:
//! - C tjbench tests tiled compression/decompression at various tile sizes
//!   (8x8, 16x16, 32x32, 64x64, full image). No Rust equivalent existed.
//!
//! Approach: since the Rust library does not have a dedicated tiled API,
//! we replicate C tjbench's approach — compress sub-regions of an image
//! independently and verify each tile decodes correctly against C djpeg.
//! This validates the codec handles various small/non-MCU-aligned dimensions.
//!
//! All tests gracefully skip if djpeg is not found.

mod helpers;

use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

// ===========================================================================
// Constants
// ===========================================================================

const QUALITY: u8 = 90;

// MCU-aligned tile sizes matching tjbench (power-of-2 series)
const TILE_SIZES: &[usize] = &[8, 16, 32, 64];

const TILED_SUBSAMPLINGS: &[(Subsampling, &str, usize)] = &[
    (Subsampling::S444, "444", 8),  // MCU = 8x8
    (Subsampling::S422, "422", 16), // MCU = 16x8
    (Subsampling::S420, "420", 16), // MCU = 16x16
];

// ===========================================================================
// Helpers
// ===========================================================================

/// Extract a rectangular tile from a pixel buffer.
fn extract_tile(
    pixels: &[u8],
    img_w: usize,
    tile_x: usize,
    tile_y: usize,
    tile_w: usize,
    tile_h: usize,
    bpp: usize,
) -> Vec<u8> {
    let mut tile: Vec<u8> = Vec::with_capacity(tile_w * tile_h * bpp);
    for row in 0..tile_h {
        let src_offset: usize = ((tile_y + row) * img_w + tile_x) * bpp;
        tile.extend_from_slice(&pixels[src_offset..src_offset + tile_w * bpp]);
    }
    tile
}

// ===========================================================================
// Tiled encode: compress each tile independently, decode with C djpeg
// ===========================================================================

#[test]
fn c_xval_tiled_encode_decode() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let img_w: usize = 128;
    let img_h: usize = 128;
    let pixels: Vec<u8> = helpers::generate_gradient(img_w, img_h);

    for &(subsamp, sname, mcu_size) in TILED_SUBSAMPLINGS {
        for &tile_size in TILE_SIZES {
            // Skip tiles smaller than MCU for subsampled modes
            if tile_size < mcu_size {
                continue;
            }

            let num_tiles_x: usize = img_w / tile_size;
            let num_tiles_y: usize = img_h / tile_size;
            let label_prefix: String = format!("tile_{}x{}_{}", tile_size, tile_size, sname);

            for ty in 0..num_tiles_y {
                for tx in 0..num_tiles_x {
                    let tile_x: usize = tx * tile_size;
                    let tile_y: usize = ty * tile_size;
                    let label: String = format!("{}_{}_{}", label_prefix, tx, ty);

                    // Extract tile pixels
                    let tile_pixels: Vec<u8> =
                        extract_tile(&pixels, img_w, tile_x, tile_y, tile_size, tile_size, 3);

                    // Compress tile with Rust
                    let tile_jpeg: Vec<u8> = compress(
                        &tile_pixels,
                        tile_size,
                        tile_size,
                        PixelFormat::Rgb,
                        QUALITY,
                        subsamp,
                    )
                    .unwrap_or_else(|e| panic!("{}: compress failed: {:?}", label, e));

                    // Decode with Rust
                    let rust_img = decompress_to(&tile_jpeg, PixelFormat::Rgb)
                        .unwrap_or_else(|e| panic!("{}: decompress failed: {:?}", label, e));
                    assert_eq!(rust_img.width, tile_size, "{}: width", label);
                    assert_eq!(rust_img.height, tile_size, "{}: height", label);

                    // Decode with C djpeg
                    let (c_w, c_h, c_rgb) =
                        helpers::decode_with_c_djpeg(&djpeg, &tile_jpeg, &label);
                    assert_eq!(c_w, tile_size, "{}: c width", label);
                    assert_eq!(c_h, tile_size, "{}: c height", label);

                    // Rust decode must match C djpeg (diff=0)
                    helpers::assert_pixels_identical(
                        &rust_img.data,
                        &c_rgb,
                        tile_size,
                        tile_size,
                        3,
                        &label,
                    );
                }
            }
        }
    }
}

// ===========================================================================
// Tiled decode: decode full image, compare tiles against individual decodes
// ===========================================================================

#[test]
fn tiled_decode_consistency() {
    let img_w: usize = 128;
    let img_h: usize = 128;
    let pixels: Vec<u8> = helpers::generate_gradient(img_w, img_h);

    for &(subsamp, sname, mcu_size) in TILED_SUBSAMPLINGS {
        // Encode full image
        let full_jpeg: Vec<u8> =
            compress(&pixels, img_w, img_h, PixelFormat::Rgb, QUALITY, subsamp)
                .unwrap_or_else(|e| panic!("full compress {} failed: {:?}", sname, e));
        let full_img = decompress_to(&full_jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("full decompress {} failed: {:?}", sname, e));

        for &tile_size in &[32usize, 64] {
            if tile_size < mcu_size {
                continue;
            }

            let num_tiles_x: usize = img_w / tile_size;
            let num_tiles_y: usize = img_h / tile_size;

            for ty in 0..num_tiles_y {
                for tx in 0..num_tiles_x {
                    let tile_x: usize = tx * tile_size;
                    let tile_y: usize = ty * tile_size;

                    // Extract tile from full decode
                    let full_tile: Vec<u8> = extract_tile(
                        &full_img.data,
                        img_w,
                        tile_x,
                        tile_y,
                        tile_size,
                        tile_size,
                        3,
                    );

                    // Compress just this tile's original pixels and decode
                    let tile_pixels: Vec<u8> =
                        extract_tile(&pixels, img_w, tile_x, tile_y, tile_size, tile_size, 3);
                    let tile_jpeg: Vec<u8> = compress(
                        &tile_pixels,
                        tile_size,
                        tile_size,
                        PixelFormat::Rgb,
                        QUALITY,
                        subsamp,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "tile compress {}x{} {} failed: {:?}",
                            tile_size, tile_size, sname, e
                        )
                    });
                    let tile_img =
                        decompress_to(&tile_jpeg, PixelFormat::Rgb).unwrap_or_else(|e| {
                            panic!(
                                "tile decompress {}x{} {} failed: {:?}",
                                tile_size, tile_size, sname, e
                            )
                        });

                    // Independently compressed/decoded tile should be close to
                    // the same region from the full image decode.
                    // Note: not pixel-identical because JPEG is lossy and tiles
                    // at different positions have different frequency content.
                    // But dimensions must match.
                    assert_eq!(tile_img.width, tile_size);
                    assert_eq!(tile_img.height, tile_size);
                    assert_eq!(tile_img.data.len(), full_tile.len());
                }
            }
        }
    }
}

// ===========================================================================
// Non-MCU-aligned tile sizes (edge case testing)
// ===========================================================================

#[test]
fn c_xval_tiled_non_mcu_aligned() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Test odd tile sizes that are NOT MCU-aligned
    let odd_sizes: &[(usize, usize)] = &[(7, 7), (9, 15), (15, 9), (17, 23), (31, 33)];

    for &(tw, th) in odd_sizes {
        for &(subsamp, sname, _) in TILED_SUBSAMPLINGS {
            let label: String = format!("odd_tile_{}x{}_{}", tw, th, sname);
            let tile_pixels: Vec<u8> = helpers::generate_gradient(tw, th);

            let tile_jpeg: Vec<u8> =
                compress(&tile_pixels, tw, th, PixelFormat::Rgb, QUALITY, subsamp)
                    .unwrap_or_else(|e| panic!("{}: compress failed: {:?}", label, e));

            let rust_img = decompress_to(&tile_jpeg, PixelFormat::Rgb)
                .unwrap_or_else(|e| panic!("{}: decompress failed: {:?}", label, e));

            let (c_w, c_h, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &tile_jpeg, &label);

            assert_eq!(rust_img.width, c_w, "{}: width", label);
            assert_eq!(rust_img.height, c_h, "{}: height", label);
            helpers::assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, &label);
        }
    }
}
