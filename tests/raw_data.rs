/// Tests for raw data encode/decode (jpeg_write_raw_data / jpeg_read_raw_data equivalent).
///
/// Raw data mode encodes from / decodes to pre-downsampled component planes
/// (e.g., separate Y, Cb, Cr at their native subsampled resolution),
/// bypassing color conversion and chroma downsampling/upsampling.
use libjpeg_turbo_rs::{
    compress, compress_raw, decompress_raw, PixelFormat, RawImage, Subsampling,
};

/// Maximum per-pixel difference allowed in lossy roundtrip tests.
/// JPEG DCT+quantization at quality 95 typically introduces up to ~6 error.
const MAX_DIFF: i16 = 8;

// ---------------------------------------------------------------------------
// Grayscale roundtrip
// ---------------------------------------------------------------------------

#[test]
fn raw_grayscale_roundtrip() {
    let width: usize = 32;
    let height: usize = 24;
    // Single Y plane with a gradient pattern
    let y_plane: Vec<u8> = (0..width * height)
        .map(|i| ((i * 7 + 13) % 256) as u8)
        .collect();

    let jpeg_data: Vec<u8> = compress_raw(
        &[&y_plane],
        &[width],
        &[height],
        width,
        height,
        95,
        Subsampling::S444, // subsampling is irrelevant for grayscale
    )
    .expect("compress_raw grayscale should succeed");

    // Verify it produces valid JPEG
    assert!(jpeg_data.len() > 2);
    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI marker

    // Decompress back to raw planes
    let raw: RawImage = decompress_raw(&jpeg_data).expect("decompress_raw should succeed");

    assert_eq!(raw.width, width);
    assert_eq!(raw.height, height);
    assert_eq!(raw.num_components, 1);
    assert_eq!(raw.planes.len(), 1);
    assert_eq!(raw.plane_widths.len(), 1);
    assert_eq!(raw.plane_heights.len(), 1);
    // Plane dimensions should match the original image dimensions (padded to MCU boundary)
    assert!(raw.plane_widths[0] >= width);
    assert!(raw.plane_heights[0] >= height);

    // Check pixel values survive the roundtrip within JPEG lossy tolerance
    for row in 0..height {
        for col in 0..width {
            let original: u8 = y_plane[row * width + col];
            let decoded: u8 = raw.planes[0][row * raw.plane_widths[0] + col];
            let diff: i16 = original as i16 - decoded as i16;
            assert!(
                diff.abs() <= MAX_DIFF,
                "grayscale pixel ({},{}) differs: orig={} decoded={} diff={}",
                col,
                row,
                original,
                decoded,
                diff
            );
        }
    }
}

// ---------------------------------------------------------------------------
// YCbCr 4:2:0 roundtrip
// ---------------------------------------------------------------------------

#[test]
fn raw_420_roundtrip() {
    let image_width: usize = 48;
    let image_height: usize = 32;

    // Y plane: full resolution
    let y_w: usize = image_width;
    let y_h: usize = image_height;
    let y_plane: Vec<u8> = (0..y_w * y_h).map(|i| ((i * 3 + 7) % 256) as u8).collect();

    // Cb, Cr planes: half resolution in both dimensions
    let c_w: usize = image_width / 2;
    let c_h: usize = image_height / 2;
    let cb_plane: Vec<u8> = (0..c_w * c_h)
        .map(|i| ((i * 5 + 100) % 256) as u8)
        .collect();
    let cr_plane: Vec<u8> = (0..c_w * c_h)
        .map(|i| ((i * 11 + 200) % 256) as u8)
        .collect();

    let jpeg_data: Vec<u8> = compress_raw(
        &[&y_plane, &cb_plane, &cr_plane],
        &[y_w, c_w, c_w],
        &[y_h, c_h, c_h],
        image_width,
        image_height,
        95,
        Subsampling::S420,
    )
    .expect("compress_raw 4:2:0 should succeed");

    let raw: RawImage = decompress_raw(&jpeg_data).expect("decompress_raw 4:2:0 should succeed");

    assert_eq!(raw.width, image_width);
    assert_eq!(raw.height, image_height);
    assert_eq!(raw.num_components, 3);
    assert_eq!(raw.planes.len(), 3);

    // Y plane should be at full resolution (MCU-aligned)
    assert!(raw.plane_widths[0] >= image_width);
    assert!(raw.plane_heights[0] >= image_height);

    // Cb/Cr planes should be at half resolution (MCU-aligned)
    assert!(raw.plane_widths[1] >= c_w);
    assert!(raw.plane_heights[1] >= c_h);
    assert!(raw.plane_widths[2] >= c_w);
    assert!(raw.plane_heights[2] >= c_h);

    // Verify Y plane values survive roundtrip
    for row in 0..y_h {
        for col in 0..y_w {
            let original: u8 = y_plane[row * y_w + col];
            let decoded: u8 = raw.planes[0][row * raw.plane_widths[0] + col];
            let diff: i16 = original as i16 - decoded as i16;
            assert!(
                diff.abs() <= MAX_DIFF,
                "Y pixel ({},{}) differs: orig={} decoded={} diff={}",
                col,
                row,
                original,
                decoded,
                diff
            );
        }
    }

    // Verify Cb plane values survive roundtrip
    for row in 0..c_h {
        for col in 0..c_w {
            let original: u8 = cb_plane[row * c_w + col];
            let decoded: u8 = raw.planes[1][row * raw.plane_widths[1] + col];
            let diff: i16 = original as i16 - decoded as i16;
            assert!(
                diff.abs() <= MAX_DIFF,
                "Cb pixel ({},{}) differs: orig={} decoded={} diff={}",
                col,
                row,
                original,
                decoded,
                diff
            );
        }
    }
}

// ---------------------------------------------------------------------------
// YCbCr 4:4:4 roundtrip
// ---------------------------------------------------------------------------

#[test]
fn raw_444_roundtrip() {
    let width: usize = 40;
    let height: usize = 30;

    // All three planes at full resolution
    let y_plane: Vec<u8> = (0..width * height)
        .map(|i| ((i * 3 + 50) % 256) as u8)
        .collect();
    let cb_plane: Vec<u8> = (0..width * height)
        .map(|i| ((i * 7 + 120) % 256) as u8)
        .collect();
    let cr_plane: Vec<u8> = (0..width * height)
        .map(|i| ((i * 11 + 180) % 256) as u8)
        .collect();

    let jpeg_data: Vec<u8> = compress_raw(
        &[&y_plane, &cb_plane, &cr_plane],
        &[width, width, width],
        &[height, height, height],
        width,
        height,
        95,
        Subsampling::S444,
    )
    .expect("compress_raw 4:4:4 should succeed");

    let raw: RawImage = decompress_raw(&jpeg_data).expect("decompress_raw 4:4:4 should succeed");

    assert_eq!(raw.width, width);
    assert_eq!(raw.height, height);
    assert_eq!(raw.num_components, 3);

    // All planes should be at full resolution (MCU-aligned)
    assert!(raw.plane_widths[0] >= width);
    assert!(raw.plane_heights[0] >= height);
    assert!(raw.plane_widths[1] >= width);
    assert!(raw.plane_heights[1] >= height);
    assert!(raw.plane_widths[2] >= width);
    assert!(raw.plane_heights[2] >= height);

    // Verify roundtrip values
    for row in 0..height {
        for col in 0..width {
            let orig_y: u8 = y_plane[row * width + col];
            let dec_y: u8 = raw.planes[0][row * raw.plane_widths[0] + col];
            let diff: i16 = orig_y as i16 - dec_y as i16;
            assert!(
                diff.abs() <= MAX_DIFF,
                "Y pixel ({},{}) differs: {} vs {}",
                col,
                row,
                orig_y,
                dec_y
            );
        }
    }
}

// ---------------------------------------------------------------------------
// decompress_raw from a standard (non-raw) JPEG produces correct plane dimensions
// ---------------------------------------------------------------------------

#[test]
fn decompress_raw_from_standard_jpeg() {
    // Create a standard JPEG from RGB pixels
    let width: usize = 64;
    let height: usize = 48;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i * 13 + 7) % 256) as u8)
        .collect();

    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .expect("compress should succeed");

    let raw: RawImage =
        decompress_raw(&jpeg_data).expect("decompress_raw from standard JPEG should succeed");

    assert_eq!(raw.width, width);
    assert_eq!(raw.height, height);
    assert_eq!(raw.num_components, 3);

    // For 4:2:0: Y is full res, Cb/Cr are half in each dimension
    // Plane widths/heights are MCU-aligned
    assert!(raw.plane_widths[0] >= width);
    assert!(raw.plane_heights[0] >= height);
    assert!(raw.plane_widths[1] >= width / 2);
    assert!(raw.plane_heights[1] >= height / 2);
    assert!(raw.plane_widths[2] >= width / 2);
    assert!(raw.plane_heights[2] >= height / 2);

    // Y plane should have reasonable luminance values (not all zeros)
    let y_sum: u64 = raw.planes[0].iter().map(|&v| v as u64).sum();
    assert!(y_sum > 0, "Y plane should not be all zeros");
}

// ---------------------------------------------------------------------------
// compress_raw with wrong plane sizes returns error
// ---------------------------------------------------------------------------

#[test]
fn compress_raw_wrong_plane_sizes_returns_error() {
    let width: usize = 32;
    let height: usize = 24;
    let y_plane: Vec<u8> = vec![128u8; width * height];

    // Cb/Cr planes are wrong size for 4:2:0 (should be 16x12 but we provide 32x24)
    let cb_plane: Vec<u8> = vec![128u8; width * height]; // too big
    let cr_plane: Vec<u8> = vec![128u8; width * height]; // too big

    let result = compress_raw(
        &[&y_plane, &cb_plane, &cr_plane],
        &[width, width, width],    // Cb/Cr widths should be width/2 for 4:2:0
        &[height, height, height], // Cb/Cr heights should be height/2 for 4:2:0
        width,
        height,
        90,
        Subsampling::S420,
    );

    assert!(
        result.is_err(),
        "compress_raw with wrong plane dimensions for 4:2:0 should fail"
    );
}

#[test]
fn compress_raw_wrong_plane_count_returns_error() {
    let width: usize = 32;
    let height: usize = 24;
    let y_plane: Vec<u8> = vec![128u8; width * height];

    // Only 1 plane for S420 (requires 3)
    let result = compress_raw(
        &[&y_plane],
        &[width],
        &[height],
        width,
        height,
        90,
        Subsampling::S420,
    );

    assert!(
        result.is_err(),
        "compress_raw with 1 plane for 4:2:0 should fail"
    );
}

#[test]
fn compress_raw_plane_data_too_small_returns_error() {
    let width: usize = 32;
    let height: usize = 24;
    let y_plane: Vec<u8> = vec![128u8; 10]; // way too small

    let result = compress_raw(
        &[&y_plane],
        &[width],
        &[height],
        width,
        height,
        90,
        Subsampling::S444,
    );

    assert!(
        result.is_err(),
        "compress_raw with undersized plane data should fail"
    );
}
