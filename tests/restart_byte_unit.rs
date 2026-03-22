//! Byte-unit (MCU block) restart interval tests.
//!
//! The C libjpeg-turbo test suite tests `-r 1b` which sets the restart
//! interval to 1 MCU block (the smallest possible). Our API exposes this
//! via `Encoder::restart_blocks(n)` where `n` is the number of MCU blocks
//! between restart markers.
//!
//! This test file:
//! - Verifies restart_blocks(1) produces DRI marker and RST markers
//! - Tests that the DRI marker value equals 1
//! - Verifies roundtrip correctness with the smallest restart interval
//! - Tests multiple subsampling modes and image sizes

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count occurrences of a 2-byte marker in JPEG data.
fn count_marker(data: &[u8], marker_byte: u8) -> usize {
    data.windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == marker_byte)
        .count()
}

/// Count all RST markers (0xFFD0 through 0xFFD7) in JPEG data.
fn count_rst_markers(data: &[u8]) -> usize {
    data.windows(2)
        .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
        .count()
}

/// Find DRI marker and return the restart interval value it contains.
/// DRI format: FF DD 00 04 Rr Rr (2-byte interval value, big-endian).
fn read_dri_interval(data: &[u8]) -> Option<u16> {
    for i in 0..data.len().saturating_sub(5) {
        if data[i] == 0xFF && data[i + 1] == 0xDD {
            // length is at i+2..i+4 (should be 0x0004)
            let interval: u16 = ((data[i + 4] as u16) << 8) | data[i + 5] as u16;
            return Some(interval);
        }
    }
    None
}

/// Create a gradient pixel buffer.
fn make_pixels(width: usize, height: usize, bpp: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * bpp);
    for y in 0..height {
        for x in 0..width {
            for c in 0..bpp {
                pixels.push(((x * 7 + y * 13 + c * 31) % 256) as u8);
            }
        }
    }
    pixels
}

// ===========================================================================
// restart_blocks(1) — smallest interval (every single MCU)
// ===========================================================================

#[test]
fn restart_blocks_1_has_dri_marker() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let dri_count: usize = count_marker(&jpeg, 0xDD);
    assert!(dri_count >= 1, "DRI marker (0xFFDD) must be present");
}

#[test]
fn restart_blocks_1_dri_value_equals_1() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let interval: u16 = read_dri_interval(&jpeg).expect("DRI marker not found");
    assert_eq!(
        interval, 1,
        "DRI interval should be 1 for restart_blocks(1), got {}",
        interval
    );
}

#[test]
fn restart_blocks_1_has_rst_markers_in_entropy_data() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let rst_count: usize = count_rst_markers(&jpeg);
    // 32x32 with S444 = 4x4 = 16 MCUs, so 15 RST markers expected (between MCUs)
    assert!(rst_count > 0, "RST markers must be present in entropy data");
    // With restart_blocks(1), there should be (total_mcus - 1) RST markers
    // For S444 32x32: MCU = 8x8, so 4*4 = 16 MCUs, 15 RSTs
    assert!(
        rst_count >= 10,
        "expected many RST markers for restart_blocks(1), got {}",
        rst_count
    );
}

#[test]
fn restart_blocks_1_roundtrip_444() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(90)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.data.len(), 32 * 32 * 3);
}

#[test]
fn restart_blocks_1_roundtrip_420() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(90)
        .subsampling(Subsampling::S420)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.data.len(), 32 * 32 * 3);
}

#[test]
fn restart_blocks_1_roundtrip_422() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(90)
        .subsampling(Subsampling::S422)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.data.len(), 32 * 32 * 3);
}

#[test]
fn restart_blocks_1_roundtrip_grayscale() {
    let pixels: Vec<u8> = make_pixels(32, 32, 1);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Grayscale)
        .quality(90)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
}

// ===========================================================================
// restart_blocks(1) with non-MCU-aligned dimensions
// ===========================================================================

#[test]
fn restart_blocks_1_non_mcu_aligned_444() {
    // 37x29: not divisible by 8 (MCU size for S444)
    let pixels: Vec<u8> = make_pixels(37, 29, 3);
    let jpeg = Encoder::new(&pixels, 37, 29, PixelFormat::Rgb)
        .quality(85)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 37);
    assert_eq!(img.height, 29);

    let interval: u16 = read_dri_interval(&jpeg).expect("DRI marker not found");
    assert_eq!(interval, 1);
}

#[test]
fn restart_blocks_1_non_mcu_aligned_420() {
    // 37x29: not divisible by 16 (MCU size for S420)
    let pixels: Vec<u8> = make_pixels(37, 29, 3);
    let jpeg = Encoder::new(&pixels, 37, 29, PixelFormat::Rgb)
        .quality(85)
        .subsampling(Subsampling::S420)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 37);
    assert_eq!(img.height, 29);

    let interval: u16 = read_dri_interval(&jpeg).expect("DRI marker not found");
    assert_eq!(interval, 1);
}

// ===========================================================================
// Comparison: restart_blocks(1) vs no restart
// ===========================================================================

#[test]
fn restart_blocks_1_decode_matches_no_restart_quality() {
    // Image content should be identical to without restart markers (at same quality)
    let pixels: Vec<u8> = make_pixels(32, 32, 3);

    let jpeg_with_rst = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let jpeg_no_rst = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(100)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    let img_rst = decompress(&jpeg_with_rst).unwrap();
    let img_no = decompress(&jpeg_no_rst).unwrap();

    assert_eq!(img_rst.width, img_no.width);
    assert_eq!(img_rst.height, img_no.height);
    // At quality 100, restart markers should not change the pixel values
    assert_eq!(
        img_rst.data, img_no.data,
        "restart_blocks(1) at q100 should produce identical pixels to no restart"
    );
}

// ===========================================================================
// restart_blocks(1) produces larger file than without restart
// ===========================================================================

#[test]
fn restart_blocks_1_increases_file_size() {
    let pixels: Vec<u8> = make_pixels(64, 64, 3);

    let jpeg_with_rst = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let jpeg_no_rst = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    // RST markers add overhead
    assert!(
        jpeg_with_rst.len() > jpeg_no_rst.len(),
        "restart_blocks(1) JPEG ({} bytes) should be larger than no-restart ({} bytes)",
        jpeg_with_rst.len(),
        jpeg_no_rst.len()
    );
}

// ===========================================================================
// Larger restart intervals for comparison
// ===========================================================================

#[test]
fn restart_blocks_2_has_fewer_rst_than_blocks_1() {
    let pixels: Vec<u8> = make_pixels(64, 64, 3);

    let jpeg_1 = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(1)
        .encode()
        .unwrap();

    let jpeg_2 = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(2)
        .encode()
        .unwrap();

    let rst_1: usize = count_rst_markers(&jpeg_1);
    let rst_2: usize = count_rst_markers(&jpeg_2);

    assert!(
        rst_1 > rst_2,
        "restart_blocks(1) should have more RST markers ({}) than restart_blocks(2) ({})",
        rst_1,
        rst_2
    );
}

#[test]
fn restart_blocks_2_dri_value_equals_2() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .restart_blocks(2)
        .encode()
        .unwrap();

    let interval: u16 = read_dri_interval(&jpeg).expect("DRI marker not found");
    assert_eq!(
        interval, 2,
        "DRI interval should be 2 for restart_blocks(2)"
    );
}

// ===========================================================================
// Decode existing fixture with restart markers
// ===========================================================================

#[test]
fn decode_fixture_with_restart_markers() {
    let data = include_bytes!("fixtures/photo_640x480_420_rst.jpg");
    let img = decompress(data).unwrap();
    // Despite the filename, the actual image dimensions are 320x240
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    // Verify it has RST markers
    let rst_count: usize = count_rst_markers(data);
    assert!(
        rst_count > 0,
        "fixture photo_640x480_420_rst.jpg should contain RST markers"
    );
}
