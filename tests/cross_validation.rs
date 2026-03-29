//! Cross-validation: compare our decoder output pixel-by-pixel against
//! C libjpeg-turbo (djpeg) reference output.
//!
//! Reference .rgb files were generated with:
//!   djpeg -ppm fixture.jpg | (skip PPM header) > fixture.rgb

use libjpeg_turbo_rs::decompress;

/// Maximum allowed per-channel difference between our output and the
/// C libjpeg-turbo reference. For 4:2:0, SIMD processing introduces
/// cumulative rounding differences across three stages:
/// (1) fancy chroma upsample (vertical blend + horizontal interpolation),
/// (2) YCbCr→RGB color conversion (i16 mulhi vs i32 precision), and
/// (3) the C reference itself has its own SIMD rounding.
/// Each stage can contribute ±1, giving a worst-case of ±3.
/// Only 4:2:0 images hit this — 4:2:2 and 4:4:4 stay within ±2.
const MAX_DIFF: u8 = 3;

fn assert_matches_reference(jpeg_path: &str, ref_path: &str, width: usize, height: usize) {
    let jpeg_data =
        std::fs::read(jpeg_path).unwrap_or_else(|_| panic!("missing fixture: {}", jpeg_path));
    let reference =
        std::fs::read(ref_path).unwrap_or_else(|_| panic!("missing reference: {}", ref_path));

    let image = decompress(&jpeg_data).unwrap();
    assert_eq!(image.width, width, "width mismatch for {}", jpeg_path);
    assert_eq!(image.height, height, "height mismatch for {}", jpeg_path);
    assert_eq!(
        image.data.len(),
        reference.len(),
        "data length mismatch for {}: ours={} ref={}",
        jpeg_path,
        image.data.len(),
        reference.len()
    );

    let mut max_seen: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&ours, &theirs)) in image.data.iter().zip(reference.iter()).enumerate() {
        let diff = (ours as i16 - theirs as i16).unsigned_abs() as u8;
        if diff > MAX_DIFF {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel = i / 3;
                let channel = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  pixel {} channel {}: ours={} ref={} diff={}",
                    pixel, channel, ours, theirs, diff
                );
            }
        }
        if diff > max_seen {
            max_seen = diff;
        }
    }

    assert_eq!(
        mismatches, 0,
        "{}: {} pixels differ by more than {} (max diff seen: {})",
        jpeg_path, mismatches, MAX_DIFF, max_seen
    );
}

// --- Resolution scaling (4:2:0) ---

#[test]
fn xval_photo_64x64_420() {
    assert_matches_reference(
        "tests/fixtures/photo_64x64_420.jpg",
        "tests/fixtures/ref/photo_64x64_420.rgb",
        64,
        64,
    );
}

#[test]
fn xval_photo_320x240_420() {
    assert_matches_reference(
        "tests/fixtures/photo_320x240_420.jpg",
        "tests/fixtures/ref/photo_320x240_420.rgb",
        320,
        240,
    );
}

// --- Subsampling modes ---

#[test]
fn xval_photo_320x240_444() {
    assert_matches_reference(
        "tests/fixtures/photo_320x240_444.jpg",
        "tests/fixtures/ref/photo_320x240_444.rgb",
        320,
        240,
    );
}

#[test]
fn xval_photo_320x240_422() {
    assert_matches_reference(
        "tests/fixtures/photo_320x240_422.jpg",
        "tests/fixtures/ref/photo_320x240_422.rgb",
        320,
        240,
    );
}

// --- Content types (640x480) ---

#[test]
fn xval_graphic_640x480_420() {
    assert_matches_reference(
        "tests/fixtures/graphic_640x480_420.jpg",
        "tests/fixtures/ref/graphic_640x480_420.rgb",
        640,
        480,
    );
}

#[test]
fn xval_checker_640x480_420() {
    assert_matches_reference(
        "tests/fixtures/checker_640x480_420.jpg",
        "tests/fixtures/ref/checker_640x480_420.rgb",
        640,
        480,
    );
}

// --- HD / FHD / 2K / 4K ---

#[test]
fn xval_photo_1280x720_420() {
    assert_matches_reference(
        "tests/fixtures/photo_1280x720_420.jpg",
        "tests/fixtures/ref/photo_1280x720_420.rgb",
        1280,
        720,
    );
}

#[test]
fn xval_photo_2560x1440_420() {
    assert_matches_reference(
        "tests/fixtures/photo_2560x1440_420.jpg",
        "tests/fixtures/ref/photo_2560x1440_420.rgb",
        2560,
        1440,
    );
}

#[test]
fn xval_photo_3840x2160_420() {
    assert_matches_reference(
        "tests/fixtures/photo_3840x2160_420.jpg",
        "tests/fixtures/ref/photo_3840x2160_420.rgb",
        3840,
        2160,
    );
}

// --- Subsampling at 640x480 ---

#[test]
fn xval_photo_640x480_444() {
    assert_matches_reference(
        "tests/fixtures/photo_640x480_444.jpg",
        "tests/fixtures/ref/photo_640x480_444.rgb",
        640,
        480,
    );
}

#[test]
fn xval_photo_640x480_422() {
    assert_matches_reference(
        "tests/fixtures/photo_640x480_422.jpg",
        "tests/fixtures/ref/photo_640x480_422.rgb",
        640,
        480,
    );
}

// --- Subsampling at 1080p ---

#[test]
fn xval_photo_1920x1080_444() {
    assert_matches_reference(
        "tests/fixtures/photo_1920x1080_444.jpg",
        "tests/fixtures/ref/photo_1920x1080_444.rgb",
        1920,
        1080,
    );
}

#[test]
fn xval_photo_1920x1080_422() {
    assert_matches_reference(
        "tests/fixtures/photo_1920x1080_422.jpg",
        "tests/fixtures/ref/photo_1920x1080_422.rgb",
        1920,
        1080,
    );
}

// --- Restart markers ---

#[test]
fn xval_photo_640x480_420_rst() {
    assert_matches_reference(
        "tests/fixtures/photo_640x480_420_rst.jpg",
        "tests/fixtures/ref/photo_640x480_420_rst.rgb",
        320,
        240,
    );
}
