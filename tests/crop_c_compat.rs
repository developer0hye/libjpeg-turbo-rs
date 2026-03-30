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

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

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

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

/// Path to C djpeg binary, or `None` if not installed.
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

/// Parse a binary PPM (P6) file and return `(width, height, data)`.
/// `data` contains raw RGB bytes.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = ppm_skip_whitespace_and_comments(&raw, idx);
    let (width, next) = ppm_read_ascii_number(&raw, idx);
    idx = ppm_skip_whitespace_and_comments(&raw, next);
    let (height, next) = ppm_read_ascii_number(&raw, idx);
    idx = ppm_skip_whitespace_and_comments(&raw, next);
    let (_maxval, next) = ppm_read_ascii_number(&raw, idx);
    // Exactly one whitespace byte after maxval before binary data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        data.len()
    );
    (width, height, data)
}

fn ppm_skip_whitespace_and_comments(data: &[u8], mut idx: usize) -> usize {
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

fn ppm_read_ascii_number(data: &[u8], idx: usize) -> (usize, usize) {
    let start: usize = idx;
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[start..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

/// Global atomic counter for unique temp file names across parallel tests.
static CROP_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path for crop tests.
fn crop_temp_path(name: &str) -> PathBuf {
    let counter: u64 = CROP_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_crop_{}_{:04}_{}", pid, counter, name))
}

/// RAII guard that removes a file when dropped (for temp file cleanup).
struct CropTempFile {
    path: PathBuf,
}

impl CropTempFile {
    fn new(name: &str) -> Self {
        Self {
            path: crop_temp_path(name),
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for CropTempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

// ===========================================================================
// C djpeg cross-validation test
// ===========================================================================

/// Encode a 64x64 JPEG with Rust, then decode a 32x32+16+16 crop region
/// using both Rust (`decompress_cropped`) and C (`djpeg -crop`), and assert
/// that the pixel output is identical (diff = 0).
#[test]
fn c_djpeg_crop_decode_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation crop test");
            return;
        }
    };

    // Crop parameters
    let crop_w: usize = 32;
    let crop_h: usize = 32;
    let crop_x: usize = 16;
    let crop_y: usize = 16;

    // Step 1: Encode a 64x64 test JPEG with Rust
    let jpeg_data: Vec<u8> = encode_64x64(Subsampling::S444);

    // Step 2: Decode with crop using Rust
    let region = CropRegion {
        x: crop_x,
        y: crop_y,
        width: crop_w,
        height: crop_h,
    };
    let rust_cropped =
        decompress_cropped(&jpeg_data, region).expect("Rust decompress_cropped failed");

    assert_eq!(rust_cropped.width, crop_w, "Rust crop width mismatch");
    assert_eq!(rust_cropped.height, crop_h, "Rust crop height mismatch");

    // Step 3: Decode with crop using C djpeg (-crop WxH+X+Y)
    let tmp_jpg = CropTempFile::new("crop_xval.jpg");
    let tmp_ppm = CropTempFile::new("crop_xval.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");

    let crop_arg: String = format!("{}x{}+{}+{}", crop_w, crop_h, crop_x, crop_y);
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-crop")
        .arg(&crop_arg)
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        eprintln!(
            "SKIP: djpeg -crop failed (may not support -crop): {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return;
    }

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());

    // djpeg -crop may output the full image width with the cropped height,
    // so we compare only the crop region from the C output.
    // djpeg -crop WxH+X+Y outputs an image of width=image_width, height=crop_h,
    // where only the rows [crop_y..crop_y+crop_h] are output but the full
    // scanline width is preserved. We extract the matching sub-rectangle.
    assert_eq!(c_h, crop_h, "C djpeg crop height mismatch");

    // If C output width equals the full image width, extract the sub-region
    let c_crop_pixels: Vec<u8> = if c_w == crop_w {
        // djpeg returned exactly the crop region
        c_pixels
    } else {
        // djpeg returned full-width scanlines; extract the crop columns
        let mut extracted: Vec<u8> = Vec::with_capacity(crop_w * crop_h * 3);
        for row in 0..crop_h {
            let row_start: usize = row * c_w * 3 + crop_x * 3;
            let row_end: usize = row_start + crop_w * 3;
            extracted.extend_from_slice(&c_pixels[row_start..row_end]);
        }
        extracted
    };

    // Step 4: Assert diff = 0 between Rust and C crop outputs
    assert_eq!(
        rust_cropped.data.len(),
        c_crop_pixels.len(),
        "pixel buffer length mismatch: Rust={} C={}",
        rust_cropped.data.len(),
        c_crop_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&r, &c)) in rust_cropped
        .data
        .iter()
        .zip(c_crop_pixels.iter())
        .enumerate()
    {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert_eq!(
        mismatches, 0,
        "crop {}x{}+{}+{}: {} pixels differ between Rust and C djpeg (max diff: {})",
        crop_w, crop_h, crop_x, crop_y, mismatches, max_diff
    );
}
