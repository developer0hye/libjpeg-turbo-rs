use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{decompress, decompress_cropped, CropRegion};

#[test]
fn crop_full_image_matches_decompress() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let region = CropRegion {
        x: 0,
        y: 0,
        width: 320,
        height: 240,
    };
    let cropped = decompress_cropped(data, region).unwrap();
    assert_eq!(full.data, cropped.data);
}

#[test]
fn crop_center_region() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion {
        x: 80,
        y: 60,
        width: 160,
        height: 120,
    };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 160);
    assert_eq!(img.height, 120);
    assert_eq!(img.data.len(), 160 * 120 * 3);
}

#[test]
fn crop_clamps_to_image_bounds() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion {
        x: 300,
        y: 200,
        width: 100,
        height: 100,
    };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 20);
    assert_eq!(img.height, 40);
}

#[test]
fn crop_top_left_corner_matches_full_decode() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let region = CropRegion {
        x: 0,
        y: 0,
        width: 64,
        height: 64,
    };
    let cropped = decompress_cropped(data, region).unwrap();
    let bpp = full.pixel_format.bytes_per_pixel();
    // First row of cropped should match first 64 pixels of full
    for x in 0..64 {
        for c in 0..bpp {
            assert_eq!(cropped.data[x * bpp + c], full.data[x * bpp + c]);
        }
    }
}

#[test]
fn crop_zero_size_returns_empty() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let region = CropRegion {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };
    let img = decompress_cropped(data, region).unwrap();
    assert_eq!(img.width, 0);
    assert_eq!(img.height, 0);
    assert_eq!(img.data.len(), 0);
}

#[test]
fn crop_region_pixel_values_match_full_decode() {
    let data = include_bytes!("fixtures/photo_320x240_420.jpg");
    let full = decompress(data).unwrap();
    let bpp = full.pixel_format.bytes_per_pixel();
    let region = CropRegion {
        x: 50,
        y: 30,
        width: 100,
        height: 80,
    };
    let cropped = decompress_cropped(data, region).unwrap();

    // Every pixel in cropped should match the corresponding pixel in full
    for row in 0..80 {
        for col in 0..100 {
            let crop_idx = (row * 100 + col) * bpp;
            let full_idx = ((30 + row) * 320 + (50 + col)) * bpp;
            for c in 0..bpp {
                assert_eq!(
                    cropped.data[crop_idx + c],
                    full.data[full_idx + c],
                    "Mismatch at row={row}, col={col}, channel={c}"
                );
            }
        }
    }
}

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

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
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = skip_ws_comments(&raw, idx);
    let (width, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (height, next) = read_number(&raw, idx);
    idx = skip_ws_comments(&raw, next);
    let (_maxval, next) = read_number(&raw, idx);
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

fn skip_ws_comments(data: &[u8], mut idx: usize) -> usize {
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

fn read_number(data: &[u8], idx: usize) -> (usize, usize) {
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

static CROP_SKIP_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn crop_skip_temp_path(name: &str) -> PathBuf {
    let counter: u64 = CROP_SKIP_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_cropskip_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: crop_skip_temp_path(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Maximum absolute per-channel difference between two pixel buffers.
fn pixel_max_diff(a: &[u8], b: &[u8]) -> u8 {
    assert_eq!(a.len(), b.len(), "pixel buffers must have equal length");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i16 - y as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0)
}

// ===========================================================================
// C djpeg crop cross-validation test
// ===========================================================================

/// Cross-validate Rust `decompress_cropped` against C `djpeg -crop` for multiple
/// crop regions: center crop, corner crops, and full-image crop.
///
/// Uses a Rust-encoded MCU-aligned S444 JPEG so that both Rust and C decoders
/// produce identical pixel output (eliminating decoder differences from the
/// comparison). djpeg -crop outputs full-width scanlines for the cropped row
/// range, so we extract the matching sub-rectangle before comparison.
#[test]
fn c_djpeg_crop_skip_diff_zero() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation crop_skip test");
            return;
        }
    };

    // Encode a 64x64 S444 gradient test JPEG with Rust. S444 with MCU-aligned
    // dimensions ensures pixel-identical decode between Rust and C decoders.
    let (img_w, img_h): (usize, usize) = (64, 64);
    let mut pixels: Vec<u8> = Vec::with_capacity(img_w * img_h * 3);
    for y in 0..img_h {
        for x in 0..img_w {
            pixels.push(((x * 4 + y) % 256) as u8);
            pixels.push(((y * 4 + x) % 256) as u8);
            pixels.push(((x * y) % 256) as u8);
        }
    }
    let jpeg_data: Vec<u8> = libjpeg_turbo_rs::compress(
        &pixels,
        img_w,
        img_h,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        95,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("compress failed");
    let full_width: usize = img_w;

    // Crop regions to test:
    // 1. Center crop
    // 2. Top-left corner crop
    // 3. Bottom-right corner crop
    // 4. Full-image crop
    let crop_cases: &[(usize, usize, usize, usize, &str)] = &[
        (16, 16, 32, 32, "center"),
        (0, 0, 24, 24, "top_left_corner"),
        (40, 40, 24, 24, "bottom_right_corner"),
        (0, 0, 64, 64, "full_image"),
    ];

    for &(crop_x, crop_y, crop_w, crop_h, label) in crop_cases {
        eprintln!(
            "  testing crop {}x{}+{}+{} ({})",
            crop_w, crop_h, crop_x, crop_y, label
        );

        // Step 1: Decode with crop using Rust
        let region = CropRegion {
            x: crop_x,
            y: crop_y,
            width: crop_w,
            height: crop_h,
        };
        let rust_cropped =
            decompress_cropped(&jpeg_data, region).expect("Rust decompress_cropped failed");

        assert_eq!(
            rust_cropped.width, crop_w,
            "{}: Rust crop width mismatch",
            label
        );
        assert_eq!(
            rust_cropped.height, crop_h,
            "{}: Rust crop height mismatch",
            label
        );

        // Step 2: Decode with crop using C djpeg
        let tmp_jpg = TempFile::new(&format!("{}_in.jpg", label));
        let tmp_ppm = TempFile::new(&format!("{}_out.ppm", label));
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

        // Step 3: Extract the crop sub-rectangle from C output.
        // djpeg -crop WxH+X+Y outputs an image of width=image_width, height=crop_h,
        // preserving the full scanline width. We extract columns [crop_x..crop_x+crop_w].
        assert_eq!(c_h, crop_h, "{}: C djpeg crop height mismatch", label);

        let c_crop_pixels: Vec<u8> = if c_w == crop_w {
            // djpeg returned exactly the crop region dimensions
            c_pixels
        } else {
            // djpeg returned full-width scanlines; extract the crop columns
            assert_eq!(
                c_w, full_width,
                "{}: unexpected C output width {}",
                label, c_w
            );
            let mut extracted: Vec<u8> = Vec::with_capacity(crop_w * crop_h * 3);
            for row in 0..crop_h {
                let row_start: usize = row * c_w * 3 + crop_x * 3;
                let row_end: usize = row_start + crop_w * 3;
                extracted.extend_from_slice(&c_pixels[row_start..row_end]);
            }
            extracted
        };

        // Step 4: Assert pixel-identical output (diff = 0)
        assert_eq!(
            rust_cropped.data.len(),
            c_crop_pixels.len(),
            "{}: pixel buffer length mismatch: Rust={} C={}",
            label,
            rust_cropped.data.len(),
            c_crop_pixels.len()
        );

        let max_diff: u8 = pixel_max_diff(&rust_cropped.data, &c_crop_pixels);

        // Log first few mismatches for debugging if any
        if max_diff > 0 {
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
                            "    pixel {} channel {}: rust={} c={} diff={}",
                            pixel, channel, r, c, diff
                        );
                    }
                }
            }
            eprintln!("    total mismatches: {}", mismatches);
        }

        assert_eq!(
            max_diff, 0,
            "{}: crop {}x{}+{}+{}: max_diff={} (must be 0 vs C djpeg)",
            label, crop_w, crop_h, crop_x, crop_y, max_diff
        );
    }
}
