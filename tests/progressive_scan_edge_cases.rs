use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress_progressive, decompress, Encoder, PixelFormat, ProgressiveDecoder, ScanScript,
    Subsampling,
};

/// Helper: create synthetic pixel data with some spatial variation.
fn make_pixels(width: usize, height: usize, bpp: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * bpp);
    for y in 0..height {
        for x in 0..width {
            for c in 0..bpp {
                pixels.push(((x * 7 + y * 3 + c * 50) % 256) as u8);
            }
        }
    }
    pixels
}

/// Helper: create grayscale pixel data.
fn make_gray_pixels(width: usize, height: usize) -> Vec<u8> {
    (0..width * height).map(|i| (i % 256) as u8).collect()
}

/// Sum of absolute pixel differences between two buffers.
fn pixel_diff(a: &[u8], b: &[u8]) -> u64 {
    let len: usize = a.len().min(b.len());
    let mut total: u64 = 0;
    for i in 0..len {
        total += (a[i] as i64 - b[i] as i64).unsigned_abs();
    }
    total
}

// ============================================================
// Scan order variations
// ============================================================

#[test]
fn progressive_dc_only_no_ac_refinement_decodes() {
    // A progressive scan script with only DC scans (no AC data at all).
    // The result should be a very blocky but valid image.
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let script: Vec<ScanScript> = vec![
        // Single interleaved DC scan, no successive approximation
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        },
    ];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();

    // Should contain SOF2 marker
    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "DC-only progressive should still have SOF2");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert!(!img.data.is_empty());
}

#[test]
fn progressive_single_all_component_scan_decodes() {
    // Degenerate progressive: a single scan that covers DC+AC for all components.
    // This is technically a valid progressive JPEG with just one scan.
    let pixels: Vec<u8> = make_pixels(16, 16, 3);

    // For progressive, each scan can only contain either DC (ss=0,se=0) or AC (ss>0).
    // Also, interleaved scans can only do DC. So we need separate scans:
    // DC interleaved + AC per-component = "degenerate" progressive but still multi-scan.
    let script: Vec<ScanScript> = vec![
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
    ];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn progressive_custom_unusual_spectral_ordering_decodes() {
    // Custom script: send low-frequency AC first, then high-frequency AC
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let script: Vec<ScanScript> = vec![
        // DC for all components
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        },
        // Y: low AC (1-5)
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 5,
            ah: 0,
            al: 0,
        },
        // Y: high AC (6-63)
        ScanScript {
            components: vec![0],
            ss: 6,
            se: 63,
            ah: 0,
            al: 0,
        },
        // Cb: full AC
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        // Cr: full AC
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
    ];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert!(!img.data.is_empty());
}

// ============================================================
// Incomplete progressive decoding
// ============================================================

#[test]
fn progressive_decoder_first_scan_only_produces_valid_image() {
    let jpeg_data: Vec<u8> = compress_progressive(
        &make_pixels(32, 32, 3),
        32,
        32,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .unwrap();

    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    assert!(decoder.num_scans() > 1, "expected multiple scans");

    // Consume only the first scan
    let consumed: bool = decoder.consume_input().unwrap();
    assert!(consumed, "should consume first scan");
    assert_eq!(decoder.scans_consumed(), 1);
    assert!(
        !decoder.input_complete(),
        "should not be complete after 1 scan"
    );

    // Output after first scan should be valid (low quality) image
    let early_image = decoder.output().unwrap();
    assert_eq!(early_image.width, 32);
    assert_eq!(early_image.height, 32);
    assert!(
        !early_image.data.is_empty(),
        "first-scan output should have pixel data"
    );
}

#[test]
fn progressive_decoder_half_scans_intermediate_quality() {
    let jpeg_data: Vec<u8> = compress_progressive(
        &make_pixels(32, 32, 3),
        32,
        32,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .unwrap();

    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    let total_scans: usize = decoder.num_scans();
    let half_scans: usize = total_scans / 2;

    // Consume first scan for early reference
    decoder.consume_input().unwrap();
    let early_image = decoder.output().unwrap();

    // Consume up to half scans
    for _ in 1..half_scans {
        decoder.consume_input().unwrap();
    }
    let half_image = decoder.output().unwrap();

    // Consume remaining scans
    while decoder.consume_input().unwrap() {}
    let final_image = decoder.output().unwrap();

    // Compare against full decompress reference
    let reference = decompress(&jpeg_data).unwrap();

    let early_diff: u64 = pixel_diff(&early_image.data, &reference.data);
    let half_diff: u64 = pixel_diff(&half_image.data, &reference.data);
    let final_diff: u64 = pixel_diff(&final_image.data, &reference.data);

    // Quality should improve monotonically: early >= half >= final
    assert!(
        early_diff >= half_diff || half_diff == 0,
        "early diff ({}) should be >= half diff ({})",
        early_diff,
        half_diff
    );
    assert!(
        half_diff >= final_diff || final_diff == 0,
        "half diff ({}) should be >= final diff ({})",
        half_diff,
        final_diff
    );
}

// ============================================================
// Progressive with various subsampling
// ============================================================

#[test]
fn progressive_420_roundtrip() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.pixel_format, PixelFormat::Rgb);

    // Also verify via ProgressiveDecoder
    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

#[test]
fn progressive_444_roundtrip() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

#[test]
fn progressive_grayscale_roundtrip() {
    let pixels: Vec<u8> = make_gray_pixels(32, 32);
    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

#[test]
fn progressive_422_roundtrip() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S422).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

// ============================================================
// Progressive + metadata
// ============================================================

#[test]
fn progressive_with_icc_profile() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let icc: Vec<u8> = vec![0x42; 500];

    // Use Encoder to create progressive JPEG with ICC
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .icc_profile(&icc)
        .encode()
        .unwrap();

    // Verify it is progressive (SOF2)
    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "should be progressive JPEG");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(
        img.icc_profile(),
        Some(icc.as_slice()),
        "ICC profile should survive progressive encoding"
    );
}

#[test]
fn progressive_with_exif() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    // Valid EXIF with orientation = 3 (rotated 180)
    let exif: Vec<u8> = build_tiff_with_orientation(3);

    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .exif_data(&exif)
        .encode()
        .unwrap();

    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "should be progressive JPEG");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.exif_data(),
        Some(exif.as_slice()),
        "EXIF data should survive progressive encoding"
    );
    assert_eq!(img.exif_orientation(), Some(3));
}

#[test]
fn progressive_with_comment() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let comment: &str = "Progressive test comment";

    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .comment(comment)
        .encode()
        .unwrap();

    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "should be progressive JPEG");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.comment.as_deref(),
        Some(comment),
        "comment should survive progressive encoding"
    );
}

#[test]
fn progressive_with_all_metadata_combined() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let icc: Vec<u8> = vec![0xCC; 2000];
    let exif: Vec<u8> = build_tiff_with_orientation(8);
    let comment: &str = "All metadata progressive test";

    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .progressive(true)
        .icc_profile(&icc)
        .exif_data(&exif)
        .comment(comment)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(icc.as_slice()));
    assert_eq!(img.exif_data(), Some(exif.as_slice()));
    assert_eq!(img.exif_orientation(), Some(8));
    assert_eq!(img.comment.as_deref(), Some(comment));
}

#[test]
fn progressive_decoder_preserves_metadata() {
    // Verify that ProgressiveDecoder.output() also includes ICC and EXIF
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let icc: Vec<u8> = vec![0xDD; 300];
    let exif: Vec<u8> = build_tiff_with_orientation(6);

    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .icc_profile(&icc)
        .exif_data(&exif)
        .comment("progressive decoder metadata test")
        .encode()
        .unwrap();

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let img = decoder.finish().unwrap();
    assert_eq!(
        img.icc_profile,
        Some(icc.clone()),
        "ProgressiveDecoder should preserve ICC"
    );
    assert_eq!(
        img.exif_data,
        Some(exif.clone()),
        "ProgressiveDecoder should preserve EXIF"
    );
    assert_eq!(
        img.comment.as_deref(),
        Some("progressive decoder metadata test"),
        "ProgressiveDecoder should preserve comment"
    );
}

// ============================================================
// Helper: build minimal TIFF with orientation (duplicated for this test file)
// ============================================================

fn build_tiff_with_orientation(orientation: u16) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&0x0112u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&orientation.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data
}

// ============================================================
// C djpeg cross-validation helpers
// ============================================================

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

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_progscan_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: temp_path(name),
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
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch"
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

/// Helper to cross-validate a progressive JPEG against C djpeg.
/// Decodes with both Rust and C djpeg and asserts pixel-identical output.
fn cross_validate_progressive_jpeg(djpeg: &Path, jpeg_data: &[u8], label: &str) {
    let tmp_jpeg: TempFile = TempFile::new(&format!("{}.jpg", label));
    let tmp_ppm: TempFile = TempFile::new(&format!("{}.ppm", label));
    std::fs::write(tmp_jpeg.path(), jpeg_data).expect("write JPEG");

    // Decode with C djpeg
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpeg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "{}: djpeg failed: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    let (c_w, c_h, c_pixels) = parse_ppm(tmp_ppm.path());

    // Decode with Rust
    let rust_img = decompress(jpeg_data)
        .unwrap_or_else(|e| panic!("{}: Rust decompress failed: {}", label, e));

    assert_eq!(
        rust_img.width, c_w,
        "{}: width mismatch Rust={} C={}",
        label, rust_img.width, c_w
    );
    assert_eq!(
        rust_img.height, c_h,
        "{}: height mismatch Rust={} C={}",
        label, rust_img.height, c_h
    );
    assert_eq!(
        rust_img.data.len(),
        c_pixels.len(),
        "{}: pixel data length mismatch",
        label
    );

    let max_diff: u8 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&r, &c)| (r as i16 - c as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "{}: Rust vs C djpeg max_diff={} (must be 0)",
        label, max_diff
    );
}

// ============================================================
// C djpeg cross-validation test
// ============================================================

#[test]
fn c_djpeg_cross_validation_progressive_edge_cases() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let pixels: Vec<u8> = make_pixels(32, 32, 3);

    // Case 1: Default progressive encoding (420)
    {
        let jpeg: Vec<u8> =
            compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420)
                .expect("compress_progressive default must succeed");
        cross_validate_progressive_jpeg(&djpeg, &jpeg, "default_progressive_420");
    }

    // Case 2: Default progressive encoding (444)
    {
        let jpeg: Vec<u8> =
            compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444)
                .expect("compress_progressive 444 must succeed");
        cross_validate_progressive_jpeg(&djpeg, &jpeg, "default_progressive_444");
    }

    // Case 3: Custom scan script — standard multi-scan with separate DC and AC
    {
        let script: Vec<ScanScript> = vec![
            // Interleaved DC for all components
            ScanScript {
                components: vec![0, 1, 2],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            },
            // Y AC
            ScanScript {
                components: vec![0],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            },
            // Cb AC
            ScanScript {
                components: vec![1],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            },
            // Cr AC
            ScanScript {
                components: vec![2],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            },
        ];
        let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
            .quality(75)
            .subsampling(Subsampling::S444)
            .progressive(true)
            .scan_script(script)
            .encode()
            .expect("custom multi-scan progressive must succeed");
        cross_validate_progressive_jpeg(&djpeg, &jpeg, "custom_multi_scan");
    }

    // Case 4: Custom scan script — unusual spectral ordering
    {
        let script: Vec<ScanScript> = vec![
            // DC for all components
            ScanScript {
                components: vec![0, 1, 2],
                ss: 0,
                se: 0,
                ah: 0,
                al: 0,
            },
            // Y: low AC (1-5)
            ScanScript {
                components: vec![0],
                ss: 1,
                se: 5,
                ah: 0,
                al: 0,
            },
            // Y: high AC (6-63)
            ScanScript {
                components: vec![0],
                ss: 6,
                se: 63,
                ah: 0,
                al: 0,
            },
            // Cb: full AC
            ScanScript {
                components: vec![1],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            },
            // Cr: full AC
            ScanScript {
                components: vec![2],
                ss: 1,
                se: 63,
                ah: 0,
                al: 0,
            },
        ];
        let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
            .quality(75)
            .subsampling(Subsampling::S444)
            .progressive(true)
            .scan_script(script)
            .encode()
            .expect("custom spectral progressive must succeed");
        cross_validate_progressive_jpeg(&djpeg, &jpeg, "custom_spectral_ordering");
    }

    // Case 5: Progressive with 422 subsampling
    {
        let jpeg: Vec<u8> =
            compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S422)
                .expect("compress_progressive 422 must succeed");
        cross_validate_progressive_jpeg(&djpeg, &jpeg, "default_progressive_422");
    }
}
