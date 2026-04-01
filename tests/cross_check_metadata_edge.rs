//! Cross-check tests for metadata edge cases against C libjpeg-turbo tools.
//!
//! Scenarios NOT covered by `cross_check_metadata.rs`:
//! 1. Large ICC multi-chunk (100KB) — encode, extract with C djpeg, compare
//! 2. All 8 EXIF orientations — encode with each, verify C tool reads same orientation
//! 3. UTF-8 comment — encode, extract with rdjpgcom, compare
//! 4. Multiple COM markers — encode with 2+ COM markers via saved_marker API, verify C reads all
//! 5. ICC survival through jpegtran rotation — encode with ICC, rotate 90 with Rust, compare with C jpegtran

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress_with_metadata, decompress, transform_jpeg_with_options, Encoder, MarkerCopyMode,
    MarkerSaveConfig, PixelFormat, SavedMarker, Subsampling, TransformOp, TransformOptions,
};

// ===========================================================================
// Tool discovery (copied from cross_check_metadata.rs)
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

fn jpegtran_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/jpegtran");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("jpegtran")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn rdjpgcom_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/rdjpgcom");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("rdjpgcom")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn exiftool_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/exiftool");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("exiftool")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Check if djpeg supports the `-icc` flag for extracting ICC profiles.
fn djpeg_supports_icc_extract(djpeg: &Path) -> bool {
    let output = Command::new(djpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("icc")
        }
        Err(_) => false,
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_meta_edge_{}_{:04}_{}", pid, counter, name))
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

/// Generate a small test RGB image with a gradient pattern.
fn generate_test_pixels(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w.max(1)) as u8);
            pixels.push(((y * 255) / h.max(1)) as u8);
            pixels.push(128u8);
        }
    }
    pixels
}

/// Build minimal TIFF/EXIF data with a given orientation value.
fn build_tiff_with_orientation(orientation: u16) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    // Little-endian byte order
    data.extend_from_slice(b"II");
    // TIFF magic (42)
    data.extend_from_slice(&42u16.to_le_bytes());
    // IFD0 offset = 8 (immediately after header)
    data.extend_from_slice(&8u32.to_le_bytes());
    // IFD0: 1 entry
    data.extend_from_slice(&1u16.to_le_bytes());
    // Entry: tag=0x0112 (Orientation), type=3 (SHORT), count=1, value=orientation
    data.extend_from_slice(&0x0112u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    data.extend_from_slice(&1u32.to_le_bytes()); // count
    data.extend_from_slice(&orientation.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes()); // padding
                                                 // Next IFD offset = 0 (end)
    data.extend_from_slice(&0u32.to_le_bytes());
    data
}

// ===========================================================================
// 1. Large ICC multi-chunk (100KB) — C djpeg extracts identical ICC
// ===========================================================================

#[test]
fn large_icc_100kb_extracted_by_c_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_icc_extract(&djpeg) {
        eprintln!("SKIP: djpeg does not support -icc flag");
        return;
    }

    // 100KB ICC profile requires multiple APP2 chunks (each chunk max 65519 bytes)
    let large_icc: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    // Encode with Rust + large ICC
    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        Some(&large_icc),
        None,
    )
    .expect("Rust compress with 100KB ICC");

    let tmp_jpg: TempFile = TempFile::new("large_icc.jpg");
    let tmp_icc: TempFile = TempFile::new("large_icc_extracted.icc");
    let tmp_ppm: TempFile = TempFile::new("large_icc.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp jpg");

    // Extract ICC with C djpeg -icc
    let output = Command::new(&djpeg)
        .arg("-icc")
        .arg(tmp_icc.path())
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg -icc failed on 100KB ICC JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Compare extracted ICC with original — must be byte-identical
    let extracted_icc: Vec<u8> = std::fs::read(tmp_icc.path()).expect("read extracted ICC");
    assert_eq!(
        extracted_icc.len(),
        large_icc.len(),
        "extracted ICC size mismatch: got {}, expected {}",
        extracted_icc.len(),
        large_icc.len()
    );
    assert_eq!(
        extracted_icc, large_icc,
        "extracted 100KB ICC profile must be byte-identical to original"
    );
}

// ===========================================================================
// 2. All 8 EXIF orientations — C exiftool reads same orientation
// ===========================================================================

#[test]
fn all_8_exif_orientations_readable_by_c_tool() {
    // We use exiftool to read orientation since djpeg does not extract EXIF fields.
    // If exiftool is not available, fall back to verifying Rust roundtrip against
    // the raw EXIF bytes that a C tool (djpeg) can decode without errors.
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let exiftool: Option<PathBuf> = exiftool_path();
    if exiftool.is_none() {
        eprintln!("INFO: exiftool not found, will verify djpeg accepts EXIF + Rust roundtrip");
    }

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    for orientation in 1u16..=8 {
        let exif_data: Vec<u8> = build_tiff_with_orientation(orientation);

        let jpeg: Vec<u8> = compress_with_metadata(
            &pixels,
            w,
            h,
            PixelFormat::Rgb,
            90,
            Subsampling::S444,
            None,
            Some(&exif_data),
        )
        .unwrap_or_else(|e| panic!("compress with orientation {} failed: {}", orientation, e));

        let tmp_jpg: TempFile = TempFile::new(&format!("orient_{}.jpg", orientation));
        let tmp_ppm: TempFile = TempFile::new(&format!("orient_{}.ppm", orientation));
        std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp jpg");

        // Verify C djpeg can decode the JPEG without error
        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        assert!(
            output.status.success(),
            "djpeg failed on JPEG with orientation {}: {}",
            orientation,
            String::from_utf8_lossy(&output.stderr)
        );

        // If exiftool is available, verify it reads the correct orientation
        if let Some(ref tool) = exiftool {
            let output = Command::new(tool)
                .arg("-Orientation")
                .arg("-n")
                .arg("-s3")
                .arg(tmp_jpg.path())
                .output()
                .expect("failed to run exiftool");

            if output.status.success() {
                let stdout: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
                let parsed: u16 = stdout.parse().unwrap_or_else(|_| {
                    panic!(
                        "exiftool returned non-numeric orientation for value {}: '{}'",
                        orientation, stdout
                    )
                });
                assert_eq!(
                    parsed, orientation,
                    "exiftool orientation mismatch: expected {}, got {}",
                    orientation, parsed
                );
            }
        }

        // Verify Rust roundtrip preserves the orientation
        let img = decompress(&jpeg).unwrap_or_else(|e| {
            panic!("Rust decompress orientation {} failed: {}", orientation, e)
        });
        assert_eq!(
            img.exif_orientation(),
            Some(orientation as u8),
            "Rust roundtrip orientation {} mismatch",
            orientation
        );
    }
}

// ===========================================================================
// 3. UTF-8 comment — rdjpgcom extracts identical text
// ===========================================================================

#[test]
fn utf8_comment_readable_by_rdjpgcom() {
    let rdjpgcom: PathBuf = match rdjpgcom_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: rdjpgcom not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    // rdjpgcom only supports Latin-1, so multi-byte UTF-8 (CJK, emoji) gets garbled.
    // Use Latin-1-compatible UTF-8 for the rdjpgcom cross-check (accents and degree sign
    // are single-byte in Latin-1 and two-byte in UTF-8, but rdjpgcom outputs raw bytes).
    // We verify full UTF-8 (including CJK) through Rust roundtrip separately.
    let latin1_comment: &str = "Caf\u{00e9} \u{00b0}C temp\u{00e9}rature";
    let full_utf8_comment: &str = "Caf\u{00e9} \u{00b0}C \u{4e16}\u{754c} \u{1F4F7}";

    // --- Part A: Latin-1-safe comment verified by rdjpgcom ---
    let jpeg_latin1: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(90)
        .comment(latin1_comment)
        .encode()
        .expect("Rust encode with Latin-1 comment");

    let tmp_jpg: TempFile = TempFile::new("utf8_comment.jpg");
    std::fs::write(tmp_jpg.path(), &jpeg_latin1).expect("write temp jpg");

    let output = Command::new(&rdjpgcom)
        .arg("-raw")
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run rdjpgcom");

    assert!(
        output.status.success(),
        "rdjpgcom failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // rdjpgcom -raw outputs raw bytes; compare at byte level since it re-interprets as Latin-1
    let raw_stdout: &[u8] = &output.stdout;
    let comment_bytes: &[u8] = latin1_comment.as_bytes();
    assert!(
        raw_stdout
            .windows(comment_bytes.len())
            .any(|w| w == comment_bytes),
        "rdjpgcom -raw output should contain the comment bytes. Got: {:?}",
        String::from_utf8_lossy(raw_stdout)
    );

    // --- Part B: Full UTF-8 (including CJK/emoji) verified through Rust roundtrip ---
    let jpeg_utf8: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(90)
        .comment(full_utf8_comment)
        .encode()
        .expect("Rust encode with full UTF-8 comment");

    let img = decompress(&jpeg_utf8).expect("Rust decompress with full UTF-8 comment");
    assert_eq!(
        img.comment.as_deref(),
        Some(full_utf8_comment),
        "Rust roundtrip should preserve full UTF-8 comment (including CJK and emoji)"
    );
}

// ===========================================================================
// 4. Multiple COM markers — rdjpgcom reads all comments
// ===========================================================================

#[test]
fn multiple_com_markers_readable_by_rdjpgcom() {
    let rdjpgcom: PathBuf = match rdjpgcom_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: rdjpgcom not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    let comment_one: &str = "first-metadata-comment";
    let comment_two: &str = "second-metadata-comment";
    let comment_three: &str = "third-metadata-comment";

    // Encode with multiple COM markers via saved_marker API
    let jpeg: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(90)
        .saved_marker(SavedMarker {
            code: 0xFE,
            data: comment_one.as_bytes().to_vec(),
        })
        .saved_marker(SavedMarker {
            code: 0xFE,
            data: comment_two.as_bytes().to_vec(),
        })
        .saved_marker(SavedMarker {
            code: 0xFE,
            data: comment_three.as_bytes().to_vec(),
        })
        .encode()
        .expect("Rust encode with multiple COM markers");

    let tmp_jpg: TempFile = TempFile::new("multi_com.jpg");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp jpg");

    // rdjpgcom should list all COM markers
    let output = Command::new(&rdjpgcom)
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run rdjpgcom");

    assert!(
        output.status.success(),
        "rdjpgcom failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout: String = String::from_utf8_lossy(&output.stdout).to_string();
    assert!(
        stdout.contains(comment_one),
        "rdjpgcom should find first comment '{}'. Got: {}",
        comment_one,
        stdout
    );
    assert!(
        stdout.contains(comment_two),
        "rdjpgcom should find second comment '{}'. Got: {}",
        comment_two,
        stdout
    );
    assert!(
        stdout.contains(comment_three),
        "rdjpgcom should find third comment '{}'. Got: {}",
        comment_three,
        stdout
    );

    // Verify Rust decode with MarkerSaveConfig::All sees all COM markers
    let mut decoder =
        libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).expect("create decoder");
    decoder.save_markers(MarkerSaveConfig::All);
    let img = decoder.decode_image().expect("decode with saved markers");

    let com_markers: Vec<&SavedMarker> = img
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xFE)
        .collect();
    assert!(
        com_markers.len() >= 3,
        "expected at least 3 COM markers via Rust decode, got {}",
        com_markers.len()
    );

    let data_list: Vec<&[u8]> = com_markers.iter().map(|m| m.data.as_slice()).collect();
    assert!(
        data_list.contains(&comment_one.as_bytes()),
        "Rust decode should find first comment"
    );
    assert!(
        data_list.contains(&comment_two.as_bytes()),
        "Rust decode should find second comment"
    );
    assert!(
        data_list.contains(&comment_three.as_bytes()),
        "Rust decode should find third comment"
    );
}

// ===========================================================================
// 5. ICC survival through jpegtran rotation — Rust transform vs C jpegtran
// ===========================================================================

#[test]
fn icc_survives_rust_rot90_matches_c_jpegtran() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_icc_extract(&djpeg) {
        eprintln!("SKIP: djpeg does not support -icc flag");
        return;
    }
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    // Use a 100KB ICC to exercise multi-chunk preservation through transform
    let icc_data: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();

    // Use MCU-aligned dimensions (multiple of 16) to avoid edge trimming issues
    let (w, h): (usize, usize) = (32, 32);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    // Step 1: Encode with Rust + ICC
    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        Some(&icc_data),
        None,
    )
    .expect("Rust compress with ICC for transform test");

    // Step 2: Rotate 90 with Rust transform (copy all markers)
    let rust_rotated: Vec<u8> = transform_jpeg_with_options(
        &jpeg,
        &TransformOptions {
            op: TransformOp::Rot90,
            copy_markers: MarkerCopyMode::All,
            ..Default::default()
        },
    )
    .expect("Rust transform rot90 with ICC");

    // Step 3: Rotate 90 with C jpegtran -copy all
    let tmp_src: TempFile = TempFile::new("icc_rot90_src.jpg");
    let tmp_c_rotated: TempFile = TempFile::new("icc_rot90_c.jpg");
    std::fs::write(tmp_src.path(), &jpeg).expect("write source jpeg");

    let output = Command::new(&jpegtran)
        .arg("-copy")
        .arg("all")
        .arg("-rotate")
        .arg("90")
        .arg("-outfile")
        .arg(tmp_c_rotated.path())
        .arg(tmp_src.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -copy all -rotate 90 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Step 4: Extract ICC from Rust-rotated JPEG using C djpeg
    let tmp_rust_jpg: TempFile = TempFile::new("icc_rot90_rust.jpg");
    let tmp_rust_icc: TempFile = TempFile::new("icc_rot90_rust_extracted.icc");
    let tmp_rust_ppm: TempFile = TempFile::new("icc_rot90_rust.ppm");
    std::fs::write(tmp_rust_jpg.path(), &rust_rotated).expect("write rust rotated jpeg");

    let output = Command::new(&djpeg)
        .arg("-icc")
        .arg(tmp_rust_icc.path())
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_rust_ppm.path())
        .arg(tmp_rust_jpg.path())
        .output()
        .expect("failed to run djpeg on Rust-rotated JPEG");

    assert!(
        output.status.success(),
        "djpeg -icc failed on Rust-rotated JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let rust_extracted_icc: Vec<u8> =
        std::fs::read(tmp_rust_icc.path()).expect("read Rust-rotated ICC");
    assert_eq!(
        rust_extracted_icc, icc_data,
        "ICC from Rust rot90 output must be byte-identical to original"
    );

    // Step 5: Extract ICC from C-rotated JPEG using C djpeg
    let tmp_c_icc: TempFile = TempFile::new("icc_rot90_c_extracted.icc");
    let tmp_c_ppm: TempFile = TempFile::new("icc_rot90_c.ppm");

    let output = Command::new(&djpeg)
        .arg("-icc")
        .arg(tmp_c_icc.path())
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_c_ppm.path())
        .arg(tmp_c_rotated.path())
        .output()
        .expect("failed to run djpeg on C-rotated JPEG");

    assert!(
        output.status.success(),
        "djpeg -icc failed on C-rotated JPEG: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_extracted_icc: Vec<u8> = std::fs::read(tmp_c_icc.path()).expect("read C-rotated ICC");
    assert_eq!(
        c_extracted_icc, icc_data,
        "ICC from C jpegtran rot90 output must be byte-identical to original"
    );

    // Step 6: Cross-compare — both tools should produce the same ICC
    assert_eq!(
        rust_extracted_icc, c_extracted_icc,
        "ICC extracted from Rust-rotated and C-rotated JPEGs must match"
    );

    // Step 7: Verify Rust decode also sees the ICC in both outputs
    let rust_img = decompress(&rust_rotated).expect("Rust decompress of Rust-rotated JPEG");
    assert_eq!(
        rust_img.icc_profile(),
        Some(icc_data.as_slice()),
        "Rust decode of Rust-rotated JPEG should find ICC"
    );

    let c_rotated_data: Vec<u8> = std::fs::read(tmp_c_rotated.path()).expect("read C-rotated JPEG");
    let c_img = decompress(&c_rotated_data).expect("Rust decompress of C-rotated JPEG");
    assert_eq!(
        c_img.icc_profile(),
        Some(icc_data.as_slice()),
        "Rust decode of C-rotated JPEG should find ICC"
    );
}
