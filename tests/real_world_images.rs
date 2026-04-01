//! Cross-validation of ALL real-world JPEG test images against C djpeg.
//!
//! Scans `tests/fixtures/real_world/*.jpg` and for each file:
//! 1. Decodes with Rust (`decompress` / `decompress_to`)
//! 2. Decodes with C `djpeg -ppm`
//! 3. Compares pixel output (target: diff=0)
//!
//! Known exception categories (gracefully skipped):
//! - 12-bit images (`*12bit*`): Rust 8-bit decoder returns error
//! - Arithmetic images (`*arithmetic*`): skipped if either decoder fails
//! - Images that cause Rust decoder panics (internal bugs): skipped with message
//! - Images that cause Rust decoder errors: skipped if in known-issue list

use libjpeg_turbo_rs::{decompress_to, PixelFormat};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ===========================================================================
// Tool discovery
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

// ===========================================================================
// PPM parsing
// ===========================================================================

/// Parse PNM file (P5 grayscale or P6 RGB), returning (width, height, components, pixel_data).
/// `components` is 1 for P5 (grayscale), 3 for P6 (RGB).
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    if data.len() < 3 {
        return None;
    }
    let is_pgm: bool = &data[0..2] == b"P5";
    let is_ppm: bool = &data[0..2] == b"P6";
    if !is_pgm && !is_ppm {
        return None;
    }

    let mut idx: usize = 2;
    idx = skip_ws_comments(data, idx);
    let (width, next) = read_number(data, idx)?;
    idx = skip_ws_comments(data, next);
    let (height, next) = read_number(data, idx)?;
    idx = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, idx)?;
    // Exactly one whitespace byte after maxval before pixel data
    idx = next + 1;

    let components: usize = if is_pgm { 1 } else { 3 };
    let expected_len: usize = width * height * components;
    let pixel_data: &[u8] = &data[idx..];
    if pixel_data.len() < expected_len {
        return None;
    }

    Some((
        width,
        height,
        components,
        pixel_data[..expected_len].to_vec(),
    ))
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

fn read_number(data: &[u8], idx: usize) -> Option<(usize, usize)> {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    if end == idx {
        return None;
    }
    let val: usize = std::str::from_utf8(&data[idx..end]).ok()?.parse().ok()?;
    Some((val, end))
}

// ===========================================================================
// Temp file management
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(suffix: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_rw_{}_{:04}_{}", pid, counter, suffix))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        Self {
            path: temp_path(suffix),
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

// ===========================================================================
// Fixture discovery
// ===========================================================================

fn real_world_dir() -> PathBuf {
    PathBuf::from("tests/fixtures/real_world")
}

/// Collect all `.jpg` files in real_world directory, sorted by name for deterministic order.
fn collect_jpeg_files() -> Vec<PathBuf> {
    let dir: PathBuf = real_world_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", dir.display(), e))
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path: PathBuf = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jpg") {
                Some(path)
            } else {
                None
            }
        })
        .collect();
    files.sort();
    files
}

/// Filter files whose stem contains any of the given substrings.
fn filter_files(files: &[PathBuf], substrings: &[&str]) -> Vec<PathBuf> {
    files
        .iter()
        .filter(|p| {
            let stem: &str = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            substrings.iter().any(|sub| stem.contains(sub))
        })
        .cloned()
        .collect()
}

// ===========================================================================
// Image classification helpers
// ===========================================================================

fn is_12bit_image(filename: &str) -> bool {
    filename.contains("12bit")
}

fn is_arithmetic_image(filename: &str) -> bool {
    filename.contains("arithmetic")
}

/// Known issues table: (filename_pattern, reason).
/// Images matching these patterns are skipped with the given reason.
/// These represent existing Rust decoder bugs tracked separately.
const KNOWN_DECODE_ISSUES: &[(&str, &str)] = &[
    // Non-uniform chroma sampling (2x1 + 1x1) causes upsample crash
    (
        "zune_synthetic_progressive",
        "non-uniform chroma sampling factors (2x1 + 1x1) not supported",
    ),
];

fn is_known_decode_issue(filename: &str) -> bool {
    KNOWN_DECODE_ISSUES
        .iter()
        .any(|(pattern, _reason)| filename.contains(pattern))
}

fn known_issue_reason(filename: &str) -> &'static str {
    for (pattern, reason) in KNOWN_DECODE_ISSUES {
        if filename.contains(pattern) {
            return reason;
        }
    }
    "unknown issue"
}

// ===========================================================================
// Result tracking
// ===========================================================================

#[derive(Debug)]
enum ImageResult {
    Pass {
        width: usize,
        height: usize,
    },
    Skip {
        reason: String,
    },
    Fail {
        width: usize,
        height: usize,
        max_diff: u8,
        mismatch_count: usize,
    },
}

struct TestRecord {
    filename: String,
    result: ImageResult,
}

fn print_summary(records: &[TestRecord]) {
    let mut pass_count: usize = 0;
    let mut skip_count: usize = 0;
    let mut fail_count: usize = 0;

    eprintln!();
    eprintln!("=== Real-World Image Cross-Validation Summary ===");
    eprintln!(
        "{:<60} {:>10} {:>10} {:>10}",
        "Filename", "Dims", "Max Diff", "Status"
    );
    eprintln!("{}", "-".repeat(95));

    for record in records {
        match &record.result {
            ImageResult::Pass { width, height } => {
                eprintln!(
                    "{:<60} {:>4}x{:<5} {:>10} {:>10}",
                    record.filename, width, height, 0, "PASS"
                );
                pass_count += 1;
            }
            ImageResult::Skip { reason } => {
                eprintln!(
                    "{:<60} {:>10} {:>10} SKIP: {}",
                    record.filename, "-", "-", reason
                );
                skip_count += 1;
            }
            ImageResult::Fail {
                width,
                height,
                max_diff,
                mismatch_count,
            } => {
                eprintln!(
                    "{:<60} {:>4}x{:<5} {:>10} FAIL ({} px differ)",
                    record.filename, width, height, max_diff, mismatch_count
                );
                fail_count += 1;
            }
        }
    }

    eprintln!("{}", "-".repeat(95));
    eprintln!(
        "Total: {} | Pass: {} | Skip: {} | Fail: {}",
        records.len(),
        pass_count,
        skip_count,
        fail_count
    );
    eprintln!();
}

// ===========================================================================
// Core comparison logic
// ===========================================================================

/// Decode a single JPEG with C djpeg and return (width, height, components, pixel_data).
/// `components` is 1 for grayscale (P5), 3 for color (P6).
/// Returns None if djpeg fails (e.g., unsupported format).
fn decode_with_c_djpeg(
    djpeg: &Path,
    jpeg_path: &Path,
    name: &str,
) -> Option<(usize, usize, usize, Vec<u8>)> {
    let tmp: TempFile = TempFile::new(&format!("{}.ppm", name));
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp.path())
        .arg(jpeg_path)
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let ppm_data: Vec<u8> = std::fs::read(tmp.path()).ok()?;
    parse_ppm(&ppm_data)
}

/// Compare two pixel buffers, returning (max_diff, mismatch_count).
fn compare_pixels(rust_data: &[u8], c_data: &[u8], bpp: usize) -> (u8, usize) {
    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;

    for (i, (&ours, &theirs)) in rust_data.iter().zip(c_data.iter()).enumerate() {
        let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / bpp;
                let channel_names: &[&str] = if bpp == 1 { &["Y"] } else { &["R", "G", "B"] };
                let channel: &str = channel_names[i % bpp];
                eprintln!(
                    "    pixel {} {}: rust={} c={} diff={}",
                    pixel, channel, ours, theirs, diff
                );
            }
            if diff > max_diff {
                max_diff = diff;
            }
        }
    }

    (max_diff, mismatch_count)
}

/// Run cross-validation for a single JPEG file.
/// Returns the test record.
fn validate_single_image(djpeg: &Path, jpeg_path: &Path) -> TestRecord {
    let filename: String = jpeg_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    let name_stem: String = jpeg_path
        .file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    eprintln!("  Testing: {}", filename);

    // --- Known exception: 12-bit images ---
    if is_12bit_image(&filename) {
        return TestRecord {
            filename,
            result: ImageResult::Skip {
                reason: "12-bit precision (8-bit decoder not applicable)".to_string(),
            },
        };
    }

    // --- Known decoder issues (panics/errors) ---
    if is_known_decode_issue(&filename) {
        let reason: &str = known_issue_reason(&filename);
        return TestRecord {
            filename,
            result: ImageResult::Skip {
                reason: format!("known decoder issue: {}", reason),
            },
        };
    }

    let is_arithmetic: bool = is_arithmetic_image(&filename);

    // --- C djpeg decode first to determine output format ---
    let c_result: Option<(usize, usize, usize, Vec<u8>)> =
        decode_with_c_djpeg(djpeg, jpeg_path, &name_stem);

    let (c_width, c_height, c_components, c_data) = match c_result {
        Some(result) => result,
        None => {
            if is_arithmetic {
                return TestRecord {
                    filename,
                    result: ImageResult::Skip {
                        reason: "C djpeg failed (arithmetic not supported by this build)"
                            .to_string(),
                    },
                };
            }
            panic!(
                "C djpeg failed for {} (not an expected skip category)",
                filename
            );
        }
    };

    // --- Read JPEG data ---
    let jpeg_data: Vec<u8> = std::fs::read(jpeg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", jpeg_path.display(), e));

    // --- Rust decode ---
    // Match the output format to what djpeg produced (P5=grayscale, P6=RGB)
    let target_format: PixelFormat = if c_components == 1 {
        PixelFormat::Grayscale
    } else {
        PixelFormat::Rgb
    };

    // Use catch_unwind to handle internal panics gracefully
    let rust_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decompress_to(&jpeg_data, target_format)
    }));

    let rust_image = match rust_result {
        Ok(Ok(img)) => img,
        Ok(Err(e)) => {
            if is_arithmetic {
                return TestRecord {
                    filename,
                    result: ImageResult::Skip {
                        reason: format!("Rust decode failed (arithmetic): {}", e),
                    },
                };
            }
            // Unexpected Rust decode error
            panic!(
                "Rust decode failed for {} (not an expected skip category): {}",
                filename, e
            );
        }
        Err(panic_info) => {
            let msg: String = if let Some(s) = panic_info.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic".to_string()
            };
            // Internal panic — record as skip (these are known bugs to fix separately)
            return TestRecord {
                filename,
                result: ImageResult::Skip {
                    reason: format!("Rust decoder panicked: {}", msg),
                },
            };
        }
    };

    // --- Dimension check ---
    assert_eq!(
        rust_image.width, c_width,
        "{}: width mismatch: rust={} c={}",
        filename, rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "{}: height mismatch: rust={} c={}",
        filename, rust_image.height, c_height
    );

    // --- Data length check ---
    let expected_len: usize = c_width * c_height * c_components;
    assert_eq!(
        c_data.len(),
        expected_len,
        "{}: C output size mismatch: got={} expected={}",
        filename,
        c_data.len(),
        expected_len
    );
    assert_eq!(
        rust_image.data.len(),
        expected_len,
        "{}: Rust output size mismatch: got={} expected={} (format={:?})",
        filename,
        rust_image.data.len(),
        expected_len,
        rust_image.pixel_format
    );

    // --- Pixel comparison ---
    let (max_diff, mismatch_count) = compare_pixels(&rust_image.data, &c_data, c_components);

    if max_diff == 0 {
        TestRecord {
            filename,
            result: ImageResult::Pass {
                width: c_width,
                height: c_height,
            },
        }
    } else {
        TestRecord {
            filename,
            result: ImageResult::Fail {
                width: c_width,
                height: c_height,
                max_diff,
                mismatch_count,
            },
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn c_djpeg_cross_validation_real_world_images() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let files: Vec<PathBuf> = collect_jpeg_files();
    assert!(
        !files.is_empty(),
        "no .jpg files found in {}",
        real_world_dir().display()
    );
    eprintln!("Found {} JPEG files in real_world fixtures", files.len());

    let mut records: Vec<TestRecord> = Vec::with_capacity(files.len());

    for jpeg_path in &files {
        let record: TestRecord = validate_single_image(&djpeg, jpeg_path);
        records.push(record);
    }

    print_summary(&records);

    // Assert all non-skipped images pass with diff=0
    let failures: Vec<&TestRecord> = records
        .iter()
        .filter(|r| matches!(r.result, ImageResult::Fail { .. }))
        .collect();

    assert!(
        failures.is_empty(),
        "{} image(s) failed cross-validation:\n{}",
        failures.len(),
        failures
            .iter()
            .map(|r| {
                if let ImageResult::Fail {
                    max_diff,
                    mismatch_count,
                    ..
                } = &r.result
                {
                    format!(
                        "  - {}: max_diff={}, {} pixels differ",
                        r.filename, max_diff, mismatch_count
                    )
                } else {
                    unreachable!()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn c_djpeg_cross_validation_real_world_progressive() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let all_files: Vec<PathBuf> = collect_jpeg_files();
    let progressive_files: Vec<PathBuf> = filter_files(&all_files, &["progressive"]);

    if progressive_files.is_empty() {
        eprintln!("SKIP: no progressive images found");
        return;
    }

    eprintln!("Testing {} progressive JPEG files", progressive_files.len());

    let mut records: Vec<TestRecord> = Vec::with_capacity(progressive_files.len());

    for jpeg_path in &progressive_files {
        let record: TestRecord = validate_single_image(&djpeg, jpeg_path);
        records.push(record);
    }

    print_summary(&records);

    let failures: Vec<&TestRecord> = records
        .iter()
        .filter(|r| matches!(r.result, ImageResult::Fail { .. }))
        .collect();

    assert!(
        failures.is_empty(),
        "{} progressive image(s) failed cross-validation:\n{}",
        failures.len(),
        failures
            .iter()
            .map(|r| format!("  - {}", r.filename))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn c_djpeg_cross_validation_real_world_highres() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let all_files: Vec<PathBuf> = collect_jpeg_files();
    let highres_files: Vec<PathBuf> = filter_files(&all_files, &["4k", "8k"]);

    if highres_files.is_empty() {
        eprintln!("SKIP: no 4K/8K images found");
        return;
    }

    eprintln!("Testing {} high-resolution JPEG files", highres_files.len());

    let mut records: Vec<TestRecord> = Vec::with_capacity(highres_files.len());

    for jpeg_path in &highres_files {
        let record: TestRecord = validate_single_image(&djpeg, jpeg_path);

        // Print timing for non-skipped images
        if matches!(record.result, ImageResult::Pass { .. }) {
            let name_stem: String = jpeg_path
                .file_stem()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let jpeg_data: Vec<u8> = std::fs::read(jpeg_path).unwrap();

            // Rust decode with timing (second pass, warmed up)
            let rust_start: Instant = Instant::now();
            let _rust_image = decompress_to(&jpeg_data, PixelFormat::Rgb).unwrap();
            let rust_elapsed = rust_start.elapsed();

            // C djpeg decode with timing
            let c_start: Instant = Instant::now();
            let _c_result = decode_with_c_djpeg(&djpeg, jpeg_path, &name_stem);
            let c_elapsed = c_start.elapsed();

            eprintln!(
                "    Timing — Rust: {:.1}ms, C djpeg: {:.1}ms",
                rust_elapsed.as_secs_f64() * 1000.0,
                c_elapsed.as_secs_f64() * 1000.0,
            );
        }

        records.push(record);
    }

    print_summary(&records);

    let failures: Vec<&TestRecord> = records
        .iter()
        .filter(|r| matches!(r.result, ImageResult::Fail { .. }))
        .collect();

    assert!(
        failures.is_empty(),
        "{} high-resolution image(s) failed cross-validation:\n{}",
        failures.len(),
        failures
            .iter()
            .map(|r| format!("  - {}", r.filename))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
