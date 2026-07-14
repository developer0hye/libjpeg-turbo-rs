//! Cross-validation of ALL real-world JPEG test images against C djpeg.
//!
//! Scans `tests/fixtures/real_world/*.jpg` and for each file:
//! 1. Decodes with Rust (`decompress` / `decompress_to`)
//! 2. Decodes with C `djpeg -ppm`
//! 3. Compares pixel output (target: diff=0)
//!
//! The fixture inventory is gated by feature and provenance minimums. Once C
//! `djpeg` accepts a fixture, every Rust error, panic, or pixel difference is a
//! hard test failure.

mod helpers;

use libjpeg_turbo_rs::{decompress_to, PixelFormat};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

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
    let (maxval, next) = read_number(data, idx)?;
    // Exactly one whitespace byte after maxval before pixel data
    idx = next + 1;

    let components: usize = if is_pgm { 1 } else { 3 };
    let num_samples: usize = width * height * components;

    if maxval > 255 {
        // 16-bit (2 bytes per sample, big-endian). Scale to 8-bit.
        let raw_len: usize = num_samples * 2;
        let pixel_data: &[u8] = &data[idx..];
        if pixel_data.len() < raw_len {
            return None;
        }
        let mut out: Vec<u8> = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let hi: u16 = pixel_data[i * 2] as u16;
            let lo: u16 = pixel_data[i * 2 + 1] as u16;
            let val: u16 = (hi << 8) | lo;
            // Scale from 0..maxval to 0..255
            out.push((val as u32 * 255 / maxval as u32) as u8);
        }
        Some((width, height, components, out))
    } else {
        // 8-bit (1 byte per sample)
        let pixel_data: &[u8] = &data[idx..];
        if pixel_data.len() < num_samples {
            return None;
        }
        Some((
            width,
            height,
            components,
            pixel_data[..num_samples].to_vec(),
        ))
    }
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

#[derive(Debug)]
struct JpegStructure {
    sof_marker: u8,
    precision: u8,
    width: usize,
    height: usize,
    components: usize,
    first_scan_components: usize,
    has_exif: bool,
}

fn inspect_jpeg_structure(data: &[u8]) -> Option<JpegStructure> {
    const SOF_MARKERS: &[u8] = &[
        0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf,
    ];
    if data.get(..2)? != [0xff, 0xd8] {
        return None;
    }

    let mut position = 2;
    let mut frame: Option<(u8, u8, usize, usize, usize)> = None;
    while position < data.len() {
        if data[position] != 0xff {
            return None;
        }
        while data.get(position) == Some(&0xff) {
            position += 1;
        }
        let marker = *data.get(position)?;
        position += 1;
        if marker == 0xda {
            let scan_components = *data.get(position + 2)? as usize;
            let (sof_marker, precision, width, height, components) = frame?;
            return Some(JpegStructure {
                sof_marker,
                precision,
                width,
                height,
                components,
                first_scan_components: scan_components,
                has_exif: data.windows(6).any(|window| window == b"Exif\0\0"),
            });
        }
        if marker == 0xd9 || marker == 0x01 || (0xd0..=0xd7).contains(&marker) {
            continue;
        }
        let segment_length =
            u16::from_be_bytes([*data.get(position)?, *data.get(position + 1)?]) as usize;
        if segment_length < 2 || position.checked_add(segment_length)? > data.len() {
            return None;
        }
        if SOF_MARKERS.contains(&marker) {
            let precision = *data.get(position + 2)?;
            let height =
                u16::from_be_bytes([*data.get(position + 3)?, *data.get(position + 4)?]) as usize;
            let width =
                u16::from_be_bytes([*data.get(position + 5)?, *data.get(position + 6)?]) as usize;
            let components = *data.get(position + 7)? as usize;
            frame = Some((marker, precision, width, height, components));
        }
        position += segment_length;
    }
    None
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
        "Total: {} | Pass: {} | Fail: {}",
        records.len(),
        pass_count,
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

    // --- C djpeg decode first to determine output format ---
    let c_result: Option<(usize, usize, usize, Vec<u8>)> =
        decode_with_c_djpeg(djpeg, jpeg_path, &name_stem);

    let (c_width, c_height, c_components, c_data) = match c_result {
        Some(result) => result,
        None => panic!("C djpeg failed for {filename}"),
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

    // Preserve the filename in decoder errors while ensuring panics remain
    // hard failures rather than silently reducing coverage.
    let rust_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        decompress_to(&jpeg_data, target_format)
    }));

    let rust_image = match rust_result {
        Ok(Ok(img)) => img,
        Ok(Err(e)) => panic!("Rust decode failed for {filename}: {e}"),
        Err(panic_info) => std::panic::resume_unwind(panic_info),
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
fn real_world_fixture_inventory_covers_required_feature_families() {
    let files = collect_jpeg_files();
    let names: Vec<&str> = files
        .iter()
        .filter_map(|path| path.file_name()?.to_str())
        .collect();
    let provenance_families = [
        "derived_",
        "exif_",
        "image_rs_",
        "libjpeg_",
        "pil_",
        "w3c_",
        "zune_",
    ]
    .iter()
    .filter(|prefix| names.iter().any(|name| name.starts_with(**prefix)))
    .count();
    let structures: Vec<JpegStructure> = files
        .iter()
        .map(|path| {
            let data = std::fs::read(path).expect("read real-world fixture inventory");
            inspect_jpeg_structure(&data)
                .unwrap_or_else(|| panic!("invalid JPEG structure: {}", path.display()))
        })
        .collect();
    let progressive = structures
        .iter()
        .filter(|jpeg| matches!(jpeg.sof_marker, 0xc2 | 0xc6 | 0xca | 0xce))
        .count();
    let arithmetic = structures
        .iter()
        .filter(|jpeg| matches!(jpeg.sof_marker, 0xc9 | 0xca | 0xcb | 0xcd | 0xce | 0xcf))
        .count();
    let four_component = structures
        .iter()
        .filter(|jpeg| jpeg.components == 4)
        .count();
    let non_interleaved = structures
        .iter()
        .filter(|jpeg| jpeg.components > 1 && jpeg.first_scan_components < jpeg.components)
        .count();
    let high_resolution = structures
        .iter()
        .filter(|jpeg| jpeg.width >= 3840 || jpeg.height >= 2160)
        .count();
    let exif = structures.iter().filter(|jpeg| jpeg.has_exif).count();

    assert!(
        names.len() >= 61,
        "real-world fixture count shrank to {}",
        names.len()
    );
    assert!(
        provenance_families >= 7,
        "only {provenance_families} provenance families remain"
    );
    assert!(
        progressive >= 13,
        "progressive coverage shrank to {progressive}"
    );
    assert!(
        arithmetic >= 3,
        "arithmetic coverage shrank to {arithmetic}"
    );
    assert!(
        structures.iter().any(|jpeg| jpeg.precision == 12),
        "12-bit coverage disappeared"
    );
    assert!(
        four_component >= 4,
        "four-component CMYK/YCCK coverage shrank to {four_component}"
    );
    assert!(
        non_interleaved >= 5,
        "non-interleaved coverage shrank to {non_interleaved}"
    );
    assert!(
        high_resolution >= 4,
        "high-resolution coverage shrank to {high_resolution}"
    );
    assert!(exif >= 7, "EXIF-bearing coverage shrank to {exif}");
}

#[test]
fn c_djpeg_cross_validation_real_world_images() {
    let djpeg: PathBuf = require_c_tool!("djpeg");

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

    // Assert every image passes with diff=0.
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
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let all_files: Vec<PathBuf> = collect_jpeg_files();
    let progressive_files: Vec<PathBuf> = filter_files(&all_files, &["progressive"]);

    assert!(
        progressive_files.len() >= 13,
        "progressive fixture coverage shrank"
    );

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
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let all_files: Vec<PathBuf> = collect_jpeg_files();
    let highres_files: Vec<PathBuf> = filter_files(&all_files, &["4k", "8k"]);

    assert!(
        highres_files.len() >= 4,
        "high-resolution fixture coverage shrank"
    );

    eprintln!("Testing {} high-resolution JPEG files", highres_files.len());

    let mut records: Vec<TestRecord> = Vec::with_capacity(highres_files.len());

    for jpeg_path in &highres_files {
        let record: TestRecord = validate_single_image(&djpeg, jpeg_path);

        // Print timing for passing images.
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
