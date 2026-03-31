use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    Encoder, MarkerCopyMode, PixelFormat, SavedMarker, Subsampling, TransformOptions,
};

/// Helper: create a small JPEG with ICC (APP2), EXIF (APP1), and COM markers embedded.
fn make_jpeg_with_all_markers() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let mut encoder = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb);
    encoder = encoder.quality(75);

    // Add a fake ICC profile in APP2 with standard ICC_PROFILE header
    let mut icc_data: Vec<u8> = Vec::new();
    icc_data.extend_from_slice(b"ICC_PROFILE\0");
    icc_data.push(1); // chunk sequence number
    icc_data.push(1); // total chunks
    icc_data.extend_from_slice(&[0xAA; 32]); // fake ICC payload
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE2, // APP2
        data: icc_data,
    });

    // Add a fake EXIF marker in APP1
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE1, // APP1 (EXIF)
        data: b"Exif\0\0FakeExifData".to_vec(),
    });

    // Add a COM marker
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xFE, // COM
        data: b"Test comment".to_vec(),
    });

    // Add another APP marker (APP5)
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE5,
        data: b"APP5-payload".to_vec(),
    });

    encoder.encode().unwrap()
}

/// Helper: read saved markers from a JPEG using decode with marker saving.
fn read_all_markers(jpeg: &[u8]) -> Vec<SavedMarker> {
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();
    image.saved_markers.clone()
}

// --- MarkerCopyMode::All preserves all markers ---

#[test]
fn copy_mode_all_preserves_all_markers() {
    let data: Vec<u8> = make_jpeg_with_all_markers();
    let opts: TransformOptions = TransformOptions {
        copy_markers: MarkerCopyMode::All,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let markers: Vec<SavedMarker> = read_all_markers(&result);

    // Should preserve APP1 (EXIF)
    assert!(
        markers.iter().any(|m| m.code == 0xE1),
        "All mode should preserve APP1 (EXIF)"
    );

    // Should preserve APP2 (ICC)
    assert!(
        markers.iter().any(|m| m.code == 0xE2),
        "All mode should preserve APP2 (ICC)"
    );

    // Should preserve COM
    assert!(
        markers.iter().any(|m| m.code == 0xFE),
        "All mode should preserve COM"
    );

    // Should preserve APP5
    assert!(
        markers.iter().any(|m| m.code == 0xE5),
        "All mode should preserve APP5"
    );
}

// --- MarkerCopyMode::None strips all markers ---

#[test]
fn copy_mode_none_strips_all_markers() {
    let data: Vec<u8> = make_jpeg_with_all_markers();
    let opts: TransformOptions = TransformOptions {
        copy_markers: MarkerCopyMode::None,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let markers: Vec<SavedMarker> = read_all_markers(&result);

    // Should NOT have APP1 (EXIF)
    assert!(
        !markers.iter().any(|m| m.code == 0xE1),
        "None mode should strip APP1 (EXIF)"
    );

    // Should NOT have APP2 (ICC)
    assert!(
        !markers.iter().any(|m| m.code == 0xE2),
        "None mode should strip APP2 (ICC)"
    );

    // Should NOT have COM
    assert!(
        !markers.iter().any(|m| m.code == 0xFE),
        "None mode should strip COM"
    );

    // Should NOT have APP5
    assert!(
        !markers.iter().any(|m| m.code == 0xE5),
        "None mode should strip APP5"
    );
}

// --- MarkerCopyMode::IccOnly preserves ICC but strips COM/EXIF ---

#[test]
fn copy_mode_icc_only_preserves_icc_strips_others() {
    let data: Vec<u8> = make_jpeg_with_all_markers();
    let opts: TransformOptions = TransformOptions {
        copy_markers: MarkerCopyMode::IccOnly,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&data, &opts).unwrap();
    let markers: Vec<SavedMarker> = read_all_markers(&result);

    // Should preserve APP2 (ICC)
    assert!(
        markers.iter().any(|m| m.code == 0xE2),
        "IccOnly mode should preserve APP2 (ICC)"
    );

    // Should NOT have APP1 (EXIF)
    assert!(
        !markers.iter().any(|m| m.code == 0xE1),
        "IccOnly mode should strip APP1 (EXIF)"
    );

    // Should NOT have COM
    assert!(
        !markers.iter().any(|m| m.code == 0xFE),
        "IccOnly mode should strip COM"
    );

    // Should NOT have APP5
    assert!(
        !markers.iter().any(|m| m.code == 0xE5),
        "IccOnly mode should strip APP5"
    );
}

// --- Default copy_markers is All ---

#[test]
fn default_copy_markers_is_all() {
    let opts: TransformOptions = TransformOptions::default();
    assert_eq!(opts.copy_markers, MarkerCopyMode::All);
}

// --- From<bool> backward compatibility ---

#[test]
fn marker_copy_mode_from_bool_true_is_all() {
    let mode: MarkerCopyMode = true.into();
    assert_eq!(mode, MarkerCopyMode::All);
}

#[test]
fn marker_copy_mode_from_bool_false_is_none() {
    let mode: MarkerCopyMode = false.into();
    assert_eq!(mode, MarkerCopyMode::None);
}

// ===========================================================================
// C jpegtran cross-validation helpers
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

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_copymode_{}_{:04}_{}", pid, counter, name))
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

/// Create a test JPEG with ICC profile and EXIF data for cross-validation.
fn make_jpeg_with_icc_and_exif() -> Vec<u8> {
    let (w, h): (usize, usize) = (48, 48);
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w) as u8);
            pixels.push(((y * 255) / h) as u8);
            pixels.push((((x + y) * 127) / (w + h)) as u8);
        }
    }

    // Build a fake ICC profile
    let icc_payload: Vec<u8> = vec![0xAA; 64];
    let mut icc_data: Vec<u8> = Vec::new();
    icc_data.extend_from_slice(b"ICC_PROFILE\0");
    icc_data.push(1); // chunk sequence number
    icc_data.push(1); // total chunks
    icc_data.extend_from_slice(&icc_payload);

    // Build minimal EXIF
    let mut exif_data: Vec<u8> = Vec::new();
    exif_data.extend_from_slice(b"Exif\0\0");
    exif_data.extend_from_slice(b"II"); // little endian
    exif_data.extend_from_slice(&42u16.to_le_bytes());
    exif_data.extend_from_slice(&8u32.to_le_bytes());
    exif_data.extend_from_slice(&1u16.to_le_bytes()); // 1 IFD entry
    exif_data.extend_from_slice(&0x0112u16.to_le_bytes()); // orientation tag
    exif_data.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    exif_data.extend_from_slice(&1u32.to_le_bytes()); // count
    exif_data.extend_from_slice(&6u16.to_le_bytes()); // orientation=6
    exif_data.extend_from_slice(&0u16.to_le_bytes());
    exif_data.extend_from_slice(&0u32.to_le_bytes());

    let mut encoder = Encoder::new(&pixels, w, h, PixelFormat::Rgb);
    encoder = encoder
        .quality(90)
        .subsampling(Subsampling::S444)
        .saved_marker(SavedMarker {
            code: 0xE2, // APP2 (ICC)
            data: icc_data,
        })
        .saved_marker(SavedMarker {
            code: 0xE1, // APP1 (EXIF)
            data: exif_data,
        });
    encoder.encode().expect("encode test JPEG with markers")
}

// ===========================================================================
// C jpegtran cross-validation test
// ===========================================================================

#[test]
fn c_jpegtran_cross_validation_copy_mode() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let source_jpeg: Vec<u8> = make_jpeg_with_icc_and_exif();

    // Write source JPEG to temp file
    let tmp_src: TempFile = TempFile::new("copymode_src.jpg");
    std::fs::write(tmp_src.path(), &source_jpeg).expect("write source JPEG");

    // Test each copy mode: (C jpegtran flag, Rust MarkerCopyMode, label)
    let modes: Vec<(&str, MarkerCopyMode, &str)> = vec![
        ("-copy all", MarkerCopyMode::All, "copy_all"),
        ("-copy none", MarkerCopyMode::None, "copy_none"),
        ("-copy icc", MarkerCopyMode::IccOnly, "copy_icc"),
    ];

    for (c_flag, rust_mode, label) in &modes {
        // --- C jpegtran ---
        let tmp_c_out: TempFile = TempFile::new(&format!("{}_c.jpg", label));
        let c_flag_parts: Vec<&str> = c_flag.split_whitespace().collect();
        let output = Command::new(&jpegtran)
            .args(&c_flag_parts)
            .arg("-outfile")
            .arg(tmp_c_out.path())
            .arg(tmp_src.path())
            .output()
            .expect("failed to run jpegtran");

        // If jpegtran does not support `-copy icc`, skip that variant
        if !output.status.success() {
            let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
            if stderr.contains("unrecognized") || stderr.contains("invalid") {
                eprintln!("SKIP: jpegtran does not support '{}': {}", c_flag, stderr);
                continue;
            }
            panic!("jpegtran {} failed: {}", label, stderr);
        }

        // --- Rust transform ---
        let rust_opts: TransformOptions = TransformOptions {
            copy_markers: *rust_mode,
            ..TransformOptions::default()
        };
        let rust_result: Vec<u8> =
            libjpeg_turbo_rs::transform_jpeg_with_options(&source_jpeg, &rust_opts)
                .unwrap_or_else(|e| panic!("Rust transform {} failed: {}", label, e));

        // --- Decode both with C djpeg and compare pixels ---
        let tmp_c_ppm: TempFile = TempFile::new(&format!("{}_c.ppm", label));
        let tmp_r_jpeg: TempFile = TempFile::new(&format!("{}_r.jpg", label));
        let tmp_r_ppm: TempFile = TempFile::new(&format!("{}_r.ppm", label));

        std::fs::write(tmp_r_jpeg.path(), &rust_result).expect("write Rust result");

        // Decode C jpegtran output with djpeg
        let c_dec = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_c_ppm.path())
            .arg(tmp_c_out.path())
            .output()
            .expect("failed to run djpeg on C output");
        assert!(
            c_dec.status.success(),
            "{}: djpeg on C output failed: {}",
            label,
            String::from_utf8_lossy(&c_dec.stderr)
        );

        // Decode Rust output with djpeg
        let r_dec = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_r_ppm.path())
            .arg(tmp_r_jpeg.path())
            .output()
            .expect("failed to run djpeg on Rust output");
        assert!(
            r_dec.status.success(),
            "{}: djpeg on Rust output failed: {}",
            label,
            String::from_utf8_lossy(&r_dec.stderr)
        );

        let (c_w, c_h, c_pixels) = parse_ppm(tmp_c_ppm.path());
        let (r_w, r_h, r_pixels) = parse_ppm(tmp_r_ppm.path());

        assert_eq!(c_w, r_w, "{}: width mismatch C={} Rust={}", label, c_w, r_w);
        assert_eq!(
            c_h, r_h,
            "{}: height mismatch C={} Rust={}",
            label, c_h, r_h
        );

        // Lossless transform: pixel data must be identical (diff=0)
        let max_diff: u8 = c_pixels
            .iter()
            .zip(r_pixels.iter())
            .map(|(&c, &r)| (c as i16 - r as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff, 0,
            "{}: C jpegtran vs Rust transform pixel max_diff={} (must be 0)",
            label, max_diff
        );
    }
}
