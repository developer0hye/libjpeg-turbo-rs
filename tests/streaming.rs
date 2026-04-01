use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::decompress;

fn assert_streaming_matches_high_level(data: &[u8]) {
    let decoder = StreamingDecoder::new(data).unwrap();
    let streaming = decoder.decode().unwrap();
    let high_level = decompress(data).unwrap();

    assert_eq!(streaming.width, high_level.width);
    assert_eq!(streaming.height, high_level.height);
    assert_eq!(streaming.pixel_format, high_level.pixel_format);
    assert_eq!(streaming.data, high_level.data);
}

#[test]
fn streaming_decoder_decode_matches_high_level_api() {
    let data = include_bytes!("fixtures/gradient_640x480.jpg");
    assert_streaming_matches_high_level(data);
}

#[test]
fn streaming_decoder_decode_matches_high_level_api_422() {
    let data = include_bytes!("fixtures/green_16x16_422.jpg");
    assert_streaming_matches_high_level(data);
}

#[test]
fn streaming_decoder_decode_matches_high_level_api_420() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");
    assert_streaming_matches_high_level(data);
}

#[test]
fn streaming_decoder_can_decode_multiple_times() {
    let data = include_bytes!("fixtures/blue_16x16_420.jpg");

    let decoder = StreamingDecoder::new(data).unwrap();
    let first = decoder.decode().unwrap();
    let second = decoder.decode().unwrap();

    assert_eq!(first.width, second.width);
    assert_eq!(first.height, second.height);
    assert_eq!(first.pixel_format, second.pixel_format);
    assert_eq!(first.data, second.data);
}

// ---------------------------------------------------------------------------
// C djpeg cross-validation helpers
// ---------------------------------------------------------------------------

/// Locate the djpeg binary. Checks /opt/homebrew/bin/djpeg first, then falls
/// back to whatever `which djpeg` returns. Returns `None` when not found.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew_path: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }

    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path: PathBuf = PathBuf::from(&path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Parse a binary PPM (P6) image into (width, height, rgb_pixels).
/// Returns `None` if the data is not a valid P6 PPM.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

    // Skip whitespace
    while pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse width
    let width_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let width: usize = std::str::from_utf8(&data[width_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Skip whitespace
    while pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse height
    let height_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let height: usize = std::str::from_utf8(&data[height_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Skip whitespace
    while pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse maxval
    let maxval_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let _maxval: usize = std::str::from_utf8(&data[maxval_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Exactly one whitespace character after maxval before binary data
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Run C djpeg on a JPEG buffer and return the decoded RGB pixels as
/// (width, height, pixels). Panics on djpeg execution failures.
fn run_djpeg(djpeg: &PathBuf, jpeg_data: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let temp_dir: PathBuf = std::env::temp_dir();
    let jpeg_path: PathBuf = temp_dir.join(format!("streaming_xval_{}.jpg", label));
    let ppm_path: PathBuf = temp_dir.join(format!("streaming_xval_{}.ppm", label));

    // Write JPEG to temp file
    {
        let mut file = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

    // Run C djpeg
    let djpeg_output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        djpeg_output.status.success(),
        "djpeg failed for {}: {}",
        label,
        String::from_utf8_lossy(&djpeg_output.stderr)
    );

    // Parse PPM output
    let ppm_data: Vec<u8> = std::fs::read(&ppm_path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_path, e));
    let (width, height, pixels) =
        parse_ppm(&ppm_data).expect("Failed to parse PPM output from djpeg");

    // Cleanup temp files
    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);

    (width, height, pixels)
}

/// Assert that Rust StreamingDecoder output is pixel-identical to C djpeg output.
fn assert_streaming_matches_djpeg(jpeg_data: &[u8], djpeg: &PathBuf, label: &str) {
    // Rust streaming decode
    let decoder = StreamingDecoder::new(jpeg_data)
        .unwrap_or_else(|e| panic!("StreamingDecoder::new failed for {}: {:?}", label, e));
    let rust_image = decoder
        .decode()
        .unwrap_or_else(|e| panic!("StreamingDecoder::decode failed for {}: {:?}", label, e));

    // C djpeg decode
    let (c_width, c_height, c_pixels) = run_djpeg(djpeg, jpeg_data, label);

    // Verify dimensions match
    assert_eq!(
        rust_image.width, c_width,
        "{}: width mismatch: Rust={} C={}",
        label, rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "{}: height mismatch: Rust={} C={}",
        label, rust_image.height, c_height
    );
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "{}: data length mismatch: Rust={} C={}",
        label,
        rust_image.data.len(),
        c_pixels.len()
    );

    // Pixel-exact comparison (diff=0)
    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&ours, &theirs)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  {} pixel {} channel {}: rust={} c={} diff={}",
                    label, pixel, channel, ours, theirs, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert_eq!(
        mismatches, 0,
        "{}: {} pixels differ (max diff={}), expected diff=0",
        label, mismatches, max_diff
    );
}

// ---------------------------------------------------------------------------
// C djpeg cross-validation tests for streaming decoder
// ---------------------------------------------------------------------------

/// Verify that StreamingDecoder produces pixel-identical output to C djpeg
/// across multiple fixtures (4:4:4, 4:2:2, 4:2:0).
#[test]
fn c_djpeg_streaming_decode_matches() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    // 4:4:4 fixture
    let gradient_444: &[u8] = include_bytes!("fixtures/gradient_640x480.jpg");
    assert_streaming_matches_djpeg(gradient_444, &djpeg, "gradient_640x480_444");

    // 4:2:2 fixture
    let green_422: &[u8] = include_bytes!("fixtures/green_16x16_422.jpg");
    assert_streaming_matches_djpeg(green_422, &djpeg, "green_16x16_422");

    // 4:2:0 fixture
    let blue_420: &[u8] = include_bytes!("fixtures/blue_16x16_420.jpg");
    assert_streaming_matches_djpeg(blue_420, &djpeg, "blue_16x16_420");
}

/// Verify that StreamingDecoder produces pixel-identical output to C djpeg
/// for a real-world 4:2:0 photo (exercises chroma upsampling thoroughly).
#[test]
fn c_djpeg_streaming_420_matches() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    let photo_420: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");
    assert_streaming_matches_djpeg(photo_420, &djpeg, "photo_320x240_420");
}
