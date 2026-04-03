//! Cross-validation of streaming I/O API against C libjpeg-turbo.
//!
//! Tests `stream::compress_to_writer`, `stream::decompress_from_reader`,
//! `stream::compress_to_file`, and `stream::decompress_from_file` by comparing
//! decoded output against C djpeg (pixel-identical, diff=0).

use std::io::Cursor;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{decompress, stream, PixelFormat, Subsampling};

// ===========================================================================
// Tool discovery
// ===========================================================================

/// Locate the djpeg binary. Checks /opt/homebrew/bin/djpeg first, then falls
/// back to whatever `which djpeg` returns. Returns `None` when not found.
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
// Helpers
// ===========================================================================

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path with the `strio_xval_` prefix.
fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("strio_xval_{}_{:04}_{}", pid, counter, name))
}

/// Generate a gradient RGB test image with varied pixel values.
/// Uses smooth gradients and diagonal patterns to exercise quantization
/// across both luma and chroma channels.
fn generate_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Parse a binary PPM (P6) file and return `(width, height, rgb_pixels)`.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

    // Skip whitespace and comments between tokens
    let skip_ws_comments = |p: &mut usize| loop {
        while *p < data.len() && data[*p].is_ascii_whitespace() {
            *p += 1;
        }
        if *p < data.len() && data[*p] == b'#' {
            while *p < data.len() && data[*p] != b'\n' {
                *p += 1;
            }
        } else {
            break;
        }
    };

    // Parse an ASCII decimal number
    let read_number = |p: &mut usize| -> Option<usize> {
        let start: usize = *p;
        while *p < data.len() && data[*p].is_ascii_digit() {
            *p += 1;
        }
        std::str::from_utf8(&data[start..*p]).ok()?.parse().ok()
    };

    skip_ws_comments(&mut pos);
    let width: usize = read_number(&mut pos)?;
    skip_ws_comments(&mut pos);
    let height: usize = read_number(&mut pos)?;
    skip_ws_comments(&mut pos);
    let _maxval: usize = read_number(&mut pos)?;

    // Exactly one whitespace byte after maxval before binary data
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Decode a JPEG with C djpeg and return the decoded RGB pixels.
/// Writes the JPEG to a temp file, runs djpeg, parses the PPM output.
fn decode_with_c_djpeg(djpeg: &PathBuf, jpeg_data: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let jpeg_path: PathBuf = temp_path(&format!("{}.jpg", label));
    let ppm_path: PathBuf = temp_path(&format!("{}.ppm", label));

    // Write JPEG to temp file
    {
        use std::io::Write;
        let mut file: std::fs::File = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

    // Run C djpeg
    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg failed for {}: {}",
        label,
        String::from_utf8_lossy(&output.stderr)
    );

    // Parse PPM output
    let ppm_data: Vec<u8> = std::fs::read(&ppm_path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_path, e));
    let result = parse_ppm(&ppm_data)
        .unwrap_or_else(|| panic!("Failed to parse PPM output from djpeg for {}", label));

    // Cleanup
    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);

    result
}

/// Assert two pixel buffers are identical (diff=0). Prints first 5 mismatches
/// on failure.
fn assert_pixels_identical(
    rust_pixels: &[u8],
    c_pixels: &[u8],
    width: usize,
    height: usize,
    label: &str,
) {
    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "{}: data length mismatch: rust={} c={}",
        label,
        rust_pixels.len(),
        c_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&ours, &theirs)) in rust_pixels.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let px: usize = pixel % width;
                let py: usize = pixel / width;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  {}: pixel ({},{}) channel {}: rust={} c={} diff={}",
                    label, px, py, channel, ours, theirs, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert_eq!(
        mismatches,
        0,
        "{}: {} of {} pixels differ (max diff={}), expected diff=0 for {}x{} image",
        label,
        mismatches,
        width * height,
        max_diff,
        width,
        height
    );
}

// ===========================================================================
// Test 1: compress_to_writer with Vec<u8> writer
// ===========================================================================

/// Compress a 48x48 gradient via `stream::compress_to_writer` into a Vec<u8>,
/// then decode with both Rust and C djpeg. Verify pixel-identical output (diff=0).
#[test]
fn c_xval_compress_to_writer_vec() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    // Compress via stream API into a Vec<u8> writer
    let mut jpeg_buf: Vec<u8> = Vec::new();
    stream::compress_to_writer(
        &mut jpeg_buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        85,
        Subsampling::S444,
    )
    .expect("stream::compress_to_writer failed");

    assert!(!jpeg_buf.is_empty(), "JPEG output must not be empty");

    // Decode with C djpeg
    let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) =
        decode_with_c_djpeg(&djpeg, &jpeg_buf, "writer_vec_48x48");

    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Decode with Rust
    let rust_image = decompress(&jpeg_buf).expect("Rust decompress failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Compare C output vs Rust output -> diff=0
    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "compress_to_writer_vec_48x48",
    );
}

// ===========================================================================
// Test 2: compress_to_writer with multiple sizes
// ===========================================================================

/// Test compress_to_writer with 3 sizes (16x16, 64x48, 128x96). For each,
/// compress, djpeg decode, verify dimensions match and pixels match Rust
/// decode (diff=0).
#[test]
fn c_xval_compress_to_writer_sizes() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let sizes: [(usize, usize); 3] = [(16, 16), (64, 48), (128, 96)];

    for &(width, height) in &sizes {
        let pixels: Vec<u8> = generate_gradient(width, height);

        // Compress via stream API
        let mut jpeg_buf: Vec<u8> = Vec::new();
        stream::compress_to_writer(
            &mut jpeg_buf,
            &pixels,
            width,
            height,
            PixelFormat::Rgb,
            85,
            Subsampling::S444,
        )
        .unwrap_or_else(|e| {
            panic!(
                "stream::compress_to_writer failed for {}x{}: {:?}",
                width, height, e
            )
        });

        assert!(
            !jpeg_buf.is_empty(),
            "JPEG output must not be empty for {}x{}",
            width,
            height
        );

        let label: String = format!("sizes_{}x{}", width, height);

        // Decode with C djpeg
        let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) =
            decode_with_c_djpeg(&djpeg, &jpeg_buf, &label);

        assert_eq!(
            c_width, width,
            "C djpeg width mismatch for {}x{}",
            width, height
        );
        assert_eq!(
            c_height, height,
            "C djpeg height mismatch for {}x{}",
            width, height
        );

        // Decode with Rust
        let rust_image = decompress(&jpeg_buf)
            .unwrap_or_else(|e| panic!("Rust decompress failed for {}x{}: {:?}", width, height, e));
        assert_eq!(rust_image.width, width);
        assert_eq!(rust_image.height, height);

        // Compare -> diff=0
        assert_pixels_identical(&rust_image.data, &c_pixels, width, height, &label);
    }
}

// ===========================================================================
// Test 3: decompress_from_reader with Cursor
// ===========================================================================

/// Compress a 48x48 gradient to JPEG bytes, create a Cursor, call
/// `stream::decompress_from_reader`. Also decode same bytes with C djpeg.
/// Compare pixel data (diff=0).
#[test]
fn c_xval_decompress_from_reader() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    // Compress to JPEG bytes using the stream writer API
    let mut jpeg_buf: Vec<u8> = Vec::new();
    stream::compress_to_writer(
        &mut jpeg_buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        85,
        Subsampling::S444,
    )
    .expect("stream::compress_to_writer failed");

    // Decompress via stream reader API using a Cursor
    let mut cursor: Cursor<&[u8]> = Cursor::new(&jpeg_buf);
    let stream_image =
        stream::decompress_from_reader(&mut cursor).expect("stream::decompress_from_reader failed");

    assert_eq!(stream_image.width, width);
    assert_eq!(stream_image.height, height);

    // Decode same bytes with C djpeg
    let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) =
        decode_with_c_djpeg(&djpeg, &jpeg_buf, "reader_cursor_48x48");

    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Compare stream reader output vs C djpeg -> diff=0
    assert_pixels_identical(
        &stream_image.data,
        &c_pixels,
        width,
        height,
        "decompress_from_reader_48x48",
    );
}

// ===========================================================================
// Test 4: compress_to_file
// ===========================================================================

/// Use `stream::compress_to_file` to write JPEG to a temp path. Run C djpeg
/// on that file. Also read file back and decompress with Rust. Compare (diff=0).
#[test]
fn c_xval_compress_to_file() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);
    let jpeg_path: PathBuf = temp_path("compress_to_file.jpg");

    // Compress to file via stream API
    stream::compress_to_file(
        &jpeg_path,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        85,
        Subsampling::S444,
    )
    .expect("stream::compress_to_file failed");

    // Run C djpeg on the written file
    let ppm_path: PathBuf = temp_path("compress_to_file.ppm");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        output.status.success(),
        "djpeg failed for compress_to_file: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(&ppm_path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_path, e));
    let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) = parse_ppm(&ppm_data)
        .unwrap_or_else(|| panic!("Failed to parse PPM output from djpeg for compress_to_file"));

    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Read file back and decompress with Rust
    let jpeg_data: Vec<u8> = std::fs::read(&jpeg_path)
        .unwrap_or_else(|e| panic!("Failed to read JPEG {:?}: {:?}", jpeg_path, e));
    let rust_image = decompress(&jpeg_data).expect("Rust decompress failed");

    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Compare -> diff=0
    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "compress_to_file_48x48",
    );

    // Cleanup
    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);
}

// ===========================================================================
// Test 5: decompress_from_file
// ===========================================================================

/// Compress JPEG bytes, write to temp file. Call `stream::decompress_from_file`
/// on that path. Also decode original bytes with C djpeg. Compare (diff=0).
#[test]
fn c_xval_decompress_from_file() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    // Compress to JPEG bytes
    let mut jpeg_buf: Vec<u8> = Vec::new();
    stream::compress_to_writer(
        &mut jpeg_buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        85,
        Subsampling::S444,
    )
    .expect("stream::compress_to_writer failed");

    // Write JPEG to a temp file
    let jpeg_path: PathBuf = temp_path("decompress_from_file.jpg");
    std::fs::write(&jpeg_path, &jpeg_buf)
        .unwrap_or_else(|e| panic!("Failed to write JPEG {:?}: {:?}", jpeg_path, e));

    // Decompress via stream file API
    let stream_image =
        stream::decompress_from_file(&jpeg_path).expect("stream::decompress_from_file failed");

    assert_eq!(stream_image.width, width);
    assert_eq!(stream_image.height, height);

    // Decode original bytes with C djpeg
    let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) =
        decode_with_c_djpeg(&djpeg, &jpeg_buf, "decompress_from_file_48x48");

    assert_eq!(c_width, width, "C djpeg width mismatch");
    assert_eq!(c_height, height, "C djpeg height mismatch");

    // Compare stream file output vs C djpeg -> diff=0
    assert_pixels_identical(
        &stream_image.data,
        &c_pixels,
        width,
        height,
        "decompress_from_file_48x48",
    );

    // Cleanup
    let _ = std::fs::remove_file(&jpeg_path);
}

// ===========================================================================
// Test 6: stream roundtrip across subsampling modes
// ===========================================================================

/// For S444, S422, S420: compress_to_writer, write temp file, C djpeg decode,
/// compare vs Rust decompress (diff=0).
#[test]
fn c_xval_stream_roundtrip_subsampling() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let pixels: Vec<u8> = generate_gradient(width, height);

    let subsamplings: [(Subsampling, &str); 3] = [
        (Subsampling::S444, "444"),
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
    ];

    for &(subsampling, ss_name) in &subsamplings {
        // Compress via stream API
        let mut jpeg_buf: Vec<u8> = Vec::new();
        stream::compress_to_writer(
            &mut jpeg_buf,
            &pixels,
            width,
            height,
            PixelFormat::Rgb,
            85,
            subsampling,
        )
        .unwrap_or_else(|e| panic!("stream::compress_to_writer failed for {}: {:?}", ss_name, e));

        assert!(
            !jpeg_buf.is_empty(),
            "JPEG output must not be empty for {}",
            ss_name
        );

        let label: String = format!("roundtrip_{}", ss_name);

        // Decode with C djpeg
        let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) =
            decode_with_c_djpeg(&djpeg, &jpeg_buf, &label);

        assert_eq!(c_width, width, "C djpeg width mismatch for {}", ss_name);
        assert_eq!(c_height, height, "C djpeg height mismatch for {}", ss_name);

        // Decode with Rust
        let rust_image = decompress(&jpeg_buf)
            .unwrap_or_else(|e| panic!("Rust decompress failed for {}: {:?}", ss_name, e));

        assert_eq!(rust_image.width, width);
        assert_eq!(rust_image.height, height);

        // Compare -> diff=0
        assert_pixels_identical(&rust_image.data, &c_pixels, width, height, &label);
    }
}

// ===========================================================================
// Test 7: large image (640x480)
// ===========================================================================

/// Test with a 640x480 image. compress_to_writer, djpeg decode -> diff=0.
/// Verifies streaming works correctly for larger buffers.
#[test]
fn c_xval_stream_large_image() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 640;
    let height: usize = 480;
    let pixels: Vec<u8> = generate_gradient(width, height);

    // Compress via stream API
    let mut jpeg_buf: Vec<u8> = Vec::new();
    stream::compress_to_writer(
        &mut jpeg_buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        85,
        Subsampling::S420,
    )
    .expect("stream::compress_to_writer failed for 640x480");

    assert!(
        !jpeg_buf.is_empty(),
        "JPEG output must not be empty for 640x480"
    );

    // Decode with C djpeg
    let (c_width, c_height, c_pixels): (usize, usize, Vec<u8>) =
        decode_with_c_djpeg(&djpeg, &jpeg_buf, "large_640x480");

    assert_eq!(c_width, width, "C djpeg width mismatch for 640x480");
    assert_eq!(c_height, height, "C djpeg height mismatch for 640x480");

    // Decode with Rust
    let rust_image = decompress(&jpeg_buf).expect("Rust decompress failed for 640x480");

    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Compare -> diff=0
    assert_pixels_identical(
        &rust_image.data,
        &c_pixels,
        width,
        height,
        "stream_large_640x480",
    );
}
