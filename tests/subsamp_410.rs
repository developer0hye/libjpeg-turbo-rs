/// Test 4:1:0 (H=4, V=2) subsampling decode support.
///
/// 4:1:0 is a rare subsampling mode where luma uses 4x2 sampling and chroma
/// uses 1x1 sampling (upsampling factor 4 horizontally, 2 vertically).
/// This test constructs a JPEG with 4:1:0 sampling factors by directly
/// manipulating the SOF header, then verifies successful decode.
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

/// Helper: create a minimal valid JPEG with custom sampling factors in the SOF0 header.
/// We start from a standard 4:2:0 JPEG and patch the SOF0 component definitions.
fn make_jpeg_with_410_sampling() -> Vec<u8> {
    // Start with a real JPEG and manually construct one with 4:1:0 sampling.
    // We use the encoder to create a baseline image, then patch the SOF marker.
    use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

    // Encode a 32x32 image at 4:2:0 (H=2, V=2 for luma).
    let width: usize = 32;
    let height: usize = 32;
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * 3;
            pixels[idx] = (x * 8) as u8; // R gradient
            pixels[idx + 1] = (y * 8) as u8; // G gradient
            pixels[idx + 2] = 128; // B constant
        }
    }

    let mut jpeg: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap();

    // Find SOF0 marker (0xFF 0xC0) and patch sampling factors.
    // SOF0 layout: FF C0 [len:2] [precision:1] [height:2] [width:2] [nf:1]
    //   then per component: [id:1] [h_samp<<4 | v_samp:1] [qt:1]
    let sof_pos: usize = find_marker(&jpeg, 0xC0).expect("SOF0 marker not found");
    // Skip marker (2 bytes) + length (2 bytes) + precision (1) + height (2) + width (2) + nf (1) = 10 bytes
    let comp_start: usize = sof_pos + 2 + 2 + 1 + 2 + 2 + 1;

    // Component 0 (Y): change from 2x2 to 4x2
    // Current: sampling byte = (2<<4)|2 = 0x22
    // Desired: sampling byte = (4<<4)|2 = 0x42
    assert_eq!(
        jpeg[comp_start + 1],
        0x22,
        "expected Y component with 2x2 sampling"
    );
    jpeg[comp_start + 1] = 0x42; // 4x2 for Y

    // Components 1,2 (Cb, Cr) stay at 1x1 (0x11) -- no change needed.

    jpeg
}

/// Find a JPEG marker by code. Returns offset of the marker (at 0xFF byte).
fn find_marker(data: &[u8], code: u8) -> Option<usize> {
    let mut i: usize = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == code {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[test]
fn decode_410_subsampling_produces_correct_dimensions() {
    let jpeg: Vec<u8> = make_jpeg_with_410_sampling();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    let image = decoder.decode_image().unwrap();

    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.pixel_format, libjpeg_turbo_rs::PixelFormat::Rgb);
    assert_eq!(image.data.len(), 32 * 32 * 3);
}

#[test]
fn decode_410_subsampling_produces_plausible_pixels() {
    let jpeg: Vec<u8> = make_jpeg_with_410_sampling();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    let image = decoder.decode_image().unwrap();

    // Verify pixels are not all zeros (i.e., decode actually produced content).
    let nonzero: usize = image.data.iter().filter(|&&b| b != 0).count();
    assert!(
        nonzero > image.data.len() / 4,
        "expected non-trivial pixel data, got {} nonzero out of {}",
        nonzero,
        image.data.len()
    );
}

#[test]
fn decode_410_subsampling_with_fast_upsample() {
    let jpeg: Vec<u8> = make_jpeg_with_410_sampling();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    decoder.set_fast_upsample(true);
    let image = decoder.decode_image().unwrap();

    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.data.len(), 32 * 32 * 3);
}

// ---------------------------------------------------------------------------
// Helpers for C djpeg cross-validation
// ---------------------------------------------------------------------------

/// Locate the djpeg binary. Check /opt/homebrew/bin first, then $PATH.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew_path = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }

    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path = PathBuf::from(&path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Parse a binary PPM (P6) file into (width, height, rgb_pixels).
/// Returns `None` if the file is not a valid P6 PPM.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

    // Skip whitespace
    while pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
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
    while pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
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
    while pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
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

    // Exactly one whitespace character after maxval
    if pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// RAII temp file that deletes on drop.
struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        let path: PathBuf = std::env::temp_dir().join(name);
        Self { path }
    }

    fn write_bytes(&self, data: &[u8]) {
        let mut file = std::fs::File::create(&self.path)
            .unwrap_or_else(|e| panic!("Failed to create temp file {:?}: {:?}", self.path, e));
        file.write_all(data)
            .unwrap_or_else(|e| panic!("Failed to write temp file {:?}: {:?}", self.path, e));
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

// ---------------------------------------------------------------------------
// Test: C djpeg cross-validation for 4:1:0
// ---------------------------------------------------------------------------

/// Cross-validate Rust 4:1:0 decode against C djpeg.
///
/// 4:1:0 (H=4, V=2 luma, 1x1 chroma) is non-standard and may not be supported
/// by C djpeg. If djpeg cannot decode the file, the test skips gracefully.
/// If djpeg can decode it, we assert pixel-identical output (diff=0).
#[test]
fn c_djpeg_cross_validation_410() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Build the 4:1:0 JPEG using the same SOF-patching approach as other tests.
    let jpeg_data: Vec<u8> = make_jpeg_with_410_sampling();

    // Write JPEG to a temp file for C djpeg.
    let jpeg_temp = TempFile::new("cross_val_410.jpg");
    jpeg_temp.write_bytes(&jpeg_data);

    let ppm_temp = TempFile::new("cross_val_410.ppm");

    // Run C djpeg — 4:1:0 may not be supported, so check for failure.
    let djpeg_output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_temp.path)
        .arg(&jpeg_temp.path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    if !djpeg_output.status.success() {
        eprintln!(
            "SKIP: djpeg cannot decode 4:1:0 (exit code {:?}): {}",
            djpeg_output.status.code(),
            String::from_utf8_lossy(&djpeg_output.stderr)
        );
        return;
    }

    // C djpeg succeeded — parse its PPM output.
    let ppm_data: Vec<u8> = std::fs::read(&ppm_temp.path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_temp.path, e));
    let (c_width, c_height, c_pixels) =
        parse_ppm(&ppm_data).expect("Failed to parse PPM output from djpeg");

    // Decode with Rust.
    let rust_image = {
        let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg_data)
            .expect("Rust decoder init failed");
        decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
        decoder
            .decode_image()
            .unwrap_or_else(|e| panic!("Rust decode failed for 4:1:0: {:?}", e))
    };

    // Verify dimensions match.
    assert_eq!(
        rust_image.width, c_width,
        "Width mismatch for 4:1:0: Rust={} C={}",
        rust_image.width, c_width
    );
    assert_eq!(
        rust_image.height, c_height,
        "Height mismatch for 4:1:0: Rust={} C={}",
        rust_image.height, c_height
    );

    // Assert pixel-exact match (diff=0).
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "Data length mismatch for 4:1:0: Rust={} C={}",
        rust_image.data.len(),
        c_pixels.len()
    );
    assert_eq!(
        rust_image.data, c_pixels,
        "Pixel data mismatch for 4:1:0: Rust and C djpeg outputs differ"
    );

    eprintln!(
        "PASS: 4:1:0 — {}x{} pixels match exactly",
        c_width, c_height
    );
}
