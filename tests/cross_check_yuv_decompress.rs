//! Cross-validation of YUV decompression APIs against C libjpeg-turbo (djpeg).
//!
//! Each test compresses RGB to JPEG, decompresses to YUV via Rust, re-encodes
//! the YUV back to JPEG, then verifies that C djpeg decodes it identically to
//! Rust direct decode of the same source.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::api::yuv;
use libjpeg_turbo_rs::{
    compress, decompress_to, yuv_buf_size, yuv_plane_size, PixelFormat, Subsampling,
};

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

// ===========================================================================
// Helpers
// ===========================================================================

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path to avoid collisions in parallel tests.
fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_rs_yuv_dec_{}_{:04}_{}", pid, counter, name))
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

/// Generate a 48x48 gradient RGB test image with varied pixel values.
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

/// Parse a binary PPM (P6) file into (width, height, rgb_pixels).
/// Returns `None` if the file is not a valid P6 PPM.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

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

    skip_ws_comments(&mut pos);

    // Parse width
    let width_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let width: usize = std::str::from_utf8(&data[width_start..pos])
        .ok()?
        .parse()
        .ok()?;

    skip_ws_comments(&mut pos);

    // Parse height
    let height_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let height: usize = std::str::from_utf8(&data[height_start..pos])
        .ok()?
        .parse()
        .ok()?;

    skip_ws_comments(&mut pos);

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
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Parse a binary PGM (P5) file into (width, height, gray_pixels).
/// Returns `None` if the file is not a valid P5 PGM.
fn parse_pgm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P5" {
        return None;
    }
    let mut pos: usize = 2;

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

    skip_ws_comments(&mut pos);

    let width_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let width: usize = std::str::from_utf8(&data[width_start..pos])
        .ok()?
        .parse()
        .ok()?;

    skip_ws_comments(&mut pos);

    let height_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let height: usize = std::str::from_utf8(&data[height_start..pos])
        .ok()?
        .parse()
        .ok()?;

    skip_ws_comments(&mut pos);

    let maxval_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let _maxval: usize = std::str::from_utf8(&data[maxval_start..pos])
        .ok()?
        .parse()
        .ok()?;

    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }

    let expected_len: usize = width * height;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

/// Compare two pixel buffers, asserting diff=0. Logs the first 5 mismatches.
fn assert_pixels_identical(label: &str, rust_pixels: &[u8], c_pixels: &[u8], channels: usize) {
    assert_eq!(
        rust_pixels.len(),
        c_pixels.len(),
        "{}: pixel buffer length mismatch: rust={} c={}",
        label,
        rust_pixels.len(),
        c_pixels.len()
    );

    let mut mismatches: usize = 0;
    let mut max_diff: i16 = 0;
    let channel_names: &[&str] = if channels == 3 {
        &["R", "G", "B"]
    } else {
        &["Y"]
    };

    for (i, (&a, &b)) in rust_pixels.iter().zip(c_pixels.iter()).enumerate() {
        let diff: i16 = (a as i16 - b as i16).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / channels;
                let ch: &str = channel_names[i % channels];
                eprintln!(
                    "  {} pixel {} channel {}: rust={} c={} diff={}",
                    label, pixel, ch, a, b, diff
                );
            }
        }
    }

    assert_eq!(
        mismatches, 0,
        "{}: {} pixels differ (max_diff={}), expected diff=0",
        label, mismatches, max_diff
    );
}

/// Decode a JPEG with C djpeg and return the raw pixel data as PPM.
/// Returns `(width, height, rgb_pixels)`.
fn decode_with_djpeg(djpeg: &Path, jpeg_data: &[u8]) -> (usize, usize, Vec<u8>) {
    let tmp: TempFile = TempFile::new("djpeg_input.jpg");
    std::fs::write(tmp.path(), jpeg_data)
        .unwrap_or_else(|e| panic!("failed to write temp JPEG: {}", e));

    let output = Command::new(djpeg)
        .args(["-ppm"])
        .arg(tmp.path())
        .output()
        .unwrap_or_else(|e| panic!("djpeg execution failed: {}", e));
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // djpeg outputs P6 for color, P5 for grayscale
    if let Some(ppm) = parse_ppm(&output.stdout) {
        ppm
    } else if let Some((w, h, gray)) = parse_pgm(&output.stdout) {
        // Return grayscale as-is (1 channel)
        (w, h, gray)
    } else {
        panic!("failed to parse djpeg output as PPM or PGM");
    }
}

// ===========================================================================
// decompress_to_yuv tests (packed buffer)
// ===========================================================================

/// Helper for decompress_to_yuv cross-validation with a given subsampling.
///
/// 1. Compress 48x48 gradient to JPEG with given subsampling
/// 2. Call decompress_to_yuv() to get packed YUV buffer
/// 3. Re-compress YUV to JPEG via compress_from_yuv()
/// 4. Decode re-compressed JPEG with C djpeg
/// 5. Also decode re-compressed JPEG with Rust decompress
/// 6. Assert Rust and C outputs are pixel-identical (diff=0)
fn decompress_to_yuv_xval_helper(djpeg: &Path, subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let rgb: Vec<u8> = generate_gradient(w, h);

    // Step 1: Compress to JPEG
    let jpeg: Vec<u8> =
        compress(&rgb, w, h, PixelFormat::Rgb, 90, subsamp).expect("compress failed");

    // Step 2: Decompress to packed YUV
    let (yuv_buf, yuv_w, yuv_h, yuv_ss) =
        yuv::decompress_to_yuv(&jpeg).expect("decompress_to_yuv failed");
    assert_eq!(
        (yuv_w, yuv_h),
        (w, h),
        "YUV dimensions mismatch for {:?}",
        subsamp
    );
    assert_eq!(yuv_ss, subsamp, "YUV subsampling mismatch");
    assert_eq!(
        yuv_buf.len(),
        yuv_buf_size(w, h, subsamp),
        "YUV buffer size mismatch for {:?}",
        subsamp
    );

    // Step 3: Re-compress YUV to JPEG
    let re_jpeg: Vec<u8> =
        yuv::compress_from_yuv(&yuv_buf, w, h, subsamp, 90).expect("compress_from_yuv failed");
    assert!(
        !re_jpeg.is_empty(),
        "re-compressed JPEG is empty for {:?}",
        subsamp
    );

    // Step 4: Decode with C djpeg
    let (c_w, c_h, c_pixels) = decode_with_djpeg(djpeg, &re_jpeg);
    assert_eq!(
        (c_w, c_h),
        (w, h),
        "C djpeg dimensions mismatch for {:?}",
        subsamp
    );

    // Step 5: Decode with Rust
    let rust_img = decompress_to(&re_jpeg, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("Rust decompress failed for {:?}: {}", subsamp, e));
    assert_eq!(
        (rust_img.width, rust_img.height),
        (w, h),
        "Rust decode dimensions mismatch for {:?}",
        subsamp
    );

    // Step 6: Compare Rust vs C djpeg (diff=0)
    let label: String = format!("decompress_to_yuv {:?}", subsamp);
    assert_pixels_identical(&label, &rust_img.data, &c_pixels, 3);
}

#[test]
fn c_xval_decompress_to_yuv_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    decompress_to_yuv_xval_helper(&djpeg, Subsampling::S420);
}

#[test]
fn c_xval_decompress_to_yuv_422() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    decompress_to_yuv_xval_helper(&djpeg, Subsampling::S422);
}

#[test]
fn c_xval_decompress_to_yuv_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    decompress_to_yuv_xval_helper(&djpeg, Subsampling::S444);
}

// ===========================================================================
// decompress_to_yuv_planes tests (separate planes)
// ===========================================================================

/// Helper for decompress_to_yuv_planes cross-validation with a given subsampling.
///
/// 1. Compress 48x48 gradient to JPEG with given subsampling
/// 2. Call decompress_to_yuv_planes() to get separate planes
/// 3. Verify plane count (3) and plane sizes match yuv_plane_size
/// 4. Reassemble planes into JPEG via compress_from_yuv_planes()
/// 5. Decode re-compressed JPEG with C djpeg and Rust
/// 6. Assert diff=0
fn decompress_to_yuv_planes_xval_helper(djpeg: &Path, subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let rgb: Vec<u8> = generate_gradient(w, h);

    // Step 1: Compress to JPEG
    let jpeg: Vec<u8> =
        compress(&rgb, w, h, PixelFormat::Rgb, 90, subsamp).expect("compress failed");

    // Step 2: Decompress to separate YUV planes
    let (planes, planes_w, planes_h, planes_ss) =
        yuv::decompress_to_yuv_planes(&jpeg).expect("decompress_to_yuv_planes failed");
    assert_eq!(
        (planes_w, planes_h),
        (w, h),
        "planes dimensions mismatch for {:?}",
        subsamp
    );
    assert_eq!(planes_ss, subsamp, "planes subsampling mismatch");

    // Step 3: Verify plane count and sizes
    assert_eq!(
        planes.len(),
        3,
        "expected 3 planes for {:?}, got {}",
        subsamp,
        planes.len()
    );
    let expected_y_size: usize = yuv_plane_size(0, w, h, subsamp);
    let expected_cb_size: usize = yuv_plane_size(1, w, h, subsamp);
    let expected_cr_size: usize = yuv_plane_size(2, w, h, subsamp);
    assert_eq!(
        planes[0].len(),
        expected_y_size,
        "Y plane size mismatch for {:?}: got {} expected {}",
        subsamp,
        planes[0].len(),
        expected_y_size
    );
    assert_eq!(
        planes[1].len(),
        expected_cb_size,
        "Cb plane size mismatch for {:?}: got {} expected {}",
        subsamp,
        planes[1].len(),
        expected_cb_size
    );
    assert_eq!(
        planes[2].len(),
        expected_cr_size,
        "Cr plane size mismatch for {:?}: got {} expected {}",
        subsamp,
        planes[2].len(),
        expected_cr_size
    );

    // Step 4: Re-compress from planes to JPEG
    let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
    let re_jpeg: Vec<u8> = yuv::compress_from_yuv_planes(&plane_refs, w, h, subsamp, 90)
        .expect("compress_from_yuv_planes failed");
    assert!(
        !re_jpeg.is_empty(),
        "re-compressed JPEG is empty for {:?}",
        subsamp
    );

    // Step 5: Decode with C djpeg
    let (c_w, c_h, c_pixels) = decode_with_djpeg(djpeg, &re_jpeg);
    assert_eq!(
        (c_w, c_h),
        (w, h),
        "C djpeg dimensions mismatch for planes {:?}",
        subsamp
    );

    // Step 6: Decode with Rust
    let rust_img = decompress_to(&re_jpeg, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("Rust decompress failed for planes {:?}: {}", subsamp, e));
    assert_eq!(
        (rust_img.width, rust_img.height),
        (w, h),
        "Rust decode dimensions mismatch for planes {:?}",
        subsamp
    );

    // Step 7: Compare Rust vs C djpeg (diff=0)
    let label: String = format!("decompress_to_yuv_planes {:?}", subsamp);
    assert_pixels_identical(&label, &rust_img.data, &c_pixels, 3);
}

#[test]
fn c_xval_decompress_to_yuv_planes_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    decompress_to_yuv_planes_xval_helper(&djpeg, Subsampling::S420);
}

#[test]
fn c_xval_decompress_to_yuv_planes_422() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    decompress_to_yuv_planes_xval_helper(&djpeg, Subsampling::S422);
}

#[test]
fn c_xval_decompress_to_yuv_planes_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    decompress_to_yuv_planes_xval_helper(&djpeg, Subsampling::S444);
}

// ===========================================================================
// Grayscale decompress_to_yuv test
// ===========================================================================

#[test]
fn c_xval_decompress_to_yuv_grayscale() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);

    // Generate grayscale image
    let mut gray_pixels: Vec<u8> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            gray_pixels.push(((x + y) * 4) as u8);
        }
    }

    // Compress grayscale to JPEG
    let jpeg: Vec<u8> = compress(
        &gray_pixels,
        w,
        h,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
    )
    .expect("grayscale compress failed");

    // Decompress to YUV (should contain only Y plane data)
    let (yuv_buf, yuv_w, yuv_h, _yuv_ss) =
        yuv::decompress_to_yuv(&jpeg).expect("decompress_to_yuv failed for grayscale");
    assert_eq!((yuv_w, yuv_h), (w, h), "grayscale YUV dimensions mismatch");

    // Verify the YUV buffer is Y-only (size = width * height, no chroma planes)
    let y_plane_size: usize = yuv_plane_size(0, w, h, Subsampling::S444);
    assert_eq!(
        yuv_buf.len(),
        y_plane_size,
        "grayscale YUV buffer should contain only Y plane: got {} expected {}",
        yuv_buf.len(),
        y_plane_size
    );

    // Re-compress from YUV to JPEG
    let re_jpeg: Vec<u8> = yuv::compress_from_yuv(&yuv_buf, w, h, Subsampling::S444, 90)
        .expect("compress_from_yuv failed for grayscale");
    assert!(!re_jpeg.is_empty(), "re-compressed grayscale JPEG is empty");

    // Decode with C djpeg (-ppm outputs P5 PGM for grayscale)
    let tmp: TempFile = TempFile::new("gray_djpeg.jpg");
    std::fs::write(tmp.path(), &re_jpeg)
        .unwrap_or_else(|e| panic!("failed to write temp JPEG: {}", e));

    let c_output = Command::new(&djpeg)
        .args(["-ppm"])
        .arg(tmp.path())
        .output()
        .unwrap_or_else(|e| panic!("djpeg execution failed: {}", e));
    assert!(
        c_output.status.success(),
        "djpeg failed for grayscale: {}",
        String::from_utf8_lossy(&c_output.stderr)
    );

    // djpeg outputs P5 PGM for grayscale JPEG
    let (c_w, c_h, c_gray) = parse_pgm(&c_output.stdout)
        .unwrap_or_else(|| panic!("failed to parse djpeg PGM output for grayscale"));
    assert_eq!((c_w, c_h), (w, h), "C djpeg grayscale dimensions mismatch");

    // Decode with Rust
    let rust_img = decompress_to(&re_jpeg, PixelFormat::Grayscale)
        .unwrap_or_else(|e| panic!("Rust decompress failed for grayscale: {}", e));
    assert_eq!(
        (rust_img.width, rust_img.height),
        (w, h),
        "Rust grayscale decode dimensions mismatch"
    );

    // Compare Rust vs C djpeg (diff=0, grayscale = 1 channel)
    assert_pixels_identical("decompress_to_yuv grayscale", &rust_img.data, &c_gray, 1);
}

// ===========================================================================
// YUV roundtrip: encode_yuv -> compress_from_yuv -> djpeg vs Rust
// ===========================================================================

#[test]
fn c_xval_yuv_roundtrip_all_subsamplings() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    let rgb: Vec<u8> = generate_gradient(w, h);
    let subsamplings: [Subsampling; 3] = [Subsampling::S444, Subsampling::S422, Subsampling::S420];

    for &ss in &subsamplings {
        // Step 1: Encode RGB to YUV
        let yuv: Vec<u8> = yuv::encode_yuv(&rgb, w, h, PixelFormat::Rgb, ss)
            .unwrap_or_else(|e| panic!("encode_yuv failed for {:?}: {}", ss, e));
        assert_eq!(
            yuv.len(),
            yuv_buf_size(w, h, ss),
            "YUV buffer size mismatch for {:?}",
            ss
        );

        // Step 2: Compress YUV to JPEG
        let jpeg: Vec<u8> = yuv::compress_from_yuv(&yuv, w, h, ss, 90)
            .unwrap_or_else(|e| panic!("compress_from_yuv failed for {:?}: {}", ss, e));
        assert!(!jpeg.is_empty(), "JPEG output is empty for {:?}", ss);

        // Step 3: Decode with C djpeg
        let (c_w, c_h, c_pixels) = decode_with_djpeg(&djpeg, &jpeg);
        assert_eq!(
            (c_w, c_h),
            (w, h),
            "C djpeg dimensions mismatch for roundtrip {:?}",
            ss
        );

        // Step 4: Decode with Rust
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("Rust decompress failed for roundtrip {:?}: {}", ss, e));
        assert_eq!(
            (rust_img.width, rust_img.height),
            (w, h),
            "Rust decode dimensions mismatch for roundtrip {:?}",
            ss
        );

        // Step 5: Both must match (diff=0)
        let label: String = format!("yuv_roundtrip {:?}", ss);
        assert_pixels_identical(&label, &rust_img.data, &c_pixels, 3);
    }
}
