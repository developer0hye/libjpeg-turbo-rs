//! Cross-validation of decompress_raw / compress_raw against C libjpeg-turbo.
//!
//! Tests raw YCbCr plane decomposition and re-encoding via compress_raw,
//! then verifies pixel-identical output between Rust and C djpeg.

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, compress_raw, decompress_raw, decompress_to, PixelFormat, RawImage, Subsampling,
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

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid: u32 = std::process::id();
        Self {
            path: std::env::temp_dir()
                .join(format!("rawdec_xval_{}_{:04}_{}", pid, counter, suffix)),
        }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Generate a gradient RGB test image with varied pixel values.
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
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM data too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    pos = next + 1; // one whitespace byte before binary data
    let expected_len: usize = width * height * 3;
    assert!(
        data.len() - pos >= expected_len,
        "PPM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );
    (width, height, data[pos..pos + expected_len].to_vec())
}

/// Parse a binary PGM (P5) file and return `(width, height, gray_pixels)`.
fn parse_pgm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PGM data too short");
    assert_eq!(&data[0..2], b"P5", "not a P5 PGM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    pos = next + 1;
    let expected_len: usize = width * height;
    assert!(
        data.len() - pos >= expected_len,
        "PGM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );
    (width, height, data[pos..pos + expected_len].to_vec())
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
        .expect("non-UTF8 in header")
        .parse()
        .expect("invalid number in header");
    (val, end)
}

/// Decode a JPEG with C djpeg to PPM and return (width, height, rgb_pixels).
fn decode_with_c_djpeg(djpeg: &PathBuf, jpeg_data: &[u8], label: &str) -> (usize, usize, Vec<u8>) {
    let tmp_jpg = TempFile::new(&format!("{label}.jpg"));
    let tmp_ppm = TempFile::new(&format!("{label}.ppm"));

    std::fs::write(tmp_jpg.path(), jpeg_data).expect("write tmp jpg");

    let output = Command::new(djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "[{label}] djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(tmp_ppm.path()).expect("read PPM output");
    parse_ppm(&ppm_data)
}

/// Decode a grayscale JPEG with C djpeg to PGM and return (width, height, gray_pixels).
fn decode_grayscale_with_c_djpeg(
    djpeg: &PathBuf,
    jpeg_data: &[u8],
    label: &str,
) -> (usize, usize, Vec<u8>) {
    let tmp_jpg = TempFile::new(&format!("{label}.jpg"));
    let tmp_pgm = TempFile::new(&format!("{label}.pgm"));

    std::fs::write(tmp_jpg.path(), jpeg_data).expect("write tmp jpg");

    let output = Command::new(djpeg)
        .arg("-grayscale")
        .arg("-outfile")
        .arg(tmp_pgm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "[{label}] djpeg -grayscale failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let pgm_data: Vec<u8> = std::fs::read(tmp_pgm.path()).expect("read PGM output");
    parse_pgm(&pgm_data)
}

/// Compare two RGB pixel buffers. Asserts diff=0 and logs first 5 mismatches.
fn compare_rgb_pixels(rust_rgb: &[u8], c_rgb: &[u8], label: &str) {
    assert_eq!(
        rust_rgb.len(),
        c_rgb.len(),
        "[{label}] RGB data length mismatch: rust={} c={}",
        rust_rgb.len(),
        c_rgb.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatches: usize = 0;
    for (i, (&r, &c)) in rust_rgb.iter().zip(c_rgb.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatches += 1;
            if mismatches <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  [{label}] pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert_eq!(
        max_diff, 0,
        "[{label}] max_diff={max_diff} mismatches={mismatches} (must be 0)"
    );
}

// ===========================================================================
// Helper: raw roundtrip via compress_raw + djpeg, compared to direct decode
// ===========================================================================

/// Compress gradient -> decompress_raw -> verify plane dimensions ->
/// compress_raw -> decode with both Rust and C djpeg -> compare diff=0.
fn raw_decompress_roundtrip(
    djpeg: &PathBuf,
    subsampling: Subsampling,
    width: usize,
    height: usize,
    label: &str,
) {
    let rgb: Vec<u8> = generate_gradient(width, height);
    let quality: u8 = 90;

    // Step 1: Compress RGB to JPEG with the given subsampling
    let jpeg_data: Vec<u8> = compress(&rgb, width, height, PixelFormat::Rgb, quality, subsampling)
        .unwrap_or_else(|e| panic!("[{label}] compress failed: {e}"));

    // Step 2: Decompress to raw YCbCr planes
    let raw: RawImage = decompress_raw(&jpeg_data)
        .unwrap_or_else(|e| panic!("[{label}] decompress_raw failed: {e}"));

    assert_eq!(raw.width, width, "[{label}] raw width mismatch");
    assert_eq!(raw.height, height, "[{label}] raw height mismatch");
    assert_eq!(
        raw.num_components, 3,
        "[{label}] expected 3 components, got {}",
        raw.num_components
    );

    // Verify plane dimensions based on subsampling
    let (expected_chroma_w, expected_chroma_h): (usize, usize) = match subsampling {
        Subsampling::S444 => (raw.plane_widths[0], raw.plane_heights[0]),
        Subsampling::S422 => ((raw.plane_widths[0] + 1) / 2, raw.plane_heights[0]),
        Subsampling::S420 => (
            (raw.plane_widths[0] + 1) / 2,
            (raw.plane_heights[0] + 1) / 2,
        ),
        _ => panic!("[{label}] unexpected subsampling"),
    };

    // Cb plane
    assert_eq!(
        raw.plane_widths[1], expected_chroma_w,
        "[{label}] Cb width: got {} expected {}",
        raw.plane_widths[1], expected_chroma_w
    );
    assert_eq!(
        raw.plane_heights[1], expected_chroma_h,
        "[{label}] Cb height: got {} expected {}",
        raw.plane_heights[1], expected_chroma_h
    );
    // Cr plane
    assert_eq!(
        raw.plane_widths[2], expected_chroma_w,
        "[{label}] Cr width: got {} expected {}",
        raw.plane_widths[2], expected_chroma_w
    );
    assert_eq!(
        raw.plane_heights[2], expected_chroma_h,
        "[{label}] Cr height: got {} expected {}",
        raw.plane_heights[2], expected_chroma_h
    );

    // Step 3: Re-encode raw planes with compress_raw
    let plane_refs: Vec<&[u8]> = raw.planes.iter().map(|p| p.as_slice()).collect();
    let re_jpeg: Vec<u8> = compress_raw(
        &plane_refs,
        &raw.plane_widths,
        &raw.plane_heights,
        raw.width,
        raw.height,
        quality,
        subsampling,
    )
    .unwrap_or_else(|e| panic!("[{label}] compress_raw failed: {e}"));

    // Step 4: Decode original JPEG with Rust
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("[{label}] Rust decompress original failed: {e}"));

    // Step 5: Decode re-encoded JPEG with C djpeg
    let (c_w, c_h, c_rgb) = decode_with_c_djpeg(djpeg, &re_jpeg, &format!("{label}_reenc"));
    assert_eq!(c_w, width, "[{label}] C djpeg width mismatch");
    assert_eq!(c_h, height, "[{label}] C djpeg height mismatch");

    // Step 6: Decode re-encoded JPEG with Rust
    let rust_reenc = decompress_to(&re_jpeg, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("[{label}] Rust decompress re-encoded failed: {e}"));

    // Step 7: Compare Rust re-encoded decode vs C djpeg re-encoded decode -> diff=0
    compare_rgb_pixels(&rust_reenc.data, &c_rgb, &format!("{label}_rust_vs_c"));

    eprintln!(
        "[{label}] PASS: original {}x{}, raw planes OK, re-encode+decode Rust==C (diff=0, {} pixels)",
        rust_img.width,
        rust_img.height,
        width * height
    );
}

// ===========================================================================
// Tests
// ===========================================================================

#[test]
fn c_xval_decompress_raw_420() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    raw_decompress_roundtrip(&djpeg, Subsampling::S420, 48, 48, "raw_420");
}

#[test]
fn c_xval_decompress_raw_444() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    raw_decompress_roundtrip(&djpeg, Subsampling::S444, 48, 48, "raw_444");
}

#[test]
fn c_xval_decompress_raw_422() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    raw_decompress_roundtrip(&djpeg, Subsampling::S422, 48, 48, "raw_422");
}

#[test]
fn c_xval_decompress_raw_grayscale() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let quality: u8 = 90;
    let label: &str = "raw_gray";

    // Generate grayscale source image
    let gray: Vec<u8> = {
        let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
        for y in 0..height {
            for x in 0..width {
                let val: u8 = (((x + y) * 255) / (width + height).max(1)) as u8;
                pixels.push(val);
            }
        }
        pixels
    };

    // Compress as grayscale JPEG
    let jpeg_data: Vec<u8> = compress(
        &gray,
        width,
        height,
        PixelFormat::Grayscale,
        quality,
        Subsampling::S444,
    )
    .unwrap_or_else(|e| panic!("[{label}] compress grayscale failed: {e}"));

    // Decompress to raw planes
    let raw: RawImage = decompress_raw(&jpeg_data)
        .unwrap_or_else(|e| panic!("[{label}] decompress_raw failed: {e}"));

    assert_eq!(raw.width, width, "[{label}] raw width mismatch");
    assert_eq!(raw.height, height, "[{label}] raw height mismatch");
    assert_eq!(
        raw.num_components, 1,
        "[{label}] expected 1 component for grayscale, got {}",
        raw.num_components
    );
    assert_eq!(
        raw.planes.len(),
        1,
        "[{label}] expected 1 plane, got {}",
        raw.planes.len()
    );

    // Re-encode raw plane with compress_raw
    let plane_refs: Vec<&[u8]> = raw.planes.iter().map(|p| p.as_slice()).collect();
    let re_jpeg: Vec<u8> = compress_raw(
        &plane_refs,
        &raw.plane_widths,
        &raw.plane_heights,
        raw.width,
        raw.height,
        quality,
        Subsampling::S444,
    )
    .unwrap_or_else(|e| panic!("[{label}] compress_raw failed: {e}"));

    // Decode re-encoded JPEG with C djpeg (grayscale output)
    let (c_w, c_h, _c_gray) =
        decode_grayscale_with_c_djpeg(&djpeg, &re_jpeg, &format!("{label}_reenc"));
    assert_eq!(c_w, width, "[{label}] C djpeg width mismatch");
    assert_eq!(c_h, height, "[{label}] C djpeg height mismatch");

    eprintln!(
        "[{label}] PASS: grayscale {}x{}, 1 plane, re-encode+C djpeg decode OK",
        width, height
    );
}

#[test]
fn c_xval_compress_raw_420_vs_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let quality: u8 = 90;
    let label: &str = "compress_raw_420";

    // Generate YCbCr planes for 4:2:0: Y full, Cb/Cr half in both dims
    let y_w: usize = width;
    let y_h: usize = height;
    let c_w: usize = width / 2;
    let c_h: usize = height / 2;

    let y_plane: Vec<u8> = (0..y_w * y_h)
        .map(|i| ((i * 200) / (y_w * y_h).max(1) + 16) as u8)
        .collect();
    let cb_plane: Vec<u8> = (0..c_w * c_h)
        .map(|i| ((i * 100) / (c_w * c_h).max(1) + 100) as u8)
        .collect();
    let cr_plane: Vec<u8> = (0..c_w * c_h)
        .map(|i| ((i * 80) / (c_w * c_h).max(1) + 120) as u8)
        .collect();

    let planes: Vec<&[u8]> = vec![&y_plane, &cb_plane, &cr_plane];
    let plane_widths: Vec<usize> = vec![y_w, c_w, c_w];
    let plane_heights: Vec<usize> = vec![y_h, c_h, c_h];

    let jpeg_data: Vec<u8> = compress_raw(
        &planes,
        &plane_widths,
        &plane_heights,
        width,
        height,
        quality,
        Subsampling::S420,
    )
    .unwrap_or_else(|e| panic!("[{label}] compress_raw failed: {e}"));

    // Decode with C djpeg
    let (djpeg_w, djpeg_h, _djpeg_rgb) =
        decode_with_c_djpeg(&djpeg, &jpeg_data, &format!("{label}_c"));
    assert_eq!(djpeg_w, width, "[{label}] C djpeg width mismatch");
    assert_eq!(djpeg_h, height, "[{label}] C djpeg height mismatch");

    // Decode with Rust
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("[{label}] Rust decompress failed: {e}"));
    assert_eq!(rust_img.width, width, "[{label}] Rust width mismatch");
    assert_eq!(rust_img.height, height, "[{label}] Rust height mismatch");

    // Compare dimensions (both must succeed with correct dims)
    compare_rgb_pixels(&rust_img.data, &_djpeg_rgb, &format!("{label}_rust_vs_c"));

    eprintln!(
        "[{label}] PASS: compress_raw 4:2:0 {}x{}, C djpeg + Rust decode OK, diff=0",
        width, height
    );
}

#[test]
fn c_xval_compress_raw_444_vs_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let quality: u8 = 90;
    let label: &str = "compress_raw_444";

    // All planes full size for 4:4:4
    let plane_size: usize = width * height;
    let y_plane: Vec<u8> = (0..plane_size)
        .map(|i| ((i * 200) / plane_size.max(1) + 16) as u8)
        .collect();
    let cb_plane: Vec<u8> = (0..plane_size)
        .map(|i| ((i * 100) / plane_size.max(1) + 100) as u8)
        .collect();
    let cr_plane: Vec<u8> = (0..plane_size)
        .map(|i| ((i * 80) / plane_size.max(1) + 120) as u8)
        .collect();

    let planes: Vec<&[u8]> = vec![&y_plane, &cb_plane, &cr_plane];
    let plane_widths: Vec<usize> = vec![width, width, width];
    let plane_heights: Vec<usize> = vec![height, height, height];

    let jpeg_data: Vec<u8> = compress_raw(
        &planes,
        &plane_widths,
        &plane_heights,
        width,
        height,
        quality,
        Subsampling::S444,
    )
    .unwrap_or_else(|e| panic!("[{label}] compress_raw failed: {e}"));

    // Decode with C djpeg
    let (djpeg_w, djpeg_h, djpeg_rgb) =
        decode_with_c_djpeg(&djpeg, &jpeg_data, &format!("{label}_c"));
    assert_eq!(djpeg_w, width, "[{label}] C djpeg width mismatch");
    assert_eq!(djpeg_h, height, "[{label}] C djpeg height mismatch");

    // Decode with Rust
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("[{label}] Rust decompress failed: {e}"));
    assert_eq!(rust_img.width, width, "[{label}] Rust width mismatch");
    assert_eq!(rust_img.height, height, "[{label}] Rust height mismatch");

    compare_rgb_pixels(&rust_img.data, &djpeg_rgb, &format!("{label}_rust_vs_c"));

    eprintln!(
        "[{label}] PASS: compress_raw 4:4:4 {}x{}, C djpeg + Rust decode OK, diff=0",
        width, height
    );
}

#[test]
fn c_xval_compress_raw_422_vs_djpeg() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let quality: u8 = 90;
    let label: &str = "compress_raw_422";

    // 4:2:2: Y full, Cb/Cr half width, full height
    let y_w: usize = width;
    let y_h: usize = height;
    let c_w: usize = width / 2;
    let c_h: usize = height;

    let y_plane: Vec<u8> = (0..y_w * y_h)
        .map(|i| ((i * 200) / (y_w * y_h).max(1) + 16) as u8)
        .collect();
    let cb_plane: Vec<u8> = (0..c_w * c_h)
        .map(|i| ((i * 100) / (c_w * c_h).max(1) + 100) as u8)
        .collect();
    let cr_plane: Vec<u8> = (0..c_w * c_h)
        .map(|i| ((i * 80) / (c_w * c_h).max(1) + 120) as u8)
        .collect();

    let planes: Vec<&[u8]> = vec![&y_plane, &cb_plane, &cr_plane];
    let plane_widths: Vec<usize> = vec![y_w, c_w, c_w];
    let plane_heights: Vec<usize> = vec![y_h, c_h, c_h];

    let jpeg_data: Vec<u8> = compress_raw(
        &planes,
        &plane_widths,
        &plane_heights,
        width,
        height,
        quality,
        Subsampling::S422,
    )
    .unwrap_or_else(|e| panic!("[{label}] compress_raw failed: {e}"));

    // Decode with C djpeg
    let (djpeg_w, djpeg_h, djpeg_rgb) =
        decode_with_c_djpeg(&djpeg, &jpeg_data, &format!("{label}_c"));
    assert_eq!(djpeg_w, width, "[{label}] C djpeg width mismatch");
    assert_eq!(djpeg_h, height, "[{label}] C djpeg height mismatch");

    // Decode with Rust
    let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
        .unwrap_or_else(|e| panic!("[{label}] Rust decompress failed: {e}"));
    assert_eq!(rust_img.width, width, "[{label}] Rust width mismatch");
    assert_eq!(rust_img.height, height, "[{label}] Rust height mismatch");

    compare_rgb_pixels(&rust_img.data, &djpeg_rgb, &format!("{label}_rust_vs_c"));

    eprintln!(
        "[{label}] PASS: compress_raw 4:2:2 {}x{}, C djpeg + Rust decode OK, diff=0",
        width, height
    );
}

#[test]
fn c_xval_raw_roundtrip_consistency() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let quality: u8 = 90;
    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "roundtrip_444"),
        (Subsampling::S422, "roundtrip_422"),
        (Subsampling::S420, "roundtrip_420"),
    ];

    let rgb: Vec<u8> = generate_gradient(width, height);

    for &(subsampling, label) in subsamplings {
        // Step 1: Compress gradient to JPEG
        let jpeg_data: Vec<u8> =
            compress(&rgb, width, height, PixelFormat::Rgb, quality, subsampling)
                .unwrap_or_else(|e| panic!("[{label}] compress failed: {e}"));

        // Step 2: decompress_raw to get YCbCr planes
        let raw: RawImage = decompress_raw(&jpeg_data)
            .unwrap_or_else(|e| panic!("[{label}] decompress_raw failed: {e}"));

        // Step 3: compress_raw from those planes
        let plane_refs: Vec<&[u8]> = raw.planes.iter().map(|p| p.as_slice()).collect();
        let re_jpeg: Vec<u8> = compress_raw(
            &plane_refs,
            &raw.plane_widths,
            &raw.plane_heights,
            raw.width,
            raw.height,
            quality,
            subsampling,
        )
        .unwrap_or_else(|e| panic!("[{label}] compress_raw failed: {e}"));

        // Step 4: Decode re-encoded with Rust
        let rust_img = decompress_to(&re_jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("[{label}] Rust decompress re-encoded failed: {e}"));
        assert_eq!(rust_img.width, width, "[{label}] Rust width mismatch");
        assert_eq!(rust_img.height, height, "[{label}] Rust height mismatch");

        // Step 5: Decode re-encoded with C djpeg
        let (c_w, c_h, c_rgb) = decode_with_c_djpeg(&djpeg, &re_jpeg, label);
        assert_eq!(c_w, width, "[{label}] C djpeg width mismatch");
        assert_eq!(c_h, height, "[{label}] C djpeg height mismatch");

        // Step 6: Rust vs C must be diff=0
        compare_rgb_pixels(&rust_img.data, &c_rgb, label);

        eprintln!(
            "[{label}] PASS: roundtrip compress->raw->compress_raw->decode Rust==C (diff=0, {}x{})",
            width, height
        );
    }
}
