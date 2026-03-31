/// Integration tests for the YUV planar encode/decode API.
///
/// These tests cover the full set of `encode_yuv`, `decode_yuv`,
/// `compress_from_yuv`, `decompress_to_yuv` functions plus buffer size helpers.
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::api::yuv;
use libjpeg_turbo_rs::{
    compress, decompress, yuv_buf_size, yuv_plane_height, yuv_plane_size, yuv_plane_width,
    PixelFormat, Subsampling,
};

/// Helper: generate a simple gradient RGB image (3 bpp).
fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
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

// ──────────────────────────────────────────────
// 1. encode_yuv roundtrip: RGB → YUV → RGB
// ──────────────────────────────────────────────

#[test]
fn encode_decode_yuv_roundtrip_444() {
    let width: usize = 16;
    let height: usize = 16;
    let original: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> = yuv::encode_yuv(
        &original,
        width,
        height,
        PixelFormat::Rgb,
        Subsampling::S444,
    )
    .unwrap();
    let expected_size: usize = yuv_buf_size(width, height, Subsampling::S444);
    assert_eq!(yuv_packed.len(), expected_size);

    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S444,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), original.len());

    // BT.601 roundtrip has rounding error; accept +/- 2 per channel
    for i in 0..original.len() {
        let diff: i16 = original[i] as i16 - decoded[i] as i16;
        assert!(
            diff.abs() <= 2,
            "pixel byte {} differs by {}: original={}, decoded={}",
            i,
            diff,
            original[i],
            decoded[i]
        );
    }
}

#[test]
fn encode_decode_yuv_roundtrip_420() {
    let width: usize = 32;
    let height: usize = 32;
    let original: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> = yuv::encode_yuv(
        &original,
        width,
        height,
        PixelFormat::Rgb,
        Subsampling::S420,
    )
    .unwrap();
    let expected_size: usize = yuv_buf_size(width, height, Subsampling::S420);
    assert_eq!(yuv_packed.len(), expected_size);

    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), original.len());

    // With 4:2:0 subsampling, chroma is averaged so more error is expected
    let max_error: i16 = 40;
    let mut total_error: i64 = 0;
    for i in 0..original.len() {
        let diff: i16 = original[i] as i16 - decoded[i] as i16;
        total_error += diff.abs() as i64;
        assert!(
            diff.abs() <= max_error,
            "pixel byte {} differs by {}: original={}, decoded={}",
            i,
            diff,
            original[i],
            decoded[i]
        );
    }
    // Average error should be modest
    let avg_error: f64 = total_error as f64 / original.len() as f64;
    assert!(
        avg_error < 10.0,
        "average error {} too high for 4:2:0 roundtrip",
        avg_error
    );
}

// ──────────────────────────────────────────────
// 2. encode_yuv_planes produces correct plane sizes
// ──────────────────────────────────────────────

#[test]
fn encode_yuv_planes_correct_sizes_444() {
    let width: usize = 24;
    let height: usize = 16;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S444)
            .unwrap();
    assert_eq!(planes.len(), 3);

    for comp in 0..3 {
        let expected: usize = yuv_plane_size(comp, width, height, Subsampling::S444);
        assert_eq!(
            planes[comp].len(),
            expected,
            "plane {} size mismatch: got {}, expected {}",
            comp,
            planes[comp].len(),
            expected
        );
    }
}

#[test]
fn encode_yuv_planes_correct_sizes_420() {
    let width: usize = 32;
    let height: usize = 24;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420)
            .unwrap();
    assert_eq!(planes.len(), 3);

    let y_size: usize = yuv_plane_size(0, width, height, Subsampling::S420);
    let cb_size: usize = yuv_plane_size(1, width, height, Subsampling::S420);
    let cr_size: usize = yuv_plane_size(2, width, height, Subsampling::S420);

    assert_eq!(planes[0].len(), y_size);
    assert_eq!(planes[1].len(), cb_size);
    assert_eq!(planes[2].len(), cr_size);
    // For 4:2:0: chroma is 1/4 of luma
    assert_eq!(cb_size, y_size / 4);
    assert_eq!(cr_size, y_size / 4);
}

#[test]
fn encode_yuv_planes_correct_sizes_422() {
    let width: usize = 32;
    let height: usize = 16;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S422)
            .unwrap();
    assert_eq!(planes.len(), 3);

    for comp in 0..3 {
        let expected: usize = yuv_plane_size(comp, width, height, Subsampling::S422);
        assert_eq!(planes[comp].len(), expected);
    }
    // For 4:2:2: chroma width is half, height is same
    assert_eq!(planes[1].len(), planes[0].len() / 2);
}

// ──────────────────────────────────────────────
// 3. compress_from_yuv → decompress produces valid image
// ──────────────────────────────────────────────

#[test]
fn compress_from_yuv_produces_valid_jpeg() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> =
        yuv::encode_yuv(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420).unwrap();

    let jpeg_data: Vec<u8> =
        yuv::compress_from_yuv(&yuv_packed, width, height, Subsampling::S420, 90).unwrap();
    assert!(jpeg_data.len() > 2);
    assert_eq!(jpeg_data[0], 0xFF);
    assert_eq!(jpeg_data[1], 0xD8); // SOI marker

    // Decompress and check dimensions
    let image = decompress(&jpeg_data).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
}

#[test]
fn compress_from_yuv_planes_produces_valid_jpeg() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S444)
            .unwrap();
    let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();

    let jpeg_data: Vec<u8> =
        yuv::compress_from_yuv_planes(&plane_refs, width, height, Subsampling::S444, 90).unwrap();

    let image = decompress(&jpeg_data).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
}

// ──────────────────────────────────────────────
// 4. decompress_to_yuv → compress_from_yuv roundtrip
// ──────────────────────────────────────────────

#[test]
fn decompress_to_yuv_roundtrip() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        95,
        Subsampling::S420,
    )
    .unwrap();

    let (yuv_buf, dec_w, dec_h, dec_sub) = yuv::decompress_to_yuv(&jpeg_data).unwrap();
    assert_eq!(dec_w, width);
    assert_eq!(dec_h, height);
    assert_eq!(dec_sub, Subsampling::S420);

    // Re-compress from YUV
    let jpeg_data2: Vec<u8> = yuv::compress_from_yuv(&yuv_buf, dec_w, dec_h, dec_sub, 95).unwrap();
    let image2 = decompress(&jpeg_data2).unwrap();
    assert_eq!(image2.width, width);
    assert_eq!(image2.height, height);
}

#[test]
fn decompress_to_yuv_planes_roundtrip() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        95,
        Subsampling::S444,
    )
    .unwrap();

    let (planes, dec_w, dec_h, dec_sub) = yuv::decompress_to_yuv_planes(&jpeg_data).unwrap();
    assert_eq!(dec_w, width);
    assert_eq!(dec_h, height);
    assert_eq!(dec_sub, Subsampling::S444);
    assert_eq!(planes.len(), 3);

    let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();
    let jpeg_data2: Vec<u8> =
        yuv::compress_from_yuv_planes(&plane_refs, dec_w, dec_h, dec_sub, 95).unwrap();
    let image2 = decompress(&jpeg_data2).unwrap();
    assert_eq!(image2.width, width);
    assert_eq!(image2.height, height);
}

// ──────────────────────────────────────────────
// 5. decode_yuv_planes with 4:2:0 produces correct output size
// ──────────────────────────────────────────────

#[test]
fn decode_yuv_planes_420_correct_output_size() {
    let width: usize = 48;
    let height: usize = 32;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let planes: Vec<Vec<u8>> =
        yuv::encode_yuv_planes(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420)
            .unwrap();
    let plane_refs: Vec<&[u8]> = planes.iter().map(|p| p.as_slice()).collect();

    let decoded: Vec<u8> = yuv::decode_yuv_planes(
        &plane_refs,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), width * height * 3);

    // Also test RGBA output
    let decoded_rgba: Vec<u8> = yuv::decode_yuv_planes(
        &plane_refs,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgba,
    )
    .unwrap();
    assert_eq!(decoded_rgba.len(), width * height * 4);
}

// ──────────────────────────────────────────────
// 6. Buffer size helpers return correct values
// ──────────────────────────────────────────────

#[test]
fn buffer_size_helpers_444() {
    let width: usize = 640;
    let height: usize = 480;

    // 4:4:4: all planes same size
    let pw0: usize = yuv_plane_width(0, width, Subsampling::S444);
    let pw1: usize = yuv_plane_width(1, width, Subsampling::S444);
    let ph0: usize = yuv_plane_height(0, height, Subsampling::S444);
    let ph1: usize = yuv_plane_height(1, height, Subsampling::S444);

    assert_eq!(pw0, 640);
    assert_eq!(pw1, 640);
    assert_eq!(ph0, 480);
    assert_eq!(ph1, 480);

    let total: usize = yuv_buf_size(width, height, Subsampling::S444);
    assert_eq!(total, 640 * 480 * 3);
}

#[test]
fn buffer_size_helpers_420() {
    let width: usize = 640;
    let height: usize = 480;

    let pw0: usize = yuv_plane_width(0, width, Subsampling::S420);
    let pw1: usize = yuv_plane_width(1, width, Subsampling::S420);
    let ph0: usize = yuv_plane_height(0, height, Subsampling::S420);
    let ph1: usize = yuv_plane_height(1, height, Subsampling::S420);

    assert_eq!(pw0, 640);
    assert_eq!(pw1, 320);
    assert_eq!(ph0, 480);
    assert_eq!(ph1, 240);

    let y_size: usize = yuv_plane_size(0, width, height, Subsampling::S420);
    let cb_size: usize = yuv_plane_size(1, width, height, Subsampling::S420);
    assert_eq!(y_size, 640 * 480);
    assert_eq!(cb_size, 320 * 240);

    let total: usize = yuv_buf_size(width, height, Subsampling::S420);
    assert_eq!(total, 640 * 480 + 2 * 320 * 240);
}

#[test]
fn buffer_size_helpers_422() {
    let width: usize = 640;
    let height: usize = 480;

    let pw0: usize = yuv_plane_width(0, width, Subsampling::S422);
    let pw1: usize = yuv_plane_width(1, width, Subsampling::S422);
    let ph0: usize = yuv_plane_height(0, height, Subsampling::S422);
    let ph1: usize = yuv_plane_height(1, height, Subsampling::S422);

    assert_eq!(pw0, 640);
    assert_eq!(pw1, 320);
    assert_eq!(ph0, 480);
    assert_eq!(ph1, 480);

    let total: usize = yuv_buf_size(width, height, Subsampling::S422);
    assert_eq!(total, 640 * 480 + 2 * 320 * 480);
}

#[test]
fn buffer_size_helpers_odd_dimensions() {
    // Odd dimensions should be padded up
    let pw0: usize = yuv_plane_width(0, 641, Subsampling::S420);
    let pw1: usize = yuv_plane_width(1, 641, Subsampling::S420);
    assert_eq!(pw0, 642); // padded to multiple of 2
    assert_eq!(pw1, 321);

    let ph0: usize = yuv_plane_height(0, 481, Subsampling::S420);
    let ph1: usize = yuv_plane_height(1, 481, Subsampling::S420);
    assert_eq!(ph0, 482); // padded to multiple of 2
    assert_eq!(ph1, 241);
}

// ──────────────────────────────────────────────
// 7. Grayscale YUV (single plane)
// ──────────────────────────────────────────────

#[test]
fn grayscale_encode_yuv_single_plane() {
    let width: usize = 16;
    let height: usize = 16;
    // Grayscale input
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let yuv_packed: Vec<u8> = yuv::encode_yuv(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        Subsampling::S444,
    )
    .unwrap();
    // For grayscale, the YUV buffer is just the Y plane (no Cb/Cr)
    let y_size: usize = yuv_plane_size(0, width, height, Subsampling::S444);
    assert_eq!(yuv_packed.len(), y_size);

    // Decode back to grayscale
    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S444,
        PixelFormat::Grayscale,
    )
    .unwrap();
    assert_eq!(decoded.len(), width * height);
    // Grayscale → Y is identity, so should be exact
    assert_eq!(decoded, pixels);
}

#[test]
fn grayscale_encode_yuv_planes_single_plane() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();

    let planes: Vec<Vec<u8>> = yuv::encode_yuv_planes(
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        Subsampling::S444,
    )
    .unwrap();
    assert_eq!(planes.len(), 1); // Only Y plane for grayscale
    assert_eq!(planes[0].len(), width * height);
}

// ──────────────────────────────────────────────
// Additional: BGRA pixel format support
// ──────────────────────────────────────────────

#[test]
fn encode_decode_yuv_bgra_format() {
    let width: usize = 16;
    let height: usize = 16;
    // BGRA input
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 4);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = 128;
            pixels.push(b); // B
            pixels.push(g); // G
            pixels.push(r); // R
            pixels.push(255); // A
        }
    }

    let yuv_packed: Vec<u8> =
        yuv::encode_yuv(&pixels, width, height, PixelFormat::Bgra, Subsampling::S444).unwrap();
    let expected_size: usize = yuv_buf_size(width, height, Subsampling::S444);
    assert_eq!(yuv_packed.len(), expected_size);

    // Decode back to BGRA
    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S444,
        PixelFormat::Bgra,
    )
    .unwrap();
    assert_eq!(decoded.len(), pixels.len());
}

// ──────────────────────────────────────────────
// Edge case: non-multiple-of-MCU dimensions
// ──────────────────────────────────────────────

#[test]
fn encode_decode_yuv_non_aligned_dimensions() {
    let width: usize = 17;
    let height: usize = 13;
    let pixels: Vec<u8> = gradient_rgb(width, height);

    let yuv_packed: Vec<u8> =
        yuv::encode_yuv(&pixels, width, height, PixelFormat::Rgb, Subsampling::S420).unwrap();

    let decoded: Vec<u8> = yuv::decode_yuv(
        &yuv_packed,
        width,
        height,
        Subsampling::S420,
        PixelFormat::Rgb,
    )
    .unwrap();
    assert_eq!(decoded.len(), width * height * 3);
}

// ──────────────────────────────────────────────
// C djpeg cross-validation helpers
// ──────────────────────────────────────────────

/// Path to C djpeg binary, or `None` if not installed.
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

/// Global atomic counter for unique temp file names across parallel tests.
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Create a unique temp file path.
fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_yuv_{}_{:04}_{}", pid, counter, name))
}

/// RAII temp file that deletes on drop.
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
/// `data` contains raw RGB bytes.
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PPM file");
    assert!(raw.len() > 3, "PPM too short");
    assert_eq!(&raw[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    idx = skip_whitespace_and_comments(&raw, idx);
    let (width, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (height, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (_maxval, next) = read_ascii_number(&raw, idx);
    // Exactly one whitespace byte after maxval before binary data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(
        data.len(),
        width * height * 3,
        "PPM pixel data length mismatch: expected {}, got {}",
        width * height * 3,
        data.len()
    );
    (width, height, data)
}

fn skip_whitespace_and_comments(data: &[u8], mut idx: usize) -> usize {
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

fn read_ascii_number(data: &[u8], idx: usize) -> (usize, usize) {
    let start: usize = idx;
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[start..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

// ──────────────────────────────────────────────
// 8. C djpeg cross-validation: YUV compress path
// ──────────────────────────────────────────────

/// Validates that the RGB → YUV → JPEG path produces output identical
/// to what Rust's standard decoder produces, and that C djpeg can decode
/// the result with diff=0 vs Rust decode.
#[test]
fn c_djpeg_cross_validation_yuv_compress() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let quality: u8 = 95;

    for &subsampling in &[Subsampling::S444, Subsampling::S422, Subsampling::S420] {
        let pixels: Vec<u8> = gradient_rgb(width, height);

        // RGB → YUV (packed)
        let yuv_packed: Vec<u8> =
            yuv::encode_yuv(&pixels, width, height, PixelFormat::Rgb, subsampling)
                .expect("encode_yuv failed");

        // YUV → JPEG
        let jpeg_data: Vec<u8> =
            yuv::compress_from_yuv(&yuv_packed, width, height, subsampling, quality)
                .expect("compress_from_yuv failed");

        // Verify JPEG starts with SOI marker
        assert_eq!(jpeg_data[0], 0xFF);
        assert_eq!(jpeg_data[1], 0xD8);

        // Decode with Rust
        let rust_image = decompress(&jpeg_data).expect("Rust decompress failed");
        assert_eq!(rust_image.width, width);
        assert_eq!(rust_image.height, height);

        // Decode with C djpeg
        let label: &str = match subsampling {
            Subsampling::S444 => "444",
            Subsampling::S422 => "422",
            Subsampling::S420 => "420",
            _ => "other",
        };
        let tmp_jpg: TempFile = TempFile::new(&format!("yuv_compress_{}.jpg", label));
        let tmp_ppm: TempFile = TempFile::new(&format!("yuv_compress_{}.ppm", label));
        std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");

        let output = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_ppm.path())
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run djpeg");

        assert!(
            output.status.success(),
            "djpeg failed for subsampling {}: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );

        let (c_width, c_height, c_pixels) = parse_ppm(tmp_ppm.path());
        assert_eq!(c_width, width, "C djpeg width mismatch for {}", label);
        assert_eq!(c_height, height, "C djpeg height mismatch for {}", label);

        // Compare Rust decode vs C djpeg decode: expect diff=0
        // Both decoders are processing the same JPEG, so output must match exactly.
        let rust_pixels: &[u8] = &rust_image.data;
        assert_eq!(
            rust_pixels.len(),
            c_pixels.len(),
            "pixel buffer size mismatch for {}",
            label
        );

        let max_diff: u8 = rust_pixels
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff, 0,
            "Rust vs C djpeg decode diff={} for subsampling {} (expected 0)",
            max_diff, label
        );
    }
}

// ──────────────────────────────────────────────
// 9. C djpeg cross-validation: YUV decompress path
// ──────────────────────────────────────────────

/// Validates that JPEG → YUV → JPEG roundtrip produces output that
/// C djpeg can decode. Compares both the original and re-encoded JPEG
/// via C djpeg to confirm C-compatibility of the YUV decompress path.
#[test]
fn c_djpeg_cross_validation_yuv_decompress() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    let quality: u8 = 95;

    for &subsampling in &[Subsampling::S444, Subsampling::S422, Subsampling::S420] {
        let pixels: Vec<u8> = gradient_rgb(width, height);

        // Create original JPEG via standard compress
        let original_jpeg: Vec<u8> = compress(
            &pixels,
            width,
            height,
            PixelFormat::Rgb,
            quality,
            subsampling,
        )
        .expect("compress failed");

        // Decompress JPEG to YUV
        let (yuv_buf, dec_w, dec_h, dec_sub) =
            yuv::decompress_to_yuv(&original_jpeg).expect("decompress_to_yuv failed");
        assert_eq!(dec_w, width);
        assert_eq!(dec_h, height);
        assert_eq!(dec_sub, subsampling);

        // Re-compress from YUV back to JPEG
        let reencoded_jpeg: Vec<u8> =
            yuv::compress_from_yuv(&yuv_buf, dec_w, dec_h, dec_sub, quality)
                .expect("compress_from_yuv failed");

        let label: &str = match subsampling {
            Subsampling::S444 => "444",
            Subsampling::S422 => "422",
            Subsampling::S420 => "420",
            _ => "other",
        };

        // Decode original JPEG with C djpeg
        let tmp_orig_jpg: TempFile = TempFile::new(&format!("yuv_decomp_orig_{}.jpg", label));
        let tmp_orig_ppm: TempFile = TempFile::new(&format!("yuv_decomp_orig_{}.ppm", label));
        std::fs::write(tmp_orig_jpg.path(), &original_jpeg).expect("write tmp orig jpg");

        let output_orig = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_orig_ppm.path())
            .arg(tmp_orig_jpg.path())
            .output()
            .expect("failed to run djpeg on original");

        assert!(
            output_orig.status.success(),
            "djpeg failed on original for {}: {}",
            label,
            String::from_utf8_lossy(&output_orig.stderr)
        );

        let (orig_c_w, orig_c_h, orig_c_pixels) = parse_ppm(tmp_orig_ppm.path());
        assert_eq!(orig_c_w, width);
        assert_eq!(orig_c_h, height);

        // Decode re-encoded JPEG with C djpeg
        let tmp_reenc_jpg: TempFile = TempFile::new(&format!("yuv_decomp_reenc_{}.jpg", label));
        let tmp_reenc_ppm: TempFile = TempFile::new(&format!("yuv_decomp_reenc_{}.ppm", label));
        std::fs::write(tmp_reenc_jpg.path(), &reencoded_jpeg).expect("write tmp reenc jpg");

        let output_reenc = Command::new(&djpeg)
            .arg("-ppm")
            .arg("-outfile")
            .arg(tmp_reenc_ppm.path())
            .arg(tmp_reenc_jpg.path())
            .output()
            .expect("failed to run djpeg on re-encoded");

        assert!(
            output_reenc.status.success(),
            "djpeg failed on re-encoded JPEG for {}: {}",
            label,
            String::from_utf8_lossy(&output_reenc.stderr)
        );

        let (reenc_c_w, reenc_c_h, reenc_c_pixels) = parse_ppm(tmp_reenc_ppm.path());
        assert_eq!(reenc_c_w, width);
        assert_eq!(reenc_c_h, height);

        // Also decode re-encoded JPEG with Rust
        let rust_reenc_image =
            decompress(&reencoded_jpeg).expect("Rust decompress of re-encoded failed");
        assert_eq!(rust_reenc_image.width, width);
        assert_eq!(rust_reenc_image.height, height);

        // Rust decode of re-encoded JPEG vs C djpeg decode of re-encoded JPEG: diff=0
        let rust_reenc_pixels: &[u8] = &rust_reenc_image.data;
        assert_eq!(
            rust_reenc_pixels.len(),
            reenc_c_pixels.len(),
            "re-encoded pixel buffer size mismatch for {}",
            label
        );

        let max_diff_reenc: u8 = rust_reenc_pixels
            .iter()
            .zip(reenc_c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff_reenc, 0,
            "Rust vs C djpeg decode of re-encoded JPEG: diff={} for {} (expected 0)",
            max_diff_reenc, label
        );

        // Verify C djpeg can decode both original and re-encoded successfully
        // (dimensions match, pixel counts match — quality may differ due to
        // re-encoding, so we do not compare original vs re-encoded pixels)
        assert_eq!(orig_c_pixels.len(), reenc_c_pixels.len());
    }
}
