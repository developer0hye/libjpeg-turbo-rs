use std::io::Cursor;

use libjpeg_turbo_rs::{compress, decompress, stream, Image, PixelFormat, Subsampling};

/// Generate a simple RGB test image: horizontal gradient.
fn make_test_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = (x * 255 / width.max(1)) as u8;
            let g: u8 = (y * 255 / height.max(1)) as u8;
            let b: u8 = 128;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

// -----------------------------------------------------------------------
// compress_to_writer
// -----------------------------------------------------------------------

#[test]
fn compress_to_writer_roundtrips_via_vec() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = make_test_pixels(width, height);

    let mut output: Vec<u8> = Vec::new();
    stream::compress_to_writer(
        &mut output,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .unwrap();

    // Output should be a valid JPEG (starts with SOI marker 0xFFD8)
    assert!(output.len() > 2);
    assert_eq!(output[0], 0xFF);
    assert_eq!(output[1], 0xD8);

    // Decompress and verify dimensions match
    let image: Image = decompress(&output).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
}

#[test]
fn compress_to_writer_different_sizes() {
    // Triangulation: test with different dimensions to prevent hardcoding
    for &(w, h) in &[(16, 16), (64, 48), (100, 75)] {
        let pixels: Vec<u8> = make_test_pixels(w, h);
        let mut output: Vec<u8> = Vec::new();
        stream::compress_to_writer(
            &mut output,
            &pixels,
            w,
            h,
            PixelFormat::Rgb,
            85,
            Subsampling::S420,
        )
        .unwrap();

        let image: Image = decompress(&output).unwrap();
        assert_eq!(image.width, w, "width mismatch for {w}x{h}");
        assert_eq!(image.height, h, "height mismatch for {w}x{h}");
    }
}

// -----------------------------------------------------------------------
// decompress_from_reader
// -----------------------------------------------------------------------

#[test]
fn decompress_from_reader_with_cursor() {
    let width: usize = 24;
    let height: usize = 24;
    let pixels: Vec<u8> = make_test_pixels(width, height);

    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        95,
        Subsampling::S444,
    )
    .unwrap();

    let mut cursor: Cursor<Vec<u8>> = Cursor::new(jpeg_data);
    let image: Image = stream::decompress_from_reader(&mut cursor).unwrap();

    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.pixel_format, PixelFormat::Rgb);
}

#[test]
fn decompress_from_reader_preserves_pixel_data() {
    let width: usize = 8;
    let height: usize = 8;
    // Use a flat color so lossy compression does not change values much
    let pixels: Vec<u8> = vec![128; width * height * 3];

    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .unwrap();

    let mut cursor: Cursor<Vec<u8>> = Cursor::new(jpeg_data);
    let image: Image = stream::decompress_from_reader(&mut cursor).unwrap();

    // At quality 100 with 4:4:4, pixel values should be very close
    for (orig, decoded) in pixels.iter().zip(image.data.iter()) {
        assert!(
            (*orig as i16 - *decoded as i16).unsigned_abs() <= 2,
            "pixel deviation too large: orig={orig} decoded={decoded}"
        );
    }
}

#[test]
fn decompress_from_reader_error_on_empty() {
    let mut cursor: Cursor<Vec<u8>> = Cursor::new(Vec::new());
    let result = stream::decompress_from_reader(&mut cursor);
    assert!(result.is_err(), "empty reader should produce an error");
}

#[test]
fn decompress_from_reader_error_on_invalid_data() {
    let garbage: Vec<u8> = vec![0x00, 0x01, 0x02, 0x03, 0xFF, 0x00];
    let mut cursor: Cursor<Vec<u8>> = Cursor::new(garbage);
    let result = stream::decompress_from_reader(&mut cursor);
    assert!(result.is_err(), "invalid JPEG data should produce an error");
}

// -----------------------------------------------------------------------
// compress_to_file / decompress_from_file
// -----------------------------------------------------------------------

#[test]
fn file_roundtrip() {
    let width: usize = 40;
    let height: usize = 30;
    let pixels: Vec<u8> = make_test_pixels(width, height);

    let dir: std::path::PathBuf = std::env::temp_dir();
    let path: std::path::PathBuf = dir.join("libjpeg_turbo_rs_stream_io_test_roundtrip.jpg");

    // Compress to file
    stream::compress_to_file(
        &path,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S422,
    )
    .unwrap();

    // File should exist and be non-empty
    let metadata: std::fs::Metadata = std::fs::metadata(&path).unwrap();
    assert!(metadata.len() > 0, "output file should be non-empty");

    // Decompress from file
    let image: Image = stream::decompress_from_file(&path).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.pixel_format, PixelFormat::Rgb);

    // Clean up
    let _ = std::fs::remove_file(&path);
}

#[test]
fn decompress_from_file_nonexistent_path() {
    let path: std::path::PathBuf = std::env::temp_dir().join("nonexistent_jpeg_file_12345.jpg");
    let result = stream::decompress_from_file(&path);
    assert!(result.is_err(), "nonexistent file should produce an error");
}

#[test]
fn compress_to_file_invalid_directory() {
    let path = std::path::Path::new("/nonexistent_dir_abc123/test.jpg");
    let pixels: Vec<u8> = make_test_pixels(8, 8);
    let result =
        stream::compress_to_file(path, &pixels, 8, 8, PixelFormat::Rgb, 90, Subsampling::S444);
    assert!(result.is_err(), "invalid directory should produce an error");
}

// -----------------------------------------------------------------------
// Writer that tracks writes (verifies streaming behavior)
// -----------------------------------------------------------------------

#[test]
fn compress_to_writer_uses_write_trait() {
    // Verify the function works with an arbitrary Write implementor, not just Vec<u8>
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = make_test_pixels(width, height);

    let buffer: Vec<u8> = Vec::new();
    let mut cursor: Cursor<Vec<u8>> = Cursor::new(buffer);
    stream::compress_to_writer(
        &mut cursor,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        80,
        Subsampling::S420,
    )
    .unwrap();

    let jpeg_bytes: Vec<u8> = cursor.into_inner();
    assert!(jpeg_bytes.len() > 2);
    assert_eq!(jpeg_bytes[0], 0xFF);
    assert_eq!(jpeg_bytes[1], 0xD8);
}

// -----------------------------------------------------------------------
// C djpeg cross-validation
// -----------------------------------------------------------------------

/// Path to C djpeg binary, or `None` if not installed.
fn djpeg_path() -> Option<std::path::PathBuf> {
    let homebrew: std::path::PathBuf = std::path::PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    std::process::Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| std::path::PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Parse a binary PPM (P6) file and return `(width, height, pixel_data)`.
fn parse_ppm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PPM too short");
    assert_eq!(&data[0..2], b"P6", "not a P6 PPM");
    let mut idx: usize = 2;
    // skip whitespace/comments
    idx = ppm_skip_ws(data, idx);
    let (width, next) = ppm_read_num(data, idx);
    idx = ppm_skip_ws(data, next);
    let (height, next) = ppm_read_num(data, idx);
    idx = ppm_skip_ws(data, next);
    let (_maxval, next) = ppm_read_num(data, idx);
    idx = next + 1; // single whitespace byte before pixel data
    let pixels: Vec<u8> = data[idx..].to_vec();
    assert_eq!(
        pixels.len(),
        width * height * 3,
        "PPM pixel data length mismatch"
    );
    (width, height, pixels)
}

fn ppm_skip_ws(data: &[u8], mut idx: usize) -> usize {
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

fn ppm_read_num(data: &[u8], idx: usize) -> (usize, usize) {
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

/// RAII temp file that deletes on drop.
struct TempFile {
    path: std::path::PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id: u64 = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            path: std::env::temp_dir().join(format!(
                "ljt_stream_io_{}_{}_{name}",
                std::process::id(),
                id
            )),
        }
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

#[test]
fn c_djpeg_cross_validation_stream_io() {
    let djpeg = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let width: usize = 48;
    let height: usize = 48;
    // Generate RGB gradient source pixels
    let mut source_pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((x * 255) / width.max(1)) as u8;
            let g: u8 = ((y * 255) / height.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (width + height).max(1)) as u8;
            source_pixels.push(r);
            source_pixels.push(g);
            source_pixels.push(b);
        }
    }

    // Encode using compress_to_writer
    let mut jpeg_buf: Vec<u8> = Vec::new();
    stream::compress_to_writer(
        &mut jpeg_buf,
        &source_pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .expect("compress_to_writer failed");

    // Write JPEG to temp file for djpeg
    let tmp_jpg = TempFile::new("stream_xval.jpg");
    let tmp_ppm = TempFile::new("stream_xval.ppm");
    std::fs::write(&tmp_jpg.path, &jpeg_buf).expect("write temp JPEG");

    // Decode with C djpeg
    let output = std::process::Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm.path)
        .arg(&tmp_jpg.path)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(&tmp_ppm.path).expect("read PPM output");
    let (c_w, c_h, c_pixels) = parse_ppm(&ppm_data);
    assert_eq!(c_w, width, "C djpeg width mismatch");
    assert_eq!(c_h, height, "C djpeg height mismatch");

    // Decode with Rust
    let rust_image: Image = decompress(&jpeg_buf).expect("Rust decompress failed");
    assert_eq!(rust_image.width, width);
    assert_eq!(rust_image.height, height);

    // Compare pixel-by-pixel: diff must be 0
    assert_eq!(
        rust_image.data.len(),
        c_pixels.len(),
        "pixel data length mismatch"
    );
    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_image.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > max_diff {
            max_diff = diff;
        }
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                let pixel: usize = i / 3;
                let channel: &str = ["R", "G", "B"][i % 3];
                eprintln!(
                    "  pixel {} channel {}: rust={} c={} diff={}",
                    pixel, channel, r, c, diff
                );
            }
        }
    }
    assert_eq!(
        max_diff, 0,
        "stream_io cross-validation: {} pixels differ, max_diff={}",
        mismatch_count, max_diff
    );
}

#[test]
fn file_roundtrip_grayscale() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = vec![200; width * height];

    let dir: std::path::PathBuf = std::env::temp_dir();
    let path: std::path::PathBuf = dir.join("libjpeg_turbo_rs_stream_io_test_gray.jpg");

    stream::compress_to_file(
        &path,
        &pixels,
        width,
        height,
        PixelFormat::Grayscale,
        95,
        Subsampling::S444,
    )
    .unwrap();

    let image: Image = stream::decompress_from_file(&path).unwrap();
    assert_eq!(image.width, width);
    assert_eq!(image.height, height);
    assert_eq!(image.pixel_format, PixelFormat::Grayscale);

    // Clean up
    let _ = std::fs::remove_file(&path);
}
