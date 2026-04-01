use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, decompress_12bit, decompress_16bit,
};
use libjpeg_turbo_rs::quantize::{dequantize, quantize, DitherMode, QuantizeOptions};
use libjpeg_turbo_rs::{
    compress, compress_into, decompress, read_scanlines_12, read_scanlines_16, requantize,
    write_scanlines_12, write_scanlines_16, Image, PixelFormat, ScanlineDecoder, Subsampling,
};

/// 12-bit scanline write/read roundtrip: encode via write_scanlines_12, then
/// decode via read_scanlines_12, and verify recovered samples are close to the
/// originals (DCT is lossy, so we allow small tolerance).
#[test]
fn scanline_12bit_roundtrip() {
    let width: usize = 16;
    let height: usize = 8;
    let num_components: usize = 1;
    // Build 12-bit sample rows (0-4095 range)
    let rows: Vec<Vec<i16>> = (0..height)
        .map(|y| {
            (0..width)
                .map(|x| ((y * width + x) * 256 % 4096) as i16)
                .collect()
        })
        .collect();

    let row_refs: Vec<&[i16]> = rows.iter().map(|r| r.as_slice()).collect();

    let jpeg_data: Vec<u8> = write_scanlines_12(
        &row_refs,
        width,
        height,
        num_components,
        95,
        Subsampling::S444,
    )
    .expect("write_scanlines_12 should succeed");

    // Should start with SOI marker
    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]);

    let decoded_rows: Vec<Vec<i16>> =
        read_scanlines_12(&jpeg_data, height).expect("read_scanlines_12 should succeed");

    assert_eq!(decoded_rows.len(), height);
    for (y, (original, decoded)) in rows.iter().zip(decoded_rows.iter()).enumerate() {
        assert_eq!(original.len(), decoded.len(), "row {} length mismatch", y);
        for (x, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            let diff: i16 = (orig - dec).abs();
            assert!(
                diff < 200,
                "12-bit sample at ({},{}) differs too much: orig={}, decoded={}, diff={}",
                x,
                y,
                orig,
                dec,
                diff
            );
        }
    }
}

/// 16-bit scanline write/read roundtrip. 16-bit is lossless, so we expect exact
/// recovery.
#[test]
fn scanline_16bit_roundtrip() {
    let width: usize = 8;
    let height: usize = 4;
    let num_components: usize = 1;
    let rows: Vec<Vec<u16>> = (0..height)
        .map(|y| {
            (0..width)
                .map(|x| ((y * width + x) * 1000 % 65536) as u16)
                .collect()
        })
        .collect();

    let row_refs: Vec<&[u16]> = rows.iter().map(|r| r.as_slice()).collect();

    let jpeg_data: Vec<u8> = write_scanlines_16(&row_refs, width, height, num_components, 1, 0)
        .expect("write_scanlines_16 should succeed");

    assert_eq!(&jpeg_data[0..2], &[0xFF, 0xD8]);

    let decoded_rows: Vec<Vec<u16>> =
        read_scanlines_16(&jpeg_data, height).expect("read_scanlines_16 should succeed");

    assert_eq!(decoded_rows.len(), height);
    for (y, (original, decoded)) in rows.iter().zip(decoded_rows.iter()).enumerate() {
        assert_eq!(original.len(), decoded.len(), "row {} length mismatch", y);
        for (x, (&orig, &dec)) in original.iter().zip(decoded.iter()).enumerate() {
            assert_eq!(
                orig, dec,
                "16-bit sample at ({},{}) differs: orig={}, decoded={}",
                x, y, orig, dec
            );
        }
    }
}

/// Bottom-up decode: when enabled, the output rows should be the reverse of
/// normal top-to-bottom decode order.
#[test]
fn bottom_up_decode_produces_flipped_rows() {
    let width: usize = 8;
    let height: usize = 8;
    // Create a gradient image: row 0 is dark, row 7 is bright
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let val: u8 = (y * 32).min(255) as u8;
        for _x in 0..width {
            pixels.extend_from_slice(&[val, val, val]);
        }
    }

    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .expect("compress should succeed");

    // Normal decode
    let normal = decompress(&jpeg_data).expect("decompress should succeed");

    // Bottom-up decode
    let mut decoder = ScanlineDecoder::new(&jpeg_data).expect("decoder should succeed");
    decoder.set_bottom_up(true);
    let flipped = decoder.finish().expect("finish should succeed");

    assert_eq!(normal.width, flipped.width);
    assert_eq!(normal.height, flipped.height);

    let bpp: usize = normal.pixel_format.bytes_per_pixel();
    let row_bytes: usize = normal.width * bpp;

    // Verify row order is reversed
    for y in 0..height {
        let normal_row: &[u8] = &normal.data[y * row_bytes..(y + 1) * row_bytes];
        let flipped_row: &[u8] =
            &flipped.data[(height - 1 - y) * row_bytes..(height - y) * row_bytes];
        assert_eq!(
            normal_row,
            flipped_row,
            "row {} of normal should match row {} of flipped",
            y,
            height - 1 - y
        );
    }
}

/// compress_into with a sufficient buffer should succeed and return byte count.
#[test]
fn compress_into_sufficient_buffer() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = vec![128u8; width * height * 3];

    // Allocate a generous buffer
    let mut buf: Vec<u8> = vec![0u8; width * height * 3 + 4096];

    let written: usize = compress_into(
        &mut buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("compress_into should succeed");

    assert!(written > 0, "should write some bytes");
    assert!(written <= buf.len(), "should not exceed buffer");
    // Verify it starts with SOI marker
    assert_eq!(&buf[0..2], &[0xFF, 0xD8]);
    // Verify it ends with EOI marker
    assert_eq!(&buf[written - 2..written], &[0xFF, 0xD9]);
}

/// compress_into with an insufficient buffer should return an error.
#[test]
fn compress_into_insufficient_buffer() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = vec![128u8; width * height * 3];

    // Allocate a tiny buffer that cannot hold the JPEG
    let mut buf: Vec<u8> = vec![0u8; 10];

    let result = compress_into(
        &mut buf,
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    );

    assert!(
        result.is_err(),
        "compress_into should fail with tiny buffer"
    );
}

/// requantize: take an already-quantized image and re-quantize it with a
/// different palette. The resulting image should use only colors from the new
/// palette.
#[test]
fn requantize_with_different_palette() {
    // Create a simple 4x4 RGB image with known colors
    let width: usize = 4;
    let height: usize = 4;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for _ in 0..width * height {
        pixels.extend_from_slice(&[200, 50, 50]); // reddish
    }

    // First quantize to a 4-color palette
    let opts = QuantizeOptions {
        num_colors: 4,
        dither_mode: DitherMode::None,
        two_pass: true,
        colormap: None,
    };
    let quantized = quantize(&pixels, width, height, &opts).expect("quantize should succeed");

    // Now re-quantize with a completely different palette
    let new_palette: Vec<[u8; 3]> = vec![[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]];
    let requ = requantize(&quantized, &new_palette, DitherMode::None);

    assert_eq!(requ.width, width);
    assert_eq!(requ.height, height);
    assert_eq!(requ.indices.len(), width * height);
    assert_eq!(requ.palette, new_palette);

    // All indices should be valid indices into the new palette
    for &idx in &requ.indices {
        assert!(
            (idx as usize) < new_palette.len(),
            "index {} out of range",
            idx
        );
    }

    // Since the original was reddish, the nearest color in the new palette
    // should be [255, 0, 0] (index 1)
    for &idx in &requ.indices {
        assert_eq!(idx, 1, "reddish pixel should map to red in new palette");
    }
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

// ---------------------------------------------------------------------------
// C djpeg cross-validation for bottom-up decode
// ---------------------------------------------------------------------------

/// Verify that bottom-up decode produces row-reversed output that, when
/// flipped back, is pixel-identical to C libjpeg-turbo's djpeg output.
///
/// Steps:
/// 1. Create a JPEG from known gradient pixel data (4:4:4, quality 100).
/// 2. Decode normally with Rust and compare to djpeg (diff=0).
/// 3. Decode bottom-up with Rust, verify rows are reversed vs normal.
/// 4. Confirm bottom-up pixels, when row-reversed, match djpeg exactly.
#[test]
fn c_djpeg_bottom_up_decode_matches() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found, skipping C cross-validation");
            return;
        }
    };

    // Build a 32x24 RGB gradient image with distinct per-row content
    let width: usize = 32;
    let height: usize = 24;
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r: u8 = ((y * 11 + x * 7) % 256) as u8;
            let g: u8 = ((y * 5 + x * 3 + 80) % 256) as u8;
            let b: u8 = ((y * 3 + x * 11 + 160) % 256) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }

    // Use quality 100 and 4:4:4 to minimize compression artifacts
    let jpeg_data: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        100,
        Subsampling::S444,
    )
    .expect("compress should succeed");

    // --- C djpeg decode ---
    let temp_dir: PathBuf = std::env::temp_dir();
    let jpeg_path: PathBuf = temp_dir.join("bottom_up_xval.jpg");
    let ppm_path: PathBuf = temp_dir.join("bottom_up_xval.ppm");

    {
        let mut file: std::fs::File = std::fs::File::create(&jpeg_path)
            .unwrap_or_else(|e| panic!("Failed to create temp JPEG {:?}: {:?}", jpeg_path, e));
        file.write_all(&jpeg_data)
            .unwrap_or_else(|e| panic!("Failed to write temp JPEG {:?}: {:?}", jpeg_path, e));
    }

    let djpeg_output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&ppm_path)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("Failed to run djpeg: {:?}", e));

    assert!(
        djpeg_output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&djpeg_output.stderr)
    );

    let ppm_data: Vec<u8> = std::fs::read(&ppm_path)
        .unwrap_or_else(|e| panic!("Failed to read PPM {:?}: {:?}", ppm_path, e));
    let (c_width, c_height, c_pixels) =
        parse_ppm(&ppm_data).expect("Failed to parse PPM output from djpeg");

    let _ = std::fs::remove_file(&jpeg_path);
    let _ = std::fs::remove_file(&ppm_path);

    assert_eq!(c_width, width, "djpeg width mismatch");
    assert_eq!(c_height, height, "djpeg height mismatch");

    // --- Rust normal decode ---
    let normal: Image = decompress(&jpeg_data).expect("Rust decompress should succeed");
    assert_eq!(normal.width, width);
    assert_eq!(normal.height, height);

    let bpp: usize = normal.pixel_format.bytes_per_pixel();
    let row_bytes: usize = width * bpp;

    // Step 2: Verify Rust normal decode matches djpeg (diff=0)
    {
        let mut max_diff: u8 = 0;
        let mut mismatches: usize = 0;
        for (i, (&ours, &theirs)) in normal.data.iter().zip(c_pixels.iter()).enumerate() {
            let diff: u8 = (ours as i16 - theirs as i16).unsigned_abs() as u8;
            if diff > 0 {
                mismatches += 1;
                if mismatches <= 5 {
                    let pixel: usize = i / 3;
                    let channel: &str = ["R", "G", "B"][i % 3];
                    eprintln!(
                        "  normal pixel {} channel {}: rust={} c={} diff={}",
                        pixel, channel, ours, theirs, diff
                    );
                }
            }
            if diff > max_diff {
                max_diff = diff;
            }
        }
        assert_eq!(
            mismatches, 0,
            "Rust normal decode vs djpeg: {} pixels differ (max diff={}), expected diff=0",
            mismatches, max_diff
        );
    }

    // --- Rust bottom-up decode ---
    let mut decoder: ScanlineDecoder =
        ScanlineDecoder::new(&jpeg_data).expect("ScanlineDecoder should succeed");
    decoder.set_bottom_up(true);
    let flipped: Image = decoder.finish().expect("bottom-up decode should succeed");

    assert_eq!(flipped.width, width);
    assert_eq!(flipped.height, height);
    assert_eq!(flipped.data.len(), normal.data.len());

    // Step 3: Verify bottom-up rows are reversed compared to normal decode
    for y in 0..height {
        let normal_row: &[u8] = &normal.data[y * row_bytes..(y + 1) * row_bytes];
        let flipped_row: &[u8] =
            &flipped.data[(height - 1 - y) * row_bytes..(height - y) * row_bytes];
        assert_eq!(
            normal_row,
            flipped_row,
            "normal row {} should match flipped row {} (row reversal check)",
            y,
            height - 1 - y
        );
    }

    // Step 4: Bottom-up pixels, when row-reversed, must match djpeg exactly (diff=0)
    {
        let mut max_diff: u8 = 0;
        let mut mismatches: usize = 0;
        for y in 0..height {
            // Row y from djpeg/normal corresponds to row (height-1-y) in bottom-up output
            let flipped_row_start: usize = (height - 1 - y) * row_bytes;
            let c_row_start: usize = y * row_bytes;
            for x_byte in 0..row_bytes {
                let flipped_val: u8 = flipped.data[flipped_row_start + x_byte];
                let c_val: u8 = c_pixels[c_row_start + x_byte];
                let diff: u8 = (flipped_val as i16 - c_val as i16).unsigned_abs() as u8;
                if diff > 0 {
                    mismatches += 1;
                    if mismatches <= 5 {
                        let pixel: usize = (y * width) + x_byte / 3;
                        let channel: &str = ["R", "G", "B"][x_byte % 3];
                        eprintln!(
                            "  bottom-up (reversed) pixel {} channel {}: rust={} c={} diff={}",
                            pixel, channel, flipped_val, c_val, diff
                        );
                    }
                }
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "bottom-up (row-reversed) vs djpeg: {} pixels differ (max diff={}), expected diff=0",
            mismatches, max_diff
        );
    }
}
