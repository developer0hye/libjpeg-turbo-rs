use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{compress_lossless, decompress, PixelFormat};

/// Build a minimal SOF3 (lossless) JPEG in memory for testing.
/// Creates a tiny 4x2 grayscale image with predictor 1 (left), no point transform.
fn make_lossless_jpeg(pixels: &[u8], width: u16, height: u16, precision: u8) -> Vec<u8> {
    let mut out = Vec::new();

    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // DQT marker (required even for lossless, though not used; write dummy)
    // Actually SOF3 doesn't use quant tables, so we skip it.

    // DHT — DC Huffman table 0
    // Encode differences using a simple variable-length code.
    // We use the standard DC luminance table for simplicity.
    let dc_bits: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    let dc_values: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    out.extend_from_slice(&[0xFF, 0xC4]); // DHT marker
    let dht_len: u16 = 2 + 1 + 16 + dc_values.len() as u16;
    out.extend_from_slice(&dht_len.to_be_bytes());
    out.push(0x00); // DC table 0
    out.extend_from_slice(&dc_bits[1..]);
    out.extend_from_slice(dc_values);

    // SOF3 — Lossless, Huffman-coded
    out.extend_from_slice(&[0xFF, 0xC3]);
    let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + 3; // 1 component
    out.extend_from_slice(&sof_len.to_be_bytes());
    out.push(precision); // sample precision
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(1); // 1 component
    out.push(1); // component id
    out.push(0x11); // h=1, v=1
    out.push(0); // quant table 0 (unused for lossless)

    // SOS — Start of Scan
    out.extend_from_slice(&[0xFF, 0xDA]);
    let sos_len: u16 = 2 + 1 + 2 + 3; // 1 component + 3 params
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(1); // 1 component
    out.push(1); // component id
    out.push(0x00); // DC table 0, AC table 0
    out.push(1); // Ss = predictor selection value (1 = left)
    out.push(0); // Se = 0
    out.push(0x00); // Ah=0, Al=0 (point transform = 0)

    // Entropy-coded data: encode differences using DC Huffman coding
    let mut bit_buf: u32 = 0;
    let mut bit_count: u32 = 0;

    // Build the Huffman encoding table from the same bits/values
    let huff_codes = build_dc_encode_table(&dc_bits, dc_values);

    let initial_pred = 1u32 << (precision as u32 - 1); // 128 for 8-bit
    let mask = (1u32 << precision) - 1;

    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = pixels[y * width as usize + x] as i32;
            let prediction = if y == 0 && x == 0 {
                initial_pred as i32
            } else if y == 0 {
                pixels[y * width as usize + x - 1] as i32
            } else if x == 0 {
                pixels[(y - 1) * width as usize + x] as i32
            } else {
                // predictor 1 = left
                pixels[y * width as usize + x - 1] as i32
            };

            let diff = ((pixel - prediction) as i32) & (mask as i32);
            // Encode diff as signed: if >= 2^(p-1), it's negative
            let signed_diff = if diff >= (1 << (precision - 1)) {
                diff - (1 << precision)
            } else {
                diff
            };

            // Determine category and encode
            let (category, extra_bits, extra_len) = categorize_dc(signed_diff);
            let (code, code_len) = huff_codes[category as usize];

            bit_buf = (bit_buf << code_len) | code as u32;
            bit_count += code_len as u32;
            if extra_len > 0 {
                bit_buf = (bit_buf << extra_len) | extra_bits as u32;
                bit_count += extra_len as u32;
            }

            // Flush complete bytes
            while bit_count >= 8 {
                bit_count -= 8;
                let byte = (bit_buf >> bit_count) as u8;
                out.push(byte);
                if byte == 0xFF {
                    out.push(0x00); // byte stuffing
                }
                bit_buf &= (1 << bit_count) - 1;
            }
        }
    }

    // Pad remaining bits with 1s
    if bit_count > 0 {
        let padding = 8 - bit_count;
        bit_buf = (bit_buf << padding) | ((1 << padding) - 1);
        let byte = bit_buf as u8;
        out.push(byte);
        if byte == 0xFF {
            out.push(0x00);
        }
    }

    // EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    out
}

/// Build a 3-component SOF3 (lossless) JPEG for testing.
/// Interleaved scan: components decoded round-robin per pixel.
fn make_lossless_jpeg_3comp(
    y_data: &[u8],
    cb_data: &[u8],
    cr_data: &[u8],
    width: u16,
    height: u16,
    precision: u8,
) -> Vec<u8> {
    let mut out = Vec::new();

    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // DHT — DC Huffman table 0
    let dc_bits: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    let dc_values: &[u8] = &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    out.extend_from_slice(&[0xFF, 0xC4]);
    let dht_len: u16 = 2 + 1 + 16 + dc_values.len() as u16;
    out.extend_from_slice(&dht_len.to_be_bytes());
    out.push(0x00); // DC table 0
    out.extend_from_slice(&dc_bits[1..]);
    out.extend_from_slice(dc_values);

    // SOF3 — Lossless, Huffman-coded, 3 components
    out.extend_from_slice(&[0xFF, 0xC3]);
    let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + 3 * 3; // 3 components
    out.extend_from_slice(&sof_len.to_be_bytes());
    out.push(precision);
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(3); // 3 components
                 // Y: id=1, h=1, v=1, qt=0
    out.push(1);
    out.push(0x11);
    out.push(0);
    // Cb: id=2, h=1, v=1, qt=0
    out.push(2);
    out.push(0x11);
    out.push(0);
    // Cr: id=3, h=1, v=1, qt=0
    out.push(3);
    out.push(0x11);
    out.push(0);

    // SOS — 3 components, interleaved, predictor 1
    out.extend_from_slice(&[0xFF, 0xDA]);
    let sos_len: u16 = 2 + 1 + 3 * 2 + 3;
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(3); // 3 components
    out.push(1);
    out.push(0x00); // Y: DC table 0
    out.push(2);
    out.push(0x00); // Cb: DC table 0
    out.push(3);
    out.push(0x00); // Cr: DC table 0
    out.push(1); // Ss = predictor 1 (left)
    out.push(0); // Se = 0
    out.push(0x00); // Ah=0, Al=0

    // Entropy-coded data: interleaved — for each pixel, encode Y diff, Cb diff, Cr diff
    let huff_codes = build_dc_encode_table(&dc_bits, dc_values);
    let initial_pred = 1u32 << (precision as u32 - 1);
    let mask = (1u32 << precision) - 1;

    let mut bit_buf: u32 = 0;
    let mut bit_count: u32 = 0;

    let planes: [&[u8]; 3] = [y_data, cb_data, cr_data];
    let mut prev_rows: [Option<Vec<u16>>; 3] = [None, None, None];
    let mut cur_rows: [Vec<u16>; 3] = [
        vec![0u16; width as usize],
        vec![0u16; width as usize],
        vec![0u16; width as usize],
    ];

    for y in 0..height as usize {
        for x in 0..width as usize {
            for c in 0..3 {
                let pixel = planes[c][y * width as usize + x] as i32;
                let prediction = if y == 0 && x == 0 {
                    initial_pred as i32
                } else if y == 0 {
                    cur_rows[c][x - 1] as i32
                } else if x == 0 {
                    prev_rows[c].as_ref().unwrap()[x] as i32
                } else {
                    cur_rows[c][x - 1] as i32
                };

                cur_rows[c][x] = pixel as u16;

                let diff = ((pixel - prediction) as i32) & (mask as i32);
                let signed_diff = if diff >= (1 << (precision - 1)) {
                    diff - (1 << precision)
                } else {
                    diff
                };

                let (category, extra_bits, extra_len) = categorize_dc(signed_diff);
                let (code, code_len) = huff_codes[category as usize];

                bit_buf = (bit_buf << code_len) | code as u32;
                bit_count += code_len as u32;
                if extra_len > 0 {
                    bit_buf = (bit_buf << extra_len) | extra_bits as u32;
                    bit_count += extra_len as u32;
                }

                while bit_count >= 8 {
                    bit_count -= 8;
                    let byte = (bit_buf >> bit_count) as u8;
                    out.push(byte);
                    if byte == 0xFF {
                        out.push(0x00);
                    }
                    bit_buf &= (1 << bit_count) - 1;
                }
            }
        }
        for c in 0..3 {
            prev_rows[c] = Some(cur_rows[c].clone());
        }
    }

    if bit_count > 0 {
        let padding = 8 - bit_count;
        bit_buf = (bit_buf << padding) | ((1 << padding) - 1);
        let byte = bit_buf as u8;
        out.push(byte);
        if byte == 0xFF {
            out.push(0x00);
        }
    }

    out.extend_from_slice(&[0xFF, 0xD9]);
    out
}

fn build_dc_encode_table(bits: &[u8; 17], values: &[u8]) -> Vec<(u16, u8)> {
    // Build codes from JPEG standard Huffman table specification
    let mut table = vec![(0u16, 0u8); 17]; // category 0..16
    let mut code: u16 = 0;
    let mut idx = 0;
    for length in 1..=16u8 {
        for _ in 0..bits[length as usize] {
            if idx < values.len() {
                table[values[idx] as usize] = (code, length);
                idx += 1;
            }
            code += 1;
        }
        code <<= 1;
    }
    table
}

fn categorize_dc(diff: i32) -> (u8, u16, u8) {
    if diff == 0 {
        return (0, 0, 0);
    }
    let abs_diff = diff.unsigned_abs();
    let category = 32 - abs_diff.leading_zeros() as u8;
    let extra = if diff > 0 {
        diff as u16
    } else {
        (diff + (1 << category) - 1) as u16
    };
    (category, extra, category)
}

#[test]
fn decode_lossless_grayscale_flat() {
    let pixels = vec![128u8; 4 * 2];
    let jpeg = make_lossless_jpeg(&pixels, 4, 2, 8);
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 4);
    assert_eq!(img.height, 2);
    assert_eq!(img.data.len(), 4 * 2);
    for &p in &img.data {
        assert_eq!(p, 128);
    }
}

#[test]
fn decode_lossless_grayscale_gradient() {
    let mut pixels = vec![0u8; 8 * 4];
    for y in 0..4 {
        for x in 0..8 {
            pixels[y * 8 + x] = (x * 30 + y * 10) as u8;
        }
    }
    let jpeg = make_lossless_jpeg(&pixels, 8, 4, 8);
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 4);
    assert_eq!(img.data, pixels);
}

#[test]
fn decode_lossless_grayscale_ramp() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = make_lossless_jpeg(&pixels, 16, 16, 8);
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(img.data, pixels);
}

#[test]
fn decode_lossless_3comp_flat() {
    let (w, h) = (4, 4);
    let y_data = vec![128u8; w * h];
    let cb_data = vec![100u8; w * h];
    let cr_data = vec![150u8; w * h];
    let jpeg = make_lossless_jpeg_3comp(&y_data, &cb_data, &cr_data, w as u16, h as u16, 8);
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
}

#[test]
fn decode_lossless_3comp_gradient() {
    let (w, h) = (8, 4);
    let y_data: Vec<u8> = (0..w * h).map(|i| (i * 3 % 256) as u8).collect();
    let cb_data: Vec<u8> = (0..w * h).map(|i| (128 + i % 128) as u8).collect();
    let cr_data: Vec<u8> = (0..w * h).map(|i| (64 + i * 2 % 192) as u8).collect();
    let jpeg = make_lossless_jpeg_3comp(&y_data, &cb_data, &cr_data, w as u16, h as u16, 8);
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
}

// ===========================================================================
// C djpeg cross-validation helpers
// ===========================================================================

/// Locate the `djpeg` binary, checking Homebrew first then PATH.
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

/// Check whether `djpeg` can handle SOF3 (lossless) by feeding it a lossless JPEG
/// and seeing if it exits successfully.
fn djpeg_supports_lossless(djpeg: &Path, lossless_jpeg: &[u8]) -> bool {
    let tmp_dir = std::env::temp_dir();
    let probe_path = tmp_dir.join(format!("ljt_lossless_probe_{}.jpg", std::process::id()));
    if std::fs::write(&probe_path, lossless_jpeg).is_err() {
        return false;
    }
    let result = Command::new(djpeg).arg("-pnm").arg(&probe_path).output();
    std::fs::remove_file(&probe_path).ok();
    match result {
        Ok(o) => o.status.success(),
        Err(_) => false,
    }
}

/// Parse a binary PGM (P5) returning `(width, height, pixel_data)`.
fn parse_pgm_data(raw: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(raw.len() > 3, "PGM too short");
    assert_eq!(&raw[0..2], b"P5", "not a P5 PGM");
    let mut idx: usize = 2;
    idx = skip_ws_comments(raw, idx);
    let (width, next) = read_number(raw, idx);
    idx = skip_ws_comments(raw, next);
    let (height, next) = read_number(raw, idx);
    idx = skip_ws_comments(raw, next);
    let (_maxval, next) = read_number(raw, idx);
    // Single whitespace byte separates header from data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(data.len(), width * height, "PGM pixel data length mismatch");
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

// ===========================================================================
// C djpeg cross-validation test
// ===========================================================================

#[test]
fn c_djpeg_lossless_decode_diff_zero() {
    // Step 1: Locate djpeg, skip if not found.
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found on this system");
            return;
        }
    };

    // Step 2: Encode a 16x16 grayscale lossless JPEG with Rust.
    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
    let jpeg: Vec<u8> =
        compress_lossless(&pixels, w, h, PixelFormat::Grayscale).expect("Rust lossless encode");

    // Step 3: Check if djpeg supports SOF3 lossless. Skip gracefully if not.
    if !djpeg_supports_lossless(&djpeg, &jpeg) {
        eprintln!("SKIP: djpeg does not support lossless JPEG (SOF3)");
        return;
    }

    // Step 4: Decode with Rust.
    let rust_img = decompress(&jpeg).expect("Rust lossless decode");
    assert_eq!(rust_img.width, w);
    assert_eq!(rust_img.height, h);

    // Step 5: Decode with C djpeg (output to PGM via stdout).
    let tmp_dir = std::env::temp_dir();
    let tmp_jpg = tmp_dir.join(format!("ljt_c_djpeg_lossless_{}.jpg", std::process::id()));
    std::fs::write(&tmp_jpg, &jpeg).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg(&tmp_jpg)
        .output()
        .expect("failed to run djpeg");

    std::fs::remove_file(&tmp_jpg).ok();

    assert!(
        output.status.success(),
        "djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Step 6: Parse PGM from djpeg stdout and compare.
    let (c_w, c_h, c_pixels) = parse_pgm_data(&output.stdout);
    assert_eq!(c_w, w, "C djpeg width mismatch");
    assert_eq!(c_h, h, "C djpeg height mismatch");

    // Step 7: Assert pixel-exact match between Rust and C djpeg.
    assert_eq!(
        rust_img.data.len(),
        c_pixels.len(),
        "Rust and C pixel buffer lengths differ: rust={} c={}",
        rust_img.data.len(),
        c_pixels.len()
    );

    let mut max_diff: u8 = 0;
    let mut mismatch_count: usize = 0;
    for (i, (&r, &c)) in rust_img.data.iter().zip(c_pixels.iter()).enumerate() {
        let diff: u8 = (r as i16 - c as i16).unsigned_abs() as u8;
        if diff > 0 {
            mismatch_count += 1;
            if mismatch_count <= 5 {
                eprintln!("  pixel {}: rust={} c={} diff={}", i, r, c, diff);
            }
        }
        if diff > max_diff {
            max_diff = diff;
        }
    }

    assert_eq!(
        max_diff, 0,
        "lossless decode must be pixel-exact: {} pixels differ, max diff={}",
        mismatch_count, max_diff
    );
}
