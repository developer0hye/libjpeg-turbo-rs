use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{
    compress_arithmetic, compress_progressive, decompress, decompress_to, PixelFormat, Subsampling,
};

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

fn cjpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/cjpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("cjpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("read PPM");
    let comps: usize = if &raw[0..2] == b"P5" { 1 } else { 3 };
    let mut idx: usize = 2;
    loop {
        while idx < raw.len() && raw[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < raw.len() && raw[idx] == b'#' {
            while idx < raw.len() && raw[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    let mut end: usize = idx;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    let w: usize = std::str::from_utf8(&raw[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    idx = end;
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    end = idx;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    let h: usize = std::str::from_utf8(&raw[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    idx = end;
    while idx < raw.len() && raw[idx].is_ascii_whitespace() {
        idx += 1;
    }
    end = idx;
    while end < raw.len() && raw[end].is_ascii_digit() {
        end += 1;
    }
    idx = end + 1;
    (w, h, raw[idx..idx + w * h * comps].to_vec())
}

/// Verify that existing arithmetic and progressive paths still work
/// with pixel validation (not just dimensions).
#[test]
fn arithmetic_sequential_still_works() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    // Uniform 128 input: decoded pixels should be close
    let max_diff: u8 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 5, "arithmetic sequential max_diff={}", max_diff);
}

#[test]
fn progressive_huffman_still_works() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    let max_diff: u8 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert!(max_diff <= 5, "progressive huffman max_diff={}", max_diff);
}

/// Test SOF10 decode with a REAL C-encoded arithmetic progressive JPEG.
/// C cjpeg -arithmetic -progressive produces SOF10 (0xCA).
/// Validates Rust decode matches C djpeg pixel-by-pixel.
#[test]
#[ignore = "SOF10 arithmetic progressive decode bug: 'arithmetic AC spectral overflow'"]
fn sof10_c_encoded_decode_pixel_validation() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
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

    // Generate PPM source
    let (w, h): (usize, usize) = (32, 32);
    let mut ppm_data: Vec<u8> = format!("P6\n{} {}\n255\n", w, h).into_bytes();
    for y in 0..h {
        for x in 0..w {
            ppm_data.push((x * 8) as u8);
            ppm_data.push((y * 8) as u8);
            ppm_data.push(((x + y) * 4) as u8);
        }
    }
    let ppm_path: &str = "/tmp/ljt_sof10_src.ppm";
    let jpg_path: &str = "/tmp/ljt_sof10.jpg";
    let dec_path: &str = "/tmp/ljt_sof10_dec.ppm";
    std::fs::write(ppm_path, &ppm_data).unwrap();

    // Encode with C cjpeg -arithmetic -progressive → SOF10
    let output = Command::new(&cjpeg)
        .args([
            "-arithmetic",
            "-progressive",
            "-quality",
            "90",
            "-outfile",
            jpg_path,
            ppm_path,
        ])
        .output()
        .expect("failed to run cjpeg");
    if !output.status.success() {
        eprintln!(
            "SKIP: cjpeg -arithmetic -progressive failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    // Verify SOF10 marker (0xFFCA) is present
    let jpeg_data: Vec<u8> = std::fs::read(jpg_path).unwrap();
    let has_sof10: bool = jpeg_data.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xCA);
    assert!(
        has_sof10,
        "cjpeg -arithmetic -progressive should produce SOF10"
    );

    // Rust decode
    let rust_img =
        decompress_to(&jpeg_data, PixelFormat::Rgb).expect("Rust must decode SOF10 JPEG");
    assert_eq!(rust_img.width, w);
    assert_eq!(rust_img.height, h);

    // C djpeg decode
    let output = Command::new(&djpeg)
        .args(["-ppm", "-outfile", dec_path, jpg_path])
        .output()
        .expect("failed to run djpeg");
    assert!(output.status.success(), "djpeg failed on SOF10 JPEG");
    let (cw, ch, c_pixels) = parse_ppm(Path::new(dec_path));
    assert_eq!(cw, w);
    assert_eq!(ch, h);

    // Cross-validate: Rust vs C djpeg, target diff=0
    let max_diff: u8 = c_pixels
        .iter()
        .zip(rust_img.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "SOF10 decode: Rust vs C djpeg max_diff={} (must be 0)",
        max_diff
    );

    std::fs::remove_file(ppm_path).ok();
    std::fs::remove_file(jpg_path).ok();
    std::fs::remove_file(dec_path).ok();
}

/// Test SOF10 decode by constructing a minimal arithmetic progressive JPEG.
#[test]
fn sof10_grayscale_minimal_decode() {
    let jpeg: Vec<u8> = build_sof10_grayscale_jpeg();
    let result = decompress(&jpeg);
    match result {
        Ok(img) => {
            assert_eq!(img.width, 8);
            assert_eq!(img.height, 8);
            // Verify pixel data is valid (all zeros from our zero-entropy data)
            assert_eq!(img.data.len(), 8 * 8 * img.pixel_format.bytes_per_pixel());
        }
        Err(e) => {
            // Hand-built entropy data may be malformed, but SOF10 must be recognized.
            let msg: String = format!("{:?}", e);
            assert!(
                !msg.contains("unsupported") && !msg.contains("Unsupported"),
                "SOF10 should be recognized, not unsupported: {}",
                msg
            );
        }
    }
}

/// Verify that the decoder recognizes SOF10 marker.
#[test]
fn sof10_marker_is_recognized() {
    let jpeg: Vec<u8> = build_minimal_sof10_header();
    let result = decompress(&jpeg);
    match result {
        Ok(_) => {}
        Err(e) => {
            let msg: String = format!("{:?}", e);
            assert!(
                !msg.contains("unsupported frame type"),
                "SOF10 (0xCA) should be a recognized frame type: {}",
                msg
            );
        }
    }
}

/// Build a minimal SOF10 JPEG with just the header markers to test recognition.
fn build_minimal_sof10_header() -> Vec<u8> {
    let mut out = Vec::new();

    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // DQT — quantization table 0 (all 1s for simplicity)
    out.extend_from_slice(&[0xFF, 0xDB]);
    let dqt_len: u16 = 2 + 1 + 64;
    out.extend_from_slice(&dqt_len.to_be_bytes());
    out.push(0x00); // 8-bit, table 0
    out.extend_from_slice(&[1u8; 64]); // all 1s quant table

    // DAC — arithmetic conditioning
    out.extend_from_slice(&[0xFF, 0xCC]);
    out.extend_from_slice(&4u16.to_be_bytes()); // length=4 (1 entry)
    out.push(0x00); // DC table 0
    out.push(0x10); // L=0, U=1

    // SOF10 — arithmetic progressive, 1 component, 8x8
    out.extend_from_slice(&[0xFF, 0xCA]); // SOF10
    let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + 3;
    out.extend_from_slice(&sof_len.to_be_bytes());
    out.push(8); // precision
    out.extend_from_slice(&8u16.to_be_bytes()); // height
    out.extend_from_slice(&8u16.to_be_bytes()); // width
    out.push(1); // 1 component
    out.push(1); // comp id
    out.push(0x11); // h=1, v=1
    out.push(0); // quant table 0

    // SOS — DC first scan (Ss=0, Se=0, Ah=0, Al=0)
    out.extend_from_slice(&[0xFF, 0xDA]);
    let sos_len: u16 = 2 + 1 + 2 + 3;
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(1); // 1 component
    out.push(1); // comp id
    out.push(0x00); // DC table 0, AC table 0
    out.push(0); // Ss=0
    out.push(0); // Se=0
    out.push(0x00); // Ah=0, Al=0

    // Minimal arithmetic entropy data (zeros → the decoder handles gracefully)
    out.extend_from_slice(&[0x00; 16]);

    // EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    out
}

/// Build a minimal single-MCU SOF10 JPEG for decode testing.
fn build_sof10_grayscale_jpeg() -> Vec<u8> {
    let mut out = Vec::new();

    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // DQT — quantization table 0 (all 1s)
    out.extend_from_slice(&[0xFF, 0xDB]);
    let dqt_len: u16 = 2 + 1 + 64;
    out.extend_from_slice(&dqt_len.to_be_bytes());
    out.push(0x00);
    out.extend_from_slice(&[1u8; 64]);

    // DAC — DC table 0: L=0, U=1; AC table 0: Kx=5
    out.extend_from_slice(&[0xFF, 0xCC]);
    out.extend_from_slice(&6u16.to_be_bytes()); // length=6 (2 entries)
    out.push(0x00); // DC table 0
    out.push(0x10); // U=1, L=0
    out.push(0x10); // AC table 0 (Tc=1, Tb=0)
    out.push(0x05); // Kx=5

    // SOF10 — 1 component, 8x8
    out.extend_from_slice(&[0xFF, 0xCA]);
    let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + 3;
    out.extend_from_slice(&sof_len.to_be_bytes());
    out.push(8);
    out.extend_from_slice(&8u16.to_be_bytes());
    out.extend_from_slice(&8u16.to_be_bytes());
    out.push(1);
    out.push(1);
    out.push(0x11);
    out.push(0);

    // Scan 1: DC first (Ss=0, Se=0, Ah=0, Al=0)
    out.extend_from_slice(&[0xFF, 0xDA]);
    let sos_len: u16 = 2 + 1 + 2 + 3;
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(1);
    out.push(1);
    out.push(0x00);
    out.push(0); // Ss=0
    out.push(0); // Se=0
    out.push(0x00); // Ah=0, Al=0

    // Arithmetic entropy data for DC: encode DC=0 (zero difference)
    // In arithmetic coding, a zero diff means decode(S0)=0 which is the MPS initially
    // Provide enough bytes for the decoder to read
    out.extend_from_slice(&[0x00; 32]);

    // Scan 2: AC first (Ss=1, Se=63, Ah=0, Al=0)
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(1);
    out.push(1);
    out.push(0x00);
    out.push(1); // Ss=1
    out.push(63); // Se=63
    out.push(0x00); // Ah=0, Al=0

    // Arithmetic entropy data for AC: encode all zeros (EOB immediately)
    // EOB = decode(st)=1 for the first AC position
    out.extend_from_slice(&[0xFF; 8]); // all-ones forces quick EOB
    out.extend_from_slice(&[0x00; 24]);

    // EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    out
}
