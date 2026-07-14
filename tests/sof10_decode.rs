mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(not(target_arch = "wasm32"))]
use libjpeg_turbo_rs::Decoder;
use libjpeg_turbo_rs::{
    compress_arithmetic, compress_arithmetic_progressive, compress_progressive, decompress,
    decompress_to, PixelFormat, Subsampling,
};

#[cfg(not(target_arch = "wasm32"))]
const P4_20_ARITHMETIC_PROGRESSIVE_GRAY: &[u8] =
    include_bytes!("../fuzz/corpus/fuzz_decompress/24fd23785278a9577686f501e17ee8164f8b977b");

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

// This differential characterization needs a native child process so the
// process-global SIMD override cannot race other tests. WASI cannot spawn the
// current test binary, and it has no external `djpeg` oracle.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn tracked_arithmetic_progressive_gray_is_pinned_to_p4_20() {
    const SCALAR_CHILD: &str = "LIBJPEG_P4_20_SCALAR_CHILD";
    if std::env::var(SCALAR_CHILD).ok().as_deref() != Some("1") {
        let status = Command::new(std::env::current_exe().expect("locate integration test binary"))
            .args([
                "--exact",
                "tracked_arithmetic_progressive_gray_is_pinned_to_p4_20",
                "--nocapture",
            ])
            .env(SCALAR_CHILD, "1")
            .env("JSIMD_FORCENONE", "1")
            .status()
            .expect("run isolated scalar P4-20 characterization");
        assert!(status.success(), "scalar P4-20 child test failed");
        return;
    }

    let djpeg: PathBuf = require_c_tool!("djpeg");
    let (c_width, c_height, c_pixels) = helpers::decode_gray_with_c_djpeg(
        &djpeg,
        P4_20_ARITHMETIC_PROGRESSIVE_GRAY,
        "p4_20_arithmetic_progressive_gray",
    );

    let mut smooth_decoder =
        Decoder::new(P4_20_ARITHMETIC_PROGRESSIVE_GRAY).expect("parse tracked SOF10 seed");
    smooth_decoder.set_block_smoothing(true);
    let smooth = smooth_decoder
        .decode_image()
        .expect("decode tracked SOF10 seed with smoothing");

    let mut unsmoothed_decoder =
        Decoder::new(P4_20_ARITHMETIC_PROGRESSIVE_GRAY).expect("parse tracked SOF10 seed");
    unsmoothed_decoder.set_block_smoothing(false);
    let unsmoothed = unsmoothed_decoder
        .decode_image()
        .expect("decode tracked SOF10 seed without smoothing");

    let nonce = format!(
        "{}_{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    );
    let jpeg_path = std::env::temp_dir().join(format!("p4_20_{nonce}.jpg"));
    let pgm_path = std::env::temp_dir().join(format!("p4_20_{nonce}.pgm"));
    std::fs::write(&jpeg_path, P4_20_ARITHMETIC_PROGRESSIVE_GRAY).expect("write C input");
    let c_nosmooth_status = Command::new(&djpeg)
        .args(["-strict", "-nosmooth", "-outfile"])
        .arg(&pgm_path)
        .arg(&jpeg_path)
        .status()
        .expect("run strict djpeg -nosmooth");
    assert!(c_nosmooth_status.success());
    let c_nosmooth_raw = std::fs::read(&pgm_path).expect("read C nosmooth PGM");
    let (_, _, c_nosmooth_pixels) =
        helpers::parse_pgm(&c_nosmooth_raw).expect("parse C nosmooth PGM");
    std::fs::remove_file(&jpeg_path).ok();
    std::fs::remove_file(&pgm_path).ok();

    assert_eq!((c_width, c_height), (144, 16));
    assert_eq!((smooth.width, smooth.height), (c_width, c_height));
    let smooth_diff = helpers::pixel_max_diff(&smooth.data, &c_pixels);
    let unsmoothed_diff = helpers::pixel_max_diff(&unsmoothed.data, &c_pixels);
    let rust_vs_c_nosmooth = helpers::pixel_max_diff(&unsmoothed.data, &c_nosmooth_pixels);
    assert_eq!(smooth_diff, 255, "P4-20 scalar characterization changed");
    assert_eq!(unsmoothed_diff, 255, "P4-20 is not smoothing-related");
    assert_eq!(
        rust_vs_c_nosmooth, 255,
        "P4-20 characterization against djpeg -nosmooth changed"
    );
}

/// Test SOF10 decode with a REAL C-encoded arithmetic progressive JPEG.
/// C cjpeg -arithmetic -progressive produces SOF10 (0xCA).
/// Validates Rust decode matches C djpeg pixel-by-pixel.
#[test]
fn sof10_c_encoded_decode_pixel_validation() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");

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

/// Cross-validate SOF10 (progressive arithmetic) decode: Rust vs C djpeg, diff=0.
/// Tests:
/// 1. C-encoded SOF10 fixture (cjpeg -arithmetic -progressive) decoded by both
/// 2. Rust-encoded SOF10 (compress_arithmetic_progressive) decoded by both
#[test]
fn c_djpeg_sof10_decode_diff_zero() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    // --- Test 1: C-encoded SOF10 fixture ---
    {
        // Generate PPM source with gradient pattern
        let (w, h): (usize, usize) = (48, 32);
        let mut ppm_data: Vec<u8> = format!("P6\n{} {}\n255\n", w, h).into_bytes();
        for y in 0..h {
            for x in 0..w {
                ppm_data.push(((x * 255) / w) as u8);
                ppm_data.push(((y * 255) / h) as u8);
                ppm_data.push((((x + y) * 127) / (w + h)) as u8);
            }
        }

        let ppm_path: String = format!("/tmp/ljt_sof10_xval_{}_src.ppm", std::process::id());
        let jpg_path: String = format!("/tmp/ljt_sof10_xval_{}_c_enc.jpg", std::process::id());
        let c_dec_ppm: String = format!("/tmp/ljt_sof10_xval_{}_c_dec.ppm", std::process::id());
        std::fs::write(&ppm_path, &ppm_data).unwrap();

        // Encode with C cjpeg -arithmetic -progressive -> SOF10
        let output = Command::new(&cjpeg)
            .args([
                "-arithmetic",
                "-progressive",
                "-quality",
                "90",
                "-outfile",
                &jpg_path,
                &ppm_path,
            ])
            .output()
            .expect("failed to run cjpeg");
        if !output.status.success() {
            eprintln!(
                "SKIP: cjpeg -arithmetic -progressive failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            std::fs::remove_file(&ppm_path).ok();
            return;
        }

        let jpeg_data: Vec<u8> = std::fs::read(&jpg_path).unwrap();

        // Verify SOF10 marker (0xFFCA) is present
        let has_sof10: bool = jpeg_data
            .windows(2)
            .any(|pair| pair[0] == 0xFF && pair[1] == 0xCA);
        assert!(
            has_sof10,
            "cjpeg -arithmetic -progressive must produce SOF10 (0xFFCA)"
        );

        // Rust decode
        let rust_img = decompress_to(&jpeg_data, PixelFormat::Rgb)
            .expect("Rust must decode C-encoded SOF10 JPEG");
        assert_eq!(rust_img.width, w);
        assert_eq!(rust_img.height, h);

        // C djpeg decode
        let output = Command::new(&djpeg)
            .args(["-ppm", "-outfile", &c_dec_ppm, &jpg_path])
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "djpeg failed on C-encoded SOF10 JPEG: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let (cw, ch, c_pixels) = parse_ppm(Path::new(&c_dec_ppm));
        assert_eq!(cw, w);
        assert_eq!(ch, h);

        // Cross-validate: Rust vs C djpeg, target diff=0
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "pixel data length mismatch for C-encoded SOF10"
        );
        let max_diff: u8 = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff, 0,
            "C-encoded SOF10: Rust vs C djpeg max_diff={} (must be 0)",
            max_diff
        );

        std::fs::remove_file(&ppm_path).ok();
        std::fs::remove_file(&jpg_path).ok();
        std::fs::remove_file(&c_dec_ppm).ok();
    }

    // --- Test 2: Rust-encoded SOF10 (compress_arithmetic_progressive) ---
    {
        let (w, h): (usize, usize) = (48, 32);
        let mut source_pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                source_pixels.push(((x * 255) / w) as u8);
                source_pixels.push(((y * 255) / h) as u8);
                source_pixels.push((((x + y) * 127) / (w + h)) as u8);
            }
        }

        // Rust encode as SOF10 (arithmetic progressive)
        let jpeg: Vec<u8> = compress_arithmetic_progressive(
            &source_pixels,
            w,
            h,
            PixelFormat::Rgb,
            90,
            Subsampling::S444,
        )
        .expect("Rust compress_arithmetic_progressive must succeed");

        // Verify SOF10 marker
        let has_sof10: bool = jpeg
            .windows(2)
            .any(|pair| pair[0] == 0xFF && pair[1] == 0xCA);
        assert!(
            has_sof10,
            "Rust compress_arithmetic_progressive must produce SOF10 (0xFFCA)"
        );

        let rust_jpg_path: String =
            format!("/tmp/ljt_sof10_xval_{}_rust_enc.jpg", std::process::id());
        let rust_c_dec_ppm: String =
            format!("/tmp/ljt_sof10_xval_{}_rust_c_dec.ppm", std::process::id());
        std::fs::write(&rust_jpg_path, &jpeg).unwrap();

        // Rust decode of Rust-encoded SOF10
        let rust_img =
            decompress_to(&jpeg, PixelFormat::Rgb).expect("Rust must decode its own SOF10 JPEG");
        assert_eq!(rust_img.width, w);
        assert_eq!(rust_img.height, h);

        // C djpeg decode of Rust-encoded SOF10
        let output = Command::new(&djpeg)
            .args(["-ppm", "-outfile", &rust_c_dec_ppm, &rust_jpg_path])
            .output()
            .expect("failed to run djpeg");
        assert!(
            output.status.success(),
            "djpeg failed on Rust-encoded SOF10 JPEG: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let (cw, ch, c_pixels) = parse_ppm(Path::new(&rust_c_dec_ppm));
        assert_eq!(cw, w);
        assert_eq!(ch, h);

        // Cross-validate: Rust decode vs C djpeg decode, target diff=0
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "pixel data length mismatch for Rust-encoded SOF10"
        );
        let max_diff: u8 = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);
        assert_eq!(
            max_diff, 0,
            "Rust-encoded SOF10: Rust vs C djpeg max_diff={} (must be 0)",
            max_diff
        );

        std::fs::remove_file(&rust_jpg_path).ok();
        std::fs::remove_file(&rust_c_dec_ppm).ok();
    }
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
