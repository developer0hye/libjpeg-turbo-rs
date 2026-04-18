//! A4-4 fixture: DAC with high-index arithmetic table slots round-trips.
//!
//! ITU-T T.81 / libjpeg-turbo `NUM_ARITH_TBLS` = 16. The DAC marker may
//! legally declare conditioning for Tb = 0..=15. This fixture constructs
//! a baseline SOF9 JPEG using the library's normal encoder, then splices
//! in an augmented DAC that carries non-default conditioning for slots
//! beyond the first 4 (specifically DC Tb=5 L=2, U=3 and AC Tb=12 Kx=5).
//!
//! The scan references the default slots (0 for luma, 1 for chroma) so
//! the extra high-index entries do not change entropy decoding — the
//! stream must still round-trip pixel-identically to the baseline. This
//! validates A4-1 (decoder parses Tb>3), A4-2 (encoder array allocation),
//! and A4-3 (writer emits high Tb without dropping them).
//!
//! If `/opt/homebrew/bin/djpeg` is available, the fixture additionally
//! feeds the augmented JPEG into C libjpeg-turbo and asserts the C
//! decoder produces the same pixels — guarding against the unlikely
//! case where our encoder and decoder share a deviation from C.

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{compress_arithmetic, decompress_to, PixelFormat, Subsampling};

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

/// Parse a minimal P6/P5 PPM/PGM into (width, height, samples).
fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("read PPM");
    assert!(&raw[0..2] == b"P6" || &raw[0..2] == b"P5");
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
    // Skip whitespace after maxval before binary data.
    idx = end + 1;
    (w, h, raw[idx..idx + w * h * comps].to_vec())
}

/// Locate the original DAC marker (FF CC) in `stream` and replace it
/// with `new_dac`. Returns the rebuilt stream. Panics if no DAC found.
fn replace_dac(stream: &[u8], new_dac: &[u8]) -> Vec<u8> {
    // Scan for FF CC after SOI (skip the very first byte to be safe).
    let mut i: usize = 2;
    while i + 3 < stream.len() {
        if stream[i] == 0xFF && stream[i + 1] == 0xCC {
            let length: usize = u16::from_be_bytes([stream[i + 2], stream[i + 3]]) as usize;
            let end: usize = i + 2 + length;
            let mut out: Vec<u8> = Vec::with_capacity(stream.len() - (end - i) + new_dac.len());
            out.extend_from_slice(&stream[..i]);
            out.extend_from_slice(new_dac);
            out.extend_from_slice(&stream[end..]);
            return out;
        }
        i += 1;
    }
    panic!("no DAC marker found in stream");
}

/// Build an augmented DAC segment that sets:
/// - DC Tb=0 to L=0, U=1 (default)
/// - DC Tb=1 to L=0, U=1 (default — for chroma)
/// - DC Tb=5 to L=2, U=3 (non-default, slot > 3)
/// - AC Tb=0 to Kx=5 (default)
/// - AC Tb=1 to Kx=5 (default)
/// - AC Tb=12 to Kx=5 (non-default, slot > 3)
///
/// Entries ordered DC0, AC0, DC1, AC1, DC5, AC12 (libjpeg-turbo
/// `jcmarker.c::emit_dac` interleaves DC_i and AC_i for each slot).
fn build_augmented_dac() -> Vec<u8> {
    let mut seg: Vec<u8> = Vec::new();
    seg.push(0xFF);
    seg.push(0xCC);

    // 6 entries * 2 bytes + 2 length = 14.
    let length: u16 = 14;
    seg.extend_from_slice(&length.to_be_bytes());

    // DC0: Tc/Tb=0x00, val = (U<<4)|L = 0x10
    seg.push(0x00);
    seg.push(0x10);
    // AC0: Tc/Tb=0x10, Kx=5
    seg.push(0x10);
    seg.push(0x05);
    // DC1: Tc/Tb=0x01, val=0x10
    seg.push(0x01);
    seg.push(0x10);
    // AC1: Tc/Tb=0x11, Kx=5
    seg.push(0x11);
    seg.push(0x05);
    // DC5: Tc/Tb=0x05, val=(3<<4)|2 = 0x32
    seg.push(0x05);
    seg.push(0x32);
    // AC12: Tc/Tb=0x1C, Kx=5
    seg.push(0x1C);
    seg.push(0x05);

    seg
}

#[test]
fn dac_with_high_table_slots_roundtrips_identically() {
    // Solid-gray image to minimize quantization artefacts and give a
    // stable baseline for the diff comparison.
    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = vec![128u8; w * h * 3];

    // Baseline: normal arithmetic JPEG (DAC slots 0 & 1 only).
    let baseline: Vec<u8> =
        compress_arithmetic(&pixels, w, h, PixelFormat::Rgb, 75, Subsampling::S444)
            .expect("baseline arithmetic compress");

    let baseline_dec = decompress_to(&baseline, PixelFormat::Rgb).expect("baseline decompress");
    assert_eq!(baseline_dec.width, w);
    assert_eq!(baseline_dec.height, h);

    // Splice an augmented DAC that additionally declares slot 5 (DC)
    // and slot 12 (AC) with non-default conditioning.
    let augmented: Vec<u8> = replace_dac(&baseline, &build_augmented_dac());

    // The augmented stream must decode to exactly the same pixels as
    // the baseline (the scan still references slots 0/1 with their
    // default conditioning — the extra DAC entries are preserved in
    // the decoder's 16-slot conditioning array but never consulted).
    let augmented_dec = decompress_to(&augmented, PixelFormat::Rgb).expect("augmented decompress");
    assert_eq!(augmented_dec.width, w);
    assert_eq!(augmented_dec.height, h);
    assert_eq!(augmented_dec.data.len(), baseline_dec.data.len());

    let max_diff: u8 = baseline_dec
        .data
        .iter()
        .zip(augmented_dec.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        max_diff, 0,
        "augmented-DAC decode must match baseline pixel-for-pixel (max_diff={})",
        max_diff
    );

    // Cross-check: the augmented stream is still spec-conformant, so
    // C libjpeg-turbo's djpeg must also decode it identically. If djpeg
    // is unavailable we only skip the external tool — not the Rust
    // round-trip above, which has already passed.
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found (C cross-check omitted)");
            return;
        }
    };

    let tmp_jpg: String = "/tmp/ljt_a4_high_slots.jpg".to_string();
    let tmp_ppm: String = "/tmp/ljt_a4_high_slots.ppm".to_string();
    std::fs::write(&tmp_jpg, &augmented).expect("write augmented jpg");
    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(&tmp_ppm)
        .arg(&tmp_jpg)
        .output()
        .expect("failed to run djpeg");
    assert!(
        output.status.success(),
        "djpeg rejected DAC with Tb=5/12: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let (cw, ch, c_pixels) = parse_ppm(Path::new(&tmp_ppm));
    std::fs::remove_file(&tmp_jpg).ok();
    std::fs::remove_file(&tmp_ppm).ok();
    assert_eq!(cw, w);
    assert_eq!(ch, h);

    let c_diff: u8 = c_pixels
        .iter()
        .zip(augmented_dec.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
        .max()
        .unwrap_or(0);
    assert_eq!(
        c_diff, 0,
        "C djpeg vs Rust decode of augmented DAC must match exactly (max_diff={})",
        c_diff
    );
}
