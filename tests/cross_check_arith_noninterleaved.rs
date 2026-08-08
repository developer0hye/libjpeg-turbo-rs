//! Regression test for P4-24: arithmetic-coded (SOF9) **non-interleaved
//! multi-scan** decode. Before the fix, `decode_arithmetic_planes` processed
//! only the first SOS (luma) and left every other component plane at its 0
//! init, decoding e.g. a 4:4:4 arithmetic stream to a strongly colour-cast
//! raster instead of the image `djpeg` produces. See
//! `docs/last_mile/phase4.md` P4-24.
//!
//! The fixture is a 16x16 4:4:4 arithmetic JPEG with three single-component
//! scans (`cjpeg -arithmetic -sample 1x1 -scans <one-comp-per-scan>`), so
//! `metadata.scans.len() == 3`. It is NOT reachable via `fuzz_decode_diff_c`
//! (that target skips arithmetic), so it is pinned here as a direct C
//! cross-check.
//!
//! Skip rule: `djpeg` absent on a dev machine is a soft skip; in CI it
//! hard-fails so the gate cannot vanish into a green skip.

mod helpers;

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use libjpeg_turbo_rs::{Decoder, PixelFormat};

/// One single-component scan per component (`cjpeg -scans "0; 1; 2;"`).
const ARITH_NONINTERLEAVED_444: &[u8] =
    include_bytes!("fixtures/fuzz_repro/arith_noninterleaved_16x16_444.jpg");
/// Partially interleaved: luma alone, then Cb+Cr interleaved
/// (`cjpeg -scans "0; 1 2;"`). The second scan carries two components, which
/// must decode via the frame-level interleaved MCU grid (T.81 A.2.2).
const ARITH_PARTIAL_INTERLEAVED_444: &[u8] =
    include_bytes!("fixtures/fuzz_repro/arith_partial_interleaved_16x16_444.jpg");

fn decode_with_djpeg(djpeg: &PathBuf, jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let mut child = Command::new(djpeg)
        .arg("-pnm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let payload: Vec<u8> = jpeg.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !out.status.success() {
        return None;
    }
    parse_pnm(&out.stdout)
}

fn parse_pnm(bytes: &[u8]) -> Option<(usize, usize, usize, Vec<u8>)> {
    let mut i: usize = 0;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 && i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            tokens.push(String::from_utf8(bytes[start..i].to_vec()).ok()?);
        }
    }
    if tokens.len() < 4 {
        return None;
    }
    let channels: usize = match tokens[0].as_str() {
        "P5" => 1,
        "P6" => 3,
        _ => return None,
    };
    let w: usize = tokens[1].parse().ok()?;
    let h: usize = tokens[2].parse().ok()?;
    if tokens[3] != "255" {
        return None;
    }
    i += 1;
    let needed = w.checked_mul(h)?.checked_mul(channels)?;
    if bytes.len() < i + needed {
        return None;
    }
    Some((w, h, channels, bytes[i..i + needed].to_vec()))
}

/// Decode `jpeg` with both Rust and `djpeg` and assert byte-exact agreement
/// (tolerance 1 for any platform IDCT rounding). `label` names the fixture in
/// failure messages.
fn assert_arith_matches_djpeg(djpeg: &PathBuf, jpeg: &[u8], label: &str) {
    let decoder = Decoder::new(jpeg).unwrap_or_else(|e| panic!("{label}: rust rejected: {e}"));
    let img = decoder
        .decode_image()
        .unwrap_or_else(|e| panic!("{label}: rust decode_image failed: {e}"));
    assert_eq!((img.width, img.height), (16, 16), "{label}: expected 16x16");
    assert_eq!(
        img.pixel_format,
        PixelFormat::Rgb,
        "{label}: expected RGB output"
    );
    let rust_pixels = img.data.clone();

    let (cw, ch, cc, c_pixels) =
        decode_with_djpeg(djpeg, jpeg).unwrap_or_else(|| panic!("{label}: djpeg rejected"));
    assert_eq!((cw, ch, cc), (16, 16, 3), "{label}: djpeg dims");
    assert_eq!(
        c_pixels.len(),
        rust_pixels.len(),
        "{label}: byte count mismatch"
    );

    let mut max_d: i32 = 0;
    for (a, b) in c_pixels.iter().zip(rust_pixels.iter()) {
        max_d = max_d.max((*a as i32 - *b as i32).abs());
    }
    assert!(
        max_d <= 1,
        "{label}: arithmetic multi-scan diverges from djpeg: max abs diff = {} \
         (expected 0; tolerance 1); first px c={:?} r={:?}",
        max_d,
        &c_pixels[..3.min(c_pixels.len())],
        &rust_pixels[..3.min(rust_pixels.len())],
    );
}

/// P4-24: arithmetic non-interleaved multi-scan (one component per scan) must
/// decode every scan and match `djpeg`. Pre-fix only the first scan (luma)
/// decoded and the chroma planes stayed at 0, diverging by 244.
#[test]
fn arith_noninterleaved_16x16_444_matches_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    assert_arith_matches_djpeg(&djpeg, ARITH_NONINTERLEAVED_444, "arith non-interleaved");
}

/// P4-24: arithmetic *partially* interleaved multi-scan (luma alone, then Cb+Cr
/// interleaved). The 2-component scan must decode via the frame interleaved MCU
/// grid. Pre-fix the decoder rejected the 2-component scan outright.
#[test]
fn arith_partial_interleaved_16x16_444_matches_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    assert_arith_matches_djpeg(
        &djpeg,
        ARITH_PARTIAL_INTERLEAVED_444,
        "arith partial-interleaved",
    );
}
