//! Regression tests for two `fuzz_decode_diff_c` divergences found by a local
//! 100,000-iteration smoke sweep (seed 424242, 2026-05-30). Both pin
//! repository fixtures against C `djpeg`, mirroring the fuzz oracle exactly
//! (lenient mode + block smoothing, pixel tolerance 24).
//!
//! - **P4-22** (`multiscan_noninterleaved_64x64_444.jpg`): a 64x64 baseline
//!   4:4:4 stream split into three single-component scans where component 1
//!   (luma) is never scanned and component 3 (Cr) is scanned twice. Both C
//!   backends decode it byte-identically; our decoder diverges by max abs 128
//!   on both NEON and SSE2 (so the defect is in shared scalar scan handling,
//!   not SIMD). See `docs/last_mile/phase4.md` P4-22.
//! - **P4-23** (`corrupt_huffman_65x65_422.jpg`): a 65x65 baseline 4:2:2 stream
//!   with corrupt entropy data. `djpeg` silently conceals it (exit 0, empty
//!   stderr) and emits a raster; our lenient decode rejects with
//!   `CorruptData("invalid Huffman code")`. See P4-23.
//!
//! Skip rule: `djpeg` absent on a developer machine is a soft skip; in CI it
//! hard-fails so the gate cannot vanish into a green skip.

mod helpers;

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use libjpeg_turbo_rs::{Decoder, PixelFormat};

/// Mirrors `fuzz_decode_diff_c.rs::PIXEL_TOLERANCE`.
const PIXEL_TOLERANCE: i32 = 24;

// Both fixtures live in the non-globbed `tests/fixtures/fuzz_repro/` subdir so
// `examples/generate_corpus.rs` (non-recursive `read_dir` of `tests/fixtures/*.jpg`)
// does NOT sweep them into `tests/corpus/`, where the CI Corpus Test would fail
// on a CRASH: the corrupt P4-23 input is a decode reject, and the P4-22
// multi-scan input — though it now decodes correctly — is rejected by the
// transform path ("baseline SOS covers 1 components but frame has 3"). They are
// pinned here against djpeg instead. (Same reason the h4v1 fuzz regression test
// inlines its crash fixture rather than storing it under tests/fixtures/.)
const MULTISCAN_444: &[u8] =
    include_bytes!("fixtures/fuzz_repro/multiscan_noninterleaved_64x64_444.jpg");
const CORRUPT_HUFFMAN_422: &[u8] =
    include_bytes!("fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg");

/// Returns `(w, h, channels, pixels, had_stderr)`. `had_stderr` mirrors the
/// fuzz oracle's `c_lenient_recovery` flag (djpeg emitted a warning).
fn decode_with_djpeg(djpeg: &PathBuf, jpeg: &[u8]) -> Option<(usize, usize, usize, Vec<u8>, bool)> {
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
    let had_stderr = !out.stderr.is_empty();
    let (w, h, c, px) = parse_pnm(&out.stdout)?;
    Some((w, h, c, px, had_stderr))
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

/// P4-22: multi-scan non-interleaved baseline must match `djpeg` pixel-for-pixel
/// (within IDCT tolerance). Pre-fix this diverges by max abs 128.
#[test]
fn multiscan_noninterleaved_64x64_444_matches_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let mut decoder = Decoder::new(MULTISCAN_444).expect("rust decoder rejected P4-22 fixture");
    decoder.set_lenient(true);
    decoder.set_block_smoothing(true);
    let img = decoder
        .decode_image()
        .expect("rust decode_image failed on P4-22 fixture");
    assert_eq!((img.width, img.height), (64, 64), "expected 64x64");
    assert_eq!(img.pixel_format, PixelFormat::Rgb, "expected RGB output");
    let rust_pixels = img.data.clone();

    let (cw, ch, cc, c_pixels, _c_warn) =
        decode_with_djpeg(&djpeg, MULTISCAN_444).expect("djpeg rejected P4-22 fixture");
    assert_eq!((cw, ch, cc), (64, 64, 3), "djpeg dims");
    assert_eq!(c_pixels.len(), rust_pixels.len(), "byte count mismatch");

    let mut max_d: i32 = 0;
    for (a, b) in c_pixels.iter().zip(rust_pixels.iter()) {
        max_d = max_d.max((*a as i32 - *b as i32).abs());
    }
    // Pinned regression: post-fix this fixture is byte-identical to djpeg
    // (measured max_diff = 0 on aarch64 + SSE2; the decoded image is a flat
    // DC-only color, so the IDCT is platform-exact). Assert `<= 1` (measured 0
    // + 1 margin per the Strict Assertion Rules) — a revert of the 0→128 plane
    // fill flips the never-scanned luma and reappears as diff 128, far above
    // even the fuzz oracle's own PIXEL_TOLERANCE.
    assert!(
        max_d <= 1,
        "P4-22: multi-scan non-interleaved baseline diverges from djpeg: max abs \
         diff = {} (expected 0; tolerance 1, fuzz-oracle tolerance {}); \
         first px c={:?} r={:?}",
        max_d,
        PIXEL_TOLERANCE,
        &c_pixels[..3.min(c_pixels.len())],
        &rust_pixels[..3.min(rust_pixels.len())],
    );
}

/// P4-23: lenient mode must be at least as accepting as `djpeg`, which silently
/// conceals the corrupt entropy here (the stream fragments into spurious
/// non-interleaved scans, and the non-interleaved decode path hit an invalid
/// Huffman code). Lenient decode must recover (gray-fill + warning), not reject
/// — otherwise the differential fuzzer's `(Some, Rejected)` arm panics.
#[test]
fn corrupt_huffman_65x65_422_lenient_matches_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");

    // djpeg accepts (exit 0): the drop-in floor requires we also accept.
    let c = decode_with_djpeg(&djpeg, CORRUPT_HUFFMAN_422);
    assert!(c.is_some(), "precondition: djpeg accepts the P4-23 fixture");
    let (cw, ch, cc, c_pixels, _c_warn) = c.unwrap();
    assert_eq!((cw, ch, cc), (65, 65, 3), "djpeg dims");

    let mut decoder = Decoder::new(CORRUPT_HUFFMAN_422).expect("decoder rejected P4-23 header");
    decoder.set_lenient(true);
    decoder.set_block_smoothing(true);
    let img = decoder
        .decode_image()
        .expect("P4-23: lenient decode must recover, not reject (drop-in floor)");
    assert_eq!((img.width, img.height), (65, 65));
    assert_eq!(img.pixel_format, PixelFormat::Rgb);

    // Lenient recovery must mark itself with at least one warning, so the fuzz
    // oracle's bilateral-OR lenient gate skips the pixel compare (both sides did
    // best-effort fill, and the two recovery strategies legitimately differ).
    // We assert acceptance + dims + warning, not pixels.
    assert!(
        !img.warnings.is_empty(),
        "P4-23: lenient decode must emit a recovery warning (djpeg concealed the corrupt scan)"
    );
    let _ = c_pixels;

    // Strict mode must still reject — the recovery is gated on lenient mode.
    let mut strict = Decoder::new(CORRUPT_HUFFMAN_422).expect("decoder rejected P4-23 header");
    strict.set_lenient(false);
    assert!(
        strict.decode_image().is_err(),
        "P4-23: strict mode must still reject the corrupt stream"
    );
}
