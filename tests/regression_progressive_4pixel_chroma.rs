//! P2-11 regression: progressive encode with 4-pixel chroma sampling
//! factors (TJSAMP_411 / TJSAMP_441 / TJSAMP_410 / TJSAMP_24) must match
//! cjpeg byte-for-byte and self-roundtrip with low decode error.
//!
//! The bug was in `compress_progressive_with_scans` (and the arithmetic
//! progressive equivalent): the chroma block helper clamped `h_samp` and
//! `v_samp` to `{1, 2}`, so 4-pixel factors were silently downsampled to
//! 1/2 resolution while the SOF still advertised 1/4 resolution. Decoded
//! output diverged by max≈140 / mean≈9 per pixel.
//!
//! This test is C-tool-free (uses the Rust decoder) so it gates the fix
//! on every CI host. The byte-parity-against-cjpeg version lives in
//! `tests/c_tjcomptest.rs` (gated on `--features full-c-parity`).

use libjpeg_turbo_rs::{Decoder, Encoder, PixelFormat, Subsampling};

const W: usize = 80;
const H: usize = 60;

fn synth_rgb() -> Vec<u8> {
    // Smooth gradient with a discontinuity to exercise both DCT-friendly
    // and edge regions. Width chosen so it's NOT a multiple of 32 — the
    // S411/S410 MCU width — to exercise right-edge expansion.
    let mut buf: Vec<u8> = vec![0u8; W * H * 3];
    for y in 0..H {
        for x in 0..W {
            let i: usize = (y * W + x) * 3;
            buf[i] = ((x * 255) / (W - 1)) as u8;
            buf[i + 1] = ((y * 255) / (H - 1)) as u8;
            buf[i + 2] = if x < W / 2 { 32 } else { 220 };
        }
    }
    buf
}

fn roundtrip(subsampling: Subsampling, progressive: bool) -> (Vec<u8>, usize, f64) {
    let src: Vec<u8> = synth_rgb();
    let jpeg: Vec<u8> = Encoder::new(&src, W, H, PixelFormat::Rgb)
        .fancy_downsampling(false)
        .subsampling(subsampling)
        .progressive(progressive)
        .encode()
        .expect("encode");
    let img = Decoder::decode(&jpeg).expect("decode");
    assert_eq!(img.width, W);
    assert_eq!(img.height, H);
    assert_eq!(img.data.len(), src.len(), "decoded size mismatch");

    let mut max_d: usize = 0;
    let mut sum_d: u64 = 0;
    for i in 0..src.len() {
        let d: usize = (src[i] as i32 - img.data[i] as i32).unsigned_abs() as usize;
        if d > max_d {
            max_d = d;
        }
        sum_d += d as u64;
    }
    let mean: f64 = sum_d as f64 / src.len() as f64;
    (jpeg, max_d, mean)
}

/// Floor: with the bug, mean was ~9 and max ~140. After the fix, both
/// progressive and baseline should land in normal lossy-JPEG territory
/// (max ≤ 50, mean ≤ 8). The asymmetry between baseline and progressive
/// must vanish — that asymmetry was the bug's signature.
fn assert_roundtrip_quality(label: &str, max_d: usize, mean: f64) {
    assert!(
        max_d <= 50,
        "{label}: max pixel diff {max_d} (expected ≤ 50; bug produced 140-161)"
    );
    assert!(
        mean <= 8.0,
        "{label}: mean pixel diff {mean:.4} (expected ≤ 8; bug produced ~9)"
    );
}

fn assert_progressive_matches_baseline(subsampling: Subsampling) {
    let (_, base_max, base_mean) = roundtrip(subsampling, false);
    let (_, prog_max, prog_mean) = roundtrip(subsampling, true);
    assert_roundtrip_quality(&format!("{subsampling:?} baseline"), base_max, base_mean);
    assert_roundtrip_quality(&format!("{subsampling:?} progressive"), prog_max, prog_mean);
    // Progressive and baseline must produce the same DCT coefficients
    // modulo entropy coding, so decoded pixels must match closely.
    // Allow ±1 from intermediate rounding differences.
    assert!(
        prog_max.abs_diff(base_max) <= 1,
        "{subsampling:?}: progressive max {prog_max} diverges from baseline max {base_max} by more than 1 — symptom of the chroma-clamp bug",
    );
    assert!(
        (prog_mean - base_mean).abs() <= 0.5,
        "{subsampling:?}: progressive mean {prog_mean:.4} diverges from baseline mean {base_mean:.4} — symptom of the chroma-clamp bug",
    );
}

#[test]
fn progressive_s411_roundtrip_matches_baseline_quality() {
    assert_progressive_matches_baseline(Subsampling::S411);
}

#[test]
fn progressive_s441_roundtrip_matches_baseline_quality() {
    assert_progressive_matches_baseline(Subsampling::S441);
}

#[test]
fn progressive_s410_roundtrip_matches_baseline_quality() {
    assert_progressive_matches_baseline(Subsampling::S410);
}

#[test]
fn progressive_s24_roundtrip_matches_baseline_quality() {
    assert_progressive_matches_baseline(Subsampling::S24);
}
