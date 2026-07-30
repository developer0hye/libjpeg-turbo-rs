//! Fuzz Smoke `fuzz_decompress` regression: the reduced-size IDCT used by
//! scaled decode overflowed `i32` and panicked.
//!
//! Scheduled runs 30420069849 (2026-07-29 03:37), 30438301770 (09:07),
//! 30461194331 (14:30) and 30485530878 (19:41) all aborted in
//! `src/decode/idct_scaled.rs` with `attempt to {add,subtract,multiply}
//! with overflow`, across five distinct minimized seeds and five distinct
//! source lines (116, 117, 126, 131, 136). Only the `data.len() % 7 == 3`
//! arm of the fuzz target reaches them, because that is the arm that sets
//! `ScalingFactor::new(1, 2)` — the sole dispatcher of `idct_4x4`.
//!
//! Root cause: `idct_scaled.rs` was ported from `jidctred.c` with plain
//! `+`/`-`/`*`. C's intermediates are `JLONG`, declared `long` at
//! `jpegint.h:62` with only a "must hold at least signed 32-bit values"
//! guarantee, so wrapping is the C contract, not an error to surface.
//! `decode/idct.rs` (the full 8x8 twin) already used `wrapping_*`
//! throughout; the reduced-size variants never got the same treatment.
//!
//! A dequantized coefficient is bounded by `i16::MAX * u16::MAX`, which
//! fits `i32` — but a single fixed-point multiply by e.g.
//! `FIX_2_562915447` (20995) does not, so any large coefficient paired
//! with a large quant value overflows.

use libjpeg_turbo_rs::{compress, Decoder, PixelFormat, ScalingFactor, Subsampling};

/// High-contrast 16x16 source: adjacent-pixel swings maximize the AC
/// coefficient magnitudes that later get multiplied by the patched quant
/// values.
fn checkerboard_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let v: u8 = if (x + y) % 2 == 0 { 0 } else { 255 };
            pixels.extend_from_slice(&[v, v, v]);
        }
    }
    pixels
}

/// Rewrite every 8-bit DQT entry to 255. This keeps the stream
/// structurally valid (segment lengths and `Pq` are untouched) while
/// driving `coeff * quant` to the top of its legal range — exactly the
/// shape the fuzzer minimized to, without depending on a corpus file.
fn saturate_quant_tables(jpeg: &mut [u8]) -> usize {
    let mut patched: usize = 0;
    let mut i: usize = 2; // skip SOI
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            break;
        }
        let marker: u8 = jpeg[i + 1];
        if marker == 0xD8 || marker == 0xD9 {
            i += 2;
            continue;
        }
        let seg_len: usize = ((jpeg[i + 2] as usize) << 8) | jpeg[i + 3] as usize;
        if seg_len < 2 || i + 2 + seg_len > jpeg.len() {
            break;
        }
        if marker == 0xDB {
            // DQT payload: [Pq/Tq][64 values] repeated.
            let mut p: usize = i + 4;
            let end: usize = i + 2 + seg_len;
            while p < end {
                let pq: u8 = jpeg[p] >> 4;
                p += 1;
                let n: usize = if pq == 0 { 64 } else { 128 };
                if p + n > end {
                    break;
                }
                for b in &mut jpeg[p..p + n] {
                    *b = 0xFF;
                }
                patched += 1;
                p += n;
            }
        }
        if marker == 0xDA {
            break; // entropy data follows; stop scanning
        }
        i += 2 + seg_len;
    }
    patched
}

/// A corrupt-quantization stream decoded at 1/2 scale must produce a
/// result or a typed error — never a panic. Before the fix this aborted
/// inside `idct_4x4`'s pass-2 odd part.
#[test]
fn scaled_decode_survives_saturated_quant_tables() {
    let (w, h) = (16usize, 16usize);
    let pixels: Vec<u8> = checkerboard_rgb(w, h);
    let mut jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 100, Subsampling::S444)
        .expect("encode the control fixture");

    let patched: usize = saturate_quant_tables(&mut jpeg);
    assert!(
        patched > 0,
        "fixture layout changed: no 8-bit DQT segment found to saturate"
    );

    let mut decoder = Decoder::new(&jpeg).expect("patched stream still parses a header");
    decoder.set_scale(ScalingFactor::new(1, 2));
    // Either outcome is acceptable; the contract under test is "no panic".
    let _ = decoder.decode_image();
}

/// The same stream at every scaling factor: 1/8 and 1/4 route to
/// `idct_1x1` / `idct_2x2`, which carry the identical multiply chains and
/// were fixed alongside `idct_4x4`.
#[test]
fn every_reduced_scale_survives_saturated_quant_tables() {
    let (w, h) = (32usize, 32usize);
    let pixels: Vec<u8> = checkerboard_rgb(w, h);
    let base: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 100, Subsampling::S420)
        .expect("encode the control fixture");

    for (num, denom) in [(1u32, 8u32), (1, 4), (1, 2), (3, 8), (5, 8), (7, 8), (1, 1)] {
        let mut jpeg: Vec<u8> = base.clone();
        assert!(saturate_quant_tables(&mut jpeg) > 0);
        let mut decoder = Decoder::new(&jpeg).expect("patched stream still parses a header");
        decoder.set_scale(ScalingFactor::new(num, denom));
        let _ = decoder.decode_image();
    }
}

/// Control: the *unpatched* fixture must decode cleanly at 1/2 scale to
/// the expected dimensions. Without this the two tests above would still
/// pass if `compress` silently produced an undecodable stream.
#[test]
fn unpatched_fixture_decodes_at_half_scale() {
    let (w, h) = (16usize, 16usize);
    let pixels: Vec<u8> = checkerboard_rgb(w, h);
    let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 100, Subsampling::S444)
        .expect("encode the control fixture");

    let mut decoder = Decoder::new(&jpeg).expect("header parses");
    decoder.set_scale(ScalingFactor::new(1, 2));
    let image = decoder
        .decode_image()
        .unwrap_or_else(|e| panic!("clean 1/2-scale decode must succeed: {e}"));
    assert_eq!((image.width, image.height), (8, 8));
}
