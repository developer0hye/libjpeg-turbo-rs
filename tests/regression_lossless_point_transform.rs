//! Regression test for lossless (SOF3) point-transform output scaling.
//!
//! Found by scheduled Fuzz Smoke `fuzz_decode_diff_c` run 29689718301
//! (2026-07-19, crash-e3ad88d5): a 16x16 3-component lossless stream with
//! component ids 'R','G','B', predictor 1, and point transform Al=2.
//! Every decoded pixel diverged from djpeg (max abs diff 189) because of
//! two C-parity gaps in our lossless path (jdlossls.c):
//!
//! 1. Undifferencing must wrap modulo 0xFFFF (16-bit), not modulo
//!    `2^precision - 1` — C's jpeg_undifference* all do
//!    `(diff + PREDICTOR) & 0xFFFF` regardless of frame precision.
//! 2. The output stage must upscale every component row by `<< Al` and
//!    truncate to the sample type (`(_JSAMPLE)(x << Al)`, i.e. `& 0xFF`
//!    for 8-bit) — our color path skipped the shift entirely and
//!    saturated instead of truncating.
//!
//! Tracked as P4-38 in docs/last_mile/phase4.md.

mod helpers;

use std::path::PathBuf;

use libjpeg_turbo_rs::{Decoder, PixelFormat};

fn decode_hex(s: &str) -> Vec<u8> {
    let compact: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    assert!(compact.len().is_multiple_of(2));
    (0..compact.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&compact[i..i + 2], 16).expect("valid hex byte"))
        .collect()
}

/// Fuzz fixture from crash-e3ad88d5 (run 29689718301): 16x16 lossless
/// (SOF3), 3 components with ids 'R'(82)/'G'(71)/'B'(66), Adobe APP14
/// transform=0, predictor Ss=1, point transform Al=2.
fn fixture_lossless_pt2() -> Vec<u8> {
    decode_hex(
        r#"
        ffd8ffee000e41646f626500640000000000ffc300110800100010035211
        00471100421100ffc4001800010101010100000000000000000000000400
        050803ffda000c03520047004200010002e7fe7fe7f7defa1234246848d0
        91a1234246848d091a1234246848d091a1234244ff0077c8d091a1234246
        848d091a1234246848d091a1234246848d091a122d043e46848d091a1234
        246848d091a1234246848d091a1234246848d0916821f234246848d091a1
        234246848d091a1234246848d091a1234246848b410f91a1234246848d09
        1a1234246848d091a1234246848d091a1234245a087c8d091a1234246848
        d091a1234246848d091a1234246848d091a122d043e46848d091a1234246
        848d091a1234246848d091a1234246848d0916821f234246848d091a1234
        246848d091a1234246848d091a1234246848b410f91a1234246848d091a1
        234246848d091a1234246848d091a1234245a087c8d091a1234246848d09
        1a1234246848d091a1234246848d091a122d043e46848d091a1234246848
        d091a1234246848d091a1234246848d0916821f234246848d091a1234246
        848d091a1234246848d091a1234246848b410f91a1234246848d091a1234
        246848d091a1234246848d091a1234245a087c8d091a1234246848d091a1
        234246848d091a1234246848d091a122d043e46848d091a1234246848d09
        1a1234246848d091a1234246848d0916821f234246848d091a1234246848
        d091a1234246848d091a1234246848ffd9
        "#,
    )
}

fn rust_decode(source: &[u8]) -> libjpeg_turbo_rs::Image {
    let mut decoder = Decoder::new(source).expect("header parse should succeed");
    decoder.set_lenient(true);
    decoder.set_block_smoothing(true);
    decoder
        .decode_image()
        .expect("lossless decode should succeed")
}

#[test]
fn lossless_point_transform_scales_output_like_c() {
    let img = rust_decode(&fixture_lossless_pt2());
    assert_eq!((img.width, img.height), (16, 16));
    assert_eq!(img.pixel_format, PixelFormat::Rgb);
    // First pixel pinned from djpeg 3.1.4.1 output: initial prediction is
    // 1 << (8 - 2 - 1) = 32, first diff is 128, so the sample is
    // (32 + 128) & 0xFFFF = 160 and the output byte is
    // (160 << 2) & 0xFF = 128 — for all three components.
    assert_eq!(&img.data[0..3], &[128, 128, 128]);
}

/// C cross-validation: lossless decode has no IDCT rounding, so the raster
/// must match djpeg exactly (diff = 0).
#[test]
fn lossless_point_transform_matches_c_djpeg_exactly() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let source: Vec<u8> = fixture_lossless_pt2();
    let img = rust_decode(&source);
    let (cw, chh, c_px) = helpers::decode_with_c_djpeg(&djpeg, &source, "lossless_pt2");
    assert_eq!((cw, chh), (img.width, img.height));
    assert_eq!(
        c_px, img.data,
        "lossless decode must be byte-exact vs djpeg (no IDCT tolerance applies)"
    );
}
