//! Issue #382: the 12/16-bit lossless path computed the initial
//! prediction `1 << (precision - pt - 1)` without validating
//! `pt < precision`, so a crafted SOF3 stream with e.g. `P=2` and SOS
//! `Al=15` made the shift amount negative — overflow panic in debug,
//! wrong pixels in release. C rejects such scans outright:
//! `jdlossls.c` start_pass requires `0 <= Al <= data_precision - 1`
//! and raises `JERR_BAD_PROGRESSION` otherwise (jdlossls.c:247-261).
//!
//! The identical 8-bit defect was clamped in `src/decode/lossless.rs`
//! (`undifference_row`) by the 2026-04-21 fuzz-hardening pass
//! (`cd98c21`); the high-precision twin in `src/api/precision.rs` never
//! received it. Encode had the same gap:
//! the precision variants only rejected `pt > 15`, so `pt >= precision`
//! slipped through and produced streams C refuses to decode.

mod helpers;

use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::precision::{compress_16bit, decompress_lossless_arbitrary};

/// Minimal single-component lossless (SOF3) stream: 1x1, precision `p`,
/// predictor 1, point transform `al`. DC table has a single 1-bit code
/// mapping to category 0 (diff = 0); the entropy data is that one code
/// padded with 1-bits.
fn crafted_lossless_stream(p: u8, al: u8) -> Vec<u8> {
    let mut s: Vec<u8> = Vec::new();
    // SOI
    s.extend_from_slice(&[0xFF, 0xD8]);
    // SOF3: len=11, precision p, 1x1, 1 component (id 1, 1x1 sampling, tq 0)
    s.extend_from_slice(&[
        0xFF, 0xC3, 0x00, 0x0B, p, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
    ]);
    // DHT: len=20, DC table 0, one code of length 1 -> value 0 (category 0)
    s.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00, 0x01]);
    s.extend_from_slice(&[0x00; 15]); // remaining bits-per-length counts
    s.push(0x00); // the single huffman value: category 0
                  // SOS: len=8, 1 component (id 1, DC table 0), Ss=1 (predictor), Se=0, Ah=0/Al
    s.extend_from_slice(&[
        0xFF,
        0xDA,
        0x00,
        0x08,
        0x01,
        0x01,
        0x00,
        0x01,
        0x00,
        al & 0x0F,
    ]);
    // Entropy: single '0' bit (category-0 code), padded with 1s
    s.push(0x7F);
    // EOI
    s.extend_from_slice(&[0xFF, 0xD9]);
    s
}

/// Issue #382: `P=2` with `Al=15` must produce a typed error, exactly as
/// C rejects the scan (`JERR_BAD_PROGRESSION`, jdlossls.c:247-261) — not
/// a shift-overflow panic (debug) or silently wrong pixels (release).
#[test]
fn issue_382_p2_al15_decode_returns_error_not_panic() {
    let stream: Vec<u8> = crafted_lossless_stream(2, 15);
    let result = decompress_lossless_arbitrary(&stream);
    assert!(
        result.is_err(),
        "P=2/Al=15 must be rejected (C raises JERR_BAD_PROGRESSION), got Ok"
    );
}

/// Same for the realistic 12-bit precision with the maximal SOS nibble.
#[test]
fn issue_382_p12_al15_decode_returns_error_not_panic() {
    let stream: Vec<u8> = crafted_lossless_stream(12, 15);
    let result = decompress_lossless_arbitrary(&stream);
    assert!(
        result.is_err(),
        "P=12/Al=15 must be rejected (C raises JERR_BAD_PROGRESSION), got Ok"
    );
}

/// Control: the same crafted skeleton with a *legal* point transform must
/// decode, proving the two rejection tests fail on the parameters rather
/// than on a malformed fixture.
#[test]
fn issue_382_crafted_stream_with_legal_pt_decodes() {
    let stream: Vec<u8> = crafted_lossless_stream(12, 4);
    let image = decompress_lossless_arbitrary(&stream)
        .unwrap_or_else(|e| panic!("legal P=12/Al=4 fixture must decode: {e}"));
    assert_eq!((image.width, image.height), (1, 1));
    // diff 0 on the first sample: value is the initial prediction
    // 1 << (12 - 4 - 1) = 128, upscaled by << Al at output (jdlossls.c
    // simple_upscale): 128 << 4 = 2048.
    assert_eq!(image.data, vec![2048u16]);
}

/// C cross-validation: djpeg (libjpeg-turbo 3.x decodes lossless) must
/// also refuse the crafted P=12/Al=15 stream.
#[test]
fn issue_382_c_djpeg_also_rejects_al_ge_precision() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    // djpeg 2.x can't decode SOF3 at all — only meaningful on 3.x, but a
    // 2.x rejection is also a rejection, so no version probe is needed.
    let stream: Vec<u8> = crafted_lossless_stream(12, 15);
    let tmp: PathBuf = std::env::temp_dir().join(format!("issue382_{}.jpg", std::process::id()));
    std::fs::write(&tmp, &stream).expect("write tmp");
    let output = Command::new(&djpeg).arg(&tmp).output().expect("run djpeg");
    let _ = std::fs::remove_file(&tmp);
    assert!(
        !output.status.success(),
        "djpeg accepted P=12/Al=15, but we reject it: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The fuzz target this issue added (`fuzz_decompress_precision`) found a
/// second overflow immediately: a corrupt DHT can map a code to any symbol
/// value, and `decode_dc_wide`'s sign extension shifted by it
/// (`-1i32 << category`), overflowing in debug for category >= 32. C
/// rejects DC symbols above 16 when building the table
/// (jdhuff.c:252-260, `JERR_BAD_HUFF_TABLE`).
#[test]
fn issue_382_dc_category_above_16_rejected() {
    let mut stream: Vec<u8> = crafted_lossless_stream(12, 4);
    // Patch the single DHT value byte (category 0 -> 200); it is the byte
    // immediately before the SOS marker in the crafted skeleton.
    let sos: usize = stream
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("SOS present");
    assert_eq!(
        stream[sos - 1],
        0x00,
        "fixture layout changed: expected the DHT value byte before SOS"
    );
    stream[sos - 1] = 200;
    let result = decompress_lossless_arbitrary(&stream);
    assert!(
        result.is_err(),
        "DC category 200 from a corrupt DHT must be a typed error, got Ok"
    );
}

/// Issue #382 encode twin: the precision encode variants only rejected
/// `pt > 15`, so `pt >= precision` slipped through and wrote streams C
/// refuses. C enforces the same `Al <= precision - 1` contract on the
/// encode side too, in `jpeg_enable_lossless` (jcparam.c:580-589,
/// `JERR_BAD_PROGRESSION`) and `validate_script` (jcmaster.c:390-399,
/// `JERR_BAD_PROG_SCRIPT`).
#[test]
fn issue_382_encode_rejects_pt_ge_precision() {
    let pixels: Vec<u16> = vec![0u16; 16];
    // 16-bit entry with an explicit 12-bit-like precision is exercised via
    // compress_16bit (precision 16): Al is a 4-bit SOS field, so pt can
    // never reach 16 — the reachable violations are the sub-16 precisions.
    // compress_lossless_arbitrary covers those.
    for (precision, pt) in [(12u8, 12u8), (12, 15), (4, 4), (4, 15)] {
        let result = libjpeg_turbo_rs::precision::compress_lossless_arbitrary(
            &pixels, 4, 4, 1, // grayscale
            precision, 1, pt,
        );
        assert!(
            result.is_err(),
            "encode with pt={pt} >= precision={precision} must be a typed error"
        );
    }
    // The 16-bit family with a sub-16 precision override had the actual
    // gap: only `pt > 15` was rejected, so pt=15 with precision 13 slipped
    // through to a negative shift in lossless_diff_16 (debug panic).
    for (precision_override, pt) in [(13u8, 13u8), (13, 15), (14, 14)] {
        let result = libjpeg_turbo_rs::api::precision::compress_16bit_with_precision(
            &pixels,
            4,
            4,
            1,
            1,
            pt,
            Some(precision_override),
        );
        assert!(
            result.is_err(),
            "compress_16bit_with_precision pt={pt} >= precision={precision_override} must error"
        );
    }
    let ok13 = libjpeg_turbo_rs::api::precision::compress_16bit_with_precision(
        &pixels,
        4,
        4,
        1,
        1,
        12,
        Some(13),
    );
    assert!(ok13.is_ok(), "pt=12 < precision=13 must encode: {ok13:?}");
    // Out-of-range overrides must be rejected outright — with the old
    // `pt > 15` guard gone, Some(17)/pt=16 would otherwise serialize a
    // self-inconsistent Al (& 0x0F) and Some(255)/pt=32 would panic.
    for (bad_precision, pt) in [(17u8, 16u8), (255, 32), (1, 0)] {
        let result = libjpeg_turbo_rs::api::precision::compress_16bit_with_precision(
            &pixels,
            4,
            4,
            1,
            1,
            pt,
            Some(bad_precision),
        );
        assert!(
            result.is_err(),
            "precision override {bad_precision} outside 2..=16 must error"
        );
    }
    // Control: legal pt still encodes.
    let ok = libjpeg_turbo_rs::precision::compress_lossless_arbitrary(&pixels, 4, 4, 1, 12, 1, 11);
    assert!(ok.is_ok(), "pt=11 < precision=12 must encode: {ok:?}");
    // The 16-bit convenience wrapper stays unaffected: every 4-bit Al is
    // legal against precision 16.
    let ok16 = compress_16bit(&pixels, 4, 4, 1, 1, 15);
    assert!(ok16.is_ok(), "pt=15 < precision=16 must encode: {ok16:?}");
}
