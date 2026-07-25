//! Regression test for transcoding coefficients whose magnitude category
//! cannot be expressed in a DHT symbol.
//!
//! Found by scheduled Fuzz Smoke `fuzz_transform_diff_c` run 30064906856
//! (2026-07-24, crash-7a0c14f3): an arithmetic progressive source decodes
//! an AC coefficient to -32768 (`(v << Al)` wrap in AC-first — C's
//! jdarith.c wraps identically, verified against `jpeg_read_coefficients`).
//! Re-encoding that value with Huffman requires magnitude category 16,
//! which the 4-bit size field of a DHT symbol cannot express. C's scalar
//! encoder rejects it with ERREXIT(JERR_BAD_DCT_COEF) (jchuff.c); its
//! x86 SIMD path silently emits an undecodable stream that djpeg flags
//! with "bad Huffman code" warnings. Our writer used to do the latter —
//! it now matches the scalar C contract and fails the transcode with
//! `CorruptData`, which the differential fuzz harness treats as an
//! inconclusive (skip) outcome.
//!
//! Tracked as P4-35 in docs/last_mile/phase4.md.

use libjpeg_turbo_rs::{
    read_coefficients, transform_jpeg_with_options, JpegError, MarkerCopyMode, TransformOp,
    TransformOptions,
};

fn decode_hex(s: &str) -> Vec<u8> {
    let compact: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    assert!(compact.len().is_multiple_of(2));
    (0..compact.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&compact[i..i + 2], 16).expect("valid hex byte"))
        .collect()
}

/// Fuzz fixture from crash-7a0c14f3 (run 30064906856): 228x186 arithmetic
/// progressive (SOF10) stream whose AC-first scan for component 2 decodes
/// one coefficient to -32768 at Al=0.
fn fixture_arith_progressive_min_coefficient() -> Vec<u8> {
    decode_hex(
        r#"
        ffd8ffe000104a4649460001010000010004ffc5ffffffffffffffffffff
        ffffffffffffffffffdb0043000302060302020303030304030304ffea05
        0504040e0a070706080c0a0c0c0bdb0b0b0d0e12100d0e110e0b0b101609
        1113141515151618170f0c141812141514ffdb004301ffd5ffffffffffff
        ffffffe000104a4649460001010000ffffffff7f03000000000000e00010
        024649c6030405060708090a0bff2d00104a4649460001030000ffffffff
        ffffffffffffffffffffffffffe000104a46494600010001010100000000
        ffca00110800ba00e403012200021101031101ffc4001500010100000000
        00000000000000000000000bffc400150101010000000000000000000000
        0000000405ffda000c0301020212031000000152426116db030000000000
        0000000800000000dd00000000000100008f000400bcbcbcbcbcbcbcbcbc
        bcbc000000000000ffda0008010301013f016ba2daf14fffc40017110101
        010100000000000000000000000002040014ffda0008010201013f0128a2
        00002d00e400dbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdb
        dbdbdbdbdbdbdbdbdbdbdbdbdbdbdbdbc6db000000000000000000000000
        000000420000000000000000dbdbdbdbdbdbdb00dbdbdbdbdbdb2500e400
        dbdbdbdb01112151ffda0008010201013f1066dbdb0100006b353532db25
        000000000000000000000000000000000000000000000000000000000000
        000000000000000000000000000000000000000000000000000000000000
        00000000000000ff00000000000000000000000000000000000000000000
        000000db001fdbdbdbdbdbdbdbdbdb0011fbdbdbdb00000000007b2dffd9
        "#,
    )
}

#[test]
fn unencodable_coefficient_transcode_fails_with_corrupt_data() {
    let source: Vec<u8> = fixture_arith_progressive_min_coefficient();

    // Pin the mechanism: the arithmetic decode really does yield an
    // i16::MIN coefficient (identical to C jpeg_read_coefficients on
    // this stream), so the writer must face category 16.
    let coeffs = read_coefficients(&source).expect("coefficient read should succeed");
    let has_min: bool = coeffs
        .components
        .iter()
        .any(|c| c.blocks.iter().any(|b| b.contains(&i16::MIN)));
    assert!(
        has_min,
        "fixture must decode to an i16::MIN coefficient to exercise category 16"
    );

    let result = transform_jpeg_with_options(
        &source,
        &TransformOptions {
            op: TransformOp::HFlip,
            copy_markers: MarkerCopyMode::All,
            ..Default::default()
        },
    );
    match result {
        Err(JpegError::CorruptData(msg)) => {
            assert!(
                msg.contains("out of range"),
                "expected out-of-range coefficient rejection, got: {msg}"
            );
        }
        Err(other) => panic!("expected CorruptData, got {other:?}"),
        Ok(_) => panic!(
            "transcode must not silently emit an undecodable Huffman stream \
             for a category-16 coefficient"
        ),
    }
}
