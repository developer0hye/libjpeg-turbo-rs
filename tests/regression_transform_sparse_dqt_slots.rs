//! Regression tests for sparse DQT slot handling in the transform path.
//!
//! Found by scheduled Fuzz Smoke `fuzz_transform_diff_c` runs
//! 29679993066..30064906856 (2026-07-19..24, ten crashes, one root cause):
//! `read_coefficients` collected quantization tables with `filter_map`,
//! compacting the four DQT slots into a dense `Vec` while each component
//! kept its *original* slot index. Any input whose defined DQT slots are
//! not exactly `0..n` (e.g. only slot 1 defined, or slots {0,1,3} with a
//! gap at 2) re-encoded into a JPEG whose SOF references a quantization
//! table the output never defines — djpeg rejects it with
//! "Quantization table 0xNN was not defined" while jpegtran's output
//! decodes fine.
//!
//! Tracked as P4-34 in docs/last_mile/phase4.md.

mod helpers;

use std::path::PathBuf;

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, MarkerCopyMode, TransformOp, TransformOptions,
};

fn decode_hex(s: &str) -> Vec<u8> {
    let compact: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    assert!(compact.len().is_multiple_of(2));
    (0..compact.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&compact[i..i + 2], 16).expect("valid hex byte"))
        .collect()
}

/// Fuzz fixture from crash-44f3519c (run 29679993066): progressive 16x16,
/// the only defined quantization table lives in DQT slot 1 and every SOF
/// component references Tq1. Slot 0 is never defined.
fn fixture_slot1_only() -> Vec<u8> {
    decode_hex(
        r#"
        ffd8ffffffffffffffffffffffee0004ffffffffffffffffffffffffffff
        ffffffee0004ff14ffdb00430103040405040509050509140d0b0d141414
        141414141414ffffffffffffffff14141414141414141414141414141414
        1414146014141414141414141414141414ffc20011080010001003012201
        021101031101ffc4001500010100000000000000000000000000000604ff
        c4001501010100000000000000000000000000000405ffda000c03010002
        10031000000135426116dbffc4001a100001050100000000000000000000
        00000441f6fa13ffefffda000801010001050230728eac6e34c82dad0014
        63da2e9fffc4001811000203000000000000000000000000000305000102
        ffda0008010301013f017b3fffd9ffffffffffffffc42a0000000102ffff
        ffffffffff0100021003100000017e30426116dbffc4001a100001050100
        00000000000000000000000441f6fa13ff7aefffda000801010001050230
        8eac6e34c82dad63da2effdfc40018110002030000e7e7e7e7e7e7e7e7e7
        e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7
        e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7e7
        e7e7e7e7e700000000000000000000000305000102ffda0008010301013f
        01000000000b0c
        "#,
    )
}

/// Fuzz fixture from crash-7dab2f09 (run 29751094520): progressive 16x16,
/// DQT slots 0, 1, and 3 are defined with a gap at slot 2; the third SOF
/// component references Tq3.
fn fixture_slot_gap_at_2() -> Vec<u8> {
    decode_hex(
        r#"
        ffd8ffe000104a46494600010100000100010000ffdb0043030202030202
        03030303040303040502ffd8ffe00010024649c6030405060708090a0bff
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        ffffffffffffffffffffe000104a4649460001010000ffffffffffffffff
        ffffffffffffffffffe000104a46494600010100000100010000ffffffff
        ffffffffffffffffffdb0043000302020302020303030304030304050805
        050404050a070706080c0a0c0010024649c6030405060708090a0bffffff
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        ffffffffffffffffe000104a4649460001010000ffffffffffffffffffff
        ffffffffffffffe000104a46494600010100000100010000ffffffffffff
        ffffffffffffffdb00430003020203020203030303040303040508050504
        04050a070706080c0a0c0c0bdb0b0b0d0e12100d0e110e0b0b1016101113
        141515151618170f0c141812141514ffdb00430103040405040509050509
        140d0b0d141414141414141414ffffffffffffffff141414141414141414
        141414141414141414141414141414141414141414141414ffc200110800
        10001003012200021101031103ffc4001500010100000000000000000000
        000000000604ffc4001501010100000000000000000000000000000405ff
        da000c0301000210031000000152426116dbffc4001a1000010501000000
        00000000000000000004010205131412ffda0008010100010502388cac6e
        34c82dad63ca2e9fffc40018110002030000000000000000000000000003
        05000102ffda0008010301013f017b3fffd9ffffffffffffffc40000000c
        0bdb0b0b0d0e12100d0e110e0b0b1016101113141515151618170f0c1418
        12141514ffdb00430103040405040509050509140d0b0d14141414141414
        1414ffffffffffffffff1414141414141414141414141414141414141414
        14141414141414141414141414ffc2001108001000100301220002110103
        1101ffc4001500010100000000000000000000000000000604ffc4001501
        010100000000000000000000000000000405ffda000c0301000210031000
        0001000000000002830152426116dbffc4001a1000010501000000000000
        00000000000004010205131412ffda0008010100010502388cac6e34c82d
        ad63ca2e9fffc40018110002030000000000000000000000000003050001
        02ffda0008010301013f017b3fffd9ffffffffffffffc40000000102ffda
        0008110301013f016ba2daf14fffc40017110000007b3fffd9
        "#,
    )
}

/// Scan a JPEG byte stream and assert every quantization table slot
/// referenced by a SOF component is defined by some DQT segment.
/// Pure-Rust structural check so the regression holds even where the C
/// tools are unavailable.
fn assert_sof_quant_refs_defined(jpeg: &[u8], label: &str) {
    let mut defined: [bool; 16] = [false; 16];
    let mut referenced: Vec<u8> = Vec::new();
    let mut i: usize = 2; // skip SOI
    while i + 4 <= jpeg.len() {
        if jpeg[i] != 0xFF {
            i += 1;
            continue;
        }
        let marker: u8 = jpeg[i + 1];
        if marker == 0xFF || marker == 0x00 {
            i += 1;
            continue;
        }
        if marker == 0xD9 || (0xD0..=0xD7).contains(&marker) {
            i += 2;
            continue;
        }
        let len: usize = ((jpeg[i + 2] as usize) << 8) | jpeg[i + 3] as usize;
        let seg: &[u8] = &jpeg[i + 4..(i + 2 + len).min(jpeg.len())];
        match marker {
            0xDB => {
                let mut j: usize = 0;
                while j < seg.len() {
                    let precision: usize = (seg[j] >> 4) as usize;
                    defined[(seg[j] & 0x0F) as usize] = true;
                    j += 1 + if precision != 0 { 128 } else { 64 };
                }
            }
            0xC0..=0xC2 => {
                let component_count: usize = seg[5] as usize;
                for c in 0..component_count {
                    referenced.push(seg[8 + 3 * c]);
                }
            }
            0xDA => break, // entropy data follows; markers of interest all precede it
            _ => {}
        }
        i += 2 + len;
    }
    assert!(
        !referenced.is_empty(),
        "{label}: no SOF component found in output"
    );
    for tq in referenced {
        assert!(
            defined[tq as usize],
            "{label}: SOF references quantization table {tq} but the output never defines it"
        );
    }
}

fn transform(source: &[u8], op: TransformOp) -> Vec<u8> {
    transform_jpeg_with_options(
        source,
        &TransformOptions {
            op,
            copy_markers: MarkerCopyMode::All,
            ..Default::default()
        },
    )
    .expect("Rust transform should succeed")
}

#[test]
fn transform_preserves_quant_reference_when_only_slot1_defined() {
    let source: Vec<u8> = fixture_slot1_only();
    let transformed: Vec<u8> = transform(&source, TransformOp::Rot180);
    assert_sof_quant_refs_defined(&transformed, "slot1_only rot180");

    let djpeg: PathBuf = require_c_tool!("djpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let c_transformed: Vec<u8> = helpers::transform_with_c_jpegtran(
        &jpegtran,
        &source,
        &["-copy", "all", "-rotate", "180"],
        "sparse_dqt_slot1_only",
    );
    let (rust_width, rust_height, rust_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &transformed, "sparse_dqt_slot1_only_rust");
    let (c_width, c_height, c_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &c_transformed, "sparse_dqt_slot1_only_c");
    // Fuzz-input oracle is acceptance + decoded-dimension agreement (see
    // fuzz_transform_diff_c); pixel parity on adversarial inputs is left
    // to curated corpus tests.
    assert_eq!((rust_width, rust_height), (16, 16));
    assert_eq!((c_width, c_height), (16, 16));
    assert_eq!(rust_pixels.len(), c_pixels.len());
}

#[test]
fn transform_preserves_quant_reference_across_dqt_slot_gap() {
    let source: Vec<u8> = fixture_slot_gap_at_2();
    let transformed: Vec<u8> = transform(&source, TransformOp::HFlip);
    assert_sof_quant_refs_defined(&transformed, "slot_gap_at_2 hflip");

    let djpeg: PathBuf = require_c_tool!("djpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let c_transformed: Vec<u8> = helpers::transform_with_c_jpegtran(
        &jpegtran,
        &source,
        &["-copy", "all", "-flip", "horizontal"],
        "sparse_dqt_slot_gap",
    );
    let (rust_width, rust_height, rust_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &transformed, "sparse_dqt_slot_gap_rust");
    let (c_width, c_height, c_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &c_transformed, "sparse_dqt_slot_gap_c");
    assert_eq!((rust_width, rust_height), (16, 16));
    assert_eq!((c_width, c_height), (16, 16));
    assert_eq!(rust_pixels.len(), c_pixels.len());
}
