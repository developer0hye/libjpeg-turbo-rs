mod helpers;

use std::path::PathBuf;

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, write_coefficients_progressive, ComponentCoefficients, Decoder,
    JpegCoefficients, MarkerCopyMode, TransformOp, TransformOptions,
};

fn decode_hex(s: &str) -> Vec<u8> {
    let compact: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    assert!(compact.len().is_multiple_of(2));
    (0..compact.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&compact[i..i + 2], 16).expect("valid hex byte"))
        .collect()
}

#[test]
fn progressive_hflip_with_large_coefficients_writes_decodable_jpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let source: Vec<u8> = decode_hex(
        r#"
        ffd8ffe000104a46494600010100000100010000ffdb0043000302020302020303030304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b1016101113141515150c0f171816141812141514ffdb0043010304
        0405040509050509140d0b0d1414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414ffc20011080010001003012200021101031101ffc400150001010000000000000000
        0000000000000604ffc4001501010100000000000000000000000000000405ffda000c0301000210031000000152426116dbffc4001a100001050100000000000000000000000004010205139912ffda0008010100010502358cac6e34c849ad
        6bca2e9fffc4001811000203000000000000000000000000000305000102ffda0008010301013f016ba2daf14fffc40017110101010100000000000000000000000002040014ffda0008010201013f0191b279d12f7fffc40014100100000000
        000000000000000000000020ffda0008010100063f021fffc4001810010101010100000000000000000000000100312141ffda0008010100013f212ac8119b09ef7660baefb7ffda000c03010002000300000010cfffc4001611010101000000
        0000000000000000000001ff00ffda0008010301013f10465c2fffc4001811000203000000000000000000000000000001113051ffda0008010201010c1076831a7fffc4001810000301010000000000000000000000000021301081ffda0008
        010100013f104ec662b070757b3fffd9
        "#,
    );

    let transformed: Vec<u8> = transform_jpeg_with_options(
        &source,
        &TransformOptions {
            op: TransformOp::HFlip,
            copy_markers: MarkerCopyMode::All,
            ..Default::default()
        },
    )
    .expect("Rust transform should succeed");

    let c_transformed: Vec<u8> = helpers::transform_with_c_jpegtran(
        &jpegtran,
        &source,
        &["-copy", "all", "-flip", "horizontal"],
        "progressive_hflip_large_coeffs",
    );
    let (rust_width, rust_height, rust_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &transformed, "progressive_hflip_large_coeffs_rust");
    let (c_width, c_height, c_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &c_transformed, "progressive_hflip_large_coeffs_c");
    // This fixture comes from `fuzz_transform_diff_c`, whose fuzz-input oracle is
    // acceptance + decoded-dimension agreement. Pixel parity on adversarial fuzz
    // inputs is intentionally left to curated corpus tests.
    assert_eq!((rust_width, rust_height), (16, 16));
    assert_eq!((c_width, c_height), (16, 16));
    assert_eq!(rust_pixels.len(), c_pixels.len());
}

#[test]
fn progressive_restart_transform_checks_dc_categories_after_restart_reset() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let mut blocks: Vec<[i16; 64]> = vec![[0i16; 64]; 2];
    blocks[0][0] = 2047;
    blocks[1][0] = 3000;

    let source: Vec<u8> = write_coefficients_progressive(
        &JpegCoefficients {
            width: 16,
            height: 8,
            data_precision: 8,
            components: vec![ComponentCoefficients {
                blocks,
                blocks_x: 2,
                blocks_y: 1,
                h_sampling: 1,
                v_sampling: 1,
                quant_table_index: 0,
                component_id: 1,
            }],
            quant_tables: vec![[1u16; 64]],
            restart_interval: 1,
            density_unit: 0,
            x_density: 1,
            y_density: 1,
            saw_jfif_marker: true,
            adobe_transform: None,
        },
        None,
    )
    .expect("progressive source should encode");

    let transformed: Vec<u8> = transform_jpeg_with_options(
        &source,
        &TransformOptions {
            op: TransformOp::None,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    )
    .expect("Rust transform should succeed");

    let c_transformed: Vec<u8> =
        helpers::transform_with_c_jpegtran(&jpegtran, &source, &[], "progressive_restart");
    let (transformed_width, transformed_height, transformed_pixels) =
        helpers::decode_gray_with_c_djpeg(&djpeg, &transformed, "progressive_restart_transformed");
    let (c_width, c_height, c_pixels) =
        helpers::decode_gray_with_c_djpeg(&djpeg, &c_transformed, "progressive_restart_c");
    assert_eq!((transformed_width, transformed_height), (16, 8));
    assert_eq!((c_width, c_height), (16, 8));
    assert_eq!(transformed_pixels, c_pixels);
}

/// The P4-181 seed minus its op-selector byte: see
/// `progressive_source_with_bogus_adobe_transform_after_sos_matches_jpegtran`.
fn p4_181_source() -> Vec<u8> {
    decode_hex(
        r#"
        ffd8ffe000104a46494602010100000100010000ffdb0043000302020302020a03030304030304050805050404050a070706080c0a0c0c0b0a0b0b0d0e12100d0e110e0b0b1016101113141515150c0f171816141812141514ffdb004301030404050405
        09050509140d0b0d1414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414141414ffc20011080010001003012200021101031101ffc4001500010100000000000000000000000000000604
        ffc4001501010100000000000000000000000000000405ffda000c0301000210031000000152426116dbffc4001a1000010501000000000000000000000000ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        ffffffffffffffffffffffffffffe000104a4649ffee4600010100ffffffffffffffffffe000104a02ffd8ffffffffffffffffffffffffffffffe00010ffc24641464600ffd8ffe10002ffffffffffffffffffe10031457869660000687474703a2f28ff
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffe000104a4649ffee4600010100ffffffffffffffffffe000104a02ffd8ffffffffffffffffffffffffffffff
        ffe00010ffc24641464600ffd8ffe10002ffffffffffffffffffe10031457869660000687474703a2f28ffd8ffffffffffffffffffffffffffffee001341646f6265000bffffffff04feffffffffffffffffffffffffee001341646f6265000bffffffff
        04feffffffffffffffee001341ee00130bffffff00000000000000005bffffffffffee001341646f6265000bffffffff04feffffffffffffffffffffffffffffffffee001341646f6265000bffffffff04feffffffffffffffee001341ee00130bffffff
        00000000000000005bffffffffffee001341646f6265000bffffffff04feffffffffffffffffffffffffee001341001341646f6265000bffffffff04feffffffffffffffee001341ee00130bffffff00000000000000005bffffffffffee001341646f62
        65000bffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffd9ffffffffffffc4001810c662b070757b3fffd9
        "#,
    )
}

/// Overwrite the transform byte of the *last* `Adobe`-identified APP14 in
/// `jpeg` — the one libjpeg's `examine_app14` leaves in `Adobe_transform`.
fn with_last_adobe_transform(jpeg: &[u8], transform: u8) -> Vec<u8> {
    let mut out: Vec<u8> = jpeg.to_vec();
    let last_adobe: usize = out
        .windows(5)
        .rposition(|w| w == b"Adobe")
        .expect("source carries an Adobe APP14");
    // "Adobe"(5) version(2) flags0(2) flags1(2) transform(1).
    out[last_adobe + 11] = transform;
    out
}

/// P4-181 (Fuzz Smoke run 34042331788, `fuzz_transform_diff_c`, HFlip):
/// a 16x16 4:2:0 progressive source whose only scan is DC-first, followed
/// by a stray DHT, six non-JFIF APP0 segments, two Exif APP1 segments
/// and nine APP14 segments — five of them identified as `Adobe`, the
/// last of those carrying transform byte 255. libjpeg never recognises
/// the leading APP0 as JFIF (its identifier is `JFIF\x02`, not
/// `JFIF\0`), and every Adobe marker sits *after* the first SOS, so
/// `jpegtran` classifies the source as YCbCr from the component IDs and
/// writes a JFIF header. We used to copy the parsed transform byte into
/// a synthesized Adobe APP14 instead, so `djpeg` decoded our output with
/// "Unknown Adobe color transform code 255" (exit 2) while it decoded
/// jpegtran's cleanly.
///
/// The three MCU-alignment-free ops the fuzz target exercises must now be
/// byte-exact with `jpegtran -copy all`, which also pins the copied
/// marker set (non-JFIF APP0 segments are copied, exactly as
/// `jcopy_markers_execute` does) and the marker order (SOI, JFIF, copied
/// markers, tables).
#[test]
fn progressive_source_with_bogus_adobe_transform_after_sos_matches_jpegtran() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let source: Vec<u8> = p4_181_source();

    for (op, c_flag, label) in [
        (
            TransformOp::HFlip,
            ["-flip", "horizontal"],
            "bogus_adobe_hflip",
        ),
        (
            TransformOp::VFlip,
            ["-flip", "vertical"],
            "bogus_adobe_vflip",
        ),
        (
            TransformOp::Rot180,
            ["-rotate", "180"],
            "bogus_adobe_rot180",
        ),
    ] {
        let transformed: Vec<u8> = transform_jpeg_with_options(
            &source,
            &TransformOptions {
                op,
                copy_markers: MarkerCopyMode::All,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{label}: Rust transform failed: {e:?}"));
        let c_transformed: Vec<u8> = helpers::transform_with_c_jpegtran(
            &jpegtran,
            &source,
            &["-copy", "all", c_flag[0], c_flag[1]],
            label,
        );
        assert_eq!(
            transformed, c_transformed,
            "{label}: Rust output must be byte-exact with jpegtran -copy all"
        );

        // `decode_with_c_djpeg` asserts exit status 0 — djpeg exits 2 when
        // it decoded but warned, which is the failure this test pins.
        let (rust_width, rust_height, rust_pixels) =
            helpers::decode_with_c_djpeg(&djpeg, &transformed, &format!("{label}_rust"));
        let (c_width, c_height, c_pixels) =
            helpers::decode_with_c_djpeg(&djpeg, &c_transformed, &format!("{label}_c"));
        assert_eq!((rust_width, rust_height), (16, 16), "{label}");
        assert_eq!((c_width, c_height), (16, 16), "{label}");
        assert_eq!(rust_pixels, c_pixels, "{label}: djpeg pixels must agree");
    }
}

/// P4-181, second half: libjpeg classifies the colorspace once, when the
/// first SOS is reached (`default_decompress_parms`), so an Adobe APP14
/// that only appears *between* scans never changes `jpeg_color_space`.
/// Flipping the seed's last Adobe transform byte from 255 to 0 turns a
/// whole-stream classifier RGB (no Adobe marker precedes the SOS, so C
/// stays YCbCr); with the wrong answer `jpegtran` writes JFIF and we wrote
/// Adobe 0, and the two outputs decode to different pixels.
#[test]
fn adobe_transform_after_first_sos_does_not_change_classification() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let source: Vec<u8> = with_last_adobe_transform(&p4_181_source(), 0);

    // Transcode: byte-exact with jpegtran, which writes a JFIF header.
    let transformed: Vec<u8> = transform_jpeg_with_options(
        &source,
        &TransformOptions {
            op: TransformOp::HFlip,
            copy_markers: MarkerCopyMode::All,
            ..Default::default()
        },
    )
    .expect("Rust transform should succeed");
    let c_transformed: Vec<u8> = helpers::transform_with_c_jpegtran(
        &jpegtran,
        &source,
        &["-copy", "all", "-flip", "horizontal"],
        "post_sos_adobe0_hflip",
    );
    assert_eq!(
        transformed, c_transformed,
        "post-SOS Adobe transform 0 must not turn the header into Adobe/RGB"
    );

    // Decode: the same classification drives YCbCr->RGB conversion, so the
    // Rust decoder must agree with djpeg pixel for pixel. djpeg smooths a
    // DC-only progressive image by default (`do_block_smoothing`), so the
    // Rust side opts in too; classification is the only other variable.
    let mut decoder: Decoder<'_> = Decoder::new(&source).expect("Rust header parse");
    decoder.set_block_smoothing(true);
    let rust_image = decoder.decode_image().expect("Rust decode should succeed");
    let (c_width, c_height, c_pixels) =
        helpers::decode_with_c_djpeg(&djpeg, &source, "post_sos_adobe0_decode");
    assert_eq!((rust_image.width, rust_image.height), (c_width, c_height));
    assert_eq!(
        rust_image.data, c_pixels,
        "decoder must classify from the first-SOS marker state like djpeg"
    );
}
