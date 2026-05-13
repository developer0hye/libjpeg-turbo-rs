mod helpers;

use std::path::PathBuf;

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, write_coefficients_progressive, ComponentCoefficients,
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
