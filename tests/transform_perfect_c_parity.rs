//! P4-227 (#655): `TransformOptions::perfect` follows upstream's
//! `jtransform_perfect_transform` (`transupp.c:2415-2450`), cross-validated
//! against `jpegtran -perfect`.
//!
//! Upstream tests only the edges an operation moves: the right edge for a
//! horizontal flip and a 270-degree rotation, the bottom edge for a vertical
//! flip and a 90-degree rotation, both for transverse and 180. The port tested
//! both edges for every rotation, so `-perfect -rotate 90` on a frame whose
//! width alone was ragged was refused where jpegtran and `tj3Transform` accept
//! it. A grayscale output uses an 8x8 iMCU (`transupp.c:1631-1655`).

#![cfg(not(target_arch = "wasm32"))]

mod helpers;

use libjpeg_turbo_rs::{
    compress, transform_jpeg_with_options, PixelFormat, Subsampling, TransformOp, TransformOptions,
};

fn source(width: usize, height: usize, subsampling: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i / 3) % width * 3 + i / (3 * width) * 5) as u8)
        .collect();
    compress(&pixels, width, height, PixelFormat::Rgb, 85, subsampling).expect("encode")
}

/// Every `-perfect` operation, with and without `-grayscale`, on frames whose
/// width, height, or both are ragged for their iMCU: the outcome — refused, or
/// the exact bytes — must equal jpegtran's.
#[test]
fn perfect_matches_jpegtran_for_every_operation() {
    let jpegtran = require_c_tool!("jpegtran");
    let ops: [(TransformOp, &[&str]); 7] = [
        (TransformOp::HFlip, &["-flip", "horizontal"]),
        (TransformOp::VFlip, &["-flip", "vertical"]),
        (TransformOp::Rot90, &["-rotate", "90"]),
        (TransformOp::Rot180, &["-rotate", "180"]),
        (TransformOp::Rot270, &["-rotate", "270"]),
        (TransformOp::Transpose, &["-transpose"]),
        (TransformOp::Transverse, &["-transverse"]),
    ];
    let mut compared: usize = 0;
    let mut accepted_by_both: usize = 0;
    for (width, height, subsampling) in [
        (76, 64, Subsampling::S444),
        (64, 76, Subsampling::S444),
        (72, 64, Subsampling::S420),
        (64, 72, Subsampling::S420),
    ] {
        let jpeg: Vec<u8> = source(width, height, subsampling);
        let input = helpers::TempFile::new("p4227_perfect_in.jpg");
        input.write_bytes(&jpeg);
        for (op, op_args) in ops {
            for grayscale in [false, true] {
                let ours = transform_jpeg_with_options(
                    &jpeg,
                    &TransformOptions {
                        op,
                        perfect: true,
                        grayscale,
                        ..TransformOptions::default()
                    },
                );
                let mut args: Vec<&str> = vec!["-copy", "all", "-perfect"];
                if grayscale {
                    args.push("-grayscale");
                }
                args.extend_from_slice(op_args);
                let output = helpers::TempFile::new("p4227_perfect_out.jpg");
                let status = std::process::Command::new(&jpegtran)
                    .args(&args)
                    .arg("-outfile")
                    .arg(output.path())
                    .arg(input.path())
                    .output()
                    .expect("run jpegtran");
                let case: String =
                    format!("{width}x{height} {subsampling:?} {op:?} grayscale={grayscale}");
                assert_eq!(
                    ours.is_ok(),
                    status.status.success(),
                    "{case}: ours {:?}, jpegtran {}",
                    ours.as_ref().map(Vec::len),
                    String::from_utf8_lossy(&status.stderr)
                );
                if let Ok(bytes) = ours {
                    let c_bytes: Vec<u8> = std::fs::read(output.path()).expect("jpegtran output");
                    assert!(bytes == c_bytes, "{case}: bytes differ from jpegtran");
                    accepted_by_both += 1;
                }
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 56);
    // Some of each: a vacuous all-refused or all-accepted run proves nothing.
    assert!(accepted_by_both > 0 && accepted_by_both < compared);
}

/// P4-227 codex review: `-grayscale` shrinks the iMCU to 8x8 only for a
/// YCbCr source. A 72-wide 4:2:0 *RGB-colour-space* frame keeps its 16-pixel
/// iMCU, so `-perfect -flip horizontal -grayscale` is refused, by jpegtran and
/// by the port alike; the same request on a YCbCr frame succeeds.
#[test]
fn perfect_grayscale_shrinks_the_imcu_only_for_ycbcr() {
    let jpegtran = require_c_tool!("jpegtran");
    let (width, height): (usize, usize) = (72, 64);
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 253) as u8).collect();
    let rgb: Vec<u8> = libjpeg_turbo_rs::Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .colorspace(libjpeg_turbo_rs::ColorSpace::Rgb)
        .subsampling(Subsampling::S420)
        .encode()
        .expect("encode an RGB-colour-space 4:2:0 frame");
    assert_eq!(
        libjpeg_turbo_rs::Decoder::new(&rgb)
            .expect("header")
            .jpeg_color_space(),
        libjpeg_turbo_rs::ColorSpace::Rgb
    );
    let ycbcr: Vec<u8> = source(width, height, Subsampling::S420);
    for (label, jpeg, accepted) in [("rgb", &rgb, false), ("ycbcr", &ycbcr, true)] {
        let ours = transform_jpeg_with_options(
            jpeg,
            &TransformOptions {
                op: TransformOp::HFlip,
                perfect: true,
                grayscale: true,
                ..TransformOptions::default()
            },
        );
        let input = helpers::TempFile::new("p4227_gray_perfect_in.jpg");
        input.write_bytes(jpeg);
        let output = helpers::TempFile::new("p4227_gray_perfect_out.jpg");
        let status = std::process::Command::new(&jpegtran)
            .args(["-perfect", "-grayscale", "-flip", "horizontal", "-outfile"])
            .arg(output.path())
            .arg(input.path())
            .output()
            .expect("run jpegtran");
        assert_eq!(status.status.success(), accepted, "{label}: jpegtran");
        assert_eq!(ours.is_ok(), accepted, "{label}: ours {ours:?}");
    }
}
