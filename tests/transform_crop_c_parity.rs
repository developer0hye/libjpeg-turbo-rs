//! P4-240 (#675): `TransformOptions::crop` follows
//! `jtransform_request_workspace`'s crop validation (`transupp.c:1705-1757`),
//! cross-validated against `jpegtran -crop`.
//!
//! A zero `width` / `height` is jpegtran's omitted `W` / `H` — "to the edge",
//! libjpeg's `JCROP_UNSET`. A region jpegtran refuses ("Invalid crop request":
//! an origin outside the transformed image, or a region running past it) is
//! refused here too; the port used to clamp it to the image and succeed.

#![cfg(not(target_arch = "wasm32"))]

mod helpers;

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, CropRegion, JpegError, TransformOp, TransformOptions,
};

const PHOTO: &[u8] = include_bytes!("fixtures/photo_64x64_420.jpg");

fn crop(x: usize, y: usize, width: usize, height: usize) -> CropRegion {
    CropRegion {
        x,
        y,
        width,
        height,
    }
}

/// Every region, under a few operations, against jpegtran's outcome and bytes.
#[test]
fn crop_regions_match_jpegtran() {
    let jpegtran = require_c_tool!("jpegtran");
    let input = helpers::TempFile::new("p4240_crop_in.jpg");
    input.write_bytes(PHOTO);
    let cases: [(TransformOp, &[&str], CropRegion, &str); 12] = [
        // Zero extents: to the edge.
        (TransformOp::None, &[], crop(16, 16, 0, 0), "+16+16"),
        (TransformOp::None, &[], crop(16, 0, 0, 16), "x16+16+0"),
        (TransformOp::None, &[], crop(0, 16, 32, 0), "32+0+16"),
        (
            TransformOp::Rot90,
            &["-rotate", "90"],
            crop(16, 0, 0, 0),
            "+16+0",
        ),
        (
            TransformOp::HFlip,
            &["-flip", "horizontal"],
            crop(0, 32, 0, 0),
            "+0+32",
        ),
        // In range, and the same with an unaligned origin (aligned down).
        (TransformOp::None, &[], crop(32, 32, 32, 32), "32x32+32+32"),
        (TransformOp::None, &[], crop(8, 8, 16, 16), "16x16+8+8"),
        // Refused: past the right edge, an origin outside, past the bottom
        // after a rotation, and a zero extent from an origin outside.
        (TransformOp::None, &[], crop(48, 0, 32, 32), "32x32+48+0"),
        (TransformOp::None, &[], crop(64, 0, 0, 0), "+64+0"),
        (
            TransformOp::Rot90,
            &["-rotate", "90"],
            crop(0, 40, 32, 32),
            "32x32+0+40",
        ),
        (
            TransformOp::Transpose,
            &["-transpose"],
            crop(0, 64, 0, 0),
            "+0+64",
        ),
        // Larger than the frame under a transform: no expansion there.
        (
            TransformOp::Rot180,
            &["-rotate", "180"],
            crop(0, 0, 80, 80),
            "80x80+0+0",
        ),
    ];
    let mut refused: usize = 0;
    for (op, op_args, region, spec) in cases {
        let ours = transform_jpeg_with_options(
            PHOTO,
            &TransformOptions {
                op,
                crop: Some(region),
                ..TransformOptions::default()
            },
        );
        let output = helpers::TempFile::new("p4240_crop_out.jpg");
        let status = std::process::Command::new(&jpegtran)
            .args(["-copy", "all"])
            .args(op_args)
            .args(["-crop", spec, "-outfile"])
            .arg(output.path())
            .arg(input.path())
            .output()
            .expect("run jpegtran");
        let case: String = format!("{op:?} -crop {spec}");
        match ours {
            Ok(bytes) => {
                assert!(
                    status.status.success(),
                    "{case}: jpegtran refused, ours succeeded"
                );
                let c_bytes: Vec<u8> = std::fs::read(output.path()).expect("jpegtran output");
                assert!(bytes == c_bytes, "{case}: bytes differ from jpegtran");
            }
            Err(error) => {
                assert!(
                    !status.status.success(),
                    "{case}: ours refused ({error}), jpegtran succeeded"
                );
                assert!(
                    String::from_utf8_lossy(&status.stderr).contains("Invalid crop request"),
                    "{case}: jpegtran refused for another reason"
                );
                assert!(
                    matches!(&error, JpegError::InvalidCropRegion { reason } if reason == "Invalid crop request"),
                    "{case}: {error:?}"
                );
                refused += 1;
            }
        }
    }
    assert_eq!(refused, 5, "the five regions jpegtran refuses");
}
