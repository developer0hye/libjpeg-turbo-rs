//! S411/S441 transform cross-validation against C jpegtran (#146).
//!
//! Verifies all 8 transform ops produce pixel-identical output to C jpegtran
//! for S411 (h_factor=4) and S441 (v_factor=4) subsampling modes.
mod helpers;

use libjpeg_turbo_rs::{compress, transform, PixelFormat, Subsampling, TransformOp};

fn assert_all_transforms_match_c(subsamp: Subsampling, sname: &str) {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    let w: usize = 48;
    let h: usize = 48;
    let pixels: Vec<u8> = helpers::generate_gradient(w, h);
    let jpeg: Vec<u8> = compress(&pixels, w, h, PixelFormat::Rgb, 90, subsamp).unwrap();

    let ops: &[(TransformOp, &str, &[&str])] = &[
        (TransformOp::None, "none", &[]),
        (TransformOp::HFlip, "hflip", &["-flip", "horizontal"]),
        (TransformOp::VFlip, "vflip", &["-flip", "vertical"]),
        (TransformOp::Transpose, "transpose", &["-transpose"]),
        (TransformOp::Transverse, "transverse", &["-transverse"]),
        (TransformOp::Rot90, "rot90", &["-rotate", "90"]),
        (TransformOp::Rot180, "rot180", &["-rotate", "180"]),
        (TransformOp::Rot270, "rot270", &["-rotate", "270"]),
    ];

    for &(op, name, c_args) in ops {
        let label: String = format!("{}_{}", sname, name);
        let rust_out: Vec<u8> =
            transform(&jpeg, op).unwrap_or_else(|e| panic!("{}: transform failed: {:?}", label, e));
        let c_out: Vec<u8> = helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, c_args, &label);

        let (rw, rh, r_rgb) =
            helpers::decode_with_c_djpeg(&djpeg, &rust_out, &format!("{}_rust", label));
        let (cw, ch, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &c_out, &format!("{}_c", label));

        assert_eq!(rw, cw, "{}: width mismatch", label);
        assert_eq!(rh, ch, "{}: height mismatch", label);
        helpers::assert_pixels_identical(&r_rgb, &c_rgb, rw, rh, 3, &label);
    }
}

#[test]
fn c_xval_s411_all_transforms() {
    assert_all_transforms_match_c(Subsampling::S411, "S411");
}

#[test]
fn c_xval_s441_all_transforms() {
    assert_all_transforms_match_c(Subsampling::S441, "S441");
}
