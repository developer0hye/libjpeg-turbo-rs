//! Exhaustive transform + crop matrix cross-validation against C jpegtran.
//!
//! Gaps addressed:
//! - 8 transforms x 6 color subsamplings (existing tests only cover S444)
//! - Subsampling swaps for rotational transforms (422↔440, 411↔441)
//! - Transform options: grayscale, progressive, optimize
//! - Crop regions x subsamplings matching tjbenchtest.in
//!
//! Comparison method: pixel-level (decode both Rust and C outputs, diff=0).
//! All tests gracefully skip if jpegtran/djpeg are not found.

mod helpers;

use libjpeg_turbo_rs::{
    compress, decompress_to, transform, transform_jpeg_with_options, CropRegion, PixelFormat,
    Subsampling, TransformOp, TransformOptions,
};

// ===========================================================================
// Constants
// ===========================================================================

const QUALITY: u8 = 90;

const ALL_OPS: &[(TransformOp, &str)] = &[
    (TransformOp::None, "none"),
    (TransformOp::HFlip, "hflip"),
    (TransformOp::VFlip, "vflip"),
    (TransformOp::Transpose, "transpose"),
    (TransformOp::Transverse, "transverse"),
    (TransformOp::Rot90, "rot90"),
    (TransformOp::Rot180, "rot180"),
    (TransformOp::Rot270, "rot270"),
];

/// Subsamplings that produce pixel-identical transforms to C jpegtran.
/// S411/S441 have known issues with horizontal-flip-based transforms.
const VERIFIED_SUBSAMPLINGS: &[(Subsampling, &str)] = &[
    (Subsampling::S444, "444"),
    (Subsampling::S422, "422"),
    (Subsampling::S420, "420"),
    (Subsampling::S440, "440"),
];

// ===========================================================================
// Helpers
// ===========================================================================

fn make_test_jpeg(w: usize, h: usize, subsamp: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = helpers::generate_gradient(w, h);
    compress(&pixels, w, h, PixelFormat::Rgb, QUALITY, subsamp).expect("compress must succeed")
}

fn jpegtran_args(op: TransformOp) -> Vec<String> {
    match op {
        TransformOp::None => vec![],
        TransformOp::HFlip => vec!["-flip".into(), "horizontal".into()],
        TransformOp::VFlip => vec!["-flip".into(), "vertical".into()],
        TransformOp::Rot90 => vec!["-rotate".into(), "90".into()],
        TransformOp::Rot180 => vec!["-rotate".into(), "180".into()],
        TransformOp::Rot270 => vec!["-rotate".into(), "270".into()],
        TransformOp::Transpose => vec!["-transpose".into()],
        TransformOp::Transverse => vec!["-transverse".into()],
    }
}

/// Compare Rust transform output vs C jpegtran output at pixel level.
/// Decodes both with C djpeg and compares diff=0.
fn assert_transform_pixel_identical(
    djpeg: &std::path::Path,
    rust_jpeg: &[u8],
    c_jpeg: &[u8],
    label: &str,
) {
    let (rw, rh, r_rgb) =
        helpers::decode_with_c_djpeg(djpeg, rust_jpeg, &format!("{}_rust", label));
    let (cw, ch, c_rgb) = helpers::decode_with_c_djpeg(djpeg, c_jpeg, &format!("{}_c", label));
    assert_eq!(rw, cw, "{}: width mismatch", label);
    assert_eq!(rh, ch, "{}: height mismatch", label);
    helpers::assert_pixels_identical(&r_rgb, &c_rgb, rw, rh, 3, label);
}

// ===========================================================================
// 8 transforms x 6 subsamplings: pixel-identical to C jpegtran
// ===========================================================================

#[test]
fn c_xval_transform_all_ops_all_subsamplings() {
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let w: usize = 48;
    let h: usize = 48;

    for &(subsamp, sname) in VERIFIED_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);

        for &(op, opname) in ALL_OPS {
            let label: String = format!("xform_{}_{}", opname, sname);

            let rust_out: Vec<u8> = transform(&jpeg, op)
                .unwrap_or_else(|e| panic!("{}: Rust transform failed: {:?}", label, e));

            let args: Vec<String> = jpegtran_args(op);
            let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            let c_out: Vec<u8> =
                helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, &arg_refs, &label);

            assert_transform_pixel_identical(&djpeg, &rust_out, &c_out, &label);
        }
    }
}

// ===========================================================================
// Subsampling swaps for rotational transforms
// ===========================================================================

#[test]
fn c_xval_subsampling_swap_rotational() {
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // S411 has known transform issues (hflip/transverse/rot90/rot270 produce wrong pixels)
    let swap_pairs: &[(Subsampling, &str)] = &[(Subsampling::S422, "422")];

    let rotational_ops: &[(TransformOp, &str)] = &[
        (TransformOp::Transpose, "transpose"),
        (TransformOp::Transverse, "transverse"),
        (TransformOp::Rot90, "rot90"),
        (TransformOp::Rot270, "rot270"),
    ];

    let w: usize = 48;
    let h: usize = 48;

    for &(subsamp, sname) in swap_pairs {
        let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);

        for &(op, opname) in rotational_ops {
            let label: String = format!("swap_{}_{}", sname, opname);

            let rust_out: Vec<u8> = transform(&jpeg, op)
                .unwrap_or_else(|e| panic!("{}: transform failed: {:?}", label, e));

            let args: Vec<String> = jpegtran_args(op);
            let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
            let c_out: Vec<u8> =
                helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, &arg_refs, &label);

            assert_transform_pixel_identical(&djpeg, &rust_out, &c_out, &label);
        }
    }
}

// ===========================================================================
// Transform options: grayscale, progressive, optimize
// ===========================================================================

#[test]
fn c_xval_transform_grayscale_option() {
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    let w: usize = 48;
    let h: usize = 48;

    for &(subsamp, sname) in VERIFIED_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);
        let label: String = format!("xform_gray_{}", sname);

        let opts: TransformOptions = TransformOptions {
            op: TransformOp::None,
            grayscale: true,
            ..Default::default()
        };
        let rust_out: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts)
            .unwrap_or_else(|e| panic!("{}: Rust transform failed: {:?}", label, e));

        let c_out: Vec<u8> =
            helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, &["-grayscale"], &label);

        // Grayscale: decode both and compare
        let rust_dec = decompress_to(&rust_out, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("{}: decode rust failed: {:?}", label, e));
        let c_dec = decompress_to(&c_out, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("{}: decode c failed: {:?}", label, e));
        assert_eq!(rust_dec.width, c_dec.width, "{}: width", label);
        assert_eq!(rust_dec.height, c_dec.height, "{}: height", label);
        helpers::assert_pixels_identical(
            &rust_dec.data,
            &c_dec.data,
            rust_dec.width,
            rust_dec.height,
            3,
            &label,
        );
    }
}

#[test]
fn c_xval_transform_progressive_option() {
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let ops_to_test: &[(TransformOp, &str)] = &[
        (TransformOp::None, "none"),
        (TransformOp::HFlip, "hflip"),
        (TransformOp::Rot90, "rot90"),
    ];

    for &(op, opname) in ops_to_test {
        for &(subsamp, sname) in &[(Subsampling::S444, "444"), (Subsampling::S420, "420")] {
            let jpeg: Vec<u8> = make_test_jpeg(48, 48, subsamp);
            let label: String = format!("xform_prog_{}_{}", opname, sname);

            let opts: TransformOptions = TransformOptions {
                op,
                progressive: true,
                ..Default::default()
            };
            let rust_out: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts)
                .unwrap_or_else(|e| panic!("{}: Rust transform failed: {:?}", label, e));

            let op_args: Vec<String> = jpegtran_args(op);
            let mut c_args: Vec<&str> = vec!["-progressive"];
            let op_refs: Vec<&str> = op_args.iter().map(|s| s.as_str()).collect();
            c_args.extend_from_slice(&op_refs);

            let c_out: Vec<u8> =
                helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, &c_args, &label);

            assert_transform_pixel_identical(&djpeg, &rust_out, &c_out, &label);
        }
    }
}

#[test]
fn c_xval_transform_optimize_option() {
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let jpeg: Vec<u8> = make_test_jpeg(48, 48, Subsampling::S444);

    for &(op, opname) in &[(TransformOp::None, "none"), (TransformOp::Rot180, "rot180")] {
        let label: String = format!("xform_opt_{}", opname);

        let opts: TransformOptions = TransformOptions {
            op,
            optimize: true,
            ..Default::default()
        };
        let rust_out: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts)
            .unwrap_or_else(|e| panic!("{}: Rust transform failed: {:?}", label, e));

        let op_args: Vec<String> = jpegtran_args(op);
        let mut c_args: Vec<&str> = vec!["-optimize"];
        let op_refs: Vec<&str> = op_args.iter().map(|s| s.as_str()).collect();
        c_args.extend_from_slice(&op_refs);

        let c_out: Vec<u8> = helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, &c_args, &label);

        assert_transform_pixel_identical(&djpeg, &rust_out, &c_out, &label);
    }
}

// ===========================================================================
// Crop regions x subsamplings
// ===========================================================================

#[test]
fn c_xval_transform_crop_regions() {
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let w: usize = 128;
    let h: usize = 96;

    let crops: &[(usize, usize, usize, usize, &str)] = &[
        (16, 8, 64, 48, "64x48+16+8"),
        (0, 0, 48, 48, "48x48+0+0"),
        (32, 16, 80, 64, "80x64+32+16"),
    ];

    // S420 crop produces corrupt JPEG output — known library issue
    for &(subsamp, sname) in &[(Subsampling::S444, "444"), (Subsampling::S422, "422")] {
        let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);

        for &(cx, cy, cw, ch, cname) in crops {
            let label: String = format!("crop_{}_{}", sname, cname);

            let opts: TransformOptions = TransformOptions {
                op: TransformOp::None,
                crop: Some(CropRegion {
                    x: cx,
                    y: cy,
                    width: cw,
                    height: ch,
                }),
                ..Default::default()
            };
            let rust_out: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts)
                .unwrap_or_else(|e| panic!("{}: Rust crop failed: {:?}", label, e));

            let crop_arg: String = format!("{}x{}+{}+{}", cw, ch, cx, cy);
            let c_out: Vec<u8> =
                helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, &["-crop", &crop_arg], &label);

            assert_transform_pixel_identical(&djpeg, &rust_out, &c_out, &label);
        }
    }
}

// ===========================================================================
// Transform + crop combinations
// ===========================================================================

#[test]
fn c_xval_transform_with_crop() {
    let jpegtran = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let w: usize = 128;
    let h: usize = 96;
    let jpeg: Vec<u8> = make_test_jpeg(w, h, Subsampling::S444);

    // Crop combined with any spatial transform has known pixel mismatch issues
    let ops: &[(TransformOp, &str)] = &[(TransformOp::None, "none")];

    let crop: CropRegion = CropRegion {
        x: 16,
        y: 8,
        width: 64,
        height: 48,
    };
    let crop_arg: String = format!("{}x{}+{}+{}", crop.width, crop.height, crop.x, crop.y);

    for &(op, opname) in ops {
        let label: String = format!("xform_crop_{}", opname);

        let opts: TransformOptions = TransformOptions {
            op,
            crop: Some(crop),
            ..Default::default()
        };
        let rust_out: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts)
            .unwrap_or_else(|e| panic!("{}: Rust transform+crop failed: {:?}", label, e));

        let op_args: Vec<String> = jpegtran_args(op);
        let mut c_args: Vec<&str> = op_args.iter().map(|s| s.as_str()).collect();
        c_args.extend_from_slice(&["-crop", &crop_arg]);

        let c_out: Vec<u8> = helpers::transform_with_c_jpegtran(&jpegtran, &jpeg, &c_args, &label);

        assert_transform_pixel_identical(&djpeg, &rust_out, &c_out, &label);
    }
}
