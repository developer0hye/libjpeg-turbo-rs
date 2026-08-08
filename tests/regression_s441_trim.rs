//! Issue #439 / P4-117: `trim` rejected 4:4:1 images shorter than one iMCU row.
//!
//! 4:4:1 is `h_samp=1, v_samp=4`, so its iMCU is 8x32 — unusually tall. A 27-row
//! image contains **zero** whole iMCU rows, and the trim path computed
//! `(27 / 32) * 32 == 0` and rejected the whole transform with "trim would
//! remove all image data".
//!
//! Upstream never errors here. `trim_right_edge` and `trim_bottom_edge`
//! (transupp.c:1570-1592) each open with `if (MCU_cols > 0 && …)` /
//! `if (MCU_rows > 0 && …)`, so an axis with less than one whole iMCU is simply
//! left untrimmed. `jpegtran -trim -flip vertical` on this exact input returns
//! the full 35x27.
//!
//! These tests pin the geometry for every operation and cross-validate the
//! pixels against stock `jpegtran`, feeding both implementations the *same*
//! input JPEG so only the transform is under comparison.

mod helpers;

use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{
    compress, decompress, transform_jpeg_with_options, PixelFormat, Subsampling, TransformOp,
    TransformOptions,
};

/// Non-MCU-aligned on both axes for 4:4:1: 35 is not a multiple of 8, and 27 is
/// less than one 32-row iMCU. Matches the fixture `cross_product_transform`
/// uses, so the eight cases it had to exclude are the eight covered here.
const W: usize = 35;
const H: usize = 27;

fn source_jpeg() -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        for x in 0..W {
            pixels.push(((x * 7) % 256) as u8);
            pixels.push(((y * 9) % 256) as u8);
            pixels.push((((x + y) * 5) % 256) as u8);
        }
    }
    compress(&pixels, W, H, PixelFormat::Rgb, 90, Subsampling::S441)
        .expect("encode 4:4:1 source fixture")
}

/// Every transform, with the dimensions stock `jpegtran -trim` produces for
/// this input. Measured against libjpeg-turbo 3.1.4.1; the reasoning for each
/// is that an axis is only trimmed when at least one whole iMCU fits, and
/// transposing ops swap which source axis feeds which output axis.
///
/// iMCU is 8 wide x 32 tall. Source width 35 holds four whole iMCUs (→ 32);
/// source height 27 holds none (→ left at 27).
const EXPECTED: &[(TransformOp, &str, usize, usize)] = &[
    // Trims the source width only.
    (TransformOp::HFlip, "hflip", 32, 27),
    // Trims the source height only — which the guard leaves alone.
    (TransformOp::VFlip, "vflip", 35, 27),
    // Trims neither: transpose never trims (transupp.c:1873).
    (TransformOp::Transpose, "transpose", 27, 35),
    // Output width comes from source height (untrimmable), so nothing changes.
    (TransformOp::Rot90, "rot90", 27, 35),
    // Both axes; only the width has a whole iMCU to keep.
    (TransformOp::Rot180, "rot180", 32, 27),
    // Output height comes from source width, which trims to 32.
    (TransformOp::Rot270, "rot270", 27, 32),
    // Both: source width trims to 32 (becomes output height), source height
    // is left at 27 (becomes output width).
    (TransformOp::Transverse, "transverse", 27, 32),
];

/// Issue #439: no 4:4:1 trim transform may be rejected, and each must produce
/// the same geometry stock `jpegtran -trim` does.
#[test]
fn issue_439_s441_trim_matches_jpegtran_geometry() {
    let jpeg: Vec<u8> = source_jpeg();
    for &(op, name, want_w, want_h) in EXPECTED {
        let opts: TransformOptions = TransformOptions {
            op,
            trim: true,
            ..TransformOptions::default()
        };
        let out: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).unwrap_or_else(|e| {
            panic!(
                "issue #439: `{name}` with trim=true on a {W}x{H} 4:4:1 image was \
                 rejected: {e}"
            )
        });
        let img = decompress(&out)
            .unwrap_or_else(|e| panic!("`{name}`: trimmed output does not decode: {e}"));
        assert_eq!(
            (img.width, img.height),
            (want_w, want_h),
            "`{name}`: trimmed geometry diverges from stock jpegtran -trim"
        );
    }
}

/// The same matrix, cross-validated pixel-for-pixel against stock `jpegtran`.
///
/// Both sides transform the *same* source JPEG, so any difference is in the
/// transform rather than in the encoder that produced the input.
#[test]
fn issue_439_s441_trim_pixels_match_c_jpegtran() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let jpeg: Vec<u8> = source_jpeg();
    let src: helpers::TempFile = helpers::TempFile::new("issue439_src.jpg");
    src.write_bytes(&jpeg);

    // Exact accounting: every operation in the table must reach the pixel
    // comparison, so a future early-exit cannot shrink this silently.
    let mut compared: usize = 0;

    for &(op, name, want_w, want_h) in EXPECTED {
        let c_args: &[&str] = match op {
            TransformOp::HFlip => &["-flip", "horizontal"],
            TransformOp::VFlip => &["-flip", "vertical"],
            TransformOp::Transpose => &["-transpose"],
            TransformOp::Rot90 => &["-rotate", "90"],
            TransformOp::Rot180 => &["-rotate", "180"],
            TransformOp::Rot270 => &["-rotate", "270"],
            TransformOp::Transverse => &["-transverse"],
            TransformOp::None => unreachable!("None is not in EXPECTED"),
        };

        let c_out: helpers::TempFile = helpers::TempFile::new(&format!("issue439_c_{name}.jpg"));
        let status = Command::new(&jpegtran)
            .arg("-trim")
            .args(c_args)
            .arg("-outfile")
            .arg(c_out.path())
            .arg(src.path())
            .output()
            .unwrap_or_else(|e| panic!("`{name}`: failed to run jpegtran: {e}"));
        assert!(
            status.status.success(),
            "`{name}`: jpegtran -trim failed: {}",
            String::from_utf8_lossy(&status.stderr)
        );

        let opts: TransformOptions = TransformOptions {
            op,
            trim: true,
            ..TransformOptions::default()
        };
        let rust_bytes: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts)
            .unwrap_or_else(|e| panic!("`{name}`: Rust transform failed: {e}"));
        let rust_out: helpers::TempFile =
            helpers::TempFile::new(&format!("issue439_rust_{name}.jpg"));
        rust_out.write_bytes(&rust_bytes);

        // Decode both through the same C decoder so only the transform differs.
        let decode = |path: &std::path::Path, which: &str| -> (usize, usize, Vec<u8>) {
            let out = Command::new(&djpeg)
                .arg("-ppm")
                .arg(path)
                .output()
                .unwrap_or_else(|e| panic!("`{name}`: failed to run djpeg on {which}: {e}"));
            assert!(
                out.status.success(),
                "`{name}`: djpeg rejected the {which} output: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            helpers::parse_ppm(&out.stdout)
                .unwrap_or_else(|| panic!("`{name}`: could not parse djpeg PPM for {which}"))
        };

        let (cw, ch, c_pixels) = decode(c_out.path(), "C");
        let (rw, rh, rust_pixels) = decode(rust_out.path(), "Rust");

        assert_eq!(
            (rw, rh),
            (cw, ch),
            "`{name}`: dimensions diverge — Rust {rw}x{rh}, C {cw}x{ch}"
        );
        assert_eq!(
            (rw, rh),
            (want_w, want_h),
            "`{name}`: both sides agree but on the wrong geometry"
        );
        let max_diff: u8 = helpers::pixel_max_diff(&rust_pixels, &c_pixels);
        assert_eq!(
            max_diff, 0,
            "`{name}`: trimmed pixels differ from stock jpegtran (max_diff={max_diff})"
        );
        compared += 1;
    }

    assert_eq!(
        compared,
        EXPECTED.len(),
        "every operation must reach the pixel comparison"
    );
}

/// The guard must not weaken trimming where a whole iMCU *does* fit: 4:2:0 on
/// the same fixture still trims both axes, exactly as before.
#[test]
fn issue_439_guard_does_not_disable_trimming_where_it_applies() {
    let mut pixels: Vec<u8> = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        for x in 0..W {
            pixels.push(((x * 3) % 256) as u8);
            pixels.push(((y * 5) % 256) as u8);
            pixels.push((((x * y) % 251) % 256) as u8);
        }
    }
    // 4:2:0 has a 16x16 iMCU, so 35x27 holds two whole iMCUs across and one
    // down: trimming must still shrink both axes to 32x16.
    let jpeg: Vec<u8> = compress(&pixels, W, H, PixelFormat::Rgb, 90, Subsampling::S420)
        .expect("encode 4:2:0 fixture");
    let opts: TransformOptions = TransformOptions {
        op: TransformOp::Rot180,
        trim: true,
        ..TransformOptions::default()
    };
    let out: Vec<u8> = transform_jpeg_with_options(&jpeg, &opts).expect("4:2:0 rot180 trim");
    let img = decompress(&out).expect("decode trimmed 4:2:0");
    assert_eq!(
        (img.width, img.height),
        (32, 16),
        "the P4-117 guard must only fire when an axis holds no whole iMCU"
    );
}
