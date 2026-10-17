//! P4-251: the 12-bit decoders counted MCUs in a `u16`.
//!
//! `precision::decompress_12bit` and `raw_data_12::decompress_raw_12` kept
//! their MCU counter in a `u16`, so a frame of more than 65,535 MCUs
//! overflowed it. Debug builds panicked ("attempt to add with overflow"); release
//! builds wrapped, and the restart check then ran on the wrapped count,
//! misplacing every restart after the wrap unless the interval divided 65,536.
//! A 2056×2056 12-bit greyscale frame has 257 × 257 = 66,049 MCUs; with one
//! restart interval per MCU row (257 MCUs, which does not divide 65,536) the
//! release build decoded one MCU row wrong and returned `Ok`.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::precision::{decompress_12bit, Image12};
use libjpeg_turbo_rs::raw_data_12::{decompress_raw_12, RawImage12};

const SIDE: usize = 2056;

/// A 12-bit greyscale PGM: gradients plus fine texture, so a misplaced
/// restart (which resets the DC predictors) changes many samples.
fn pgm_12bit(width: usize, height: usize) -> Vec<u8> {
    let mut pgm: Vec<u8> = format!("P5\n{width} {height}\n4095\n").into_bytes();
    for y in 0..height {
        for x in 0..width {
            let value: usize =
                (x * 4095 / (width - 1) + y * 1777 / (height - 1) + (x * 7 + y * 13) % 97 * 9)
                    % 4096;
            pgm.extend_from_slice(&(value as u16).to_be_bytes());
        }
    }
    pgm
}

fn cjpeg_supports_precision(cjpeg: &Path) -> bool {
    Command::new(cjpeg)
        .arg("-help")
        .output()
        .is_ok_and(|output| {
            String::from_utf8_lossy(&output.stderr).contains("precision")
                || String::from_utf8_lossy(&output.stdout).contains("precision")
        })
}

/// P4-251: a 12-bit frame past 65,535 MCUs, with a restart interval that does
/// not divide 65,536, decodes identically to `djpeg` through both 12-bit
/// entry points — no overflow panic in debug builds, no misplaced restart in
/// release builds.
#[test]
fn twelve_bit_frame_past_65535_mcus_with_restarts_matches_djpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    if !cjpeg_supports_precision(&cjpeg) {
        helpers::skip_missing_c_capability("cjpeg", "-precision 12");
        return;
    }
    let jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
        &cjpeg,
        &pgm_12bit(SIDE, SIDE),
        &["-precision", "12", "-quality", "95", "-restart", "1"],
        "p4_251_twelve_bit_restarts",
    );
    let (width, height, components, maxval, expected): (usize, usize, usize, usize, Vec<i16>) =
        helpers::decode_with_c_djpeg_i16(&djpeg, &jpeg, "p4_251_twelve_bit_restarts");
    assert_eq!(
        (width, height, components, maxval),
        (SIDE, SIDE, 1, 4095),
        "djpeg must decode the frame as 12-bit greyscale"
    );

    let image: Image12 = decompress_12bit(&jpeg).expect("decompress_12bit");
    assert_eq!(
        (image.width, image.height, image.num_components),
        (SIDE, SIDE, 1)
    );
    let differing: usize = image
        .data
        .iter()
        .zip(&expected)
        .filter(|(a, b)| a != b)
        .count();
    assert_eq!(
        differing,
        0,
        "decompress_12bit: {differing} of {} samples differ from djpeg",
        expected.len()
    );

    let raw: RawImage12 = decompress_raw_12(&jpeg).expect("decompress_raw_12");
    let stride: usize = raw.plane_widths[0];
    let plane: Vec<i16> = (0..SIDE)
        .flat_map(|y| raw.planes[0][y * stride..y * stride + SIDE].iter().copied())
        .collect();
    let differing: usize = plane.iter().zip(&expected).filter(|(a, b)| a != b).count();
    assert_eq!(
        differing,
        0,
        "decompress_raw_12: {differing} of {} samples differ from djpeg",
        expected.len()
    );
}
