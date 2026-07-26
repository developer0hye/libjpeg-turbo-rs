//! Issues #313, #339, #340: the CMYK encode path had no C reference at all.
//!
//! `cjpeg` reads PNM, BMP, GIF and Targa — none of which carry CMYK — so the
//! four-component path was the one encode path that could only ever be
//! compared against itself. Three defects lived there undisturbed:
//!
//! - a JFIF APP0 marker C never writes, 18 bytes in every CMYK file (#339),
//! - component IDs `1,2,3,4` where libjpeg writes `'C','M','Y','K'` (#339),
//! - bottom padding that clamped the last row where C repeats the last row
//!   group, wrong for every subsampling with `v_samp > 1` (#340),
//!
//! plus `optimize_huffman` rejected outright and `smoothing_factor` ignored
//! (#313), both of which C applies to CMYK like any other colorspace.
//!
//! `examples/cmyk_encode_c_oracle.c` supplies the missing oracle by driving
//! libjpeg directly. `tests/helpers/c_oracle.rs` compiles it on demand against
//! whatever libjpeg development install it can find, or honours a prebuilt
//! binary named by `CMYK_C_ORACLE`.

mod helpers;

use libjpeg_turbo_rs::encode::pipeline::{
    compress_optimized_with_params, compress_with_params, CompressParams,
};
use libjpeg_turbo_rs::{decompress_to, DctMethod, PixelFormat, Subsampling};
use std::path::PathBuf;

/// Deterministic CMYK content with a hard-edged region, so quantization and
/// entropy-coding differences are observable rather than washed out.
fn cmyk_image(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * 4];
    let mut rng_state: u32 = 0x9e37_79b9;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let in_rect: bool = x * 3 >= width && x * 3 < width * 2;
            let offset: usize = (y * width + x) * 4;
            pixels[offset] = ((x * 255 / width.max(1)) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 1] = ((y * 255 / height.max(1)) as i32 - noise).clamp(0, 255) as u8;
            pixels[offset + 2] = (if in_rect { 220 } else { 40 } + noise).clamp(0, 255) as u8;
            pixels[offset + 3] = ((x + y) % 256) as u8;
        }
    }
    pixels
}

/// The subsamplings CMYK can legally carry: components 0 and 3 both take the
/// luma factors, so `2 * h * v + 2` must stay inside the 10-block MCU cap.
const SUBSAMPLINGS: &[(Subsampling, usize, usize)] = &[
    (Subsampling::S444, 1, 1),
    (Subsampling::S422, 2, 1),
    (Subsampling::S440, 1, 2),
    (Subsampling::S420, 2, 2),
];

/// Geometries chosen for their MCU residues: 32x2 and 33x18 have heights that
/// are multiples of `v_samp` but not of the MCU height, which is the exact
/// shape #340 got wrong; 1x1 and 7x16 pin the degenerate cases.
const GEOMETRIES: &[(usize, usize)] = &[
    (64, 48),
    (17, 17),
    (32, 2),
    (7, 16),
    (33, 18),
    (1, 1),
    (48, 48),
];

const QUALITIES: &[u8] = &[25, 75, 95];

fn oracle_args(
    width: usize,
    height: usize,
    quality: u8,
    h_samp: usize,
    v_samp: usize,
    extra: &[&str],
) -> Vec<String> {
    let mut args: Vec<String> = vec![
        width.to_string(),
        height.to_string(),
        quality.to_string(),
        h_samp.to_string(),
        v_samp.to_string(),
    ];
    args.extend(extra.iter().map(|value| value.to_string()));
    args
}

/// Resolve the oracle, or skip locally / fail in CI, matching the policy the
/// `require_c_tool!` macro applies to `cjpeg` and friends.
macro_rules! require_cmyk_oracle {
    () => {
        match helpers::c_oracle::cmyk_c_oracle() {
            Some(path) => path,
            None => {
                assert!(
                    !helpers::c_tools::is_ci(),
                    "no libjpeg development install found in CI — the CMYK C \
                     oracle cannot be built and cross-validation would silently \
                     drop to zero (issues #313 / #339 / #340)"
                );
                eprintln!("SKIP: no libjpeg headers/library found to build the CMYK C oracle");
                return;
            }
        }
    };
}

/// The whole matrix, byte for byte. Every option C applies to CMYK is swept
/// here because each of the three defects showed up under a different one.
#[test]
fn cmyk_matches_c_byte_for_byte_across_the_option_matrix() {
    let oracle: PathBuf = require_cmyk_oracle!();

    let mut failures: Vec<String> = Vec::new();
    let mut compared: usize = 0;

    for &(subsampling, h_samp, v_samp) in SUBSAMPLINGS {
        for &(width, height) in GEOMETRIES {
            for &quality in QUALITIES {
                let pixels: Vec<u8> = cmyk_image(width, height);
                let base = || {
                    CompressParams::new(
                        &pixels,
                        width,
                        height,
                        PixelFormat::Cmyk,
                        quality,
                        subsampling,
                    )
                };

                // (label, our bytes, the oracle's extra arguments)
                let cases: Vec<(&str, Vec<u8>, Vec<&str>)> = vec![
                    (
                        "plain",
                        compress_with_params(&base()).expect("CMYK encode"),
                        vec![],
                    ),
                    (
                        "restart3",
                        compress_with_params(&base().restart_interval(3))
                            .expect("CMYK encode with restart"),
                        vec!["--restart", "3"],
                    ),
                    (
                        "dct-fast",
                        compress_with_params(&base().dct_method(DctMethod::IsFast))
                            .expect("CMYK encode with ifast"),
                        vec!["--dct", "fast"],
                    ),
                    (
                        "optimize",
                        compress_optimized_with_params(&base().optimize_huffman(true))
                            .expect("CMYK encode with optimized Huffman"),
                        vec!["--optimize"],
                    ),
                    (
                        "smooth25",
                        compress_optimized_with_params(&base().smoothing_factor(25))
                            .expect("CMYK encode with smoothing"),
                        vec!["--smooth", "25"],
                    ),
                    (
                        "smooth100",
                        compress_optimized_with_params(&base().smoothing_factor(100))
                            .expect("CMYK encode with smoothing"),
                        vec!["--smooth", "100"],
                    ),
                ];

                for (label, ours, extra) in cases {
                    let theirs: Vec<u8> = helpers::c_oracle::encode_with_cmyk_c_oracle(
                        &oracle,
                        &pixels,
                        &oracle_args(width, height, quality, h_samp, v_samp, &extra),
                    );
                    compared += 1;
                    if ours != theirs {
                        let first_difference: usize = ours
                            .iter()
                            .zip(theirs.iter())
                            .take_while(|(a, b)| a == b)
                            .count();
                        failures.push(format!(
                            "  {label} {h_samp}x{v_samp} {width}x{height} q{quality}: \
                             ours={} c={} first diff at byte {first_difference}",
                            ours.len(),
                            theirs.len()
                        ));
                    }
                }
            }
        }
    }

    // A silent zero here would look exactly like success.
    let expected: usize = SUBSAMPLINGS.len() * GEOMETRIES.len() * QUALITIES.len() * 6;
    assert_eq!(
        compared, expected,
        "the sweep must compare every case; a short run reads as a pass"
    );
    assert!(
        failures.is_empty(),
        "{} of {compared} CMYK cases diverged from C (issues #313 / #339 / #340):\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// `float` is deliberately not byte-exact — the AA&N float transform's
/// operation ordering differs — so it gets the weaker guarantee the other DCT
/// paths document: same picture, within a measured bound.
#[test]
fn cmyk_float_dct_is_pixel_equivalent_to_c() {
    let oracle: PathBuf = require_cmyk_oracle!();

    let (width, height) = (64usize, 48usize);
    let pixels: Vec<u8> = cmyk_image(width, height);
    let ours: Vec<u8> = compress_with_params(
        &CompressParams::new(
            &pixels,
            width,
            height,
            PixelFormat::Cmyk,
            75,
            Subsampling::S444,
        )
        .dct_method(DctMethod::Float),
    )
    .expect("CMYK encode with float DCT");
    let theirs: Vec<u8> = helpers::c_oracle::encode_with_cmyk_c_oracle(
        &oracle,
        &pixels,
        &oracle_args(width, height, 75, 1, 1, &["--dct", "float"]),
    );

    let ours_decoded = decompress_to(&ours, PixelFormat::Cmyk).expect("decode ours");
    let theirs_decoded = decompress_to(&theirs, PixelFormat::Cmyk).expect("decode C's");
    let max_difference: i32 = ours_decoded
        .data
        .iter()
        .zip(theirs_decoded.data.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);

    // Measured 4 on this fixture; the bound is that plus margin. Anything
    // approaching islow-scale divergence would be a real defect, not ordering.
    assert!(
        max_difference <= 10,
        "CMYK float diverged from C by {max_difference} per sample — beyond \
         floating-point ordering"
    );
}

/// The marker sequence itself, with no C tool involved, so the #339 contract
/// holds even where the oracle cannot be built.
#[test]
fn cmyk_stream_carries_the_adobe_marker_and_no_jfif() {
    let (width, height) = (16usize, 16usize);
    let pixels: Vec<u8> = cmyk_image(width, height);
    let jpeg: Vec<u8> = compress_with_params(&CompressParams::new(
        &pixels,
        width,
        height,
        PixelFormat::Cmyk,
        75,
        Subsampling::S444,
    ))
    .expect("CMYK encode");

    assert_eq!(
        &jpeg[..4],
        &[0xFF, 0xD8, 0xFF, 0xEE],
        "a CMYK stream must open SOI then Adobe APP14; a JFIF APP0 (FF E0) \
         claims a colorspace JFIF does not define (issue #339)"
    );

    // Component IDs live in SOF0: SOI(2) + APP14(16) + DQT(2+2+64) + the SOF
    // header, then one 3-byte record per component.
    let sof_start: usize = jpeg
        .windows(2)
        .position(|window| window == [0xFF, 0xC0])
        .expect("SOF0 marker");
    let components_start: usize = sof_start + 10;
    let ids: Vec<u8> = (0..4)
        .map(|index| jpeg[components_start + index * 3])
        .collect();
    assert_eq!(
        ids,
        vec![b'C', b'M', b'Y', b'K'],
        "component IDs must be the ASCII initials libjpeg writes (issue #339)"
    );
}
