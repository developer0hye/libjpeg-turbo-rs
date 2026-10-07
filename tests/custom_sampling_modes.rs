//! Issue #664 (P4-236): `Encoder::sampling_factors` with factors that map to
//! no standard `Subsampling` (3x2, 1x4, a chroma component out-sampling
//! another, …) used to route to a baseline-only Huffman encoder ahead of every
//! mode switch, so `.progressive(true)`, `.arithmetic(true)`,
//! `.lossless(true)` and `.restart_*` were dropped and `encode` returned `Ok`
//! with an SOF0 stream. Custom quantisation, the DCT method and optimised
//! Huffman tables never reached it either.
//!
//! The C oracle is `cjpeg -sample`, which composes with all of these: the
//! sampling factors are only component fields to `jcmaster.c`. Every case
//! here must be byte-identical to the matching `cjpeg` invocation, and every
//! factor set C refuses must be refused here.

mod helpers;

use std::process::Command;

use libjpeg_turbo_rs::{DctMethod, Encoder, JpegError, PixelFormat};

/// Not a multiple of any MCU size used below on either axis. The height is
/// 2 mod 4 so that a component with `v = 2` under `max_v = 4` hits C's
/// row-group padding (`jcprepct.c` pads the input to a multiple of
/// `max_v_samp_factor` rows, then the downsampled output to a full iMCU):
/// C's 38th Cb row averages two copies of input row 73, where padding the
/// downsampled rows alone would repeat the 37th.
const WIDTH: usize = 45;
const HEIGHT: usize = 74;

/// A frame with real AC energy in both axes, so every coefficient band and
/// every downsampling bias matters.
fn textured_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 37 + y * 11) % 256) as u8);
            pixels.push(((x * x + y * 3) % 256) as u8);
            pixels.push((((x ^ y) * 29) % 256) as u8);
        }
    }
    pixels
}

fn textured_gray(width: usize, height: usize) -> Vec<u8> {
    (0..width * height)
        .map(|i: usize| (((i % width) * 13 + (i / width) * 7 + (i % 7) * 31) % 256) as u8)
        .collect()
}

/// One encoder configuration, as builder options and `cjpeg` switches.
#[derive(Clone, Copy)]
struct Mode {
    label: &'static str,
    cjpeg_args: &'static [&'static str],
    apply: fn(Encoder<'_>) -> Encoder<'_>,
}

const MODES: &[Mode] = &[
    Mode {
        label: "baseline",
        cjpeg_args: &[],
        apply: |encoder| encoder,
    },
    Mode {
        label: "optimize",
        cjpeg_args: &["-optimize"],
        apply: |encoder| encoder.optimize_huffman(true),
    },
    Mode {
        label: "progressive",
        cjpeg_args: &["-progressive"],
        apply: |encoder| encoder.progressive(true),
    },
    Mode {
        label: "arithmetic",
        cjpeg_args: &["-arithmetic"],
        apply: |encoder| encoder.arithmetic(true),
    },
    Mode {
        label: "arithmetic progressive",
        cjpeg_args: &["-arithmetic", "-progressive"],
        apply: |encoder| encoder.arithmetic(true).progressive(true),
    },
    Mode {
        label: "restart every MCU",
        cjpeg_args: &["-restart", "1B"],
        apply: |encoder| encoder.restart_blocks(1),
    },
    Mode {
        label: "restart every row",
        cjpeg_args: &["-restart", "1"],
        apply: |encoder| encoder.restart_rows(1),
    },
    Mode {
        label: "restart every row, progressive",
        cjpeg_args: &["-restart", "1", "-progressive"],
        apply: |encoder| encoder.restart_rows(1).progressive(true),
    },
    Mode {
        label: "restart every MCU, arithmetic",
        cjpeg_args: &["-restart", "1B", "-arithmetic"],
        apply: |encoder| encoder.restart_blocks(1).arithmetic(true),
    },
    Mode {
        label: "ifast DCT",
        cjpeg_args: &["-dct", "fast"],
        apply: |encoder| encoder.dct_method(DctMethod::IsFast),
    },
    Mode {
        label: "quality 10 (16-bit tables, SOF1)",
        cjpeg_args: &["-quality", "10"],
        apply: |encoder| encoder.quality(10),
    },
    Mode {
        label: "quality 10, forced baseline",
        cjpeg_args: &["-quality", "10", "-baseline"],
        apply: |encoder| encoder.quality(10).force_baseline(true),
    },
    Mode {
        label: "per-table quality 90,40, progressive",
        cjpeg_args: &["-quality", "90,40", "-progressive"],
        apply: |encoder| {
            encoder
                .quality_factor(0, 90)
                .quality_factor(1, 40)
                .progressive(true)
        },
    },
];

/// Factor sets that map to no standard `Subsampling`, each taking a different
/// downsampling path in `jcsample.c`'s `jinit_downsampler`.
const COLOR_FACTORS: &[(&str, [(u8, u8); 3])] = &[
    // Chroma by int_downsample (3x2 is no h2v1/h2v2 ratio).
    ("3x2,1x1,1x1", [(3, 2), (1, 1), (1, 1)]),
    // Cb at ratio (1,2) -> int_downsample; Cr at (2,2) -> h2v2.
    ("2x2,2x1,1x1", [(2, 2), (2, 1), (1, 1)]),
    // Cb has v = 2 under max_v = 4: the row-group padding case.
    ("1x4,1x2,1x1", [(1, 4), (1, 2), (1, 1)]),
    // Component 0 is not the largest: luma itself is downsampled (h2v2).
    ("1x1,2x2,1x1", [(1, 1), (2, 2), (1, 1)]),
    // Cb at ratio (2,1) -> h2v1 under a 4-wide MCU.
    ("4x1,2x1,1x1", [(4, 1), (2, 1), (1, 1)]),
];

fn rust_encode(
    mode: Mode,
    format: PixelFormat,
    pixels: &[u8],
    factors: &[(u8, u8)],
) -> Result<Vec<u8>, JpegError> {
    (mode.apply)(
        Encoder::new(pixels, WIDTH, HEIGHT, format)
            .quality(75)
            .sampling_factors(factors.to_vec()),
    )
    .encode()
}

/// What `cjpeg` writes, or its stderr when it refuses.
fn cjpeg_encode(
    cjpeg: &std::path::Path,
    extra_args: &[&str],
    grayscale: bool,
    pixels: &[u8],
    sample: &str,
) -> Result<Vec<u8>, String> {
    let input = helpers::TempFile::new(if grayscale { "cs.pgm" } else { "cs.ppm" });
    if grayscale {
        helpers::write_pgm_file(input.path(), WIDTH, HEIGHT, pixels);
    } else {
        helpers::write_ppm_file(input.path(), WIDTH, HEIGHT, pixels);
    }
    let output_jpeg = helpers::TempFile::new("cs.jpg");
    let output: std::process::Output = Command::new(cjpeg)
        .args(["-quality", "75"])
        .args(extra_args)
        .args(["-sample", sample])
        .arg("-outfile")
        .arg(output_jpeg.path())
        .arg(input.path())
        .output()
        .unwrap_or_else(|error| panic!("failed to run cjpeg: {error}"));
    if output.status.success() {
        Ok(std::fs::read(output_jpeg.path())
            .unwrap_or_else(|error| panic!("cjpeg wrote no output: {error}")))
    } else {
        Err(String::from_utf8_lossy(&output.stderr).into_owned())
    }
}

fn assert_same_stream(label: &str, rust: &[u8], c: &[u8]) {
    if rust != c {
        let first_difference: usize = rust
            .iter()
            .zip(c)
            .position(|(a, b)| a != b)
            .unwrap_or(rust.len().min(c.len()));
        panic!(
            "{label}: Rust and cjpeg streams differ ({} vs {} bytes, first at {first_difference})",
            rust.len(),
            c.len()
        );
    }
}

/// The SOF marker byte (0xC0..=0xCF, skipping DHT/JPG/DAC) of a stream.
fn sof_marker(jpeg: &[u8]) -> u8 {
    jpeg.windows(2)
        .find_map(|pair: &[u8]| {
            (pair[0] == 0xFF
                && (0xC0..=0xCF).contains(&pair[1])
                && ![0xC4, 0xC8, 0xCC].contains(&pair[1]))
            .then_some(pair[1])
        })
        .unwrap_or_else(|| panic!("no SOF marker"))
}

/// Issue #664: every mode composes with every non-standard factor set,
/// byte-identical to `cjpeg -sample`.
#[test]
fn issue_664_every_mode_matches_cjpeg_for_nonstandard_color_factors() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels: Vec<u8> = textured_rgb(WIDTH, HEIGHT);
    for &(sample, factors) in COLOR_FACTORS {
        for &mode in MODES {
            let label: String = format!("{sample} / {}", mode.label);
            let c_jpeg: Vec<u8> = cjpeg_encode(&cjpeg, mode.cjpeg_args, false, &pixels, sample)
                .unwrap_or_else(|stderr| panic!("{label}: cjpeg refused: {stderr}"));
            let jpeg: Vec<u8> = rust_encode(mode, PixelFormat::Rgb, &pixels, &factors)
                .unwrap_or_else(|error| panic!("{label}: Rust refused: {error}"));
            assert_same_stream(&label, &jpeg, &c_jpeg);
        }
    }
}

/// Issue #664: a single-component frame with a non-unit factor, in every mode.
/// The scan is non-interleaved, so the factor only reaches the SOF.
#[test]
fn issue_664_every_mode_matches_cjpeg_for_grayscale_factor() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels: Vec<u8> = textured_gray(WIDTH, HEIGHT);
    for &mode in MODES {
        let label: String = format!("gray 2x2 / {}", mode.label);
        let c_jpeg: Vec<u8> = cjpeg_encode(&cjpeg, mode.cjpeg_args, true, &pixels, "2x2")
            .unwrap_or_else(|stderr| panic!("{label}: cjpeg refused: {stderr}"));
        let jpeg: Vec<u8> = rust_encode(mode, PixelFormat::Grayscale, &pixels, &[(2, 2)])
            .unwrap_or_else(|error| panic!("{label}: Rust refused: {error}"));
        assert_same_stream(&label, &jpeg, &c_jpeg);
    }
}

/// Issue #664: lossless mode resets every component to 1x1 in C
/// (`jcmaster.c`, "Disable smoothing and subsampling in lossless mode"), so
/// `cjpeg -lossless 1 -sample 3x2,1x1,1x1` writes the same stream as without
/// `-sample`. Rust used to write a baseline SOF0 stream instead.
#[test]
fn issue_664_lossless_ignores_sampling_as_c_does() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels: Vec<u8> = textured_rgb(WIDTH, HEIGHT);
    let c_jpeg: Vec<u8> = cjpeg_encode(&cjpeg, &["-lossless", "1"], false, &pixels, "3x2,1x1,1x1")
        .unwrap_or_else(|stderr| panic!("cjpeg refused: {stderr}"));
    let jpeg: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .sampling_factors(vec![(3, 2), (1, 1), (1, 1)])
        .lossless(true)
        .lossless_predictor(1)
        .encode()
        .unwrap_or_else(|error| panic!("Rust refused: {error}"));
    assert_eq!(sof_marker(&jpeg), 0xC3, "lossless must write SOF3");
    assert_same_stream("lossless 3x2", &jpeg, &c_jpeg);
}

/// Issue #664: the reproducer from the issue — none of these may come back
/// as a baseline SOF0 stream.
#[test]
fn issue_664_modes_are_not_silently_dropped() {
    let pixels: Vec<u8> = vec![100u8; 48 * 48 * 3];
    let base = || {
        Encoder::new(&pixels, 48, 48, PixelFormat::Rgb).sampling_factors(vec![
            (3, 2),
            (1, 1),
            (1, 1),
        ])
    };
    let sof = |encoder: Encoder<'_>, label: &str| -> u8 {
        sof_marker(
            &encoder
                .encode()
                .unwrap_or_else(|error| panic!("{label}: {error}")),
        )
    };
    assert_eq!(sof(base().progressive(true), "progressive"), 0xC2);
    assert_eq!(sof(base().arithmetic(true), "arithmetic"), 0xC9);
    assert_eq!(sof(base().arithmetic(true).progressive(true), "both"), 0xCA);
    assert_eq!(sof(base().lossless(true), "lossless"), 0xC3);
    let restarted: Vec<u8> = base()
        .restart_blocks(1)
        .encode()
        .unwrap_or_else(|error| panic!("restart: {error}"));
    assert!(
        restarted.windows(2).any(|pair: &[u8]| pair == [0xFF, 0xDD]),
        "restart_blocks(1) must write a DRI"
    );
}

/// Issue #664: factor sets C refuses are refused here, before any pixels are
/// read. `4x4,1x1,1x1` puts 18 blocks in an MCU (`C_MAX_BLOCKS_IN_MCU` is 10,
/// `JERR_BAD_MCU_SIZE`); `3x1,2x1,1x1` is a fractional ratio
/// (`JERR_FRACT_SAMPLE_NOTIMPL`).
#[test]
fn issue_664_factor_sets_c_refuses_are_refused() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels: Vec<u8> = textured_rgb(WIDTH, HEIGHT);
    let cases: [(&str, [(u8, u8); 3], &str); 2] = [
        (
            "4x4,1x1,1x1",
            [(4, 4), (1, 1), (1, 1)],
            "Sampling factors too large for interleaved scan",
        ),
        (
            "3x1,2x1,1x1",
            [(3, 1), (2, 1), (1, 1)],
            "Fractional sampling not implemented yet",
        ),
    ];
    for (sample, factors, c_message) in cases {
        for &mode in MODES {
            let label: String = format!("{sample} / {}", mode.label);
            match cjpeg_encode(&cjpeg, mode.cjpeg_args, false, &pixels, sample) {
                Ok(_) => panic!("{label}: cjpeg accepted a factor set this case says C refuses"),
                Err(stderr) => assert!(
                    stderr.contains(c_message),
                    "{label}: cjpeg refused with {stderr:?}, expected {c_message:?}"
                ),
            }
            match rust_encode(mode, PixelFormat::Rgb, &pixels, &factors) {
                Err(JpegError::CorruptData(message)) => assert!(
                    message.contains("sampling"),
                    "{label}: message does not name the sampling factors: {message}"
                ),
                Err(other) => panic!("{label}: expected CorruptData, got {other:?}"),
                Ok(jpeg) => panic!("{label}: expected a refusal, got {} bytes", jpeg.len()),
            }
        }
    }
}

/// Issue #664: options with no C counterpart on this path, or that C applies
/// only partly here, are refused rather than dropped.
///
/// * `smoothing_factor`: C smooths only full-size components and h2v2 ones
///   and traces `JTRC_SMOOTH_NOTIMPL` for the rest (`jcsample.c`).
/// * `fancy_downsampling`: a Rust-only prefilter keyed on `subsampling()`,
///   which these factors bypass.
/// * custom Huffman tables: not carried by this encoder.
#[test]
fn issue_664_options_this_path_cannot_carry_are_refused() {
    let pixels: Vec<u8> = textured_rgb(WIDTH, HEIGHT);
    let base = || {
        Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb).sampling_factors(vec![
            (3, 2),
            (1, 1),
            (1, 1),
        ])
    };
    let table = libjpeg_turbo_rs::HuffmanTableDef {
        bits: [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        values: (0..12).collect(),
    };
    for (label, encoder) in [
        ("smoothing", base().smoothing_factor(10)),
        ("fancy downsampling", base().fancy_downsampling(true)),
        ("custom Huffman table", base().huffman_dc_table(0, table)),
    ] {
        match encoder.encode() {
            Err(JpegError::Unsupported(message)) => assert!(
                message.contains("sampling_factors"),
                "{label}: message does not name sampling_factors: {message}"
            ),
            Err(other) => panic!("{label}: expected Unsupported, got {other:?}"),
            Ok(jpeg) => panic!("{label}: expected Unsupported, got {} bytes", jpeg.len()),
        }
    }
}

/// Issue #664: the same lossless rule for a single-component frame. A
/// grayscale encode with an explicit sampling request had its SOF patched to
/// that factor after encoding, so `lossless(true)` wrote 2x2 where
/// `cjpeg -lossless 1 -sample 2x2` writes 1x1 — for a non-standard factor
/// and for `subsampling(S420)` alike.
#[test]
fn issue_664_grayscale_lossless_ignores_sampling_as_c_does() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels: Vec<u8> = textured_gray(WIDTH, HEIGHT);
    let c_jpeg: Vec<u8> = cjpeg_encode(&cjpeg, &["-lossless", "1"], true, &pixels, "2x2")
        .unwrap_or_else(|stderr| panic!("cjpeg refused: {stderr}"));
    let lossless = || {
        Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Grayscale)
            .lossless(true)
            .lossless_predictor(1)
    };
    for (label, encoder) in [
        (
            "sampling_factors 2x2",
            lossless().sampling_factors(vec![(2, 2)]),
        ),
        (
            "subsampling S420",
            lossless().subsampling(libjpeg_turbo_rs::Subsampling::S420),
        ),
    ] {
        let jpeg: Vec<u8> = encoder
            .encode()
            .unwrap_or_else(|error| panic!("{label}: Rust refused: {error}"));
        assert_same_stream(label, &jpeg, &c_jpeg);
    }
}
