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
    cjpeg_encode_size(cjpeg, extra_args, grayscale, pixels, sample, WIDTH, HEIGHT)
}

fn cjpeg_encode_size(
    cjpeg: &std::path::Path,
    extra_args: &[&str],
    grayscale: bool,
    pixels: &[u8],
    sample: &str,
    width: usize,
    height: usize,
) -> Result<Vec<u8>, String> {
    let input = helpers::TempFile::new(if grayscale { "cs.pgm" } else { "cs.ppm" });
    if grayscale {
        helpers::write_pgm_file(input.path(), width, height, pixels);
    } else {
        helpers::write_ppm_file(input.path(), width, height, pixels);
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

/// The Rust-only fancy prefilter has no arbitrary-factor implementation.
#[test]
fn issue_664_fancy_downsampling_is_explicitly_refused() {
    let pixels = textured_rgb(WIDTH, HEIGHT);
    assert!(
        matches!(Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .sampling_factors(vec![(3,2),(1,1),(1,1)]).fancy_downsampling(true).encode(),
        Err(JpegError::Unsupported(message)) if message.contains("fancy_downsampling"))
    );
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

/// Issue #664: the lossless reset holds for RGB-direct output too.
/// `cjpeg -lossless 1 -rgb -sample 3x2,1x1,1x1` codes every component at 1x1;
/// Rust used to refuse the combination before reaching the lossless encoder.
#[test]
fn issue_664_rgb_direct_lossless_ignores_sampling_as_c_does() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels: Vec<u8> = textured_rgb(WIDTH, HEIGHT);
    let c_jpeg: Vec<u8> = cjpeg_encode(
        &cjpeg,
        &["-lossless", "1", "-rgb"],
        false,
        &pixels,
        "3x2,1x1,1x1",
    )
    .unwrap_or_else(|stderr| panic!("cjpeg refused: {stderr}"));
    let jpeg: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .colorspace(libjpeg_turbo_rs::ColorSpace::Rgb)
        .sampling_factors(vec![(3, 2), (1, 1), (1, 1)])
        .lossless(true)
        .lossless_predictor(1)
        .encode()
        .unwrap_or_else(|error| panic!("Rust refused: {error}"));
    assert_same_stream("RGB-direct lossless 3x2", &jpeg, &c_jpeg);
}

/// Issue #673: color conversion and smoothing compose with every entropy mode.
#[test]
fn issue_673_rgb_gray_and_smoothing_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels = textured_rgb(WIDTH, HEIGHT);
    for &(sample, factors) in COLOR_FACTORS {
        for &mode in MODES {
            for kind in ["rgb", "gray", "smooth"] {
                let mut args = mode.cjpeg_args.to_vec();
                let mut enc = (mode.apply)(
                    Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
                        .quality(75)
                        .sampling_factors(factors.to_vec()),
                );
                match kind {
                    "rgb" => {
                        args.push("-rgb");
                        enc = enc.colorspace(libjpeg_turbo_rs::ColorSpace::Rgb);
                    }
                    "gray" => {
                        args.push("-grayscale");
                        enc = enc.grayscale_from_color(true);
                    }
                    _ => {
                        args.extend(["-smooth", "10"]);
                        enc = enc.smoothing_factor(10);
                    }
                }
                let label = format!("{sample} / {} / {kind}", mode.label);
                let c = cjpeg_encode(&cjpeg, &args, false, &pixels, sample).expect(&label);
                let rust = enc.encode().expect(&label);
                assert_same_stream(&label, &rust, &c);
            }
        }
    }
}

fn component_script(n: u8) -> Vec<libjpeg_turbo_rs::ScanScript> {
    use libjpeg_turbo_rs::ScanScript;
    let mut script = Vec::new();
    // Separate DC first/refinement scans expose component versus frame MCU grids.
    // Reverse scan order also checks table emission follows the scan, not the frame.
    for ci in (0..n).rev() {
        script.push(ScanScript {
            components: vec![ci],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        });
    }
    for ci in (0..n).rev() {
        script.push(ScanScript {
            components: vec![ci],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        });
        script.push(ScanScript {
            components: vec![ci],
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        });
    }
    script
}

/// Issue #673: scripted DC scans must traverse each component's real block grid.
#[test]
fn issue_673_component_scans_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels = textured_rgb(WIDTH, HEIGHT);
    let script = component_script(3);
    let file = helpers::TempFile::new("custom-scans.txt");
    let text: String = script
        .iter()
        .map(|s| {
            format!(
                "{}: {} {} {} {};\n",
                s.components[0], s.ss, s.se, s.ah, s.al
            )
        })
        .collect();
    std::fs::write(file.path(), text).unwrap();
    for &(sample, factors) in COLOR_FACTORS {
        for arithmetic in [false, true] {
            for rgb in [false, true] {
                for restart in ["0", "1", "1B"] {
                    let mut args =
                        vec!["-scans", file.path().to_str().unwrap(), "-restart", restart];
                    let mut enc = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
                        .quality(75)
                        .sampling_factors(factors.to_vec())
                        .progressive(true)
                        .arithmetic(arithmetic)
                        .scan_script(script.clone());
                    if arithmetic {
                        args.push("-arithmetic");
                    }
                    if rgb {
                        args.push("-rgb");
                        enc = enc.colorspace(libjpeg_turbo_rs::ColorSpace::Rgb);
                    }
                    enc = match restart {
                        "1" => enc.restart_rows(1),
                        "1B" => enc.restart_blocks(1),
                        _ => enc,
                    };
                    let label =
                        format!("{sample} arithmetic={arithmetic} rgb={rgb} restart={restart}");
                    let c = cjpeg_encode(&cjpeg, &args, false, &pixels, sample).expect(&label);
                    let rust = enc.encode().expect(&label);
                    assert_same_stream(&label, &rust, &c);
                }
            }
        }
    }
}

fn installed_tables(mut enc: Encoder<'_>, cmyk: bool) -> Encoder<'_> {
    use libjpeg_turbo_rs::{encode::tables::*, HuffmanTableDef};
    for i in 0..if cmyk { 1 } else { 2 } {
        let (db, dv, ab, av) = if i == 0 {
            (
                DC_LUMINANCE_BITS,
                DC_LUMINANCE_VALUES.to_vec(),
                AC_LUMINANCE_BITS,
                AC_LUMINANCE_VALUES.to_vec(),
            )
        } else {
            (
                DC_CHROMINANCE_BITS,
                DC_CHROMINANCE_VALUES.to_vec(),
                AC_CHROMINANCE_BITS,
                AC_CHROMINANCE_VALUES.to_vec(),
            )
        };
        let mut dc = HuffmanTableDef {
            bits: db,
            values: dv,
        };
        let mut ac = HuffmanTableDef {
            bits: ab,
            values: av,
        };
        dc.values.swap(0, 1);
        ac.values.swap(0, 1);
        enc = enc.huffman_dc_table(i, dc).huffman_ac_table(i, ac);
    }
    enc
}

/// Issue #673: four independently sampled CMYK planes and nondefault Huffman
/// tables, including modes where C regenerates or does not use those tables.
#[test]
fn issue_673_cmyk_and_installed_tables_match_stock_api() {
    let Some(oracle) = helpers::c_oracle::custom_sampling_c_oracle() else {
        assert!(
            !helpers::is_ci(),
            "CI requires a libjpeg development install for the sampling oracle"
        );
        eprintln!("SKIP: libjpeg development install not found");
        return;
    };
    for cmyk in [false, true] {
        let factors = if cmyk {
            vec![(1, 1), (1, 1), (1, 1), (3, 2)]
        } else {
            vec![(3, 2), (1, 1), (1, 1)]
        };
        let sample = if cmyk {
            "1x1,1x1,1x1,3x2"
        } else {
            "3x2,1x1,1x1"
        };
        let format = if cmyk {
            PixelFormat::Cmyk
        } else {
            PixelFormat::Rgb
        };
        let pixels: Vec<u8> = (0..WIDTH * HEIGHT * format.bytes_per_pixel())
            .map(|i| ((i * 37 + i / 7) % 256) as u8)
            .collect();
        for arithmetic in [false, true] {
            for progressive in [false, true] {
                for script in [false, true] {
                    if script && !progressive {
                        continue;
                    }
                    for custom in [false, true] {
                        for smooth in [0, 10] {
                            for restart in [0, 1, 2] {
                                for (quality, optimize) in [(75, false), (75, true), (10, true)] {
                                    let mut enc = Encoder::new(&pixels, WIDTH, HEIGHT, format)
                                        .quality(quality)
                                        .optimize_huffman(optimize)
                                        .sampling_factors(factors.clone())
                                        .arithmetic(arithmetic)
                                        .progressive(progressive)
                                        .smoothing_factor(smooth);
                                    if custom {
                                        enc = installed_tables(enc, cmyk);
                                    }
                                    if script {
                                        enc = enc.scan_script(component_script(if cmyk {
                                            4
                                        } else {
                                            3
                                        }));
                                    }
                                    if restart == 1 {
                                        enc = enc.restart_blocks(1);
                                    }
                                    if restart == 2 {
                                        enc = enc.restart_rows(1);
                                    }
                                    let args: Vec<String> = vec![
                                        WIDTH.to_string(),
                                        HEIGHT.to_string(),
                                        u8::from(cmyk).to_string(),
                                        sample.into(),
                                        quality.to_string(),
                                        u8::from(progressive).to_string(),
                                        u8::from(arithmetic).to_string(),
                                        smooth.to_string(),
                                        u8::from(restart == 1).to_string(),
                                        u8::from(restart == 2).to_string(),
                                        u8::from(custom).to_string(),
                                        u8::from(script).to_string(),
                                        u8::from(optimize).to_string(),
                                    ];
                                    let c = helpers::c_oracle::encode_with_cmyk_c_oracle(
                                        &oracle, &pixels, &args,
                                    );
                                    let label = format!("cmyk={cmyk} arithmetic={arithmetic} progressive={progressive} script={script} custom={custom} smooth={smooth} restart={restart} quality={quality} optimize={optimize}");
                                    let rust = enc.encode().expect(&label);
                                    assert_same_stream(&label, &rust, &c);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Issue #673: tiny/odd dimensions and factors whose maximum horizontal and
/// vertical sampling live on different components stress padding/context rows.
#[test]
fn issue_673_smoothing_edges_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    for (width, height) in [(1, 1), (7, 9), (25, 17), (74, 45)] {
        let pixels = textured_rgb(width, height);
        for (sample, factors) in COLOR_FACTORS.iter().copied().chain([
            ("4x1,1x4,1x1", [(4, 1), (1, 4), (1, 1)]),
            ("2x1,1x2,1x1", [(2, 1), (1, 2), (1, 1)]),
        ]) {
            for smooth in [1, 50, 100] {
                let c = cjpeg_encode_size(
                    &cjpeg,
                    &[
                        "-smooth",
                        &smooth.to_string(),
                        "-progressive",
                        "-restart",
                        "1",
                    ],
                    false,
                    &pixels,
                    sample,
                    width,
                    height,
                )
                .unwrap();
                let rust = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                    .quality(75)
                    .sampling_factors(factors.to_vec())
                    .smoothing_factor(smooth)
                    .progressive(true)
                    .restart_rows(1)
                    .encode()
                    .unwrap();
                assert_same_stream(
                    &format!("{width}x{height} {sample} smooth={smooth}"),
                    &rust,
                    &c,
                );
            }
        }
    }
}

/// Issue #673: C's 10-block limit applies to each interleaved scan, not to
/// the whole frame when every scan contains just one component.
#[test]
fn issue_673_large_factors_with_separate_scans_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels = textured_rgb(WIDTH, HEIGHT);
    let script = component_script(3);
    let file = helpers::TempFile::new("large-sampling-scans.txt");
    std::fs::write(
        file.path(),
        script
            .iter()
            .map(|s| {
                format!(
                    "{}: {} {} {} {};\n",
                    s.components[0], s.ss, s.se, s.ah, s.al
                )
            })
            .collect::<String>(),
    )
    .unwrap();
    for arithmetic in [false, true] {
        let mut args = vec!["-scans", file.path().to_str().unwrap(), "-restart", "1"];
        if arithmetic {
            args.push("-arithmetic");
        }
        let c = cjpeg_encode(&cjpeg, &args, false, &pixels, "4x4,1x1,1x1").unwrap();
        let rust = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
            .quality(75)
            .sampling_factors(vec![(4, 4), (1, 1), (1, 1)])
            .progressive(true)
            .arithmetic(arithmetic)
            .restart_rows(1)
            .scan_script(script.clone())
            .encode()
            .unwrap();
        assert_same_stream("4x4 separate scans", &rust, &c);
    }
}

/// Issue #673: mixed interleaved/non-interleaved scripts. Stock 3.2.0 ARM
/// NEON's h2v2 downsampler uses outrow instead of 2*outrow when v > 1;
/// explicitly use scalar stock C for this upstream-divergent factor set.
#[test]
fn issue_673_mixed_scans_match_scalar_cjpeg() {
    use libjpeg_turbo_rs::ScanScript;
    let cjpeg = require_c_tool!("cjpeg");
    let pixels = textured_rgb(51, 29);
    let input = helpers::TempFile::new("mixed-scans.ppm");
    helpers::write_ppm_file(input.path(), 51, 29, &pixels);
    let file = helpers::TempFile::new("mixed-scans.txt");
    std::fs::write(file.path(), "0 2: 0 0 0 1;\n1: 0 0 0 1;\n2: 1 63 0 0;\n0: 1 63 0 0;\n1: 1 63 0 0;\n0 1: 0 0 1 0;\n2: 0 0 1 0;\n").unwrap();
    let script: Vec<ScanScript> = [
        (vec![0, 2], 0, 0, 0, 1),
        (vec![1], 0, 0, 0, 1),
        (vec![2], 1, 63, 0, 0),
        (vec![0], 1, 63, 0, 0),
        (vec![1], 1, 63, 0, 0),
        (vec![0, 1], 0, 0, 1, 0),
        (vec![2], 0, 0, 1, 0),
    ]
    .into_iter()
    .map(|(components, ss, se, ah, al)| ScanScript {
        components,
        ss,
        se,
        ah,
        al,
    })
    .collect();
    for arithmetic in [false, true] {
        let output = helpers::TempFile::new("mixed-scans.jpg");
        let mut cmd = Command::new(&cjpeg);
        cmd.env("JSIMD_FORCENONE", "1")
            .args([
                "-quality",
                "75",
                "-sample",
                "2x2,1x4,4x1",
                "-restart",
                "1",
                "-scans",
            ])
            .arg(file.path());
        if arithmetic {
            cmd.arg("-arithmetic");
        }
        let result = cmd
            .arg("-outfile")
            .arg(output.path())
            .arg(input.path())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let rust = Encoder::new(&pixels, 51, 29, PixelFormat::Rgb)
            .quality(75)
            .sampling_factors(vec![(2, 2), (1, 4), (4, 1)])
            .progressive(true)
            .arithmetic(arithmetic)
            .restart_rows(1)
            .scan_script(script.clone())
            .encode()
            .unwrap();
        assert_same_stream(
            "mixed scans / scalar C",
            &rust,
            &std::fs::read(output.path()).unwrap(),
        );
    }
}

/// Issue #673 review: RGB override must not reinterpret a grayscale or CMYK
/// input as three packed channels. Preserve the builder's format guard.
#[test]
fn issue_673_rgb_override_respects_effective_input_format() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels = textured_rgb(WIDTH, HEIGHT);
    let c = cjpeg_encode(&cjpeg, &["-grayscale"], false, &pixels, "3x2,1x1,1x1").unwrap();
    let rust = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .colorspace(libjpeg_turbo_rs::ColorSpace::Rgb)
        .grayscale_from_color(true)
        .sampling_factors(vec![(3, 2), (1, 1), (1, 1)])
        .encode()
        .unwrap();
    assert_same_stream("grayscale overrides RGB-direct", &rust, &c);
    let Some(oracle) = helpers::c_oracle::custom_sampling_c_oracle() else {
        assert!(
            !helpers::is_ci(),
            "CI requires a libjpeg development install for the sampling oracle"
        );
        eprintln!("SKIP: libjpeg development install not found");
        return;
    };
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * 4).map(|i| (i % 256) as u8).collect();
    let args: Vec<String> = [
        WIDTH.to_string(),
        HEIGHT.to_string(),
        "1".into(),
        "1x1,1x1,1x1,3x2".into(),
        "75".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "0".into(),
        "0".into(),
    ]
    .into();
    let c = helpers::c_oracle::encode_with_cmyk_c_oracle(&oracle, &pixels, &args);
    let rust = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Cmyk)
        .colorspace(libjpeg_turbo_rs::ColorSpace::Rgb)
        .sampling_factors(vec![(1, 1), (1, 1), (1, 1), (3, 2)])
        .encode()
        .unwrap();
    assert_same_stream("CMYK retains four components", &rust, &c);
}

/// Issue #673 review: the coefficient path must preserve RGB ICC metadata,
/// which the older pixel-domain RGB writers used to insert themselves.
#[test]
fn issue_673_rgb_direct_icc_matches_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    let pixels = textured_rgb(WIDTH, HEIGHT);
    let profile = b"issue-673 ICC profile bytes";
    let file = helpers::TempFile::new("custom-sampling.icc");
    std::fs::write(file.path(), profile).unwrap();
    for &mode in MODES {
        let mut args = mode.cjpeg_args.to_vec();
        args.extend(["-rgb", "-icc", file.path().to_str().unwrap()]);
        let c = cjpeg_encode(&cjpeg, &args, false, &pixels, "3x2,1x1,1x1").unwrap();
        let rust = (mode.apply)(
            Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
                .colorspace(libjpeg_turbo_rs::ColorSpace::Rgb)
                .sampling_factors(vec![(3, 2), (1, 1), (1, 1)])
                .icc_profile(profile),
        )
        .encode()
        .unwrap();
        assert_same_stream(mode.label, &rust, &c);
    }
}
