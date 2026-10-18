//! Custom progressive scan scripts are validated the way C's `validate_script`
//! (`references/libjpeg-turbo/src/jcmaster.c:276-436`) validates them, before
//! any encoding work runs.
//!
//! Issue #610 (P4-192): `Encoder::scan_script` used to store the script
//! verbatim, and on x86_64 with the default `simd` feature an AC-first band
//! with `se > 63` reached `prepare_ac_first_sse2`, which indexes through raw
//! pointers with no bound — safe Rust wrote past two `[u16; 64]` stack arrays
//! and the encode returned `Ok`. The aarch64 / scalar arm panicked on the same
//! script instead. Both must now return the same typed error.
//!
//! The C oracle is `cjpeg -scans`, whose script file goes through the same
//! `validate_script`. Every case that file format can express is run through
//! it: a script C refuses must be refused here at the same entry, and a script
//! C accepts must be accepted here and encoded to the same bytes.
//!
//! Issue #636 (P4-210): the script used to reach only the Huffman YCbCr /
//! grayscale encode; arithmetic coding and RGB-direct output silently used
//! C's default script instead. C honours `scan_info` independently of the
//! entropy coder and the colorspace, so every case now runs in every
//! [`Mode`], each against the matching `cjpeg -arithmetic` / `-rgb` call.

mod helpers;

use std::process::Command;

use libjpeg_turbo_rs::{
    decompress, ColorSpace, Encoder, JpegError, PixelFormat, ScanScript, Subsampling,
};

/// 40 so the 4:2:0 luma grid (5 blocks) is narrower than the MCU-padded grid
/// (6): a non-interleaved scan that walked MCUs instead of the component's own
/// blocks would encode a column C does not.
const SIDE: usize = 40;

fn scan(components: &[u8], ss: u8, se: u8, ah: u8, al: u8) -> ScanScript {
    ScanScript {
        components: components.to_vec(),
        ss,
        se,
        ah,
        al,
    }
}

/// One way of encoding the frame: the entropy coder and the output
/// colorspace, as the builder options and the matching `cjpeg` switches.
#[derive(Clone, Copy, Debug)]
struct Mode {
    label: &'static str,
    arithmetic: bool,
    /// `JCS_RGB` output (`.colorspace(ColorSpace::Rgb)`, `cjpeg -rgb`).
    rgb_direct: bool,
    /// For RGB-direct: request 2x2 luma-slot sampling (`cjpeg -sample
    /// 2x2,1x1,1x1`) instead of `-rgb`'s 1x1, so the first component's grid
    /// differs from the MCU grid as it does for YCbCr 4:2:0.
    rgb_subsampled: bool,
}

const HUFFMAN_YCBCR: Mode = Mode {
    label: "Huffman YCbCr",
    arithmetic: false,
    rgb_direct: false,
    rgb_subsampled: false,
};

const MODES: [Mode; 5] = [
    HUFFMAN_YCBCR,
    Mode {
        label: "arithmetic YCbCr",
        arithmetic: true,
        rgb_direct: false,
        rgb_subsampled: false,
    },
    Mode {
        label: "Huffman RGB-direct",
        arithmetic: false,
        rgb_direct: true,
        rgb_subsampled: false,
    },
    Mode {
        label: "Huffman RGB-direct 2x2",
        arithmetic: false,
        rgb_direct: true,
        rgb_subsampled: true,
    },
    Mode {
        label: "arithmetic RGB-direct 2x2",
        arithmetic: true,
        rgb_direct: true,
        rgb_subsampled: true,
    },
];

impl Mode {
    /// RGB-direct has no grayscale form: `cjpeg -rgb` on a PGM is not a
    /// grayscale encode.
    fn covers(&self, grayscale: bool) -> bool {
        !(grayscale && self.rgb_direct)
    }

    fn apply<'a>(&self, mut encoder: Encoder<'a>) -> Encoder<'a> {
        if self.arithmetic {
            encoder = encoder.arithmetic(true);
        }
        if self.rgb_direct {
            encoder = encoder.colorspace(ColorSpace::Rgb);
            if self.rgb_subsampled {
                encoder = encoder.subsampling(Subsampling::S420);
            }
        }
        encoder
    }

    fn cjpeg_args(&self) -> Vec<&'static str> {
        let mut args: Vec<&'static str> = Vec::new();
        if self.arithmetic {
            args.push("-arithmetic");
        }
        if self.rgb_direct {
            args.push("-rgb");
            if self.rgb_subsampled {
                args.extend(["-sample", "2x2,1x1,1x1"]);
            }
        }
        args
    }
}

fn encode_rgb_in(mode: Mode, script: Vec<ScanScript>) -> Result<Vec<u8>, JpegError> {
    let pixels: Vec<u8> = helpers::generate_gradient(SIDE, SIDE);
    mode.apply(Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb))
        .quality(75)
        .progressive(true)
        .scan_script(script)
        .encode()
}

fn encode_gray_in(mode: Mode, script: Vec<ScanScript>) -> Result<Vec<u8>, JpegError> {
    let pixels: Vec<u8> = helpers::generate_gradient_gray(SIDE, SIDE);
    mode.apply(Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Grayscale))
        .quality(75)
        .progressive(true)
        .scan_script(script)
        .encode()
}

fn encode_rgb(script: Vec<ScanScript>) -> Result<Vec<u8>, JpegError> {
    encode_rgb_in(HUFFMAN_YCBCR, script)
}

/// The 1-based script entry a refusal names (0 = the script as a whole), or a
/// panic naming whatever else came back.
fn refused_entry(result: Result<Vec<u8>, JpegError>, label: &str) -> usize {
    match result {
        Err(JpegError::InvalidScanScript { entry, .. }) => entry,
        Err(other) => panic!("{label}: expected InvalidScanScript, got {other:?}"),
        Ok(jpeg) => panic!(
            "{label}: expected InvalidScanScript, got Ok with {} bytes",
            jpeg.len()
        ),
    }
}

/// A script in `cjpeg -scans` syntax: `c0 c1: Ss Se Ah Al;` per entry.
fn cjpeg_scan_file(script: &[ScanScript]) -> String {
    script
        .iter()
        .map(|entry: &ScanScript| {
            let components: Vec<String> = entry
                .components
                .iter()
                .map(|component: &u8| component.to_string())
                .collect();
            format!(
                "{}: {} {} {} {};\n",
                components.join(" "),
                entry.ss,
                entry.se,
                entry.ah,
                entry.al
            )
        })
        .collect()
}

/// What `cjpeg -scans` says about `script` in `mode`: the stream it writes
/// when it encodes, or the stderr text when it refuses. Quality 75 and
/// cjpeg's default sampling for the mode, which is what `encode_rgb_in` /
/// `encode_gray_in` ask for.
fn cjpeg_verdict(
    cjpeg: &std::path::Path,
    mode: Mode,
    script: &[ScanScript],
    grayscale: bool,
) -> Result<Vec<u8>, String> {
    let input = helpers::TempFile::new(if grayscale { "scan.pgm" } else { "scan.ppm" });
    if grayscale {
        helpers::write_pgm_file(
            input.path(),
            SIDE,
            SIDE,
            &helpers::generate_gradient_gray(SIDE, SIDE),
        );
    } else {
        helpers::write_ppm_file(
            input.path(),
            SIDE,
            SIDE,
            &helpers::generate_gradient(SIDE, SIDE),
        );
    }
    let scans = helpers::TempFile::new("scans.txt");
    scans.write_bytes(cjpeg_scan_file(script).as_bytes());
    let output_jpeg = helpers::TempFile::new("scan.jpg");
    let output: std::process::Output = Command::new(cjpeg)
        .args(mode.cjpeg_args())
        .arg("-scans")
        .arg(scans.path())
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

/// A script C refuses, the entry C names, and C's message for it.
struct RefusedCase {
    label: &'static str,
    grayscale: bool,
    script: Vec<ScanScript>,
    entry: usize,
    c_message: String,
}

fn bad_prog(entry: usize) -> String {
    format!("Invalid progressive/lossless parameters at scan script entry {entry}")
}

fn bad_scan(entry: usize) -> String {
    format!("Invalid scan script at entry {entry}")
}

fn dc_all() -> ScanScript {
    scan(&[0, 1, 2], 0, 0, 0, 0)
}

/// Every refusal `validate_script` makes for a progressive script that the
/// `cjpeg -scans` file format can express. Each script is otherwise valid, so
/// the rule under test is the only one it breaks — except the `ah` ceiling,
/// which cannot be broken alone: the refinement chain already requires
/// `ah` to equal an earlier `al`, itself at most 10.
fn refused_cases() -> Vec<RefusedCase> {
    vec![
        RefusedCase {
            label: "issue #610 reproducer: AC-first band with se = 200",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 1, 200, 0, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "smallest overrun: se = 64",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 1, 64, 0, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "AC-refine band with se = 64",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 1, 63, 0, 1), scan(&[0], 1, 64, 1, 0)],
            entry: 3,
            c_message: bad_prog(3),
        },
        RefusedCase {
            label: "ss = 64",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 64, 64, 0, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "se < ss",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 5, 3, 0, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "DC and AC in one progressive scan",
            grayscale: false,
            // First, so no earlier DC scan makes the refinement chain fail
            // too; the second entry only keeps the script otherwise complete.
            script: vec![scan(&[0], 0, 5, 0, 0), scan(&[1, 2], 0, 0, 0, 0)],
            entry: 1,
            c_message: bad_prog(1),
        },
        RefusedCase {
            label: "AC scan with two components",
            grayscale: false,
            script: vec![dc_all(), scan(&[0, 1], 1, 63, 0, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "AC scan before that component's DC",
            grayscale: false,
            script: vec![scan(&[0], 0, 0, 0, 0), scan(&[1], 1, 63, 0, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "first scan of a band with ah != 0",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 1, 63, 1, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "refinement whose al is not ah - 1",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 1, 63, 0, 2), scan(&[0], 1, 63, 2, 0)],
            entry: 3,
            c_message: bad_prog(3),
        },
        RefusedCase {
            label: "refinement whose ah is not the previous al",
            grayscale: false,
            script: vec![dc_all(), scan(&[0], 1, 63, 0, 2), scan(&[0], 1, 63, 1, 0)],
            entry: 3,
            c_message: bad_prog(3),
        },
        RefusedCase {
            label: "al above the 8-bit ceiling of 10",
            grayscale: false,
            script: vec![scan(&[0, 1, 2], 0, 0, 0, 11)],
            entry: 1,
            c_message: bad_prog(1),
        },
        RefusedCase {
            label: "ah above the 8-bit ceiling of 10",
            grayscale: false,
            script: vec![dc_all(), scan(&[0, 1, 2], 0, 0, 11, 10)],
            entry: 2,
            c_message: bad_prog(2),
        },
        RefusedCase {
            label: "component index past the frame",
            grayscale: false,
            script: vec![scan(&[0, 1, 3], 0, 0, 0, 0)],
            entry: 1,
            c_message: bad_scan(1),
        },
        RefusedCase {
            label: "components out of frame order",
            grayscale: false,
            script: vec![scan(&[0, 2, 1], 0, 0, 0, 0)],
            entry: 1,
            c_message: bad_scan(1),
        },
        RefusedCase {
            label: "a component repeated within a scan",
            grayscale: false,
            script: vec![scan(&[0, 0, 1], 0, 0, 0, 0)],
            entry: 1,
            c_message: bad_scan(1),
        },
        RefusedCase {
            label: "a component that never gets DC",
            grayscale: false,
            script: vec![scan(&[0, 1], 0, 0, 0, 0), scan(&[0], 1, 63, 0, 0)],
            entry: 0,
            c_message: "Scan script does not transmit all data".to_string(),
        },
        RefusedCase {
            label: "grayscale: component 1 does not exist",
            grayscale: true,
            script: vec![scan(&[1], 0, 0, 0, 0)],
            entry: 1,
            c_message: bad_scan(1),
        },
        RefusedCase {
            label: "grayscale: se = 64",
            grayscale: true,
            script: vec![scan(&[0], 0, 0, 0, 0), scan(&[0], 1, 64, 0, 0)],
            entry: 2,
            c_message: bad_prog(2),
        },
    ]
}

/// Scripts C accepts: the encode must succeed, decode to the frame size and
/// match `cjpeg -scans` byte for byte.
fn accepted_cases() -> Vec<(&'static str, bool, Vec<ScanScript>)> {
    vec![
        (
            "DC with al = 10 (the 8-bit ceiling), then its refinements",
            false,
            vec![scan(&[0, 1, 2], 0, 0, 0, 10), scan(&[0, 1, 2], 0, 0, 10, 9)],
        ),
        (
            "full spectral band 1..63 per component",
            false,
            vec![
                dc_all(),
                scan(&[0], 1, 63, 0, 0),
                scan(&[1], 1, 63, 0, 0),
                scan(&[2], 1, 63, 0, 0),
            ],
        ),
        (
            "split luma bands and a full successive-approximation chain",
            false,
            vec![
                scan(&[0, 1, 2], 0, 0, 0, 1),
                scan(&[0], 1, 5, 0, 2),
                scan(&[0], 6, 63, 0, 2),
                scan(&[0], 1, 63, 2, 1),
                scan(&[0], 1, 63, 1, 0),
                scan(&[0, 1, 2], 0, 0, 1, 0),
            ],
        ),
        (
            "DC only: C transmits only DC and still accepts it",
            false,
            vec![dc_all()],
        ),
        (
            "per-component DC scans (non-interleaved)",
            false,
            vec![
                scan(&[0], 0, 0, 0, 0),
                scan(&[1], 0, 0, 0, 0),
                scan(&[2], 0, 0, 0, 0),
            ],
        ),
        (
            "non-interleaved DC first and refinement, then AC",
            false,
            vec![
                scan(&[0], 0, 0, 0, 1),
                scan(&[1, 2], 0, 0, 0, 1),
                scan(&[0], 1, 63, 0, 0),
                scan(&[0], 0, 0, 1, 0),
                scan(&[1], 0, 0, 1, 0),
                scan(&[2], 0, 0, 1, 0),
                scan(&[1], 1, 63, 0, 0),
                scan(&[2], 1, 63, 0, 0),
            ],
        ),
        (
            "grayscale: single-coefficient band at 63",
            true,
            vec![scan(&[0], 0, 0, 0, 0), scan(&[0], 63, 63, 0, 0)],
        ),
    ]
}

fn encode_case(mode: Mode, grayscale: bool, script: Vec<ScanScript>) -> Result<Vec<u8>, JpegError> {
    if grayscale {
        encode_gray_in(mode, script)
    } else {
        encode_rgb_in(mode, script)
    }
}

/// Issue #610: the reproducer from the issue, on its own so a failure names it.
///
/// Before the fix this returned `Ok` on x86_64 release builds after writing
/// 200 `u16` into each of two `[u16; 64]` stack arrays, and panicked on the
/// scalar arm. The CI x86_64 legs are what make this discriminating for the
/// SSE2 path; the aarch64 legs prove the scalar arm agrees.
#[test]
fn issue_610_se_past_63_is_refused_not_written() {
    let script: Vec<ScanScript> = vec![dc_all(), scan(&[0], 1, 200, 0, 0)];
    assert_eq!(refused_entry(encode_rgb(script), "se = 200"), 2);
    let script: Vec<ScanScript> = vec![dc_all(), scan(&[0], 1, 64, 0, 0)];
    assert_eq!(refused_entry(encode_rgb(script), "se = 64"), 2);
}

/// Every refusal agrees with `cjpeg -scans` on the entry it names, in every
/// mode (issue #636: the arithmetic and RGB-direct encodes used to return
/// `Ok` for all of these).
#[test]
fn refusals_match_cjpeg_entry_for_entry() {
    let cjpeg = require_c_tool!("cjpeg");
    for mode in MODES {
        for case in refused_cases() {
            if !mode.covers(case.grayscale) {
                continue;
            }
            let label: String = format!("{} / {}", mode.label, case.label);
            let entry: usize = refused_entry(
                encode_case(mode, case.grayscale, case.script.clone()),
                &label,
            );
            assert_eq!(entry, case.entry, "{label}: Rust named the wrong entry");
            match cjpeg_verdict(&cjpeg, mode, &case.script, case.grayscale) {
                Ok(_) => panic!("{label}: cjpeg accepted a script this case says C refuses"),
                Err(stderr) => assert!(
                    stderr.contains(&case.c_message),
                    "{label}: cjpeg refused with {stderr:?}, expected {:?}",
                    case.c_message
                ),
            }
        }
    }
}

/// The refusals hold without the C tool too, so a machine without `cjpeg`
/// still runs every Rust-side assertion.
#[test]
fn refusals_name_the_offending_entry() {
    for mode in MODES {
        for case in refused_cases() {
            if !mode.covers(case.grayscale) {
                continue;
            }
            let label: String = format!("{} / {}", mode.label, case.label);
            let entry: usize =
                refused_entry(encode_case(mode, case.grayscale, case.script), &label);
            assert_eq!(entry, case.entry, "{label}");
        }
    }
}

/// Every script C accepts is accepted here, decodes to the frame, and is
/// byte-identical to `cjpeg -scans`'s stream — in every mode (issue #636: the
/// arithmetic and RGB-direct encodes used to write C's default script).
#[test]
fn accepted_scripts_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    for mode in MODES {
        for (case_label, grayscale, script) in accepted_cases() {
            if !mode.covers(grayscale) {
                continue;
            }
            let label: String = format!("{} / {case_label}", mode.label);
            let c_jpeg: Vec<u8> =
                cjpeg_verdict(&cjpeg, mode, &script, grayscale).unwrap_or_else(|stderr| {
                    panic!("{label}: cjpeg refused a script this case says C accepts: {stderr}")
                });
            let jpeg: Vec<u8> = encode_case(mode, grayscale, script)
                .unwrap_or_else(|error| panic!("{label}: Rust refused a valid script: {error}"));
            let image = decompress(&jpeg).unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!((image.width, image.height), (SIDE, SIDE), "{label}");
            assert!(
                jpeg == c_jpeg,
                "{label}: Rust and cjpeg streams differ ({} vs {} bytes)",
                jpeg.len(),
                c_jpeg.len()
            );
        }
    }
}

/// Refusals that `cjpeg -scans` cannot express, because its file reader stops
/// them first (`rdswitch.c` caps a scan at four components and needs at least
/// one) or because the file cannot be empty. These take C's
/// `JERR_COMPONENT_COUNT` and `JERR_BAD_SCAN_SCRIPT` arms of
/// `validate_script`, which the library reaches through `jpeg_scan_info`.
#[test]
fn refusals_the_cjpeg_file_format_cannot_express() {
    assert_eq!(refused_entry(encode_rgb(Vec::new()), "empty script"), 0);
    assert_eq!(
        refused_entry(
            encode_rgb(vec![scan(&[], 0, 0, 0, 0)]),
            "scan with no components"
        ),
        1
    );
    assert_eq!(
        refused_entry(
            encode_rgb(vec![scan(&[0, 1, 2, 0, 1], 0, 0, 0, 0)]),
            "five components in one scan"
        ),
        1
    );
}

/// The builder's script always selects progressive coding, so the two shapes
/// with which C's *first* entry selects another mode are refused rather than
/// reinterpreted: `Ss = 0, Se = 63` (sequential in C) breaks the
/// progressive DC/AC separation rule, and `Ss != 0, Se = 0` (lossless in C)
/// breaks `Se >= Ss`. Deliberately not cross-validated — C would encode
/// these in a mode this API does not offer through `scan_script`.
#[test]
fn sequential_and_lossless_shaped_scripts_are_refused() {
    assert_eq!(
        refused_entry(
            encode_rgb(vec![scan(&[0, 1, 2], 0, 63, 0, 0)]),
            "sequential-shaped"
        ),
        1
    );
    assert_eq!(
        refused_entry(
            encode_rgb(vec![scan(&[0, 1, 2], 1, 0, 0, 0)]),
            "lossless-shaped"
        ),
        1
    );
}

/// Issue #636: where no encode path can follow a script, the builder refuses
/// it rather than writing a stream that ignores it.
///
/// * Without `progressive(true)` the builder encodes a sequential stream; it
///   does not switch mode on the caller's behalf (in C the script itself
///   selects the mode, `jcmaster.c` `validate_script`).
/// * `lossless(true)` has no progressive form for a script to describe.
#[test]
fn issue_636_scripts_no_path_can_honour_are_refused() {
    let script: Vec<ScanScript> = vec![dc_all(), scan(&[0], 1, 63, 0, 0)];
    let pixels: Vec<u8> = helpers::generate_gradient(SIDE, SIDE);
    let unsupported = |result: Result<Vec<u8>, JpegError>, label: &str| match result {
        Err(JpegError::Unsupported(message)) => assert!(
            message.contains("scan_script"),
            "{label}: message does not name the option: {message}"
        ),
        Err(other) => panic!("{label}: expected Unsupported, got {other:?}"),
        Ok(jpeg) => panic!(
            "{label}: expected Unsupported, got Ok with {} bytes",
            jpeg.len()
        ),
    };
    unsupported(
        Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb)
            .scan_script(script.clone())
            .encode(),
        "progressive off",
    );
    unsupported(
        Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb)
            .arithmetic(true)
            .scan_script(script.clone())
            .encode(),
        "arithmetic, progressive off",
    );
    unsupported(
        Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb)
            .lossless(true)
            .progressive(true)
            .scan_script(script.clone())
            .encode(),
        "lossless",
    );
    // Without a script each of these still encodes as before.
    for (label, encoder) in [
        (
            "custom sampling",
            Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb).sampling_factors(vec![
                (3, 2),
                (1, 1),
                (1, 1),
            ]),
        ),
        (
            "sequential",
            Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb),
        ),
        (
            "lossless",
            Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb).lossless(true),
        ),
    ] {
        encoder
            .encode()
            .unwrap_or_else(|error| panic!("{label} without a script: {error}"));
    }
}

/// Issue #636: an invalid script is refused on the arithmetic and RGB-direct
/// paths with the typed error, ahead of the frame-size checks, as on the
/// Huffman path (`validate_script` runs before `initial_setup` in
/// `jcmaster.c`).
#[test]
fn issue_636_invalid_script_outranks_frame_errors_on_every_path() {
    let script: Vec<ScanScript> = vec![dc_all(), scan(&[0], 1, 64, 0, 0)];
    for mode in MODES {
        let result = mode
            .apply(Encoder::new(&[], 0, 0, PixelFormat::Rgb))
            .progressive(true)
            .scan_script(script.clone())
            .encode();
        assert_eq!(refused_entry(result, mode.label), 2, "{}", mode.label);
    }
}
