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

mod helpers;

use std::process::Command;

use libjpeg_turbo_rs::{decompress, Encoder, JpegError, PixelFormat, ScanScript};

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

fn encode_rgb(script: Vec<ScanScript>) -> Result<Vec<u8>, JpegError> {
    let pixels: Vec<u8> = helpers::generate_gradient(SIDE, SIDE);
    Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Rgb)
        .quality(75)
        .progressive(true)
        .scan_script(script)
        .encode()
}

fn encode_gray(script: Vec<ScanScript>) -> Result<Vec<u8>, JpegError> {
    let pixels: Vec<u8> = helpers::generate_gradient_gray(SIDE, SIDE);
    Encoder::new(&pixels, SIDE, SIDE, PixelFormat::Grayscale)
        .quality(75)
        .progressive(true)
        .scan_script(script)
        .encode()
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

/// What `cjpeg -scans` says about `script`: the stream it writes when it
/// encodes, or the stderr text when it refuses. Quality 75 and cjpeg's
/// default 4:2:0, which is what `encode_rgb` / `encode_gray` ask for.
fn cjpeg_verdict(
    cjpeg: &std::path::Path,
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

fn encode_case(grayscale: bool, script: Vec<ScanScript>) -> Result<Vec<u8>, JpegError> {
    if grayscale {
        encode_gray(script)
    } else {
        encode_rgb(script)
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

/// Every refusal agrees with `cjpeg -scans` on the entry it names.
#[test]
fn refusals_match_cjpeg_entry_for_entry() {
    let cjpeg = require_c_tool!("cjpeg");
    for case in refused_cases() {
        let entry: usize =
            refused_entry(encode_case(case.grayscale, case.script.clone()), case.label);
        assert_eq!(
            entry, case.entry,
            "{}: Rust named the wrong entry",
            case.label
        );
        match cjpeg_verdict(&cjpeg, &case.script, case.grayscale) {
            Ok(_) => panic!(
                "{}: cjpeg accepted a script this case says C refuses",
                case.label
            ),
            Err(stderr) => assert!(
                stderr.contains(&case.c_message),
                "{}: cjpeg refused with {stderr:?}, expected {:?}",
                case.label,
                case.c_message
            ),
        }
    }
}

/// The refusals hold without the C tool too, so a machine without `cjpeg`
/// still runs every Rust-side assertion.
#[test]
fn refusals_name_the_offending_entry() {
    for case in refused_cases() {
        let entry: usize = refused_entry(encode_case(case.grayscale, case.script), case.label);
        assert_eq!(entry, case.entry, "{}", case.label);
    }
}

/// Every script C accepts is accepted here, decodes to the frame, and is
/// byte-identical to `cjpeg -scans`'s stream.
#[test]
fn accepted_scripts_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    for (label, grayscale, script) in accepted_cases() {
        let c_jpeg: Vec<u8> = cjpeg_verdict(&cjpeg, &script, grayscale).unwrap_or_else(|stderr| {
            panic!("{label}: cjpeg refused a script this case says C accepts: {stderr}")
        });
        let jpeg: Vec<u8> = encode_case(grayscale, script)
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
