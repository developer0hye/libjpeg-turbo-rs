//! P4-247, part 1: the arbitrary-precision lossless decoders ignored
//! restart intervals.
//!
//! `precision::decompress_lossless_arbitrary` and its 16-bit twin
//! `precision::decompress_16bit` read the DRI interval but never acted on
//! it: they neither consumed the RSTn markers nor reset prediction. T.81
//! H.1.2.1 and libjpeg (`jddiffct.c` `process_restart`, which re-runs the
//! undifferencer's `start_pass`) decode the first row of every restart
//! interval like the first row of the scan. `cjpeg -lossless 1 -restart 1`
//! on a 33×17 greyscale image decoded with 527 of 561 samples different
//! from `djpeg`, and returned `Ok`.
//!
//! T.81 H.1.1 also requires a lossless restart interval to be a whole
//! number of MCU rows. libjpeg refuses any other, when decoding
//! (`jddiffct.c` `start_input_pass`) and when encoding (`jclossls.c`
//! `start_pass`), with `JERR_BAD_RESTART`. The three Huffman lossless
//! decoders now refuse it too: `Decoder` used to restart prediction mid-row
//! and returned `Ok` with 550 of 561 samples wrong on the stream below. The
//! lossless encoder refuses it as well; it used to write such a stream,
//! which `djpeg` refuses.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::precision::{decompress_16bit, decompress_lossless_arbitrary, Image16};
use libjpeg_turbo_rs::{Encoder, PixelFormat};

const WIDTH: usize = 33;
const HEIGHT: usize = 17;

/// A PGM (one component) or PPM (three) at `precision` bits: a gradient
/// plus xorshift noise, so a prediction reset in the wrong place changes
/// most of the samples after it.
fn pnm(components: usize, precision: u8) -> Vec<u8> {
    let maxval: u32 = (1u32 << precision) - 1;
    let magic: &str = if components == 1 { "P5" } else { "P6" };
    let mut out: Vec<u8> = format!("{magic}\n{WIDTH} {HEIGHT}\n{maxval}\n").into_bytes();
    let mut state: u32 = 0x9E37_79B9 ^ u32::from(precision);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            for c in 0..components {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let gradient: u32 =
                    (x as u32 * 977 + y as u32 * 2311 + c as u32 * 4099) % (maxval + 1);
                let value: u32 = (gradient + state % (maxval / 4 + 1)) % (maxval + 1);
                if maxval > 255 {
                    out.extend_from_slice(&(value as u16).to_be_bytes());
                } else {
                    out.push(value as u8);
                }
            }
        }
    }
    out
}

fn cjpeg_supports_lossless_precision(cjpeg: &Path) -> bool {
    Command::new(cjpeg)
        .arg("-help")
        .output()
        .is_ok_and(|output| {
            let text: String = String::from_utf8_lossy(&output.stderr).to_string()
                + &String::from_utf8_lossy(&output.stdout);
            text.contains("-lossless") && text.contains("-precision")
        })
}

fn tools() -> Option<(PathBuf, PathBuf)> {
    let cjpeg: PathBuf = helpers::require_c_tool("cjpeg").ok()?;
    let djpeg: PathBuf = helpers::require_c_tool("djpeg").ok()?;
    Some((cjpeg, djpeg))
}

/// Positions where `got` differs from djpeg's samples (read as `u16`).
fn differing(got: &Image16, expected: &[i16]) -> usize {
    assert_eq!(got.data.len(), expected.len());
    got.data
        .iter()
        .zip(expected)
        .filter(|(&a, &b)| a != b as u16)
        .count()
}

/// Every precision class (2, 8, 12, 16 bits), greyscale and interleaved
/// RGB, three predictor / point-transform pairs, and restart intervals of
/// 0, 1, 2 and 20 MCU rows (the last longer than the image) decode
/// identically to `djpeg`, through `decompress_lossless_arbitrary` and, at
/// 16 bits, `decompress_16bit`.
#[test]
fn lossless_restart_intervals_match_djpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    if !cjpeg_supports_lossless_precision(&cjpeg) {
        helpers::skip_missing_c_capability("cjpeg", "-lossless with -precision");
        return;
    }
    let mut failures: Vec<String> = Vec::new();
    let mut cases: usize = 0;
    for precision in [2u8, 8, 12, 16] {
        for components in [1usize, 3] {
            let source: Vec<u8> = pnm(components, precision);
            for (psv, pt) in [(1u8, 0u8), (4, 0), (7, 1)] {
                for restart_rows in ["0", "1", "2", "20"] {
                    let label: String =
                        format!("p{precision}_c{components}_psv{psv}_pt{pt}_rst{restart_rows}");
                    let precision_arg: String = precision.to_string();
                    let lossless_arg: String = format!("{psv},{pt}");
                    let mut args: Vec<&str> = vec![
                        "-precision",
                        &precision_arg,
                        "-lossless",
                        &lossless_arg,
                        "-restart",
                        restart_rows,
                    ];
                    if components == 3 {
                        args.push("-rgb");
                    }
                    let jpeg: Vec<u8> =
                        helpers::encode_with_c_cjpeg(&cjpeg, &source, &args, &label);
                    let (w, h, nc, _maxval, expected) =
                        helpers::decode_with_c_djpeg_i16(&djpeg, &jpeg, &label);
                    assert_eq!((w, h, nc), (WIDTH, HEIGHT, components), "{label}");
                    cases += 1;

                    match decompress_lossless_arbitrary(&jpeg) {
                        Ok(image) => {
                            let wrong: usize = differing(&image, &expected);
                            if wrong != 0 {
                                failures.push(format!(
                                    "{label}: decompress_lossless_arbitrary: {wrong} of {} \
                                     samples differ",
                                    expected.len()
                                ));
                            }
                        }
                        Err(e) => failures.push(format!(
                            "{label}: decompress_lossless_arbitrary refused: {e}"
                        )),
                    }
                    if precision == 16 {
                        match decompress_16bit(&jpeg) {
                            Ok(image) => {
                                let wrong: usize = differing(&image, &expected);
                                if wrong != 0 {
                                    failures.push(format!(
                                        "{label}: decompress_16bit: {wrong} of {} samples differ",
                                        expected.len()
                                    ));
                                }
                            }
                            Err(e) => {
                                failures.push(format!("{label}: decompress_16bit refused: {e}"))
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} cases diverge from djpeg:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A lossless restart interval that is not a whole number of MCU rows is
/// refused, as libjpeg refuses it (`JERR_BAD_RESTART`, "Invalid restart
/// interval 10; must be an integer multiple of the number of MCUs in an
/// MCU row (33)").
#[test]
fn lossless_partial_row_restart_interval_is_refused_like_djpeg() {
    let Some((cjpeg, djpeg)) = tools() else {
        eprintln!("SKIP: cjpeg/djpeg not found");
        assert!(!helpers::is_ci(), "CI requires cjpeg and djpeg");
        return;
    };
    if !cjpeg_supports_lossless_precision(&cjpeg) {
        helpers::skip_missing_c_capability("cjpeg", "-lossless with -precision");
        return;
    }
    for precision in [8u8, 16] {
        let label: String = format!("partial_row_restart_p{precision}");
        let precision_arg: String = precision.to_string();
        let mut jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
            &cjpeg,
            &pnm(1, precision),
            &[
                "-precision",
                &precision_arg,
                "-lossless",
                "1",
                "-restart",
                "1",
            ],
            &label,
        );
        // DRI = 33 (one MCU row) as written; declare 10 instead.
        let dri: usize = jpeg
            .windows(4)
            .position(|w| w == [0xFF, 0xDD, 0x00, 0x04])
            .expect("cjpeg -restart writes a DRI segment");
        assert_eq!(&jpeg[dri + 4..dri + 6], &[0x00, WIDTH as u8], "{label}");
        jpeg[dri + 5] = 10;

        let file: helpers::TempFile = helpers::TempFile::new(&format!("{label}.jpg"));
        file.write_bytes(&jpeg);
        let c: std::process::Output = Command::new(&djpeg)
            .arg("-outfile")
            .arg(if cfg!(windows) { "NUL" } else { "/dev/null" })
            .arg(file.path())
            .output()
            .expect("run djpeg");
        let c_stderr = String::from_utf8_lossy(&c.stderr);
        assert!(
            !c.status.success(),
            "{label}: djpeg must refuse the interval: {c_stderr}"
        );
        // The official Linux .deb builds of 3.1.4.1 and 3.2.0 refuse the
        // 16-bit stream with the wrong text ("Corrupt JPEG data: bad ICC
        // marker"; the stream has no APP2), so the message is pinned only at
        // 8 bits. The unpatched stream decodes, so the refusal is the DRI's.
        if precision == 8 {
            assert!(
                c_stderr.contains("Invalid restart interval"),
                "{label}: djpeg must refuse with JERR_BAD_RESTART: {c_stderr}"
            );
        }

        let err = decompress_lossless_arbitrary(&jpeg)
            .err()
            .unwrap_or_else(|| panic!("{label}: decompress_lossless_arbitrary accepted it"));
        assert!(
            err.to_string().contains("restart interval"),
            "{label}: {err}"
        );
        if precision == 8 {
            let err = libjpeg_turbo_rs::decompress(&jpeg)
                .err()
                .unwrap_or_else(|| panic!("{label}: decompress accepted it"));
            assert!(
                err.to_string().contains("restart interval"),
                "{label}: {err}"
            );
        }
        if precision == 16 {
            let err = decompress_16bit(&jpeg)
                .err()
                .unwrap_or_else(|| panic!("{label}: decompress_16bit accepted it"));
            assert!(
                err.to_string().contains("restart interval"),
                "{label}: {err}"
            );
        }
    }
}

/// The lossless encoder writes whole-row restart intervals that `djpeg` and
/// every Rust lossless decoder decode back to the source, and refuses an
/// interval that is not a whole number of rows, as `cjpeg` does.
#[test]
fn lossless_encoder_restart_intervals_follow_libjpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    for (format, components) in [(PixelFormat::Grayscale, 1usize), (PixelFormat::Rgb, 3)] {
        let source: Vec<u8> = pnm(components, 8);
        let pixels: &[u8] = &source[source.len() - WIDTH * HEIGHT * components..];
        for restart_rows in [1u16, 2] {
            let label: String = format!("encode_c{components}_rst{restart_rows}");
            let jpeg: Vec<u8> = Encoder::new(pixels, WIDTH, HEIGHT, format)
                .lossless(true)
                .restart_rows(restart_rows)
                .encode()
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let c_pixels: Vec<u8> = if components == 1 {
                helpers::decode_gray_with_c_djpeg(&djpeg, &jpeg, &label).2
            } else {
                helpers::decode_with_c_djpeg(&djpeg, &jpeg, &label).2
            };
            assert!(
                c_pixels == pixels,
                "{label}: djpeg does not decode the source"
            );
            let image =
                libjpeg_turbo_rs::decompress(&jpeg).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(
                image.data == pixels,
                "{label}: decompress does not decode the source"
            );
            let image16: Image16 =
                decompress_lossless_arbitrary(&jpeg).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(
                image16
                    .data
                    .iter()
                    .map(|&v| v as u8)
                    .eq(pixels.iter().copied()),
                "{label}: decompress_lossless_arbitrary does not decode the source"
            );
        }
        let err = Encoder::new(pixels, WIDTH, HEIGHT, format)
            .lossless(true)
            .restart_blocks(10)
            .encode()
            .expect_err("a 10-MCU interval on a 33-MCU row must be refused");
        assert!(err.to_string().contains("restart interval"), "{err}");
    }
}
