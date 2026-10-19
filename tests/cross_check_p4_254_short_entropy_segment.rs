//! P4-254: an entropy-coded segment that ends before its last MCU.
//!
//! C libjpeg-turbo calls this `JWRN_HIT_MARKER` ("Corrupt JPEG data: premature
//! end of data segment"): once a read needs more bits than the segment holds
//! (`jpeg_fill_bit_buffer`, jdhuff.c:361-373) it warns, and every later MCU is
//! left zero, which decodes as mid-grey (`decode_mcu`, jdhuff.c:787-799;
//! lossless `decode_mcus`, jdlhuff.c:224-239). A stream that simply stops gets
//! `JWRN_JPEG_EOF` and a fake EOI first (`fill_input_buffer`,
//! jdatasrc.c:109-113; `fill_mem_input_buffer`, :135-140, for a memory
//! source), so it decodes the same way. A restart marker after a short
//! interval starts a new segment (`process_restart`, jdhuff.c:533-539;
//! lossless jddiffct.c:178-183).
//!
//! Strict decoding refuses every such stream with `UnexpectedEof`. Lenient
//! decoding returns djpeg's pixels exactly, with one `TruncatedData` warning.
//! djpeg exits with `EXIT_WARNING` (2) on these inputs, so this file runs it
//! directly rather than through the helpers that require a zero status.

mod helpers;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{
    decompress_from_reader_incremental, precision, raw_data_12, read_coefficients, DecodeWarning,
    Decoder, JpegError,
};

/// Offset of the first byte of entropy-coded data after the first SOS.
fn entropy_start(jpeg: &[u8]) -> usize {
    let sos: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xDA])
        .expect("JPEG has an SOS marker");
    let length: usize = u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]) as usize;
    sos + 2 + length
}

/// The two shapes of a short segment, each cut at three points of the
/// entropy-coded data: the stream simply stops, or an EOI follows the cut.
fn short_variants(complete: &[u8]) -> Vec<(String, Vec<u8>)> {
    assert_eq!(
        &complete[complete.len() - 2..],
        &[0xFF, 0xD9],
        "ends in EOI"
    );
    let start: usize = entropy_start(complete);
    let entropy_len: usize = complete.len() - 2 - start;
    let mut variants: Vec<(String, Vec<u8>)> = Vec::new();
    for percent in [25usize, 60, 90] {
        let cut: Vec<u8> = complete[..start + entropy_len * percent / 100].to_vec();
        let mut with_eoi: Vec<u8> = cut.clone();
        with_eoi.extend_from_slice(&[0xFF, 0xD9]);
        variants.push((format!("cut at {percent}%"), cut));
        variants.push((format!("cut at {percent}% + EOI"), with_eoi));
    }
    variants
}

/// The positions of the RSTn markers in `jpeg`'s entropy-coded data.
fn restart_markers(jpeg: &[u8]) -> Vec<usize> {
    (entropy_start(jpeg)..jpeg.len() - 1)
        .filter(|&i| jpeg[i] == 0xFF && (0xD0..=0xD7).contains(&jpeg[i + 1]))
        .collect()
}

/// `complete` with the second half of each interval `k` in `intervals` (the
/// bytes between RSTk-1 and RSTk) dropped, every marker kept.
fn shorten_intervals(complete: &[u8], rst: &[usize], intervals: &[usize]) -> Vec<u8> {
    let mut short: Vec<u8> = Vec::new();
    let mut copied: usize = 0;
    for &k in intervals {
        let interval_start: usize = rst[k - 1] + 2;
        let interval_end: usize = rst[k];
        short.extend_from_slice(&complete[copied..interval_start]);
        short.extend_from_slice(
            &complete[interval_start..interval_start + (interval_end - interval_start) / 2],
        );
        copied = interval_end;
    }
    short.extend_from_slice(&complete[copied..]);
    short
}

/// Run djpeg, which must report the stream as short with `EXIT_WARNING` (2)
/// unless `expect_clean`, and return its samples (16-bit outputs as their
/// big-endian values, 8-bit outputs as is).
fn djpeg_samples(djpeg: &Path, jpeg: &[u8], expect_clean: bool, label: &str) -> Vec<u16> {
    let input: helpers::TempFile = helpers::TempFile::new(&format!("p4_254_{label}.jpg"));
    input.write_bytes(jpeg);
    let output: std::process::Output = Command::new(djpeg)
        .arg("-pnm")
        .arg(input.path())
        .output()
        .unwrap_or_else(|e| panic!("failed to run djpeg: {e:?}"));
    let expected_status: i32 = if expect_clean { 0 } else { 2 };
    assert_eq!(
        output.status.code(),
        Some(expected_status),
        "{label}: djpeg status (stderr: {})",
        String::from_utf8_lossy(&output.stderr)
    );
    let (_, _, _, _, samples) =
        helpers::parse_pnm_i16(&output.stdout).unwrap_or_else(|| panic!("{label}: djpeg PNM"));
    // `parse_pnm_i16` stores 16-bit samples as their bit pattern.
    samples.iter().map(|&s| s as u16).collect()
}

fn truncated_count(warnings: &[DecodeWarning]) -> usize {
    warnings
        .iter()
        .filter(|w| matches!(w, DecodeWarning::TruncatedData { .. }))
        .count()
}

fn widen(samples: &[u8]) -> Vec<u16> {
    samples.iter().map(|&s| u16::from(s)).collect()
}

fn label_of(case: &str) -> String {
    case.replace([' ', ',', '%', '+', '[', ']'], "_")
}

fn cjpeg_encode(cjpeg: &Path, source: &Path, args: &[&str], label: &str) -> Vec<u8> {
    let pnm: Vec<u8> = std::fs::read(source).unwrap_or_else(|e| panic!("{source:?}: {e}"));
    helpers::encode_with_c_cjpeg(cjpeg, &pnm, args, label)
}

/// An 8-bit greyscale PGM of `testorig`, for one-component lossless fixtures:
/// `cjpeg -grayscale` has no effect in lossless mode, which keeps every
/// input component.
fn testorig_pgm(djpeg: &Path) -> Vec<u8> {
    let testorig: PathBuf = require_c_testimage_path("testorig.jpg");
    let output: std::process::Output = Command::new(djpeg)
        .args(["-grayscale", "-pnm"])
        .arg(&testorig)
        .output()
        .expect("run djpeg");
    assert!(output.status.success(), "djpeg -grayscale");
    assert_eq!(&output.stdout[..2], b"P5", "a greyscale PGM");
    output.stdout
}

fn require_c_testimage_path(name: &str) -> PathBuf {
    helpers::require_c_testimage(name).unwrap_or_else(|e| panic!("{name}: {e:?}"))
}

/// The 8-bit `Decoder` refuses each short stream in strict mode and, in
/// lenient mode, returns djpeg's output exactly with one `TruncatedData`.
fn assert_decoder_matches_djpeg(djpeg: &Path, complete: &[u8], label: &str) {
    assert_decoder_matches_djpeg_on(djpeg, short_variants(complete), label);
}

fn assert_decoder_matches_djpeg_on(djpeg: &Path, variants: Vec<(String, Vec<u8>)>, label: &str) {
    for (shape, jpeg) in variants {
        let case: String = format!("{label}, {shape}");
        let strict = Decoder::new(&jpeg).and_then(|d| d.decode_image());
        assert!(
            matches!(strict, Err(JpegError::UnexpectedEof)),
            "{case}: strict decode must refuse a short segment, got {:?}",
            strict.as_ref().map(|image| &image.warnings)
        );
        let lenient = Decoder::new(&jpeg)
            .and_then(|d| d.with_lenient(true).decode_image())
            .unwrap_or_else(|e| panic!("{case}: lenient decode failed: {e}"));
        assert_eq!(
            truncated_count(&lenient.warnings),
            1,
            "{case}: {:?}",
            lenient.warnings
        );
        let reference: Vec<u16> = djpeg_samples(djpeg, &jpeg, false, &label_of(&case));
        assert!(
            widen(&lenient.data) == reference,
            "{case}: lenient pixels must equal djpeg's"
        );
    }
}

/// With a restart marker per MCU row, losing the tail of an interval leaves
/// that interval short but every later one intact: djpeg greys the rest of
/// the short interval and resumes at the next RST. One and two short
/// intervals, through the `Decoder` and, for lossless, through
/// `decompress_lossless_arbitrary` as well.
fn assert_restart_resumes_like_djpeg(djpeg: &Path, complete: &[u8], label: &str, lossless: bool) {
    let rst: Vec<usize> = restart_markers(complete);
    assert!(
        rst.len() >= 6,
        "{label}: expected several restart intervals"
    );
    for intervals in [vec![rst.len() / 2], vec![rst.len() / 3, 2 * rst.len() / 3]] {
        let case: String = format!("{label}, short intervals {intervals:?}");
        let short: Vec<u8> = shorten_intervals(complete, &rst, &intervals);
        let reference: Vec<u16> = djpeg_samples(djpeg, &short, false, &label_of(&case));

        let strict = Decoder::new(&short).and_then(|d| d.decode_image());
        assert!(matches!(strict, Err(JpegError::UnexpectedEof)), "{case}");
        let lenient = Decoder::new(&short)
            .and_then(|d| d.with_lenient(true).decode_image())
            .unwrap_or_else(|e| panic!("{case}: lenient decode failed: {e}"));
        assert_eq!(
            truncated_count(&lenient.warnings),
            1,
            "{case}: TruncatedData at most once per decode"
        );
        assert!(
            widen(&lenient.data) == reference,
            "{case}: pixels must equal djpeg's"
        );
        // The intervals after the short ones are real image, not grey.
        let row: usize = lenient.data.len() / lenient.height;
        assert!(
            lenient.data[lenient.data.len() - row..]
                .iter()
                .any(|&s| s != 128),
            "{case}: decoding resumed after the RST"
        );

        if lossless {
            assert!(matches!(
                precision::decompress_lossless_arbitrary(&short),
                Err(JpegError::UnexpectedEof)
            ));
            let (image, warnings) = precision::decompress_lossless_arbitrary_lenient(&short)
                .unwrap_or_else(|e| panic!("{case}: arbitrary lenient: {e}"));
            assert_eq!(truncated_count(&warnings), 1, "{case}: arbitrary");
            assert!(image.data == reference, "{case}: arbitrary pixels");
        }
    }
}

#[test]
fn baseline_color_short_segment_matches_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let testorig: PathBuf = require_c_testimage!("testorig.jpg");
    let complete: Vec<u8> = std::fs::read(testorig).unwrap();
    assert_decoder_matches_djpeg(&djpeg, &complete, "testorig 4:2:0");
}

#[test]
fn baseline_grayscale_short_segment_matches_djpeg() {
    // A one-component scan takes the non-interleaved decode path.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let source: PathBuf = require_c_testimage!("testorig.ppm");
    let complete: Vec<u8> = cjpeg_encode(&cjpeg, &source, &["-grayscale"], "p4_254_gray");
    assert_decoder_matches_djpeg(&djpeg, &complete, "grayscale");
}

#[test]
fn multi_scan_short_segment_matches_djpeg() {
    // One scan per component: the non-interleaved path over several scans,
    // each with its own reader; components whose scans are lost stay grey.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let source: PathBuf = require_c_testimage!("testorig.ppm");
    let script: helpers::TempFile = helpers::TempFile::new("p4_254_scans.txt");
    script.write_bytes(b"0;\n1;\n2;\n");
    let script_path: String = script.path().to_string_lossy().into_owned();
    let complete: Vec<u8> =
        cjpeg_encode(&cjpeg, &source, &["-scans", &script_path], "p4_254_scans");
    assert_eq!(
        complete.windows(2).filter(|w| *w == [0xFF, 0xDA]).count(),
        3,
        "three scans"
    );
    // Only the cuts an EOI follows: a multi-scan stream that just stops fails
    // in the marker walk, before any entropy decoding, in both modes (P4-257).
    let with_eoi: Vec<(String, Vec<u8>)> = short_variants(&complete)
        .into_iter()
        .filter(|(shape, _)| shape.ends_with("+ EOI"))
        .collect();
    assert_decoder_matches_djpeg_on(&djpeg, with_eoi, "one scan per component");
}

#[test]
fn short_restart_interval_resumes_at_the_next_marker_like_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let source: PathBuf = require_c_testimage!("testorig.ppm");
    let complete: Vec<u8> = cjpeg_encode(&cjpeg, &source, &["-restart", "1"], "p4_254_rst");
    assert_restart_resumes_like_djpeg(&djpeg, &complete, "4:2:0 restart per row", false);
}

#[test]
fn twelve_bit_short_segment_matches_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let monkey: PathBuf = require_c_testimage!("monkey12.jpg");
    let complete: Vec<u8> = std::fs::read(monkey).unwrap();
    for (shape, jpeg) in short_variants(&complete) {
        let case: String = format!("12-bit, {shape}");
        assert!(
            matches!(
                precision::decompress_12bit(&jpeg),
                Err(JpegError::UnexpectedEof)
            ),
            "{case}: strict decompress_12bit must refuse it"
        );
        let (image, warnings) = precision::decompress_12bit_lenient(&jpeg)
            .unwrap_or_else(|e| panic!("{case}: lenient: {e}"));
        assert_eq!(truncated_count(&warnings), 1, "{case}: {warnings:?}");
        let samples: Vec<u16> = image.data.iter().map(|&s| s as u16).collect();
        let reference: Vec<u16> = djpeg_samples(&djpeg, &jpeg, false, &label_of(&case));
        assert!(samples == reference, "{case}: pixels must equal djpeg's");

        // The 8-bit `Decoder` reaches the same 12-bit decode.
        let strict = Decoder::new(&jpeg).and_then(|d| d.decode_image());
        assert!(matches!(strict, Err(JpegError::UnexpectedEof)), "{case}");
        let lenient = Decoder::new(&jpeg)
            .and_then(|d| d.with_lenient(true).decode_image())
            .unwrap_or_else(|e| panic!("{case}: Decoder lenient: {e}"));
        assert_eq!(
            truncated_count(&lenient.warnings),
            1,
            "{case}: Decoder lenient"
        );
    }
}

#[test]
fn raw_twelve_bit_short_segment_fills_mid_grey() {
    // No C tool emits raw 12-bit planes, so this pins the C behaviour the
    // pixel tests above prove: rows decoded before the short segment are
    // unchanged, and the image's last MCU row is CENTERJSAMPLE (2048).
    let monkey: PathBuf = require_c_testimage!("monkey12.jpg");
    let complete: Vec<u8> = std::fs::read(monkey).unwrap();
    let whole = raw_data_12::decompress_raw_12(&complete).expect("complete raw decode");
    for (shape, jpeg) in short_variants(&complete) {
        assert!(
            matches!(
                raw_data_12::decompress_raw_12(&jpeg),
                Err(JpegError::UnexpectedEof)
            ),
            "{shape}: strict decompress_raw_12 must refuse it"
        );
        let (raw, warnings) =
            raw_data_12::decompress_raw_12_lenient(&jpeg).expect("lenient raw decode");
        assert_eq!(truncated_count(&warnings), 1, "{shape}: {warnings:?}");
        for (c, plane) in raw.planes.iter().enumerate() {
            let width: usize = raw.plane_widths[c];
            assert_eq!(
                plane[..width * 8],
                whole.planes[c][..width * 8],
                "{shape}: top rows"
            );
            assert!(
                plane[plane.len() - width * 8..].iter().all(|&s| s == 2048),
                "{shape}: component {c}'s last MCU row must be mid-grey"
            );
        }
    }
}

/// 8-bit RGB, 8-bit greyscale and 16-bit greyscale lossless fixtures.
fn lossless_fixtures(djpeg: &Path, cjpeg: &Path) -> Vec<(&'static str, Vec<u8>)> {
    let rgb: PathBuf = require_c_testimage_path("testorig.ppm");
    let gray16: PathBuf = require_c_testimage_path("monkey16.pgm");
    let gray8: Vec<u8> = testorig_pgm(djpeg);
    vec![
        (
            "8-bit RGB lossless",
            cjpeg_encode(cjpeg, &rgb, &["-lossless", "1"], "ll_rgb8"),
        ),
        (
            "8-bit grey lossless",
            helpers::encode_with_c_cjpeg(cjpeg, &gray8, &["-lossless", "1"], "ll_gray8"),
        ),
        (
            "16-bit grey lossless",
            cjpeg_encode(
                cjpeg,
                &gray16,
                &["-precision", "16", "-lossless", "1"],
                "ll_gray16",
            ),
        ),
    ]
}

#[test]
fn lossless_short_segment_matches_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    for (label, complete) in lossless_fixtures(&djpeg, &cjpeg) {
        // The 8-bit `Decoder` has its own lossless path (grey and colour).
        if !label.starts_with("16-bit") {
            assert_decoder_matches_djpeg(&djpeg, &complete, label);
        }
        for (shape, jpeg) in short_variants(&complete) {
            let case: String = format!("{label}, {shape}");
            assert!(
                matches!(
                    precision::decompress_lossless_arbitrary(&jpeg),
                    Err(JpegError::UnexpectedEof)
                ),
                "{case}: strict decompress_lossless_arbitrary must refuse it"
            );
            if label.starts_with("16-bit") {
                assert!(
                    matches!(
                        precision::decompress_16bit(&jpeg),
                        Err(JpegError::UnexpectedEof)
                    ),
                    "{case}: strict decompress_16bit must refuse it"
                );
            }
            let (image, warnings) = precision::decompress_lossless_arbitrary_lenient(&jpeg)
                .unwrap_or_else(|e| panic!("{case}: lenient: {e}"));
            assert_eq!(truncated_count(&warnings), 1, "{case}: {warnings:?}");
            let reference: Vec<u16> = djpeg_samples(&djpeg, &jpeg, false, &label_of(&case));
            assert!(image.data == reference, "{case}: pixels must equal djpeg's");
        }
    }
}

#[test]
fn lossless_short_restart_interval_resumes_like_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let rgb: PathBuf = require_c_testimage!("testorig.ppm");
    let gray8: Vec<u8> = testorig_pgm(&djpeg);
    let rgb_rst: Vec<u8> = cjpeg_encode(
        &cjpeg,
        &rgb,
        &["-lossless", "1", "-restart", "1"],
        "ll_rgb_rst",
    );
    let gray_rst: Vec<u8> = helpers::encode_with_c_cjpeg(
        &cjpeg,
        &gray8,
        &["-lossless", "1", "-restart", "1"],
        "ll_gray_rst",
    );
    assert_restart_resumes_like_djpeg(&djpeg, &rgb_rst, "8-bit RGB lossless", true);
    assert_restart_resumes_like_djpeg(&djpeg, &gray_rst, "8-bit grey lossless", true);
}

#[test]
fn read_coefficients_refuses_a_short_segment() {
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let testorig: PathBuf = require_c_testimage!("testorig.jpg");
    let source: PathBuf = require_c_testimage!("testorig.ppm");
    let color: Vec<u8> = std::fs::read(testorig).unwrap();
    // One component takes `read_coefficients`' single-component loop.
    let gray: Vec<u8> = cjpeg_encode(&cjpeg, &source, &["-grayscale"], "p4_254_coef_gray");
    for (label, complete) in [("4:2:0", &color), ("grayscale", &gray)] {
        for (shape, jpeg) in short_variants(complete) {
            assert!(
                matches!(read_coefficients(&jpeg), Err(JpegError::UnexpectedEof)),
                "{label}, {shape}: read_coefficients must refuse a short segment"
            );
            // jpegtran reads the same coefficients and reports the stream short.
            let input: helpers::TempFile = helpers::TempFile::new("p4_254_coefficients.jpg");
            input.write_bytes(&jpeg);
            let status = Command::new(&jpegtran)
                .arg(input.path())
                .output()
                .expect("run jpegtran")
                .status;
            assert_eq!(
                status.code(),
                Some(2),
                "{label}, {shape}: jpegtran warns too"
            );
        }
    }
}

#[test]
fn incremental_reader_refuses_a_short_segment() {
    // The windowed incremental decode is strict.
    let testorig: PathBuf = require_c_testimage!("testorig.jpg");
    let complete: Vec<u8> = std::fs::read(testorig).unwrap();
    let whole = decompress_from_reader_incremental(Cursor::new(&complete)).expect("complete");
    let reference = Decoder::new(&complete)
        .and_then(|d| d.decode_image())
        .unwrap();
    assert!(whole.data == reference.data, "complete stream unchanged");
    for (shape, jpeg) in short_variants(&complete) {
        assert!(
            matches!(
                decompress_from_reader_incremental(Cursor::new(&jpeg)),
                Err(JpegError::UnexpectedEof)
            ),
            "{shape}: the incremental reader must refuse a short segment"
        );
    }
}

#[test]
fn complete_streams_decode_unchanged() {
    // The new check must never fire on a whole stream, with or without its
    // EOI: every entry point decodes it without a warning, as djpeg does.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let testorig: PathBuf = require_c_testimage!("testorig.jpg");
    let source: PathBuf = require_c_testimage!("testorig.ppm");
    let monkey: PathBuf = require_c_testimage!("monkey12.jpg");
    let mut fixtures: Vec<(&str, Vec<u8>)> = vec![
        ("4:2:0", std::fs::read(testorig).unwrap()),
        (
            "grayscale",
            cjpeg_encode(&cjpeg, &source, &["-grayscale"], "p4_254_c_gray"),
        ),
        (
            "restart per row",
            cjpeg_encode(&cjpeg, &source, &["-restart", "1"], "p4_254_c_rst"),
        ),
        ("12-bit", std::fs::read(monkey).unwrap()),
    ];
    fixtures.extend(lossless_fixtures(&djpeg, &cjpeg));
    for (label, complete) in &fixtures {
        let mut without_eoi: Vec<u8> = complete.to_vec();
        without_eoi.truncate(complete.len() - 2);
        for (shape, jpeg, clean) in [
            ("complete", complete.to_vec(), true),
            ("no EOI", without_eoi, false),
        ] {
            let case: String = format!("{label} {shape}");
            let reference: Vec<u16> = djpeg_samples(&djpeg, &jpeg, clean, &label_of(&case));
            if label.contains("lossless") {
                let image = precision::decompress_lossless_arbitrary(&jpeg)
                    .unwrap_or_else(|e| panic!("{case}: {e}"));
                assert!(image.data == reference, "{case}: pixels");
                let (_, warnings) = precision::decompress_lossless_arbitrary_lenient(&jpeg)
                    .unwrap_or_else(|e| panic!("{case}: lenient: {e}"));
                assert!(warnings.is_empty(), "{case}: {warnings:?}");
            }
            if *label == "12-bit" {
                let image =
                    precision::decompress_12bit(&jpeg).unwrap_or_else(|e| panic!("{case}: {e}"));
                let samples: Vec<u16> = image.data.iter().map(|&s| s as u16).collect();
                assert!(samples == reference, "{case}: pixels");
                raw_data_12::decompress_raw_12(&jpeg)
                    .unwrap_or_else(|e| panic!("{case}: raw: {e}"));
            } else if !label.starts_with("16-bit") {
                let image = Decoder::new(&jpeg)
                    .and_then(|d| d.decode_image())
                    .unwrap_or_else(|e| panic!("{case}: {e}"));
                assert!(image.warnings.is_empty(), "{case}: {:?}", image.warnings);
                assert!(widen(&image.data) == reference, "{case}: pixels");
                if !label.contains("lossless") {
                    read_coefficients(&jpeg)
                        .unwrap_or_else(|e| panic!("{case}: coefficients: {e}"));
                }
            }
        }
    }
}
