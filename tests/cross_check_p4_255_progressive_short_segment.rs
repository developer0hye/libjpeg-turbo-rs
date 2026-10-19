//! P4-255: a progressive scan whose entropy-coded segment ends early.
//!
//! P4-254 covered the sequential decoders. C's progressive decoder raises the
//! same `JWRN_HIT_MARKER` once a read needs more bits than the segment holds,
//! then leaves every later block of the segment as it was
//! (`decode_mcu_DC_first`, `decode_mcu_AC_first`, `decode_mcu_AC_refine`:
//! jdphuff.c:312, 387, 526; DC refinement reads on, but zero bits change
//! nothing, jdphuff.c:467-469) until a restart marker starts a new segment.
//! Block smoothing then treats the iMCU rows past the last one decoded with
//! data as missing the final scan, using the coefficient bits from before it
//! (`last_good_iMCU_row`, jdcoefct.c:257-258, 556-563).
//!
//! Strict decoding refuses every such stream with `UnexpectedEof`, at every
//! entry point. Lenient decoding with block smoothing, as djpeg decodes,
//! returns djpeg's pixels exactly, with one `TruncatedData` warning. djpeg
//! exits with `EXIT_WARNING` (2) on these inputs, so this file runs it
//! directly rather than through the helpers that require a zero status.
//!
//! Each stream is cut inside one scan's entropy-coded data, so the cut never
//! falls in a later scan's header. The tests also pin P4-259: the scan-by-scan
//! `ProgressiveDecoder` walked non-interleaved scans over the MCU-padded
//! block grid.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{read_coefficients, DecodeWarning, Decoder, JpegError, ProgressiveDecoder};

/// One scan's entropy-coded data, as `start..end` offsets in the stream.
struct Scan {
    start: usize,
    end: usize,
}

/// Every scan of `jpeg`, found by walking its marker segments. A scan's data
/// runs to the next marker that is not a stuffed byte, a fill byte or RSTn.
fn scans(jpeg: &[u8]) -> Vec<Scan> {
    let mut found: Vec<Scan> = Vec::new();
    let mut i: usize = 2;
    while i + 4 <= jpeg.len() && jpeg[i] == 0xFF && jpeg[i + 1] != 0xD9 {
        let length: usize = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if jpeg[i + 1] != 0xDA {
            i += 2 + length;
            continue;
        }
        let start: usize = i + 2 + length;
        let mut end: usize = start;
        while end + 1 < jpeg.len() {
            match (jpeg[end], jpeg[end + 1]) {
                (0xFF, 0x00) | (0xFF, 0xD0..=0xD7) => end += 2,
                (0xFF, 0xFF) => end += 1,
                (0xFF, _) => break,
                _ => end += 1,
            }
        }
        found.push(Scan { start, end });
        i = end;
    }
    assert!(found.len() > 1, "a multi-scan stream");
    found
}

/// Scan `k` cut at the middle of its data, with an EOI after the cut: the
/// truncated stream, which ends in that scan.
fn cut_in_scan(complete: &[u8], scan: &Scan) -> Vec<u8> {
    let mut jpeg: Vec<u8> = complete[..scan.start + (scan.end - scan.start) / 2].to_vec();
    jpeg.extend_from_slice(&[0xFF, 0xD9]);
    jpeg
}

/// Scan `k` with the second half of its data dropped and every later scan
/// kept: a marker comes early, in the middle of the stream.
fn tail_dropped(complete: &[u8], scan: &Scan) -> Vec<u8> {
    let mut jpeg: Vec<u8> = complete[..scan.start + (scan.end - scan.start) / 2].to_vec();
    jpeg.extend_from_slice(&complete[scan.end..]);
    jpeg
}

/// `scan` with the second half of its middle restart interval dropped, every
/// marker kept.
fn short_interval(complete: &[u8], scan: &Scan) -> Vec<u8> {
    let rst: Vec<usize> = (scan.start..scan.end - 1)
        .filter(|&i| complete[i] == 0xFF && (0xD0..=0xD7).contains(&complete[i + 1]))
        .collect();
    assert!(rst.len() >= 4, "expected several restart intervals");
    let k: usize = rst.len() / 2;
    let interval_start: usize = rst[k - 1] + 2;
    let interval_end: usize = rst[k];
    let mut jpeg: Vec<u8> =
        complete[..interval_start + (interval_end - interval_start) / 2].to_vec();
    jpeg.extend_from_slice(&complete[interval_end..]);
    jpeg
}

/// Run djpeg, which must exit with `status`, and return its 8-bit samples.
fn djpeg_samples(djpeg: &Path, jpeg: &[u8], status: i32, label: &str) -> Vec<u8> {
    let input: helpers::TempFile = helpers::TempFile::new(&format!("p4_255_{label}.jpg"));
    input.write_bytes(jpeg);
    let output: std::process::Output = Command::new(djpeg)
        .arg("-pnm")
        .arg(input.path())
        .output()
        .unwrap_or_else(|e| panic!("failed to run djpeg: {e:?}"));
    assert_eq!(
        output.status.code(),
        Some(status),
        "{label}: djpeg status (stderr: {})",
        String::from_utf8_lossy(&output.stderr)
    );
    let (_, _, _, _, samples) =
        helpers::parse_pnm_i16(&output.stdout).unwrap_or_else(|| panic!("{label}: djpeg PNM"));
    samples.iter().map(|&s| s as u8).collect()
}

/// jpegtran's exit status on `jpeg`: 2 when it reports the stream short.
fn jpegtran_status(jpegtran: &Path, jpeg: &[u8]) -> Option<i32> {
    let input: helpers::TempFile = helpers::TempFile::new("p4_255_coefficients.jpg");
    input.write_bytes(jpeg);
    Command::new(jpegtran)
        .arg(input.path())
        .output()
        .expect("run jpegtran")
        .status
        .code()
}

fn truncated_count(warnings: &[DecodeWarning]) -> usize {
    warnings
        .iter()
        .filter(|w| matches!(w, DecodeWarning::TruncatedData { .. }))
        .count()
}

fn label_of(case: &str) -> String {
    case.replace([' ', ',', ':', '+', '/'], "_")
}

/// `cjpeg -progressive` fixtures from `testorig.ppm` (227×149, so partial
/// MCUs on both axes): the default ten-scan colour script with DC scans
/// interleaved, a greyscale one with non-interleaved DC scans, 4:4:4, and a
/// restart marker per MCU row.
fn fixtures(cjpeg: &Path) -> Vec<(&'static str, Vec<u8>)> {
    let source: PathBuf = helpers::require_c_testimage("testorig.ppm")
        .unwrap_or_else(|e| panic!("testorig.ppm: {e:?}"));
    let ppm: Vec<u8> = std::fs::read(&source).unwrap();
    let encode =
        |args: &[&str], label: &str| helpers::encode_with_c_cjpeg(cjpeg, &ppm, args, label);
    vec![
        ("4:2:0", encode(&["-progressive"], "p4_255_420")),
        (
            "grayscale",
            encode(&["-progressive", "-grayscale"], "p4_255_gray"),
        ),
        (
            "4:4:4",
            encode(&["-progressive", "-sample", "1x1"], "p4_255_444"),
        ),
        (
            "restart per row",
            encode(&["-progressive", "-restart", "1"], "p4_255_rst"),
        ),
    ]
}

/// Strict decoding refuses `jpeg` at the `Decoder` (both `decode_image` and
/// `decode_raw`), the scan-by-scan `ProgressiveDecoder` and
/// `read_coefficients`, which jpegtran reports as a warning.
fn assert_every_strict_entry_point_refuses(jpegtran: &Path, jpeg: &[u8], case: &str) {
    let image = Decoder::new(jpeg).and_then(|d| d.decode_image());
    assert!(
        matches!(image, Err(JpegError::UnexpectedEof)),
        "{case}: strict decode_image, got {:?}",
        image.as_ref().map(|i| &i.warnings)
    );
    let raw = Decoder::new(jpeg).and_then(|d| d.decode_raw());
    assert!(
        matches!(raw, Err(JpegError::UnexpectedEof)),
        "{case}: strict decode_raw, got {:?}",
        raw.as_ref().map(|r| r.width)
    );
    let scan_by_scan = ProgressiveDecoder::new(jpeg).and_then(|d| d.finish());
    assert!(
        matches!(scan_by_scan, Err(JpegError::UnexpectedEof)),
        "{case}: ProgressiveDecoder, got {:?}",
        scan_by_scan.as_ref().map(|i| i.width)
    );
    assert!(
        matches!(read_coefficients(jpeg), Err(JpegError::UnexpectedEof)),
        "{case}: read_coefficients"
    );
    assert_eq!(
        jpegtran_status(jpegtran, jpeg),
        Some(2),
        "{case}: jpegtran warns"
    );
}

/// Lenient decoding with block smoothing returns djpeg's pixels exactly, with
/// one `TruncatedData`; without smoothing it still warns once.
fn assert_lenient_matches_djpeg(djpeg: &Path, jpeg: &[u8], case: &str) {
    let smoothed = Decoder::new(jpeg)
        .and_then(|d| {
            d.with_lenient(true)
                .with_block_smoothing(true)
                .decode_image()
        })
        .unwrap_or_else(|e| panic!("{case}: lenient decode failed: {e}"));
    assert_eq!(
        truncated_count(&smoothed.warnings),
        1,
        "{case}: {:?}",
        smoothed.warnings
    );
    let reference: Vec<u8> = djpeg_samples(djpeg, jpeg, 2, &label_of(case));
    assert!(
        smoothed.data == reference,
        "{case}: lenient pixels must equal djpeg's ({} of {} differ)",
        smoothed
            .data
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a != b)
            .count(),
        reference.len()
    );
    let unsmoothed = Decoder::new(jpeg)
        .and_then(|d| d.with_lenient(true).decode_image())
        .unwrap_or_else(|e| panic!("{case}: lenient decode without smoothing failed: {e}"));
    assert_eq!(truncated_count(&unsmoothed.warnings), 1, "{case}");
}

#[test]
fn a_scan_cut_short_is_refused_strictly_and_matches_djpeg_leniently() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    // Every scan of every fixture: DC first and refinement (interleaved and
    // not), AC first and AC refinement, with and without restart markers.
    for (label, complete) in fixtures(&cjpeg) {
        for (k, scan) in scans(&complete).iter().enumerate() {
            let case: String = format!("{label}, scan {k} cut + EOI");
            let jpeg: Vec<u8> = cut_in_scan(&complete, scan);
            assert_every_strict_entry_point_refuses(&jpegtran, &jpeg, &case);
            assert_lenient_matches_djpeg(&djpeg, &jpeg, &case);
        }
    }
}

#[test]
fn a_scan_cut_short_mid_stream_is_refused_strictly() {
    // The scan runs short because the next scan's markers come early. In
    // lenient mode the later refinement scans can then desync on blocks the
    // short scan left unset, and lenient progressive decoding has no Huffman
    // error recovery (P4-260), so only the strict refusal is pinned here.
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    for (label, complete) in fixtures(&cjpeg) {
        let all: Vec<Scan> = scans(&complete);
        for (k, scan) in all.iter().enumerate().take(all.len() - 1) {
            let case: String = format!("{label}, scan {k} tail dropped");
            assert_every_strict_entry_point_refuses(
                &jpegtran,
                &tail_dropped(&complete, scan),
                &case,
            );
        }
    }
}

#[test]
fn a_short_restart_interval_resumes_at_the_next_marker_like_djpeg() {
    // A short interval in the first scan (interleaved DC, which no later scan
    // depends on to stay in sync) and in the last: the rest of the interval
    // is left as it was, decoding resumes at the next RST, and the smoothing
    // rows follow from where the last scan last had data.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let jpegtran: PathBuf = require_c_tool!("jpegtran");
    let (_, complete) = fixtures(&cjpeg)
        .into_iter()
        .find(|(label, _)| *label == "restart per row")
        .unwrap();
    let all: Vec<Scan> = scans(&complete);
    for k in [0, all.len() - 1] {
        let case: String = format!("restart per row, scan {k} short interval");
        let jpeg: Vec<u8> = short_interval(&complete, &all[k]);
        assert_every_strict_entry_point_refuses(&jpegtran, &jpeg, &case);
        assert_lenient_matches_djpeg(&djpeg, &jpeg, &case);
    }
}

#[test]
fn progressive_decoder_walks_each_components_own_blocks_p4_259() {
    // 227×149 leaves partial MCUs, so a subsampled component's own block grid
    // is narrower than the MCU-padded one. ProgressiveDecoder used to walk
    // the padded grid in non-interleaved scans, misplacing blocks; 89% of the
    // 4:2:0 samples differed from djpeg's.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let source: PathBuf = require_c_testimage!("testorig.ppm");
    let ppm: Vec<u8> = std::fs::read(&source).unwrap();
    for (label, sample) in [("4:2:0", "2x2"), ("4:2:2", "2x1"), ("4:4:0", "1x2")] {
        let jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
            &cjpeg,
            &ppm,
            &["-progressive", "-sample", sample],
            &format!("p4_259_{}", label_of(label)),
        );
        let reference: Vec<u8> = djpeg_samples(&djpeg, &jpeg, 0, &format!("p4_259_{sample}"));
        let image = ProgressiveDecoder::new(&jpeg)
            .and_then(|d| d.finish())
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert!(
            image.data == reference,
            "{label}: ProgressiveDecoder pixels"
        );
    }
}

#[test]
fn complete_progressive_streams_decode_unchanged() {
    // The new checks must never fire on a whole stream: every entry point
    // decodes it without a warning, as djpeg does.
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    for (label, jpeg) in fixtures(&cjpeg) {
        let reference: Vec<u8> = djpeg_samples(&djpeg, &jpeg, 0, &label_of(label));
        for (mode, lenient, smoothing) in [
            ("strict", false, false),
            ("lenient", true, false),
            ("lenient, smoothed", true, true),
        ] {
            let image = Decoder::new(&jpeg)
                .and_then(|d| {
                    d.with_lenient(lenient)
                        .with_block_smoothing(smoothing)
                        .decode_image()
                })
                .unwrap_or_else(|e| panic!("{label}, {mode}: {e}"));
            assert!(
                image.warnings.is_empty(),
                "{label}, {mode}: {:?}",
                image.warnings
            );
            assert!(image.data == reference, "{label}, {mode}: pixels");
        }
        Decoder::new(&jpeg)
            .and_then(|d| d.decode_raw())
            .unwrap_or_else(|e| panic!("{label}: decode_raw: {e}"));
        let scan_by_scan = ProgressiveDecoder::new(&jpeg)
            .and_then(|d| d.finish())
            .unwrap_or_else(|e| panic!("{label}: ProgressiveDecoder: {e}"));
        assert!(
            scan_by_scan.data == reference,
            "{label}: ProgressiveDecoder pixels"
        );
        read_coefficients(&jpeg).unwrap_or_else(|e| panic!("{label}: coefficients: {e}"));
    }
}
