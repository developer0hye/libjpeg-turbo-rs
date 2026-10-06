//! Every configured `DecodeLimits` field refuses, with small fixtures, on
//! every decode path that takes a budget — 8-bit baseline, progressive,
//! 12-bit and 16-bit (#635 Milestone A, P4-199 #620).
//!
//! `tests/decode_limits.rs` and `tests/memory_limits.rs` cover the 8-bit
//! `Decoder` (and the 65535x65535 header bomb); `tests/tj3_decomp_parameters.rs`
//! covers the `TjHandle` parameters at 12 and 16 bits. What neither had is the
//! full matrix at the boundary: each limit one below what the frame needs must
//! refuse with the documented `what`, and exactly at it must decode — so a
//! check that fires too early, too late, or on the wrong quantity fails here.
//! docs/STABILITY.md "What the memory budget covers" is the prose these numbers pin.
//!
//! Fixtures are embedded so the `wasm32-wasip1` leg runs this file too.

use libjpeg_turbo_rs::precision::{decompress_12bit_with_limits, decompress_16bit_with_limits};
use libjpeg_turbo_rs::{DecodeLimits, Decoder, JpegError};

const BASELINE_GRAY: &[u8] = include_bytes!("fixtures/gray_8x8.jpg");
const PROGRESSIVE: &[u8] = include_bytes!("fixtures/photo_64x64_420_prog.jpg");
const LOSSY12: &[u8] = include_bytes!("fixtures/real_world/libjpeg_testorig12_227x149_12bit.jpg");
const LOSSLESS16: &[u8] = include_bytes!("inputs/api_sequence_lossless16_gray_8x8.jpg");
/// 12-bit, three single-component scans (stock `cjpeg -scans`); see
/// `tests/inputs/README.md`.
const NONINTERLEAVED12: &[u8] = include_bytes!("inputs/p4199_noninterleaved12_16x16_444.jpg");

/// One decode path under a caller's limits.
#[derive(Clone, Copy, Debug)]
enum Path {
    /// `Decoder::new_with_limits` + `decode_image`.
    Decoder,
    /// `precision::decompress_12bit_with_limits`.
    Precision12,
    /// `precision::decompress_16bit_with_limits`.
    Precision16,
}

fn decode(path: Path, jpeg: &[u8], limits: DecodeLimits) -> Result<(), JpegError> {
    match path {
        Path::Decoder => Decoder::new_with_limits(jpeg, limits)?
            .decode_image()
            .map(|_| ()),
        Path::Precision12 => decompress_12bit_with_limits(jpeg, &limits).map(|_| ()),
        Path::Precision16 => decompress_16bit_with_limits(jpeg, &limits).map(|_| ()),
    }
}

/// `limits` one below the frame's need refuses with exactly `(what, actual,
/// limit)`; `at` — the same field set to the need — decodes.
fn assert_boundary(
    label: &str,
    path: Path,
    jpeg: &[u8],
    below: DecodeLimits,
    at: DecodeLimits,
    expected: (&str, u64, u64),
) {
    match decode(path, jpeg, below) {
        Err(JpegError::LimitExceeded {
            what,
            actual,
            limit,
        }) => assert_eq!(
            (what, actual, limit),
            expected,
            "{label} via {path:?}: wrong refusal"
        ),
        other => panic!("{label} via {path:?}: expected LimitExceeded {expected:?}, got {other:?}"),
    }
    decode(path, jpeg, at)
        .unwrap_or_else(|e| panic!("{label} via {path:?}: at the limit it must decode: {e}"));
}

/// The dimension and pixel caps, against the SOF, on every path.
#[test]
fn dimension_and_pixel_caps_refuse_one_below_the_frame_on_every_path() {
    let cases: [(&str, Path, &[u8], u64, u64); 5] = [
        ("8-bit baseline", Path::Decoder, BASELINE_GRAY, 8, 8),
        ("progressive", Path::Decoder, PROGRESSIVE, 64, 64),
        ("12-bit through Decoder", Path::Decoder, LOSSY12, 227, 149),
        ("12-bit", Path::Precision12, LOSSY12, 227, 149),
        ("16-bit lossless", Path::Precision16, LOSSLESS16, 8, 8),
    ];
    let defaults: DecodeLimits = DecodeLimits::default();
    for (label, path, jpeg, width, height) in cases {
        assert_boundary(
            label,
            path,
            jpeg,
            DecodeLimits {
                max_width: width as usize - 1,
                ..defaults
            },
            DecodeLimits {
                max_width: width as usize,
                ..defaults
            },
            ("image width", width, width - 1),
        );
        assert_boundary(
            label,
            path,
            jpeg,
            DecodeLimits {
                max_height: height as usize - 1,
                ..defaults
            },
            DecodeLimits {
                max_height: height as usize,
                ..defaults
            },
            ("image height", height, height - 1),
        );
        let pixels: u64 = width * height;
        assert_boundary(
            label,
            path,
            jpeg,
            DecodeLimits {
                max_pixels: pixels - 1,
                ..defaults
            },
            DecodeLimits {
                max_pixels: pixels,
                ..defaults
            },
            ("total pixels", pixels, pixels - 1),
        );
    }
}

/// `max_memory` against each path's own estimate, at the byte. The numbers are
/// the formulas docs/STABILITY.md "What the memory budget covers" states, worked for each fixture.
#[test]
fn the_memory_budget_refuses_one_byte_below_each_paths_estimate() {
    let cases: [(&str, Path, &[u8], u64); 5] = [
        // 8x8 gray to gray: 64 px x (1 output byte + 1 plane byte).
        ("8-bit baseline", Path::Decoder, BASELINE_GRAY, 128),
        // 64x64 4:2:0 to RGB: 4096 px x (3 output + 3 plane bytes), plus the
        // progressive coefficients: 4096 x 2 x 3 + 4096 x 3 / 64.
        ("progressive", Path::Decoder, PROGRESSIVE, 24_576 + 24_768),
        // 227x149 to RGB through Decoder: the 8-bit estimate alone,
        // 33,823 px x (3 + 3). The 12-bit staging is not counted (P4-224).
        ("12-bit through Decoder", Path::Decoder, LOSSY12, 202_938),
        // 4:2:0 12-bit, two bytes a sample: component planes padded to
        // 15 x 10 iMCUs (240x160 + 2 x 120x80 = 57,600 samples), then
        // 3 x 33,823 upsampled and 3 x 33,823 interleaved.
        (
            "12-bit",
            Path::Precision12,
            LOSSY12,
            (57_600 + 2 * 101_469) * 2,
        ),
        // 8x8 single-component 16-bit: the output alone, 64 x 2 bytes.
        ("16-bit lossless", Path::Precision16, LOSSLESS16, 128),
    ];
    for (label, path, jpeg, estimate) in cases {
        assert_boundary(
            label,
            path,
            jpeg,
            DecodeLimits {
                max_memory: Some(estimate - 1),
                ..DecodeLimits::default()
            },
            DecodeLimits {
                max_memory: Some(estimate),
                ..DecodeLimits::default()
            },
            ("estimated decode memory", estimate, estimate - 1),
        );
    }
}

/// `max_scans` wherever a stream's header walk counts more than one scan: a
/// progressive frame, and a multi-scan 12-bit frame on both 12-bit routes. An
/// interleaved sequential stream ends the walk at its only SOS, so there is no
/// boundary to test on one.
#[test]
fn the_scan_cap_refuses_one_below_the_streams_scan_count() {
    // Every SOS marker the header walk will find; entropy data cannot hold
    // `FF DA`, since a coded `FF` is always stuffed with `00`.
    let progressive_scans: u64 = PROGRESSIVE
        .windows(2)
        .filter(|pair| *pair == [0xFF, 0xDA])
        .count() as u64;
    assert!(progressive_scans > 1, "{progressive_scans} scans");
    let cases: [(&str, Path, &[u8], u64); 3] = [
        ("progressive", Path::Decoder, PROGRESSIVE, progressive_scans),
        (
            "multi-scan 12-bit through Decoder",
            Path::Decoder,
            NONINTERLEAVED12,
            3,
        ),
        ("multi-scan 12-bit", Path::Precision12, NONINTERLEAVED12, 3),
    ];
    for (label, path, jpeg, scans) in cases {
        // Below the count the header walk itself refuses, before any decode.
        match decode(
            path,
            jpeg,
            DecodeLimits {
                max_scans: scans as usize - 1,
                ..DecodeLimits::default()
            },
        ) {
            Err(JpegError::LimitExceeded { actual, limit, .. }) => {
                assert_eq!((actual, limit), (scans, scans - 1), "{label} via {path:?}")
            }
            other => panic!("{label} via {path:?}: expected a scan refusal, got {other:?}"),
        }
        // At the count nothing refuses for a limit. (The multi-scan 12-bit
        // stream decodes to wrong pixels at present — P4-223 — which is a
        // different defect from the one pinned here.)
        let outcome = decode(
            path,
            jpeg,
            DecodeLimits {
                max_scans: scans as usize,
                ..DecodeLimits::default()
            },
        );
        assert!(
            !matches!(outcome, Err(JpegError::LimitExceeded { .. })),
            "{label} via {path:?}: at the scan count it must not refuse for a limit: {outcome:?}"
        );
    }
}
