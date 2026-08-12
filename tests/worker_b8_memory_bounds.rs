//! Worker-B8 pathological-input memory bounds.
//!
//! Replicates coverage from `tests/memory_limits.rs` **and** records peak RSS
//! and wall-clock for every test, asserting against DOCUMENTED measured
//! baselines rather than hand-picked magic numbers.
//!
//! P4-152: peak RSS is the asserted bound. The wall-clock baselines below are
//! still measured and reported, but only *asserted* on platforms that cannot
//! report RSS — a ~1000x margin beside a real memory bound catches nothing and
//! fails under contention.
//!
//! Measured baselines (darwin arm64, debug test build, --test-threads=1):
//!
//! | Test                                    | wall (ms) | RSS delta (MiB) |
//! |-----------------------------------------|-----------|-----------------|
//! | max_mem_default / max_mem_high          | 0.28-0.43 | 0.00-0.27       |
//! | max_mem_low / max_mem_tight / rejects   | 0.09-0.10 | 0.00-0.03       |
//! | max_pixels_* (all)                      | 0.09-0.27 | 0.00-0.02       |
//! | scan_limit_baseline (32x32)             | 0.27      | 0.02            |
//! | scan_limit_generous (16x16 prog)        | 0.13      | 0.05            |
//! | scan_limit_reject (320x240 prog)        | 0.50      | 0.00            |
//! | scan_limit_allow (320x240 prog decode)  | 10.92     | 0.73            |
//! | stop_on_warn_reject (320x240 truncated) | 2.43      | 0.23            |
//! | stop_on_warn_clean                      | 0.26      | 0.00            |
//!
//! Tolerances:
//!   - small-decode wall: 500ms (measured <1 ms; ~500x for CI jitter on
//!     contended runners). Still three orders of magnitude tighter than
//!     "doesn't panic".
//!   - medium-prog wall:  2_000ms (measured ~11ms; same ratio, same rationale).
//!   - peak RSS delta:    32 MiB. Measured <1 MiB; catches order-of-magnitude
//!     regressions (e.g., accidental O(w*h) intermediate allocation) while
//!     absorbing per-process cache warmup drift. RSS is a high-water mark, so
//!     a small image sandwiched between other tests may inherit the prior
//!     peak without this test contributing — 32 MiB accommodates the
//!     alloc_16mib self-test from worker_b8_measure bleeding into this run.

#[path = "worker_b8_measure.rs"]
mod measure;

use std::time::Duration;

use measure::{measure, rss_supported};

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{
    compress, compress_progressive, decompress_lenient, PixelFormat, Subsampling,
};

// --------------------------------------------------------------------------
// Baselines (see table above). Wall bounds intentionally generous to tolerate
// CI jitter; RSS bounds chosen to catch order-of-magnitude regressions only
// (e.g., accidental O(width*height) temp allocations on top of normal decode).
// --------------------------------------------------------------------------

/// Upper bound for peak RSS delta on any of the B8-2 tests. Measured max
/// observed delta is 0.73 MiB (scan_limit_allow 320x240 prog). 32 MiB absorbs
/// process-wide peak drift from sibling tests sharing the binary.
const SMALL_DECODE_PEAK_RSS_DELTA_LIMIT: u64 = 32 * 1024 * 1024;
/// Wall-clock fallback for platforms with no RSS reporting (P4-152). Measured
/// worst case 0.43 ms; 500 ms is ~1000x, which is why it is a fallback and not
/// a bound worth asserting beside a real memory measurement.
const SMALL_DECODE_WALL_CLOCK_MS: u128 = 500;

/// Upper bound for peak RSS delta during a 320x240 progressive decode. Same
/// 32 MiB bound — the 320x240 fixture measured only 0.73 MiB delta.
const MEDIUM_PROG_DECODE_PEAK_RSS_DELTA_LIMIT: u64 = 32 * 1024 * 1024;
/// Wall-clock fallback, as above. Measured worst case 10.92 ms; 2000 ms is
/// ~180x.
const MEDIUM_PROG_DECODE_WALL_CLOCK_MS: u128 = 2_000;

// --------------------------------------------------------------------------
// Fixtures
// --------------------------------------------------------------------------

fn make_32x32_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = (0..32 * 32 * 3)
        .map(|i| ((i * 37 + 13) % 256) as u8)
        .collect();
    compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap()
}

fn make_64x64_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 64 * 64 * 3];
    compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap()
}

/// P4-152: peak RSS where it can be measured, wall clock only where it cannot.
///
/// The bound this file is about is *memory*, and where RSS is available that is
/// both asserted and deterministic. The 500 ms clock that used to run first
/// added nothing there — its own comment called it "~1000x to tolerate
/// contended CI runners", a margin that cannot fire on a regression the RSS
/// bound would miss, only on contention. That is the failure P4-147 (#523)
/// documented: a green-on-rerun failure naming a regression that did not
/// happen.
///
/// It is kept as a *fallback* rather than deleted outright, because
/// `worker_b8_measure` reports no RSS on platforms other than Linux and macOS
/// and its contract says callers "should skip RSS assertions but still run
/// wall-clock bounds so the test remains useful". Dropping it everywhere would
/// have left Windows with no resource bound at all — trading a flaky assertion
/// for no assertion. The same loose margin that makes it useless beside a real
/// memory bound is what makes it tolerable as the only one.
/// The bound check itself, with RSS availability injected.
///
/// Returns `Err(reason)` instead of panicking so the *fallback* path can be
/// exercised on a platform that has RSS. That matters: CI runs these binaries
/// on Ubuntu, where `rss_supported()` is always true, and the Windows job
/// builds them with `--no-run` — so a fallback written as an untestable `if`
/// would never execute anywhere in CI. An inverted comparison or a wrong limit
/// would then ship unnoticed.
fn check_decode_bounds(
    label: &str,
    m: &measure::Measurement,
    rss_available: bool,
    wall_clock_limit_ms: u128,
    rss_limit_bytes: u64,
) -> Result<(), String> {
    if rss_available {
        // Where memory can be measured it is the bound, and the wall clock is
        // deliberately *not* also asserted — that ~1000x margin is the flaky
        // one P4-152 removed.
        if m.peak_rss_delta_bytes >= rss_limit_bytes {
            return Err(format!(
                "{label}: peak_rss_delta={:.2}MiB exceeds bound {:.2}MiB",
                m.peak_rss_delta_mib(),
                rss_limit_bytes as f64 / (1024.0 * 1024.0),
            ));
        }
        return Ok(());
    }
    // No RSS on this platform, so the clock is the only bound left.
    // `worker_b8_measure`'s contract asks callers to keep it for exactly this
    // case; dropping it would leave the test asserting nothing here.
    if m.wall_clock.as_millis() >= wall_clock_limit_ms {
        return Err(format!(
            "{label}: wall_clock={:?} exceeds fallback bound {wall_clock_limit_ms}ms \
             (no RSS on this platform)",
            m.wall_clock,
        ));
    }
    Ok(())
}

fn assert_within_small_decode_bounds(label: &str, m: measure::Measurement) {
    if let Err(reason) = check_decode_bounds(
        label,
        &m,
        rss_supported(),
        SMALL_DECODE_WALL_CLOCK_MS,
        SMALL_DECODE_PEAK_RSS_DELTA_LIMIT,
    ) {
        panic!("{reason}");
    }
}

fn assert_within_medium_prog_bounds(label: &str, m: measure::Measurement) {
    if let Err(reason) = check_decode_bounds(
        label,
        &m,
        rss_supported(),
        MEDIUM_PROG_DECODE_WALL_CLOCK_MS,
        MEDIUM_PROG_DECODE_PEAK_RSS_DELTA_LIMIT,
    ) {
        panic!("{reason}");
    }
}

// --------------------------------------------------------------------------
// max_pixels tests
// --------------------------------------------------------------------------

#[test]
fn max_pixels_rejects_image_exceeding_limit_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (err_msg, m) = measure("max_pixels_reject", || {
        let mut decoder: Decoder = Decoder::new(&jpeg).unwrap();
        decoder.set_max_pixels(100);
        let err = decoder
            .decode_image()
            .expect_err("must reject 1024-pixel image with max_pixels=100");
        format!("{}", err)
    });
    assert!(
        err_msg.contains("exceeds limit"),
        "error should mention exceeds limit, got: {}",
        err_msg
    );
    // Rejection is fail-fast before any allocation of the output buffer.
    assert_within_small_decode_bounds("max_pixels_reject", m);
}

#[test]
fn max_pixels_allows_image_within_limit_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (img, m) = measure("max_pixels_allow", || {
        let mut decoder: Decoder = Decoder::new(&jpeg).unwrap();
        decoder.set_max_pixels(10_000);
        decoder.decode_image().unwrap()
    });
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_within_small_decode_bounds("max_pixels_allow", m);
}

#[test]
fn max_pixels_zero_rejects_any_nonzero_image_bounded() {
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let (err, m) = measure("max_pixels_zero", || {
        let mut decoder: Decoder = Decoder::new(&jpeg).unwrap();
        decoder.set_max_pixels(0);
        decoder.decode_image().expect_err(
            "max_pixels=0 must reject any non-zero image (0 means zero, not unlimited, here)",
        )
    });
    let _ = err;
    assert_within_small_decode_bounds("max_pixels_zero", m);
}

#[test]
fn max_pixels_exact_boundary_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (img, m_ok) = measure("max_pixels_eq", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_max_pixels(1024);
        d.decode_image().unwrap()
    });
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_within_small_decode_bounds("max_pixels_eq", m_ok);

    let (_err, m_err) = measure("max_pixels_lt", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_max_pixels(1023);
        d.decode_image()
            .expect_err("max_pixels=1023 must reject 1024-pixel image")
    });
    assert_within_small_decode_bounds("max_pixels_lt", m_err);
}

// --------------------------------------------------------------------------
// max_memory tests
// --------------------------------------------------------------------------

#[test]
fn max_memory_very_low_rejects_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (err_msg, m) = measure("max_mem_low", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_max_memory(1024);
        format!(
            "{}",
            d.decode_image()
                .expect_err("must reject with max_memory=1024")
        )
    });
    assert!(
        err_msg.contains("memory") || err_msg.contains("exceeds"),
        "error should mention memory limit, got: {}",
        err_msg
    );
    assert_within_small_decode_bounds("max_mem_low", m);
}

#[test]
fn max_memory_high_enough_succeeds_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (img, m) = measure("max_mem_high", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_max_memory(10 * 1024 * 1024);
        d.decode_image().unwrap()
    });
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_within_small_decode_bounds("max_mem_high", m);
}

#[test]
fn max_memory_default_unlimited_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (img, m) = measure("max_mem_default", || {
        let decoder: Decoder = Decoder::new(&jpeg).unwrap();
        decoder.decode_image().unwrap()
    });
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_within_small_decode_bounds("max_mem_default", m);
}

#[test]
fn max_memory_large_image_tight_limit_bounded() {
    let jpeg: Vec<u8> = make_64x64_jpeg();
    let (_err, m) = measure("max_mem_tight", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_max_memory(1_000);
        d.decode_image()
            .expect_err("should reject 64x64 with max_memory=1000")
    });
    assert_within_small_decode_bounds("max_mem_tight", m);
}

// --------------------------------------------------------------------------
// scan_limit tests
// --------------------------------------------------------------------------

#[test]
fn scan_limit_rejects_progressive_with_many_scans_bounded() {
    let jpeg: &[u8] = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let (err_msg, m) = measure("scan_limit_reject", || {
        let mut d: Decoder = Decoder::new(jpeg).unwrap();
        d.set_scan_limit(1);
        format!(
            "{}",
            d.decode_image().expect_err(
                "scan_limit=1 must reject multi-scan progressive JPEG — \
                 this is the DoS mitigation under test",
            )
        )
    });
    // Current implementation emits Unsupported("progressive scan count N exceeds limit of M").
    // Accept either the existing phrasing or a future ScanLimitExceeded-style wording.
    assert!(
        (err_msg.contains("scan") && err_msg.contains("limit"))
            || err_msg.contains("ScanLimitExceeded"),
        "error should mention scan limit, got: {}",
        err_msg
    );
    assert_within_medium_prog_bounds("scan_limit_reject", m);
}

#[test]
fn scan_limit_high_allows_progressive_bounded() {
    let jpeg: &[u8] = include_bytes!("fixtures/photo_320x240_420_prog.jpg");
    let (img, m) = measure("scan_limit_allow", || {
        let mut d: Decoder = Decoder::new(jpeg).unwrap();
        d.set_scan_limit(100);
        d.decode_image().unwrap()
    });
    assert_eq!(img.width, 320);
    assert_eq!(img.height, 240);
    assert_within_medium_prog_bounds("scan_limit_allow", m);
}

#[test]
fn scan_limit_does_not_affect_baseline_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (img, m) = measure("scan_limit_baseline", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_scan_limit(1);
        d.decode_image().unwrap()
    });
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_within_small_decode_bounds("scan_limit_baseline", m);
}

#[test]
fn scan_limit_just_above_scan_count_succeeds_bounded() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let (img, m) = measure("scan_limit_generous", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_scan_limit(100);
        d.decode_image().unwrap()
    });
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_within_small_decode_bounds("scan_limit_generous", m);
}

// --------------------------------------------------------------------------
// stop_on_warning tests
// --------------------------------------------------------------------------

#[test]
fn stop_on_warning_rejects_truncated_jpeg_bounded() {
    let data: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");
    let truncated: &[u8] = &data[..2000.min(data.len())];

    // Baseline: lenient alone succeeds and emits warnings.
    let lenient_img = decompress_lenient(truncated).unwrap();
    assert!(
        !lenient_img.warnings.is_empty(),
        "lenient truncated decode must produce warnings"
    );

    let (err_msg, m) = measure("stop_on_warn_reject", || {
        let mut d: Decoder = Decoder::new(truncated).unwrap();
        d.set_lenient(true);
        d.set_stop_on_warning(true);
        format!(
            "{}",
            d.decode_image()
                .expect_err("stop_on_warning must convert warning to error")
        )
    });
    assert!(
        err_msg.contains("stop_on_warning"),
        "error should mention stop_on_warning, got: {}",
        err_msg
    );
    assert_within_medium_prog_bounds("stop_on_warn_reject", m);
}

#[test]
fn stop_on_warning_allows_clean_jpeg_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();
    let (img, m) = measure("stop_on_warn_clean", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_stop_on_warning(true);
        d.decode_image().unwrap()
    });
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_within_small_decode_bounds("stop_on_warn_clean", m);
}

// --------------------------------------------------------------------------
// Combined limits
// --------------------------------------------------------------------------

#[test]
fn max_pixels_and_max_memory_both_enforced_bounded() {
    let jpeg: Vec<u8> = make_32x32_jpeg();

    let (_e1, m1) = measure("combined_pixels", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_max_pixels(100);
        d.set_max_memory(10 * 1024 * 1024);
        d.decode_image()
            .expect_err("pixel limit must fire before mem limit")
    });
    assert_within_small_decode_bounds("combined_pixels", m1);

    let (_e2, m2) = measure("combined_memory", || {
        let mut d: Decoder = Decoder::new(&jpeg).unwrap();
        d.set_max_pixels(100_000);
        d.set_max_memory(100);
        d.decode_image()
            .expect_err("memory limit must fire independently")
    });
    assert_within_small_decode_bounds("combined_memory", m2);
}

/// P4-152: the no-RSS fallback is exercised here, on every platform.
///
/// Checking only a predicate would not have done it — an inverted comparison, a
/// wrong limit, or a deleted assertion inside the helper would all still pass.
/// So these drive `check_decode_bounds` itself with a synthetic `Measurement`
/// and `rss_available` forced both ways, which is the only way the fallback
/// runs at all in CI: the integration binaries run on Ubuntu, where RSS is
/// always available, and the Windows job builds them with `--no-run`.
#[test]
fn the_wall_clock_fallback_is_asserted_exactly_when_rss_is_unavailable() {
    let over_clock: measure::Measurement = measure::Measurement {
        peak_rss_delta_bytes: 0,
        peak_rss_bytes: 0,
        wall_clock: Duration::from_millis(SMALL_DECODE_WALL_CLOCK_MS as u64 + 1),
        rss_supported: false,
    };
    let over_rss: measure::Measurement = measure::Measurement {
        peak_rss_delta_bytes: SMALL_DECODE_PEAK_RSS_DELTA_LIMIT + 1,
        peak_rss_bytes: 0,
        wall_clock: Duration::from_millis(1),
        rss_supported: true,
    };
    let within: measure::Measurement = measure::Measurement {
        peak_rss_delta_bytes: 0,
        peak_rss_bytes: 0,
        wall_clock: Duration::from_millis(1),
        rss_supported: true,
    };
    let check = |m: &measure::Measurement, rss: bool| {
        check_decode_bounds(
            "unit",
            m,
            rss,
            SMALL_DECODE_WALL_CLOCK_MS,
            SMALL_DECODE_PEAK_RSS_DELTA_LIMIT,
        )
    };

    // Without RSS the clock is enforced — this is the branch CI never reaches
    // on its own, and the one whose deletion this guards against.
    assert!(
        check(&over_clock, false).is_err(),
        "a platform with no RSS must still reject an over-budget wall clock; \
         otherwise this file asserts nothing there"
    );
    assert!(check(&within, false).is_ok(), "a fast decode must pass");

    // With RSS the memory bound is enforced and the clock is deliberately not,
    // which is the flaky assertion P4-152 removed.
    assert!(
        check(&over_rss, true).is_err(),
        "an over-budget peak RSS must be rejected where RSS is measurable"
    );
    assert!(
        check(&over_clock, true).is_ok(),
        "where peak RSS is measurable a slow wall clock must NOT fail the test — \
         re-asserting it is the ~1000x margin that fails under contention"
    );

    // A bound of 0 ms would reject every decode it is meant to permit.
    const {
        assert!(SMALL_DECODE_WALL_CLOCK_MS > 0);
        assert!(MEDIUM_PROG_DECODE_WALL_CLOCK_MS > 0);
    }
}
