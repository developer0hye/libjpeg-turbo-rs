//! Worker-B8 progressive-scan bomb: a crafted progressive JPEG that carries
//! thousands of SOS headers so the parser's `scan_limit` mitigation can be
//! exercised end-to-end.
//!
//! # Construction
//!
//! The bomb is built by byte-splicing:
//!
//!   [ header from a real 16x16 progressive JPEG up to the first SOS ]
//!   ++ [ SOS + minimal entropy-data + RST ]  * N
//!   ++ EOI
//!
//! Each replicated SOS advertises the same scan parameters as the first scan
//! of the real fixture, but with a single-byte minimal entropy payload (`0x00`)
//! so the skip_entropy_data() loop advances one byte before finding the next
//! 0xFF marker. The parser counts each SOS into `metadata.scans`, letting us
//! drive the total count to any value without decoding to completion.
//!
//! # What we assert
//!
//! - **With `set_scan_limit(1000)`**: decode errors out, error message mentions
//!   "scan" and "limit" (or the future `ScanLimitExceeded` enum variant).
//! - **Without limit**: the decoder either completes or surfaces an error,
//!   within bounded peak RSS.
//!
//! Neither of those asserts a wall-clock ceiling any more (P4-152). The two
//! timing properties they used to carry — that the scan loop is linear, and
//! that `scan_limit` stops early — are now ratios against a control, at the
//! bottom of this file, `#[ignore]`d out of the parallel run and executed by a
//! serial CI step.

#[path = "worker_b8_measure.rs"]
mod measure;

use measure::{measure, rss_supported};

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{compress_progressive, PixelFormat, Subsampling};

const TARGET_SCAN_COUNT: usize = 5_000;
const SCAN_LIMIT_UNDER_TEST: u32 = 1_000;

/// P4-152: the two absolute wall-clock bounds that used to live here are gone.
/// Each asserted a fixed millisecond ceiling while `cargo test` runs binaries
/// and threads in parallel, so a contended runner failed them with a message
/// naming a regression that did not happen — the cost P4-147 (#523) documented.
/// They are replaced by the two *ratios* below, which compare a workload
/// against a control and so cancel machine speed and load.
///
/// **Scan-loop scaling.** Quadrupling the scan count must roughly quadruple the
/// work, not multiply it by sixteen. Measured min-of-9 over five rounds on
/// darwin arm64 release: 3.87, 3.91, 3.91, 3.91, 3.87 against a linear
/// expectation of 4.0. A quadratic scan loop — the regression the demoted bound
/// claimed to catch — gives ~16. The bound is the measured worst case plus a
/// small margin, per the tolerance rule: 5.0 is 3.91 + ~28 %. It still rejects
/// quadratic threefold, and unlike a bound placed halfway to 16 it also rejects
/// a merely *superlinear* regression that doubles the constant factor.
const SCAN_LOOP_SCALING_RATIO_LIMIT: f64 = 5.0;
/// Scan counts for that ratio, a 4x span. The upper one is near the decoder's
/// own 8192-scan parse limit, which is what caps the achievable signal.
const SCALING_SMALL_SCANS: usize = 1_999;
const SCALING_LARGE_SCANS: usize = 7_999;

/// **Fail-fast.** A `scan_limit` of 1000 against ~5000 scans must stop early
/// rather than walk them all, so the limited decode costs a fraction of the
/// unlimited one. Measured min-of-9 over five rounds: 0.278, 0.269, 0.270,
/// 0.269, 0.269 — close to the 0.2 the scan ratio implies, plus fixed header
/// cost. A mitigation that stopped firing would sit near 1.0, so the bound is
/// set at roughly twice the measurement and well clear of it.
const FAIL_FAST_RATIO_LIMIT: f64 = 0.6;
/// Timed repetitions per workload. Each workload's own minimum is taken across
/// all of them and the ratio is formed from the two minima — see
/// [`best_decode_ms`] for why the *ratio* must not be minimised directly.
const RATIO_SAMPLES: usize = 27;
/// Peak RSS delta bound. Parser stores a ScanInfo per SOS; since issue #351
/// those hold `Arc<HuffmanTable>` handles, so 5000 scans share one copy of
/// each table (8 pointers per scan, not ~4 KB). Scan state + coefficient
/// buffers still dominate. 100 MiB catches order-of-magnitude regressions.
// Measured ~155 MiB peak_rss_delta on aarch64 macOS under `cargo test --tests`
// BEFORE the #351 Arc sharing landed; the bound is now conservative, not
// re-measured. Tighten only against a fresh measurement.
// aggregation (the decoder allocates scan-state before/while the scan_limit
// fires and before max_memory caps hit). Bound set to measured + ~45 MiB
// margin; catches true runaway (GiB-scale) allocation, still well under the
// 256 MiB set_max_memory used in the unlimited test.
const BOMB_PEAK_RSS_DELTA_LIMIT: u64 = 200 * 1024 * 1024;

// -----------------------------------------------------------------------------
// Fixture builder
// -----------------------------------------------------------------------------

/// Build a seed progressive JPEG from a 16x16 uniform image, then clone its
/// first SOS+entropy region `extra_scan_count` extra times. The replicated
/// scans are valid DC refinement scans (Ss=0, Se=0) with a minimal
/// byte-stuffed entropy payload so the parser skips them and advances.
fn build_progressive_scan_bomb(extra_scan_count: usize) -> Vec<u8> {
    // Step 1: real progressive seed so the decoder's header parsing succeeds
    // (SOI, DQT, SOF2, DHT, SOS). compress_progressive emits a standard
    // simple_progression() scan script (10 scans for grayscale, more for RGB).
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let seed: Vec<u8> =
        compress_progressive(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444)
            .expect("seed progressive JPEG must encode");

    // Step 2: locate the first SOS marker and the first post-SOS marker.
    let first_sos: usize = find_marker(&seed, 0, 0xDA).expect("seed has SOS");
    let first_post_sos_marker: usize =
        find_post_sos_marker(&seed, first_sos).expect("seed has a marker after its first SOS");

    // Step 3: take a snapshot of the first full scan segment (SOS header +
    // entropy bytes up to next marker). We replay this byte-for-byte so each
    // clone is already byte-stuffed correctly.
    let first_scan_segment: &[u8] = &seed[first_sos..first_post_sos_marker];

    // Step 4: assemble bomb = seed_prefix + first_scan_segment (count = N+1)
    //                        + original_tail_from_first_post_sos_marker.
    let mut bomb: Vec<u8> =
        Vec::with_capacity(seed.len() + extra_scan_count * first_scan_segment.len());
    bomb.extend_from_slice(&seed[..first_sos]);
    for _ in 0..(extra_scan_count + 1) {
        bomb.extend_from_slice(first_scan_segment);
    }
    // Append the remainder of the original JPEG (subsequent scans + EOI).
    bomb.extend_from_slice(&seed[first_post_sos_marker..]);

    bomb
}

fn find_marker(data: &[u8], start: usize, marker_code: u8) -> Option<usize> {
    let mut i: usize = start;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == marker_code {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// Given the position of a 0xFF 0xDA (SOS), scan forward past the SOS header
/// and entropy data until the next real marker (non-stuffing, non-RST) and
/// return its offset.
fn find_post_sos_marker(data: &[u8], sos_pos: usize) -> Option<usize> {
    // Skip SOS header: 2 (marker) + 2 (length) + Ls-2 bytes.
    let len_pos: usize = sos_pos + 2;
    if len_pos + 2 > data.len() {
        return None;
    }
    let sos_len: usize = ((data[len_pos] as usize) << 8) | data[len_pos + 1] as usize;
    let mut i: usize = sos_pos + 2 + sos_len;
    while i + 1 < data.len() {
        if data[i] == 0xFF {
            let next: u8 = data[i + 1];
            if next == 0x00 || (0xD0..=0xD7).contains(&next) {
                // stuffed or restart — keep walking
                i += 2;
                continue;
            }
            return Some(i);
        }
        i += 1;
    }
    None
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn progressive_bomb_fixture_is_well_formed() {
    // Guard against silent fake-pass: confirm headers still parse.
    let bomb: Vec<u8> = build_progressive_scan_bomb(TARGET_SCAN_COUNT - 1);
    assert_eq!(&bomb[..2], &[0xFF, 0xD8], "missing SOI");
    assert_eq!(&bomb[bomb.len() - 2..], &[0xFF, 0xD9], "missing EOI");
    // Decoder::new parses headers only — it must succeed to prove the bomb is
    // syntactically valid before bound testing.
    let _ = Decoder::new(&bomb).unwrap_or_else(|e| {
        panic!(
            "progressive bomb fixture failed header parse — builder bug \
             (NOT a decoder bug), scans={}: {}",
            TARGET_SCAN_COUNT, e
        )
    });
}

#[test]
fn progressive_bomb_with_scan_limit_errors_out_bounded() {
    let bomb: Vec<u8> = build_progressive_scan_bomb(TARGET_SCAN_COUNT - 1);

    let (err_msg, m) = measure("prog_bomb_limited", || {
        let mut decoder: Decoder = Decoder::new(&bomb).unwrap();
        decoder.set_scan_limit(SCAN_LIMIT_UNDER_TEST);
        decoder.set_max_memory(128 * 1024 * 1024);
        format!(
            "{}",
            decoder.decode_image().err().unwrap_or_else(|| panic!(
                "scan_limit={} must reject a ~{}-scan progressive JPEG — \
                 DoS mitigation did not fire",
                SCAN_LIMIT_UNDER_TEST, TARGET_SCAN_COUNT
            ))
        )
    });

    // Match current implementation wording OR a future ScanLimitExceeded enum
    // variant (see mission brief). Either proves the mitigation fired.
    assert!(
        (err_msg.contains("scan") && err_msg.contains("limit"))
            || err_msg.contains("ScanLimitExceeded"),
        "error should mention scan limit, got: {}",
        err_msg
    );

    // P4-152: the wall-clock bound that used to sit here is now
    // `scan_limit_stops_early_rather_than_walking_every_scan`, which measures
    // the same property — fail-fast, not walking all scans — against a control
    // instead of a fixed ceiling.
    if rss_supported() {
        assert!(
            m.peak_rss_delta_bytes < BOMB_PEAK_RSS_DELTA_LIMIT,
            "scan_limit rejection peak_rss_delta={:.2}MiB exceeds {:.2}MiB",
            m.peak_rss_delta_mib(),
            BOMB_PEAK_RSS_DELTA_LIMIT as f64 / (1024.0 * 1024.0),
        );
    }
}

#[test]
fn progressive_bomb_without_limit_is_bounded() {
    // Without scan_limit, the decoder must still complete or surface a
    // structured error within the loose wall-clock bound. This catches
    // pathological O(N^2) regressions in the scan-iteration loop.
    let bomb: Vec<u8> = build_progressive_scan_bomb(TARGET_SCAN_COUNT - 1);

    let (result, m) = measure("prog_bomb_unlimited", || {
        let mut decoder: Decoder = Decoder::new(&bomb).unwrap();
        decoder.set_max_memory(256 * 1024 * 1024);
        decoder.decode_image()
    });

    // Either outcome is acceptable as long as we terminate in bounded time:
    // - Ok(img): decoder successfully walked all scans
    // - Err(_):  decoder surfaced a structured error (e.g., DC refinement on
    //            an already-refined coefficient)
    match &result {
        Ok(img) => {
            eprintln!(
                "prog_bomb_unlimited decoded Ok {}x{}",
                img.width, img.height
            );
        }
        Err(e) => {
            eprintln!("prog_bomb_unlimited returned error (accepted): {}", e);
        }
    }

    // P4-152: the O(N^2) guard is now
    // `scan_loop_cost_scales_linearly_with_scan_count`, which compares two scan
    // counts rather than asserting a millisecond ceiling.
    if rss_supported() {
        assert!(
            m.peak_rss_delta_bytes < BOMB_PEAK_RSS_DELTA_LIMIT,
            "unlimited progressive bomb peak_rss_delta={:.2}MiB exceeds {:.2}MiB",
            m.peak_rss_delta_mib(),
            BOMB_PEAK_RSS_DELTA_LIMIT as f64 / (1024.0 * 1024.0),
        );
    }
}

// ---------------------------------------------------------------------------
// P4-152: ratios against a control, in place of absolute wall-clock bounds.
//
// Both are `#[ignore]`d out of the default parallel run and executed by the
// serial CI step P4-147 added. A ratio cancels machine speed, but not
// *contention*: two workloads timed while other test binaries compete for the
// same cores are not comparable to each other either. Running them serially is
// what makes the comparison mean anything.
// ---------------------------------------------------------------------------

/// Best-of-`RATIO_SAMPLES` wall clock for one decode configuration, in
/// milliseconds.
///
/// Minimum rather than mean: scheduler noise only ever *adds* time to a
/// measurement, so the fastest observation of a workload is the one closest to
/// the work it actually did.
///
/// That reasoning applies to each workload on its own, and **not** to the ratio
/// between two of them — a distinction the first version of this file got
/// wrong. Noise landing on the denominator inflates it and therefore *deflates*
/// the ratio, so taking the minimum across several (numerator, denominator)
/// pairs selects for the most-deflated one and can hide the very regression the
/// test exists to catch. Each workload is minimised here; the ratio is formed
/// once, from two numbers that are each already the least-noisy estimate
/// available.
fn best_decode_ms(scans: usize, scan_limit: Option<u32>) -> f64 {
    let bomb: Vec<u8> = build_progressive_scan_bomb(scans);
    let mut best: f64 = f64::MAX;
    for _ in 0..RATIO_SAMPLES {
        let (_result, m) = measure("ratio", || {
            let mut decoder: Decoder = Decoder::new(&bomb).unwrap();
            decoder.set_max_memory(256 * 1024 * 1024);
            if let Some(limit) = scan_limit {
                decoder.set_scan_limit(limit);
            }
            decoder.decode_image()
        });
        best = best.min(m.wall_clock_ms());
    }
    best
}

/// Quadrupling the scan count must roughly quadruple the work.
///
/// This is what the deleted `UNLIMITED_PARSE_WALL_CLOCK_MS` bound was *for* —
/// its comment named an O(N^2) scan loop as the regression — but a fixed
/// ceiling could only ever catch a regression large enough to cross it from a
/// measurement 1000x below, while failing on a loaded runner for no reason.
/// A ratio between two sizes of the same fixture measures the scaling directly.
///
/// The span is capped by the decoder's own 8192-scan parse limit, so 2000 to
/// 8000 is the widest 4x window available.
#[ignore = "timing ratio — runs serially in CI's --test-threads=1 step (P4-152)"]
#[test]
fn scan_loop_cost_scales_linearly_with_scan_count() {
    // Warm the allocator and instruction caches so the first size measured is
    // not charged for them.
    let _ = best_decode_ms(SCALING_LARGE_SCANS, None);

    let small: f64 = best_decode_ms(SCALING_SMALL_SCANS, None);
    let large: f64 = best_decode_ms(SCALING_LARGE_SCANS, None);
    assert!(
        small > 0.0,
        "the small decode measured 0 ms, so the ratio would be meaningless"
    );
    let ratio: f64 = large / small;

    assert!(
        ratio < SCAN_LOOP_SCALING_RATIO_LIMIT,
        "scan-loop cost scaled {ratio:.2}x for a 4x scan count ({} -> {}), \
         over the {SCAN_LOOP_SCALING_RATIO_LIMIT} bound. Linear is ~4.0 and \
         measured 3.87-3.91; quadratic is ~16. \
         small={small:.3}ms large={large:.3}ms, each the best of {RATIO_SAMPLES}.",
        SCALING_SMALL_SCANS,
        SCALING_LARGE_SCANS,
    );
}

/// A `scan_limit` must stop the decode early, not walk every scan and then
/// report.
///
/// The control is the same bomb decoded with no limit, which is what makes
/// "early" measurable: the mitigation is supposed to do a fraction of that
/// work. The deleted `LIMITED_DECODE_WALL_CLOCK_MS` asserted a millisecond
/// ceiling instead, which a mitigation that had stopped firing entirely would
/// still have satisfied — the unlimited decode of this fixture also finishes in
/// about a millisecond.
#[ignore = "timing ratio — runs serially in CI's --test-threads=1 step (P4-152)"]
#[test]
fn scan_limit_stops_early_rather_than_walking_every_scan() {
    let _ = best_decode_ms(TARGET_SCAN_COUNT - 1, None);

    let unlimited: f64 = best_decode_ms(TARGET_SCAN_COUNT - 1, None);
    let limited: f64 = best_decode_ms(TARGET_SCAN_COUNT - 1, Some(SCAN_LIMIT_UNDER_TEST));
    assert!(
        unlimited > 0.0,
        "the unlimited decode measured 0 ms, so the ratio would be meaningless"
    );
    let ratio: f64 = limited / unlimited;

    assert!(
        ratio < FAIL_FAST_RATIO_LIMIT,
        "a scan_limit of {SCAN_LIMIT_UNDER_TEST} against ~{TARGET_SCAN_COUNT} scans cost \
         {ratio:.3} of an unlimited decode, over the {FAIL_FAST_RATIO_LIMIT} bound. \
         Measured 0.269-0.278; a mitigation that no longer fires early sits near 1.0. \
         limited={limited:.3}ms unlimited={unlimited:.3}ms, each the best of {RATIO_SAMPLES}.",
    );
}
