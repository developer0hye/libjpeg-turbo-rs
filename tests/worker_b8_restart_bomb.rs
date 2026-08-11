//! Worker-B8 restart-interval bomb: 4096x4096 image with restart_interval=1
//! forces the decoder to process one RST marker per MCU — the maximum RST
//! parsing pressure achievable within the JPEG spec.
//!
//! For RGB 4:2:0 subsampling the luma MCU grid is (4096/16) x (4096/16) =
//! 256 x 256 = 65 536 iMCUs. With RI=1 that's 65 536 RST markers in a single
//! decode path — ~1 MiB of marker bytes plus per-RST state resets.
//!
//! # What we assert
//!
//! **The cost of the restart markers, not the cost of the decode.** This file
//! used to assert an absolute wall-clock bound (`< 100 ms`, ~4x a measured
//! 21-24 ms). That flaked under `cargo test`'s parallel threads — and its
//! failure message blamed "RST parsing regression", sending the reader after a
//! bug that had not happened. `CLAUDE.md` already forbids parallel
//! benchmarking for exactly this reason; a clock assertion inside a test the
//! default suite runs in parallel is the same mistake (P4-147, #523).
//!
//! What the test is really guarding is a *complexity* claim: RI=1 restart
//! parsing must not be quadratic in the MCU count. That is measurable without
//! an absolute clock — decode the same image **twice**, once with RI=1 and
//! once with no restart markers at all, and compare. The ratio is what carries
//! the claim:
//!
//! - a machine under load slows both halves together, so the ratio holds;
//! - an `O(MCUs * RST)` scan multiplies only the RI=1 half, by orders of
//!   magnitude — 65 536 restarts against 65 536 iMCUs.
//!
//! Measured on darwin arm64 release over ten rounds of min-of-three: min
//! 0.988, median 0.997, max 1.007. The bound is **1.5** — the measured worst
//! case plus ~50%. See `BOMB_RST_OVERHEAD_RATIO` for why it is not looser.
//!
//! Peak RSS delta bound catches per-RST allocation regressions.
//!
//! # Notes on test runtime
//!
//! This fixture is large — the 4096x4096 source RGB buffer is 48 MiB and the
//! resulting JPEG is multi-MiB. Two fixtures are now built (RI=1 and RI=0) and
//! each is decoded three times, taking the **minimum**: under contention the
//! minimum is the least-noisy statistic, since load can only ever add time.

#[path = "worker_b8_measure.rs"]
mod measure;

#[path = "helpers/mod.rs"]
mod helpers;

use measure::{measure, rss_supported};

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{Encoder, PixelFormat, Subsampling};

const BOMB_SIDE: usize = 4096;

/// How much slower the RI=1 decode may be than the identical image with no
/// restart markers.
///
/// Measured on darwin arm64 release, ten rounds of min-of-three:
/// **min 0.988, median 0.997, max 1.007**. Restart parsing costs nothing
/// measurable today, and taking the minimum of each variant makes the ratio
/// tight enough to bound closely.
///
/// 1.5 is the measured worst case plus ~50%. An earlier draft used 10x on
/// flake-avoidance grounds; that was a guess, not a measurement, and it would
/// have accepted a 5x restart-handling regression — worse than the absolute
/// bound it replaced. `CLAUDE.md` requires a tolerance to be measured reality
/// plus a small margin.
const BOMB_RST_OVERHEAD_RATIO: f64 = 1.5;
/// Decodes per variant; the minimum of each is compared. Three is enough for
/// the minimum to be stable and cheap enough to keep the suite fast.
const BOMB_DECODE_RUNS: usize = 3;
/// Peak RSS delta bound. The 4096^2 RGB output alone is 48 MiB; decode
/// working set is a few coefficient buffers + 1-component planes so we
/// expect ~100 MiB peak delta. 400 MiB catches per-RST leak regressions.
const BOMB_PEAK_RSS_DELTA_LIMIT: u64 = 400 * 1024 * 1024;

// -----------------------------------------------------------------------------
// Fixture builder
// -----------------------------------------------------------------------------

/// Build a 4096x4096 RGB JPEG with restart_interval = 1. The pixel pattern is
/// a cheap deterministic gradient so encoding is fast and reproducible.
fn build_restart_bomb_jpeg() -> Vec<u8> {
    build_bomb_jpeg(true)
}

/// The same 4096x4096 image with or without restart markers. The pixels are
/// identical, so the two decodes differ in exactly one thing: whether the
/// entropy stream carries 65 536 RST markers.
fn build_bomb_jpeg(with_restarts: bool) -> Vec<u8> {
    let n: usize = BOMB_SIDE * BOMB_SIDE * 3;
    let mut pixels: Vec<u8> = Vec::with_capacity(n);
    // Cheap gradient — avoids calling rand / alloc-heavy constructors.
    for y in 0..BOMB_SIDE {
        let yrow: u8 = (y & 0xFF) as u8;
        for x in 0..BOMB_SIDE {
            let xcol: u8 = (x & 0xFF) as u8;
            pixels.push(xcol);
            pixels.push(yrow);
            pixels.push(xcol ^ yrow);
        }
    }
    assert_eq!(pixels.len(), n);

    let encoder = Encoder::new(&pixels, BOMB_SIDE, BOMB_SIDE, PixelFormat::Rgb)
        .quality(50)
        .subsampling(Subsampling::S420);
    let encoder = if with_restarts {
        encoder.restart_blocks(1)
    } else {
        encoder
    };
    encoder
        .encode()
        .unwrap_or_else(|e| panic!("restart bomb JPEG encoding must succeed: {}", e))
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[test]
fn restart_bomb_fixture_has_ri_1_and_decodes() {
    let jpeg: Vec<u8> = build_restart_bomb_jpeg();

    // Guard against fake-pass: inspect the JPEG for a DRI marker with value 1.
    let dri_ri: u16 = find_dri_value(&jpeg).expect("bomb must contain a DRI marker");
    assert_eq!(
        dri_ri, 1,
        "bomb must carry restart_interval=1 (got {}), else the RST pressure \
         assertion is meaningless",
        dri_ri
    );

    // Decoder::new only parses headers; cheap sanity check.
    let _header = Decoder::new(&jpeg).unwrap();
}

/// Decode `jpeg` with stock `djpeg` and require byte-identical output.
///
/// Skips only when the C tool is absent — a developer machine without
/// libjpeg-turbo installed — never on a Rust-side failure.
fn assert_matches_djpeg(label: &str, jpeg: &[u8], ours: &libjpeg_turbo_rs::Image) {
    let Some(djpeg) = helpers::djpeg_path() else {
        eprintln!("SKIP: djpeg not found; {label} fixture not cross-validated against C");
        return;
    };
    let dir: std::path::PathBuf =
        std::env::temp_dir().join(format!("p4147_{}_{}", std::process::id(), label));
    std::fs::create_dir_all(&dir).expect("temp dir");
    let src: std::path::PathBuf = dir.join("in.jpg");
    std::fs::write(&src, jpeg).expect("write fixture");

    let out = std::process::Command::new(&djpeg)
        .arg("-ppm")
        .arg(&src)
        .output()
        .unwrap_or_else(|e| panic!("running djpeg: {e}"));
    assert!(
        out.status.success(),
        "{label}: djpeg rejected the fixture: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Skip the PPM header: "P6\n<w> <h>\n255\n".
    let body: &[u8] = ppm_body(&out.stdout);
    assert_eq!(
        body.len(),
        ours.data.len(),
        "{label}: djpeg produced {} bytes, we produced {}",
        body.len(),
        ours.data.len()
    );
    assert!(
        body == ours.data.as_slice(),
        "{label}: decode differs from djpeg — the fixture or the restart \
         handling disagrees with C, which a Rust-only comparison cannot see"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Strip a binary PPM's three header lines.
fn ppm_body(ppm: &[u8]) -> &[u8] {
    let mut seen: usize = 0;
    for (i, b) in ppm.iter().enumerate() {
        if *b == b'\n' {
            seen += 1;
            if seen == 3 {
                return &ppm[i + 1..];
            }
        }
    }
    panic!("malformed PPM from djpeg");
}

/// Count `FF D0`..`FF D7` restart markers in the entropy stream.
///
/// Byte-level and deliberately naive: a `FF` in entropy-coded data is always
/// stuffed as `FF 00`, so an `FF Dn` sequence is a real marker.
fn count_rst_markers(data: &[u8]) -> usize {
    data.windows(2)
        .filter(|w| w[0] == 0xFF && (0xD0..=0xD7).contains(&w[1]))
        .count()
}

/// Decode once for content comparison, outside any timing.
fn decode_for_content(jpeg: &[u8]) -> libjpeg_turbo_rs::Image {
    let mut decoder: Decoder = Decoder::new(jpeg).unwrap();
    decoder.set_max_pixels(BOMB_SIDE * BOMB_SIDE);
    decoder.set_max_memory(512 * 1024 * 1024);
    decoder
        .decode_image()
        .unwrap_or_else(|e| panic!("content decode failed: {e}"))
}

fn decode_once(label: &str, jpeg: &[u8]) -> measure::Measurement {
    let (image, m) = measure(label, || {
        let mut decoder: Decoder = Decoder::new(jpeg).unwrap();
        decoder.set_max_pixels(BOMB_SIDE * BOMB_SIDE);
        decoder.set_max_memory(512 * 1024 * 1024);
        decoder
            .decode_image()
            .unwrap_or_else(|e| panic!("{label} decode failed: {e}"))
    });
    assert_eq!(image.width, BOMB_SIDE);
    assert_eq!(image.height, BOMB_SIDE);
    m
}

/// Decode both variants `BOMB_DECODE_RUNS` times, **alternating** between
/// them, returning the minimum of each plus the first RI=1 measurement.
///
/// Alternating matters. Running every RI=1 decode and then every control
/// decode leaves the two groups exposed to different moments: a load spike
/// during the first group inflates the numerator alone, and the ratio then
/// reports a regression that is really a scheduling artefact. Interleaving
/// puts the two halves in the same conditions round by round.
///
/// The minimum is then the right statistic: contention can only ever *add*
/// time, so the fastest run of each is the closest available estimate of the
/// work required.
fn interleaved_decodes(
    bomb: &[u8],
    plain: &[u8],
) -> (
    std::time::Duration,
    std::time::Duration,
    measure::Measurement,
) {
    let mut best_bomb: Option<std::time::Duration> = None;
    let mut best_plain: Option<std::time::Duration> = None;
    let mut first_bomb: Option<measure::Measurement> = None;

    for _ in 0..BOMB_DECODE_RUNS {
        let b: measure::Measurement = decode_once("restart_bomb_decode", bomb);
        let p: measure::Measurement = decode_once("restart_bomb_decode_no_rst", plain);
        best_bomb =
            Some(best_bomb.map_or(b.wall_clock, |x: std::time::Duration| x.min(b.wall_clock)));
        best_plain =
            Some(best_plain.map_or(p.wall_clock, |x: std::time::Duration| x.min(p.wall_clock)));
        if first_bomb.is_none() {
            first_bomb = Some(b);
        }
    }
    (
        best_bomb.expect("at least one run"),
        best_plain.expect("at least one run"),
        first_bomb.expect("at least one run"),
    )
}

/// Ignored by default **on purpose**: this is the one assertion here that
/// reads a clock, and `cargo test` runs its binaries and threads in parallel.
/// P4-147's criterion is that no wall-clock comparison remains in the default
/// suite; CI runs it in a named step with `--test-threads=1`, which is the
/// "serial-only harness" the item's own options list.
#[test]
#[ignore = "timing ratio — runs serially in CI's `--test-threads=1` step (P4-147)"]
fn restart_markers_do_not_multiply_decode_cost() {
    let bomb: Vec<u8> = build_bomb_jpeg(true);
    let plain: Vec<u8> = build_bomb_jpeg(false);

    // Guard the control, before timing anything. If the encoder emitted
    // restart markers by default, `plain` would be a second bomb, the ratio
    // would sit near 1 for the wrong reason, and this test would pass while
    // measuring nothing.
    assert_eq!(
        find_dri_value(&bomb),
        Some(1),
        "the bomb must carry restart_interval=1"
    );
    assert_eq!(
        find_dri_value(&plain),
        None,
        "the control must carry no DRI marker at all — otherwise it is a \
         second bomb and the ratio compares like with like for the wrong reason"
    );
    assert_eq!(
        count_rst_markers(&plain),
        0,
        "the control's entropy stream must contain no RST markers"
    );
    let bomb_rsts: usize = count_rst_markers(&bomb);
    assert!(
        bomb_rsts > 60_000,
        "the bomb must actually carry one RST per iMCU (~65 536); found \
         {bomb_rsts}, so the RST pressure this measures is not there"
    );

    // The measured decodes run *first*. `measure()` reports a delta against
    // the process high-water mark, so any decode performed before it — the
    // content comparison below, say — would fold its transient peak into that
    // mark and leave the RSS assertion measuring only what the later run
    // *adds*. A per-RST allocation regression would then sail through the
    // guard advertised to catch it.
    let (with_rst, without_rst, m) = interleaved_decodes(&bomb, &plain);

    // Only now: the two must decode to the *same image*. Restart markers are a
    // framing device, not a content change, and if that were not true the
    // ratio above would be comparing two different workloads.
    let from_bomb = decode_for_content(&bomb);
    let from_plain = decode_for_content(&plain);
    assert_eq!(
        from_bomb.data, from_plain.data,
        "RI=1 and RI=0 encodes of the same pixels must decode identically"
    );

    // And both fixtures are checked against C. Comparing two Rust-encoded
    // streams with the Rust decoder cannot see a shared restart-marker bug:
    // marker counts, content equality and the ratio would all agree while C
    // rejected the stream or decoded it differently.
    assert_matches_djpeg("bomb", &bomb, &from_bomb);
    assert_matches_djpeg("control", &plain, &from_plain);

    // The claim: 65 536 restart markers are a linear cost, not a quadratic
    // one. Comparing the two decodes rather than the clock is what makes this
    // survive a loaded machine — load slows both, and the ratio holds.
    let ratio: f64 = with_rst.as_secs_f64() / without_rst.as_secs_f64().max(f64::EPSILON);
    assert!(
        ratio < BOMB_RST_OVERHEAD_RATIO,
        "RI=1 decode is {ratio:.2}x the cost of the same image without restart \
         markers ({with_rst:?} vs {without_rst:?}), over the {BOMB_RST_OVERHEAD_RATIO}x \
         bound — an O(MCUs * RST) scan is the regression this guards. A loaded \
         machine slows both halves, so this ratio is not a timing flake."
    );

    if rss_supported() {
        assert!(
            m.peak_rss_delta_bytes < BOMB_PEAK_RSS_DELTA_LIMIT,
            "4096x4096 RI=1 decode peak_rss_delta={:.2}MiB exceeds {:.2}MiB",
            m.peak_rss_delta_mib(),
            BOMB_PEAK_RSS_DELTA_LIMIT as f64 / (1024.0 * 1024.0),
        );
    }
}

// -----------------------------------------------------------------------------
// DRI inspection helper
// -----------------------------------------------------------------------------

/// Find the first 0xFF 0xDD (DRI) marker and return its 16-bit restart value.
fn find_dri_value(data: &[u8]) -> Option<u16> {
    let mut i: usize = 0;
    while i + 5 < data.len() {
        if data[i] == 0xFF && data[i + 1] == 0xDD {
            // DRI length is always 4, payload is the 16-bit interval.
            let ri: u16 = ((data[i + 4] as u16) << 8) | data[i + 5] as u16;
            return Some(ri);
        }
        i += 1;
    }
    None
}
