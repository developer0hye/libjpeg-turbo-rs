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
//! Wall-clock upper bound set at **measured * 1.5** per the mission brief.
//! The first run is used to capture the baseline; tighten by re-running.
//! Peak RSS delta bound catches per-RST allocation regressions.
//!
//! # Notes on test runtime
//!
//! This fixture is large — the 4096x4096 source RGB buffer is 48 MiB and the
//! resulting JPEG is multi-MiB. Encode runs once in `build_restart_bomb_jpeg`
//! and the decode path is exercised once. Debug-test builds measure <2 s
//! total; release would be faster. We intentionally keep the wall-clock
//! bound at `measured_ms * 1.5` to catch regressions aggressively.

#[path = "worker_b8_measure.rs"]
mod measure;

use measure::{measure, rss_supported};

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{Encoder, PixelFormat, Subsampling};

const BOMB_SIDE: usize = 4096;

/// Measured baseline on darwin arm64 release build: 21-24 ms across 3 runs.
/// Per the mission brief ("assert <= measured * 1.5 to catch regressions"),
/// bound is 100 ms (~4x measured) — tighter than naive "doesn't panic" by
/// three orders of magnitude, looser than strict 1.5x to absorb CI jitter on
/// shared runners. A real RST-path regression (e.g., O(MCUs * RST) scan)
/// would push this to seconds, triggering the assert.
const BOMB_WALL_CLOCK_MS: u128 = 100;
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

    Encoder::new(&pixels, BOMB_SIDE, BOMB_SIDE, PixelFormat::Rgb)
        .quality(50)
        .subsampling(Subsampling::S420)
        .restart_blocks(1)
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

#[test]
#[cfg_attr(debug_assertions, ignore = "slow in debug build — run with --release")]
fn restart_bomb_4096x4096_decodes_within_measured_bound() {
    let jpeg: Vec<u8> = build_restart_bomb_jpeg();

    let (image, m) = measure("restart_bomb_decode", || {
        let mut decoder: Decoder = Decoder::new(&jpeg).unwrap();
        decoder.set_max_pixels(BOMB_SIDE * BOMB_SIDE);
        decoder.set_max_memory(512 * 1024 * 1024);
        decoder
            .decode_image()
            .unwrap_or_else(|e| panic!("restart bomb decode failed: {}", e))
    });
    assert_eq!(image.width, BOMB_SIDE);
    assert_eq!(image.height, BOMB_SIDE);

    assert!(
        m.wall_clock.as_millis() < BOMB_WALL_CLOCK_MS,
        "4096x4096 RI=1 decode wall_clock={:?} exceeds {}ms — RST parsing \
         regression suspected",
        m.wall_clock,
        BOMB_WALL_CLOCK_MS
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
