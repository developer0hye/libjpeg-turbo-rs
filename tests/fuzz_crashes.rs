//! Regression tests for fuzzer-discovered crashes.
//!
//! Every `crash-*` file under `fuzz/corpus/<target>/` is re-fed into the
//! same API the matching fuzz target exercises. A malformed JPEG must
//! never panic or abort — it must return `Err` (or at minimum not crash
//! the process). libFuzzer runs the same seeds nightly; this harness
//! catches regressions fast between nightly runs.
//!
//! **The replay must drive the same decoder options the fuzz target
//! does.** `fuzz_decompress` keys an option matrix (scale 1/2, RGB565,
//! crop, grayscale, …) off `data.len() % 7`; this harness used to call
//! bare `decompress()`, so every seed whose crash needed one of those
//! options replayed green here while still failing in CI.
//! `crash-8d3c593a…` sat committed and "passing" in exactly that state
//! until the 2026-07-29 Fuzz Smoke runs — see `drive_decompress` below,
//! which is kept in lock-step with `fuzz/fuzz_targets/fuzz_decompress.rs`.
//!
//! Gated off `wasm` because the corpus is loaded via `std::fs`.

#![cfg(not(target_family = "wasm"))]

use libjpeg_turbo_rs::precision::{
    decompress_12bit, decompress_16bit, decompress_lossless_arbitrary,
};
use libjpeg_turbo_rs::{
    decompress, decompress_lenient, read_coefficients, transform_jpeg_with_options, Decoder,
    MarkerCopyMode, PixelFormat, TransformOp, TransformOptions,
};
use std::path::PathBuf;

/// Mirrors `MAX_FUZZ_PIXELS` in the fuzz targets.
const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fn crash_seeds(target: &str) -> Vec<PathBuf> {
    let dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "fuzz", "corpus", target]
        .iter()
        .collect();
    let read_dir = match std::fs::read_dir(&dir) {
        Ok(rd) => rd,
        Err(_) => return Vec::new(),
    };
    let mut seeds: Vec<PathBuf> = read_dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("crash-"))
        })
        .collect();
    seeds.sort();
    seeds
}

fn run<F: Fn(&[u8])>(target: &str, call: F) {
    let seeds = crash_seeds(target);
    assert!(
        !seeds.is_empty(),
        "no crash-* seeds under fuzz/corpus/{}",
        target
    );
    for seed in seeds {
        let data = std::fs::read(&seed).unwrap_or_else(|e| panic!("read {:?}: {}", seed, e));
        call(&data);
    }
}

/// Replay of `fuzz/fuzz_targets/fuzz_decompress.rs`. Keep the option
/// matrix and its `data.len() % 7` keying identical to that target — a
/// seed only reproduces under the arm the fuzzer picked for it.
fn drive_decompress(data: &[u8]) {
    let Ok(mut decoder) = Decoder::new(data) else {
        return;
    };
    let (frame_w, frame_h) = {
        let header = decoder.header();
        (header.width, header.height)
    };
    let pixels = (frame_w as u64).saturating_mul(frame_h as u64);
    if frame_w == 0 || frame_h == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    decoder.set_scan_limit(100);
    match data.len() % 7 {
        1 => decoder.set_output_format(PixelFormat::Rgba),
        2 => {
            decoder.set_output_format(PixelFormat::Rgb565);
            decoder.set_dither_565(true);
        }
        // The reduced-size IDCT lives behind this arm only; five of the
        // 2026-07-29 crash seeds panic on i32 overflow without it.
        3 => decoder.set_scale(libjpeg_turbo_rs::ScalingFactor::new(1, 2)),
        4 => decoder.set_crop_region(3, 0, 40, frame_h as usize),
        5 => decoder.set_output_format(PixelFormat::Xrgb),
        6 => decoder.set_output_format(PixelFormat::Grayscale),
        _ => {}
    }
    let _ = decoder.decode_image();
}

#[test]
fn fuzz_decompress_crashes_are_panic_safe() {
    run("fuzz_decompress", drive_decompress);
}

#[test]
fn fuzz_decompress_crashes_are_panic_safe_under_every_option_arm() {
    // The `% 7` keying pins each seed to one arm, so a seed minimized
    // under "scale 1/2" never exercises Rgb565 and vice versa. Sweep all
    // arms over all seeds by padding the input length.
    //
    // Padding is an arm-selection lever, not a pixel-preserving one: for a
    // seed that ends at EOI the appended bytes are ignored, but many
    // crash seeds are truncated mid-entropy, where the extra zero bytes
    // do feed the bit reader. That is fine — the contract asserted here
    // is only "no panic, for any input", which every variant must meet.
    let seeds = crash_seeds("fuzz_decompress");
    assert!(!seeds.is_empty(), "no crash-* seeds under fuzz_decompress");
    for seed in seeds {
        let data = std::fs::read(&seed).unwrap_or_else(|e| panic!("read {:?}: {}", seed, e));
        for pad in 0..7 {
            let mut padded = data.clone();
            padded.extend(std::iter::repeat_n(0u8, pad));
            drive_decompress(&padded);
        }
    }
}

#[test]
fn fuzz_decompress_lenient_crashes_are_panic_safe() {
    run("fuzz_decompress_lenient", |d| {
        let _ = decompress_lenient(d);
    });
}

#[test]
fn fuzz_decompress_precision_crashes_are_panic_safe() {
    run("fuzz_decompress_precision", |d| {
        let _ = decompress_12bit(d);
        let _ = decompress_16bit(d);
        let _ = decompress_lossless_arbitrary(d);
    });
}

#[test]
fn fuzz_progressive_decoder_crashes_are_panic_safe() {
    run("fuzz_progressive_decoder", |d| {
        let _ = decompress(d);
    });
}

#[test]
fn fuzz_read_coefficients_crashes_are_panic_safe() {
    run("fuzz_read_coefficients", |d| {
        let _ = read_coefficients(d);
    });
}

/// Replay of the Rust half of `fuzz/fuzz_targets/fuzz_transform_diff_c.rs`:
/// byte 0 selects HFlip / VFlip / Rot180 (`op_for`), the rest is the JPEG,
/// and markers are copied like `jpegtran -copy all`. The C-oracle half
/// (byte-exactness with jpegtran, clean djpeg decode) is pinned per seed in
/// `tests/regression_transform_fuzz_progressive.rs`; this replay only
/// guarantees the transform itself stays panic-free.
#[test]
fn fuzz_transform_diff_c_crashes_are_panic_safe() {
    run("fuzz_transform_diff_c", |d| {
        const HEADER_LEN: usize = 1;
        if d.len() < HEADER_LEN + 32 {
            return;
        }
        let op: TransformOp = match d[0] % 3 {
            0 => TransformOp::HFlip,
            1 => TransformOp::VFlip,
            _ => TransformOp::Rot180,
        };
        let _ = transform_jpeg_with_options(
            &d[HEADER_LEN..],
            &TransformOptions {
                op,
                copy_markers: MarkerCopyMode::All,
                ..Default::default()
            },
        );
    });
}

#[test]
fn fuzz_transform_crashes_are_panic_safe() {
    // Transform goes through the same entropy-decode entry points as
    // `read_coefficients`; we exercise that path here.
    run("fuzz_transform", |d| {
        let _ = read_coefficients(d);
    });
}
