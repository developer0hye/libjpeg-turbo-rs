//! Regression tests for fuzzer-discovered crashes.
//!
//! Every `crash-*` file under `fuzz/corpus/<target>/` is re-fed into the
//! same API the matching fuzz target exercises. A malformed JPEG must
//! never panic or abort — it must return `Err` (or at minimum not crash
//! the process). libFuzzer runs the same seeds nightly; this harness
//! catches regressions fast between nightly runs.
//!
//! Gated off `wasm` because the corpus is loaded via `std::fs`.

#![cfg(not(target_family = "wasm"))]

use libjpeg_turbo_rs::{decompress, decompress_lenient, read_coefficients};
use std::path::PathBuf;

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

#[test]
fn fuzz_decompress_crashes_are_panic_safe() {
    run("fuzz_decompress", |d| {
        let _ = decompress(d);
    });
}

#[test]
fn fuzz_decompress_lenient_crashes_are_panic_safe() {
    run("fuzz_decompress_lenient", |d| {
        let _ = decompress_lenient(d);
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

#[test]
fn fuzz_transform_crashes_are_panic_safe() {
    // Transform goes through the same entropy-decode entry points as
    // `read_coefficients`; we exercise that path here.
    run("fuzz_transform", |d| {
        let _ = read_coefficients(d);
    });
}
