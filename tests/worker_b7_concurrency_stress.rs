//! Worker B7 — multi-thread stress coverage for `Encoder` / `Decoder`
//! lifecycles and shared-table borrows.
//!
//! The mission brief asked for `rayon::par_iter`; rayon is not a direct
//! dev-dependency of the workspace (see `COORDINATOR_NOTES.md`), so these
//! tests use `std::thread` with an explicit manual work split. The
//! observable coverage — many threads concurrently hitting a shared
//! immutable source / a shared custom quant table — is identical.
//!
//! Covered scenarios:
//!   B7-2: 1000 concurrent decodes of the same JPEG bytes from fixtures,
//!         asserting every thread produces bit-identical pixels.

// WASM has no std::thread support.
#![cfg(not(target_arch = "wasm32"))]

use libjpeg_turbo_rs::{decompress, Image};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

// --- Fixtures ---------------------------------------------------------------

/// Absolute path to `tests/fixtures/<name>`.
fn fixture_path(name: &str) -> PathBuf {
    let mut p: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests");
    p.push("fixtures");
    p.push(name);
    p
}

/// Load a fixture JPEG as bytes. Panics on I/O failure so any issue with the
/// test harness surfaces as a Rust failure rather than a silent skip.
fn load_fixture(name: &str) -> Vec<u8> {
    let path: PathBuf = fixture_path(name);
    std::fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", path.display(), e))
}

/// A small set of real fixtures with different subsamplings / dimensions —
/// exercises multiple decoder code paths in parallel.
fn fixture_set() -> Vec<(&'static str, Vec<u8>)> {
    let names: &[&str] = &[
        "cjpeg_31x33_420.jpg",
        "cjpeg_31x33_422.jpg",
        "cjpeg_31x33_444.jpg",
        "blue_16x16_420.jpg",
        "checker_640x480_420.jpg",
    ];
    names.iter().map(|n| (*n, load_fixture(n))).collect()
}

// --- B7-2: 1000 concurrent decodes of shared &[u8] -------------------------

/// Spawn `TOTAL_TASKS` decode jobs distributed across `worker_threads`, all
/// reading the same `Arc<Vec<u8>>` via `&[u8]`. Every job must produce the
/// same pixel buffer as the serial baseline.
#[test]
fn b7_2_stress_thousand_decodes_shared_source() {
    const TOTAL_TASKS: usize = 1000;
    // Use enough threads to create real contention on the shared source; cap
    // at a reasonable number so CI runners are not oversubscribed.
    let worker_threads: usize = std::cmp::min(
        16,
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );

    let fixtures: Vec<(&'static str, Vec<u8>)> = fixture_set();
    assert!(!fixtures.is_empty(), "fixture set must not be empty");

    // Compute serial reference pixels for each fixture once.
    let references: Vec<Image> = fixtures
        .iter()
        .map(|(name, bytes)| {
            decompress(bytes).unwrap_or_else(|e| panic!("ref decode {name} failed: {e:?}"))
        })
        .collect();

    // Share every source buffer across threads via Arc. The Arc allows
    // moving a borrow into each worker without copying bytes.
    let shared_sources: Arc<Vec<(&'static str, Arc<Vec<u8>>)>> = Arc::new(
        fixtures
            .into_iter()
            .map(|(name, bytes)| (name, Arc::new(bytes)))
            .collect(),
    );
    let shared_refs: Arc<Vec<Image>> = Arc::new(references);

    let mut handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(worker_threads);
    let tasks_per_worker: usize = TOTAL_TASKS.div_ceil(worker_threads);

    for worker_id in 0..worker_threads {
        let sources: Arc<Vec<(&'static str, Arc<Vec<u8>>)>> = Arc::clone(&shared_sources);
        let refs: Arc<Vec<Image>> = Arc::clone(&shared_refs);
        let start: usize = worker_id * tasks_per_worker;
        let end: usize = std::cmp::min(start + tasks_per_worker, TOTAL_TASKS);
        handles.push(thread::spawn(move || {
            for task_idx in start..end {
                let fixture_idx: usize = task_idx % sources.len();
                let (name, bytes) = &sources[fixture_idx];
                let image: Image = decompress(bytes.as_slice()).unwrap_or_else(|e| {
                    panic!("task {task_idx} fixture {name} decode failed: {e:?}")
                });
                let reference: &Image = &refs[fixture_idx];
                assert_eq!(
                    image.width, reference.width,
                    "task {task_idx} fixture {name}: width mismatch"
                );
                assert_eq!(
                    image.height, reference.height,
                    "task {task_idx} fixture {name}: height mismatch"
                );
                assert_eq!(
                    image.pixel_format, reference.pixel_format,
                    "task {task_idx} fixture {name}: pixel format mismatch"
                );
                assert_eq!(
                    image.data, reference.data,
                    "task {task_idx} fixture {name}: decoded pixels diverged from serial reference"
                );
            }
        }));
    }

    for handle in handles {
        handle.join().expect("worker thread panicked");
    }
}
