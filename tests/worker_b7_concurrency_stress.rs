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
//!   B7-3: interleaved Encoder/Decoder lifetimes across threads via channels.
//!   B7-4: custom `[u16; 64]` quant table reused simultaneously by two
//!         encoder threads, asserting byte-equal output vs. serial baseline.

// WASM has no std::thread support.
#![cfg(not(target_arch = "wasm32"))]

use libjpeg_turbo_rs::{decompress, Encoder, Image, PixelFormat, Subsampling};
use std::path::PathBuf;
use std::sync::mpsc;
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

// --- B7-3: interleaved Encoder/Decoder across threads ----------------------

/// Each worker: build an `Encoder` on thread `E`, hand the encoded bytes to a
/// decode on thread `D` via an mpsc channel, then compare decoded pixels
/// against the serial baseline. Exercises cross-thread handoff of the
/// encoded byte buffer.
#[test]
fn b7_3_interleaved_encoder_decoder_across_threads() {
    const NUM_WORKERS: usize = 8;
    // Shared immutable source that every encoder will compress. Shared
    // `&[u8]` exercises the immutable-borrow side of the encoder contract.
    let width: usize = 32;
    let height: usize = 32;
    let source_pixels: Arc<Vec<u8>> = Arc::new(
        (0..width * height * 3)
            .map(|i| ((i as u32 * 7 + 11) % 256) as u8)
            .collect(),
    );

    // Serial baseline: encode + decode once — this is the pixel-equivalence target.
    let baseline_jpeg: Vec<u8> = Encoder::new(&source_pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .expect("baseline encode failed");
    let baseline_image: Image = decompress(&baseline_jpeg).expect("baseline decode failed");

    // Each worker gets its own (encoder-thread, decoder-thread, channel) trio.
    let mut worker_handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(NUM_WORKERS);
    for worker_id in 0..NUM_WORKERS {
        let pixels: Arc<Vec<u8>> = Arc::clone(&source_pixels);
        let expected_data: Vec<u8> = baseline_image.data.clone();
        let expected_w: usize = baseline_image.width;
        let expected_h: usize = baseline_image.height;

        worker_handles.push(thread::spawn(move || {
            let (tx, rx) = mpsc::channel::<Vec<u8>>();

            // Encoder thread — owns the send side.
            let pixels_for_enc: Arc<Vec<u8>> = Arc::clone(&pixels);
            let encoder_thread: thread::JoinHandle<()> = thread::spawn(move || {
                let jpeg: Vec<u8> = Encoder::new(&pixels_for_enc, width, height, PixelFormat::Rgb)
                    .quality(75)
                    .subsampling(Subsampling::S444)
                    .encode()
                    .unwrap_or_else(|e| panic!("worker {worker_id} encode failed: {e:?}"));
                tx.send(jpeg)
                    .expect("worker encoder->decoder channel closed unexpectedly");
            });

            // Decoder thread — owns the receive side.
            let decoder_thread: thread::JoinHandle<Image> = thread::spawn(move || {
                let jpeg: Vec<u8> = rx
                    .recv()
                    .expect("decoder never received bytes from encoder thread");
                decompress(&jpeg)
                    .unwrap_or_else(|e| panic!("worker {worker_id} decode failed: {e:?}"))
            });

            encoder_thread.join().expect("encoder thread panicked");
            let decoded: Image = decoder_thread.join().expect("decoder thread panicked");

            assert_eq!(
                decoded.width, expected_w,
                "worker {worker_id} width mismatch"
            );
            assert_eq!(
                decoded.height, expected_h,
                "worker {worker_id} height mismatch"
            );
            assert_eq!(
                decoded.data, expected_data,
                "worker {worker_id} pixels diverged from serial baseline across cross-thread handoff"
            );
        }));
    }

    for handle in worker_handles {
        handle.join().expect("outer worker thread panicked");
    }
}

// --- B7-4: shared custom quant table across encoder threads ----------------

/// Construct a custom `[u16; 64]` luma quantization table, then encode from
/// two threads concurrently using a shared reference to it. Both outputs
/// must be byte-equal to the serial baseline produced with the same table.
///
/// This proves that custom quant tables can be safely referenced from
/// multiple `Encoder` instances running in parallel, and that the resulting
/// JPEG bytes are deterministic (no encoder-side global state mutation).
#[test]
fn b7_4_shared_quant_table_across_encoders_byte_equal() {
    // A plausible custom quant table — monotonically increasing so the
    // output actually depends on every slot, not just DC. Values clamp to
    // the valid 8-bit JPEG quant range (1..=255) so we don't accidentally
    // exercise 16-bit precision code paths in this test.
    let mut custom: [u16; 64] = [0; 64];
    for (i, slot) in custom.iter_mut().enumerate() {
        *slot = (8 + i as u16).min(200);
    }
    let shared_table: Arc<[u16; 64]> = Arc::new(custom);

    // Simple deterministic 16x16 RGB source — small enough to keep the
    // test fast, large enough to produce multiple DCT blocks per component.
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| ((i as u32 * 13 + 29) % 256) as u8)
        .collect();

    // Serial baseline: encode once with the custom table.
    let baseline: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .subsampling(Subsampling::S444)
        .quant_table(0, *shared_table)
        .encode()
        .expect("serial baseline encode failed");

    // Two encoder threads, each encoding the same source with a borrow of
    // the same shared quant table. `[u16; 64]` is `Copy`, so we copy it
    // out of the `Arc` inside each thread — this is the exact pattern a
    // user with a "template" table shared across encoder workers would use.
    let t1_pixels: Vec<u8> = pixels.clone();
    let t1_table: Arc<[u16; 64]> = Arc::clone(&shared_table);
    let t1 = thread::spawn(move || {
        Encoder::new(&t1_pixels, width, height, PixelFormat::Rgb)
            .subsampling(Subsampling::S444)
            .quant_table(0, *t1_table)
            .encode()
            .expect("t1 encode failed")
    });

    let t2_pixels: Vec<u8> = pixels.clone();
    let t2_table: Arc<[u16; 64]> = Arc::clone(&shared_table);
    let t2 = thread::spawn(move || {
        Encoder::new(&t2_pixels, width, height, PixelFormat::Rgb)
            .subsampling(Subsampling::S444)
            .quant_table(0, *t2_table)
            .encode()
            .expect("t2 encode failed")
    });

    let out1: Vec<u8> = t1.join().expect("t1 panicked");
    let out2: Vec<u8> = t2.join().expect("t2 panicked");

    assert_eq!(
        out1, baseline,
        "t1 output must be byte-equal to serial baseline when using same custom quant table"
    );
    assert_eq!(
        out2, baseline,
        "t2 output must be byte-equal to serial baseline when using same custom quant table"
    );
    assert_eq!(
        out1, out2,
        "both concurrent encoders using the same shared quant table must produce identical bytes"
    );
}
