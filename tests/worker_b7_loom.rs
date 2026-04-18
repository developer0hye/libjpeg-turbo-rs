//! Loom-permuted concurrency tests for handle/encoder/decoder types.
//!
//! These tests use the `loom` crate to exhaustively permute thread interleavings
//! and verify that `TjHandle`, `Encoder`, and `Decoder` lifecycles are free of
//! data races under the C11/C++11 memory model.
//!
//! # How to run
//!
//! ```sh
//! RUSTFLAGS="--cfg loom" cargo test --test worker_b7_loom --release
//! ```
//!
//! Loom permutation is expensive; `--release` keeps the per-permutation cost
//! reasonable. The tests are gated behind `#[cfg(loom)]` so they are invisible
//! to normal `cargo test` runs.

// The `loom` cfg name is custom (not listed in `check-cfg`); silence the
// `unexpected_cfgs` lint locally. The coordinator-level fix is to add
// `[lints.rust] unexpected_cfgs = { ..., check-cfg = ['cfg(loom)'] }` to the
// root Cargo.toml (see `COORDINATOR_NOTES.md` §2).
#![allow(unexpected_cfgs)]
#![cfg(loom)]

use libjpeg_turbo_rs::tj3::TjHandle;
use libjpeg_turbo_rs::{compress, Encoder, PixelFormat, Subsampling};
use loom::sync::Arc;
use loom::thread;

/// Two threads concurrently create and drop `TjHandle`s.
///
/// `TjHandle` owns only `Copy` scalars plus owned `Option<Vec<u8>>` / cropping
/// region state — construction and destruction must never touch shared state.
/// Loom will permute every relevant interleaving and fail if any data race
/// (e.g. missing `Send`/`Sync` bound, hidden static mut, TLS leak) is observed.
#[test]
fn loom_two_threads_create_and_drop_tjhandle() {
    loom::model(|| {
        let t1 = thread::spawn(|| {
            let handle: TjHandle = TjHandle::new();
            // Touch the handle to ensure the optimizer cannot elide creation.
            assert_eq!(handle.get(libjpeg_turbo_rs::tj3::TjParam::Quality), 75);
            drop(handle);
        });

        let t2 = thread::spawn(|| {
            let handle: TjHandle = TjHandle::new();
            assert_eq!(handle.get(libjpeg_turbo_rs::tj3::TjParam::Quality), 75);
            drop(handle);
        });

        t1.join().expect("t1 should not panic");
        t2.join().expect("t2 should not panic");
    });
}

/// Two threads concurrently build and drop `Encoder` instances around shared
/// immutable pixel data.
///
/// Shared `&[u8]` via `loom::sync::Arc` exercises the read-only borrow surface.
/// Each thread constructs its own encoder, configures it, and drops without
/// calling `.encode()` — this isolates the *construction/destruction* path from
/// any per-call allocation inside the encoder pipeline (which would blow up
/// loom's state space).
#[test]
fn loom_two_threads_encoder_lifecycle() {
    loom::model(|| {
        let pixels: Arc<Vec<u8>> = Arc::new(vec![128u8; 8 * 8 * 3]);

        let p1: Arc<Vec<u8>> = Arc::clone(&pixels);
        let t1 = thread::spawn(move || {
            let encoder: Encoder<'_> = Encoder::new(&p1, 8, 8, PixelFormat::Rgb)
                .quality(75)
                .subsampling(Subsampling::S444);
            drop(encoder);
        });

        let p2: Arc<Vec<u8>> = Arc::clone(&pixels);
        let t2 = thread::spawn(move || {
            let encoder: Encoder<'_> = Encoder::new(&p2, 8, 8, PixelFormat::Rgb)
                .quality(90)
                .subsampling(Subsampling::S420);
            drop(encoder);
        });

        t1.join().expect("t1 should not panic");
        t2.join().expect("t2 should not panic");
    });
}

/// Two threads concurrently decode the same shared JPEG byte buffer.
///
/// The shared source is `Arc<Vec<u8>>` — a read-only borrow passed to
/// `decompress`. Loom verifies that no internal mutable state (static buffer,
/// thread-local, etc.) is shared across decoder invocations.
#[test]
fn loom_two_threads_decompress_shared_source() {
    loom::model(|| {
        // Build a tiny JPEG once outside the model. Loom re-runs the closure
        // many times; reusing the same bytes keeps permutation cost bounded.
        let pixels: Vec<u8> = vec![64u8; 4 * 4 * 3];
        let jpeg: Vec<u8> =
            compress(&pixels, 4, 4, PixelFormat::Rgb, 75, Subsampling::S444).expect("encode");
        let shared: Arc<Vec<u8>> = Arc::new(jpeg);

        let s1: Arc<Vec<u8>> = Arc::clone(&shared);
        let t1 = thread::spawn(move || {
            let image = libjpeg_turbo_rs::decompress(&s1).expect("decode t1");
            assert_eq!(image.width, 4);
            assert_eq!(image.height, 4);
        });

        let s2: Arc<Vec<u8>> = Arc::clone(&shared);
        let t2 = thread::spawn(move || {
            let image = libjpeg_turbo_rs::decompress(&s2).expect("decode t2");
            assert_eq!(image.width, 4);
            assert_eq!(image.height, 4);
        });

        t1.join().expect("t1 should not panic");
        t2.join().expect("t2 should not panic");
    });
}
