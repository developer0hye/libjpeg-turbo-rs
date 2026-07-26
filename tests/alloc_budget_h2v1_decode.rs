//! Allocation-budget regression for the 4:2:2 (H2V1) decode path
//! (issue #350): H2V1 used to fall through to the generic upsample path,
//! which materialises two full-resolution chroma planes (`cb_full` /
//! `cr_full`, ~4.2 MB extra for 1080p) before colour conversion, while
//! H2V2 (4:2:0) streams rows and never allocates them.
//!
//! Budget model for 1920×1080 4:2:2 with the row-streaming path
//! (4:2:2 iMCUs are 16×8, so plane heights align to 8 — 1080 rows, not
//! 4:2:0's 1088):
//!   output RGB   1920×1080×3      ≈ 6.22 MB
//!   Y plane      1920×1080        ≈ 2.07 MB
//!   Cb+Cr planes 2 × 960×1080     ≈ 2.07 MB   (decoded, half-res — intrinsic)
//!   row scratch  a few × full_width
//!   ------------------------------------------------
//!   total        ≈ 10.4 MB
//!
//! The pre-fix path allocated ≈ 14.5 MB (the two full-res chroma planes
//! on top). The assertion pins the post-fix ceiling with margin so the
//! full-plane fallback cannot silently return for standard 4:2:2.
//!
//! Note: 4:2:2 cannot reach 4:2:0's total (≈ 9.4 MB) exactly — its
//! decoded chroma planes are half-resolution rather than
//! quarter-resolution, an intrinsic +1.03 MB at 1080p. Removing the
//! whole-image plane architecture is tracked separately (issue #353).

#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_BYTES: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record(layout.size());
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

fn record(size: usize) {
    COUNTING.with(|c| {
        if c.get() {
            ALLOC_BYTES.with(|b| b.set(b.get() + size));
        }
    });
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Total bytes allocated by `f` on a dedicated thread.
fn measure_bytes<F: FnOnce() + Send + 'static>(f: F) -> usize {
    std::thread::spawn(move || {
        ALLOC_BYTES.with(|b| b.set(0));
        COUNTING.with(|c| c.set(true));
        f();
        COUNTING.with(|c| c.set(false));
        ALLOC_BYTES.with(|b| b.get())
    })
    .join()
    .expect("measurement thread panicked")
}

#[test]
fn h2v1_1080p_decode_does_not_materialise_full_res_chroma() {
    let jpeg_422 = std::fs::read("tests/fixtures/photo_1920x1080_422.jpg")
        .expect("photo_1920x1080_422.jpg fixture required");

    // Warm-up decode outside the measured window (process-global state).
    let warm = libjpeg_turbo_rs::decompress(&jpeg_422).expect("warm-up decode");
    assert_eq!((warm.width, warm.height), (1920, 1080));

    let bytes_422 = measure_bytes(move || {
        let img = libjpeg_turbo_rs::decompress(&jpeg_422).expect("decode 422");
        assert_eq!((img.width, img.height), (1920, 1080));
        assert_eq!(img.data.len(), 1920 * 1080 * 3);
    });

    eprintln!("photo_1920x1080_422 decode: {bytes_422} bytes allocated");

    // Budget: output (6.22 MB) + Y (2.07 MB) + half-res chroma (2.07 MB)
    // + small fixed cost, with ~10% headroom => 11 MiB. The pre-fix
    // full-plane path allocates ≈ 14.5 MB and must fail this.
    assert!(
        bytes_422 <= 11 * 1024 * 1024,
        "4:2:2 1080p decode must stream chroma rows (≤ 11 MiB), got {bytes_422} — \
         full-resolution cb_full/cr_full buffers are back (issue #350)"
    );
}

/// H1V2 (4:4:0) streaming-path witness: byte-exactness alone cannot
/// detect a gate regression (the generic fallback produces identical
/// pixels), but the fallback's two full-resolution chroma planes add
/// ~4.2 MB at 1080p, which this ceiling catches (review of #350).
#[test]
fn h1v2_1080p_decode_does_not_materialise_full_res_chroma() {
    // No committed 1080p 4:4:0 fixture exists; synthesise one by
    // re-encoding the 4:2:0 photo fixture's pixels at S440.
    let jpeg_420 = std::fs::read("tests/fixtures/photo_1920x1080_420.jpg")
        .expect("photo_1920x1080_420.jpg fixture required");
    let src = libjpeg_turbo_rs::decompress(&jpeg_420).expect("decode source");
    let jpeg_440 = libjpeg_turbo_rs::compress(
        &src.data,
        1920,
        1080,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        libjpeg_turbo_rs::Subsampling::S440,
    )
    .expect("encode 4:4:0");

    let warm = libjpeg_turbo_rs::decompress(&jpeg_440).expect("warm-up decode");
    assert_eq!((warm.width, warm.height), (1920, 1080));

    let bytes_440 = measure_bytes(move || {
        let img = libjpeg_turbo_rs::decompress(&jpeg_440).expect("decode 440");
        assert_eq!((img.width, img.height), (1920, 1080));
        assert_eq!(img.data.len(), 1920 * 1080 * 3);
    });

    eprintln!("synthetic_1920x1080_440 decode: {bytes_440} bytes allocated");

    // Same budget model as 4:2:2, with 4:4:0's 8×16 iMCUs padding the
    // planes to 1088 rows: output (6.22 MB) + Y (2.09 MB) + half-res
    // chroma (2 x 1.04 MB) + row scratch => ~10.4 MB; the full-plane
    // fallback adds ~4.2 MB and must fail this.
    assert!(
        bytes_440 <= 11 * 1024 * 1024,
        "4:4:0 1080p decode must stream chroma rows (≤ 11 MiB), got {bytes_440} — \
         the H1V2 streaming gate regressed to the full-plane path (issue #350)"
    );
}
