//! Allocation-budget regression for the 4:2:2 (H2V1) decode path
//! (issue #350): H2V1 used to fall through to the generic upsample path,
//! which materialises two full-resolution chroma planes (`cb_full` /
//! `cr_full`, ~4.2 MB extra for 1080p) before colour conversion, while
//! H2V2 (4:2:0) streams rows and never allocates them.
//!
//! Budget model for 1920×1080 4:2:2 with the row-streaming path:
//!   output RGB   1920×1080×3      ≈ 6.22 MB
//!   Y plane      1920×1088        ≈ 2.09 MB
//!   Cb+Cr planes 2 × 960×1088     ≈ 2.09 MB   (decoded, half-res — intrinsic)
//!   row scratch  a few × full_width
//!   ------------------------------------------------
//!   total        ≈ 10.5 MB
//!
//! The pre-fix path allocated ≈ 14.6 MB (the two full-res chroma planes
//! on top). The assertion pins the post-fix ceiling with margin so the
//! full-plane fallback cannot silently return for standard 4:2:2.
//!
//! Note: 4:2:2 cannot reach 4:2:0's total (≈ 9.4 MB) exactly — its
//! decoded chroma planes are half-resolution rather than
//! quarter-resolution, an intrinsic +1.04 MB at 1080p. Removing the
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

    // Budget: output (6.22 MB) + Y (2.09 MB) + half-res chroma (2.09 MB)
    // + small fixed cost, with ~5% headroom => 11 MB. The pre-fix
    // full-plane path allocates ≈ 14.6 MB and must fail this.
    assert!(
        bytes_422 <= 11 * 1024 * 1024,
        "4:2:2 1080p decode must stream chroma rows (≤ 11 MB), got {bytes_422} — \
         full-resolution cb_full/cr_full buffers are back (issue #350)"
    );
}
