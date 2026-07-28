//! Allocation-budget regression tests for small-image decode fixed cost.
//!
//! Issue #351: an 8×8 grayscale decode used to cost 44 heap allocations /
//! 60,542 bytes because `Decoder::new` eagerly built all four standard
//! Huffman tables (4 KB fast-LUT each) and deep-cloned them into every
//! unset slot and every per-scan snapshot. zune-jpeg does the same decode
//! in 3 allocations / 816 bytes.
//!
//! These tests pin the fixed cost with a counting global allocator so a
//! future change that reintroduces eager table building or per-decode
//! deep clones fails loudly.
//!
//! The counting allocator is process-global, so each budget test decodes
//! on a dedicated thread and counts only that thread's allocations
//! (thread-local counters); the cargo test harness's own threads don't
//! pollute the measurement.
//!
//! Excluded on wasm32: WASI has no `std::thread::spawn`, and allocation
//! budgets are pinned by the native jobs anyway (codex P1 on #351).

#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
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
        // A grow-in-place is still a logical allocation event for budget
        // purposes; count the full new size like a fresh allocation.
        record(new_size);
        System.realloc(ptr, layout, new_size)
    }
}

fn record(size: usize) {
    // `thread_local` access can itself allocate on first touch; the
    // counters are primed before counting is enabled (see `measure`).
    COUNTING.with(|c| {
        if c.get() {
            ALLOC_COUNT.with(|n| n.set(n.get() + 1));
            ALLOC_BYTES.with(|b| b.set(b.get() + size));
        }
    });
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Run `f` on a fresh thread and return (allocations, bytes) performed by
/// the calls inside `f` on that thread.
fn measure<F: FnOnce() + Send + 'static>(f: F) -> (usize, usize) {
    std::thread::spawn(move || {
        // Prime the thread-local counters before enabling counting.
        ALLOC_COUNT.with(|n| n.set(0));
        ALLOC_BYTES.with(|b| b.set(0));
        COUNTING.with(|c| c.set(true));
        f();
        COUNTING.with(|c| c.set(false));
        (ALLOC_COUNT.with(|n| n.get()), ALLOC_BYTES.with(|b| b.get()))
    })
    .join()
    .expect("measurement thread panicked")
}

fn gray_8x8_jpeg() -> Vec<u8> {
    // Non-uniform content so the AC table is exercised.
    let mut pixels = vec![0u8; 8 * 8];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = ((i % 8) * 32) as u8 ^ ((i / 8) * 16) as u8;
    }
    libjpeg_turbo_rs::compress(
        &pixels,
        8,
        8,
        libjpeg_turbo_rs::PixelFormat::Grayscale,
        90,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("encode gray 8x8")
}

fn blue_16x16_420_jpeg() -> Vec<u8> {
    let pixels = vec![[30u8, 60, 200]; 16 * 16].concat();
    libjpeg_turbo_rs::compress(
        &pixels,
        16,
        16,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        90,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("encode blue 16x16")
}

/// Issue #351 acceptance: ≤ 8 allocations and ≤ 12 KB for an 8×8
/// grayscale decode (was 44 allocations / 60,542 bytes).
#[test]
fn gray_8x8_decode_allocation_budget() {
    let jpeg = gray_8x8_jpeg();

    // Warm up process-global lazy state (std Huffman tables, SIMD
    // dispatch) so the budget measures per-decode cost only.
    let warm = libjpeg_turbo_rs::decompress(&jpeg).expect("warm-up decode");
    assert_eq!((warm.width, warm.height), (8, 8));

    let (allocs, bytes) = measure(move || {
        let img = libjpeg_turbo_rs::decompress(&jpeg).expect("decode");
        assert_eq!((img.width, img.height), (8, 8));
        assert_eq!(img.data.len(), 64);
    });

    eprintln!("gray_8x8 decode: {allocs} allocations, {bytes} bytes");
    // Windows measures one more allocation than the 8 pinned elsewhere —
    // 9 allocations / 9,634 bytes on x86_64-pc-windows-msvc, reproduced
    // on a clean tree (P4-62, 2026-07-27) and stable across the CI job
    // issue #378 added. The extra event's origin inside std's Windows
    // plumbing is unattributed; the budget stays exact-measured per
    // platform rather than loosened globally.
    let alloc_budget: usize = if cfg!(windows) { 9 } else { 8 };
    assert!(
        allocs <= alloc_budget,
        "8x8 grayscale decode must stay within {alloc_budget} allocations, got {allocs}"
    );
    assert!(
        bytes <= 12 * 1024,
        "8x8 grayscale decode must stay within 12 KB allocated, got {bytes}"
    );
}

/// The 16×16 4:2:0 colour case from issue #351's measurement table
/// (was 56 allocations / 64,896 bytes). Budget: the two entropy tables
/// plus per-component plane/output buffers — no eager std-table cost.
#[test]
fn blue_16x16_420_decode_allocation_budget() {
    let jpeg = blue_16x16_420_jpeg();

    let warm = libjpeg_turbo_rs::decompress(&jpeg).expect("warm-up decode");
    assert_eq!((warm.width, warm.height), (16, 16));

    let (allocs, bytes) = measure(move || {
        let img = libjpeg_turbo_rs::decompress(&jpeg).expect("decode");
        assert_eq!((img.width, img.height), (16, 16));
        assert_eq!(img.data.len(), 16 * 16 * 3);
    });

    eprintln!("blue_16x16_420 decode: {allocs} allocations, {bytes} bytes");
    assert!(
        allocs <= 24,
        "16x16 4:2:0 decode must stay within 24 allocations, got {allocs}"
    );
    assert!(
        bytes <= 24 * 1024,
        "16x16 4:2:0 decode must stay within 24 KB allocated, got {bytes}"
    );
}
