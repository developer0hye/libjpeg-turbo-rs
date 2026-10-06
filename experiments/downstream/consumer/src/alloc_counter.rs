//! Counting global allocator.
//!
//! Wraps the system allocator and keeps four process-wide counters: number of
//! allocation events, cumulative bytes requested, bytes currently live, and the
//! peak of live bytes. `measure` snapshots them around a closure.
//!
//! The counters are atomics with `Relaxed` ordering. The harness is
//! single-threaded and none of the measured backends spawn threads in the
//! configuration built here (image without `rayon`, zune-jpeg without its
//! threading feature, libjpeg-turbo-rs is single-threaded), so the counts are
//! exact. Atomics are used only because `GlobalAlloc` must be `Sync`; if a
//! backend ever did allocate from another thread the totals would still be
//! correct, but "peak live" would be an approximation.
//!
//! Allocation stats are taken in a dedicated pass outside the timed loop:
//! the atomic traffic is cheap but not free, and it is the same for every
//! backend, so it does not bias the comparison — it is kept out of the timings
//! anyway so that the timing numbers have nothing in them but the decode.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

pub struct CountingAllocator;

static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);

fn record_growth(bytes: usize) {
    let live: usize = LIVE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK_LIVE_BYTES.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only adds counter updates, so `System`'s guarantees carry
// over unchanged.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer: *mut u8 = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            record_growth(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer: *mut u8 = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOCATED_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
            record_growth(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        LIVE_BYTES.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    /// A reallocation counts as one allocation event. Cumulative bytes grow
    /// by the *growth* only (a `Vec` doubling from 8 MiB to 16 MiB requested
    /// 8 MiB more, not 16), and live bytes move by the signed difference.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer: *mut u8 = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            let old_size: usize = layout.size();
            if new_size >= old_size {
                let growth: usize = new_size - old_size;
                ALLOCATED_BYTES.fetch_add(growth as u64, Ordering::Relaxed);
                record_growth(growth);
            } else {
                LIVE_BYTES.fetch_sub(old_size - new_size, Ordering::Relaxed);
            }
        }
        new_pointer
    }
}

/// What one closure allocated.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocStats {
    /// Allocation events (`alloc`, `alloc_zeroed`, `realloc`).
    pub count: u64,
    /// Bytes requested, cumulative over the closure.
    pub bytes: u64,
    /// Highest live heap during the closure, measured above the live heap at
    /// its start — i.e. the closure's own working set, including the output
    /// it returns.
    pub peak_live: u64,
}

/// Run `work` once and report what it allocated. The return value is dropped
/// *after* the snapshot so a library-owned output buffer is counted in
/// `peak_live` like any other allocation the caller would hold.
pub fn measure<R>(work: impl FnOnce() -> R) -> (R, AllocStats) {
    let live_at_start: usize = LIVE_BYTES.load(Ordering::Relaxed);
    PEAK_LIVE_BYTES.store(live_at_start, Ordering::Relaxed);
    let count_at_start: u64 = ALLOCATION_COUNT.load(Ordering::Relaxed);
    let bytes_at_start: u64 = ALLOCATED_BYTES.load(Ordering::Relaxed);

    let result: R = work();

    let stats: AllocStats = AllocStats {
        count: ALLOCATION_COUNT.load(Ordering::Relaxed) - count_at_start,
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed) - bytes_at_start,
        peak_live: PEAK_LIVE_BYTES
            .load(Ordering::Relaxed)
            .saturating_sub(live_at_start) as u64,
    };
    (result, stats)
}
