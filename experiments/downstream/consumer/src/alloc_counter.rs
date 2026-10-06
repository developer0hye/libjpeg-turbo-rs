//! Counting global allocator.
//!
//! Wraps the system allocator and, while a [`measure`] window is open, keeps
//! four process-wide counters: allocation events, cumulative bytes requested,
//! the live-heap delta since the window opened, and that delta's peak.
//!
//! Counting is switched on only inside [`measure`]. Outside it — in particular
//! in every timed region — each allocator call costs one `Relaxed` load of the
//! `COUNTING` flag on top of `System`, the same for every backend, and no
//! counter is written.
//!
//! Live bytes are tracked as a *signed* delta from the moment the window
//! opens, so a block allocated before the window and freed inside it lowers
//! the delta below zero instead of underflowing an unsigned counter. Peak live
//! is the highest that delta reached, i.e. the closure's working set above
//! the heap it started with.
//!
//! The counters are `Relaxed` atomics, and every figure is exact even when the
//! measured closure allocates and frees from several threads, provided the
//! closure joins every thread it starts before it returns. The concurrent
//! workload's `std::thread::scope` does, and the measured libraries start no
//! threads in this build (image without `rayon`; zune-jpeg 0.5 has no
//! threading feature; libjpeg-turbo-rs is single-threaded):
//!
//! - count and cumulative bytes are sums of `fetch_add`, which loses no update;
//! - every update of `LIVE_DELTA` is a read-modify-write on one atomic, so they
//!   form a single modification order and each `fetch_add` returns the value
//!   immediately before its own. `previous + delta` is therefore exactly the
//!   next value in that order, every value the counter ever holds is passed to
//!   `fetch_max`, and a maximum does not depend on the order those calls land
//!   in. A block allocated on one thread and freed on another moves the same
//!   global delta, so it needs no per-thread bookkeeping;
//! - joining the threads orders all their updates before the final loads.
//!
//! "Exact" is about the counter: a call is counted a few instructions after
//! `System` returns. The peak is the true peak of the live heap in the
//! interleaving where each call takes effect at its counter update — one the
//! same threads could have run. The allocator's own instantaneous state can
//! differ from it only by calls caught between `System` returning and their
//! update, at most one per thread. The figure is heap requested through
//! `GlobalAlloc`: thread stacks (mapped directly), allocator overhead and
//! retention, and code pages are not in it, so it is not resident memory.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};

pub struct CountingAllocator;

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATION_COUNT: AtomicU64 = AtomicU64::new(0);
static ALLOCATED_BYTES: AtomicU64 = AtomicU64::new(0);
static LIVE_DELTA: AtomicI64 = AtomicI64::new(0);
static PEAK_LIVE_DELTA: AtomicI64 = AtomicI64::new(0);

fn counting() -> bool {
    COUNTING.load(Ordering::Relaxed)
}

fn record_allocation(bytes: usize) {
    ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    record_live_change(bytes as i64);
}

fn record_live_change(delta: i64) {
    let live: i64 = LIVE_DELTA.fetch_add(delta, Ordering::Relaxed) + delta;
    PEAK_LIVE_DELTA.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: every method forwards to `System` with the caller's arguments
// unchanged and only adds counter updates, so `System`'s guarantees carry
// over unchanged.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer: *mut u8 = unsafe { System.alloc(layout) };
        if !pointer.is_null() && counting() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer: *mut u8 = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() && counting() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
        if counting() {
            record_live_change(-(layout.size() as i64));
        }
    }

    /// A reallocation counts as one allocation event. Cumulative bytes grow
    /// by the *growth* only (a `Vec` doubling from 8 MiB to 16 MiB requested
    /// 8 MiB more, not 16), and live bytes move by the signed difference.
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_pointer: *mut u8 = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && counting() {
            ALLOCATION_COUNT.fetch_add(1, Ordering::Relaxed);
            let old_size: usize = layout.size();
            ALLOCATED_BYTES.fetch_add(new_size.saturating_sub(old_size) as u64, Ordering::Relaxed);
            record_live_change(new_size as i64 - old_size as i64);
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
    ALLOCATION_COUNT.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LIVE_DELTA.store(0, Ordering::Relaxed);
    PEAK_LIVE_DELTA.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);

    let result: R = work();

    COUNTING.store(false, Ordering::Relaxed);
    let stats: AllocStats = AllocStats {
        count: ALLOCATION_COUNT.load(Ordering::Relaxed),
        bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live: PEAK_LIVE_DELTA.load(Ordering::Relaxed).max(0) as u64,
    };
    (result, stats)
}

/// The counters are process-wide and `cargo test` runs tests on parallel
/// threads, so a test that allocates heavily (a decode) holds this while the
/// window test below runs, or its allocations would land in that window.
#[cfg(test)]
pub static HEAVY_ALLOCATION_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    // Lower bounds only: the test harness runs other tests on other threads,
    // and their small allocations land in an open window too.
    #[test]
    fn a_window_sees_the_closures_allocation_and_is_closed_afterwards() {
        let _serial = HEAVY_ALLOCATION_TESTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let (buffer, stats) = measure(|| vec![7u8; 100_000]);
        assert!(stats.count >= 1);
        assert!(stats.bytes >= 100_000);
        assert!(stats.peak_live >= 100_000);
        assert!(!counting(), "the window must close when measure returns");
        // Freeing a block allocated before the window drives the live delta
        // negative; peak is clamped at zero rather than underflowing.
        let ((), after) = measure(move || drop(buffer));
        assert!(after.peak_live < 100_000);
    }
}
