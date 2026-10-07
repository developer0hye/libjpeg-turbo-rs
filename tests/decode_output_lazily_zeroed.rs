//! Issue #659 (P4-228): the owned decode destination comes from the
//! allocator's zeroed path, as 0.8.0's `vec![0u8; size]` did.
//!
//! P4-209 made that allocation fallible with `try_reserve_exact` followed by
//! `resize`, which writes every byte before the decoder overwrites them. On
//! a 33 MP output that extra pass made fresh decodes 2–5 % slower than
//! 0.8.0 in every downstream run (`experiments/downstream/BUDGETS.md`).
//! `alloc_zeroed` (`calloc`) lets the kernel hand out pages that are
//! already zero, so the buffer is written once. The timing lives in the
//! downstream harness; this pins the mechanism: the output-sized block must
//! arrive through `alloc_zeroed`, not through `alloc` plus a fill.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;

use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

struct ZeroedProbe;

static WATCHING: AtomicBool = AtomicBool::new(false);
static WATCHED_SIZE: AtomicUsize = AtomicUsize::new(0);
static ZEROED_HITS: AtomicUsize = AtomicUsize::new(0);
static PLAIN_HITS: AtomicUsize = AtomicUsize::new(0);
/// The counters are process-wide; tests in this binary take turns.
static SERIAL: Mutex<()> = Mutex::new(());

// SAFETY: every method forwards to `System` unchanged and only reads or bumps
// atomics, so `System`'s guarantees carry over.
unsafe impl GlobalAlloc for ZeroedProbe {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if WATCHING.load(Ordering::Relaxed) && layout.size() == WATCHED_SIZE.load(Ordering::Relaxed)
        {
            PLAIN_HITS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if WATCHING.load(Ordering::Relaxed) && layout.size() == WATCHED_SIZE.load(Ordering::Relaxed)
        {
            ZEROED_HITS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: ZeroedProbe = ZeroedProbe;

/// A photo-like gradient, so the encoder produces a real baseline stream.
fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 255 / width) as u8);
            pixels.push((y * 255 / height) as u8);
            pixels.push(((x + y) * 127 / (width + height)) as u8);
        }
    }
    pixels
}

fn assert_output_is_zeroed_allocation(subsampling: Subsampling, format: PixelFormat) {
    let _turn = SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    // Odd dimensions so the output size matches no plane or MCU-padded buffer.
    let (width, height): (usize, usize) = (333, 197);
    let jpeg: Vec<u8> = compress(
        &gradient_rgb(width, height),
        width,
        height,
        PixelFormat::Rgb,
        90,
        subsampling,
    )
    .expect("encode the probe image");
    let output_bytes: usize = width * height * format.bytes_per_pixel();

    ZEROED_HITS.store(0, Ordering::Relaxed);
    PLAIN_HITS.store(0, Ordering::Relaxed);
    WATCHED_SIZE.store(output_bytes, Ordering::Relaxed);
    WATCHING.store(true, Ordering::Relaxed);
    let decoded = decompress_to(&jpeg, format);
    WATCHING.store(false, Ordering::Relaxed);
    let image = decoded.expect("decode the probe image");

    assert_eq!(image.data.len(), output_bytes);
    assert_eq!(
        ZEROED_HITS.load(Ordering::Relaxed),
        1,
        "the {output_bytes}-byte output must come from alloc_zeroed ({subsampling:?}, {format:?})"
    );
    assert_eq!(
        PLAIN_HITS.load(Ordering::Relaxed),
        0,
        "no output-sized block may come from plain alloc + fill ({subsampling:?}, {format:?})"
    );
}

/// Issue #659: 4:2:0 RGB, the downstream harness's 8K and 12 MP shape.
#[test]
fn owned_rgb_output_420_is_allocated_zeroed() {
    assert_output_is_zeroed_allocation(Subsampling::S420, PixelFormat::Rgb);
}

/// Issue #659: the same contract for 4:4:4 and a four-byte format.
#[test]
fn owned_rgba_output_444_is_allocated_zeroed() {
    assert_output_is_zeroed_allocation(Subsampling::S444, PixelFormat::Rgba);
}
