//! What the adapter allocates, measured with a counting global allocator
//! (#637 criteria 1 and 2).
//!
//! `libjpeg-turbo-rs-image 0.1.0` decoded the whole image inside
//! `JpegDecoder::new` and copied it out in `read_image`, so a read held two
//! full decoded images at once — the decoder's and the caller's. These
//! assertions pin the replacement: construction parses headers only, and the
//! read's peak working set stays below one decoded image.
//!
//! Its own test binary with a single `#[test]` so nothing else allocates on
//! another thread while it measures; the counters are thread-local anyway.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use image::ImageDecoder;
use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_image::JpegDecoder;

struct CountingAllocator;

thread_local! {
    static LIVE_BYTES: Cell<usize> = const { Cell::new(0) };
    static PEAK_BYTES: Cell<usize> = const { Cell::new(0) };
}

fn record_alloc(size: usize) {
    LIVE_BYTES.with(|live: &Cell<usize>| {
        let now: usize = live.get() + size;
        live.set(now);
        PEAK_BYTES.with(|peak: &Cell<usize>| peak.set(peak.get().max(now)));
    });
}

fn record_dealloc(size: usize) {
    LIVE_BYTES.with(|live: &Cell<usize>| live.set(live.get().saturating_sub(size)));
}

// SAFETY: every method forwards to `System` with the caller's layout and
// pointer unchanged; the bookkeeping only touches const-initialised
// thread-locals, which never allocate.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_alloc(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_alloc(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        record_dealloc(layout.size());
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        record_dealloc(layout.size());
        record_alloc(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Live bytes now, and resets the peak to it.
fn checkpoint() -> usize {
    let live: usize = LIVE_BYTES.with(Cell::get);
    PEAK_BYTES.with(|peak: &Cell<usize>| peak.set(live));
    live
}

fn peak_since_checkpoint() -> usize {
    PEAK_BYTES.with(Cell::get)
}

#[test]
fn construction_is_header_only_and_read_holds_no_second_image() {
    let (width, height): (usize, usize) = (2048, 1536);
    let mut rgb: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.extend_from_slice(&[(x % 251) as u8, (y % 241) as u8, ((x ^ y) % 256) as u8]);
        }
    }
    let jpeg: Vec<u8> =
        compress(&rgb, width, height, PixelFormat::Rgb, 85, Subsampling::S420).expect("encode");
    drop(rgb);
    let decoded_image_bytes: usize = width * height * 3;

    // Construction: a header parse over the owned stream.
    let before_new: usize = checkpoint();
    let decoder: JpegDecoder = JpegDecoder::from_vec(jpeg).expect("from_vec");
    let construction_peak: usize = peak_since_checkpoint() - before_new;
    eprintln!(
        "construction peak: {construction_peak} bytes (decoded image: {decoded_image_bytes})"
    );
    assert!(
        construction_peak < 64 * 1024,
        "construction allocated {construction_peak} bytes — it must not decode pixels"
    );

    // The read: the caller's buffer is allocated before the checkpoint, so the
    // peak above it is the decoder's own working set.
    let mut pixels: Vec<u8> = vec![0u8; decoder.total_bytes() as usize];
    let before_read: usize = checkpoint();
    decoder.read_image(&mut pixels).expect("read_image");
    let read_peak: usize = peak_since_checkpoint() - before_read;
    eprintln!(
        "read_image working-set peak: {read_peak} bytes (decoded image: {decoded_image_bytes})"
    );
    assert!(
        read_peak < decoded_image_bytes,
        "read_image peaked {read_peak} bytes above the caller's buffer, at least one more \
         decoded image ({decoded_image_bytes} bytes)"
    );
}
