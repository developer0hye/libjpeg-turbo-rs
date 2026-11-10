//! Peak heap of `Decoder::into_raw_strips` on sequential streams.
//!
//! The strip reader exists so a caller can consume raw planes without the
//! decoder holding a whole-image copy (P4-218 is the same defect on the pixel
//! path). On a single-scan Huffman stream its working set must therefore be
//! bounded by the frame's width times one iMCU row, not by width x height.
//! `decode_raw` is measured alongside, so a broken meter cannot pass.
//!
//! One `#[test]`, because the counting allocator is process-global and
//! parallel tests would add their allocations to each other's peaks.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use libjpeg_turbo_rs::{compress, decompress_raw, Decoder, PixelFormat, RawImage, Subsampling};

struct CountingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: forwarded unchanged to the system allocator.
        let ptr: *mut u8 = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live: usize = LIVE.fetch_add(layout.size(), Ordering::SeqCst) + layout.size();
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above with this `layout`.
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::SeqCst);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Peak live heap above the level at entry, while `work` runs.
fn peak_during(work: impl FnOnce()) -> usize {
    let start: usize = LIVE.load(Ordering::SeqCst);
    PEAK.store(start, Ordering::SeqCst);
    work();
    PEAK.load(Ordering::SeqCst) - start
}

fn noise_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut state: u32 = 0x1234_5678;
    (0..width * height * 3)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

/// Decode every strip, touching each sample so nothing is optimized out.
fn drain_strips(jpeg: &[u8]) -> u64 {
    let decoder: Decoder<'_> = Decoder::new(jpeg).expect("header");
    let mut sum: u64 = 0;
    decoder
        .into_raw_strips()
        .expect("reader")
        .for_each_strip(|strip| {
            for plane in strip.planes() {
                for row in plane.rows() {
                    sum += row.iter().map(|&b| u64::from(b)).sum::<u64>();
                }
            }
            Ok(())
        })
        .expect("strips");
    sum
}

#[test]
fn strip_reader_working_set_is_one_imcu_row_on_sequential_streams() {
    const WIDTH: usize = 2048;
    const HEIGHT: usize = 2048;
    let pixels: Vec<u8> = noise_rgb(WIDTH, HEIGHT);

    for (label, subsampling, format) in [
        ("4:2:0", Subsampling::S420, PixelFormat::Rgb),
        ("4:4:4", Subsampling::S444, PixelFormat::Rgb),
        ("grayscale", Subsampling::S444, PixelFormat::Grayscale),
    ] {
        let source: Vec<u8> = if format == PixelFormat::Grayscale {
            pixels.iter().step_by(3).copied().collect()
        } else {
            pixels.clone()
        };
        let jpeg: Vec<u8> =
            compress(&source, WIDTH, HEIGHT, format, 85, subsampling).expect("encode");

        let raw_peak: usize = peak_during(|| {
            let raw: RawImage = decompress_raw(&jpeg).expect("decode_raw");
            assert!(!raw.planes.is_empty());
        });
        // Sanity: the whole-frame decode holds at least the luma plane.
        assert!(
            raw_peak >= WIDTH * HEIGHT,
            "[{label}] meter broken: decode_raw peak {raw_peak} below one plane"
        );

        let scratch: usize = Decoder::new(&jpeg)
            .unwrap()
            .into_raw_strips()
            .unwrap()
            .geometry()
            .components
            .iter()
            .map(|c| c.padded_width * c.rows_per_strip)
            .sum();
        let strip_peak: usize = peak_during(|| {
            drain_strips(&jpeg);
        });
        // The scratch strip plus fixed overhead (Huffman lookup tables, the
        // MCU plan, the decoder): measured at 18.6 KiB for the three-component
        // layouts and 9.6 KiB for grayscale (aarch64-darwin, 2026-11-09), so
        // 24 KiB leaves a small margin. A
        // regression that kept even a few extra image rows would exceed it.
        let bound: usize = scratch + 24 * 1024;
        eprintln!("[{label}] decode_raw peak {raw_peak} B, strip peak {strip_peak} B");
        assert!(
            strip_peak <= bound,
            "[{label}] strip reader peak {strip_peak} B exceeds the one-row bound {bound} B \
             (decode_raw: {raw_peak} B)"
        );
    }
}
