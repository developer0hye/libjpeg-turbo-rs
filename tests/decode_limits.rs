//! Decoder resource limits (issue #355): configurable caps on
//! dimensions, pixel count, scan count, and estimated memory, enforced
//! before the corresponding allocation with typed errors.
//!
//! The counting-allocator test is native-only (WASI has no threads).

#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

mod helpers;

use libjpeg_turbo_rs::{DecodeLimits, Decoder, JpegError, PixelFormat, Subsampling};

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

/// A syntactically valid stream whose SOF declares 65535x65535 —
/// 4.29 gigapixels from a few hundred header bytes.
fn header_bomb_65535() -> Vec<u8> {
    let pixels = vec![128u8; 8 * 8];
    let mut jpeg =
        libjpeg_turbo_rs::compress(&pixels, 8, 8, PixelFormat::Grayscale, 75, Subsampling::S444)
            .expect("encode");
    let sof = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("SOF0 present");
    // SOF payload: len(2) precision(1) height(2) width(2) ...
    jpeg[sof + 5] = 0xFF;
    jpeg[sof + 6] = 0xFF;
    jpeg[sof + 7] = 0xFF;
    jpeg[sof + 8] = 0xFF;
    jpeg
}

/// Issue #355 criterion: the 65535x65535 header bomb errors out with the
/// DEFAULT limits, before any plane allocation (< 1 MB total).
#[test]
fn header_bomb_rejected_with_bounded_allocation_by_default() {
    let jpeg = header_bomb_65535();
    // Warm process-global lazy state (std Huffman tables, SIMD dispatch)
    // with a normal decode so the measurement is the rejection cost only.
    let pixels = vec![128u8; 8 * 8];
    let warm =
        libjpeg_turbo_rs::compress(&pixels, 8, 8, PixelFormat::Grayscale, 75, Subsampling::S444)
            .expect("warm encode");
    let _ = libjpeg_turbo_rs::decompress(&warm).expect("warm decode");

    let bytes = measure_bytes(move || {
        let result = libjpeg_turbo_rs::decompress(&jpeg);
        match result {
            Err(JpegError::LimitExceeded { what, actual, .. }) => {
                // 65_535 > the 65_500 default width cap (C's
                // JPEG_MAX_DIMENSION), which fires before the pixel
                // product check.
                assert_eq!(what, "image width");
                assert_eq!(actual, 65_535);
            }
            other => panic!("expected LimitExceeded, got {other:?}"),
        }
    });
    eprintln!("header bomb allocation: {bytes} bytes");
    assert!(
        bytes < 1024 * 1024,
        "rejecting the header bomb must not allocate ({bytes} bytes)"
    );
}

/// Width/height caps fire individually (before the pixel product).
#[test]
fn dimension_caps_are_enforced() {
    let jpeg = header_bomb_65535();
    let mut decoder = Decoder::new(&jpeg).expect("parse");
    decoder.set_limits(DecodeLimits {
        max_width: 1024,
        ..DecodeLimits::default()
    });
    match decoder.decode_image() {
        Err(JpegError::LimitExceeded { what, .. }) => assert_eq!(what, "image width"),
        other => panic!("expected width LimitExceeded, got {other:?}"),
    }
}

/// Issue #355 criterion: a progressive stream with more scans than
/// `max_scans` is rejected cleanly.
#[test]
fn scan_cap_rejects_progressive_scan_counts_above_limit() {
    let mut pixels = vec![0u8; 32 * 32];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i % 251) as u8;
    }
    let jpeg = libjpeg_turbo_rs::compress_progressive(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        85,
        Subsampling::S444,
    )
    .expect("progressive encode");

    // Sanity: decodes fine with defaults.
    let baseline = libjpeg_turbo_rs::decompress(&jpeg).expect("default decode");
    assert_eq!((baseline.width, baseline.height), (32, 32));

    let mut decoder = Decoder::new(&jpeg).expect("parse");
    decoder.set_limits(DecodeLimits {
        max_scans: 2,
        ..DecodeLimits::default()
    });
    match decoder.decode_image() {
        Err(JpegError::LimitExceeded { what, limit, .. }) => {
            // The upfront header check fires before any coefficient
            // allocation (codex P2) — earlier than the per-scan loop
            // check, which remains for incremental consumers.
            assert_eq!(what, "scan count");
            assert_eq!(limit, 2);
        }
        other => panic!("expected scan LimitExceeded, got {other:?}"),
    }
}

/// `DecodeLimits::strict()` still accepts normal images.
#[test]
fn strict_limits_accept_ordinary_images() {
    let jpeg = std::fs::read("tests/fixtures/photo_320x240_420.jpg").expect("fixture");
    let mut decoder = Decoder::new(&jpeg).expect("parse");
    decoder.set_limits(DecodeLimits::strict());
    let img = decoder
        .decode_image()
        .expect("strict limits must accept 320x240");
    assert_eq!((img.width, img.height), (320, 240));
}

/// The memory ceiling uses the estimation model and errors typed.
#[test]
fn memory_ceiling_is_typed() {
    let jpeg = std::fs::read("tests/fixtures/photo_320x240_420.jpg").expect("fixture");
    let mut decoder = Decoder::new(&jpeg).expect("parse");
    decoder.set_limits(DecodeLimits {
        max_memory: Some(1024),
        ..DecodeLimits::default()
    });
    match decoder.decode_image() {
        Err(JpegError::LimitExceeded { what, .. }) => {
            assert_eq!(what, "estimated decode memory")
        }
        other => panic!("expected memory LimitExceeded, got {other:?}"),
    }
}

/// Codex P1 on #355: `output_buffer_size` is the first step of the
/// documented untrusted-input workflow (size -> allocate -> decode), so
/// the limits must fire there or the caller OOMs on its own allocation.
#[test]
fn output_buffer_size_enforces_limits() {
    let jpeg = header_bomb_65535();
    let mut decoder = Decoder::new(&jpeg).expect("parse");
    decoder.set_output_format(PixelFormat::Rgb);
    match decoder.output_buffer_size() {
        Err(JpegError::LimitExceeded { what, .. }) => assert_eq!(what, "image width"),
        other => panic!("expected LimitExceeded from sizing, got {other:?}"),
    }
}

/// Codex P1 on #355: `decode_raw` bypassed every limit.
#[test]
fn decode_raw_enforces_limits() {
    let jpeg = header_bomb_65535();
    let decoder = Decoder::new(&jpeg).expect("parse");
    match decoder.decode_raw() {
        Err(JpegError::LimitExceeded { what, .. }) => assert_eq!(what, "image width"),
        other => panic!(
            "expected LimitExceeded from decode_raw, got {:?}",
            other.map(|_| "raw image")
        ),
    }
}

/// Codex P2 on #355: sequential multi-scan (non-interleaved baseline)
/// streams never reached the progressive-loop scan check.
#[test]
fn scan_cap_applies_to_non_interleaved_baseline() {
    let cjpeg: std::path::PathBuf = require_c_tool!("cjpeg");
    let dir = std::env::temp_dir().join(format!("scan_cap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let ppm = dir.join("in.ppm");
    let scans = dir.join("scans.txt");
    let mut body = Vec::new();
    for y in 0..32u32 {
        for x in 0..32u32 {
            body.extend_from_slice(&[(x * 8) as u8, (y * 8) as u8, ((x ^ y) * 8) as u8]);
        }
    }
    let mut f = std::fs::File::create(&ppm).expect("ppm");
    use std::io::Write;
    write!(f, "P6\n32 32\n255\n").unwrap();
    f.write_all(&body).unwrap();
    drop(f);
    std::fs::write(&scans, "0: 0 63 0 0;\n1: 0 63 0 0;\n2: 0 63 0 0;\n").unwrap();
    let out = std::process::Command::new(&cjpeg)
        .args(["-sample", "1x1", "-scans"])
        .arg(&scans)
        .arg(&ppm)
        .output()
        .expect("cjpeg");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let jpeg = out.stdout;
    let _ = std::fs::remove_dir_all(&dir);

    // Three-scan sequential stream: decodes with defaults...
    let mut d = Decoder::new(&jpeg).expect("parse");
    d.set_output_format(PixelFormat::Rgb);
    d.decode_image().expect("default decode");
    // ...and is rejected when max_scans is below the scan count.
    let mut d = Decoder::new(&jpeg).expect("parse");
    d.set_limits(DecodeLimits {
        max_scans: 1,
        ..DecodeLimits::default()
    });
    match d.decode_image() {
        Err(JpegError::LimitExceeded { what, .. }) => assert_eq!(what, "scan count"),
        other => panic!("expected scan-count LimitExceeded, got {other:?}"),
    }
}

/// Codex P2 on #355: `new_with_limits` bounds ScanInfo buffering during
/// the header walk itself, so a strict policy pays no parse-stage cost
/// for a scan bomb.
#[test]
fn new_with_limits_caps_parse_stage_scans() {
    // A 200-scan progressive stream (synthetic scan bomb, small).
    let mut pixels = vec![0u8; 16 * 16];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i * 7 % 251) as u8;
    }
    let base = libjpeg_turbo_rs::compress_progressive(
        &pixels,
        16,
        16,
        PixelFormat::Grayscale,
        85,
        Subsampling::S444,
    )
    .expect("encode");
    // Duplicate the final AC-refine scan many times by splicing before EOI.
    let eoi = base.len() - 2;
    // Find the last SOS to duplicate (marker walk from the end is fiddly;
    // duplicating the tail region between the last FFDA and EOI works for
    // this synthetic stream).
    let last_sos = base
        .windows(2)
        .rposition(|w| w == [0xFF, 0xDA])
        .expect("SOS");
    let tail = base[last_sos..eoi].to_vec();
    let mut bomb = base[..eoi].to_vec();
    for _ in 0..200 {
        bomb.extend_from_slice(&tail);
    }
    bomb.extend_from_slice(&[0xFF, 0xD9]);

    match Decoder::new_with_limits(&bomb, DecodeLimits::strict()) {
        Err(JpegError::LimitExceeded { what, limit, .. }) => {
            assert_eq!(what, "scan count at parse");
            assert_eq!(limit, 100, "strict max_scans threads into the parser");
        }
        Ok(_) => panic!("strict limits must reject a 200-scan bomb at parse"),
        Err(other) => panic!("expected parse-stage LimitExceeded, got {other:?}"),
    }
}
