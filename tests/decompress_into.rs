//! Caller-buffer decode API (issue #354): `decompress_into` /
//! `output_buffer_size` / `Decoder::decode_image_into`.
//!
//! Gates: (1) byte-identity with `decompress()` across a representative
//! fixture matrix including paths that stage internally; (2) a typed
//! `BufferTooSmall` error — never a panic or truncation — for short
//! buffers; (3) buffer reuse across differently-sized frames; (4) the
//! zero-output-allocation claim, pinned by a counting allocator
//! (native-only — WASI has no threads).

#![cfg(not(target_arch = "wasm32"))]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use libjpeg_turbo_rs::{decompress, decompress_into, output_buffer_size, PixelFormat};

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

/// Representative decode-path matrix: streamed subsamplings, 4:4:4,
/// grayscale, progressive, restart, non-interleaved 4:4:0, arithmetic,
/// and an ICC-carrying file (metadata must survive the into-path).
const FIXTURES: &[&str] = &[
    "tests/fixtures/photo_320x240_420.jpg",
    "tests/fixtures/photo_640x480_422.jpg",
    "tests/fixtures/photo_320x240_444.jpg",
    "tests/fixtures/gray_8x8.jpg",
    "tests/fixtures/photo_320x240_420_prog.jpg",
    "tests/fixtures/photo_640x480_420_rst.jpg",
    "tests/fixtures/real_world/zune_non_interleaved_440_64x64.jpg",
    "tests/fixtures/real_world/derived_227x149_arithmetic.jpg",
    "tests/fixtures/real_world/derived_227x149_icc_profile.jpg",
    "tests/fixtures/real_world/derived_227x149_grayscale_q90.jpg",
];

/// Issue #354 criterion: byte-identical to `decompress()` everywhere,
/// including the staged fallback paths.
#[test]
fn decompress_into_matches_decompress_across_fixture_matrix() {
    for path in FIXTURES {
        let jpeg = std::fs::read(path).unwrap_or_else(|_| panic!("{path} fixture required"));
        let reference = decompress(&jpeg).expect("decompress");
        let format = reference.pixel_format;

        let size = output_buffer_size(&jpeg, format).expect("output_buffer_size");
        assert!(
            size >= reference.data.len(),
            "{path}: output_buffer_size {size} under-reports actual {}",
            reference.data.len()
        );

        let mut out = vec![0xA5u8; size];
        let info = decompress_into(&jpeg, format, &mut out).expect("decompress_into");
        assert_eq!(
            (info.width, info.height, info.pixel_format),
            (reference.width, reference.height, reference.pixel_format),
            "{path}: metadata mismatch"
        );
        assert_eq!(info.bytes_written, reference.data.len(), "{path}");
        assert_eq!(
            &out[..info.bytes_written],
            &reference.data[..],
            "{path}: pixels differ from decompress()"
        );
        assert_eq!(
            info.icc_profile.is_some(),
            reference.icc_profile.is_some(),
            "{path}: ICC metadata lost"
        );
    }
}

/// Issue #354 criterion: a short buffer is a typed error, never a panic
/// or a truncated write.
#[test]
fn short_buffer_is_a_typed_error() {
    let jpeg = std::fs::read("tests/fixtures/photo_320x240_420.jpg").expect("fixture");
    let size = output_buffer_size(&jpeg, PixelFormat::Rgb).expect("size");

    for short in [0usize, 1, size - 1] {
        let mut out = vec![0u8; short];
        match decompress_into(&jpeg, PixelFormat::Rgb, &mut out) {
            Err(libjpeg_turbo_rs::JpegError::BufferTooSmall { need, got }) => {
                assert_eq!(got, short);
                assert!(need > short);
            }
            other => panic!("len {short}: expected BufferTooSmall, got {other:?}"),
        }
    }
}

/// Issue #354 criterion: one reused buffer serves ≥ 3 differently-sized
/// frames in sequence (the MJPEG/video pattern).
#[test]
fn reused_buffer_decodes_multiple_frame_sizes() {
    let frames = [
        "tests/fixtures/photo_640x480_422.jpg",
        "tests/fixtures/gray_8x8.jpg",
        "tests/fixtures/photo_320x240_444.jpg",
    ];
    let max_size = frames
        .iter()
        .map(|p| {
            let jpeg = std::fs::read(p).expect("fixture");
            output_buffer_size(&jpeg, PixelFormat::Rgb).expect("size")
        })
        .max()
        .unwrap();
    let mut out = vec![0u8; max_size];

    for path in frames {
        let jpeg = std::fs::read(path).expect("fixture");
        let reference = libjpeg_turbo_rs::decompress_to(&jpeg, PixelFormat::Rgb).expect("ref");
        let info = decompress_into(&jpeg, PixelFormat::Rgb, &mut out).expect("into");
        assert_eq!(info.bytes_written, reference.data.len(), "{path}");
        assert_eq!(&out[..info.bytes_written], &reference.data[..], "{path}");
    }
}

/// Issue #354 criterion: `decompress_into` performs no output-sized heap
/// allocation — allocated bytes for a 1080p decode drop by at least the
/// 6.2 MB output size versus `decompress()`.
#[test]
fn into_path_skips_the_output_allocation() {
    let jpeg = std::fs::read("tests/fixtures/photo_1920x1080_420.jpg").expect("fixture");
    let out_size = output_buffer_size(&jpeg, PixelFormat::Rgb).expect("size");
    assert_eq!(out_size, 1920 * 1080 * 3);

    let jpeg_a = jpeg.clone();
    let owned_bytes = measure_bytes(move || {
        let img = libjpeg_turbo_rs::decompress_to(&jpeg_a, PixelFormat::Rgb).expect("decompress");
        assert_eq!(img.data.len(), 1920 * 1080 * 3);
    });

    let mut out = vec![0u8; out_size];
    let into_bytes = measure_bytes(move || {
        let info = decompress_into(&jpeg, PixelFormat::Rgb, &mut out).expect("into");
        assert_eq!(info.bytes_written, 1920 * 1080 * 3);
    });

    eprintln!("owned: {owned_bytes} B, into: {into_bytes} B");
    assert!(
        owned_bytes >= into_bytes + out_size,
        "decompress_into must skip the {out_size}-byte output allocation: \
         owned {owned_bytes} vs into {into_bytes}"
    );
}

/// Codex P2 on #354: the sizing contract must hold for staged paths
/// that ignore scale/crop (12-bit, lossless) — a buffer allocated from
/// `output_buffer_size` must always be accepted, for every option
/// combination.
#[test]
fn advertised_size_is_always_sufficient_across_options() {
    // Lossless ignores scaled decode: pre-fix an 8x8 at 1/2 scale
    // advertised 16 bytes but wrote 64.
    let mut pixels = vec![0u8; 8 * 8];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i * 3) as u8;
    }
    let lossless = libjpeg_turbo_rs::compress_lossless(&pixels, 8, 8, PixelFormat::Grayscale)
        .expect("lossless encode");

    // 12-bit ignores scale/crop the same way (routes through
    // decode_12bit_as_8bit before either applies).
    let pixels12: Vec<i16> = (0..16 * 16).map(|i| ((i * 16) % 4096) as i16).collect();
    let twelve_bit = libjpeg_turbo_rs::precision::compress_12bit(
        &pixels12,
        16,
        16,
        1,
        90,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("12-bit encode");

    let baseline = std::fs::read("tests/fixtures/photo_320x240_420.jpg").expect("fixture");

    type Config = fn(&mut libjpeg_turbo_rs::Decoder);
    let configs: [(&str, Config); 3] = [
        ("plain", |_d| {}),
        ("half-scale", |d| {
            d.set_scale(libjpeg_turbo_rs::ScalingFactor::new(1, 2))
        }),
        ("vcrop", |d| d.set_crop_region(0, 8, 64, 16)),
    ];
    for (name, configure) in configs {
        for (label, jpeg) in [
            ("lossless", &lossless),
            ("12bit", &twelve_bit),
            ("baseline", &baseline),
        ] {
            let mut decoder = libjpeg_turbo_rs::Decoder::new(jpeg).expect("parse");
            configure(&mut decoder);
            let size = decoder.output_buffer_size().expect("size");
            let mut out = vec![0u8; size];
            let info = decoder
                .decode_image_into(&mut out)
                .unwrap_or_else(|e| panic!("{label}/{name}: advertised {size} rejected: {e:?}"));
            assert!(
                info.bytes_written <= size,
                "{label}/{name}: wrote {} into advertised {size}",
                info.bytes_written
            );
        }
    }
}
