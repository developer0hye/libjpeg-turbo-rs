//! P4-134 (issue #465) riscv64 measurement probe. Mirrors the two
//! `tjbench` operations upstream's 3.2.0 RVV kernels accelerate — baseline
//! JPEG → RGB decompression and RGB → baseline 4:2:0 q95 compression — so the
//! Rust and C numbers describe the same work. Run on the riscv64 target
//! (under `qemu-riscv64` or on RVV hardware) against
//! `tjbench <fixture>.jpg -rgb -nowrite` (decompress-only mode, which decodes
//! the fixture itself — `tjbench <ppm> 95 -subsamp 420` would decode its own
//! q95 re-encode instead) and `tjbench <ppm> 95 -subsamp 420 -rgb -nowrite`
//! for the compress leg.
//!
//! What matches `tjbench` and what does not, so the ratios are read right:
//! the decode destination is allocated once outside the timed region, as
//! `tjbench` does; the encode still allocates its output per iteration
//! (`tjbench` pre-allocates with `TJPARAM_NOREALLOC`), a bias against us
//! with no public-API remedy; and `tjbench` reports the *mean* over its
//! benchtime window after a warmup, where this probe reports the *median*
//! of `runs` after one discarded warmup iteration, which drops the slow
//! tail C's mean keeps. Absolute times under emulation are meaningless;
//! ratios within one run are the data.
//!
//! Usage: `bench_p4134_riscv64 [runs]` (default 7; even counts return the
//! upper middle sample).

use std::time::Instant;

use libjpeg_turbo_rs::PixelFormat;
use libjpeg_turbo_rs::Subsampling;

/// Median of `runs` timed iterations after one discarded warmup iteration
/// (QEMU translates code on first execution, so a cold first run is not a
/// measurement of the kernels). Nanosecond samples: on RVV hardware a
/// 640×480 decode is around a millisecond, where microsecond truncation
/// would already cost precision.
fn median_ns<F: FnMut()>(mut measured: F, runs: usize) -> u128 {
    measured();
    let mut samples: Vec<u128> = Vec::with_capacity(runs);
    for _ in 0..runs {
        let started = Instant::now();
        measured();
        samples.push(started.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples[runs / 2]
}

/// `tjbench -quiet` reports megapixels/second; pixels per microsecond is
/// numerically the same unit.
fn megapixels_per_second(width: usize, height: usize, nanos: u128) -> f64 {
    (width * height) as f64 * 1000.0 / nanos as f64
}

fn main() {
    // Zero runs would index an empty sample vector; treat it as the default.
    let runs: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse::<usize>().ok())
        .filter(|&count| count > 0)
        .unwrap_or(7);
    let fixtures: [(&[u8], &str); 2] = [
        (
            include_bytes!("../tests/fixtures/photo_640x480_420.jpg"),
            "photo_640x480_420",
        ),
        (
            include_bytes!("../tests/fixtures/photo_1920x1080_420.jpg"),
            "photo_1920x1080_420",
        ),
    ];

    println!("operation\tfixture\tmedian_us\tmpixels_per_s\truns");
    for (fixture_jpeg, name) in fixtures {
        let source_rgb = libjpeg_turbo_rs::decompress_to(fixture_jpeg, PixelFormat::Rgb)
            .expect("fixture decodes");
        let (width, height) = (source_rgb.width, source_rgb.height);

        let output_len: usize =
            libjpeg_turbo_rs::output_buffer_size(fixture_jpeg, PixelFormat::Rgb)
                .expect("output size");
        let mut rgb_out: Vec<u8> = vec![0u8; output_len];
        let mut sink: usize = 0;
        let decode_ns: u128 = median_ns(
            || {
                let info =
                    libjpeg_turbo_rs::decompress_into(fixture_jpeg, PixelFormat::Rgb, &mut rgb_out)
                        .expect("decode");
                sink ^= info.bytes_written;
            },
            runs,
        );
        println!(
            "decode_rgb\t{name}\t{}\t{:.3}\t{runs}",
            decode_ns / 1000,
            megapixels_per_second(width, height, decode_ns)
        );

        let encode_ns: u128 = median_ns(
            || {
                let encoded: Vec<u8> = libjpeg_turbo_rs::compress(
                    &source_rgb.data,
                    width,
                    height,
                    PixelFormat::Rgb,
                    95,
                    Subsampling::S420,
                )
                .expect("encode");
                sink ^= encoded.len();
            },
            runs,
        );
        println!(
            "encode_420_q95\t{name}\t{}\t{:.3}\t{runs}",
            encode_ns / 1000,
            megapixels_per_second(width, height, encode_ns)
        );
        std::hint::black_box(sink);
    }
}
