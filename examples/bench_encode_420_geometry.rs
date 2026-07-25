//! Measures 4:2:0 encode throughput either side of the dummy-block boundary.
//!
//! Issue #314's fix narrows the x86_64 AVX2 4:2:0 row fast path so it no longer
//! runs when the last MCU column needs dummy blocks (`ceil(width/8)` odd).
//! Those widths now take the generic per-MCU path, so this quantifies what that
//! costs, and how both sides compare against C.
//!
//! Widths are paired so each "odd" (fast path now disabled) case sits next to a
//! near-identical "even" (fast path still active) control.

use libjpeg_turbo_rs::encode::pipeline;
use libjpeg_turbo_rs::{DctMethod, PixelFormat, Subsampling};
use std::time::Instant;

fn synthetic_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let offset: usize = (y * width + x) * 3;
            pixels[offset] = ((x * 255 / width.max(1)) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 1] = ((y * 255 / height.max(1)) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 2] =
                (((x + y) * 255 / (width + height)) as i32 - noise).clamp(0, 255) as u8;
        }
    }
    pixels
}

fn main() {
    // (width, height): pairs differing by 8px so image area is nearly equal but
    // `ceil(width/8)` parity flips.
    let cases: &[(usize, usize)] = &[
        (1000, 750),  // ceil(1000/8)=125 odd  -> generic path
        (1008, 750),  // ceil(1008/8)=126 even -> AVX2 fast path
        (1920, 1080), // ceil(1920/8)=240 even -> AVX2 fast path
        (1928, 1080), // ceil(1928/8)=241 odd  -> generic path
        (3840, 2160), // ceil(3840/8)=480 even -> AVX2 fast path
        (3848, 2160), // ceil(3848/8)=481 odd  -> generic path
    ];

    println!(
        "{:<12} {:>10} {:>12} {:>12} {:>10}",
        "size", "blocks", "path", "median_ms", "MP/s"
    );

    for &(width, height) in cases {
        let pixels: Vec<u8> = synthetic_pixels(width, height);
        let blocks_across: usize = width.div_ceil(8);
        let path: &str = if blocks_across % 2 == 0 {
            "avx2-fast"
        } else {
            "generic"
        };

        // Warm up so the first timed run is not paying for page faults.
        for _ in 0..3 {
            let _ = pipeline::compress(
                &pixels,
                width,
                height,
                PixelFormat::Rgb,
                75,
                Subsampling::S420,
                DctMethod::IsLow,
            )
            .expect("encode");
        }

        let runs: usize = 15;
        let mut timings_ms: Vec<f64> = Vec::with_capacity(runs);
        for _ in 0..runs {
            let started = Instant::now();
            let encoded = pipeline::compress(
                &pixels,
                width,
                height,
                PixelFormat::Rgb,
                75,
                Subsampling::S420,
                DctMethod::IsLow,
            )
            .expect("encode");
            timings_ms.push(started.elapsed().as_secs_f64() * 1000.0);
            std::hint::black_box(encoded);
        }
        timings_ms.sort_by(|a, b| a.partial_cmp(b).expect("no NaN timings"));
        let median_ms: f64 = timings_ms[runs / 2];
        let megapixels: f64 = (width * height) as f64 / 1e6;

        println!(
            "{:<12} {:>10} {:>12} {:>12.3} {:>10.1}",
            format!("{width}x{height}"),
            blocks_across,
            path,
            median_ms,
            megapixels / (median_ms / 1000.0)
        );
    }
}
