//! Best-of-N wall-clock comparison vs `zune-jpeg` on small-image fixed
//! cost (issue #351) — mirrors the methodology of the 2026-07-26 gap
//! analysis (issue #361): warm-up = N/20, metric = best-of-N per decode.
//!
//! Run with: `cargo run --release --example bench_small_vs_zune`

use std::time::Instant;

fn gray_8x8() -> Vec<u8> {
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
    .expect("encode")
}

fn blue_16x16_420() -> Vec<u8> {
    let pixels = vec![[30u8, 60, 200]; 16 * 16].concat();
    libjpeg_turbo_rs::compress(
        &pixels,
        16,
        16,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        90,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("encode")
}

fn photo_64x64_420() -> Vec<u8> {
    let mut pixels = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            pixels.push((x * 4) as u8);
            pixels.push((y * 4) as u8);
            pixels.push((((x * 7) ^ (y * 13)) & 0xff) as u8);
        }
    }
    libjpeg_turbo_rs::compress(
        &pixels,
        64,
        64,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        85,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("encode")
}

fn best_of<F: FnMut()>(n: usize, mut f: F) -> f64 {
    for _ in 0..n / 20 {
        f();
    }
    let mut best = f64::INFINITY;
    for _ in 0..n {
        let t = Instant::now();
        f();
        let dt = t.elapsed().as_secs_f64() * 1e6;
        if dt < best {
            best = dt;
        }
    }
    best
}

fn main() {
    let cases: Vec<(&str, Vec<u8>, usize)> = vec![
        ("gray_8x8", gray_8x8(), 20000),
        ("blue_16x16_420", blue_16x16_420(), 20000),
        ("photo_64x64_420", photo_64x64_420(), 10000),
    ];

    println!(
        "{:<18} {:>10} {:>10} {:>10}",
        "case", "ours (us)", "zune (us)", "ours/zune"
    );
    for (name, jpeg, n) in &cases {
        let ours = best_of(*n, || {
            let img = libjpeg_turbo_rs::decompress(std::hint::black_box(jpeg)).unwrap();
            std::hint::black_box(&img.data);
        });
        let zune = best_of(*n, || {
            let cursor = std::io::Cursor::new(std::hint::black_box(jpeg));
            let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
            let pixels = decoder.decode().unwrap();
            std::hint::black_box(&pixels);
        });
        println!(
            "{:<18} {:>10.2} {:>10.2} {:>10.2}",
            name,
            ours,
            zune,
            ours / zune
        );
    }
}
