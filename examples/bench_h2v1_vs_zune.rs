//! Best-of-N wall-clock comparison vs `zune-jpeg` for the 4:2:2 (H2V1)
//! decode path (issue #350) — same methodology as the #361 gap analysis
//! and `bench_small_vs_zune`: warm-up = N/20, metric = best-of-N.
//!
//! Run with: `cargo run --release --example bench_h2v1_vs_zune`

use std::time::Instant;

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
    let cases: [(&str, &str, usize); 4] = [
        (
            "photo_320x240_422",
            "tests/fixtures/photo_320x240_422.jpg",
            3000,
        ),
        (
            "photo_640x480_422",
            "tests/fixtures/photo_640x480_422.jpg",
            1500,
        ),
        (
            "photo_1920x1080_422",
            "tests/fixtures/photo_1920x1080_422.jpg",
            300,
        ),
        (
            "photo_1920x1080_420",
            "tests/fixtures/photo_1920x1080_420.jpg",
            300,
        ),
    ];

    println!(
        "{:<20} {:>10} {:>10} {:>10}",
        "case", "ours (us)", "zune (us)", "ours/zune"
    );
    for (name, path, n) in &cases {
        let jpeg = std::fs::read(path).unwrap_or_else(|_| panic!("{path} fixture required"));
        let ours = best_of(*n, || {
            let img = libjpeg_turbo_rs::decompress(std::hint::black_box(&jpeg)).unwrap();
            std::hint::black_box(&img.data);
        });
        let zune = best_of(*n, || {
            let cursor = std::io::Cursor::new(std::hint::black_box(&jpeg));
            let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
            let pixels = decoder.decode().unwrap();
            std::hint::black_box(&pixels);
        });
        println!(
            "{:<20} {:>10.2} {:>10.2} {:>10.2}",
            name,
            ours,
            zune,
            ours / zune
        );
    }
}
