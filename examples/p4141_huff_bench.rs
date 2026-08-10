//! P4-141 criterion 5: decode throughput with safe vs unchecked zigzag indexing.
use std::time::Instant;
fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        "tests/fixtures/real_world/derived_7680x4320_8k_progressive.jpg".to_string()
    });
    let data = std::fs::read(&path).expect("fixture");
    for _ in 0..3 {
        let _ = libjpeg_turbo_rs::decompress(&data).unwrap();
    }
    let mut best = f64::MAX;
    for _ in 0..9 {
        let t = Instant::now();
        let img = libjpeg_turbo_rs::decompress(&data).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(&img);
        if ms < best {
            best = ms;
        }
    }
    println!("DECODE_BEST_MS {best:.2}");
}
