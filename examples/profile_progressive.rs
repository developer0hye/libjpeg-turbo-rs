/// Tight progressive decode loop for profiling with samply.
/// Usage: samply record target/release/examples/profile_progressive
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/photo_1920x1080_420_prog.jpg".to_string());
    let jpeg_data = std::fs::read(&path).expect("missing test fixture");

    // Warmup
    for _ in 0..100 {
        let _ = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
    }

    // Profile loop
    let iters = if jpeg_data.len() > 2_000_000 {
        200u64
    } else if jpeg_data.len() > 500_000 {
        500u64
    } else {
        2000u64
    };
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let img = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        std::hint::black_box(&img.data);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{} iters in {:?} => {:.1} us/iter  ({})",
        iters,
        elapsed,
        elapsed.as_micros() as f64 / iters as f64,
        path
    );
}
