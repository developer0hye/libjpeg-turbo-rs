/// Tight progressive encode loop for profiling with samply.
/// Usage: samply record target/release/examples/profile_progressive_encode
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/photo_1920x1080_420.jpg".to_string());
    let jpeg_data = std::fs::read(&path).expect("missing test fixture");
    let image = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();

    // Warmup
    for _ in 0..50 {
        let _ = libjpeg_turbo_rs::compress_progressive(
            &image.data,
            image.width,
            image.height,
            libjpeg_turbo_rs::PixelFormat::Rgb,
            75,
            libjpeg_turbo_rs::Subsampling::S420,
        )
        .unwrap();
    }

    // Profile loop
    let iters = 500u64;
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let result = libjpeg_turbo_rs::compress_progressive(
            &image.data,
            image.width,
            image.height,
            libjpeg_turbo_rs::PixelFormat::Rgb,
            75,
            libjpeg_turbo_rs::Subsampling::S420,
        )
        .unwrap();
        std::hint::black_box(&result);
    }
    let elapsed = start.elapsed();
    eprintln!(
        "{} iters in {:?} => {:.1} us/iter  (progressive encode, {})",
        iters,
        elapsed,
        elapsed.as_micros() as f64 / iters as f64,
        path
    );
}
