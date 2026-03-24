/// Standalone benchmark matrix matching benches/c_baseline.c
/// Usage: cargo run --release --example bench_matrix
fn main() {
    let cases: Vec<(&str, u32)> = vec![
        // Resolution scaling (4:2:0, photo-like)
        ("tests/fixtures/photo_64x64_420.jpg", 20000),
        ("tests/fixtures/photo_320x240_420.jpg", 5000),
        ("tests/fixtures/gradient_640x480.jpg", 5000),
        ("tests/fixtures/photo_1280x720_420.jpg", 2000),
        ("tests/fixtures/photo_1920x1080_420.jpg", 500),
        ("tests/fixtures/photo_2560x1440_420.jpg", 200),
        ("tests/fixtures/photo_3840x2160_420.jpg", 100),
        // Subsampling modes
        ("tests/fixtures/photo_320x240_444.jpg", 5000),
        ("tests/fixtures/photo_320x240_422.jpg", 5000),
        ("tests/fixtures/photo_640x480_444.jpg", 5000),
        ("tests/fixtures/photo_640x480_422.jpg", 5000),
        ("tests/fixtures/photo_1920x1080_444.jpg", 500),
        ("tests/fixtures/photo_1920x1080_422.jpg", 500),
        // Content types
        ("tests/fixtures/graphic_640x480_420.jpg", 5000),
        ("tests/fixtures/checker_640x480_420.jpg", 5000),
        ("tests/fixtures/graphic_1920x1080_420.jpg", 500),
        // Restart markers
        ("tests/fixtures/photo_640x480_420_rst.jpg", 5000),
    ];

    println!(
        "{:<45} {:>10} {:>12} {:>8}",
        "File", "Size", "Time", "Iters"
    );
    println!("{}", "-".repeat(80));

    for (path, iters) in &cases {
        let jpeg_data = match std::fs::read(path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: {} (not found)", path);
                continue;
            }
        };

        // Warmup
        for _ in 0..100 {
            let _ = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        }

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..*iters {
            let img = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
            std::hint::black_box(&img.data);
        }
        let elapsed = start.elapsed();
        let us: f64 = elapsed.as_nanos() as f64 / *iters as f64 / 1000.0;

        // Get image dimensions from the result
        let img = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        println!(
            "{:<45} {:>4}x{:<4} {:>10.1} µs  ({} iters)",
            path, img.width, img.height, us, iters
        );
    }
}
