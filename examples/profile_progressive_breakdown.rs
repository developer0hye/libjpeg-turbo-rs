/// Instrumented progressive decode profiling.
/// Measures each phase: scan decode, IDCT, color, upsample.
/// Usage: cargo run --release --example profile_progressive_breakdown
fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "tests/fixtures/photo_1920x1080_420_prog.jpg".to_string());
    let jpeg_data = std::fs::read(&path).expect("missing test fixture");

    // Compare with baseline (non-progressive) version of same image
    let baseline_path = path.replace("_prog.jpg", ".jpg");
    let baseline_data = std::fs::read(&baseline_path).ok();

    let iters = 200u64;

    // Warmup
    for _ in 0..50 {
        let _ = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
    }

    // Time progressive
    let start = std::time::Instant::now();
    for _ in 0..iters {
        let img = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        std::hint::black_box(&img.data);
    }
    let prog_us = start.elapsed().as_micros() as f64 / iters as f64;

    // Time baseline if available
    if let Some(ref bl_data) = baseline_data {
        for _ in 0..50 {
            let _ = libjpeg_turbo_rs::decompress(bl_data).unwrap();
        }
        let start = std::time::Instant::now();
        for _ in 0..iters {
            let img = libjpeg_turbo_rs::decompress(bl_data).unwrap();
            std::hint::black_box(&img.data);
        }
        let bl_us = start.elapsed().as_micros() as f64 / iters as f64;
        eprintln!("Baseline:     {:.1} µs", bl_us);
        eprintln!("Progressive:  {:.1} µs", prog_us);
        eprintln!(
            "Overhead:     {:.1} µs ({:.1}x)",
            prog_us - bl_us,
            prog_us / bl_us
        );
    } else {
        eprintln!("Progressive:  {:.1} µs  ({})", prog_us, path);
    }
}
