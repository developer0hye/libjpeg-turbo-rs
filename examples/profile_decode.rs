/// Tight decode loop for profiling with samply.
/// Usage: samply record target/release/examples/profile_decode
fn main() {
    let jpeg_data =
        std::fs::read("tests/fixtures/gradient_640x480.jpg").expect("missing test fixture");

    // Warmup
    for _ in 0..100 {
        let _ = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
    }

    // Profile loop
    for _ in 0..5000 {
        let img = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        std::hint::black_box(&img.data);
    }
}
