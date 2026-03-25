/// Profiling target for encoding.
/// Usage: samply record cargo run --release --example profile_encode
fn main() {
    use libjpeg_turbo_rs::{PixelFormat, Subsampling};

    let jpeg_data = std::fs::read("tests/fixtures/photo_1920x1080_420.jpg").unwrap();
    let image = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();

    // Run enough iterations for samply to get good samples
    for _ in 0..500 {
        let result = libjpeg_turbo_rs::compress(
            &image.data,
            image.width,
            image.height,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
        )
        .unwrap();
        std::hint::black_box(&result);
    }
}
