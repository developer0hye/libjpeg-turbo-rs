fn main() {
    let jpeg_data =
        std::fs::read("tests/fixtures/photo_1920x1080_420.jpg").expect("missing test fixture");
    // Warmup
    for _ in 0..10 {
        let _ = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
    }
    // Profile loop
    for _ in 0..500 {
        let img = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        std::hint::black_box(&img.data);
    }
}
