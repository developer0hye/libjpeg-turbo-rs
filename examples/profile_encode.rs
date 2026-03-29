fn main() {
    let jpeg = std::fs::read("tests/fixtures/photo_1920x1080_420.jpg").unwrap();
    let img = libjpeg_turbo_rs::decompress(&jpeg).unwrap();
    // Warmup
    for _ in 0..5 {
        let _ = libjpeg_turbo_rs::compress(&img.data, img.width, img.height, libjpeg_turbo_rs::PixelFormat::Rgb, 90, libjpeg_turbo_rs::Subsampling::S420).unwrap();
    }
    // Profile
    for _ in 0..100 {
        let out = libjpeg_turbo_rs::compress(&img.data, img.width, img.height, libjpeg_turbo_rs::PixelFormat::Rgb, 90, libjpeg_turbo_rs::Subsampling::S420).unwrap();
        std::hint::black_box(&out);
    }
}
