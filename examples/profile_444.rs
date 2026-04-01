fn main() {
    let jpeg_data =
        std::fs::read("tests/fixtures/photo_640x480_444.jpg").expect("missing test fixture");
    for _ in 0..10 {
        let _ = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
    }
    for _ in 0..2000 {
        let img = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();
        std::hint::black_box(&img.data);
    }
}
