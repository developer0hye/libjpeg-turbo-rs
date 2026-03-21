use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_ours(c: &mut Criterion) {
    let jpeg_data = std::fs::read("tests/fixtures/gradient_640x480.jpg")
        .expect("gradient_640x480.jpg fixture required");

    c.bench_function("ours_640x480", |b| {
        b.iter(|| {
            let image = libjpeg_turbo_rs::decompress(black_box(&jpeg_data)).unwrap();
            black_box(&image.data);
        })
    });
}

fn bench_zune(c: &mut Criterion) {
    let jpeg_data = std::fs::read("tests/fixtures/gradient_640x480.jpg")
        .expect("gradient_640x480.jpg fixture required");

    c.bench_function("zune_640x480", |b| {
        b.iter(|| {
            let cursor = std::io::Cursor::new(black_box(&jpeg_data));
            let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
            let pixels = decoder.decode().unwrap();
            black_box(&pixels);
        })
    });
}

criterion_group!(benches, bench_ours, bench_zune);
criterion_main!(benches);
