use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::simd;

fn bench_idct_8x8(c: &mut Criterion) {
    let routines = simd::detect();

    // Typical JPEG block: DC + some AC coefficients (zigzag order)
    let mut coeffs = [0i16; 64];
    coeffs[0] = 1024;
    coeffs[1] = -50;
    coeffs[2] = 30;
    coeffs[3] = -10;
    coeffs[4] = 5;
    coeffs[5] = -3;

    let quant = [16u16; 64];
    let mut output = [0u8; 64];

    c.bench_function("idct_8x8", |b| {
        b.iter(|| {
            (routines.idct_islow)(black_box(&coeffs), black_box(&quant), &mut output);
            black_box(&output);
        })
    });
}

fn bench_ycbcr_to_rgb_row(c: &mut Criterion) {
    let routines = simd::detect();
    let width = 640;
    let y: Vec<u8> = (0..width).map(|i| (i % 256) as u8).collect();
    let cb: Vec<u8> = (0..width).map(|i| ((i + 64) % 256) as u8).collect();
    let cr: Vec<u8> = (0..width).map(|i| ((i + 128) % 256) as u8).collect();
    let mut rgb = vec![0u8; width * 3];

    c.bench_function("ycbcr_to_rgb_row_640", |b| {
        b.iter(|| {
            (routines.ycbcr_to_rgb_row)(
                black_box(&y),
                black_box(&cb),
                black_box(&cr),
                &mut rgb,
                width,
            );
            black_box(&rgb);
        })
    });
}

fn bench_fancy_upsample_h2v1(c: &mut Criterion) {
    let routines = simd::detect();
    let in_width = 320;
    let input: Vec<u8> = (0..in_width).map(|i| (i % 256) as u8).collect();
    let mut output = vec![0u8; in_width * 2];

    c.bench_function("fancy_h2v1_320", |b| {
        b.iter(|| {
            (routines.fancy_upsample_h2v1)(black_box(&input), in_width, &mut output);
            black_box(&output);
        })
    });
}

fn bench_full_decode(c: &mut Criterion) {
    let jpeg_data = std::fs::read("tests/fixtures/gradient_640x480.jpg")
        .expect("gradient_640x480.jpg fixture required for benchmarks");

    c.bench_function("decode_640x480", |b| {
        b.iter(|| {
            let image = libjpeg_turbo_rs::decompress(black_box(&jpeg_data)).unwrap();
            black_box(&image.data);
        })
    });
}

fn bench_decoder_new(c: &mut Criterion) {
    let jpeg_data = std::fs::read("tests/fixtures/gradient_640x480.jpg")
        .expect("gradient_640x480.jpg fixture required for benchmarks");

    c.bench_function("decoder_new_640x480", |b| {
        b.iter(|| {
            let decoder = StreamingDecoder::new(black_box(&jpeg_data)).unwrap();
            black_box(decoder.header());
        })
    });
}

fn bench_full_decode_reuse_header(c: &mut Criterion) {
    let jpeg_data = std::fs::read("tests/fixtures/gradient_640x480.jpg")
        .expect("gradient_640x480.jpg fixture required for benchmarks");
    let decoder = StreamingDecoder::new(&jpeg_data).expect("streaming decoder fixture");

    c.bench_function("decode_640x480_reuse_header", |b| {
        b.iter(|| {
            let image = decoder.decode().unwrap();
            black_box(&image.data);
        })
    });
}

criterion_group!(
    benches,
    bench_idct_8x8,
    bench_ycbcr_to_rgb_row,
    bench_fancy_upsample_h2v1,
    bench_decoder_new,
    bench_full_decode,
    bench_full_decode_reuse_header,
);
criterion_main!(benches);
