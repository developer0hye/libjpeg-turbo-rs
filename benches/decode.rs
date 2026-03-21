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

// --- Full decode benchmarks: various resolutions, subsampling, content ---

struct BenchCase {
    name: &'static str,
    path: &'static str,
}

const BENCH_CASES: &[BenchCase] = &[
    // Resolution scaling (all 4:2:0, photo-like content)
    BenchCase {
        name: "photo_64x64_420",
        path: "tests/fixtures/photo_64x64_420.jpg",
    },
    BenchCase {
        name: "photo_320x240_420",
        path: "tests/fixtures/photo_320x240_420.jpg",
    },
    BenchCase {
        name: "decode_640x480",
        path: "tests/fixtures/gradient_640x480.jpg",
    },
    BenchCase {
        name: "photo_1280x720_420",
        path: "tests/fixtures/photo_1280x720_420.jpg",
    },
    BenchCase {
        name: "photo_1920x1080_420",
        path: "tests/fixtures/photo_1920x1080_420.jpg",
    },
    BenchCase {
        name: "photo_2560x1440_420",
        path: "tests/fixtures/photo_2560x1440_420.jpg",
    },
    BenchCase {
        name: "photo_3840x2160_420",
        path: "tests/fixtures/photo_3840x2160_420.jpg",
    },
    // Subsampling modes
    BenchCase {
        name: "photo_320x240_444",
        path: "tests/fixtures/photo_320x240_444.jpg",
    },
    BenchCase {
        name: "photo_320x240_422",
        path: "tests/fixtures/photo_320x240_422.jpg",
    },
    BenchCase {
        name: "photo_640x480_444",
        path: "tests/fixtures/photo_640x480_444.jpg",
    },
    BenchCase {
        name: "photo_640x480_422",
        path: "tests/fixtures/photo_640x480_422.jpg",
    },
    BenchCase {
        name: "photo_1920x1080_444",
        path: "tests/fixtures/photo_1920x1080_444.jpg",
    },
    BenchCase {
        name: "photo_1920x1080_422",
        path: "tests/fixtures/photo_1920x1080_422.jpg",
    },
    // Content types (640x480)
    BenchCase {
        name: "graphic_640x480_420",
        path: "tests/fixtures/graphic_640x480_420.jpg",
    },
    BenchCase {
        name: "checker_640x480_420",
        path: "tests/fixtures/checker_640x480_420.jpg",
    },
    // Large graphic
    BenchCase {
        name: "graphic_1920x1080_420",
        path: "tests/fixtures/graphic_1920x1080_420.jpg",
    },
    // Restart markers
    BenchCase {
        name: "photo_640x480_420_rst",
        path: "tests/fixtures/photo_640x480_420_rst.jpg",
    },
];

fn bench_full_decode_matrix(c: &mut Criterion) {
    for case in BENCH_CASES {
        let jpeg_data = std::fs::read(case.path)
            .unwrap_or_else(|_| panic!("{} fixture required for benchmarks", case.path));

        c.bench_function(case.name, |b| {
            b.iter(|| {
                let image = libjpeg_turbo_rs::decompress(black_box(&jpeg_data)).unwrap();
                black_box(&image.data);
            })
        });
    }
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
    bench_full_decode_matrix,
    bench_full_decode_reuse_header,
);
criterion_main!(benches);
