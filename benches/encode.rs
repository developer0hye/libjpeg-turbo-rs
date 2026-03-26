use criterion::{black_box, criterion_group, criterion_main, Criterion};
use libjpeg_turbo_rs::simd::{self, QuantDivisors};

fn bench_fdct_quantize_8x8(c: &mut Criterion) {
    let enc = simd::detect_encoder();

    let mut input = [0i16; 64];
    for i in 0..64 {
        input[i] = (i as i16 * 3) - 96;
    }
    let divisors = [128u16; 64]; // pre-scaled (16 * 8)
    let mut reciprocals = [0u16; 64];
    for i in 0..64 {
        let d = divisors[i] as u32;
        reciprocals[i] = (((1u32 << 16) + d - 1) / d) as u16;
    }
    let quant = QuantDivisors {
        divisors,
        reciprocals,
    };
    let mut output = [0i16; 64];

    c.bench_function("fdct_quantize_8x8", |b| {
        b.iter(|| {
            (enc.fdct_quantize)(black_box(&input), black_box(&quant), &mut output);
            black_box(&output);
        })
    });
}

fn bench_rgb_to_ycbcr_row(c: &mut Criterion) {
    let enc = simd::detect_encoder();

    for width in [320, 640, 1920] {
        let rgb: Vec<u8> = (0..width * 3).map(|i| (i % 256) as u8).collect();
        let mut y = vec![0u8; width];
        let mut cb = vec![0u8; width];
        let mut cr = vec![0u8; width];

        c.bench_function(&format!("rgb_to_ycbcr_row_{width}"), |b| {
            b.iter(|| {
                (enc.rgb_to_ycbcr_row)(black_box(&rgb), &mut y, &mut cb, &mut cr, width);
                black_box((&y, &cb, &cr));
            })
        });
    }
}

fn bench_full_encode(c: &mut Criterion) {
    use libjpeg_turbo_rs::{PixelFormat, Subsampling};

    struct EncodeCase {
        name: &'static str,
        fixture: &'static str,
        subsampling: Subsampling,
    }

    let cases = [
        EncodeCase {
            name: "encode_320x240_420",
            fixture: "tests/fixtures/photo_320x240_420.jpg",
            subsampling: Subsampling::S420,
        },
        EncodeCase {
            name: "encode_320x240_422",
            fixture: "tests/fixtures/photo_320x240_422.jpg",
            subsampling: Subsampling::S422,
        },
        EncodeCase {
            name: "encode_320x240_444",
            fixture: "tests/fixtures/photo_320x240_444.jpg",
            subsampling: Subsampling::S444,
        },
        EncodeCase {
            name: "encode_640x480_422",
            fixture: "tests/fixtures/photo_640x480_422.jpg",
            subsampling: Subsampling::S422,
        },
        EncodeCase {
            name: "encode_640x480_444",
            fixture: "tests/fixtures/photo_640x480_444.jpg",
            subsampling: Subsampling::S444,
        },
        EncodeCase {
            name: "encode_1920x1080_420",
            fixture: "tests/fixtures/photo_1920x1080_420.jpg",
            subsampling: Subsampling::S420,
        },
        EncodeCase {
            name: "encode_1920x1080_422",
            fixture: "tests/fixtures/photo_1920x1080_422.jpg",
            subsampling: Subsampling::S422,
        },
        EncodeCase {
            name: "encode_1920x1080_444",
            fixture: "tests/fixtures/photo_1920x1080_444.jpg",
            subsampling: Subsampling::S444,
        },
    ];

    for case in &cases {
        let jpeg_data = match std::fs::read(case.fixture) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let image = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();

        c.bench_function(case.name, |b| {
            b.iter(|| {
                let result = libjpeg_turbo_rs::compress(
                    black_box(&image.data),
                    image.width,
                    image.height,
                    PixelFormat::Rgb,
                    75,
                    case.subsampling,
                )
                .unwrap();
                black_box(&result);
            })
        });
    }
}

criterion_group!(
    benches,
    bench_fdct_quantize_8x8,
    bench_rgb_to_ycbcr_row,
    bench_full_encode,
);
criterion_main!(benches);
