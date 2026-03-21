use criterion::{black_box, criterion_group, criterion_main, Criterion};

struct CompareCase {
    name: &'static str,
    path: &'static str,
}

const COMPARE_CASES: &[CompareCase] = &[
    CompareCase {
        name: "640x480",
        path: "tests/fixtures/gradient_640x480.jpg",
    },
    CompareCase {
        name: "1280x720",
        path: "tests/fixtures/photo_1280x720_420.jpg",
    },
    CompareCase {
        name: "1920x1080",
        path: "tests/fixtures/photo_1920x1080_420.jpg",
    },
    CompareCase {
        name: "2560x1440",
        path: "tests/fixtures/photo_2560x1440_420.jpg",
    },
    CompareCase {
        name: "3840x2160",
        path: "tests/fixtures/photo_3840x2160_420.jpg",
    },
    CompareCase {
        name: "320x240_444",
        path: "tests/fixtures/photo_320x240_444.jpg",
    },
    CompareCase {
        name: "graphic_640x480",
        path: "tests/fixtures/graphic_640x480_420.jpg",
    },
];

fn bench_ours_matrix(c: &mut Criterion) {
    for case in COMPARE_CASES {
        let jpeg_data =
            std::fs::read(case.path).unwrap_or_else(|_| panic!("{} fixture required", case.path));

        c.bench_function(&format!("ours_{}", case.name), |b| {
            b.iter(|| {
                let image = libjpeg_turbo_rs::decompress(black_box(&jpeg_data)).unwrap();
                black_box(&image.data);
            })
        });
    }
}

fn bench_zune_matrix(c: &mut Criterion) {
    for case in COMPARE_CASES {
        let jpeg_data =
            std::fs::read(case.path).unwrap_or_else(|_| panic!("{} fixture required", case.path));

        c.bench_function(&format!("zune_{}", case.name), |b| {
            b.iter(|| {
                let cursor = std::io::Cursor::new(black_box(&jpeg_data));
                let mut decoder = zune_jpeg::JpegDecoder::new(cursor);
                let pixels = decoder.decode().unwrap();
                black_box(&pixels);
            })
        });
    }
}

criterion_group!(benches, bench_ours_matrix, bench_zune_matrix);
criterion_main!(benches);
