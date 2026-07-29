use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

struct CompareCase {
    name: &'static str,
    path: &'static str,
}

// Every subsampling × {baseline, progressive} × size tier must be
// represented: the pre-#360 matrix covered only 4:2:0/4:4:4 baseline,
// which is exactly why the 4:2:2, small-image, and progressive losses
// to zune-jpeg went unnoticed (issues #350–#353). The wide sweep with
// allocation metrics lives in `examples/bench_zune_matrix.rs`; this
// criterion set is the statistically-rigorous subset.
const COMPARE_CASES: &[CompareCase] = &[
    // tiny — fixed cost dominates (#351)
    CompareCase {
        name: "gray_8x8",
        path: "tests/fixtures/gray_8x8.jpg",
    },
    CompareCase {
        name: "16x16_420",
        path: "tests/fixtures/blue_16x16_420.jpg",
    },
    CompareCase {
        name: "64x64_420",
        path: "tests/fixtures/photo_64x64_420.jpg",
    },
    // small / medium, all subsamplings (#350)
    CompareCase {
        name: "320x240_422",
        path: "tests/fixtures/photo_320x240_422.jpg",
    },
    CompareCase {
        name: "320x240_444",
        path: "tests/fixtures/photo_320x240_444.jpg",
    },
    CompareCase {
        name: "640x480",
        path: "tests/fixtures/gradient_640x480.jpg",
    },
    CompareCase {
        name: "640x480_422",
        path: "tests/fixtures/photo_640x480_422.jpg",
    },
    CompareCase {
        name: "640x480_444",
        path: "tests/fixtures/photo_640x480_444.jpg",
    },
    CompareCase {
        name: "640x480_420_rst",
        path: "tests/fixtures/photo_640x480_420_rst.jpg",
    },
    CompareCase {
        name: "graphic_640x480",
        path: "tests/fixtures/graphic_640x480_420.jpg",
    },
    // HD and up
    CompareCase {
        name: "1280x720",
        path: "tests/fixtures/photo_1280x720_420.jpg",
    },
    CompareCase {
        name: "1920x1080",
        path: "tests/fixtures/photo_1920x1080_420.jpg",
    },
    CompareCase {
        name: "1920x1080_422",
        path: "tests/fixtures/photo_1920x1080_422.jpg",
    },
    CompareCase {
        name: "1920x1080_444",
        path: "tests/fixtures/photo_1920x1080_444.jpg",
    },
    CompareCase {
        name: "2560x1440",
        path: "tests/fixtures/photo_2560x1440_420.jpg",
    },
    CompareCase {
        name: "3840x2160",
        path: "tests/fixtures/photo_3840x2160_420.jpg",
    },
    // progressive (#352)
    CompareCase {
        name: "320x240_420_prog",
        path: "tests/fixtures/photo_320x240_420_prog.jpg",
    },
    CompareCase {
        name: "1920x1080_420_prog",
        path: "tests/fixtures/photo_1920x1080_420_prog.jpg",
    },
    CompareCase {
        name: "3840x2160_420_prog",
        path: "tests/fixtures/photo_3840x2160_420_prog.jpg",
    },
    // 8K real-world (#352's superlinear-scaling witness)
    CompareCase {
        name: "8k_420",
        path: "tests/fixtures/real_world/derived_7680x4320_8k_420_q75.jpg",
    },
    CompareCase {
        name: "8k_progressive",
        path: "tests/fixtures/real_world/derived_7680x4320_8k_progressive.jpg",
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
