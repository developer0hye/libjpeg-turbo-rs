/// Rust progressive encoding benchmark matrix.
/// Matches the C bench_c_progressive_encode_matrix format for side-by-side comparison.
///
/// Usage: cargo run --release --example bench_rust_progressive_encode_matrix
use libjpeg_turbo_rs::{PixelFormat, Subsampling};

struct EncodeCase {
    fixture: &'static str,
    subsampling: Subsampling,
    sub_str: &'static str,
    iters: u64,
}

fn main() {
    let cases = [
        // Resolution scaling (4:2:0)
        EncodeCase {
            fixture: "tests/fixtures/photo_64x64_420.jpg",
            subsampling: Subsampling::S420,
            sub_str: "420",
            iters: 20000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_320x240_420.jpg",
            subsampling: Subsampling::S420,
            sub_str: "420",
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_640x480_422.jpg",
            subsampling: Subsampling::S420,
            sub_str: "420",
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_1280x720_420.jpg",
            subsampling: Subsampling::S420,
            sub_str: "420",
            iters: 2000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_1920x1080_420.jpg",
            subsampling: Subsampling::S420,
            sub_str: "420",
            iters: 500,
        },
        // Subsampling modes (320x240)
        EncodeCase {
            fixture: "tests/fixtures/photo_320x240_444.jpg",
            subsampling: Subsampling::S444,
            sub_str: "444",
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_320x240_422.jpg",
            subsampling: Subsampling::S422,
            sub_str: "422",
            iters: 5000,
        },
        // Subsampling modes (640x480)
        EncodeCase {
            fixture: "tests/fixtures/photo_640x480_444.jpg",
            subsampling: Subsampling::S444,
            sub_str: "444",
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_640x480_422.jpg",
            subsampling: Subsampling::S422,
            sub_str: "422",
            iters: 5000,
        },
        // Subsampling modes (1920x1080)
        EncodeCase {
            fixture: "tests/fixtures/photo_1920x1080_444.jpg",
            subsampling: Subsampling::S444,
            sub_str: "444",
            iters: 500,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_1920x1080_422.jpg",
            subsampling: Subsampling::S422,
            sub_str: "422",
            iters: 500,
        },
    ];

    println!(
        "{:<50} {:>10} {:>12} {:>8}",
        "Case", "Size", "Time", "Iters"
    );
    println!("{}", "-".repeat(85));

    for case in &cases {
        let jpeg_data = match std::fs::read(case.fixture) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skip: {} (not found)", case.fixture);
                continue;
            }
        };
        let image = libjpeg_turbo_rs::decompress(&jpeg_data).unwrap();

        // Warmup
        for _ in 0..100 {
            let _ = libjpeg_turbo_rs::compress_progressive(
                &image.data,
                image.width,
                image.height,
                PixelFormat::Rgb,
                75,
                case.subsampling,
            )
            .unwrap();
        }

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..case.iters {
            let result = libjpeg_turbo_rs::compress_progressive(
                &image.data,
                image.width,
                image.height,
                PixelFormat::Rgb,
                75,
                case.subsampling,
            )
            .unwrap();
            std::hint::black_box(&result);
        }
        let elapsed = start.elapsed();
        let per_iter = elapsed.as_micros() as f64 / case.iters as f64;

        println!(
            "RS_prog_encode_{:>4}x{:<4}_{:<3}                          {:>4}x{:<4} {:>10.1} us  ({} iters)",
            image.width,
            image.height,
            case.sub_str,
            image.width,
            image.height,
            per_iter,
            case.iters,
        );
    }
}
