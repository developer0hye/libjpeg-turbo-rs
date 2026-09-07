/// Standalone encoding benchmark matrix.
/// Usage: cargo run --release --example bench_encode_matrix
///
/// `BENCH_DCT_METHOD=islow|ifast|float` selects the forward DCT (default
/// `islow`, the same default `compress()` uses). The float method is the one
/// path where FMA codegen matters (P4-133 / #464), so the portable-vs-native
/// A/B runs the matrix once per method from the same binary.
fn main() {
    use libjpeg_turbo_rs::{DctMethod, Encoder, PixelFormat, Subsampling};

    let dct_method: DctMethod = match std::env::var("BENCH_DCT_METHOD").as_deref() {
        Err(_) | Ok("") | Ok("islow") => DctMethod::IsLow,
        Ok("ifast") => DctMethod::IsFast,
        Ok("float") => DctMethod::Float,
        Ok(other) => panic!("BENCH_DCT_METHOD must be islow, ifast or float, got {other:?}"),
    };
    let dct_label: &str = match dct_method {
        DctMethod::IsLow => "islow",
        DctMethod::IsFast => "ifast",
        DctMethod::Float => "float",
    };
    // Quality stays 75 on purpose: below 50 the builder takes the
    // force_baseline-aware quantization path that `compress()` never did,
    // and the numbers stop being comparable with experiments/encode.tsv.
    let encode = |pixels: &[u8], width: usize, height: usize, subsampling: Subsampling| {
        Encoder::new(pixels, width, height, PixelFormat::Rgb)
            .quality(75)
            .subsampling(subsampling)
            .dct_method(dct_method)
            .encode()
            .unwrap()
    };

    struct EncodeCase {
        fixture: &'static str,
        subsampling: Subsampling,
        iters: u32,
    }

    let cases: Vec<EncodeCase> = vec![
        // Resolution scaling (4:2:0)
        EncodeCase {
            fixture: "tests/fixtures/photo_64x64_420.jpg",
            subsampling: Subsampling::S420,
            iters: 20000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_320x240_420.jpg",
            subsampling: Subsampling::S420,
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_640x480_422.jpg",
            subsampling: Subsampling::S420,
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_1280x720_420.jpg",
            subsampling: Subsampling::S420,
            iters: 2000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_1920x1080_420.jpg",
            subsampling: Subsampling::S420,
            iters: 500,
        },
        // Subsampling modes (320x240)
        EncodeCase {
            fixture: "tests/fixtures/photo_320x240_444.jpg",
            subsampling: Subsampling::S444,
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_320x240_422.jpg",
            subsampling: Subsampling::S422,
            iters: 5000,
        },
        // Subsampling modes (640x480)
        EncodeCase {
            fixture: "tests/fixtures/photo_640x480_444.jpg",
            subsampling: Subsampling::S444,
            iters: 5000,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_640x480_422.jpg",
            subsampling: Subsampling::S422,
            iters: 5000,
        },
        // Subsampling modes (1920x1080)
        EncodeCase {
            fixture: "tests/fixtures/photo_1920x1080_444.jpg",
            subsampling: Subsampling::S444,
            iters: 500,
        },
        EncodeCase {
            fixture: "tests/fixtures/photo_1920x1080_422.jpg",
            subsampling: Subsampling::S422,
            iters: 500,
        },
    ];

    println!("dct_method: {dct_label}");
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
        let pixels = &image.data;
        let width = image.width;
        let height = image.height;

        // Warmup
        for _ in 0..100 {
            let _ = encode(pixels, width, height, case.subsampling);
        }

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..case.iters {
            let result = encode(pixels, width, height, case.subsampling);
            std::hint::black_box(&result);
        }
        let elapsed = start.elapsed();
        let us: f64 = elapsed.as_nanos() as f64 / case.iters as f64 / 1000.0;

        let sub_str = match case.subsampling {
            Subsampling::S420 => "420",
            Subsampling::S422 => "422",
            Subsampling::S444 => "444",
            _ => "???",
        };

        println!(
            "encode_{:>4}x{:<4}_{:<3}                                {:>4}x{:<4} {:>10.1} us  ({} iters)",
            width, height, sub_str, width, height, us, case.iters
        );
    }
}
