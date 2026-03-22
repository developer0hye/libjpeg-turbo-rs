use std::fs;
use std::path::Path;

/// Generate minimal seed JPEG files for the fuzzing corpus.
///
/// Run manually with: `cargo test generate_seeds -- --ignored`
///
/// This creates small but valid JPEG files in `fuzz/corpus/<target>/` so the
/// fuzzer starts from structurally valid inputs rather than pure random bytes.
#[test]
#[ignore]
fn generate_seeds() {
    let corpus_base: &Path = Path::new("fuzz/corpus");

    // Directories to populate with decoder-oriented seeds
    let decoder_targets: &[&str] = &[
        "fuzz_decompress",
        "fuzz_decompress_lenient",
        "fuzz_read_coefficients",
        "fuzz_transform",
        "fuzz_progressive_decoder",
    ];

    for target in decoder_targets {
        let target_dir = corpus_base.join(target);
        fs::create_dir_all(&target_dir).expect("failed to create corpus directory");

        // Copy small fixture files as seeds
        let fixtures: &[&str] = &[
            "tests/fixtures/gray_8x8.jpg",
            "tests/fixtures/red_16x16_444.jpg",
            "tests/fixtures/blue_16x16_420.jpg",
            "tests/fixtures/photo_64x64_420.jpg",
        ];

        for fixture_path in fixtures {
            let src: &Path = Path::new(fixture_path);
            if src.exists() {
                let filename: &str = src.file_name().unwrap().to_str().unwrap();
                let dest = target_dir.join(filename);
                fs::copy(src, &dest).expect("failed to copy fixture");
            }
        }
    }

    // Add progressive seed to the progressive decoder target
    let prog_seed: &Path = Path::new("tests/fixtures/blue_16x16_420_prog.jpg");
    if prog_seed.exists() {
        let dest = corpus_base.join("fuzz_progressive_decoder/blue_16x16_420_prog.jpg");
        fs::copy(prog_seed, &dest).expect("failed to copy progressive fixture");
    }

    // Generate encoder-produced seeds at various qualities and subsamplings
    let roundtrip_dir = corpus_base.join("fuzz_roundtrip");
    fs::create_dir_all(&roundtrip_dir).expect("failed to create roundtrip corpus directory");

    let width: usize = 8;
    let height: usize = 8;
    let pixel_count: usize = width * height * 3;

    // Solid red pixels
    let red_pixels: Vec<u8> = vec![255, 0, 0].repeat(width * height);
    // Gradient pixels
    let gradient_pixels: Vec<u8> = (0..pixel_count as u8).collect();

    let subsamplings: &[(libjpeg_turbo_rs::Subsampling, &str)] = &[
        (libjpeg_turbo_rs::Subsampling::S420, "420"),
        (libjpeg_turbo_rs::Subsampling::S422, "422"),
        (libjpeg_turbo_rs::Subsampling::S444, "444"),
    ];

    for (subsampling, label) in subsamplings {
        for quality in [10, 50, 90] {
            for (pixels, name) in [(&red_pixels, "red"), (&gradient_pixels, "grad")] {
                let jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
                    pixels,
                    width,
                    height,
                    libjpeg_turbo_rs::PixelFormat::Rgb,
                    quality,
                    *subsampling,
                )
                .expect("compress must succeed for seed generation");

                let filename: String = format!("{name}_{width}x{height}_{label}_q{quality}.jpg");
                fs::write(roundtrip_dir.join(&filename), &jpeg)
                    .expect("failed to write roundtrip seed");

                // Also copy encoder-produced seeds into decoder targets
                for target in decoder_targets {
                    let dest = corpus_base.join(target).join(&filename);
                    fs::write(&dest, &jpeg).expect("failed to write decoder seed");
                }
            }
        }
    }

    println!("Seed corpus generated successfully.");
}
