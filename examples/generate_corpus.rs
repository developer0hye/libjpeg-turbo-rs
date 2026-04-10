/// Generate a diverse JPEG test corpus in tests/corpus/.
///
/// Usage: cargo run --example generate_corpus
///
/// Output layout:
///   tests/corpus/generated/  — JPEGs produced by C cjpeg from synthetic/reference PPMs
///   tests/corpus/fuzz_seeds/ — copies of fuzz/corpus/fuzz_decompress/*.jpg
///   tests/corpus/fixtures/   — copies of tests/fixtures/*.jpg
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// C tool helpers
// ---------------------------------------------------------------------------

fn c_tool_path(name: &str) -> Option<PathBuf> {
    let homebrew = PathBuf::from(format!("/opt/homebrew/bin/{}", name));
    if homebrew.exists() {
        return Some(homebrew);
    }
    std::process::Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn run_cjpeg(cjpeg: &Path, input_ppm: &Path, output_jpg: &Path, args: &[&str]) -> bool {
    let output = std::process::Command::new(cjpeg)
        .args(args)
        .arg("-outfile")
        .arg(output_jpg)
        .arg(input_ppm)
        .output();
    match output {
        Ok(o) if o.status.success() => true,
        Ok(o) => {
            eprintln!("cjpeg failed: {}", String::from_utf8_lossy(&o.stderr));
            false
        }
        Err(e) => {
            eprintln!("cjpeg error: {}", e);
            false
        }
    }
}

// ---------------------------------------------------------------------------
// PPM generation (P6 binary format)
// ---------------------------------------------------------------------------

fn write_ppm(path: &Path, width: usize, height: usize, pixels: &[u8]) {
    let header = format!("P6\n{} {}\n255\n", width, height);
    let mut data = Vec::with_capacity(header.len() + pixels.len());
    data.extend_from_slice(header.as_bytes());
    data.extend_from_slice(pixels);
    std::fs::write(path, &data).expect("failed to write PPM");
}

fn make_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width.max(1)) as u8;
            let g = ((y * 255) / height.max(1)) as u8;
            let b = (((x + y) * 255) / (width + height).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

fn make_solid(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    vec![r, g, b]
        .into_iter()
        .cycle()
        .take(width * height * 3)
        .collect()
}

fn make_checkerboard(width: usize, height: usize, block: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let is_white = ((x / block) + (y / block)) % 2 == 0;
            let v = if is_white { 255u8 } else { 0u8 };
            pixels.push(v);
            pixels.push(v);
            pixels.push(v);
        }
    }
    pixels
}

/// Deterministic pseudo-random noise (LCG-based, no external deps)
fn make_noise(width: usize, height: usize, seed: u64) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    let mut state = seed;
    for _ in 0..(width * height * 3) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        pixels.push((state >> 33) as u8);
    }
    pixels
}

/// High-contrast edges (vertical and horizontal bars with color transitions)
fn make_edges(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let h_edge = (x % 16 < 8) as u8 * 255;
            let v_edge = (y % 16 < 8) as u8 * 255;
            let diag = ((x + y) % 8 < 4) as u8 * 200;
            pixels.push(h_edge);
            pixels.push(v_edge);
            pixels.push(diag);
        }
    }
    pixels
}

/// Sine wave pattern (smooth gradients with varying frequency)
fn make_sine(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64 / width.max(1) as f64;
            let fy = y as f64 / height.max(1) as f64;
            let r = ((fx * 4.0 * std::f64::consts::PI).sin() * 127.0 + 128.0) as u8;
            let g = ((fy * 6.0 * std::f64::consts::PI).sin() * 127.0 + 128.0) as u8;
            let b = (((fx + fy) * 3.0 * std::f64::consts::PI).sin() * 127.0 + 128.0) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Mixed-frequency pattern simulating natural-looking content
fn make_natural(width: usize, height: usize) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);
    let mut state: u64 = 42;
    for y in 0..height {
        for x in 0..width {
            let fx = x as f64 / width.max(1) as f64;
            let fy = y as f64 / height.max(1) as f64;
            // Low-frequency base
            let base_r = ((fx * 2.0 * std::f64::consts::PI).sin() * 60.0 + 128.0) as f64;
            let base_g = ((fy * 1.5 * std::f64::consts::PI).cos() * 60.0 + 128.0) as f64;
            let base_b = (((fx * fy) * 4.0 * std::f64::consts::PI).sin() * 40.0 + 128.0) as f64;
            // High-frequency noise
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((state >> 33) as f64 / 255.0 - 0.5) * 30.0;
            let r = (base_r + noise).clamp(0.0, 255.0) as u8;
            let g = (base_g + noise * 0.7).clamp(0.0, 255.0) as u8;
            let b = (base_b + noise * 0.5).clamp(0.0, 255.0) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// Source PPM descriptors
// ---------------------------------------------------------------------------

struct SourcePpm {
    name: String,
    path: PathBuf,
}

fn prepare_sources(tmp_dir: &Path) -> Vec<SourcePpm> {
    let mut sources: Vec<SourcePpm> = Vec::new();

    // Reference PPMs from libjpeg-turbo testimages
    for name in &["testorig", "testimg"] {
        let path = PathBuf::from(format!("references/libjpeg-turbo/testimages/{}.ppm", name));
        if path.exists() {
            sources.push(SourcePpm {
                name: name.to_string(),
                path,
            });
        }
    }

    if sources.is_empty() {
        eprintln!("warning: no reference PPMs found in references/libjpeg-turbo/testimages/");
    }

    // Synthetic images: (name, width, height, pixel_fn)
    type PixelFn = Box<dyn Fn(usize, usize) -> Vec<u8>>;
    let mut synthetics: Vec<(&str, usize, usize, PixelFn)> = vec![
        (
            "gradient_64x64",
            64,
            64,
            Box::new(|w, h| make_gradient(w, h)),
        ),
        (
            "gradient_640x480",
            640,
            480,
            Box::new(|w, h| make_gradient(w, h)),
        ),
        (
            "solid_red_8x8",
            8,
            8,
            Box::new(|w, h| make_solid(w, h, 255, 0, 0)),
        ),
        (
            "checkerboard_32x32",
            32,
            32,
            Box::new(|w, h| make_checkerboard(w, h, 2)),
        ),
        ("tiny_1x1", 1, 1, Box::new(|w, h| make_gradient(w, h))),
        ("tiny_3x3", 3, 3, Box::new(|w, h| make_gradient(w, h))),
        ("odd_7x11", 7, 11, Box::new(|w, h| make_gradient(w, h))),
        ("odd_33x17", 33, 17, Box::new(|w, h| make_gradient(w, h))),
        ("strip_100x1", 100, 1, Box::new(|w, h| make_gradient(w, h))),
        ("strip_1x100", 1, 100, Box::new(|w, h| make_gradient(w, h))),
        // Phase 2 additions: more resolutions, content patterns, edge cases
        (
            "noise_320x240",
            320,
            240,
            Box::new(|w, h| make_noise(w, h, 12345)),
        ),
        (
            "noise_17x31",
            17,
            31,
            Box::new(|w, h| make_noise(w, h, 67890)),
        ),
        ("edges_256x256", 256, 256, Box::new(|w, h| make_edges(w, h))),
        ("sine_800x600", 800, 600, Box::new(|w, h| make_sine(w, h))),
        (
            "natural_640x480",
            640,
            480,
            Box::new(|w, h| make_natural(w, h)),
        ),
        (
            "natural_1280x720",
            1280,
            720,
            Box::new(|w, h| make_natural(w, h)),
        ),
        ("odd_127x63", 127, 63, Box::new(|w, h| make_gradient(w, h))),
        (
            "odd_255x127",
            255,
            127,
            Box::new(|w, h| make_noise(w, h, 99999)),
        ),
        ("tall_16x256", 16, 256, Box::new(|w, h| make_gradient(w, h))),
        ("wide_256x16", 256, 16, Box::new(|w, h| make_edges(w, h))),
        (
            "solid_black_8x8",
            8,
            8,
            Box::new(|w, h| make_solid(w, h, 0, 0, 0)),
        ),
        (
            "solid_white_8x8",
            8,
            8,
            Box::new(|w, h| make_solid(w, h, 255, 255, 255)),
        ),
        // Large resolutions: 4K (always included)
        (
            "natural_3840x2160",
            3840,
            2160,
            Box::new(|w, h| make_natural(w, h)),
        ),
    ];

    // 8K sources are opt-in via CORPUS_INCLUDE_8K=1 (too slow for CI)
    if std::env::var("CORPUS_INCLUDE_8K").as_deref() == Ok("1") {
        synthetics.push((
            "gradient_7680x4320",
            7680,
            4320,
            Box::new(|w, h| make_gradient(w, h)),
        ));
        synthetics.push((
            "noise_7680x4320",
            7680,
            4320,
            Box::new(|w, h| make_noise(w, h, 77777)),
        ));
        synthetics.push((
            "edges_7680x4320",
            7680,
            4320,
            Box::new(|w, h| make_edges(w, h)),
        ));
        // Odd 8K dimensions (non-MCU-aligned)
        synthetics.push((
            "natural_7681x4321",
            7681,
            4321,
            Box::new(|w, h| make_natural(w, h)),
        ));
    }

    for (name, width, height, gen) in &synthetics {
        let ppm_path = tmp_dir.join(format!("{}.ppm", name));
        let pixels = gen(*width, *height);
        write_ppm(&ppm_path, *width, *height, &pixels);
        sources.push(SourcePpm {
            name: name.to_string(),
            path: ppm_path,
        });
    }

    sources
}

// ---------------------------------------------------------------------------
// cjpeg variant matrix
// ---------------------------------------------------------------------------

struct CjpegVariant {
    /// Used in the output filename: {source}_{label}.jpg
    label: String,
    args: Vec<String>,
}

fn all_variants() -> Vec<CjpegVariant> {
    let mut variants: Vec<CjpegVariant> = Vec::new();

    // Axes
    let subsampling: &[(&str, Option<&str>)] = &[
        ("420", Some("2x2")),
        ("422", Some("2x1")),
        ("444", Some("1x1")),
        ("440", Some("1x2")),
        ("411", Some("4x1")),
        ("gray", None), // uses -grayscale instead
    ];
    let qualities: &[u32] = &[1, 10, 25, 50, 75, 90, 95, 100];

    // Flags that get OR'd with the base baseline pass
    // (label suffix, extra args)
    let flag_sets: &[(&str, &[&str])] = &[
        ("baseline", &[]),
        ("progressive", &["-progressive"]),
        ("optimized", &["-optimize"]),
        ("arithmetic", &["-arithmetic"]),
        ("arithmetic_progressive", &["-arithmetic", "-progressive"]),
        ("restart1", &["-restart", "1"]),
        ("restart4", &["-restart", "4"]),
        ("smooth50", &["-smooth", "50"]),
    ];

    for (subsamp_label, sample_arg) in subsampling {
        for &quality in qualities {
            for (flag_label, extra_args) in flag_sets {
                // Arithmetic + progressive is only meaningful at some qualities; generate for all.
                let label = format!("{}_{}_{}", subsamp_label, quality, flag_label);
                let mut args: Vec<String> = Vec::new();

                if let Some(sample) = sample_arg {
                    args.push("-sample".to_string());
                    args.push(sample.to_string());
                } else {
                    args.push("-grayscale".to_string());
                }

                args.push("-quality".to_string());
                args.push(quality.to_string());

                for a in *extra_args {
                    args.push(a.to_string());
                }

                variants.push(CjpegVariant { label, args });
            }
        }
    }

    variants
}

fn generate_jpegs(cjpeg: &Path, sources: &[SourcePpm], out_dir: &Path) -> (usize, usize) {
    let variants = all_variants();

    // Collect all (source, variant) pairs for parallel execution
    let work: Vec<(&SourcePpm, &CjpegVariant)> = sources
        .iter()
        .flat_map(|source| variants.iter().map(move |variant| (source, variant)))
        .collect();

    let generated = std::sync::atomic::AtomicUsize::new(0);
    let failed = std::sync::atomic::AtomicUsize::new(0);

    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let chunk_size = (work.len() + parallelism - 1) / parallelism;

    let generated_ref = &generated;
    let failed_ref = &failed;

    std::thread::scope(|s| {
        for chunk in work.chunks(chunk_size) {
            s.spawn(move || {
                for (source, variant) in chunk {
                    let filename = format!("{}_{}.jpg", source.name, variant.label);
                    let output_jpg = out_dir.join(&filename);
                    let args: Vec<&str> = variant.args.iter().map(|s| s.as_str()).collect();
                    if run_cjpeg(cjpeg, &source.path, &output_jpg, &args) {
                        generated_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        failed_ref.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });
        }
    });

    (
        generated.load(std::sync::atomic::Ordering::Relaxed),
        failed.load(std::sync::atomic::Ordering::Relaxed),
    )
}

// ---------------------------------------------------------------------------
// File copying
// ---------------------------------------------------------------------------

fn copy_jpgs(src_dir: &Path, dst_dir: &Path) -> usize {
    let mut count = 0usize;
    let entries = match std::fs::read_dir(src_dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("jpg") {
            let dst = dst_dir.join(path.file_name().unwrap());
            if std::fs::copy(&path, &dst).is_ok() {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cjpeg = match c_tool_path("cjpeg") {
        Some(p) => p,
        None => {
            eprintln!(
                "warning: cjpeg not found in /opt/homebrew/bin or PATH — cannot generate JPEG corpus"
            );
            eprintln!("install libjpeg-turbo (e.g. `brew install libjpeg-turbo`) and re-run.");
            std::process::exit(1);
        }
    };
    println!("cjpeg: {}", cjpeg.display());

    // Create output directories
    let corpus_dir = PathBuf::from("tests/corpus");
    let generated_dir = corpus_dir.join("generated");
    let fuzz_seeds_dir = corpus_dir.join("fuzz_seeds");
    let fixtures_dir = corpus_dir.join("fixtures");

    for dir in [&generated_dir, &fuzz_seeds_dir, &fixtures_dir] {
        std::fs::create_dir_all(dir).expect("failed to create output directory");
    }

    // Temp directory for synthetic PPMs
    let tmp_dir = std::env::temp_dir().join("libjpeg_turbo_rs_corpus_ppms");
    std::fs::create_dir_all(&tmp_dir).expect("failed to create temp directory");

    // Prepare source images
    println!("Preparing source PPM images...");
    let sources = prepare_sources(&tmp_dir);
    println!("  {} source images ready", sources.len());

    // Generate JPEG matrix
    let parallelism = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    println!(
        "Running cjpeg matrix ({} variants × {} sources, {} threads)...",
        all_variants().len(),
        sources.len(),
        parallelism,
    );
    let (generated, failed) = generate_jpegs(&cjpeg, &sources, &generated_dir);
    println!("  generated: {}  failed: {}", generated, failed);

    // Cleanup temp PPMs
    let _ = std::fs::remove_dir_all(&tmp_dir);

    // Copy fuzz seeds
    let fuzz_src = PathBuf::from("fuzz/corpus/fuzz_decompress");
    let fuzz_count = copy_jpgs(&fuzz_src, &fuzz_seeds_dir);
    println!("  fuzz_seeds: {} files copied", fuzz_count);

    // Copy fixtures
    let fixtures_src = PathBuf::from("tests/fixtures");
    let fixtures_count = copy_jpgs(&fixtures_src, &fixtures_dir);
    println!("  fixtures: {} files copied", fixtures_count);

    // Summary
    let total = generated + fuzz_count + fixtures_count;
    println!();
    println!("Corpus summary:");
    println!("  generated/  : {}", generated);
    println!("  fuzz_seeds/ : {}", fuzz_count);
    println!("  fixtures/   : {}", fixtures_count);
    println!("  total       : {}", total);
    println!();
    println!(
        "Output: {}",
        corpus_dir.canonicalize().unwrap_or(corpus_dir).display()
    );
}
