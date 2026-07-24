/// Lossless-transform benchmark (issue #308).
///
/// Synthesizes a photo-like 24 MP JPEG (6000x4000, Q90, 4:2:0), then times
/// `transform_jpeg_with_options` (Rot90, copy all markers) plus the
/// `read_coefficients` / `write_coefficients` halves separately so the
/// entropy-decode vs entropy-encode split is visible.
///
/// Usage:
///   cargo run --release --example bench_transform [iters] [--save <path>]
///
/// `--save` writes the synthesized JPEG so the same input can be fed to C
/// `jpegtran -rot 90` for a side-by-side comparison.
use libjpeg_turbo_rs::{
    compress, read_coefficients, transform_jpeg_with_options, write_coefficients, MarkerCopyMode,
    PixelFormat, Subsampling, TransformOp, TransformOptions,
};
use std::time::Instant;

/// Deterministic xorshift32 PRNG so every run benches identical content.
struct XorShift32(u32);

impl XorShift32 {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
}

/// Photo-like content: smooth low-frequency gradients (sky/ground) plus
/// medium-amplitude noise (foliage/texture). Tuned so 6000x4000 @ Q90 4:2:0
/// lands around the ~6 MB the issue #308 report used.
fn synthesize_photo(width: usize, height: usize) -> Vec<u8> {
    let mut rng = XorShift32(0x9E3779B9);
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let fy = y as f32 / height as f32;
        for x in 0..width {
            let fx = x as f32 / width as f32;
            // Low-frequency base: diagonal sky-to-ground gradient with a
            // couple of broad sinusoidal "hills".
            let base_r = 90.0 + 100.0 * fy + 30.0 * (fx * 6.3).sin();
            let base_g = 120.0 + 60.0 * fx + 30.0 * (fy * 9.4).cos();
            let base_b = 160.0 - 90.0 * fy + 20.0 * ((fx + fy) * 12.6).sin();
            // Texture noise: +-12 levels, uncorrelated per pixel/channel.
            let n = rng.next();
            let nr = ((n & 0x1F) as f32) - 15.5;
            let ng = (((n >> 5) & 0x1F) as f32) - 15.5;
            let nb = (((n >> 10) & 0x1F) as f32) - 15.5;
            pixels.push((base_r + nr).clamp(0.0, 255.0) as u8);
            pixels.push((base_g + ng).clamp(0.0, 255.0) as u8);
            pixels.push((base_b + nb).clamp(0.0, 255.0) as u8);
        }
    }
    pixels
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iters: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let save_path: Option<&str> = args
        .iter()
        .position(|a| a == "--save")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());

    let (w, h): (usize, usize) = (6000, 4000);
    eprintln!("synthesizing {}x{} photo-like RGB...", w, h);
    let pixels = synthesize_photo(w, h);
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420)
        .expect("encode 24MP source JPEG");
    eprintln!("source JPEG: {:.2} MB", jpeg.len() as f64 / 1e6);

    if let Some(path) = save_path {
        std::fs::write(path, &jpeg).expect("save source JPEG");
        eprintln!("saved to {}", path);
    }

    let opts = TransformOptions {
        op: TransformOp::Rot90,
        copy_markers: MarkerCopyMode::All,
        ..Default::default()
    };

    // Warmup.
    let out = transform_jpeg_with_options(&jpeg, &opts).expect("warmup transform");
    eprintln!("rot90 output: {:.2} MB", out.len() as f64 / 1e6);

    // Full transform.
    let start = Instant::now();
    for _ in 0..iters {
        let out = transform_jpeg_with_options(&jpeg, &opts).expect("transform");
        std::hint::black_box(&out);
    }
    let full_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Entropy-decode half.
    let start = Instant::now();
    for _ in 0..iters {
        let coeffs = read_coefficients(&jpeg).expect("read_coefficients");
        std::hint::black_box(&coeffs);
    }
    let read_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Entropy-encode half (re-encode of unrotated coefficients).
    let coeffs = read_coefficients(&jpeg).expect("read_coefficients");
    let start = Instant::now();
    for _ in 0..iters {
        let out = write_coefficients(&coeffs).expect("write_coefficients");
        std::hint::black_box(&out);
    }
    let write_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    println!(
        "transform_rot90 {:>8.1} ms/iter  ({} iters)",
        full_ms, iters
    );
    println!("read_coefficients {:>6.1} ms/iter", read_ms);
    println!("write_coefficients {:>5.1} ms/iter", write_ms);
}
