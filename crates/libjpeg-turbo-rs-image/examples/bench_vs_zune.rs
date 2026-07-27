//! Bridge-vs-zune decode benchmark (issue #381 acceptance criterion).
//!
//! Decodes the same JPEG through (a) this bridge's `JpegDecoder` and
//! (b) the `image` crate's built-in JPEG path (zune-jpeg), timing both.
//! Run sequentially, never in parallel with other benchmarks:
//!
//! ```sh
//! cargo run --release -p libjpeg-turbo-rs-image --example bench_vs_zune
//! ```
//!
//! Results are recorded in `experiments/image_bridge.md`.

use image::ImageDecoder;
use std::time::Instant;

fn synth_photo_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut px: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;
            let r = (255.0 * (0.5 + 0.5 * (fx * 6.7).sin() * (fy * 3.1).cos())) as u8;
            let g = (255.0 * (0.5 + 0.5 * (fx * 2.3 + fy * 5.9).sin())) as u8;
            let b = (255.0 * fy) as u8;
            px.extend_from_slice(&[r, g, b]);
        }
    }
    px
}

fn time_ms<F: FnMut()>(iters: u32, mut f: F) -> f64 {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let (width, height) = (1920usize, 1080usize);
    let rgb = synth_photo_rgb(width, height);
    let jpeg = libjpeg_turbo_rs::compress(
        &rgb,
        width,
        height,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        90,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("encode");
    println!(
        "fixture: {width}x{height} 4:2:0 q90, {} bytes; simd_and_std_features={}",
        jpeg.len(),
        libjpeg_turbo_rs::simd_and_std_features_enabled()
    );

    let iters: u32 = 50;

    // Warm up + correctness cross-check first.
    let mut ours_buf = {
        let dec = libjpeg_turbo_rs_image::JpegDecoder::new(&jpeg).expect("bridge decoder");
        let mut buf = vec![0u8; dec.total_bytes() as usize];
        dec.read_image(&mut buf).expect("bridge decode");
        buf
    };
    let zune_buf = {
        let dec = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg))
            .expect("zune decoder");
        let mut buf = vec![0u8; dec.total_bytes() as usize];
        dec.read_image(&mut buf).expect("zune decode");
        buf
    };
    assert_eq!(ours_buf.len(), zune_buf.len(), "output size mismatch");
    // Decoders differ in IDCT/upsample rounding, so byte-identity is not
    // the contract — but a timing for wrong pixels is worthless. Measured
    // max per-channel diff on this fixture is 3 (2026-07-28); assert
    // measured + 1.
    let max_diff: u8 = ours_buf
        .iter()
        .zip(zune_buf.iter())
        .map(|(a, b)| a.abs_diff(*b))
        .max()
        .unwrap_or(0);
    println!("bridge-vs-zune max per-channel diff: {max_diff}");
    assert!(
        max_diff <= 4,
        "bridge and zune outputs diverge (max_diff={max_diff}, allowed 4): \
         one of the decoders is producing wrong pixels"
    );

    // Both closures reuse a preallocated output buffer: read_image takes
    // caller storage for both implementations, and charging the ~6 MiB
    // allocation to only one leg would skew the ratio.
    let mut zune_reuse_buf = vec![0u8; zune_buf.len()];
    let ours_ms = time_ms(iters, || {
        let dec = libjpeg_turbo_rs_image::JpegDecoder::new(&jpeg).expect("bridge decoder");
        dec.read_image(&mut ours_buf).expect("bridge decode");
    });
    let zune_ms = time_ms(iters, || {
        let dec = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg))
            .expect("zune decoder");
        dec.read_image(&mut zune_reuse_buf).expect("zune decode");
    });

    println!("bridge (libjpeg-turbo-rs): {ours_ms:.3} ms/decode");
    println!("image built-in (zune):     {zune_ms:.3} ms/decode");
    println!("ratio (zune/bridge):       {:.2}x", zune_ms / ours_ms);
}
