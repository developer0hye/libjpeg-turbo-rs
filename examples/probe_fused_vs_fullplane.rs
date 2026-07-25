//! Probe: which `compress_*` buffering strategy matches C cjpeg, and where does
//! the public encoder diverge from cjpeg at partial-MCU geometries?
//!
//! `compress()` uses a fused single-pass color-convert + encode strategy for
//! RGB-family input; every other `compress_*` variant uses full-plane
//! `convert_to_ycbcr`. The characterization fixture shows the two disagree on
//! partial-MCU geometries with chroma subsampling, so at most one matches C.
//!
//! Writes the input as PPM plus both Rust encodings for every case, so a shell
//! wrapper can run stock `cjpeg` over the same PPM and diff all three. Prints
//! one tab-separated record per case.
//!
//! Usage: `probe_fused_vs_fullplane <output-dir> [sweep|realworld]`

use libjpeg_turbo_rs::encode::pipeline;
use libjpeg_turbo_rs::{DctMethod, PixelFormat, Subsampling};
use std::io::Write;

/// Must stay identical to `synthetic_pixels` in `tests/encode_pipeline_golden.rs`
/// so the probe reproduces the exact fixture cases that diverged.
fn synthetic_pixels(width: usize, height: usize, bytes_per_pixel: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * bytes_per_pixel];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let gradient_x: i32 = (x * 255 / width.max(1)) as i32;
            let gradient_y: i32 = (y * 255 / height.max(1)) as i32;
            let in_rect: bool =
                x * 3 >= width && x * 3 < width * 2 && y * 3 >= height && y * 3 < height * 2;
            let edge: i32 = if in_rect { 200 } else { 0 };
            let red: u8 = (gradient_x + noise + edge).clamp(0, 255) as u8;
            let green: u8 = (gradient_y + noise).clamp(0, 255) as u8;
            let blue: u8 = ((gradient_x + gradient_y) / 2 - noise + edge / 2).clamp(0, 255) as u8;
            let offset: usize = (y * width + x) * bytes_per_pixel;
            for channel in 0..bytes_per_pixel {
                pixels[offset + channel] = match channel {
                    0 => red,
                    1 => green,
                    2 => blue,
                    _ => red ^ green,
                };
            }
        }
    }
    pixels
}

fn emit_case(out_dir: &str, width: usize, height: usize, subsampling: Subsampling, sample: &str) {
    let quality: u8 = 50;
    let pixels: Vec<u8> = synthetic_pixels(width, height, 3);
    let stem: String = format!("{width}x{height}_{sample}_q{quality}");

    // P6 binary PPM — cjpeg's native RGB input format.
    let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.extend_from_slice(&pixels);
    std::fs::write(format!("{out_dir}/{stem}.ppm"), &ppm).expect("write ppm");

    // The public high-level entry point routes here with DctMethod::IsLow, so
    // this is exactly what a library user gets from `libjpeg_turbo_rs::compress`.
    let fused: Vec<u8> = pipeline::compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        quality,
        subsampling,
        DctMethod::IsLow,
    )
    .expect("fused compress");
    std::fs::write(format!("{out_dir}/{stem}.fused.jpg"), &fused).expect("write fused");

    // restart_interval = 0 emits no RST markers, so this isolates the buffering
    // strategy as the only difference from the fused path.
    let full_plane: Vec<u8> = pipeline::compress_with_restart(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        quality,
        subsampling,
        0,
        DctMethod::IsLow,
    )
    .expect("full-plane compress");
    std::fs::write(format!("{out_dir}/{stem}.fullplane.jpg"), &full_plane)
        .expect("write full-plane");

    // The two-pass optimized-Huffman variant, which is what `Encoder` uses
    // whenever `optimize_huffman(true)` or a smoothing factor is set. Compared
    // against `cjpeg -optimize`.
    let optimized: Vec<u8> = pipeline::compress_optimized(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        quality,
        subsampling,
        0,
        DctMethod::IsLow,
        0,
    )
    .expect("optimized compress");
    std::fs::write(format!("{out_dir}/{stem}.optimized.jpg"), &optimized).expect("write optimized");

    // A real restart interval, compared against `cjpeg -restart 3B` (the `B`
    // suffix means "in MCU blocks", which is what our interval counts).
    let restarted: Vec<u8> = pipeline::compress_with_restart(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        quality,
        subsampling,
        3,
        DctMethod::IsLow,
    )
    .expect("restart compress");
    std::fs::write(format!("{out_dir}/{stem}.restart3.jpg"), &restarted).expect("write restart3");

    let _ = writeln!(
        std::io::stdout(),
        "{stem}\t{sample}\t{width}\t{height}\t{}",
        fused == full_plane
    );
}

fn main() {
    let out_dir: String = std::env::args()
        .nth(1)
        .expect("usage: probe_fused_vs_fullplane <output-dir> [sweep|realworld]");
    let mode: String = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "sweep".to_string());

    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "1x1"),
        (Subsampling::S422, "2x1"),
        (Subsampling::S420, "2x2"),
        (Subsampling::S440, "1x2"),
    ];

    match mode.as_str() {
        // Every partial-MCU residue for 16-pixel MCUs, so the boundary
        // condition is characterized rather than sampled.
        "sweep" => {
            let dimensions: &[usize] = &[7, 8, 15, 16, 17, 23, 24, 31, 32, 33, 48, 64];
            for &(subsampling, sample) in subsamplings {
                for &width in dimensions {
                    for &height in dimensions {
                        emit_case(&out_dir, width, height, subsampling, sample);
                    }
                }
            }
        }
        // Ordinary photo dimensions whose width leaves >= 8 pixels of MCU
        // padding at 4:2:0 — i.e. width % 16 in 1..=8.
        "realworld" => {
            let cases: &[(usize, usize)] = &[
                (1000, 750),  // 1000 % 16 == 8
                (1000, 1000), // 1000 % 16 == 8
                (1080, 1080), // 1080 % 16 == 8
                (500, 375),   // 500  % 16 == 4
                (1200, 900),  // 1200 % 16 == 0 — control, must match
                (1920, 1080), // 1920 % 16 == 0 — control, must match
            ];
            for &(width, height) in cases {
                for &(subsampling, sample) in subsamplings {
                    emit_case(&out_dir, width, height, subsampling, sample);
                }
            }
        }
        other => panic!("unknown mode {other:?}; expected sweep or realworld"),
    }
}
