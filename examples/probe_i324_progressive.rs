//! Issue #324 debug probe: dump our progressive output for a given geometry so
//! its scan structure can be diffed against `cjpeg`'s.
//!
//! Usage: `probe_i324_progressive <out.jpg> <width> <height> [sample]`
//! where `sample` is `2x2` (default), `1x2`, `2x1` or `1x1`.

use libjpeg_turbo_rs::{compress_progressive, PixelFormat, Subsampling};

fn pixels(width: usize, height: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = vec![0u8; width * height * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let offset: usize = (y * width + x) * 3;
            buffer[offset] = ((x * 255 / width) as i32 + noise).clamp(0, 255) as u8;
            buffer[offset + 1] = ((y * 255 / height) as i32 - noise).clamp(0, 255) as u8;
            buffer[offset + 2] = (((x ^ y) & 0xff) as i32 + noise).clamp(0, 255) as u8;
        }
    }
    buffer
}

fn main() {
    let mut args = std::env::args().skip(1);
    let out: String = args.next().expect("usage: <out.jpg> <w> <h> [sample]");
    let width: usize = args.next().expect("width").parse().expect("width");
    let height: usize = args.next().expect("height").parse().expect("height");
    let subsampling: Subsampling = match args.next().as_deref().unwrap_or("2x2") {
        "1x1" => Subsampling::S444,
        "2x1" => Subsampling::S422,
        "1x2" => Subsampling::S440,
        _ => Subsampling::S420,
    };

    let raw: Vec<u8> = pixels(width, height);
    let encoded: Vec<u8> =
        compress_progressive(&raw, width, height, PixelFormat::Rgb, 90, subsampling)
            .expect("progressive encode");
    std::fs::write(&out, &encoded).expect("write output");

    // Also write the PPM so the C side encodes byte-identical input.
    let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.extend_from_slice(&raw);
    std::fs::write(format!("{out}.ppm"), &ppm).expect("write ppm");

    println!("{} bytes -> {out}", encoded.len());
}
