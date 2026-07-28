//! Use libjpeg-turbo-rs as the JPEG decoder behind the `image` crate's
//! traits — drop-in speed for existing `image` pipelines.
//!
//! ```sh
//! cargo run -p libjpeg-turbo-rs-image --example decode_with_image [file.jpg]
//! ```
//!
//! With no argument, a JPEG is synthesized in-process so the example is
//! self-contained even in the published crate (which ships no fixtures).

use image::ImageDecoder;
use libjpeg_turbo_rs_image::JpegDecoder;

fn synthesize_jpeg() -> Vec<u8> {
    let (width, height): (usize, usize) = (320, 240);
    let mut rgb: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.extend_from_slice(&[
                (x * 255 / width) as u8,
                (y * 255 / height) as u8,
                ((x + y) % 256) as u8,
            ]);
        }
    }
    libjpeg_turbo_rs::compress(
        &rgb,
        width,
        height,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        88,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("synthesized encode cannot fail")
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (label, data): (String, Vec<u8>) = match std::env::args().nth(1) {
        Some(path) => {
            let bytes = std::fs::read(&path)?;
            (path, bytes)
        }
        None => ("<synthesized 320x240>".to_string(), synthesize_jpeg()),
    };

    let decoder = JpegDecoder::new(&data)?;
    let (width, height) = decoder.dimensions();
    let mut pixels: Vec<u8> = vec![0u8; decoder.total_bytes() as usize];
    decoder.read_image(&mut pixels)?;
    println!(
        "{label}: {width}x{height}, {} pixel bytes via the image-crate traits",
        pixels.len()
    );
    Ok(())
}
