//! Encode raw RGB pixels to a JPEG, one-shot and via the builder.
//!
//! ```sh
//! cargo run --example encode
//! ```

use libjpeg_turbo_rs::{compress, Encoder, PixelFormat, Subsampling};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (width, height): (usize, usize) = (256, 192);
    let mut rgb: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.extend_from_slice(&[(x * 255 / width) as u8, (y * 255 / height) as u8, 128]);
        }
    }

    // One-shot.
    let jpeg = compress(&rgb, width, height, PixelFormat::Rgb, 85, Subsampling::S420)?;
    println!("one-shot: {} bytes at quality 85", jpeg.len());

    // Builder: progressive, optimized Huffman, a comment marker.
    let fancy = Encoder::new(&rgb, width, height, PixelFormat::Rgb)
        .quality(92)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .optimize_huffman(true)
        .comment("made by examples/encode.rs")
        .encode()?;
    println!(
        "builder (progressive 4:4:4 q92, optimized): {} bytes",
        fancy.len()
    );

    let out = std::env::temp_dir().join("libjpeg_turbo_rs_example.jpg");
    std::fs::write(&out, &fancy)?;
    println!("wrote {}", out.display());
    Ok(())
}
