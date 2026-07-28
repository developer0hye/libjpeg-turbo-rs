//! Decode a JPEG to raw pixels.
//!
//! ```sh
//! cargo run --example decode [path/to/image.jpg]
//! ```
//!
//! Defaults to a bundled test fixture so it runs without arguments.

use libjpeg_turbo_rs::{decompress, decompress_to, PixelFormat};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default fixture, resolved from the crate root regardless of the
    // invocation directory.
    let path: String = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/tests/fixtures/photo_640x480_420.jpg",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let jpeg: Vec<u8> = std::fs::read(&path)?;

    // Default: RGB (or grayscale for 1-component sources).
    let image = decompress(&jpeg)?;
    println!(
        "{path}: {}x{} {:?}, {} bytes of pixels",
        image.width,
        image.height,
        image.pixel_format,
        image.data.len()
    );

    // Any other pixel format is one call away.
    let rgba = decompress_to(&jpeg, PixelFormat::Rgba)?;
    println!(
        "as RGBA: {} bytes ({} per pixel)",
        rgba.data.len(),
        rgba.pixel_format.bytes_per_pixel()
    );
    Ok(())
}
