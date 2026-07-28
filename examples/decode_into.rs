//! Decode into a caller-owned, reusable buffer — no per-frame output
//! allocation on the standard paths (issue #354's API).
//!
//! ```sh
//! cargo run --example decode_into [path/to/image.jpg]
//! ```

use libjpeg_turbo_rs::{decompress_into, output_buffer_size, PixelFormat};

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

    // Size once, reuse across frames (e.g. a thumbnail farm).
    let size: usize = output_buffer_size(&jpeg, PixelFormat::Rgb)?;
    let mut buffer: Vec<u8> = vec![0u8; size];

    let info = decompress_into(&jpeg, PixelFormat::Rgb, &mut buffer)?;
    println!(
        "{path}: {}x{}, wrote {} bytes into a caller buffer of {}",
        info.width,
        info.height,
        info.bytes_written,
        buffer.len()
    );
    Ok(())
}
