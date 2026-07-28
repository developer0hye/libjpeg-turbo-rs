//! Read dimensions, EXIF orientation, and metadata without decoding
//! any pixels — `Decoder::new` parses markers only.
//!
//! ```sh
//! cargo run --example probe_header [path/to/image.jpg]
//! ```

use libjpeg_turbo_rs::Decoder;

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

    let decoder = Decoder::new(&jpeg)?;
    let header = decoder.header();
    println!("{path}:");
    println!(
        "  {}x{} px, {} component(s), precision {}",
        header.width,
        header.height,
        header.components.len(),
        header.precision
    );
    println!(
        "  progressive: {}, lossless: {}",
        header.is_progressive, header.is_lossless
    );
    match decoder.exif_orientation() {
        Some(o) => println!("  EXIF orientation: {o} (1 = upright)"),
        None => println!("  EXIF orientation: none"),
    }
    Ok(())
}
