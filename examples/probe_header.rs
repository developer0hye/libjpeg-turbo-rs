//! Read dimensions, coding mode, subsampling, colorspace, EXIF
//! orientation and metadata presence without decoding any pixels —
//! `probe` parses markers only.
//!
//! ```sh
//! cargo run --example probe_header [path/to/image.jpg]
//! ```

use libjpeg_turbo_rs::{probe, JpegInfo};

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

    // One call gets everything (issue #386). For accessors beyond
    // JpegInfo — JFIF version, saved markers, the raw ICC/EXIF/XMP
    // payloads rather than a presence flag — construct a `Decoder` and
    // use its getters; it too never decodes pixels.
    let info: JpegInfo = probe(&jpeg)?;
    println!("{path}:");
    println!(
        "  {}x{} px, {} component(s), precision {}",
        info.width, info.height, info.components, info.precision
    );
    println!(
        "  progressive: {}, lossless: {}, arithmetic: {}",
        info.progressive, info.lossless, info.arithmetic
    );
    println!(
        "  subsampling: {:?}, colorspace: {:?}",
        info.subsampling, info.color_space
    );
    match info.exif_orientation {
        Some(o) => println!("  EXIF orientation: {o} (1 = upright)"),
        None => println!("  EXIF orientation: none"),
    }
    println!(
        "  metadata: exif={} icc={} xmp={} iptc={}",
        info.has_exif, info.has_icc, info.has_xmp, info.has_iptc
    );
    Ok(())
}
