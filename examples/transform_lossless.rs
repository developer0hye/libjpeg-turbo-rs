//! Lossless (DCT-domain) rotation: no decode, no re-encode loss.
//!
//! ```sh
//! cargo run --example transform_lossless [path/to/image.jpg]
//! ```

use libjpeg_turbo_rs::{transform, Decoder, TransformOp};

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

    let rotated: Vec<u8> = transform(&jpeg, TransformOp::Rot90)?;
    let header = Decoder::new(&rotated)?.header().clone();
    println!(
        "{path}: rotated 90 degrees losslessly -> {}x{} ({} bytes)",
        header.width,
        header.height,
        rotated.len()
    );

    // Tip: map an EXIF orientation tag to the right op with
    // TransformOp::from_exif_orientation (see the docs for the
    // iMCU-alignment and marker-copy caveats).
    let out = std::env::temp_dir().join("libjpeg_turbo_rs_rot90.jpg");
    std::fs::write(&out, &rotated)?;
    println!("wrote {}", out.display());
    Ok(())
}
