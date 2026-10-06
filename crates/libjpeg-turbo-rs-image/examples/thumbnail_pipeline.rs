//! Migrating an `image` thumbnail pipeline to this backend:
//! decode → apply EXIF orientation → resize → encode.
//!
//! ```sh
//! cargo run -p libjpeg-turbo-rs-image --example thumbnail_pipeline [in.jpg] [out.jpg]
//! ```
//!
//! The only lines that change from a stock `image` pipeline are marked
//! `// was:` — construction of the decoder and of the encoder. Everything in
//! between is plain `image` API, so the rest of an application keeps working.
//!
//! With no input argument a JPEG is synthesized in-process, so the example is
//! self-contained even in the published crate (which ships no fixtures).

use std::error::Error;
use std::fs;

use image::imageops::FilterType;
use image::{DynamicImage, ImageDecoder, Limits};
use libjpeg_turbo_rs_image::{JpegDecoder, JpegEncoder};

const THUMBNAIL_EDGE: u32 = 256;

fn synthesize_jpeg() -> Vec<u8> {
    let (width, height): (usize, usize) = (1200, 800);
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
        90,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("synthesized encode cannot fail")
}

fn make_thumbnail(jpeg: Vec<u8>) -> Result<Vec<u8>, Box<dyn Error>> {
    // was: let mut decoder = image::codecs::jpeg::JpegDecoder::new(Cursor::new(&jpeg))?;
    let mut decoder: JpegDecoder = JpegDecoder::from_vec(jpeg)?;

    // An upload service bounds what it will decode before decoding it. The
    // refusal is an `ImageError::Limits`, raised here, before any pixel
    // buffer exists.
    let mut limits: Limits = Limits::default();
    limits.max_image_width = Some(12_000);
    limits.max_image_height = Some(12_000);
    decoder.set_limits(limits)?;

    // Orientation is reported, not applied, exactly as with `image`'s own
    // decoder — apply it once, here.
    let orientation = decoder.orientation()?;
    let mut image: DynamicImage = DynamicImage::from_decoder(decoder)?;
    image.apply_orientation(orientation);

    let thumbnail: DynamicImage =
        image.resize(THUMBNAIL_EDGE, THUMBNAIL_EDGE, FilterType::Triangle);

    let mut output: Vec<u8> = Vec::new();
    // was: let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 80);
    let encoder: JpegEncoder<&mut Vec<u8>> = JpegEncoder::new_with_quality(&mut output, 80);
    thumbnail.to_rgb8().write_with_encoder(encoder)?;
    Ok(output)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args().skip(1);
    let (label, jpeg): (String, Vec<u8>) = match arguments.next() {
        Some(path) => {
            let bytes: Vec<u8> = fs::read(&path)?;
            (path, bytes)
        }
        None => ("<synthesized 1200x800>".to_string(), synthesize_jpeg()),
    };
    let thumbnail: Vec<u8> = make_thumbnail(jpeg)?;
    let decoded: DynamicImage = DynamicImage::from_decoder(JpegDecoder::new(&thumbnail)?)?;
    println!(
        "{label} -> {}x{} thumbnail, {} bytes",
        decoded.width(),
        decoded.height(),
        thumbnail.len()
    );
    if let Some(path) = arguments.next() {
        fs::write(&path, &thumbnail)?;
        println!("wrote {path}");
    }
    Ok(())
}
