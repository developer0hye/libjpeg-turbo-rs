use crate::common::error::Result;
use crate::common::types::{PixelFormat, Subsampling};
use crate::decode::pipeline::{Decoder, Image};
use crate::encode::pipeline as encoder;

/// Decompress a JPEG byte slice into an Image (default: RGB for color, Grayscale for gray).
pub fn decompress(data: &[u8]) -> Result<Image> {
    Decoder::decode(data)
}

/// Decompress a JPEG byte slice into an Image with the specified pixel format.
pub fn decompress_to(data: &[u8], format: PixelFormat) -> Result<Image> {
    Decoder::decode_to(data, format)
}

/// Decompress a JPEG in lenient mode — continue on errors, filling corrupt areas with gray.
/// The returned Image may have non-empty `warnings` if errors were encountered.
pub fn decompress_lenient(data: &[u8]) -> Result<Image> {
    let mut decoder = Decoder::new(data)?;
    decoder.set_lenient(true);
    decoder.decode_image()
}

/// Compress raw pixel data into a JPEG byte stream.
///
/// # Arguments
/// * `pixels` - Raw pixel data in the format specified by `pixel_format`
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pixel_format` - Pixel format of the input data
/// * `quality` - JPEG quality factor (1-100, where 100 is best quality)
/// * `subsampling` - Chroma subsampling mode
///
/// # Returns
/// A `Vec<u8>` containing the complete JPEG file data.
pub fn compress(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    encoder::compress(pixels, width, height, pixel_format, quality, subsampling)
}
