use crate::common::error::Result;
use crate::common::types::{CropRegion, PixelFormat, Subsampling};
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

/// Decompress a cropped region of a JPEG.
///
/// Performs a full decode then extracts the specified region.
/// Coordinates that exceed image bounds are clamped.
pub fn decompress_cropped(data: &[u8], region: CropRegion) -> Result<Image> {
    let full = Decoder::decode(data)?;
    let bpp = full.pixel_format.bytes_per_pixel();

    let x = region.x.min(full.width);
    let y = region.y.min(full.height);
    let w = region.width.min(full.width.saturating_sub(x));
    let h = region.height.min(full.height.saturating_sub(y));

    let mut cropped_data = Vec::with_capacity(w * h * bpp);
    for row in y..y + h {
        let start = (row * full.width + x) * bpp;
        cropped_data.extend_from_slice(&full.data[start..start + w * bpp]);
    }

    Ok(Image {
        width: w,
        height: h,
        pixel_format: full.pixel_format,
        data: cropped_data,
        icc_profile: full.icc_profile,
        exif_data: full.exif_data,
        warnings: full.warnings,
    })
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

/// Compress with optimized Huffman tables (2-pass encoding).
///
/// Produces smaller output than `compress()` at the cost of an extra encoding pass.
pub fn compress_optimized(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    encoder::compress_optimized(pixels, width, height, pixel_format, quality, subsampling)
}
