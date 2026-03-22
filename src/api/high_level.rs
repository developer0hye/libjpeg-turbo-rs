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
/// Uses MCU-level IDCT skip to avoid unnecessary computation on rows
/// outside the crop region, then extracts the exact pixel region.
/// Coordinates that exceed image bounds are clamped.
pub fn decompress_cropped(data: &[u8], region: CropRegion) -> Result<Image> {
    let mut decoder = Decoder::new(data)?;
    let header = decoder.header();
    let img_w = header.width as usize;
    let img_h = header.height as usize;

    let x = region.x.min(img_w);
    let y = region.y.min(img_h);
    let w = region.width.min(img_w.saturating_sub(x));
    let h = region.height.min(img_h.saturating_sub(y));

    if w == 0 || h == 0 {
        return Ok(Image {
            width: w,
            height: h,
            pixel_format: PixelFormat::Rgb,
            data: Vec::new(),
            icc_profile: None,
            exif_data: None,
            warnings: Vec::new(),
        });
    }

    // Set crop region so the decoder skips IDCT on out-of-range MCU rows
    decoder.set_crop_region(x, y, w, h);
    let full = decoder.decode_image()?;
    let bpp = full.pixel_format.bytes_per_pixel();

    // Extract the exact pixel region from the full-width output
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

/// Compress as progressive JPEG (SOF2, multi-scan).
///
/// Produces a progressive JPEG that renders incrementally during download.
pub fn compress_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    encoder::compress_progressive(pixels, width, height, pixel_format, quality, subsampling)
}

/// Compress with optional ICC profile and/or EXIF metadata embedded.
pub fn compress_with_metadata(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    encoder::compress_with_metadata(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        icc_profile,
        exif_data,
    )
}

/// Compress with arithmetic entropy coding (SOF9).
///
/// Uses QM-coder binary arithmetic coding instead of Huffman coding.
pub fn compress_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    encoder::compress_arithmetic(pixels, width, height, pixel_format, quality, subsampling)
}
