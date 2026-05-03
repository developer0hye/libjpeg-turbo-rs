use crate::common::error::Result;
use crate::common::types::{CropRegion, DctMethod, DensityInfo, PixelFormat, Subsampling};
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

    // Compute iMCU column alignment matching C jpeg_crop_scanline():
    //   align = min_DCT_scaled_size * max_h_samp_factor
    // For standard (no scaling): min_DCT_scaled_size = 8 (DCTSIZE).
    let max_h_samp: usize = header
        .components
        .iter()
        .map(|c| c.horizontal_sampling as usize)
        .max()
        .unwrap_or(1);
    let align: usize = if header.components.len() == 1 {
        8 // single-component: align to DCTSIZE only
    } else {
        8 * max_h_samp // multi-component: align to iMCU column width
    };

    let input_x: usize = region.x.min(img_w);
    let y: usize = region.y.min(img_h);

    // Snap x down to nearest iMCU boundary (C: xoffset = (input_xoffset / align) * align)
    let x: usize = (input_x / align) * align;
    // Extend width to keep the right edge at the originally requested position
    // (C: width = width + input_xoffset - xoffset)
    let w: usize = (region.width + input_x - x).min(img_w.saturating_sub(x));
    let h: usize = region.height.min(img_h.saturating_sub(y));

    if w == 0 || h == 0 {
        return Ok(Image {
            width: w,
            height: h,
            pixel_format: PixelFormat::Rgb,
            precision: 8,
            data: Vec::new(),
            icc_profile: None,
            exif_data: None,
            comment: None,
            density: DensityInfo::default(),
            saved_markers: Vec::new(),
            warnings: Vec::new(),
        });
    }

    // Disable merged upsample for crop decode: C djpeg uses fancy merged
    // upsample by default, but Rust's merged path is box-filter. The fancy
    // row-streaming H2V2 path matches C's behavior at crop boundaries.
    decoder.merged_upsample = false;

    // Set crop region: decode_image() handles both horizontal (MCU-aligned
    // crop width matching C jpeg_crop_scanline) and vertical (exact row slice)
    // cropping internally.
    decoder.set_crop_region(x, y, w, h);
    decoder.decode_image()
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
    encoder::compress(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        crate::common::types::DctMethod::IsLow,
    )
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
    encoder::compress_optimized(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        0,
        crate::common::types::DctMethod::IsLow,
        0,
    )
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
    encoder::compress_progressive(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        DctMethod::IsLow,
    )
}

/// Compress with optional ICC profile and/or EXIF metadata embedded.
#[allow(clippy::too_many_arguments)]
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

/// Compress as lossless JPEG (SOF3).
///
/// Uses predictor 1 (left) with no point transform. Produces exact
/// pixel-identical output when decoded. Currently supports grayscale only.
pub fn compress_lossless(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    encoder::compress_lossless(pixels, width, height, pixel_format)
}

/// Compress as lossless JPEG (SOF3) with configurable predictor and point transform.
///
/// Supports grayscale (1-component) and RGB (3-component interleaved via YCbCr).
///
/// # Arguments
/// * `predictor` - Predictor selection value (1-7)
/// * `point_transform` - Point transform value (0-15)
pub fn compress_lossless_extended(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    encoder::compress_lossless_extended(
        pixels,
        width,
        height,
        pixel_format,
        predictor,
        point_transform,
        0, // no restart interval for high-level API
    )
}

/// Like `compress_lossless_extended` but with an explicit sample precision
/// (2..=8). The precision field controls the SOF3 marker and the lossless
/// predictor arithmetic; source samples remain `u8`.
#[allow(clippy::too_many_arguments)]
pub fn compress_lossless_extended_precision(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
    precision: u8,
) -> Result<Vec<u8>> {
    encoder::compress_lossless_extended_precision(
        pixels,
        width,
        height,
        pixel_format,
        predictor,
        point_transform,
        restart_interval,
        precision,
    )
}

/// Compress with arithmetic progressive encoding (SOF10).
///
/// Combines progressive multi-scan encoding with arithmetic entropy coding.
/// Produces a progressive JPEG that renders incrementally and uses arithmetic
/// coding for better compression than Huffman-based progressive.
pub fn compress_arithmetic_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    encoder::compress_arithmetic_progressive(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        DctMethod::IsLow,
    )
}

/// Compress as lossless JPEG with arithmetic entropy coding (SOF11).
///
/// Uses predictor-based lossless encoding with arithmetic (QM-coder) entropy
/// coding instead of Huffman. Produces exact pixel-identical output when decoded.
pub fn compress_lossless_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    encoder::compress_lossless_arithmetic(
        pixels,
        width,
        height,
        pixel_format,
        predictor,
        point_transform,
    )
}

/// Compress into a pre-allocated buffer. Returns the number of bytes written.
///
/// This avoids allocating a new `Vec<u8>` for the output. If the buffer is
/// too small to hold the compressed JPEG, returns `JpegError::BufferTooSmall`.
/// Matches libjpeg-turbo's `TJPARAM_NOREALLOC` behavior.
pub fn compress_into(
    buf: &mut [u8],
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
) -> Result<usize> {
    let jpeg_data: Vec<u8> = encoder::compress(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        crate::common::types::DctMethod::IsLow,
    )?;
    if jpeg_data.len() > buf.len() {
        return Err(crate::common::error::JpegError::BufferTooSmall {
            need: jpeg_data.len(),
            got: buf.len(),
        });
    }
    buf[..jpeg_data.len()].copy_from_slice(&jpeg_data);
    Ok(jpeg_data.len())
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
    encoder::compress_arithmetic(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        DctMethod::IsLow,
        0,
    )
}
