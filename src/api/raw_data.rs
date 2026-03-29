/// Raw data encode/decode API.
///
/// Provides functions to encode JPEG from pre-downsampled component planes
/// and decode JPEG to raw component planes, bypassing color conversion
/// and chroma upsampling/downsampling. This matches libjpeg-turbo's
/// `jpeg_write_raw_data()` / `jpeg_read_raw_data()` functionality.
use crate::common::error::Result;
use crate::common::types::Subsampling;
use crate::decode::pipeline::Decoder;
use crate::encode::pipeline as encoder;

/// Decoded raw component plane data.
///
/// Contains separate planes for each component at their native
/// (potentially subsampled) resolution. For a 4:2:0 YCbCr JPEG,
/// the Y plane is at full resolution while Cb and Cr are at half
/// resolution in each dimension.
#[derive(Debug, Clone)]
pub struct RawImage {
    /// Component plane data. planes\[0\] is Y (or the single grayscale component),
    /// planes\[1\] is Cb, planes\[2\] is Cr.
    pub planes: Vec<Vec<u8>>,
    /// Width of each plane in samples (MCU-aligned).
    pub plane_widths: Vec<usize>,
    /// Height of each plane in rows (MCU-aligned).
    pub plane_heights: Vec<usize>,
    /// Original image width in pixels.
    pub width: usize,
    /// Original image height in pixels.
    pub height: usize,
    /// Number of color components (1 for grayscale, 3 for YCbCr).
    pub num_components: usize,
}

/// Encode JPEG from raw downsampled component data.
///
/// Each plane is a separate component at its native (subsampled) resolution.
/// This bypasses color conversion and downsampling — the caller provides
/// data already in the YCbCr color space at the correct subsampled dimensions.
///
/// # Arguments
/// * `planes` - Component plane data (Y, Cb, Cr for color; just Y for grayscale)
/// * `plane_widths` - Width of each plane in samples
/// * `plane_heights` - Height of each plane in rows
/// * `image_width` - Full image width in pixels
/// * `image_height` - Full image height in pixels
/// * `quality` - JPEG quality factor (1-100)
/// * `subsampling` - Chroma subsampling mode
///
/// # Errors
/// Returns error if plane counts or dimensions are inconsistent with the
/// subsampling mode, or if plane data is too small for the declared dimensions.
pub fn compress_raw(
    planes: &[&[u8]],
    plane_widths: &[usize],
    plane_heights: &[usize],
    image_width: usize,
    image_height: usize,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    encoder::compress_raw(
        planes,
        plane_widths,
        plane_heights,
        image_width,
        image_height,
        quality,
        subsampling,
    )
}

/// Decompress JPEG to raw downsampled component planes.
///
/// Returns separate planes for each component at their native resolution.
/// For subsampled JPEGs, chroma planes will be smaller than the luma plane.
/// No color conversion or upsampling is performed.
///
/// # Arguments
/// * `data` - Complete JPEG file data
///
/// # Returns
/// A `RawImage` containing the decoded component planes and their dimensions.
pub fn decompress_raw(data: &[u8]) -> Result<RawImage> {
    let decoder = Decoder::new(data)?;
    decoder.decode_raw()
}
