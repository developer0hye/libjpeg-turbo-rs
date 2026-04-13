//! Bridge crate connecting [`libjpeg-turbo-rs`] with the [`image`] ecosystem.
//!
//! Provides [`JpegDecoder`] and [`JpegEncoder`] that implement the
//! [`image::ImageDecoder`] and [`image::ImageEncoder`] traits respectively,
//! backed by the high-performance `libjpeg-turbo-rs` codec.
//!
//! # Example
//!
//! ```rust,no_run
//! use libjpeg_turbo_rs_image::JpegDecoder;
//! use image::ImageDecoder;
//! use std::fs;
//!
//! let data = fs::read("photo.jpg").unwrap();
//! let mut decoder = JpegDecoder::new(&data).unwrap();
//! let (width, height) = decoder.dimensions();
//! let color_type = decoder.color_type();
//! let mut buf = vec![0u8; decoder.total_bytes() as usize];
//! decoder.read_image(&mut buf).unwrap();
//! ```

use image::{ColorType, ExtendedColorType, ImageDecoder, ImageEncoder, ImageError, ImageResult};
use libjpeg_turbo_rs::{
    compress, decompress_to, JpegError, PixelFormat, ScanlineDecoder, Subsampling,
};
use std::io::Write;

// ===== Error conversion =====

/// Convert a [`JpegError`] into an [`ImageError`].
fn jpeg_error_to_image_error(err: JpegError) -> ImageError {
    ImageError::Decoding(image::error::DecodingError::new(
        image::error::ImageFormatHint::Name("JPEG".to_string()),
        err,
    ))
}

/// Convert a [`JpegError`] into an [`ImageError`] for encoding context.
fn jpeg_encode_error_to_image_error(err: JpegError) -> ImageError {
    ImageError::Encoding(image::error::EncodingError::new(
        image::error::ImageFormatHint::Name("JPEG".to_string()),
        err,
    ))
}

// ===== ColorType mapping =====

/// Map our [`PixelFormat`] to an `image::ColorType`.
fn pixel_format_to_color_type(fmt: PixelFormat) -> Option<ColorType> {
    match fmt {
        PixelFormat::Grayscale => Some(ColorType::L8),
        PixelFormat::Rgb => Some(ColorType::Rgb8),
        PixelFormat::Rgba => Some(ColorType::Rgba8),
        // image crate has no native BGR/BGRA/CMYK color types — callers
        // should request RGB/RGBA explicitly via decompress_to().
        _ => None,
    }
}

/// Map an `image::ExtendedColorType` to our [`PixelFormat`] for encoding.
fn extended_color_type_to_pixel_format(color_type: ExtendedColorType) -> Option<PixelFormat> {
    match color_type {
        ExtendedColorType::L8 => Some(PixelFormat::Grayscale),
        ExtendedColorType::Rgb8 => Some(PixelFormat::Rgb),
        ExtendedColorType::Rgba8 => Some(PixelFormat::Rgba),
        _ => None,
    }
}

// ===== JpegDecoder =====

/// JPEG decoder backed by `libjpeg-turbo-rs`, implementing [`image::ImageDecoder`].
///
/// Decodes the full image eagerly on construction and stores the decoded pixel
/// data in memory. This matches the contract of `ImageDecoder` which requires
/// all metadata (`dimensions`, `color_type`) to be available before
/// `read_image` is called.
pub struct JpegDecoder {
    width: u32,
    height: u32,
    color_type: ColorType,
    pixels: Vec<u8>,
    icc_profile: Option<Vec<u8>>,
}

impl JpegDecoder {
    /// Create a new decoder from raw JPEG bytes.
    ///
    /// Decodes the image eagerly. Returns an error if the data is not valid
    /// JPEG or if the output color type cannot be mapped to an `image` type.
    pub fn new(data: &[u8]) -> ImageResult<Self> {
        // Peek at the JPEG header to determine the source color space so we
        // can choose the appropriate output pixel format. Grayscale JPEGs
        // (1 component) must decode to L8, not RGB8.
        let header_reader = ScanlineDecoder::new(data).map_err(jpeg_error_to_image_error)?;
        let jpeg_color_space = header_reader.header().components.len();
        drop(header_reader);

        let output_format = if jpeg_color_space == 1 {
            PixelFormat::Grayscale
        } else {
            PixelFormat::Rgb
        };

        let decoded = decompress_to(data, output_format).map_err(jpeg_error_to_image_error)?;

        let color_type = pixel_format_to_color_type(decoded.pixel_format).ok_or_else(|| {
            ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
                image::error::ImageFormatHint::Name("JPEG".to_string()),
                image::error::UnsupportedErrorKind::Color(ExtendedColorType::Unknown(
                    decoded.pixel_format.bytes_per_pixel() as u8 * 8,
                )),
            ))
        })?;

        Ok(Self {
            width: decoded.width as u32,
            height: decoded.height as u32,
            color_type,
            pixels: decoded.data,
            icc_profile: decoded.icc_profile,
        })
    }

    /// Create a decoder that decodes to a specific pixel format.
    ///
    /// Use this when you need a format not representable as a standard
    /// `image::ColorType` (e.g., BGR, BGRA).
    pub fn new_with_format(data: &[u8], format: PixelFormat) -> ImageResult<Self> {
        let decoded = decompress_to(data, format).map_err(jpeg_error_to_image_error)?;

        let color_type = pixel_format_to_color_type(decoded.pixel_format).ok_or_else(|| {
            ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
                image::error::ImageFormatHint::Name("JPEG".to_string()),
                image::error::UnsupportedErrorKind::Color(ExtendedColorType::Unknown(
                    decoded.pixel_format.bytes_per_pixel() as u8 * 8,
                )),
            ))
        })?;

        Ok(Self {
            width: decoded.width as u32,
            height: decoded.height as u32,
            color_type,
            pixels: decoded.data,
            icc_profile: decoded.icc_profile,
        })
    }
}

impl ImageDecoder for JpegDecoder {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn color_type(&self) -> ColorType {
        self.color_type
    }

    fn read_image(self, buf: &mut [u8]) -> ImageResult<()> {
        let expected = self.total_bytes() as usize;
        if buf.len() < expected {
            return Err(ImageError::Parameter(
                image::error::ParameterError::from_kind(
                    image::error::ParameterErrorKind::DimensionMismatch,
                ),
            ));
        }
        buf[..expected].copy_from_slice(&self.pixels);
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        Ok(self.icc_profile.take())
    }
}

// ===== JpegEncoder =====

/// JPEG encoder backed by `libjpeg-turbo-rs`, implementing [`image::ImageEncoder`].
///
/// Writes compressed JPEG output to any `Write` sink. Quality defaults to 75
/// and subsampling to 4:2:0, matching common JPEG encoder defaults.
pub struct JpegEncoder<W: Write> {
    writer: W,
    /// JPEG quality (1–100). Default: 75.
    quality: u8,
    /// Chroma subsampling. Default: 4:2:0.
    subsampling: Subsampling,
}

impl<W: Write> JpegEncoder<W> {
    /// Create a new encoder that writes to `writer` with default quality 75.
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            quality: 75,
            subsampling: Subsampling::S420,
        }
    }

    /// Create a new encoder with the specified quality (1–100).
    pub fn new_with_quality(writer: W, quality: u8) -> Self {
        Self {
            writer,
            quality,
            subsampling: Subsampling::S420,
        }
    }

    /// Set the JPEG quality (1–100).
    pub fn set_quality(&mut self, quality: u8) {
        self.quality = quality;
    }

    /// Set the chroma subsampling mode.
    pub fn set_subsampling(&mut self, subsampling: Subsampling) {
        self.subsampling = subsampling;
    }
}

impl<W: Write> ImageEncoder for JpegEncoder<W> {
    fn write_image(
        mut self,
        buf: &[u8],
        width: u32,
        height: u32,
        color_type: ExtendedColorType,
    ) -> ImageResult<()> {
        let pixel_format = extended_color_type_to_pixel_format(color_type).ok_or_else(|| {
            ImageError::Unsupported(image::error::UnsupportedError::from_format_and_kind(
                image::error::ImageFormatHint::Name("JPEG".to_string()),
                image::error::UnsupportedErrorKind::Color(color_type),
            ))
        })?;

        let jpeg_data = compress(
            buf,
            width as usize,
            height as usize,
            pixel_format,
            self.quality,
            self.subsampling,
        )
        .map_err(jpeg_encode_error_to_image_error)?;

        self.writer
            .write_all(&jpeg_data)
            .map_err(ImageError::IoError)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageDecoder;

    /// Path to a small JPEG fixture available in the workspace fuzz corpus.
    const FIXTURE_RGB: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fuzz/corpus/fuzz_decompress/photo_64x64_420.jpg"
    );
    const FIXTURE_GRAY: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fuzz/corpus/fuzz_decompress/gray_8x8.jpg"
    );

    fn read_fixture(path: &str) -> Vec<u8> {
        std::fs::read(path).unwrap_or_else(|_| panic!("missing fixture: {path}"))
    }

    /// Decode via bridge matches direct libjpeg-turbo-rs decode.
    #[test]
    fn decode_rgb_matches_direct() {
        let data = read_fixture(FIXTURE_RGB);

        // Decode via bridge
        let decoder = JpegDecoder::new(&data).expect("JpegDecoder::new failed");
        let (width, height) = decoder.dimensions();
        let color_type = decoder.color_type();
        let total = decoder.total_bytes() as usize;
        let mut bridge_pixels = vec![0u8; total];
        decoder
            .read_image(&mut bridge_pixels)
            .expect("read_image failed");

        // Decode directly
        let direct = decompress_to(&data, PixelFormat::Rgb).expect("direct decode failed");

        assert_eq!(width, direct.width as u32, "width mismatch");
        assert_eq!(height, direct.height as u32, "height mismatch");
        assert_eq!(color_type, ColorType::Rgb8, "unexpected color type");
        assert_eq!(bridge_pixels, direct.data, "pixel data mismatch");
    }

    /// Decode grayscale JPEG returns L8 color type.
    #[test]
    fn decode_grayscale_returns_l8() {
        let data = read_fixture(FIXTURE_GRAY);
        let decoder = JpegDecoder::new(&data).expect("JpegDecoder::new failed");
        assert_eq!(
            decoder.color_type(),
            ColorType::L8,
            "expected L8 for grayscale JPEG"
        );
    }

    /// Encode then decode round-trip preserves dimensions.
    #[test]
    fn encode_decode_roundtrip_preserves_dimensions() {
        let data = read_fixture(FIXTURE_RGB);
        let decoded = decompress_to(&data, PixelFormat::Rgb).expect("decode failed");
        let width = decoded.width as u32;
        let height = decoded.height as u32;

        // Encode via bridge
        let mut jpeg_out: Vec<u8> = Vec::new();
        let encoder = JpegEncoder::new(&mut jpeg_out);
        encoder
            .write_image(&decoded.data, width, height, ExtendedColorType::Rgb8)
            .expect("encode failed");

        assert!(!jpeg_out.is_empty(), "encoded JPEG must not be empty");

        // Decode the re-encoded JPEG
        let re_decoded = JpegDecoder::new(&jpeg_out).expect("re-decode failed");
        assert_eq!(
            re_decoded.dimensions(),
            (width, height),
            "dimensions changed after roundtrip"
        );
        assert_eq!(re_decoded.color_type(), ColorType::Rgb8);
    }

    /// Encode with custom quality produces a smaller file than quality 95.
    #[test]
    fn encode_quality_affects_file_size() {
        let data = read_fixture(FIXTURE_RGB);
        let decoded = decompress_to(&data, PixelFormat::Rgb).expect("decode failed");
        let width = decoded.width as u32;
        let height = decoded.height as u32;

        let encode_at_quality = |q: u8| -> Vec<u8> {
            let mut out: Vec<u8> = Vec::new();
            JpegEncoder::new_with_quality(&mut out, q)
                .write_image(&decoded.data, width, height, ExtendedColorType::Rgb8)
                .expect("encode failed");
            out
        };

        let low_quality = encode_at_quality(10);
        let high_quality = encode_at_quality(95);

        assert!(
            low_quality.len() < high_quality.len(),
            "low quality ({} bytes) should be smaller than high quality ({} bytes)",
            low_quality.len(),
            high_quality.len()
        );
    }

    /// JpegDecoder::icc_profile() returns the embedded ICC profile if present.
    #[test]
    fn icc_profile_is_passed_through() {
        // Encode a JPEG with an ICC profile using the main library, then decode
        // via our bridge and verify the ICC profile is returned.
        let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
        let icc_data: Vec<u8> = b"fake-icc-profile-data".to_vec();

        let jpeg_bytes = libjpeg_turbo_rs::Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
            .quality(75)
            .icc_profile(&icc_data)
            .encode()
            .expect("encode with ICC failed");

        let mut decoder = JpegDecoder::new(&jpeg_bytes).expect("decode failed");
        let profile = decoder.icc_profile().expect("icc_profile() returned Err");

        // ICC profile must be present and match what we embedded.
        assert!(profile.is_some(), "ICC profile not returned from decoder");
        assert_eq!(profile.unwrap(), icc_data, "ICC profile data mismatch");
    }

    /// Encoding an unsupported color type returns an UnsupportedError.
    #[test]
    fn encode_unsupported_color_type_returns_error() {
        let pixels: Vec<u8> = vec![0u8; 4 * 4 * 2]; // LA8: 2 channels
        let mut out: Vec<u8> = Vec::new();
        let result = JpegEncoder::new(&mut out).write_image(
            &pixels,
            4,
            4,
            ExtendedColorType::La8, // Not supported by JPEG
        );
        assert!(
            result.is_err(),
            "encoding LA8 should fail with UnsupportedError"
        );
        assert!(
            matches!(result.unwrap_err(), ImageError::Unsupported(_)),
            "expected UnsupportedError for LA8"
        );
    }
}
