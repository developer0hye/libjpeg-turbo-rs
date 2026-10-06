//! Bridge crate connecting [`libjpeg-turbo-rs`] with the [`image`] ecosystem.
//!
//! Provides [`JpegDecoder`] and [`JpegEncoder`] that implement the
//! [`image::ImageDecoder`] and [`image::ImageEncoder`] traits respectively,
//! backed by the `libjpeg-turbo-rs` codec.
//!
//! # Which `image` entry points use this backend
//!
//! Only the ones you construct explicitly. This crate does not register
//! itself with `image`: `image::open`, `ImageReader::decode`,
//! `load_from_memory` and `DynamicImage::save` keep using `image`'s own JPEG
//! codec. To decode through this crate, build a [`JpegDecoder`] and hand it
//! to [`image::DynamicImage::from_decoder`] (or call
//! [`ImageDecoder::read_image`] yourself); to encode, pass a [`JpegEncoder`]
//! to [`image::DynamicImage::write_with_encoder`].
//!
//! # Example
//!
//! ```rust,no_run
//! use libjpeg_turbo_rs_image::JpegDecoder;
//! use image::{DynamicImage, ImageDecoder};
//! use std::fs;
//!
//! let data = fs::read("photo.jpg").unwrap();
//! let mut decoder = JpegDecoder::new(&data).unwrap();
//! let orientation = decoder.orientation().unwrap();
//! let mut image = DynamicImage::from_decoder(decoder).unwrap();
//! image.apply_orientation(orientation);
//! ```

use image::error::{
    DecodingError, EncodingError, ImageFormatHint, LimitError, LimitErrorKind, ParameterError,
    ParameterErrorKind, UnsupportedError, UnsupportedErrorKind,
};
use image::metadata::Orientation;
use image::{
    ColorType, ExtendedColorType, ImageDecoder, ImageEncoder, ImageError, ImageFormat, ImageResult,
    Limits,
};
use libjpeg_turbo_rs::{
    compress, DecodeLimits, Decoder, FrameHeader, JpegError, MarkerSaveConfig, PixelFormat,
    SavedMarker, Subsampling,
};
use std::io::Write;

const APP1: u8 = 0xE1;
const APP13: u8 = 0xED;
const EXIF_SIGNATURE: &[u8] = b"Exif\0\0";
const XMP_SIGNATURE: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const PHOTOSHOP_SIGNATURE: &[u8] = b"Photoshop 3.0\0";

// ===== Error conversion =====

fn jpeg_format_hint() -> ImageFormatHint {
    ImageFormatHint::Exact(ImageFormat::Jpeg)
}

/// The `ImageError` category a codec error belongs to, shared by decode and
/// encode so both directions classify a refusal the same way.
///
/// Limit and allocation refusals are `ImageError::Limits` — the category
/// `image` itself uses when `Limits` stop a decode — rather than a decoding
/// failure, so a caller can tell "this file is too big for the budget I set"
/// from "this file is broken".
fn classify_error(err: JpegError, is_encoding: bool) -> ImageError {
    match err {
        JpegError::LimitExceeded { what, .. } => {
            // The three frame-geometry refusals `DecodeLimits::check_frame`
            // raises are dimension errors; every other limit (memory
            // estimate, scan count, marker and ICC chunk lists) bounds a
            // resource, which `image` calls insufficient memory.
            let kind: LimitErrorKind = match what {
                "image width" | "image height" | "total pixels" => LimitErrorKind::DimensionError,
                _ => LimitErrorKind::InsufficientMemory,
            };
            ImageError::Limits(LimitError::from_kind(kind))
        }
        JpegError::AllocationFailed { .. } => {
            ImageError::Limits(LimitError::from_kind(LimitErrorKind::InsufficientMemory))
        }
        JpegError::Unsupported(feature) => {
            ImageError::Unsupported(UnsupportedError::from_format_and_kind(
                jpeg_format_hint(),
                UnsupportedErrorKind::GenericFeature(feature),
            ))
        }
        JpegError::BufferTooSmall { .. } => ImageError::Parameter(ParameterError::from_kind(
            ParameterErrorKind::DimensionMismatch,
        )),
        JpegError::Io(io_error) => ImageError::IoError(io_error),
        other if is_encoding => ImageError::Encoding(EncodingError::new(jpeg_format_hint(), other)),
        other => ImageError::Decoding(DecodingError::new(jpeg_format_hint(), other)),
    }
}

fn decode_error(err: JpegError) -> ImageError {
    classify_error(err, false)
}

fn encode_error(err: JpegError) -> ImageError {
    classify_error(err, true)
}

// ===== ColorType mapping =====

/// The `image::ColorType` a decode to `format` produces, if `image` has one.
fn pixel_format_to_color_type(format: PixelFormat) -> Option<ColorType> {
    match format {
        PixelFormat::Grayscale => Some(ColorType::L8),
        PixelFormat::Rgb => Some(ColorType::Rgb8),
        PixelFormat::Rgba => Some(ColorType::Rgba8),
        // image has no BGR/BGRA/ARGB/CMYK/RGB565 color types, and
        // `ImageDecoder::read_image` must fill a buffer of `color_type()`.
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
/// Construction parses the headers only: dimensions, color type and metadata
/// are known, and no pixel has been decoded or allocated. The pixel decode
/// runs in [`ImageDecoder::read_image`]. For 8-bit grayscale and
/// three-component (YCbCr/RGB) streams, baseline or progressive, it decodes
/// straight into the caller's buffer, so no second decoded copy of the image
/// exists. Four-component (CMYK/YCCK), 12-bit and lossless streams are still
/// staged by the core decoder in a full-size buffer and copied — tracked as
/// P4-213. What the decoder holds between calls is the compressed stream: [`JpegDecoder::new`] copies it once (it
/// takes a borrowed slice, and `read_image(self, ..)` must own what it decodes
/// from); [`JpegDecoder::from_vec`] takes an owned buffer and copies nothing.
///
/// Decoding still allocates working memory — component planes for upsampling
/// and colour conversion, and a coefficient buffer for progressive streams —
/// so it is not allocation-free. [`ImageDecoder::set_limits`] bounds the
/// core's *estimate* of that memory together with the output size; the
/// estimate does not count the staging buffer of the paths above, so
/// `max_alloc` is non-strict there.
///
/// # Color types
///
/// | JPEG                       | `color_type()` | `original_color_type()` |
/// |----------------------------|----------------|-------------------------|
/// | 1 component (grayscale)    | `L8`           | `L8`                    |
/// | 3 components (YCbCr / RGB) | `Rgb8`         | `Rgb8`                  |
/// | 4 components (CMYK / YCCK) | `Rgb8`         | `Cmyk8`                 |
///
/// [`JpegDecoder::new_with_format`] selects `Rgba8` (or forces `L8`/`Rgb8`).
///
/// # Differences from `image`'s built-in JPEG decoder
///
/// * Corrupt and truncated streams are errors by default, as in C
///   libjpeg-turbo with `-strict`; `image`'s built-in decoder fills what it
///   cannot decode. [`JpegDecoder::set_lenient`] opts into the lenient
///   behaviour. Construction parses headers only in both, so a stream whose
///   entropy data is corrupt constructs and fails in `read_image`.
/// * `original_color_type()` reports `Cmyk8` for a four-component stream,
///   where the built-in decoder reports the converted `Rgb8`.
///
/// EXIF, XMP, IPTC and ICC bytes and the orientation are the same as the
/// built-in decoder's for the same file, including its choice of the *last*
/// segment when one repeats and its omission of Extended XMP.
pub struct JpegDecoder {
    input: Vec<u8>,
    width: u32,
    height: u32,
    output_format: PixelFormat,
    color_type: ColorType,
    original_color_type: ExtendedColorType,
    limits: DecodeLimits,
    lenient: bool,
}

impl JpegDecoder {
    /// Create a decoder from borrowed JPEG bytes, copying them once.
    ///
    /// Parses the headers only. Grayscale JPEGs decode to `L8`, everything
    /// else to `Rgb8`.
    pub fn new(data: &[u8]) -> ImageResult<Self> {
        Self::from_vec(copy_input(data)?)
    }

    /// Create a decoder that takes ownership of the JPEG bytes (no copy).
    ///
    /// Parses the headers only. Grayscale JPEGs decode to `L8`, everything
    /// else to `Rgb8`.
    pub fn from_vec(data: Vec<u8>) -> ImageResult<Self> {
        Self::with_output_format(data, None)
    }

    /// Create a decoder that decodes to a specific pixel format.
    ///
    /// `format` must be one `image` has a color type for: `Grayscale` (`L8`),
    /// `Rgb` (`Rgb8`) or `Rgba` (`Rgba8`). Any other format (BGR, BGRA,
    /// CMYK, ...) is refused here with `ImageError::Unsupported`, because
    /// `read_image` must fill a buffer laid out as `color_type()`; decode those
    /// with `libjpeg_turbo_rs::Decoder` directly.
    pub fn new_with_format(data: &[u8], format: PixelFormat) -> ImageResult<Self> {
        Self::with_output_format(copy_input(data)?, Some(format))
    }

    /// One header parse for everything construction needs. `None` picks the
    /// default output: `L8` for one component, `Rgb8` otherwise.
    fn with_output_format(
        input: Vec<u8>,
        requested_format: Option<PixelFormat>,
    ) -> ImageResult<Self> {
        let decoder: Decoder<'_> = Decoder::new(&input).map_err(decode_error)?;
        let header: &FrameHeader = decoder.header();
        let component_count: usize = header.components.len();
        let output_format: PixelFormat = requested_format.unwrap_or(if component_count == 1 {
            PixelFormat::Grayscale
        } else {
            PixelFormat::Rgb
        });
        let color_type: ColorType = pixel_format_to_color_type(output_format).ok_or_else(|| {
            ImageError::Unsupported(UnsupportedError::from_format_and_kind(
                jpeg_format_hint(),
                UnsupportedErrorKind::Color(ExtendedColorType::Unknown(
                    (output_format.bytes_per_pixel() * 8) as u8,
                )),
            ))
        })?;
        // The core has no CMYK/YCCK -> grayscale conversion; say so now rather
        // than after the caller has allocated a destination.
        if component_count == 4 && output_format == PixelFormat::Grayscale {
            return Err(ImageError::Unsupported(
                UnsupportedError::from_format_and_kind(
                    jpeg_format_hint(),
                    UnsupportedErrorKind::Color(ExtendedColorType::L8),
                ),
            ));
        }
        let original_color_type: ExtendedColorType = match component_count {
            1 => ExtendedColorType::L8,
            4 => ExtendedColorType::Cmyk8,
            _ => ExtendedColorType::Rgb8,
        };
        let (width, height): (u32, u32) = (u32::from(header.width), u32::from(header.height));
        drop(decoder);
        Ok(Self {
            input,
            width,
            height,
            output_format,
            color_type,
            original_color_type,
            limits: DecodeLimits::default(),
            lenient: false,
        })
    }

    /// Decode corrupt or truncated streams best-effort instead of failing.
    ///
    /// Off by default. With it on, entropy-coded data that cannot be decoded
    /// is filled rather than reported, which is what `image`'s built-in JPEG
    /// decoder does unconditionally.
    pub fn set_lenient(&mut self, lenient: bool) {
        self.lenient = lenient;
    }

    /// A header-parsed core decoder over the stored input, with this
    /// decoder's limits and options applied.
    fn core_decoder(&self) -> ImageResult<Decoder<'_>> {
        let mut decoder: Decoder<'_> =
            Decoder::new_with_limits(&self.input, self.limits).map_err(decode_error)?;
        decoder.set_output_format(self.output_format);
        decoder.set_lenient(self.lenient);
        Ok(decoder)
    }

    /// The payload of the last `marker` segment whose data starts with
    /// `signature` and has at least one byte after it — the rule `image`'s
    /// built-in decoder (zune-jpeg) applies to EXIF, XMP and IPTC. The core
    /// crate's own accessors differ on purpose (first EXIF wins, Extended XMP
    /// appended, IPTC reduced to the IIM payload), and an application
    /// switching backends should get the bytes it got before.
    fn last_segment_payload(&self, marker: u8, signature: &[u8]) -> ImageResult<Option<Vec<u8>>> {
        let mut decoder: Decoder<'_> = Decoder::new(&self.input).map_err(decode_error)?;
        decoder.save_markers(MarkerSaveConfig::Specific(vec![marker]));
        Ok(decoder
            .saved_markers()
            .iter()
            .rev()
            .filter(|saved: &&SavedMarker| saved.code == marker)
            .filter_map(|saved: &SavedMarker| saved.data.strip_prefix(signature))
            .find(|payload: &&[u8]| !payload.is_empty())
            .map(<[u8]>::to_vec))
    }

    /// The EXIF payload (TIFF header onward).
    fn exif_payload(&self) -> ImageResult<Option<Vec<u8>>> {
        self.last_segment_payload(APP1, EXIF_SIGNATURE)
    }
}

/// Copy a borrowed stream into an owned buffer, reporting allocator refusal
/// instead of aborting: the size is the caller's input, which may be large.
fn copy_input(data: &[u8]) -> ImageResult<Vec<u8>> {
    let mut input: Vec<u8> = Vec::new();
    input.try_reserve_exact(data.len()).map_err(|_| {
        ImageError::Limits(LimitError::from_kind(LimitErrorKind::InsufficientMemory))
    })?;
    input.extend_from_slice(data);
    Ok(input)
}

impl ImageDecoder for JpegDecoder {
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn color_type(&self) -> ColorType {
        self.color_type
    }

    fn original_color_type(&self) -> ExtendedColorType {
        self.original_color_type
    }

    fn icc_profile(&mut self) -> ImageResult<Option<Vec<u8>>> {
        let decoder: Decoder<'_> = Decoder::new(&self.input).map_err(decode_error)?;
        decoder.icc_profile().map_err(decode_error)
    }

    fn exif_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        self.exif_payload()
    }

    /// The standard XMP packet only, as `image`'s built-in decoder returns
    /// it; Extended XMP segments (`http://ns.adobe.com/xmp/extension/`) are
    /// not appended. Use `libjpeg_turbo_rs::Decoder::xmp_data` for the
    /// reassembled extension.
    fn xmp_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        self.last_segment_payload(APP1, XMP_SIGNATURE)
    }

    /// The Photoshop image resource block from APP13 — everything after the
    /// `Photoshop 3.0\0` signature, `8BIM` resource headers included — which
    /// is what `image`'s own decoder returns here. (The core crate's
    /// `Decoder::iptc_data` returns only the IIM payload inside resource
    /// 0x0404.)
    fn iptc_metadata(&mut self) -> ImageResult<Option<Vec<u8>>> {
        self.last_segment_payload(APP13, PHOTOSHOP_SIGNATURE)
    }

    /// The EXIF orientation, reported and never applied: `read_image`
    /// returns the stored pixel order, so `DynamicImage::apply_orientation`
    /// with this value rotates exactly once. Same parse as `image`'s own
    /// decoder (`Orientation::from_exif_chunk`), defaulting to
    /// `NoTransforms` when the stream has no EXIF or no valid tag.
    fn orientation(&mut self) -> ImageResult<Orientation> {
        Ok(self
            .exif_payload()?
            .as_deref()
            .and_then(Orientation::from_exif_chunk)
            .unwrap_or(Orientation::NoTransforms))
    }

    fn read_image(self, buf: &mut [u8]) -> ImageResult<()> {
        let expected: u64 = self.total_bytes();
        if buf.len() as u64 != expected {
            return Err(ImageError::Parameter(ParameterError::from_kind(
                ParameterErrorKind::DimensionMismatch,
            )));
        }
        let decoder: Decoder<'_> = self.core_decoder()?;
        let written: usize = decoder
            .decode_image_into(buf)
            .map_err(decode_error)?
            .bytes_written;
        // decode_image_into reports exactly what it wrote; anything short of
        // the advertised size would leave caller bytes stale.
        if written as u64 != expected {
            return Err(ImageError::Decoding(DecodingError::new(
                jpeg_format_hint(),
                format!("decoded {written} bytes where the header advertised {expected}"),
            )));
        }
        Ok(())
    }

    fn read_image_boxed(self: Box<Self>, buf: &mut [u8]) -> ImageResult<()> {
        (*self).read_image(buf)
    }

    /// Applies `max_image_width`, `max_image_height` and `max_alloc`.
    ///
    /// The dimension limits are checked against the header here, before any
    /// pixel allocation, and again by the core decoder. `max_alloc` becomes
    /// the core's decode-memory ceiling, whose estimate counts the output
    /// buffer, the component planes and — for progressive streams — the
    /// coefficient buffer; a stream over it is refused here as well, so
    /// `ImageError::Limits` arrives before `read_image` allocates anything.
    /// On refusal the previous limits stay in force.
    fn set_limits(&mut self, limits: Limits) -> ImageResult<()> {
        limits.check_support(&image::LimitSupport::default())?;
        limits.check_dimensions(self.width, self.height)?;
        let defaults: DecodeLimits = DecodeLimits::default();
        let mut core_limits: DecodeLimits = defaults;
        if let Some(max_width) = limits.max_image_width {
            core_limits.max_width = defaults.max_width.min(max_width as usize);
        }
        if let Some(max_height) = limits.max_image_height {
            core_limits.max_height = defaults.max_height.min(max_height as usize);
        }
        core_limits.max_memory = limits.max_alloc;
        let previous: DecodeLimits = std::mem::replace(&mut self.limits, core_limits);
        // `output_buffer_size` runs the core's header-limit checks — the same
        // ones `read_image` will hit — without decoding anything.
        let checked: ImageResult<usize> = self
            .core_decoder()
            .and_then(|decoder: Decoder<'_>| decoder.output_buffer_size().map_err(decode_error));
        if let Err(error) = checked {
            self.limits = previous;
            return Err(error);
        }
        Ok(())
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
        let pixel_format: PixelFormat = extended_color_type_to_pixel_format(color_type)
            .ok_or_else(|| {
                ImageError::Unsupported(UnsupportedError::from_format_and_kind(
                    jpeg_format_hint(),
                    UnsupportedErrorKind::Color(color_type),
                ))
            })?;

        // `ImageEncoder::write_image` takes exactly `width * height` pixels;
        // the core accepts a longer buffer, so the contract is checked here.
        let expected_len: u64 =
            u64::from(width) * u64::from(height) * pixel_format.bytes_per_pixel() as u64;
        if buf.len() as u64 != expected_len {
            return Err(ImageError::Parameter(ParameterError::from_kind(
                ParameterErrorKind::DimensionMismatch,
            )));
        }
        let jpeg_data: Vec<u8> = compress(
            buf,
            width as usize,
            height as usize,
            pixel_format,
            self.quality,
            self.subsampling,
        )
        .map_err(encode_error)?;

        self.writer
            .write_all(&jpeg_data)
            .map_err(ImageError::IoError)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libjpeg_turbo_rs::decompress_to;

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
