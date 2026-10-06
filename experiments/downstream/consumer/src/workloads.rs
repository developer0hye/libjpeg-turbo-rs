//! Encode backends and the decode → orient → resize → encode thumbnail
//! workload.

use std::io::Cursor;

use image::imageops::FilterType;
use image::{DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, RgbImage};

use crate::ljt_api::{baseline, candidate};

/// Quality for every encode row and the thumbnail's final encode.
pub const ENCODE_QUALITY: u8 = 85;

/// Longest side of the thumbnail.
pub const THUMBNAIL_LONG_SIDE: u32 = 256;

/// One resampling filter for every thumbnail row, so the rows differ only in
/// codec. Triangle (bilinear) is what thumbnailers commonly default to.
const THUMBNAIL_FILTER: FilterType = FilterType::Triangle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodeBackend {
    Baseline,
    Candidate,
    CandidateImageAdapter,
    ImageBuiltin,
}

impl EncodeBackend {
    pub const ALL: [EncodeBackend; 4] = [
        EncodeBackend::Baseline,
        EncodeBackend::Candidate,
        EncodeBackend::CandidateImageAdapter,
        EncodeBackend::ImageBuiltin,
    ];

    pub fn id(self) -> &'static str {
        match self {
            EncodeBackend::Baseline => "baseline",
            EncodeBackend::Candidate => "candidate",
            EncodeBackend::CandidateImageAdapter => "candidate-image-adapter",
            EncodeBackend::ImageBuiltin => "image-builtin",
        }
    }

    pub fn api(self) -> &'static str {
        match self {
            EncodeBackend::Baseline => {
                "libjpeg-turbo-rs 0.8.0 (crates.io) `compress(q85, S420)`"
            }
            EncodeBackend::Candidate => "candidate `compress(q85, S420)`",
            EncodeBackend::CandidateImageAdapter => {
                "candidate libjpeg-turbo-rs-image `JpegEncoder::new_with_quality(85)` (4:2:0 default)"
            }
            EncodeBackend::ImageBuiltin => {
                "image `codecs::jpeg::JpegEncoder::new_with_quality(85)` (subsampling is image's choice)"
            }
        }
    }

    /// Output is always a new `Vec` per encode: no backend offers encoding
    /// into a caller-owned buffer through the API used here.
    pub fn encode(self, rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
        match self {
            EncodeBackend::Baseline => baseline::compress_rgb(rgb, width, height, ENCODE_QUALITY),
            EncodeBackend::Candidate => candidate::compress_rgb(rgb, width, height, ENCODE_QUALITY),
            EncodeBackend::CandidateImageAdapter => {
                let mut out: Vec<u8> = Vec::new();
                libjpeg_turbo_rs_image::JpegEncoder::new_with_quality(&mut out, ENCODE_QUALITY)
                    .write_image(rgb, width as u32, height as u32, ExtendedColorType::Rgb8)
                    .unwrap_or_else(|error| panic!("adapter encode failed: {error}"));
                out
            }
            EncodeBackend::ImageBuiltin => {
                let mut out: Vec<u8> = Vec::new();
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, ENCODE_QUALITY)
                    .encode(rgb, width as u32, height as u32, ExtendedColorType::Rgb8)
                    .unwrap_or_else(|error| panic!("image encode failed: {error}"));
                out
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailBackend {
    Baseline,
    Candidate,
    CandidateScaledDecode,
    CandidateImageAdapter,
    ImageBuiltin,
    Zune,
}

impl ThumbnailBackend {
    pub const ALL: [ThumbnailBackend; 6] = [
        ThumbnailBackend::Baseline,
        ThumbnailBackend::Candidate,
        ThumbnailBackend::CandidateScaledDecode,
        ThumbnailBackend::CandidateImageAdapter,
        ThumbnailBackend::ImageBuiltin,
        ThumbnailBackend::Zune,
    ];

    pub fn id(self) -> &'static str {
        match self {
            ThumbnailBackend::Baseline => "baseline",
            ThumbnailBackend::Candidate => "candidate",
            ThumbnailBackend::CandidateScaledDecode => "candidate-scaled-decode",
            ThumbnailBackend::CandidateImageAdapter => "candidate-image-adapter",
            ThumbnailBackend::ImageBuiltin => "image-builtin",
            ThumbnailBackend::Zune => "zune",
        }
    }

    pub fn api(self) -> &'static str {
        match self {
            ThumbnailBackend::Baseline => {
                "0.8.0 decode + `Image::apply_orientation` → imageops::resize → 0.8.0 `compress`"
            }
            ThumbnailBackend::Candidate => {
                "candidate decode + `Image::apply_orientation` → imageops::resize → candidate `compress`"
            }
            ThumbnailBackend::CandidateScaledDecode => {
                "candidate decode at the largest DCT scale ≥ thumbnail size + orient → resize → candidate `compress`"
            }
            ThumbnailBackend::CandidateImageAdapter => {
                "adapter `JpegDecoder` → `orientation()` → DynamicImage::apply_orientation → resize → adapter `JpegEncoder`"
            }
            ThumbnailBackend::ImageBuiltin => {
                "image `JpegDecoder` → `orientation()` → DynamicImage::apply_orientation → resize → image `JpegEncoder`"
            }
            ThumbnailBackend::Zune => "zune-jpeg alone",
        }
    }

    pub fn not_applicable(self) -> Option<&'static str> {
        match self {
            ThumbnailBackend::Zune => {
                Some("zune-jpeg is a decoder only: no orientation, resize or encode")
            }
            _ => None,
        }
    }
}

/// Fit `width`x`height` inside a `THUMBNAIL_LONG_SIDE` square, keeping the
/// aspect ratio (rounded, never 0).
pub fn thumbnail_size(width: u32, height: u32) -> (u32, u32) {
    let long_side: u32 = width.max(height);
    let scale = |side: u32| -> u32 {
        ((side as u64 * THUMBNAIL_LONG_SIDE as u64 + long_side as u64 / 2) / long_side as u64)
            .max(1) as u32
    };
    (scale(width), scale(height))
}

/// The largest libjpeg DCT reduction (1/8, 1/4, 1/2) that still leaves the
/// long side at least `THUMBNAIL_LONG_SIDE`, so the resize afterwards is
/// always a downscale — what a thumbnailer using libjpeg's scaling does.
pub fn thumbnail_scale(width: usize, height: usize) -> Option<(u32, u32)> {
    let long_side: usize = width.max(height);
    [8usize, 4, 2]
        .into_iter()
        .find(|denominator| long_side.div_ceil(*denominator) >= THUMBNAIL_LONG_SIDE as usize)
        .map(|denominator| (1, denominator as u32))
}

fn resize(rgb: RgbImage) -> RgbImage {
    let (width, height) = thumbnail_size(rgb.width(), rgb.height());
    image::imageops::resize(&rgb, width, height, THUMBNAIL_FILTER)
}

fn rgb_image(width: usize, height: usize, pixels: Vec<u8>) -> RgbImage {
    RgbImage::from_raw(width as u32, height as u32, pixels)
        .expect("decoder output length matches its reported dimensions")
}

/// Run the whole workload once; returns the encoded thumbnail.
pub fn thumbnail(
    backend: ThumbnailBackend,
    jpeg: &[u8],
    source_width: usize,
    source_height: usize,
) -> Vec<u8> {
    match backend {
        ThumbnailBackend::Baseline => {
            let (width, height, pixels) = baseline::decode_oriented_rgb(jpeg, None);
            let small: RgbImage = resize(rgb_image(width, height, pixels));
            baseline::compress_rgb(
                small.as_raw(),
                small.width() as usize,
                small.height() as usize,
                ENCODE_QUALITY,
            )
        }
        ThumbnailBackend::Candidate | ThumbnailBackend::CandidateScaledDecode => {
            let scale: Option<(u32, u32)> = if backend == ThumbnailBackend::CandidateScaledDecode {
                thumbnail_scale(source_width, source_height)
            } else {
                None
            };
            let (width, height, pixels) = candidate::decode_oriented_rgb(jpeg, scale);
            let small: RgbImage = resize(rgb_image(width, height, pixels));
            candidate::compress_rgb(
                small.as_raw(),
                small.width() as usize,
                small.height() as usize,
                ENCODE_QUALITY,
            )
        }
        ThumbnailBackend::CandidateImageAdapter => {
            let mut decoder: libjpeg_turbo_rs_image::JpegDecoder =
                libjpeg_turbo_rs_image::JpegDecoder::new(jpeg)
                    .unwrap_or_else(|error| panic!("adapter decode failed: {error}"));
            let orientation: image::metadata::Orientation = decoder
                .orientation()
                .unwrap_or_else(|error| panic!("adapter orientation failed: {error}"));
            let mut decoded: DynamicImage = DynamicImage::from_decoder(decoder)
                .unwrap_or_else(|error| panic!("adapter read failed: {error}"));
            decoded.apply_orientation(orientation);
            let small: RgbImage = resize(decoded.into_rgb8());
            let mut out: Vec<u8> = Vec::new();
            libjpeg_turbo_rs_image::JpegEncoder::new_with_quality(&mut out, ENCODE_QUALITY)
                .write_image(
                    small.as_raw(),
                    small.width(),
                    small.height(),
                    ExtendedColorType::Rgb8,
                )
                .unwrap_or_else(|error| panic!("adapter encode failed: {error}"));
            out
        }
        ThumbnailBackend::ImageBuiltin => {
            let mut decoder = image::codecs::jpeg::JpegDecoder::new(Cursor::new(jpeg))
                .unwrap_or_else(|error| panic!("image decoder construction failed: {error}"));
            let orientation: image::metadata::Orientation = decoder
                .orientation()
                .unwrap_or_else(|error| panic!("image orientation failed: {error}"));
            let mut decoded: DynamicImage = DynamicImage::from_decoder(decoder)
                .unwrap_or_else(|error| panic!("image decode failed: {error}"));
            decoded.apply_orientation(orientation);
            let small: RgbImage = resize(decoded.into_rgb8());
            let mut out: Vec<u8> = Vec::new();
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, ENCODE_QUALITY)
                .encode_image(&small)
                .unwrap_or_else(|error| panic!("image encode failed: {error}"));
            out
        }
        ThumbnailBackend::Zune => unreachable!("callers skip not-applicable rows"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portrait_phone_thumbnail_geometry() {
        // 4032x3024 rotated by EXIF 6 is 3024x4032 → 192x256.
        assert_eq!(thumbnail_size(3024, 4032), (192, 256));
        assert_eq!(thumbnail_size(4032, 3024), (256, 192));
        // 4032 / 8 = 504 ≥ 256, so the scaled path decodes at 1/8.
        assert_eq!(thumbnail_scale(4032, 3024), Some((1, 8)));
        // 300 / 2 = 150 < 256: no reduction leaves enough pixels.
        assert_eq!(thumbnail_scale(300, 200), None);
    }
}
