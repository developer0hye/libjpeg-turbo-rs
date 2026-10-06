//! Decode backends: one row each in the report.

use std::io::Cursor;

use image::ImageDecoder;
use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

use crate::corpus::OutputLayout;
use crate::ljt_api::{baseline, candidate};

/// A decode case: one input stream, one requested output.
pub struct DecodeCase {
    pub id: String,
    pub corpus_id: String,
    pub description: String,
    pub jpeg: Vec<u8>,
    pub layout: OutputLayout,
    /// `Some((1, 4))` for the scaled-decode case.
    pub scale: Option<(u32, u32)>,
    /// Source (unscaled) dimensions; MP/s is always over source pixels so a
    /// scaled decode's MP/s is directly comparable with the full decode's.
    pub source_width: usize,
    pub source_height: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeBackend {
    BaselineFresh,
    BaselineReuse,
    CandidateFresh,
    CandidateReuse,
    CandidateImageAdapter,
    ImageBuiltin,
    ZuneFresh,
    ZuneReuse,
}

impl DecodeBackend {
    pub const ALL: [DecodeBackend; 8] = [
        DecodeBackend::BaselineFresh,
        DecodeBackend::BaselineReuse,
        DecodeBackend::CandidateFresh,
        DecodeBackend::CandidateReuse,
        DecodeBackend::CandidateImageAdapter,
        DecodeBackend::ImageBuiltin,
        DecodeBackend::ZuneFresh,
        DecodeBackend::ZuneReuse,
    ];

    pub fn id(self) -> &'static str {
        match self {
            DecodeBackend::BaselineFresh => "baseline-fresh",
            DecodeBackend::BaselineReuse => "baseline-reuse",
            DecodeBackend::CandidateFresh => "candidate-fresh",
            DecodeBackend::CandidateReuse => "candidate-reuse",
            DecodeBackend::CandidateImageAdapter => "candidate-image-adapter",
            DecodeBackend::ImageBuiltin => "image-builtin",
            DecodeBackend::ZuneFresh => "zune-fresh",
            DecodeBackend::ZuneReuse => "zune-reuse",
        }
    }

    pub fn api(self) -> &'static str {
        match self {
            DecodeBackend::BaselineFresh => "libjpeg-turbo-rs 0.8.0 (crates.io) `decompress_to`",
            DecodeBackend::BaselineReuse => "libjpeg-turbo-rs 0.8.0 (crates.io) `decompress_into`",
            DecodeBackend::CandidateFresh => "candidate `decompress_to`",
            DecodeBackend::CandidateReuse => "candidate `decompress_into`",
            DecodeBackend::CandidateImageAdapter => {
                "candidate libjpeg-turbo-rs-image `JpegDecoder::new` + `read_image`"
            }
            DecodeBackend::ImageBuiltin => {
                "image `codecs::jpeg::JpegDecoder::new` + `read_image` (zune-jpeg inside)"
            }
            DecodeBackend::ZuneFresh => "zune-jpeg `JpegDecoder::decode`",
            DecodeBackend::ZuneReuse => "zune-jpeg `JpegDecoder::decode_into`",
        }
    }

    pub fn buffer(self) -> &'static str {
        match self {
            DecodeBackend::BaselineFresh
            | DecodeBackend::CandidateFresh
            | DecodeBackend::ZuneFresh => "library-owned, new Vec per decode",
            DecodeBackend::BaselineReuse
            | DecodeBackend::CandidateReuse
            | DecodeBackend::ZuneReuse => "caller-owned, reused",
            DecodeBackend::CandidateImageAdapter => {
                "caller-owned, reused (adapter decodes into its own Vec in `new`, then copies)"
            }
            DecodeBackend::ImageBuiltin => {
                "caller-owned, reused (image copies the input stream into a Vec first)"
            }
        }
    }

    /// Fresh rows construct and drop everything per decode; reuse rows keep
    /// only the output buffer between iterations.
    pub fn path(self) -> &'static str {
        match self {
            DecodeBackend::BaselineFresh
            | DecodeBackend::CandidateFresh
            | DecodeBackend::ZuneFresh => "fresh",
            _ => "buffer-reuse",
        }
    }
}

fn zune_options(layout: OutputLayout) -> DecoderOptions {
    let colorspace: ColorSpace = match layout {
        OutputLayout::Rgb => ColorSpace::RGB,
        OutputLayout::Gray => ColorSpace::Luma,
    };
    DecoderOptions::default().jpeg_set_out_colorspace(colorspace)
}

fn expected_image_color(layout: OutputLayout) -> image::ColorType {
    match layout {
        OutputLayout::Rgb => image::ColorType::Rgb8,
        OutputLayout::Gray => image::ColorType::L8,
    }
}

/// A backend ready to decode one case. Reuse buffers are allocated here,
/// outside every timed and allocation-counted region.
pub struct PreparedDecode<'case> {
    pub backend: DecodeBackend,
    case: &'case DecodeCase,
    reuse_buffer: Vec<u8>,
}

pub struct Decoded {
    pub width: usize,
    pub height: usize,
    owned: Option<Vec<u8>>,
}

pub enum Preparation<'case> {
    Ready(PreparedDecode<'case>),
    /// The backend has no API for this case (e.g. zune-jpeg has no scaled
    /// decode). The reason is printed in the report.
    NotApplicable(&'static str),
}

impl<'case> PreparedDecode<'case> {
    pub fn prepare(backend: DecodeBackend, case: &'case DecodeCase) -> Preparation<'case> {
        let scaled: bool = case.scale.is_some();
        // Arm order matters: the libjpeg-turbo-rs rows support every case,
        // the `_ if scaled` arm then turns every other backend's scaled case
        // into a reported N/A before the per-backend arms run.
        let reuse_len: usize = match backend {
            DecodeBackend::BaselineFresh | DecodeBackend::CandidateFresh => 0,
            DecodeBackend::BaselineReuse => {
                baseline::output_size(&case.jpeg, case.layout, case.scale)
            }
            DecodeBackend::CandidateReuse => {
                candidate::output_size(&case.jpeg, case.layout, case.scale)
            }
            _ if scaled => {
                return Preparation::NotApplicable(
                    "no scaled-decode API (DCT-domain scaling is a libjpeg feature)",
                );
            }
            DecodeBackend::ZuneFresh => 0,
            DecodeBackend::CandidateImageAdapter => {
                let decoder: libjpeg_turbo_rs_image::JpegDecoder =
                    libjpeg_turbo_rs_image::JpegDecoder::new(&case.jpeg)
                        .unwrap_or_else(|error| panic!("adapter decode failed: {error}"));
                assert_eq!(
                    decoder.color_type(),
                    expected_image_color(case.layout),
                    "adapter picked a different output color type than the case requests"
                );
                decoder.total_bytes() as usize
            }
            DecodeBackend::ImageBuiltin => {
                let decoder = image::codecs::jpeg::JpegDecoder::new(Cursor::new(&case.jpeg[..]))
                    .unwrap_or_else(|error| panic!("image decoder construction failed: {error}"));
                assert_eq!(
                    decoder.color_type(),
                    expected_image_color(case.layout),
                    "image's decoder picked a different output color type than the case requests"
                );
                decoder.total_bytes() as usize
            }
            DecodeBackend::ZuneReuse => {
                let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
                    ZCursor::new(&case.jpeg[..]),
                    zune_options(case.layout),
                );
                decoder
                    .decode_headers()
                    .unwrap_or_else(|error| panic!("zune header parse failed: {error:?}"));
                decoder
                    .output_buffer_size()
                    .expect("zune reports a size once headers are decoded")
            }
        };
        Preparation::Ready(PreparedDecode {
            backend,
            case,
            reuse_buffer: vec![0u8; reuse_len],
        })
    }

    /// One complete decode, constructing the decoder from the JPEG bytes as an
    /// application would. Any error is a harness failure, never a skip.
    pub fn decode(&mut self) -> Decoded {
        let case: &DecodeCase = self.case;
        let jpeg: &[u8] = &case.jpeg;
        match self.backend {
            DecodeBackend::BaselineFresh => {
                let (width, height, pixels) = baseline::decode_fresh(jpeg, case.layout, case.scale);
                Decoded {
                    width,
                    height,
                    owned: Some(pixels),
                }
            }
            DecodeBackend::CandidateFresh => {
                let (width, height, pixels) =
                    candidate::decode_fresh(jpeg, case.layout, case.scale);
                Decoded {
                    width,
                    height,
                    owned: Some(pixels),
                }
            }
            DecodeBackend::BaselineReuse => {
                let (width, height) =
                    baseline::decode_into(jpeg, case.layout, case.scale, &mut self.reuse_buffer);
                Decoded {
                    width,
                    height,
                    owned: None,
                }
            }
            DecodeBackend::CandidateReuse => {
                let (width, height) =
                    candidate::decode_into(jpeg, case.layout, case.scale, &mut self.reuse_buffer);
                Decoded {
                    width,
                    height,
                    owned: None,
                }
            }
            DecodeBackend::CandidateImageAdapter => {
                let decoder: libjpeg_turbo_rs_image::JpegDecoder =
                    libjpeg_turbo_rs_image::JpegDecoder::new(jpeg)
                        .unwrap_or_else(|error| panic!("adapter decode failed: {error}"));
                let (width, height) = decoder.dimensions();
                decoder
                    .read_image(&mut self.reuse_buffer)
                    .unwrap_or_else(|error| panic!("adapter read_image failed: {error}"));
                Decoded {
                    width: width as usize,
                    height: height as usize,
                    owned: None,
                }
            }
            DecodeBackend::ImageBuiltin => {
                let decoder = image::codecs::jpeg::JpegDecoder::new(Cursor::new(jpeg))
                    .unwrap_or_else(|error| panic!("image decoder construction failed: {error}"));
                let (width, height) = decoder.dimensions();
                decoder
                    .read_image(&mut self.reuse_buffer)
                    .unwrap_or_else(|error| panic!("image read_image failed: {error}"));
                Decoded {
                    width: width as usize,
                    height: height as usize,
                    owned: None,
                }
            }
            DecodeBackend::ZuneFresh => {
                let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
                    ZCursor::new(jpeg),
                    zune_options(case.layout),
                );
                let pixels: Vec<u8> = decoder
                    .decode()
                    .unwrap_or_else(|error| panic!("zune decode failed: {error:?}"));
                let (width, height) = decoder.dimensions().expect("dimensions after decode");
                Decoded {
                    width,
                    height,
                    owned: Some(pixels),
                }
            }
            DecodeBackend::ZuneReuse => {
                let mut decoder = zune_jpeg::JpegDecoder::new_with_options(
                    ZCursor::new(jpeg),
                    zune_options(case.layout),
                );
                decoder
                    .decode_into(&mut self.reuse_buffer)
                    .unwrap_or_else(|error| panic!("zune decode_into failed: {error:?}"));
                let (width, height) = decoder.dimensions().expect("dimensions after decode");
                Decoded {
                    width,
                    height,
                    owned: None,
                }
            }
        }
    }

    pub fn pixels<'a>(&'a self, decoded: &'a Decoded) -> &'a [u8] {
        match &decoded.owned {
            Some(pixels) => pixels,
            None => &self.reuse_buffer,
        }
    }
}
