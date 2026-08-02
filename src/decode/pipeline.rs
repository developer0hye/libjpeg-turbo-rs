//! Stable public decoder types backed by private responsibility modules.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
use crate::common::error::DecodeWarning;
use crate::common::types::{
    ColorSpace, DctMethod, DecodeLimits, DensityInfo, PixelFormat, SavedMarker, ScalingFactor,
    Subsampling,
};
use crate::decode::marker::JpegMetadata;
use crate::simd::SimdRoutines;
use alloc::{boxed::Box, string::String, vec::Vec};

#[path = "pipeline_impl/api.rs"]
mod api;
#[path = "pipeline_impl/arithmetic.rs"]
mod arithmetic;
#[path = "pipeline_impl/baseline.rs"]
mod baseline;
#[path = "pipeline_impl/color.rs"]
mod color;
#[path = "pipeline_impl/colorspace.rs"]
mod colorspace;
#[path = "pipeline_impl/lossless.rs"]
mod lossless;
#[path = "pipeline_impl/output.rs"]
mod output;
#[path = "pipeline_impl/progressive.rs"]
mod progressive;
#[path = "pipeline_impl/raw.rs"]
mod raw;
#[path = "pipeline_impl/scan.rs"]
mod scan;
#[cfg(feature = "std")]
#[path = "pipeline_impl/streaming.rs"]
mod streaming;

pub use api::probe;
pub(crate) use color::upsample_generic_nearest;

/// Decoded image data.
///
/// `Clone`/`PartialEq` (issue #386) compare every field including the
/// pixel buffer — handy in tests and caches; clone consciously copies
/// the full pixel allocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    pub precision: u8,
    pub data: Vec<u8>,
    /// Reassembled ICC profile from APP2 markers, if present and valid.
    pub icc_profile: Option<Vec<u8>>,
    /// Raw EXIF TIFF data from APP1 marker, if present.
    pub exif_data: Option<Vec<u8>>,
    /// Raw XMP packet from APP1 (Extended XMP reassembled + appended).
    pub xmp_data: Option<Vec<u8>>,
    /// Raw IPTC IIM payload from the APP13 Photoshop IRB.
    pub iptc_data: Option<Vec<u8>>,
    /// COM marker text, if present.
    pub comment: Option<String>,
    /// Pixel density from JFIF header.
    pub density: DensityInfo,
    /// Saved APP/COM markers.
    pub saved_markers: Vec<SavedMarker>,
    /// Warnings accumulated during lenient decoding.
    pub warnings: Vec<DecodeWarning>,
}

/// Metadata for a decode that wrote pixels into a caller-provided buffer
/// (`Decoder::decode_image_into` / `decompress_into`): everything
/// [`Image`] carries except the pixel `data`.
#[derive(Debug)]
pub struct ImageInfo {
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    pub precision: u8,
    /// Number of bytes written into the caller buffer
    /// (`width * height * pixel_format.bytes_per_pixel()`).
    pub bytes_written: usize,
    /// Reassembled ICC profile from APP2 markers, if present and valid.
    pub icc_profile: Option<Vec<u8>>,
    /// Raw EXIF TIFF data from APP1 marker, if present.
    pub exif_data: Option<Vec<u8>>,
    /// Raw XMP packet (see [`Image::xmp_data`]).
    pub xmp_data: Option<Vec<u8>>,
    /// Raw IPTC IIM payload (see [`Image::iptc_data`]).
    pub iptc_data: Option<Vec<u8>>,
    /// COM marker text, if present.
    pub comment: Option<String>,
    /// Pixel density from JFIF header.
    pub density: DensityInfo,
    /// Saved APP/COM markers.
    pub saved_markers: Vec<SavedMarker>,
    /// Warnings accumulated during lenient decoding.
    pub warnings: Vec<DecodeWarning>,
}

/// Everything a caller usually wants to know about a JPEG before
/// deciding whether/how to decode it — returned by [`probe`] in one
/// call (issue #386). Header-parse only; no pixels are decoded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JpegInfo {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Number of frame components (1 gray, 3 YCbCr/RGB, 4 CMYK/YCCK).
    pub components: usize,
    /// Sample precision in bits, reported verbatim from SOF: 8 for
    /// baseline/progressive, 12 for the extended path, and anything in
    /// 2-16 for arbitrary-precision lossless. Not validated here — a
    /// corrupt stream can report any value.
    pub precision: u8,
    /// True for progressive DCT (SOF2/SOF10).
    pub progressive: bool,
    /// True for lossless (SOF3/SOF11).
    pub lossless: bool,
    /// True for arithmetic entropy coding (SOF9-SOF11).
    pub arithmetic: bool,
    /// Chroma subsampling derived from the SOF sampling factors.
    /// Grayscale sources report [`Subsampling::Unknown`] (there is no
    /// `Gray` variant) — check `components == 1` / `color_space`.
    pub subsampling: Subsampling,
    /// Source color space (from component count + APP14/JFIF heuristics,
    /// same detection the decoder itself uses).
    pub color_space: ColorSpace,
    /// EXIF orientation tag value (1-8) if present.
    pub exif_orientation: Option<u8>,
    /// An APP1 EXIF segment is present.
    pub has_exif: bool,
    /// APP2 ICC profile chunks are present.
    pub has_icc: bool,
    /// An APP1 XMP packet is present.
    pub has_xmp: bool,
    /// An APP13 Photoshop IRB with IPTC data is present.
    pub has_iptc: bool,
    /// COM marker text, if present.
    pub comment: Option<String>,
    /// Pixel density from the JFIF APP0 marker.
    pub density: DensityInfo,
}

/// JPEG decoder. Orchestrates the full decoding pipeline.
///
/// # Threading
///
/// `Decoder` is [`Send`] — a configured decoder can move to another
/// thread (rayon, `tokio::task::spawn_blocking`) and decode there
/// (issue #384). It is deliberately **not** [`Sync`]: in-decode state
/// lives behind interior mutability (`RefCell`) and the installed
/// callbacks are `Send`-only boxes, so one decoder serves one thread
/// at a time — upstream libjpeg-turbo's per-`cinfo` rule. Our own C
/// ABI shim is stricter still: a `cinfo` may not leave the thread that
/// created it (`docs/ABI_COMPATIBILITY.md`, "Threading contract").
/// Decode the same bytes concurrently by giving each thread its own
/// `Decoder`; construction from `&[u8]` is cheap (header parse only).
pub struct Decoder<'a> {
    metadata: JpegMetadata,
    raw_data: &'a [u8],
    routines: SimdRoutines,
    output_format: Option<PixelFormat>,
    scale: ScalingFactor,
    lenient: bool,
    /// Horizontal crop offset (iMCU-aligned).
    crop_x: Option<usize>,
    /// Horizontal crop width.
    crop_width: Option<usize>,
    /// Vertical crop offset in pixels (auto-aligned to MCU boundary).
    crop_y: Option<usize>,
    /// Vertical crop height in pixels.
    crop_height: Option<usize>,
    stop_on_warning: bool,
    limits: DecodeLimits,
    /// P4-58: planes decoded externally by the incremental reader,
    /// consumed (once) by `decode_baseline_planes` in place of its own
    /// entropy decode.
    prefilled_baseline_planes: core::cell::RefCell<Option<Vec<Vec<u8>>>>,
    /// Fast upsampling toggle.
    pub(crate) fast_upsample: bool,
    /// Fast DCT toggle.
    pub(crate) fast_dct: bool,
    /// DCT method for decode.
    pub(crate) dct_method: DctMethod,
    /// Block smoothing toggle.
    pub(crate) block_smoothing: bool,
    /// Output colorspace override.
    pub(crate) output_colorspace: Option<ColorSpace>,
    /// Apply ordered dithering when outputting RGB565.
    pub(crate) dither_565: bool,
    /// Enable merged upsampling (combined upsample + color convert for H2V1/H2V2).
    pub(crate) merged_upsample: bool,
    /// Custom marker processor callbacks, keyed by marker code.
    #[allow(clippy::type_complexity)]
    /// `+ Send` so the whole `Decoder` stays `Send` (issue #384):
    /// installed callbacks must be movable with the decoder they live in.
    marker_processors:
        alloc::collections::BTreeMap<u8, Box<dyn Fn(&[u8]) -> Option<Vec<u8>> + Send>>,
    /// Optional RST-marker desync recovery strategy (A6-3, mirrors
    /// `jpeg_resync_to_restart`). `None` means the historical Rust
    /// behavior of unconditionally skipping past the RST.
    ///
    /// Uses `RefCell` because `decode_image(&self)` must mutate the
    /// strategy through an immutable receiver. `+ Send` on the box for
    /// the same issue #384 reason as `marker_processors`.
    pub(crate) resync_strategy:
        core::cell::RefCell<Option<Box<dyn crate::decode::resync::RestartResyncStrategy + Send>>>,
}
