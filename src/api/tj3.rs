//! TJ3-compatible handle/parameter API for JPEG compression/decompression.
//!
//! Provides a handle-based interface matching libjpeg-turbo's TurboJPEG 3 API
//! pattern: `tj3Init()`/`tj3Set()`/`tj3Get()`/`tj3Destroy()`. All JPEG
//! parameters are stored in a single `TjHandle` and accessed via `TjParam`.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
use crate::common::error::{JpegError, Result};
use crate::common::types::{
    ColorSpace, CropRegion, DctMethod, DensityUnit, FrameHeader, MarkerSaveConfig, PixelFormat,
    ScalingFactor, Subsampling,
};
use crate::decode::pipeline::{Decoder, Image};
#[allow(unused_imports)]
use alloc::vec::Vec;
#[allow(unused_imports)]
use alloc::{format, vec};

/// All TJPARAM parameter identifiers from libjpeg-turbo TJ3 API.
///
/// Maps 1-to-1 with the C `TJPARAM_*` constants. Integer encoding matches
/// libjpeg-turbo conventions (e.g., subsampling as 0-5, booleans as 0/1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TjParam {
    /// TJPARAM_QUALITY: Quality factor 1-100 once set. A fresh handle
    /// reports -1 (unset, as upstream's does — P4-155); `set` rejects
    /// writing -1 back, as upstream's `SET_PARAM(quality, 1, 100)` does.
    Quality,
    /// TJPARAM_SUBSAMP: Chroma subsampling (0=444, 1=422, 2=420, 3=Gray, 4=440, 5=411).
    Subsampling,
    /// TJPARAM_JPEGWIDTH: the frame header's width, as the last decompress or
    /// header read published it — unaffected by scaling or cropping. -1 until
    /// a header has been read.
    Width,
    /// TJPARAM_JPEGHEIGHT: the frame header's height; see `Width`.
    Height,
    /// TJPARAM_PRECISION: the frame header's sample precision after a
    /// decompress or header read (8, 12 or 2-16 for lossless).
    Precision,
    /// TJPARAM_COLORSPACE: Color space (-1=Default/auto, 0=RGB, 1=YCbCr, 2=Gray, 3=CMYK, 4=YCCK).
    ColorSpace,
    /// TJPARAM_FASTUPSAMPLE: Use nearest-neighbor upsampling (boolean).
    FastUpSample,
    /// TJPARAM_FASTDCT: Use fast DCT algorithm (boolean).
    FastDct,
    /// TJPARAM_OPTIMIZE: Use optimized Huffman tables (boolean).
    Optimize,
    /// TJPARAM_PROGRESSIVE: Enable progressive JPEG (boolean).
    Progressive,
    /// TJPARAM_SCANLIMIT: Max number of progressive scans before error.
    ScanLimit,
    /// TJPARAM_ARITHMETIC: Use arithmetic entropy coding (boolean).
    Arithmetic,
    /// TJPARAM_LOSSLESS: Enable lossless mode (boolean).
    Lossless,
    /// TJPARAM_LOSSLESSPSV: Lossless predictor selection value (1-7).
    LosslessPsv,
    /// TJPARAM_LOSSLESSPT: Lossless point transform (0-15).
    LosslessPt,
    /// TJPARAM_RESTARTBLOCKS: Restart interval in MCU blocks.
    RestartBlocks,
    /// TJPARAM_RESTARTROWS: Restart interval in MCU rows.
    RestartRows,
    /// TJPARAM_XDENSITY: Horizontal pixel density.
    XDensity,
    /// TJPARAM_YDENSITY: Vertical pixel density.
    YDensity,
    /// TJPARAM_DENSITYUNITS: Density units (0=unknown, 1=DPI, 2=DPCM).
    DensityUnits,
    /// TJPARAM_MAXMEMORY: Max memory in MEGABYTES (0=unlimited), per the C contract; converted x1048576 at decode time.
    MaxMemory,
    /// TJPARAM_MAXPIXELS: Max image size in pixels (0=unlimited).
    MaxPixels,
    /// TJPARAM_BOTTOMUP: Bottom-up row order (boolean).
    BottomUp,
    /// TJPARAM_NOREALLOC: Use pre-allocated output buffer (boolean).
    /// N/A in Rust: `Vec<u8>` return type handles allocation automatically.
    /// Stored for API compatibility but has no behavioral effect.
    NoRealloc,
    /// TJPARAM_STOPONWARNING: Treat warnings as fatal (boolean).
    StopOnWarning,
    /// TJPARAM_SAVEMARKERS: Marker preservation level (0-4, matching C libjpeg-turbo).
    ///
    /// 0 = none, 1 = COM only, 2 = all (default in C), 3 = all except ICC, 4 = ICC only.
    SaveMarkers,
}

/// Frame facts determined by the JPEG header alone.
///
/// Returned by [`TjHandle::inspect_header`], which reads them without decoding
/// pixel data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameInfo {
    /// Image width in pixels, unscaled.
    pub width: usize,
    /// Image height in pixels, unscaled.
    pub height: usize,
    /// Number of components in the SOF marker: 1 for grayscale, 3 for YCbCr,
    /// 4 for CMYK/YCCK.
    pub num_components: usize,
    /// Chroma subsampling implied by the components' sampling factors.
    pub subsampling: Subsampling,
}

/// The frame-header facts `tj3SetCroppingRegion` validates a region against.
///
/// Upstream's `setDecompParameters` (`turbojpeg.c:514-536`) records them from
/// the SOF every time a header is read, and `tj3SetCroppingRegion`
/// (`turbojpeg.c:2068-2115`) reads them back. The handle's published
/// `Width`/`Height`/`Subsampling`/`Precision`/`Lossless` hold the same values
/// once a header is read (P4-199, P4-200); this copy exists because a caller
/// may `set` the writable ones in between, and upstream validates against
/// `jpegWidth`, which `tj3Set` cannot reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CroppingGeometry {
    /// `jpegWidth`: the SOF's width, unscaled.
    width: usize,
    /// `jpegHeight`: the SOF's height, unscaled.
    height: usize,
    /// `subsamp` as a `TJSAMP_*` value; -1 is `TJSAMP_UNKNOWN`.
    subsampling: i32,
    /// `precision`: the SOF's sample precision.
    precision: u8,
    /// `lossless`: an SOF3/SOF11 frame.
    lossless: bool,
}

impl CroppingGeometry {
    fn of(decoder: &Decoder<'_>) -> Self {
        let frame = decoder.header();
        let subsampling: i32 = Self::turbojpeg_subsampling(decoder);
        Self {
            width: frame.width as usize,
            height: frame.height as usize,
            subsampling,
            precision: frame.precision,
            lossless: frame.is_lossless,
        }
    }

    /// Upstream's `getSubsamp` (`turbojpeg.c:431-510`), ported as written:
    /// the iMCU width a crop is checked against comes from this
    /// classification, so it has to agree with upstream's on every frame,
    /// including the non-standard sampling layouts it deliberately accepts
    /// (4:2:2 and 4:4:0 spelled with a 2x2 luma, 4:4:4 with equal non-unit
    /// factors) and the ones it leaves at TJSAMP_UNKNOWN. `Decoder::
    /// jpeg_subsampling` compares only luma with the first chroma component,
    /// so `2x2,1x1,2x2` read as 4:2:0 there where upstream refuses to crop it
    /// (codex review of P4-197). `numSamp` is TJ_NUMSAMP, as for a 3.2 handle
    /// (`:611`).
    fn turbojpeg_subsampling(decoder: &Decoder<'_>) -> i32 {
        // tjMCUWidth[] / tjMCUHeight[] (turbojpeg.h:247, :277).
        const MCU_WIDTH: [usize; 9] = [8, 16, 16, 8, 8, 32, 8, 32, 16];
        const MCU_HEIGHT: [usize; 9] = [8, 8, 16, 8, 16, 8, 32, 16, 32];
        const TJSAMP_444: usize = 0;
        const TJSAMP_422: usize = 1;
        const TJSAMP_GRAY: usize = 3;
        const TJSAMP_440: usize = 4;
        // jpeglib.h D_MAX_BLOCKS_IN_MCU.
        const MAX_BLOCKS_IN_MCU: usize = 10;

        let components = &decoder.header().components;
        let color_space: ColorSpace = decoder.jpeg_color_space();
        let num_components: usize = components.len();
        if num_components == 1 && color_space == ColorSpace::Grayscale {
            return TJSAMP_GRAY as i32;
        }
        let is_cmyk_like: bool = matches!(color_space, ColorSpace::Cmyk | ColorSpace::Ycck);
        let factors = |k: usize| -> (usize, usize) {
            (
                components[k].horizontal_sampling as usize,
                components[k].vertical_sampling as usize,
            )
        };
        let mut result: i32 = -1;
        for i in 0..MCU_WIDTH.len() {
            if i == TJSAMP_GRAY {
                continue;
            }
            if !(num_components == 3 || (is_cmyk_like && num_components == 4)) {
                continue;
            }
            let (h0, v0): (usize, usize) = factors(0);
            if h0 == MCU_WIDTH[i] / 8 && v0 == MCU_HEIGHT[i] / 8 {
                let matched: usize = (1..num_components)
                    .filter(|&k| {
                        let (href, vref): (usize, usize) = if is_cmyk_like && k == 3 {
                            (MCU_WIDTH[i] / 8, MCU_HEIGHT[i] / 8)
                        } else {
                            (1, 1)
                        };
                        factors(k) == (href, vref)
                    })
                    .count();
                if matched == num_components - 1 {
                    result = i as i32;
                    break;
                }
            }
            // 4:2:2 and 4:4:0 images whose sampling factors are specified in
            // non-standard ways.
            if h0 == 2 && v0 == 2 && (i == TJSAMP_422 || i == TJSAMP_440) {
                let matched: usize = (1..num_components)
                    .filter(|&k| {
                        let (href, vref): (usize, usize) = if is_cmyk_like && k == 3 {
                            (2, 2)
                        } else {
                            (MCU_HEIGHT[i] / 8, MCU_WIDTH[i] / 8)
                        };
                        factors(k) == (href, vref)
                    })
                    .count();
                if matched == num_components - 1 {
                    result = i as i32;
                    break;
                }
            }
            // 4:4:4 images whose sampling factors are specified in
            // non-standard ways. Upstream's `break` here leaves only the inner
            // loop, so the outer one keeps going; `matched` counting to the
            // end gives the same answer, since the test is "all matched".
            if h0 * v0 <= MAX_BLOCKS_IN_MCU / 3 && i == TJSAMP_444 {
                let matched: usize = (1..num_components)
                    .filter(|&k| factors(k) == (h0, v0))
                    .count();
                if matched == num_components - 1 {
                    result = i as i32;
                }
            }
        }
        result
    }

    /// `tj3SetCroppingRegion`'s checks after its sign and header-read guards,
    /// in upstream's order (`turbojpeg.c:2088-2111`), returning the region
    /// upstream stores: a zero `width` or `height` means "to the right/bottom
    /// edge" and is filled in.
    ///
    /// The iMCU-divisibility rule is upstream's, not the `Decoder`'s: a left
    /// boundary that is not a multiple of the scaled iMCU width is refused
    /// here, where `Decoder::set_crop` aligns it down. TurboJPEG documents the
    /// refusal and its callers size their buffers from the region they passed,
    /// so silently widening the output would break them (P4-197, #618).
    fn resolve(&self, region: CropRegion, scaling: ScalingFactor) -> Result<CropRegion> {
        let refuse = |reason: alloc::string::String| JpegError::InvalidCropRegion { reason };
        if (self.precision != 8 && self.precision != 12) || self.lossless {
            return Err(refuse(alloc::string::String::from(
                "Cannot partially decompress lossless JPEG images",
            )));
        }
        // tjMCUWidth[] (turbojpeg.h:247), indexed by TJSAMP_*.
        const MCU_WIDTH: [usize; 9] = [8, 16, 16, 8, 8, 32, 8, 32, 16];
        let mcu_width: usize = match usize::try_from(self.subsampling)
            .ok()
            .and_then(|index| MCU_WIDTH.get(index))
        {
            Some(&width) => width,
            None => {
                return Err(refuse(alloc::string::String::from(
                    "Could not determine subsampling level of JPEG image",
                )))
            }
        };
        let scaled_width: usize = scaling.scale_dim(self.width);
        let scaled_height: usize = scaling.scale_dim(self.height);
        let scaled_imcu_width: usize = scaling.scale_dim(mcu_width);
        if !region.x.is_multiple_of(scaled_imcu_width) {
            return Err(refuse(format!(
                "The left boundary of the cropping region ({}) is not\n\
                 divisible by the scaled iMCU width ({})",
                region.x, scaled_imcu_width
            )));
        }
        // C computes `scaledWidth - x` in `int`, so a left boundary past the
        // edge yields a non-positive width and falls into the refusal below;
        // `checked_sub` reaches the same refusal without wrapping.
        let width: Option<usize> = if region.width == 0 {
            scaled_width.checked_sub(region.x)
        } else {
            Some(region.width)
        };
        let height: Option<usize> = if region.height == 0 {
            scaled_height.checked_sub(region.y)
        } else {
            Some(region.height)
        };
        let fits = |origin: usize, extent: Option<usize>, limit: usize| -> Option<usize> {
            let extent: usize = extent.filter(|&extent| extent > 0)?;
            (origin.checked_add(extent)? <= limit).then_some(extent)
        };
        match (
            fits(region.x, width, scaled_width),
            fits(region.y, height, scaled_height),
        ) {
            (Some(width), Some(height)) => Ok(CropRegion {
                x: region.x,
                y: region.y,
                width,
                height,
            }),
            _ => Err(refuse(alloc::string::String::from(
                "The cropping region exceeds the scaled image dimensions",
            ))),
        }
    }
}

/// `TJCS_DEFAULT` (`turbojpeg.h:541`): what `setDecompParameters` publishes
/// for a `jpeg_color_space` it has no `TJCS_*` for (`turbojpeg.c:526`).
const TJCS_DEFAULT: i32 = -1;

/// TJ3-compatible handle for JPEG compression/decompression.
///
/// Wraps all parameters in a single object with get/set accessors,
/// matching the libjpeg-turbo `tjhandle` pattern. Create with `new()`,
/// configure with `set()`, compress/decompress, then drop.
pub struct TjHandle {
    quality: i32,
    subsampling: i32,
    width: i32,
    height: i32,
    precision: i32,
    color_space: i32,
    fast_upsample: i32,
    fast_dct: i32,
    optimize: i32,
    progressive: i32,
    scan_limit: i32,
    arithmetic: i32,
    lossless: i32,
    lossless_psv: i32,
    lossless_pt: i32,
    restart_blocks: i32,
    restart_rows: i32,
    x_density: i32,
    y_density: i32,
    density_units: i32,
    max_memory: i32,
    max_pixels: i32,
    bottom_up: i32,
    no_realloc: i32,
    stop_on_warning: i32,
    save_markers: i32,
    icc_profile: Option<Vec<u8>>,
    scaling_factor: ScalingFactor,
    cropping_region: Option<CropRegion>,
    /// The last frame header a decompress read, for `resolve_cropping_region`;
    /// `None` until one has been read, as upstream's `jpegWidth == -1`.
    cropping_geometry: Option<CroppingGeometry>,
}

impl TjHandle {
    /// Create a new TJ3 handle with default parameters (like `tj3Init`).
    pub fn new() -> Self {
        Self {
            // Upstream initialises both to *unset* — quality -1 and
            // TJSAMP_UNKNOWN — and every lossy compress path refuses until
            // the caller supplies them (`turbojpeg-mp.c:95-98`). Defaulting
            // to 75 / 4:2:0 made those errors unreachable and silently
            // substituted values a caller never chose (P4-155, #539).
            quality: -1,
            subsampling: -1, // TJSAMP_UNKNOWN
            // tj3InitVersion seeds both to -1, the "no header read yet"
            // sentinel (`turbojpeg.c:600-601`); 0 is a dimension a caller
            // cannot tell from a degenerate frame (P4-200, #621).
            width: -1,
            height: -1,
            precision: 8,
            color_space: TJCS_DEFAULT,
            fast_upsample: 0,
            fast_dct: 0,
            optimize: 0,
            progressive: 0,
            scan_limit: 0,
            arithmetic: 0,
            lossless: 0,
            lossless_psv: 1,
            lossless_pt: 0,
            restart_blocks: 0,
            restart_rows: 0,
            x_density: 1,
            y_density: 1,
            density_units: 0, // unknown
            max_memory: 0,
            max_pixels: 0,
            bottom_up: 0,
            no_realloc: 0,
            stop_on_warning: 0,
            save_markers: 2, // TJSM_ALL (C default)
            icc_profile: None,
            scaling_factor: ScalingFactor::default(),
            cropping_region: None,
            cropping_geometry: None,
        }
    }

    /// Set a parameter value (like `tj3Set`).
    ///
    /// Returns an error if the value is out of the valid range for the parameter.
    pub fn set(&mut self, param: TjParam, value: i32) -> Result<()> {
        match param {
            TjParam::Quality => {
                if !(1..=100).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "quality must be 1-100, got {value}"
                    )));
                }
                self.quality = value;
            }
            TjParam::Subsampling => {
                // 0=444, 1=422, 2=420, 3=GRAY, 4=440, 5=411,
                // 6=441, 7=410, 8=24 (libjpeg-turbo 3.x widened range).
                if !(0..=8).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "subsampling must be 0-8, got {value}"
                    )));
                }
                self.subsampling = value;
            }
            TjParam::Width => {
                self.width = value;
            }
            TjParam::Height => {
                self.height = value;
            }
            TjParam::Precision => {
                // Mirror upstream
                // `references/libjpeg-turbo/src/turbojpeg.c:769`:
                //   `SET_PARAM(precision, 2, 16);`
                // Reject globally-invalid values at set time so
                // out-of-spec writes surface as -1 from `tj3Set` rather
                // than silently encoding at the entry-point default.
                if !(2..=16).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "precision must be 2-16, got {value}"
                    )));
                }
                self.precision = value;
            }
            TjParam::ColorSpace => {
                if !(-1..=4).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "color space must be -1..4, got {value}"
                    )));
                }
                self.color_space = value;
            }
            TjParam::FastUpSample => {
                self.fast_upsample = if value != 0 { 1 } else { 0 };
            }
            TjParam::FastDct => {
                self.fast_dct = if value != 0 { 1 } else { 0 };
            }
            TjParam::Optimize => {
                self.optimize = if value != 0 { 1 } else { 0 };
            }
            TjParam::Progressive => {
                self.progressive = if value != 0 { 1 } else { 0 };
            }
            TjParam::ScanLimit => {
                self.scan_limit = value;
            }
            TjParam::Arithmetic => {
                self.arithmetic = if value != 0 { 1 } else { 0 };
            }
            TjParam::Lossless => {
                self.lossless = if value != 0 { 1 } else { 0 };
            }
            TjParam::LosslessPsv => {
                if !(1..=7).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "lossless PSV must be 1-7, got {value}"
                    )));
                }
                self.lossless_psv = value;
            }
            TjParam::LosslessPt => {
                if !(0..=15).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "lossless point transform must be 0-15, got {value}"
                    )));
                }
                self.lossless_pt = value;
            }
            TjParam::RestartBlocks => {
                self.restart_blocks = value;
            }
            TjParam::RestartRows => {
                self.restart_rows = value;
            }
            TjParam::XDensity => {
                self.x_density = value;
            }
            TjParam::YDensity => {
                self.y_density = value;
            }
            TjParam::DensityUnits => {
                if !(0..=2).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "density units must be 0-2, got {value}"
                    )));
                }
                self.density_units = value;
            }
            TjParam::MaxMemory => {
                self.max_memory = value;
            }
            TjParam::MaxPixels => {
                self.max_pixels = value;
            }
            TjParam::BottomUp => {
                self.bottom_up = if value != 0 { 1 } else { 0 };
            }
            TjParam::NoRealloc => {
                self.no_realloc = if value != 0 { 1 } else { 0 };
            }
            TjParam::StopOnWarning => {
                self.stop_on_warning = if value != 0 { 1 } else { 0 };
            }
            TjParam::SaveMarkers => {
                if !(0..=4).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "save markers level must be 0-4, got {value}"
                    )));
                }
                self.save_markers = value;
            }
        }
        Ok(())
    }

    /// Get a parameter value (like `tj3Get`).
    pub fn get(&self, param: TjParam) -> i32 {
        match param {
            TjParam::Quality => self.quality,
            TjParam::Subsampling => self.subsampling,
            TjParam::Width => self.width,
            TjParam::Height => self.height,
            TjParam::Precision => self.precision,
            TjParam::ColorSpace => self.color_space,
            TjParam::FastUpSample => self.fast_upsample,
            TjParam::FastDct => self.fast_dct,
            TjParam::Optimize => self.optimize,
            TjParam::Progressive => self.progressive,
            TjParam::ScanLimit => self.scan_limit,
            TjParam::Arithmetic => self.arithmetic,
            TjParam::Lossless => self.lossless,
            TjParam::LosslessPsv => self.lossless_psv,
            TjParam::LosslessPt => self.lossless_pt,
            TjParam::RestartBlocks => self.restart_blocks,
            TjParam::RestartRows => self.restart_rows,
            TjParam::XDensity => self.x_density,
            TjParam::YDensity => self.y_density,
            TjParam::DensityUnits => self.density_units,
            TjParam::MaxMemory => self.max_memory,
            TjParam::MaxPixels => self.max_pixels,
            TjParam::BottomUp => self.bottom_up,
            TjParam::NoRealloc => self.no_realloc,
            TjParam::StopOnWarning => self.stop_on_warning,
            TjParam::SaveMarkers => self.save_markers,
        }
    }

    /// Set ICC profile (like `tj3SetICCProfile`).
    pub fn set_icc_profile(&mut self, profile: Option<Vec<u8>>) {
        self.icc_profile = profile;
    }

    /// Get ICC profile (like `tj3GetICCProfile`).
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Set scaling factor for decompression.
    ///
    /// Accepts exactly what [`ScalingFactor::try_new`] accepts — the 16 JPEG
    /// IDCT scaling factors from 1/8 to 2/1, in upstream's form — and returns
    /// its [`JpegError::Unsupported`] for anything else, leaving the current
    /// factor unchanged.
    pub fn set_scaling_factor(&mut self, num: u32, denom: u32) -> Result<()> {
        self.scaling_factor = ScalingFactor::try_new(num, denom)?;
        Ok(())
    }

    /// Set the cropping region for decompression; `None`, or a region whose
    /// four fields are all zero (`TJUNCROPPED`), clears it.
    ///
    /// The region is stored as given and validated by [`Self::decompress`],
    /// against the image it decodes and the scaling factor in force then,
    /// with `tj3SetCroppingRegion`'s rules and messages
    /// (`turbojpeg.c:2068-2115`); see [`Self::resolve_cropping_region`]. A
    /// `width` or `height` of 0 means "to the right / bottom edge", as it
    /// does upstream. A region that does not fit is refused with
    /// [`JpegError::InvalidCropRegion`], never clamped (P4-197, #618).
    ///
    /// Upstream validates at *set* time instead, against the header its
    /// handle last read, and refuses outright when it has read none. That
    /// check is [`Self::resolve_cropping_region`], which the C ABI's
    /// `tj3SetCroppingRegion` calls before storing. It is kept out of this
    /// setter so a decode's outcome depends only on the handle's
    /// configuration and its input — not on which image an earlier call
    /// happened to read.
    pub fn set_cropping_region(&mut self, region: Option<CropRegion>) {
        self.cropping_region = region.filter(|region| *region != Self::UNCROPPED);
    }

    /// `tj3SetCroppingRegion`'s set-time validation (`turbojpeg.c:2086-2111`),
    /// against the frame header this handle last read — through
    /// [`Self::decompress_header`] or [`Self::decompress`] — at the current
    /// scaling factor. Returns the region upstream would store: a `width` or
    /// `height` of 0 filled in to the edge. It stores nothing.
    ///
    /// Refusals, in upstream's order, each an
    /// [`JpegError::InvalidCropRegion`] carrying upstream's message:
    ///
    /// * no header read yet — "JPEG header has not yet been read";
    /// * a lossless frame, or a precision other than 8 or 12;
    /// * a subsampling TurboJPEG cannot classify;
    /// * `x` not a multiple of the scaled iMCU width — refused, where
    ///   [`Decoder::set_crop`] aligns it down, because TurboJPEG documents the
    ///   refusal and its callers size their buffers from the region they
    ///   passed;
    /// * the region extending past the scaled image.
    ///
    /// The all-zero `TJUNCROPPED` region and negative fields are the caller's
    /// to handle first, as upstream handles them before this point.
    pub fn resolve_cropping_region(&self, region: CropRegion) -> Result<CropRegion> {
        let geometry: CroppingGeometry =
            self.cropping_geometry
                .ok_or_else(|| JpegError::InvalidCropRegion {
                    reason: alloc::string::String::from("JPEG header has not yet been read"),
                })?;
        geometry.resolve(region, self.scaling_factor)
    }

    /// `TJUNCROPPED`: the all-zero region that means "no cropping".
    const UNCROPPED: CropRegion = CropRegion {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
    };

    /// The 12- and 16-bit decompress paths take no region yet (P4-219), so a
    /// stored one is refused rather than ignored: ignoring it returns the
    /// whole image to a caller — through the C ABI, into a buffer — that
    /// sized for the region. Upstream crops at 12 bits and refuses a crop on
    /// the lossless frames 16-bit output requires.
    fn refuse_unhonoured_crop(region: Option<CropRegion>, path: &str) -> Result<()> {
        match region {
            Some(_) => Err(JpegError::Unsupported(format!(
                "cropping is not implemented for {path} decompression (P4-219)"
            ))),
            None => Ok(()),
        }
    }

    /// Get available scaling factors.
    ///
    /// Returns all supported (numerator, denominator) pairs for JPEG decompression scaling.
    pub fn scaling_factors() -> Vec<(u32, u32)> {
        ScalingFactor::SUPPORTED
            .iter()
            .map(|factor: &ScalingFactor| (factor.num(), factor.denom()))
            .collect()
    }

    /// Convert TJ3 integer to `ColorSpace` enum. -1 (TJCS_DEFAULT) returns None.
    fn tj_to_color_space(val: i32) -> Option<ColorSpace> {
        match val {
            -1 => None, // TJCS_DEFAULT: auto-detect
            0 => Some(ColorSpace::Rgb),
            1 => Some(ColorSpace::YCbCr),
            2 => Some(ColorSpace::Grayscale),
            3 => Some(ColorSpace::Cmyk),
            4 => Some(ColorSpace::Ycck),
            _ => None,
        }
    }

    /// Upstream's "must be specified" gates (`turbojpeg-mp.c:95-98`): a
    /// lossy compress refuses until the caller supplies quality and
    /// subsampling; a lossless one consults neither (P4-155, #539).
    fn require_lossy_params(&self) -> Result<()> {
        if self.lossless != 0 {
            return Ok(());
        }
        if self.quality == -1 {
            return Err(JpegError::CorruptData(alloc::string::String::from(
                "TJPARAM_QUALITY must be specified",
            )));
        }
        if self.subsampling == -1 {
            return Err(JpegError::CorruptData(alloc::string::String::from(
                "TJPARAM_SUBSAMP must be specified",
            )));
        }
        Ok(())
    }

    /// Convert the subsampling integer to the `Subsampling` enum.
    fn subsampling_enum(&self) -> Subsampling {
        match self.subsampling {
            0 => Subsampling::S444,
            1 => Subsampling::S422,
            2 => Subsampling::S420,
            // 3 => Grayscale — handled by pixel format, default to S444
            3 => Subsampling::S444,
            4 => Subsampling::S440,
            5 => Subsampling::S411,
            6 => Subsampling::S441,
            7 => Subsampling::S410,
            8 => Subsampling::S24,
            // `set` rejects anything outside 0-8, so the only value reaching
            // here is the P4-155 unset sentinel, and only on the lossless
            // path that `require_lossy_params` waves through. Upstream's
            // `setCompDefaults` returns before touching the sampling factors
            // there (`turbojpeg.c:381-386`), leaving `jpeg_set_defaults`'
            // 2x2 luma (`jcparam.c:378-381`) — which is this S420.
            _ => Subsampling::S420,
        }
    }

    /// Apply all handle parameters to an Encoder builder.
    fn configure_encoder<'a>(
        &'a self,
        encoder: crate::api::encoder::Encoder<'a>,
    ) -> crate::api::encoder::Encoder<'a> {
        // `-1` is the P4-155 unset sentinel. Lossy paths never reach here
        // with it — `require_lossy_params` refused already — so it flows only
        // where quality is not consulted (lossless), and the placeholder is
        // unobservable. Casting the sentinel itself would wrap to 255 and
        // underflow `quality_scaling`'s `200 - q * 2`.
        let effective_quality: u8 = if self.quality == -1 {
            75
        } else {
            self.quality as u8
        };
        // The `_ => S420` arm of `subsampling_enum` is only correct for the
        // unset sentinel on the lossless path (see its comment); make the
        // invariant enforceable rather than documentary.
        debug_assert!(
            self.subsampling != -1 || self.lossless != 0,
            "unset subsampling reached configure_encoder on a lossy path — \
             require_lossy_params must refuse first (P4-155)"
        );
        let mut enc = encoder
            .quality(effective_quality)
            .subsampling(self.subsampling_enum())
            .optimize_huffman(self.optimize != 0)
            .progressive(self.progressive != 0)
            .arithmetic(self.arithmetic != 0)
            .lossless(self.lossless != 0)
            .lossless_predictor(self.lossless_psv as u8)
            .lossless_point_transform(self.lossless_pt as u8);

        if self.fast_dct != 0 {
            enc = enc.dct_method(DctMethod::IsFast);
        }

        if self.x_density != 1 || self.y_density != 1 || self.density_units != 0 {
            enc = enc.density(
                self.density_units as u8,
                self.x_density as u16,
                self.y_density as u16,
            );
        }

        if self.restart_blocks > 0 {
            enc = enc.restart_blocks(self.restart_blocks as u16);
        } else if self.restart_rows > 0 {
            enc = enc.restart_rows(self.restart_rows as u16);
        }

        if let Some(ref icc) = self.icc_profile {
            enc = enc.icc_profile(icc);
        }

        // Wire ColorSpace override (TJCS_DEFAULT=-1 means auto-detect)
        if let Some(cs) = Self::tj_to_color_space(self.color_space) {
            enc = enc.colorspace(cs);
        }

        // TJSAMP_GRAY (3): RGB/CMYK input must be converted to a grayscale
        // JPEG. `subsampling_enum()` falls back to S444 for this case; the
        // actual "make it grayscale" switch is `grayscale_from_color`.
        if self.subsampling == 3 {
            enc = enc.grayscale_from_color(true);
        }

        enc
    }

    /// Compress pixels to JPEG using current handle parameters.
    ///
    /// Delegates to the existing `Encoder` builder, translating handle parameters
    /// into the appropriate encoder configuration.
    pub fn compress(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
    ) -> Result<Vec<u8>> {
        use crate::api::encoder::Encoder;

        self.require_lossy_params()?;

        // Wire BottomUp: flip rows before encoding
        if self.bottom_up != 0 {
            let bpp: usize = pixel_format.bytes_per_pixel();
            let row_bytes: usize = width * bpp;
            let mut flipped: Vec<u8> = Vec::with_capacity(pixels.len());
            for row in (0..height).rev() {
                flipped.extend_from_slice(&pixels[row * row_bytes..(row + 1) * row_bytes]);
            }
            let encoder = Encoder::new(&flipped, width, height, pixel_format);
            return self.configure_encoder(encoder).encode();
        }

        let encoder = Encoder::new(pixels, width, height, pixel_format);
        self.configure_encoder(encoder).encode()
    }

    /// Compress pixels into a caller-supplied output buffer (like `tj3Compress8`
    /// with `TJPARAM_NOREALLOC`).
    ///
    /// Encodes with the current handle parameters and writes the resulting JPEG
    /// bytes into `out`. Returns the number of bytes written on success.
    ///
    /// If the encoded stream does not fit in `out`, returns
    /// `JpegError::BufferTooSmall { need, got }`. The `TJPARAM_NOREALLOC`
    /// parameter is honored: because `out` is a borrowed slice it cannot be
    /// grown, so `BufferTooSmall` is returned regardless of the parameter
    /// value; the parameter primarily documents the caller's intent and
    /// matches C libjpeg-turbo's API shape. Callers wanting automatic growth
    /// should use `compress()` (returns `Vec<u8>`).
    pub fn compress_into(
        &self,
        pixels: &[u8],
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
        out: &mut [u8],
    ) -> Result<usize> {
        let jpeg: Vec<u8> = self.compress(pixels, width, height, pixel_format)?;
        if jpeg.len() > out.len() {
            return Err(JpegError::BufferTooSmall {
                need: jpeg.len(),
                got: out.len(),
            });
        }
        out[..jpeg.len()].copy_from_slice(&jpeg);
        Ok(jpeg.len())
    }

    /// Compress 12-bit pixels to JPEG (like `tj3Compress12`).
    ///
    /// Uses handle quality, subsampling, and lossless parameters.
    /// When `TJPARAM_LOSSLESS` is set and `TJPARAM_PRECISION` is in 9..=12,
    /// the SOF marker precision field reflects the stored precision value.
    pub fn compress_12bit(
        &self,
        pixels: &[i16],
        width: usize,
        height: usize,
        num_components: usize,
    ) -> Result<Vec<u8>> {
        self.require_lossy_params()?;
        // The unset sentinel flows here only under `TJPARAM_LOSSLESS` — but
        // unlike `configure_encoder`, this path does not route the flag:
        // `compress_12bit_with_precision` encodes lossy regardless and feeds
        // the placeholder into its quant tables (P4-158). The placeholder
        // keeps that pre-existing mis-shape at its historical quality rather
        // than wrapping -1 to 255; P4-158 is the fix, not this substitution.
        crate::api::precision::compress_12bit_with_precision(
            pixels,
            width,
            height,
            num_components,
            if self.quality == -1 {
                75
            } else {
                self.quality as u8
            },
            self.subsampling_enum(),
            None,
        )
    }

    /// Compress 12-bit pixels to JPEG with an explicit SOF precision override.
    ///
    /// `precision` must be in 9..=12 when lossless is active. For lossy encode
    /// the precision is still set in the SOF1 marker but quantisation tables
    /// remain 12-bit scaled.
    pub fn compress_12bit_with_precision(
        &self,
        pixels: &[i16],
        width: usize,
        height: usize,
        num_components: usize,
        precision: u8,
    ) -> Result<Vec<u8>> {
        self.require_lossy_params()?;
        // The unset sentinel flows here only under `TJPARAM_LOSSLESS` — but
        // unlike `configure_encoder`, this path does not route the flag:
        // `compress_12bit_with_precision` encodes lossy regardless and feeds
        // the placeholder into its quant tables (P4-158). The placeholder
        // keeps that pre-existing mis-shape at its historical quality rather
        // than wrapping -1 to 255; P4-158 is the fix, not this substitution.
        crate::api::precision::compress_12bit_with_precision(
            pixels,
            width,
            height,
            num_components,
            if self.quality == -1 {
                75
            } else {
                self.quality as u8
            },
            self.subsampling_enum(),
            Some(precision),
        )
    }

    /// Compress 16-bit pixels to lossless JPEG (the encode `tj3Compress16`
    /// performs).
    ///
    /// 16-bit is always lossless (SOF3). Uses handle lossless predictor
    /// and point transform parameters.
    ///
    /// Note this is *not* the C entry point's full contract: `tj3Compress16`
    /// first rejects a handle with `TJPARAM_LOSSLESS` unset, because upstream
    /// does (`jcmaster.c:206`). Encoding lossless anyway was P4-150. This
    /// method is a Rust API whose signature already says lossless, so it has
    /// nothing to reject — the acceptance rule belongs to the shim.
    pub fn compress_16bit(
        &self,
        pixels: &[u16],
        width: usize,
        height: usize,
        num_components: usize,
    ) -> Result<Vec<u8>> {
        crate::api::precision::compress_16bit_with_precision(
            pixels,
            width,
            height,
            num_components,
            self.lossless_psv as u8,
            self.lossless_pt as u8,
            None,
        )
    }

    /// Compress 16-bit pixels with an explicit SOF precision override.
    ///
    /// `precision` must be in 2..=16 (SOF3-legal, lossless-only path).
    pub fn compress_16bit_with_precision(
        &self,
        pixels: &[u16],
        width: usize,
        height: usize,
        num_components: usize,
        precision: u8,
    ) -> Result<Vec<u8>> {
        crate::api::precision::compress_16bit_with_precision(
            pixels,
            width,
            height,
            num_components,
            self.lossless_psv as u8,
            self.lossless_pt as u8,
            Some(precision),
        )
    }

    /// Header-only read (matches `tj3DecompressHeader`, `turbojpeg.c:1872-1927`).
    ///
    /// Parses the markers up to the first SOS and no further, as
    /// `jpeg_read_header` does — no entropy data is decoded and no later scan
    /// is located (P4-142) — and publishes the thirteen parameters
    /// `setDecompParameters` writes, exactly as [`Self::decompress`] does.
    /// Like upstream, it consults neither the scaling factor nor the cropping
    /// region, and applies neither `TJPARAM_MAXPIXELS` nor
    /// `TJPARAM_SCANLIMIT`: those belong to the decompress that follows.
    ///
    /// With `TJPARAM_SAVEMARKERS` 2 or 4 the ICC profile is captured, as by a
    /// decompress; at 0, 1 or 3 the handle's profile is cleared, also as by a
    /// decompress (the two share one buffer here, P4-198).
    ///
    /// Refuses, after publishing, a frame whose colour space TurboJPEG cannot
    /// name ("Could not determine colorspace of JPEG image", `:1919-1920`).
    pub fn decompress_header(&mut self, data: &[u8]) -> Result<()> {
        self.read_header(data)?;
        if self.color_space == TJCS_DEFAULT {
            return Err(JpegError::Unsupported(alloc::string::String::from(
                "Could not determine colorspace of JPEG image",
            )));
        }
        Ok(())
    }

    /// `jpeg_read_header` and the `setDecompParameters` call after it, shared
    /// by all four decompress entry points (`turbojpeg.c:1904-1907`,
    /// `turbojpeg-mp.c:187-190`): parse up to the first SOS, refuse what
    /// libjpeg's `get_sof` / `initial_setup` refuse at that point — so such a
    /// frame publishes nothing, as upstream's never reaches
    /// `setDecompParameters` — then publish.
    ///
    /// Only the first SOS is read, so a stream whose later scans are
    /// truncated, or exceed `TJPARAM_SCANLIMIT`, still publishes before the
    /// decode that follows refuses it — upstream's order too.
    fn read_header(&mut self, data: &[u8]) -> Result<()> {
        let header: Decoder<'_> = Decoder::new_header_only(data, self.decode_limits())?;
        Self::check_frame_like_initial_setup(header.header())?;
        self.publish_header(&header)
    }

    /// The frame checks libjpeg makes while `jpeg_read_header` runs, before
    /// TurboJPEG can publish anything: an empty frame (`get_sof`,
    /// `JERR_EMPTY_IMAGE`), a dimension above `JPEG_MAX_DIMENSION` and an
    /// illegal sample precision (`initial_setup`, `jdinput.c:54-70`). The
    /// component count and sampling factors are refused by the marker parser
    /// already. A dimension refusal keeps the `LimitExceeded` shape the
    /// decode-time check (`DecodeLimits::check_frame`) gives the same frame.
    fn check_frame_like_initial_setup(frame: &FrameHeader) -> Result<()> {
        // jmorecfg.h JPEG_MAX_DIMENSION.
        const JPEG_MAX_DIMENSION: u64 = 65_500;
        if frame.height == 0 {
            return Err(JpegError::CorruptData(alloc::string::String::from(
                "Empty JPEG image (DNL not supported)",
            )));
        }
        for (what, actual) in [
            ("image width", u64::from(frame.width)),
            ("image height", u64::from(frame.height)),
        ] {
            if actual > JPEG_MAX_DIMENSION {
                return Err(JpegError::LimitExceeded {
                    what,
                    actual,
                    limit: JPEG_MAX_DIMENSION,
                });
            }
        }
        let legal_precision: bool = if frame.is_lossless {
            (2..=16).contains(&frame.precision)
        } else {
            matches!(frame.precision, 8 | 12)
        };
        if !legal_precision {
            return Err(JpegError::Unsupported(format!(
                "Unsupported JPEG data precision {}",
                frame.precision
            )));
        }
        Ok(())
    }

    /// Everything a header read leaves on the handle, done once, right after
    /// the parse, by all four decompress entry points: the thirteen
    /// parameters ([`Self::publish_decomp_parameters`]) and the ICC profile —
    /// with `TJPARAM_SAVEMARKERS` 2 or 4 the frame's (or `None` when it has
    /// none), otherwise `None`.
    ///
    /// The ICC half is here, rather than after a successful decode, because a
    /// later compress reads the field: the handle keeps one ICC buffer where
    /// upstream keeps two (P4-198, #619), so an entry point that wrote it at a
    /// different point — or not at all — would make a compress depend on
    /// *which* decode ran last, and whether it got past the header, rather
    /// than on the last header read.
    fn publish_header(&mut self, decoder: &Decoder<'_>) -> Result<()> {
        self.publish_decomp_parameters(decoder);
        self.icc_profile = match self.save_markers {
            2 | 4 => decoder.icc_profile()?,
            _ => None,
        };
        Ok(())
    }

    /// Publish what upstream's `setDecompParameters` (`turbojpeg.c:514-536`)
    /// writes, from the frame header `decoder` parsed: all thirteen of
    /// `SUBSAMP`, `JPEGWIDTH`, `JPEGHEIGHT`, `PRECISION`, `COLORSPACE`,
    /// `PROGRESSIVE`, `ARITHMETIC`, `LOSSLESS`, `LOSSLESSPSV`, `LOSSLESSPT`,
    /// `XDENSITY`, `YDENSITY` and `DENSITYUNITS` (P4-199, #620).
    ///
    /// Every value is the *frame's*: the SOF's dimensions rather than the
    /// scaled or cropped output's (P4-200, #621) and the SOF's precision
    /// rather than the decode path's (P4-203, #625). `LOSSLESSPSV` and
    /// `LOSSLESSPT` are `dinfo.Ss` / `dinfo.Al` of the first scan, which on a
    /// progressive stream is the DC scan's point transform.
    ///
    /// Called, through [`Self::publish_header`], by every decompress entry
    /// point right after the header parse and before any limit or crop
    /// check, as upstream's shared body calls it before its
    /// `TJPARAM_MAXPIXELS` refusal (`turbojpeg-mp.c:190`, `:195-198`) — so a
    /// refused decode has still published.
    fn publish_decomp_parameters(&mut self, decoder: &Decoder<'_>) {
        let geometry: CroppingGeometry = CroppingGeometry::of(decoder);
        let frame = decoder.header();
        let first_scan = decoder.first_scan_header();
        let density = decoder.density();
        self.subsampling = geometry.subsampling;
        self.width = i32::from(frame.width);
        self.height = i32::from(frame.height);
        self.precision = i32::from(frame.precision);
        self.color_space = match decoder.jpeg_color_space() {
            ColorSpace::Grayscale => 2,
            ColorSpace::Rgb => 0,
            ColorSpace::YCbCr => 1,
            ColorSpace::Cmyk => 3,
            ColorSpace::Ycck => 4,
            ColorSpace::Unknown => TJCS_DEFAULT,
        };
        self.progressive = i32::from(frame.is_progressive);
        self.arithmetic = i32::from(decoder.is_arithmetic());
        self.lossless = i32::from(frame.is_lossless);
        self.lossless_psv = i32::from(first_scan.spec_start);
        self.lossless_pt = i32::from(first_scan.succ_low);
        self.x_density = i32::from(density.x);
        self.y_density = i32::from(density.y);
        self.density_units = match density.unit {
            DensityUnit::Unknown => 0,
            DensityUnit::Dpi => 1,
            DensityUnit::Dpcm => 2,
        };
        self.cropping_geometry = Some(geometry);
    }

    /// Resource limits derived from this handle's TurboJPEG params.
    ///
    /// Built from the params FIRST so SCANLIMIT applies from marker parsing
    /// onward, and 0 keeps the C contract's no-library-level-cap semantics
    /// (turbojpeg.h TJPARAM_SCANLIMIT / TJPARAM_MAXMEMORY; the latter is in
    /// MEGABYTES, converted x1048576 like turbojpeg.c).
    fn decode_limits(&self) -> crate::common::types::DecodeLimits {
        let mut limits = crate::common::types::DecodeLimits {
            // C's JPEG_MAX_DIMENSION (65,500) is an unconditional
            // library-level cap ("Maximum supported image dimension"),
            // not a TurboJPEG param — keep it.
            max_width: 65_500,
            max_height: 65_500,
            max_pixels: u64::MAX,
            max_scans: usize::MAX,
            max_memory: None,
        };
        if self.max_pixels > 0 {
            limits.max_pixels = self.max_pixels as u64;
        }
        if self.max_memory > 0 {
            limits.max_memory = Some(self.max_memory as u64 * 1_048_576);
        }
        if self.scan_limit > 0 {
            limits.max_scans = self.scan_limit as usize;
        }
        limits
    }

    /// Everything about a frame that its header alone determines, read
    /// **without decoding any pixel data**.
    ///
    /// `Decoder::new_with_limits` stops after marker parsing, so this costs the
    /// header rather than the image. The handle's resource limits apply —
    /// including `TJPARAM_MAXPIXELS`, which is enforced here rather than left to
    /// the decode that may never happen.
    ///
    /// This exists so C-ABI entry points can apply upstream's header-time
    /// validation order (`turbojpeg.c:2223-2239`) instead of decoding first and
    /// rejecting after. The free functions `decompress_to_yuv_planes` and
    /// friends take no handle, so they cannot see these limits at all (P4-127).
    pub fn inspect_header(&self, data: &[u8]) -> Result<FrameInfo> {
        let decoder = Decoder::new_with_limits(data, self.decode_limits())?;
        let frame = decoder.header();
        let (width, height): (usize, usize) = (frame.width(), frame.height());
        // Upstream applies its maxPixels test right after reading the header
        // and before anything else (turbojpeg.c:2228-2231).
        decoder.limits().check_frame(width, height)?;
        Ok(FrameInfo {
            width,
            height,
            num_components: frame.components.len(),
            subsampling: decoder.jpeg_subsampling(),
        })
    }

    /// Decompress JPEG data using current handle parameters (like
    /// `tj3Decompress8`).
    ///
    /// Publishes the thirteen header parameters first — see
    /// [`Self::decompress_header`] — then applies `TJPARAM_MAXPIXELS` to the
    /// frame, then locates every scan under `TJPARAM_SCANLIMIT`, then applies
    /// the scaling factor and cropping region, then decodes.
    pub fn decompress(&mut self, data: &[u8]) -> Result<Image> {
        self.read_header(data)?;
        let limits: crate::common::types::DecodeLimits = self.decode_limits();
        // Upstream's maxPixels test sits right after setDecompParameters and
        // before scaling or cropping is consulted (`turbojpeg-mp.c:195-198`).
        limits.check_frame(self.width as usize, self.height as usize)?;
        // The full walk, which the decode needs and the header read skipped.
        let mut decoder = Decoder::new_with_limits(data, limits)?;

        // Apply scaling
        if self.scaling_factor != ScalingFactor::default() {
            decoder.set_scale(self.scaling_factor);
        }

        // Apply stop-on-warning
        if self.stop_on_warning != 0 {
            decoder.set_stop_on_warning(true);
        }

        // Wire FastUpSample
        if self.fast_upsample != 0 {
            decoder.set_fast_upsample(true);
        }

        // Wire FastDct
        if self.fast_dct != 0 {
            decoder.set_fast_dct(true);
        }

        // Published above, so a later resolve_cropping_region validates
        // against this frame.
        let geometry: CroppingGeometry = CroppingGeometry::of(&decoder);

        // Apply the crop region, resolved against *this* image with
        // tj3SetCroppingRegion's rules: refused with upstream's message
        // rather than handed to the decoder to clamp (P4-197, #618). For a
        // region the C ABI already resolved at set time this is a no-op
        // unless the image or the scaling factor changed since.
        let applied_crop: Option<CropRegion> = match self.cropping_region {
            Some(region) => {
                let crop: CropRegion = geometry.resolve(region, self.scaling_factor)?;
                // `resolve` checks `x` against `tjMCUWidth[subsamp]`, as
                // upstream does, but the decoder aligns to its own iMCU —
                // `max_h_samp` scaled blocks — and the two differ for sampling
                // factors TurboJPEG classifies loosely (all three components
                // 2x1 is TJSAMP_444, an 8-pixel iMCU, decoded in 16-pixel
                // columns). Upstream catches the difference after
                // `jpeg_crop_scanline` moves `x` (`turbojpeg-mp.c:217-221`);
                // so must we, or the decode is wider than the buffer a caller
                // sized from the region.
                let max_h_samp: usize = decoder
                    .header()
                    .components
                    .iter()
                    .map(|component| component.horizontal_sampling as usize)
                    .max()
                    .unwrap_or(1);
                let decoder_imcu_width: usize = max_h_samp * decoder.output_block_size();
                let aligned_x: usize = (crop.x / decoder_imcu_width) * decoder_imcu_width;
                if aligned_x != crop.x {
                    return Err(JpegError::InvalidCropRegion {
                        reason: format!(
                            "Unexplained mismatch between specified ({}) and\n\
                             actual ({}) cropping region left boundary",
                            crop.x, aligned_x
                        ),
                    });
                }
                decoder.set_crop_region(crop.x, crop.y, crop.width, crop.height);
                Some(crop)
            }
            None => None,
        };

        // Wire SaveMarkers: configure which markers to preserve
        // Matches C TJSM_NONE(0), TJSM_COM(1), TJSM_ALL(2), TJSM_NOICC(3), TJSM_ICC(4)
        match self.save_markers {
            1 => decoder.save_markers(MarkerSaveConfig::Specific(vec![0xFE])),
            2 => decoder.save_markers(MarkerSaveConfig::All),
            3 => decoder.save_markers(MarkerSaveConfig::All), // filter ICC post-decode
            4 => decoder.save_markers(MarkerSaveConfig::Specific(vec![0xE2])),
            _ => {} // 0 = none (default)
        }

        let mut img: Image = decoder.decode_image()?;

        // The decode must have produced exactly the region, as upstream's
        // `turbojpeg-mp.c:222-225` and `:265-276` require. A decode path that
        // does not honour the horizontal crop — the 12-bit one today
        // (P4-219) — would otherwise hand the C ABI an image wider than the
        // caller's buffer.
        if let Some(crop) = applied_crop {
            if img.width != crop.width {
                return Err(JpegError::InvalidCropRegion {
                    reason: format!(
                        "Unexplained mismatch between specified ({}) and\n\
                         actual ({}) cropping region width",
                        crop.width, img.width
                    ),
                });
            }
            if img.height != crop.height {
                return Err(JpegError::InvalidCropRegion {
                    reason: format!(
                        "Unexplained mismatch between specified ({}) and\n\
                         actual ({}) cropping region lower boundary",
                        crop.y + crop.height,
                        crop.y + img.height
                    ),
                });
            }
        }

        // The image's copy of the ICC profile follows the SaveMarkers level
        // too; the handle's was stored with the header (`publish_header`).
        // Level 0/1: no ICC; level 3: all markers except ICC; 2 and 4 keep it.
        match self.save_markers {
            0 | 1 => {
                img.icc_profile = None;
            }
            3 => {
                img.icc_profile = None;
                // Remove ICC APP2 markers from saved_markers
                img.saved_markers
                    .retain(|m| !(m.code == 0xE2 && m.data.starts_with(b"ICC_PROFILE\0")));
            }
            _ => {}
        }

        // Wire BottomUp: flip rows after decoding
        if self.bottom_up != 0 {
            let bpp: usize = img.pixel_format.bytes_per_pixel();
            let row_bytes: usize = img.width * bpp;
            let mut flipped: Vec<u8> = Vec::with_capacity(img.data.len());
            for row in (0..img.height).rev() {
                flipped.extend_from_slice(&img.data[row * row_bytes..(row + 1) * row_bytes]);
            }
            img.data = flipped;
        }

        Ok(img)
    }

    /// Decompress JPEG to 12-bit pixels (like `tj3Decompress12`).
    ///
    /// Returns 12-bit sample data (0-4095). Upstream reaches all three
    /// precisions through one body (`turbojpeg-mp.c:153`), so this publishes
    /// the same thirteen parameters as [`Self::decompress`] and applies the
    /// same handle limits — `TJPARAM_MAXPIXELS`, `TJPARAM_SCANLIMIT` and
    /// `TJPARAM_MAXMEMORY` (P4-199, #620).
    pub fn decompress_12bit(&mut self, data: &[u8]) -> Result<crate::api::precision::Image12> {
        let limits: crate::common::types::DecodeLimits = self.prepare_precision_decode(data)?;
        Self::refuse_unhonoured_crop(self.cropping_region, "12-bit")?;
        crate::api::precision::decompress_12bit_with_limits(data, &limits)
    }

    /// Decompress JPEG to 16-bit pixels (like `tj3Decompress16`).
    ///
    /// Returns 16-bit sample data. Publishes and limits exactly as
    /// [`Self::decompress_12bit`] does.
    pub fn decompress_16bit(&mut self, data: &[u8]) -> Result<crate::api::precision::Image16> {
        let limits: crate::common::types::DecodeLimits = self.prepare_precision_decode(data)?;
        Self::refuse_unhonoured_crop(self.cropping_region, "16-bit")?;
        crate::api::precision::decompress_16bit_with_limits(data, &limits)
    }

    /// The shared head of the 12/16-bit paths, in upstream's order: read the
    /// header and publish, then apply `TJPARAM_MAXPIXELS`
    /// (`turbojpeg-mp.c:190`, `:195-198`). Returns the limits the decode
    /// itself must honour, `TJPARAM_SCANLIMIT` among them.
    fn prepare_precision_decode(
        &mut self,
        data: &[u8],
    ) -> Result<crate::common::types::DecodeLimits> {
        self.read_header(data)?;
        let limits: crate::common::types::DecodeLimits = self.decode_limits();
        limits.check_frame(self.width as usize, self.height as usize)?;
        Ok(limits)
    }
}

impl Default for TjHandle {
    fn default() -> Self {
        Self::new()
    }
}
