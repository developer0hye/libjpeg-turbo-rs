//! TJ3-compatible handle/parameter API for JPEG compression/decompression.
//!
//! Provides a handle-based interface matching libjpeg-turbo's TurboJPEG 3 API
//! pattern: `tj3Init()`/`tj3Set()`/`tj3Get()`/`tj3Destroy()`. All JPEG
//! parameters are stored in a single `TjHandle` and accessed via `TjParam`.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
use crate::common::error::{JpegError, Result};
use crate::common::types::{
    ColorSpace, CropRegion, DctMethod, DensityUnit, MarkerSaveConfig, PixelFormat, ScalingFactor,
    Subsampling,
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
    /// TJPARAM_QUALITY: Quality factor 1-100.
    Quality,
    /// TJPARAM_SUBSAMP: Chroma subsampling (0=444, 1=422, 2=420, 3=Gray, 4=440, 5=411).
    Subsampling,
    /// TJPARAM_JPEGWIDTH: Image width (read-only after decompress).
    Width,
    /// TJPARAM_JPEGHEIGHT: Image height (read-only after decompress).
    Height,
    /// TJPARAM_PRECISION: Sample precision in bits (read-only).
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
}

impl TjHandle {
    /// Create a new TJ3 handle with default parameters (like `tj3Init`).
    pub fn new() -> Self {
        Self {
            quality: 75,
            subsampling: 2, // S420
            width: 0,
            height: 0,
            precision: 8,
            color_space: -1, // TJCS_DEFAULT (auto-detect)
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
    /// Supports all 16 JPEG IDCT scaling factors from 1/8 to 2/1.
    pub fn set_scaling_factor(&mut self, num: u32, denom: u32) -> Result<()> {
        let valid = Self::scaling_factors();
        if !valid.contains(&(num, denom)) {
            return Err(JpegError::CorruptData(format!(
                "unsupported scaling factor {num}/{denom}"
            )));
        }
        self.scaling_factor = ScalingFactor::new(num, denom);
        Ok(())
    }

    /// Set cropping region for decompression.
    pub fn set_cropping_region(&mut self, region: Option<CropRegion>) {
        self.cropping_region = region;
    }

    /// Get available scaling factors.
    ///
    /// Returns all supported (numerator, denominator) pairs for JPEG decompression scaling.
    pub fn scaling_factors() -> Vec<(u32, u32)> {
        vec![
            (2, 1),
            (15, 8),
            (7, 4),
            (13, 8),
            (3, 2),
            (11, 8),
            (5, 4),
            (9, 8),
            (1, 1),
            (7, 8),
            (3, 4),
            (5, 8),
            (1, 2),
            (3, 8),
            (1, 4),
            (1, 8),
        ]
    }

    /// Convert `ColorSpace` enum to TJ3 integer (TJCS_* constants).
    fn color_space_to_tj(cs: ColorSpace) -> i32 {
        match cs {
            ColorSpace::Rgb => 0,
            ColorSpace::YCbCr => 1,
            ColorSpace::Grayscale => 2,
            ColorSpace::Cmyk => 3,
            ColorSpace::Ycck => 4,
            ColorSpace::Unknown => 1, // default to YCbCr
        }
    }

    /// Convert `Subsampling` enum to TJ3 integer (TJSAMP_* constants).
    fn subsampling_to_tj(ss: Subsampling) -> i32 {
        match ss {
            Subsampling::S444 => 0, // TJSAMP_444
            Subsampling::S422 => 1, // TJSAMP_422
            Subsampling::S420 => 2, // TJSAMP_420
            Subsampling::S440 => 4, // TJSAMP_440
            Subsampling::S411 => 5, // TJSAMP_411
            Subsampling::S441 => 6, // TJSAMP_441
            Subsampling::S410 => 7, // TJSAMP_410
            Subsampling::S24 => 8,  // TJSAMP_24
            Subsampling::Unknown => 0,
        }
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
            _ => Subsampling::S420,
        }
    }

    /// Apply all handle parameters to an Encoder builder.
    fn configure_encoder<'a>(
        &'a self,
        encoder: crate::api::encoder::Encoder<'a>,
    ) -> crate::api::encoder::Encoder<'a> {
        let mut enc = encoder
            .quality(self.quality as u8)
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
        crate::api::precision::compress_12bit_with_precision(
            pixels,
            width,
            height,
            num_components,
            self.quality as u8,
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
        crate::api::precision::compress_12bit_with_precision(
            pixels,
            width,
            height,
            num_components,
            self.quality as u8,
            self.subsampling_enum(),
            Some(precision),
        )
    }

    /// Compress 16-bit pixels to lossless JPEG (like `tj3Compress16`).
    ///
    /// 16-bit is always lossless (SOF3). Uses handle lossless predictor
    /// and point transform parameters.
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

    /// Decompress JPEG data using current handle parameters.
    ///
    /// Delegates to the existing `Decoder`, translating handle parameters.
    /// After successful decompression, updates handle state to reflect the
    /// decoded image: `Width`, `Height`, `Precision`, `ColorSpace`,
    /// `Subsampling`, density, and ICC profile.
    /// Header-only decompress (matches `tj3DecompressHeader` semantics).
    ///
    /// Unlike `decompress()`, this method:
    /// - Ignores `scaling_factor` (`JPEGWIDTH`/`JPEGHEIGHT` must reflect the
    ///   ORIGINAL JPEG dimensions per the libjpeg-turbo spec, not the
    ///   scaled output).
    /// - Populates `width`, `height`, `precision`, `color_space`,
    ///   `subsampling`, density, and ICC exactly as `decompress()` would,
    ///   but without producing pixel data.
    ///
    /// Implementation: temporarily reset the scaling factor to 1:1, call
    /// `decompress()`, then restore. The scaled output from a subsequent
    /// `decompress()` call is still governed by the original scaling
    /// factor — this method does NOT clobber user-visible state beyond the
    /// read-only header params.
    pub fn decompress_header(&mut self, data: &[u8]) -> Result<()> {
        let saved: ScalingFactor = self.scaling_factor;
        self.scaling_factor = ScalingFactor::default();
        let result: Result<Image> = self.decompress(data);
        self.scaling_factor = saved;
        result.map(|_| ())
    }

    pub fn decompress(&mut self, data: &[u8]) -> Result<Image> {
        // Build limits from the TurboJPEG params FIRST and construct
        // with them, so SCANLIMIT applies from marker parsing onward and
        // 0 keeps the C contract's no-library-level-cap semantics
        // (turbojpeg.h TJPARAM_SCANLIMIT / TJPARAM_MAXMEMORY; the
        // latter is in MEGABYTES, converted x1048576 like turbojpeg.c).
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

        // Apply crop region
        if let Some(crop) = self.cropping_region {
            decoder.set_crop_region(crop.x, crop.y, crop.width, crop.height);
        }

        // Wire SaveMarkers: configure which markers to preserve
        // Matches C TJSM_NONE(0), TJSM_COM(1), TJSM_ALL(2), TJSM_NOICC(3), TJSM_ICC(4)
        match self.save_markers {
            1 => decoder.save_markers(MarkerSaveConfig::Specific(vec![0xFE])),
            2 => decoder.save_markers(MarkerSaveConfig::All),
            3 => decoder.save_markers(MarkerSaveConfig::All), // filter ICC post-decode
            4 => decoder.save_markers(MarkerSaveConfig::Specific(vec![0xE2])),
            _ => {} // 0 = none (default)
        }

        // Capture JPEG header metadata before decoding
        let jpeg_color_space: ColorSpace = decoder.jpeg_color_space();
        let jpeg_subsampling: Subsampling = decoder.jpeg_subsampling();

        let mut img: Image = decoder.decode_image()?;

        // Update read-only params from decoded image
        self.width = img.width as i32;
        self.height = img.height as i32;
        self.precision = img.precision as i32;

        // Update color space from JPEG header (matches C tj3DecompressHeader)
        self.color_space = Self::color_space_to_tj(jpeg_color_space);

        // Update subsampling from JPEG header
        // Grayscale has no chroma subsampling; map to TJSAMP_GRAY=3 matching C
        self.subsampling = if jpeg_color_space == ColorSpace::Grayscale {
            3 // TJSAMP_GRAY
        } else {
            Self::subsampling_to_tj(jpeg_subsampling)
        };

        // Update density from JFIF header
        self.density_units = match img.density.unit {
            DensityUnit::Unknown => 0,
            DensityUnit::Dpi => 1,
            DensityUnit::Dpcm => 2,
        };
        self.x_density = img.density.x as i32;
        self.y_density = img.density.y as i32;

        // Capture ICC profile based on SaveMarkers level (matches C tj3GetICCProfile semantics)
        // Level 0/1: no ICC extraction; Level 2/4: extract ICC; Level 3: all except ICC
        match self.save_markers {
            0 | 1 => {
                self.icc_profile = None;
                img.icc_profile = None;
            }
            3 => {
                self.icc_profile = None;
                img.icc_profile = None;
                // Remove ICC APP2 markers from saved_markers
                img.saved_markers
                    .retain(|m| !(m.code == 0xE2 && m.data.starts_with(b"ICC_PROFILE\0")));
            }
            _ => {
                // Level 2 (all) and 4 (ICC only): extract ICC to handle while
                // leaving the image copy intact (tj3GetICCProfile symmetry).
                self.icc_profile = img.icc_profile.clone();
            }
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
    /// Returns 12-bit sample data (0-4095). Updates handle `Width`, `Height`,
    /// and `Precision` from the decoded image.
    pub fn decompress_12bit(&mut self, data: &[u8]) -> Result<crate::api::precision::Image12> {
        let img = crate::api::precision::decompress_12bit(data)?;
        self.width = img.width as i32;
        self.height = img.height as i32;
        self.precision = 12;
        Ok(img)
    }

    /// Decompress JPEG to 16-bit pixels (like `tj3Decompress16`).
    ///
    /// Returns 16-bit sample data. Updates handle `Width`, `Height`,
    /// and `Precision` from the decoded image.
    pub fn decompress_16bit(&mut self, data: &[u8]) -> Result<crate::api::precision::Image16> {
        let img = crate::api::precision::decompress_16bit(data)?;
        self.width = img.width as i32;
        self.height = img.height as i32;
        self.precision = img.precision as i32;
        Ok(img)
    }
}

impl Default for TjHandle {
    fn default() -> Self {
        Self::new()
    }
}
