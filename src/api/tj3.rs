//! TJ3-compatible handle/parameter API for JPEG compression/decompression.
//!
//! Provides a handle-based interface matching libjpeg-turbo's TurboJPEG 3 API
//! pattern: `tj3Init()`/`tj3Set()`/`tj3Get()`/`tj3Destroy()`. All JPEG
//! parameters are stored in a single `TjHandle` and accessed via `TjParam`.

use crate::common::error::{JpegError, Result};
use crate::common::types::{CropRegion, PixelFormat, ScalingFactor, Subsampling};
use crate::decode::pipeline::{Decoder, Image};

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
    /// TJPARAM_COLORSPACE: Color space (0=RGB, 1=YCbCr, 2=Gray, 3=CMYK, 4=YCCK).
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
    /// TJPARAM_MAXMEMORY: Max memory in bytes (0=unlimited).
    MaxMemory,
    /// TJPARAM_MAXPIXELS: Max image size in pixels (0=unlimited).
    MaxPixels,
    /// TJPARAM_BOTTOMUP: Bottom-up row order (boolean).
    BottomUp,
    /// TJPARAM_NOREALLOC: Use pre-allocated output buffer (boolean).
    NoRealloc,
    /// TJPARAM_STOPONWARNING: Treat warnings as fatal (boolean).
    StopOnWarning,
    /// TJPARAM_SAVEMARKERS: Marker saving config (0=None, 1=All).
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
            color_space: 1, // YCbCr
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
            x_density: 72,
            y_density: 72,
            density_units: 1, // DPI
            max_memory: 0,
            max_pixels: 0,
            bottom_up: 0,
            no_realloc: 0,
            stop_on_warning: 0,
            save_markers: 0,
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
                if !(0..=5).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "subsampling must be 0-5, got {value}"
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
                self.precision = value;
            }
            TjParam::ColorSpace => {
                if !(0..=4).contains(&value) {
                    return Err(JpegError::CorruptData(format!(
                        "color space must be 0-4, got {value}"
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
    /// Only the standard JPEG scaling factors are supported: 1/1, 1/2, 1/4, 1/8.
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
            _ => Subsampling::S420,
        }
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

        let mut encoder = Encoder::new(pixels, width, height, pixel_format)
            .quality(self.quality as u8)
            .subsampling(self.subsampling_enum())
            .optimize_huffman(self.optimize != 0)
            .progressive(self.progressive != 0)
            .arithmetic(self.arithmetic != 0)
            .lossless(self.lossless != 0)
            .lossless_predictor(self.lossless_psv as u8)
            .lossless_point_transform(self.lossless_pt as u8);

        if self.restart_blocks > 0 {
            encoder = encoder.restart_blocks(self.restart_blocks as u16);
        } else if self.restart_rows > 0 {
            encoder = encoder.restart_rows(self.restart_rows as u16);
        }

        if let Some(ref icc) = self.icc_profile {
            encoder = encoder.icc_profile(icc);
        }

        encoder.encode()
    }

    /// Decompress JPEG data using current handle parameters.
    ///
    /// Delegates to the existing `Decoder`, translating handle parameters.
    /// After successful decompression, updates `Width`, `Height`, and `Precision`
    /// to reflect the decoded image.
    pub fn decompress(&mut self, data: &[u8]) -> Result<Image> {
        let mut decoder = Decoder::new(data)?;

        // Apply scaling
        if self.scaling_factor != ScalingFactor::default() {
            decoder.set_scale(self.scaling_factor);
        }

        // Apply stop-on-warning
        if self.stop_on_warning != 0 {
            decoder.set_stop_on_warning(true);
        }

        // Apply max pixels limit
        if self.max_pixels > 0 {
            decoder.set_max_pixels(self.max_pixels as usize);
        }

        // Apply max memory limit
        if self.max_memory > 0 {
            decoder.set_max_memory(self.max_memory as usize);
        }

        // Apply scan limit
        if self.scan_limit > 0 {
            decoder.set_scan_limit(self.scan_limit as u32);
        }

        // Apply crop region
        if let Some(crop) = self.cropping_region {
            decoder.set_crop_region(crop.x, crop.y, crop.width, crop.height);
        }

        let img = decoder.decode_image()?;

        // Update read-only params from decoded image
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
