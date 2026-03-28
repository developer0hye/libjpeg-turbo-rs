use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::icc;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use crate::decode::huffman;
use crate::decode::idct_scaled;
use crate::decode::lossless;
use crate::decode::marker::{JpegMetadata, MarkerReader, ScanInfo};
use crate::decode::progressive;
use crate::simd::{self, SimdRoutines};

/// Vertical triangle-filter blend: out[i] = (3*cur[i] + neighbor[i] + 2) >> 2.
/// Auto-vectorizes well on aarch64 with NEON.
#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
#[inline]
fn vertical_blend(cur: &[u8], neighbor: &[u8], output: &mut [u8], width: usize) {
    for i in 0..width {
        output[i] = ((3 * cur[i] as u16 + neighbor[i] as u16 + 2) >> 2) as u8;
    }
}

/// Generic nearest-neighbor upsampling for arbitrary h/v factor combinations.
///
/// Handles non-standard sampling factors like 3x2, 3x1, 1x3, 4x2 that lack
/// dedicated optimized paths. Each input sample is replicated h_factor times
/// horizontally and v_factor times vertically.
fn upsample_generic_nearest(
    input: &[u8],
    in_width: usize,
    in_height: usize,
    output: &mut [u8],
    out_stride: usize,
    h_factor: usize,
    v_factor: usize,
) {
    for y in 0..in_height {
        let in_row: &[u8] = &input[y * in_width..y * in_width + in_width];
        // Build one upsampled row (horizontal replication)
        let out_y_base: usize = y * v_factor;
        let first_out_row: usize = out_y_base * out_stride;
        for (x, &val) in in_row.iter().enumerate() {
            let out_x: usize = x * h_factor;
            for dx in 0..h_factor {
                output[first_out_row + out_x + dx] = val;
            }
        }
        // Replicate the row vertically
        for dy in 1..v_factor {
            let src_start: usize = first_out_row;
            let dst_start: usize = (out_y_base + dy) * out_stride;
            let copy_len: usize = in_width * h_factor;
            output.copy_within(src_start..src_start + copy_len, dst_start);
        }
    }
}

/// Per-component layout info for progressive decoding.
struct CompInfo {
    blocks_x: usize,
    blocks_y: usize,
    h_samp: usize,
    v_samp: usize,
    comp_w: usize,
}

/// Decoded image data.
#[derive(Debug)]
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
    /// COM marker text, if present.
    pub comment: Option<String>,
    /// Pixel density from JFIF header.
    pub density: DensityInfo,
    /// Saved APP/COM markers.
    pub saved_markers: Vec<SavedMarker>,
    /// Warnings accumulated during lenient decoding.
    pub warnings: Vec<DecodeWarning>,
}

impl Image {
    /// Returns the ICC color profile embedded in this JPEG, if any.
    pub fn icc_profile(&self) -> Option<&[u8]> {
        self.icc_profile.as_deref()
    }

    /// Returns the raw EXIF TIFF data, if present.
    pub fn exif_data(&self) -> Option<&[u8]> {
        self.exif_data.as_deref()
    }

    /// Parses and returns the EXIF orientation tag (1-8), if present.
    pub fn exif_orientation(&self) -> Option<u8> {
        self.exif_data
            .as_ref()
            .and_then(|d| crate::common::exif::parse_orientation(d))
    }

    /// Returns all saved markers (APP and COM) collected during decoding.
    ///
    /// Only populated when the decoder was configured with `save_markers()`.
    pub fn markers(&self) -> &[SavedMarker] {
        &self.saved_markers
    }
}

/// JPEG decoder. Orchestrates the full decoding pipeline.
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
    max_pixels: Option<usize>,
    max_memory: Option<usize>,
    scan_limit: Option<u32>,
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
    marker_processors: std::collections::HashMap<u8, Box<dyn Fn(&[u8]) -> Option<Vec<u8>>>>,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let mut reader = MarkerReader::new(data);
        let metadata = reader.read_markers()?;
        let routines = simd::detect();
        Ok(Self {
            metadata,
            raw_data: data,
            routines,
            output_format: None,
            scale: ScalingFactor::default(),
            lenient: false,
            crop_x: None,
            crop_width: None,
            crop_y: None,
            crop_height: None,
            stop_on_warning: false,
            max_pixels: None,
            max_memory: None,
            scan_limit: None,
            fast_upsample: false,
            fast_dct: false,
            dct_method: DctMethod::IsLow,
            block_smoothing: false,
            output_colorspace: None,
            dither_565: false,
            merged_upsample: false,
            marker_processors: std::collections::HashMap::new(),
        })
    }

    pub fn header(&self) -> &FrameHeader {
        &self.metadata.frame
    }

    /// Set the desired output pixel format.
    pub fn set_output_format(&mut self, format: PixelFormat) {
        self.output_format = Some(format);
    }

    /// Set the decompression scaling factor (e.g., 1/2, 1/4, 1/8).
    pub fn set_scale(&mut self, scale: ScalingFactor) {
        self.scale = scale;
    }

    /// Enable lenient mode: continue decoding on errors, filling corrupt areas with gray.
    pub fn set_lenient(&mut self, lenient: bool) {
        self.lenient = lenient;
    }

    /// Set horizontal crop region. Offsets are auto-aligned to iMCU boundaries.
    pub fn set_crop(&mut self, x: usize, width: usize) {
        self.crop_x = Some(x);
        self.crop_width = Some(width);
    }

    /// Set full crop region (horizontal + vertical).
    /// MCU rows outside the vertical range will skip IDCT during decoding.
    pub fn set_crop_region(&mut self, x: usize, y: usize, width: usize, height: usize) {
        self.crop_x = Some(x);
        self.crop_width = Some(width);
        self.crop_y = Some(y);
        self.crop_height = Some(height);
    }

    /// Treat warnings as fatal errors.
    pub fn set_stop_on_warning(&mut self, stop: bool) {
        self.stop_on_warning = stop;
    }

    /// Set maximum allowed image size in pixels. Reject images exceeding this.
    pub fn set_max_pixels(&mut self, limit: usize) {
        self.max_pixels = Some(limit);
    }

    /// Set maximum memory usage in bytes.
    pub fn set_max_memory(&mut self, limit: usize) {
        self.max_memory = Some(limit);
    }

    /// Set maximum number of progressive scans before error.
    pub fn set_scan_limit(&mut self, limit: u32) {
        self.scan_limit = Some(limit);
    }

    /// Enable or disable fast (nearest-neighbor) upsampling.
    pub fn set_fast_upsample(&mut self, fast: bool) {
        self.fast_upsample = fast;
    }

    /// Enable or disable fast DCT for decoding.
    pub fn set_fast_dct(&mut self, fast: bool) {
        self.fast_dct = fast;
    }

    /// Set the DCT/IDCT method for decoding.
    pub fn set_dct_method(&mut self, method: DctMethod) {
        self.dct_method = method;
    }

    /// Enable or disable inter-block smoothing.
    pub fn set_block_smoothing(&mut self, smooth: bool) {
        self.block_smoothing = smooth;
    }

    /// Override the output color space.
    pub fn set_output_colorspace(&mut self, cs: ColorSpace) {
        self.output_colorspace = Some(cs);
    }

    /// Enable or disable ordered dithering for RGB565 output.
    ///
    /// When enabled, applies a 4x4 ordered dither pattern before truncating
    /// 8-bit RGB to 5-6-5, reducing visible banding in smooth gradients.
    /// Matches libjpeg-turbo's dithered RGB565 output mode.
    pub fn set_dither_565(&mut self, dither: bool) {
        self.dither_565 = dither;
    }

    /// Enable merged upsampling optimization (combines upsample + color convert).
    ///
    /// When enabled and subsampling is 4:2:0 or 4:2:2, uses a merged path that
    /// performs chroma upsampling and YCbCr->RGB conversion in a single pass.
    /// This avoids writing upsampled chroma to intermediate buffers, improving
    /// cache behavior. Slightly less accurate than separate fancy upsample
    /// because merged uses box-filter (nearest-neighbor) chroma replication.
    pub fn set_merged_upsample(&mut self, enabled: bool) {
        self.merged_upsample = enabled;
    }

    /// Configure which markers to save during decoding.
    ///
    /// By default, the decoder only parses known markers (JFIF, ICC, EXIF, Adobe, COM)
    /// and discards unknown APP markers. Call this to preserve arbitrary APP/COM markers
    /// in the decoded `Image.saved_markers` field.
    ///
    /// This re-parses the JPEG header with the new configuration.
    pub fn save_markers(&mut self, config: MarkerSaveConfig) {
        let mut reader: MarkerReader<'_> = MarkerReader::new(self.raw_data);
        reader.set_marker_save_config(config);
        if let Ok(metadata) = reader.read_markers() {
            self.metadata = metadata;
        }
    }

    /// Register a custom marker processor callback for a specific marker type.
    pub fn set_marker_processor<F>(&mut self, marker_type: u8, processor: F)
    where
        F: Fn(&[u8]) -> Option<Vec<u8>> + 'static,
    {
        let has_marker: bool = self
            .metadata
            .saved_markers
            .iter()
            .any(|m| m.code == marker_type);
        if !has_marker {
            let mut reader: MarkerReader<'_> = MarkerReader::new(self.raw_data);
            reader.set_marker_save_config(MarkerSaveConfig::Specific(vec![marker_type]));
            if let Ok(metadata) = reader.read_markers() {
                self.metadata = metadata;
            }
        }
        self.marker_processors
            .insert(marker_type, Box::new(processor));
    }

    pub fn decode(data: &'a [u8]) -> Result<Image> {
        let decoder = Self::new(data)?;
        decoder.decode_image()
    }

    pub fn decode_to(data: &'a [u8], format: PixelFormat) -> Result<Image> {
        let mut decoder = Self::new(data)?;
        decoder.set_output_format(format);
        decoder.decode_image()
    }

    #[inline(always)]
    #[allow(dead_code)]
    fn idct_islow(&self, coeffs: &[i16; 64], quant: &[u16; 64], output: &mut [u8; 64]) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::idct::neon_idct_islow(coeffs, quant, output);
        }

        #[allow(unreachable_code)]
        (self.routines.idct_islow)(coeffs, quant, output)
    }

    /// IDCT writing directly to a strided destination buffer (no intermediate copy).
    ///
    /// # Safety
    /// `output` must point to at least `7 * stride + 8` writable bytes.
    #[inline(always)]
    unsafe fn idct_islow_strided(
        &self,
        coeffs: &[i16; 64],
        quant: &[u16; 64],
        output: *mut u8,
        stride: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::idct::neon_idct_islow_strided(
                coeffs, quant, output, stride,
            );
        }

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                return crate::simd::x86_64::avx2_idct::avx2_idct_islow_strided(
                    coeffs, quant, output, stride,
                );
            }
            // SSE2 fallback: IDCT into temp buffer, then copy row-by-row.
            let mut tmp = [0u8; 64];
            (self.routines.idct_islow)(coeffs, quant, &mut tmp);
            for row in 0..8 {
                std::ptr::copy_nonoverlapping(
                    tmp.as_ptr().add(row * 8),
                    output.add(row * stride),
                    8,
                );
            }
            return;
        }

        // Scalar fallback: IDCT into temp buffer, then copy row-by-row.
        #[allow(unreachable_code)]
        {
            let mut tmp = [0u8; 64];
            (self.routines.idct_islow)(coeffs, quant, &mut tmp);
            for row in 0..8 {
                std::ptr::copy_nonoverlapping(
                    tmp.as_ptr().add(row * 8),
                    output.add(row * stride),
                    8,
                );
            }
        }
    }

    /// Scale-aware IDCT dispatch: picks 8x8, 4x4, 2x2, or 1x1 based on block_size.
    ///
    /// # Safety
    /// `output` must point to sufficient writable bytes for the chosen block_size × stride.
    #[inline(always)]
    unsafe fn idct_scaled_strided(
        &self,
        coeffs: &[i16; 64],
        quant: &[u16; 64],
        output: *mut u8,
        stride: usize,
        block_size: usize,
    ) {
        match block_size {
            8 => self.idct_islow_strided(coeffs, quant, output, stride),
            4 => idct_scaled::idct_4x4_strided(coeffs, quant, output, stride),
            2 => idct_scaled::idct_2x2_strided(coeffs, quant, output, stride),
            1 => idct_scaled::idct_1x1_strided(coeffs, quant, output, stride),
            _ => unreachable!("invalid block_size: {}", block_size),
        }
    }

    #[inline(always)]
    fn ycbcr_to_rgb_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_rgb_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        (self.routines.ycbcr_to_rgb_row)(y, cb, cr, out, width)
    }

    #[inline(always)]
    fn ycbcr_to_rgba_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_rgba_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_rgba_row(y, cb, cr, out, width)
    }

    #[inline(always)]
    fn ycbcr_to_bgr_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_bgr_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_bgr_row(y, cb, cr, out, width)
    }

    #[inline(always)]
    fn ycbcr_to_bgra_row(&self, y: &[u8], cb: &[u8], cr: &[u8], out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::color::neon_ycbcr_to_bgra_row(y, cb, cr, out, width);
        }

        #[allow(unreachable_code)]
        crate::decode::color::ycbcr_to_bgra_row(y, cb, cr, out, width)
    }

    /// Dispatch color conversion for one row based on the target pixel format.
    /// `row_index` is the output row number, used for ordered dithering in RGB565 mode.
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn color_convert_row(
        &self,
        format: PixelFormat,
        y: &[u8],
        cb: &[u8],
        cr: &[u8],
        out: &mut [u8],
        width: usize,
        row_index: usize,
    ) {
        match format {
            PixelFormat::Rgb => self.ycbcr_to_rgb_row(y, cb, cr, out, width),
            PixelFormat::Rgba => self.ycbcr_to_rgba_row(y, cb, cr, out, width),
            PixelFormat::Bgr => self.ycbcr_to_bgr_row(y, cb, cr, out, width),
            PixelFormat::Bgra => self.ycbcr_to_bgra_row(y, cb, cr, out, width),
            PixelFormat::Rgbx => {
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 0, 1, 2, 3)
            }
            PixelFormat::Bgrx => {
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 2, 1, 0, 3)
            }
            PixelFormat::Xrgb => {
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 1, 2, 3, 0)
            }
            PixelFormat::Xbgr => {
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 3, 2, 1, 0)
            }
            PixelFormat::Argb => {
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 1, 2, 3, 0)
            }
            PixelFormat::Abgr => {
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 3, 2, 1, 0)
            }
            PixelFormat::Rgb565 => {
                if self.dither_565 {
                    crate::decode::color::ycbcr_to_rgb565_dithered_row(
                        y, cb, cr, out, width, row_index,
                    )
                } else {
                    crate::decode::color::ycbcr_to_rgb565_row(y, cb, cr, out, width)
                }
            }
            PixelFormat::Grayscale | PixelFormat::Cmyk => {
                unreachable!("grayscale/cmyk handled separately")
            }
        }
    }

    /// Merged H2V1 upsample + color convert dispatch.
    #[inline(always)]
    fn merged_h2v1(y_row: &[u8], cb_row: &[u8], cr_row: &[u8], rgb_out: &mut [u8], width: usize) {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                crate::simd::x86_64::avx2_merged::avx2_merged_h2v1_ycbcr_to_rgb(
                    y_row, cb_row, cr_row, rgb_out, width,
                );
                return;
            }
        }

        crate::decode::merged_upsample::merged_h2v1_ycbcr_to_rgb(
            y_row, cb_row, cr_row, rgb_out, width,
        );
    }

    /// Merged H2V2 upsample + color convert dispatch.
    #[inline(always)]
    fn merged_h2v2(
        y_row0: &[u8],
        y_row1: &[u8],
        cb_row: &[u8],
        cr_row: &[u8],
        rgb_out0: &mut [u8],
        rgb_out1: &mut [u8],
        width: usize,
    ) {
        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                crate::simd::x86_64::avx2_merged::avx2_merged_h2v2_ycbcr_to_rgb(
                    y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width,
                );
                return;
            }
        }

        crate::decode::merged_upsample::merged_h2v2_ycbcr_to_rgb(
            y_row0, y_row1, cb_row, cr_row, rgb_out0, rgb_out1, width,
        );
    }

    #[inline(always)]
    fn fancy_upsample_h2v1(&self, input: &[u8], in_width: usize, output: &mut [u8]) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            return crate::simd::aarch64::upsample::neon_fancy_upsample_h2v1(
                input, in_width, output,
            );
        }

        #[allow(unreachable_code)]
        (self.routines.fancy_upsample_h2v1)(input, in_width, output)
    }

    /// Fancy h2v2 upsample. On aarch64 this uses a dedicated helper that
    /// fuses the two vertical blends into one pass before the h2v1 stage.
    fn fancy_h2v2(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
    ) {
        #[cfg(all(target_arch = "aarch64", feature = "simd"))]
        {
            crate::simd::aarch64::upsample::neon_fancy_upsample_h2v2(
                input, in_width, in_height, output, out_width,
            )
        }

        #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
        {
            let mut row_above = vec![0u8; in_width];
            let mut row_below = vec![0u8; in_width];

            for y in 0..in_height {
                let cur_row = &input[y * in_width..(y + 1) * in_width];
                let above = if y > 0 {
                    &input[(y - 1) * in_width..y * in_width]
                } else {
                    cur_row
                };
                let below = if y + 1 < in_height {
                    &input[(y + 1) * in_width..(y + 2) * in_width]
                } else {
                    cur_row
                };

                vertical_blend(cur_row, above, &mut row_above, in_width);
                vertical_blend(cur_row, below, &mut row_below, in_width);

                let out_y_top = y * 2;
                let out_y_bot = y * 2 + 1;
                self.fancy_upsample_h2v1(
                    &row_above,
                    in_width,
                    &mut output[out_y_top * out_width..],
                );
                self.fancy_upsample_h2v1(
                    &row_below,
                    in_width,
                    &mut output[out_y_bot * out_width..],
                );
            }
        }
    }

    /// Fancy h1v2 upsample: vertical-only 2x (for S440).
    /// Each input row produces two output rows using triangle filter vertically.
    /// Horizontal samples are copied 1:1.
    fn fancy_h1v2(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
    ) {
        for y in 0..in_height {
            let cur_row = &input[y * in_width..(y + 1) * in_width];
            let above = if y > 0 {
                &input[(y - 1) * in_width..y * in_width]
            } else {
                cur_row
            };
            let below = if y + 1 < in_height {
                &input[(y + 1) * in_width..(y + 2) * in_width]
            } else {
                cur_row
            };

            let out_y_top = y * 2;
            let out_y_bot = y * 2 + 1;
            // split_at_mut to get non-overlapping mutable slices
            let (top_half, bot_half) = output.split_at_mut(out_y_bot * out_width);
            let out_top = &mut top_half[out_y_top * out_width..out_y_top * out_width + in_width];
            let out_bot = &mut bot_half[..in_width];
            // Inline vertical blend: out[i] = (3*cur[i] + neighbor[i] + 2) >> 2
            for i in 0..in_width {
                out_top[i] = ((3 * cur_row[i] as u16 + above[i] as u16 + 2) >> 2) as u8;
                out_bot[i] = ((3 * cur_row[i] as u16 + below[i] as u16 + 2) >> 2) as u8;
            }
        }
    }

    /// Fancy h1v4 upsample: vertical-only 4x (for S441).
    /// Each input row produces four output rows using triangle filter vertically.
    /// Horizontal samples are copied 1:1.
    fn fancy_h1v4(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
    ) {
        for y in 0..in_height {
            let cur_row = &input[y * in_width..(y + 1) * in_width];
            let above = if y > 0 {
                &input[(y - 1) * in_width..y * in_width]
            } else {
                cur_row
            };
            let below = if y + 1 < in_height {
                &input[(y + 1) * in_width..(y + 2) * in_width]
            } else {
                cur_row
            };

            // 4 output rows per input row, mirroring h4v1 interpolation positions
            // Positions: -3/8, -1/8, +1/8, +3/8 relative to center
            let out_row_0 = y * 4;
            let out_row_1 = y * 4 + 1;
            let out_row_2 = y * 4 + 2;
            let out_row_3 = y * 4 + 3;

            for i in 0..in_width {
                let a = above[i] as u16;
                let c = cur_row[i] as u16;
                let b = below[i] as u16;
                output[out_row_0 * out_width + i] = ((a * 3 + c * 5 + 4) >> 3) as u8;
                output[out_row_1 * out_width + i] = ((a + c * 7 + 4) >> 3) as u8;
                output[out_row_2 * out_width + i] = ((c * 7 + b + 4) >> 3) as u8;
                output[out_row_3 * out_width + i] = ((c * 5 + b * 3 + 4) >> 3) as u8;
            }
        }
    }

    /// Fancy h4v2 upsample: 4x horizontal, 2x vertical (for 4:1:0).
    /// Each input row produces two output rows (vertical 2x via triangle filter),
    /// then each row is horizontally expanded 4x using the h4v1 filter.
    fn fancy_h4v2(
        &self,
        input: &[u8],
        in_width: usize,
        in_height: usize,
        output: &mut [u8],
        out_width: usize,
    ) {
        // Temporary buffer for one vertically-blended row before horizontal expansion.
        let mut vert_row = vec![0u8; in_width];

        for y in 0..in_height {
            let cur_row = &input[y * in_width..(y + 1) * in_width];
            let above = if y > 0 {
                &input[(y - 1) * in_width..y * in_width]
            } else {
                cur_row
            };
            let below = if y + 1 < in_height {
                &input[(y + 1) * in_width..(y + 2) * in_width]
            } else {
                cur_row
            };

            let out_y_top: usize = y * 2;
            let out_y_bot: usize = y * 2 + 1;

            // Vertical blend toward above, then horizontal 4x expand.
            for i in 0..in_width {
                vert_row[i] = ((3 * cur_row[i] as u16 + above[i] as u16 + 2) >> 2) as u8;
            }
            Self::fancy_upsample_h4v1(&vert_row, in_width, &mut output[out_y_top * out_width..]);

            // Vertical blend toward below, then horizontal 4x expand.
            for i in 0..in_width {
                vert_row[i] = ((3 * cur_row[i] as u16 + below[i] as u16 + 2) >> 2) as u8;
            }
            Self::fancy_upsample_h4v1(&vert_row, in_width, &mut output[out_y_bot * out_width..]);
        }
    }

    /// Fancy h4v1 upsample: horizontal-only 4x (for S411).
    /// Each input sample produces 4 output samples using triangle filter horizontally.
    fn fancy_upsample_h4v1(input: &[u8], in_width: usize, output: &mut [u8]) {
        if in_width == 0 {
            return;
        }
        for x in 0..in_width {
            let left = if x > 0 { input[x - 1] } else { input[x] };
            let cur = input[x];
            let right = if x + 1 < in_width {
                input[x + 1]
            } else {
                input[x]
            };
            // Generate 4 output samples using linear interpolation
            // Positions: -3/8, -1/8, +1/8, +3/8 relative to center
            let ox = x * 4;
            output[ox] = ((left as u16 * 3 + cur as u16 * 5 + 4) >> 3) as u8;
            output[ox + 1] = ((left as u16 + cur as u16 * 7 + 4) >> 3) as u8;
            output[ox + 2] = ((cur as u16 * 7 + right as u16 + 4) >> 3) as u8;
            output[ox + 3] = ((cur as u16 * 5 + right as u16 * 3 + 4) >> 3) as u8;
        }
    }

    /// Decode baseline (single-scan) into component planes.
    /// Returns component planes and any warnings (in lenient mode).
    /// `mcu_row_range`: optional (start, end) MCU row range for IDCT skip optimization.
    /// When set, only MCU rows in [start, end) get IDCT; planes are sized for this range only.
    fn decode_baseline_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        block_size: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        let scan = &self.metadata.scan;

        // Determine MCU row range for IDCT
        let (mcu_y_start, mcu_y_end) = self.mcu_row_range(mcus_y, block_size, frame);

        // Allocate component planes (full MCU-aligned size, uninitialized).
        // SAFETY: The MCU decode loop + IDCT writes every pixel before reading.
        #[allow(clippy::uninit_vec)]
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * block_size;
                let comp_h = mcus_y * comp.vertical_sampling as usize * block_size;
                let size: usize = comp_w * comp_h;
                let mut v: Vec<u8> = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        let mcu_plan = entropy::resolve_mcu_plan(
            frame,
            scan,
            &self.metadata.dc_huffman_tables,
            &self.metadata.ac_huffman_tables,
        )?;

        struct CompLayout {
            comp_w: usize,
            h_blocks: usize,
            v_blocks: usize,
        }
        let comp_layouts: Vec<CompLayout> = frame
            .components
            .iter()
            .map(|comp| CompLayout {
                comp_w: mcus_x * comp.horizontal_sampling as usize * block_size,
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
            })
            .collect();

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut mcu_count: u16 = 0;
        let mut coeffs = [0i16; 64];
        let mut warnings: Vec<DecodeWarning> = Vec::new();
        let total_mcus = mcus_x * mcus_y;

        // Fast path: non-lenient, no cropping — tight loop with minimal branching.
        // The lenient/crop path is below with full error recovery support.
        if !self.lenient && mcu_y_start == 0 && mcu_y_end == mcus_y {
            let restart_interval: u16 = self.metadata.restart_interval;
            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        bit_reader.reset();
                        mcu_decoder.reset();
                    }

                    for (comp_idx, layout) in comp_layouts.iter().enumerate() {
                        let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;
                        let plan = &mcu_plan[comp_idx];

                        for v in 0..layout.v_blocks {
                            for h in 0..layout.h_blocks {
                                mcu_decoder.decode_block(
                                    &mut bit_reader,
                                    plan.comp_idx,
                                    plan.dc_table,
                                    plan.ac_table,
                                    &mut coeffs,
                                )?;

                                let block_x: usize = (mcu_x * layout.h_blocks + h) * block_size;
                                let block_y: usize = (mcu_y * layout.v_blocks + v) * block_size;
                                let dst_offset: usize = block_y * layout.comp_w + block_x;

                                unsafe {
                                    let dst: *mut u8 =
                                        component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                    self.idct_scaled_strided(
                                        &coeffs,
                                        qt_values,
                                        dst,
                                        layout.comp_w,
                                        block_size,
                                    );
                                }
                            }
                        }
                    }

                    mcu_count += 1;

                    if bit_reader.is_eof() && (mcu_count as usize) < total_mcus {
                        return Err(JpegError::UnexpectedEof);
                    }
                }
            }
        } else {
            // General path: lenient mode with error recovery + crop support
            'mcu_loop: for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if self.metadata.restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(self.metadata.restart_interval)
                    {
                        bit_reader.reset();
                        mcu_decoder.reset();
                    }

                    let mut mcu_error = false;

                    for (comp_idx, layout) in comp_layouts.iter().enumerate() {
                        let qt_values = &quant_tables[comp_idx].values;
                        let plan = &mcu_plan[comp_idx];

                        for v in 0..layout.v_blocks {
                            for h in 0..layout.h_blocks {
                                let decode_result = mcu_decoder.decode_block(
                                    &mut bit_reader,
                                    plan.comp_idx,
                                    plan.dc_table,
                                    plan.ac_table,
                                    &mut coeffs,
                                );

                                match decode_result {
                                    Ok(()) => {}
                                    Err(e) if self.lenient => {
                                        coeffs = [0i16; 64];
                                        if !mcu_error {
                                            warnings.push(DecodeWarning::HuffmanError {
                                                mcu_x,
                                                mcu_y,
                                                message: e.to_string(),
                                            });
                                            mcu_error = true;
                                        }
                                        if matches!(e, JpegError::UnexpectedEof) {
                                            warnings.push(DecodeWarning::TruncatedData {
                                                decoded_mcus: mcu_count as usize,
                                                total_mcus,
                                            });
                                            for plane in &mut component_planes {
                                                plane.fill(128);
                                            }
                                            break 'mcu_loop;
                                        }
                                        mcu_decoder.reset();
                                    }
                                    Err(e) => return Err(e),
                                }

                                if mcu_y >= mcu_y_start && mcu_y < mcu_y_end {
                                    let block_x: usize = (mcu_x * layout.h_blocks + h) * block_size;
                                    let block_y: usize = (mcu_y * layout.v_blocks + v) * block_size;
                                    let dst_offset: usize = block_y * layout.comp_w + block_x;

                                    unsafe {
                                        let dst: *mut u8 =
                                            component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                        self.idct_scaled_strided(
                                            &coeffs,
                                            qt_values,
                                            dst,
                                            layout.comp_w,
                                            block_size,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    mcu_count += 1;

                    if bit_reader.is_eof() && (mcu_count as usize) < total_mcus {
                        if self.lenient {
                            warnings.push(DecodeWarning::TruncatedData {
                                decoded_mcus: mcu_count as usize,
                                total_mcus,
                            });
                            break 'mcu_loop;
                        } else {
                            return Err(JpegError::UnexpectedEof);
                        }
                    }
                }
            }
        }

        Ok((component_planes, warnings))
    }

    /// Decode arithmetic-coded planes (SOF9 sequential).
    fn decode_arithmetic_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        block_size: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        let scan = &self.metadata.scan;

        // Allocate component planes
        #[allow(clippy::uninit_vec)]
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .map(|comp| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * block_size;
                let comp_h = mcus_y * comp.vertical_sampling as usize * block_size;
                let size = comp_w * comp_h;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        struct CompLayout {
            comp_w: usize,
            h_blocks: usize,
            v_blocks: usize,
        }
        let comp_layouts: Vec<CompLayout> = frame
            .components
            .iter()
            .map(|comp| CompLayout {
                comp_w: mcus_x * comp.horizontal_sampling as usize * block_size,
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
            })
            .collect();

        // Build component map from scan selectors
        let scan_comps: Vec<(usize, usize, usize)> = scan
            .components
            .iter()
            .map(|sc| {
                let comp_idx = frame
                    .components
                    .iter()
                    .position(|fc| fc.id == sc.component_id)
                    .unwrap_or(0);
                (
                    comp_idx,
                    sc.dc_table_index as usize,
                    sc.ac_table_index as usize,
                )
            })
            .collect();

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut arith = ArithDecoder::new(entropy_data, 0);

        // Set conditioning parameters
        for i in 0..4 {
            let (l, u) = self.metadata.arith_dc_params[i];
            arith.set_dc_conditioning(i, l, u);
            arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
        }

        let mut coeffs: [i16; 64];

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                for &(comp_idx, dc_tbl, ac_tbl) in &scan_comps {
                    let layout = &comp_layouts[comp_idx];
                    let qt_values = &quant_tables[comp_idx].values;

                    for v in 0..layout.v_blocks {
                        for h in 0..layout.h_blocks {
                            coeffs = [0i16; 64];

                            // Arithmetic decode DC + AC
                            arith.decode_dc_sequential(&mut coeffs, comp_idx, dc_tbl)?;
                            arith.decode_ac_sequential(&mut coeffs, ac_tbl)?;

                            // IDCT
                            let bx = mcu_x * layout.h_blocks + h;
                            let by = mcu_y * layout.v_blocks + v;
                            let x_offset = bx * block_size;
                            let y_offset = by * block_size;

                            let plane = &mut component_planes[comp_idx];
                            let stride = layout.comp_w;

                            if block_size == 8 {
                                let out_ptr =
                                    unsafe { plane.as_mut_ptr().add(y_offset * stride + x_offset) };
                                unsafe {
                                    self.idct_islow_strided(&coeffs, qt_values, out_ptr, stride);
                                }
                            } else {
                                // Scaled IDCT
                                unsafe {
                                    let out_ptr =
                                        plane.as_mut_ptr().add(y_offset * stride + x_offset);
                                    self.idct_scaled_strided(
                                        &coeffs, qt_values, out_ptr, stride, block_size,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok((component_planes, Vec::new()))
    }

    /// Decode arithmetic progressive (SOF10) into component planes.
    /// Accumulates DCT coefficients across all scans using ArithDecoder, then runs IDCT.
    #[allow(clippy::too_many_arguments)]
    fn decode_arithmetic_progressive_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        _max_h: usize,
        _max_v: usize,
        block_size: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        // Per-component coefficient buffers
        let comp_infos: Vec<CompInfo> = frame
            .components
            .iter()
            .map(|comp| {
                let h_samp = comp.horizontal_sampling as usize;
                let v_samp = comp.vertical_sampling as usize;
                CompInfo {
                    blocks_x: mcus_x * h_samp,
                    blocks_y: mcus_y * v_samp,
                    h_samp,
                    v_samp,
                    comp_w: mcus_x * h_samp * block_size,
                }
            })
            .collect();

        // Allocate coefficient buffers (zero-initialized for progressive accumulation)
        let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_infos
            .iter()
            .map(|ci| vec![[0i16; 64]; ci.blocks_x * ci.blocks_y])
            .collect();

        // Process each scan, enforcing scan_limit if set
        for (scan_idx, scan_info) in self.metadata.scans.iter().enumerate() {
            if let Some(limit) = self.scan_limit {
                if scan_idx as u32 >= limit {
                    return Err(JpegError::Unsupported(format!(
                        "progressive scan count {} exceeds limit of {}",
                        scan_idx + 1,
                        limit
                    )));
                }
            }
            let scan_header = &scan_info.header;
            let is_dc = scan_header.spec_start == 0 && scan_header.spec_end == 0;
            let ah = scan_header.succ_high;
            let al = scan_header.succ_low;
            let ss = scan_header.spec_start;
            let se = scan_header.spec_end;

            let entropy_data = &self.raw_data[scan_info.data_offset..];
            let mut arith = ArithDecoder::new(entropy_data, 0);

            // Set conditioning parameters from DAC markers
            for i in 0..4 {
                let (l, u) = self.metadata.arith_dc_params[i];
                arith.set_dc_conditioning(i, l, u);
                arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
            }

            // Resolve component indices for this scan
            let scan_comp_indices: Vec<usize> = scan_header
                .components
                .iter()
                .map(|sc| {
                    frame
                        .components
                        .iter()
                        .position(|fc| fc.id == sc.component_id)
                        .ok_or_else(|| {
                            JpegError::CorruptData(format!(
                                "scan references unknown component {}",
                                sc.component_id
                            ))
                        })
                })
                .collect::<Result<Vec<_>>>()?;

            if scan_header.components.len() > 1 {
                // Interleaved scan (DC only in progressive)
                for mcu_y in 0..mcus_y {
                    for mcu_x in 0..mcus_x {
                        for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                            let ci = &comp_infos[comp_idx];
                            let scan_comp = &scan_header.components[si];
                            let dc_tbl = scan_comp.dc_table_index as usize;

                            for v in 0..ci.v_samp {
                                for h in 0..ci.h_samp {
                                    let bx = mcu_x * ci.h_samp + h;
                                    let by = mcu_y * ci.v_samp + v;
                                    let block_idx = by * ci.blocks_x + bx;
                                    let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                                    if is_dc && ah == 0 {
                                        arith.decode_dc_first_progressive(
                                            coeffs, comp_idx, dc_tbl, al,
                                        )?;
                                    } else if is_dc {
                                        arith.decode_dc_refine_progressive(coeffs, al)?;
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                // Non-interleaved scan (single component)
                let comp_idx = scan_comp_indices[0];
                let scan_comp = &scan_header.components[0];
                let dc_tbl = scan_comp.dc_table_index as usize;
                let ac_tbl = scan_comp.ac_table_index as usize;
                let ci = &comp_infos[comp_idx];

                for by in 0..ci.blocks_y {
                    for bx in 0..ci.blocks_x {
                        let block_idx = by * ci.blocks_x + bx;
                        let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                        if is_dc && ah == 0 {
                            arith.decode_dc_first_progressive(coeffs, comp_idx, dc_tbl, al)?;
                        } else if is_dc {
                            arith.decode_dc_refine_progressive(coeffs, al)?;
                        } else if ah == 0 {
                            arith.decode_ac_first_progressive(coeffs, ac_tbl, ss, se, al)?;
                        } else {
                            arith.decode_ac_refine_progressive(coeffs, ac_tbl, ss, se, al)?;
                        }
                    }
                }
            }
        }

        // IDCT all blocks into component planes
        #[allow(clippy::uninit_vec)]
        let mut component_planes: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let size = ci.comp_w * ci.blocks_y * block_size;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        for (comp_idx, ci) in comp_infos.iter().enumerate() {
            let qt_values = &quant_tables[comp_idx].values;
            for by in 0..ci.blocks_y {
                for bx in 0..ci.blocks_x {
                    let block_idx = by * ci.blocks_x + bx;
                    let coeffs = &coeff_bufs[comp_idx][block_idx];

                    let px_x = bx * block_size;
                    let px_y = by * block_size;
                    let dst_offset = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(coeffs, qt_values, dst, ci.comp_w, block_size);
                    }
                }
            }
        }

        Ok((component_planes, Vec::new()))
    }

    /// Decode progressive (multi-scan) into component planes.
    /// Accumulates DCT coefficients across all scans, then runs IDCT.
    #[allow(clippy::too_many_arguments)]
    fn decode_progressive_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        block_size: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        // Per-component coefficient buffers: blocks_x * blocks_y blocks of 64 coefficients
        let comp_infos: Vec<CompInfo> = frame
            .components
            .iter()
            .map(|comp| {
                let h_samp = comp.horizontal_sampling as usize;
                let v_samp = comp.vertical_sampling as usize;
                CompInfo {
                    blocks_x: mcus_x * h_samp,
                    blocks_y: mcus_y * v_samp,
                    h_samp,
                    v_samp,
                    comp_w: mcus_x * h_samp * block_size,
                }
            })
            .collect();

        // Allocate coefficient buffers (zero-initialized for progressive accumulation)
        let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_infos
            .iter()
            .map(|ci| vec![[0i16; 64]; ci.blocks_x * ci.blocks_y])
            .collect();

        // Process each scan, enforcing scan_limit if set
        for (scan_idx, scan_info) in self.metadata.scans.iter().enumerate() {
            if let Some(limit) = self.scan_limit {
                if scan_idx as u32 >= limit {
                    return Err(JpegError::Unsupported(format!(
                        "progressive scan count {} exceeds limit of {}",
                        scan_idx + 1,
                        limit
                    )));
                }
            }
            self.decode_progressive_scan(
                frame,
                scan_info,
                &comp_infos,
                &mut coeff_bufs,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
            )?;
        }

        // IDCT all blocks into component planes
        #[allow(clippy::uninit_vec)]
        let mut component_planes: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let size = ci.comp_w * ci.blocks_y * block_size;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        for (comp_idx, ci) in comp_infos.iter().enumerate() {
            let qt_values = &quant_tables[comp_idx].values;
            for by in 0..ci.blocks_y {
                for bx in 0..ci.blocks_x {
                    let block_idx = by * ci.blocks_x + bx;
                    let coeffs = &coeff_bufs[comp_idx][block_idx];

                    let px_x = bx * block_size;
                    let px_y = by * block_size;
                    let dst_offset = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(coeffs, qt_values, dst, ci.comp_w, block_size);
                    }
                }
            }
        }

        // Progressive decoding doesn't have per-MCU error recovery yet;
        // errors in scans propagate normally.
        Ok((component_planes, Vec::new()))
    }

    /// Decode one progressive scan's entropy data into the coefficient buffers.
    #[allow(clippy::too_many_arguments)]
    fn decode_progressive_scan(
        &self,
        frame: &FrameHeader,
        scan_info: &ScanInfo,
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
    ) -> Result<()> {
        let scan = &scan_info.header;
        let ss = scan.spec_start;
        let se = scan.spec_end;
        let ah = scan.succ_high;
        let al = scan.succ_low;
        let is_dc = ss == 0 && se == 0;

        let entropy_data = &self.raw_data[scan_info.data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);

        // Resolve component indices for this scan
        let scan_comp_indices: Vec<usize> = scan
            .components
            .iter()
            .map(|sc| {
                frame
                    .components
                    .iter()
                    .position(|fc| fc.id == sc.component_id)
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "scan references unknown component {}",
                            sc.component_id
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        if scan.components.len() > 1 {
            // Interleaved scan (DC only in progressive)
            self.decode_progressive_interleaved(
                scan_info,
                &scan_comp_indices,
                comp_infos,
                coeff_bufs,
                &mut bit_reader,
                mcus_x,
                mcus_y,
                is_dc,
                ss,
                se,
                ah,
                al,
            )
        } else {
            // Non-interleaved scan (single component)
            let comp_idx = scan_comp_indices[0];
            let scan_comp = &scan.components[0];
            self.decode_progressive_non_interleaved(
                scan_info,
                scan_comp,
                comp_idx,
                comp_infos,
                coeff_bufs,
                &mut bit_reader,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                is_dc,
                ss,
                se,
                ah,
                al,
            )
        }
    }

    /// Decode an interleaved progressive scan (multiple components, DC only).
    #[allow(clippy::too_many_arguments)]
    fn decode_progressive_interleaved(
        &self,
        scan_info: &ScanInfo,
        scan_comp_indices: &[usize],
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        bit_reader: &mut BitReader,
        mcus_x: usize,
        mcus_y: usize,
        is_dc: bool,
        _ss: u8,
        _se: u8,
        ah: u8,
        al: u8,
    ) -> Result<()> {
        let scan = &scan_info.header;
        let mut dc_preds = [0i16; 4];

        // Pre-resolve Huffman tables outside the MCU loop
        let dc_tables: Vec<&HuffmanTable> = scan
            .components
            .iter()
            .map(|sc| Self::resolve_table(&scan_info.dc_huffman_tables, sc.dc_table_index, "DC"))
            .collect::<Result<Vec<_>>>()?;

        // Use countdown for restart interval to avoid modulo in hot loop
        let restart_interval = scan_info.restart_interval as u32;
        // Start at restart_interval so the first MCU doesn't trigger a reset.
        // When restart_interval is 0, countdown is never checked.
        let mut restart_countdown: u32 = restart_interval;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if restart_interval > 0 {
                    if restart_countdown == 0 {
                        bit_reader.reset();
                        dc_preds = [0i16; 4];
                        restart_countdown = restart_interval;
                    }
                    restart_countdown -= 1;
                }

                for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                    let ci = &comp_infos[comp_idx];
                    let dc_table = dc_tables[si];

                    for v in 0..ci.v_samp {
                        for h in 0..ci.h_samp {
                            let bx = mcu_x * ci.h_samp + h;
                            let by = mcu_y * ci.v_samp + v;
                            let block_idx = by * ci.blocks_x + bx;
                            let coeffs = &mut coeff_bufs[comp_idx][block_idx];

                            if is_dc {
                                if ah == 0 {
                                    progressive::decode_dc_first(
                                        bit_reader,
                                        dc_table,
                                        &mut dc_preds[comp_idx],
                                        coeffs,
                                        al,
                                    )?;
                                } else {
                                    progressive::decode_dc_refine(bit_reader, coeffs, al)?;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Decode a non-interleaved progressive scan (single component).
    #[allow(clippy::too_many_arguments)]
    fn decode_progressive_non_interleaved(
        &self,
        scan_info: &ScanInfo,
        scan_comp: &ScanComponentSelector,
        comp_idx: usize,
        comp_infos: &[CompInfo],
        coeff_bufs: &mut [Vec<[i16; 64]>],
        bit_reader: &mut BitReader,
        _mcus_x: usize,
        _mcus_y: usize,
        _max_h: usize,
        _max_v: usize,
        is_dc: bool,
        ss: u8,
        se: u8,
        ah: u8,
        al: u8,
    ) -> Result<()> {
        let ci = &comp_infos[comp_idx];
        let mut dc_pred = 0i16;
        let mut eob_run = 0u16;

        let restart_interval = scan_info.restart_interval as u32;
        let mut restart_countdown: u32 = restart_interval;

        // Pre-resolve tables once before the block loop
        let dc_table = if is_dc {
            Some(Self::resolve_table(
                &scan_info.dc_huffman_tables,
                scan_comp.dc_table_index,
                "DC",
            )?)
        } else {
            None
        };
        let ac_table = if !is_dc || se > 0 {
            Some(Self::resolve_table(
                &scan_info.ac_huffman_tables,
                scan_comp.ac_table_index,
                "AC",
            )?)
        } else {
            None
        };

        // Macro to handle restart interval countdown in each specialized loop.
        macro_rules! restart_check_dc {
            ($bit_reader:expr, $dc_pred:expr, $countdown:expr, $interval:expr) => {
                if $interval > 0 {
                    if $countdown == 0 {
                        $bit_reader.reset();
                        $dc_pred = 0;
                        $countdown = $interval;
                    }
                    $countdown -= 1;
                }
            };
        }
        macro_rules! restart_check_ac {
            ($bit_reader:expr, $eob_run:expr, $countdown:expr, $interval:expr) => {
                if $interval > 0 {
                    if $countdown == 0 {
                        $bit_reader.reset();
                        $eob_run = 0;
                        $countdown = $interval;
                    }
                    $countdown -= 1;
                }
            };
        }

        // Split into four specialized loops to eliminate per-block branches.
        // Use direct iterator over the flat coefficient buffer to avoid
        // per-block `by * blocks_x + bx` index computation and bounds checks.
        let coeff_slice = &mut coeff_bufs[comp_idx];
        let total_blocks = ci.blocks_x * ci.blocks_y;

        if is_dc && ah == 0 {
            let dc_table = dc_table.unwrap();
            for coeffs in coeff_slice[..total_blocks].iter_mut() {
                restart_check_dc!(bit_reader, dc_pred, restart_countdown, restart_interval);
                progressive::decode_dc_first(bit_reader, dc_table, &mut dc_pred, coeffs, al)?;
            }
        } else if is_dc {
            for coeffs in coeff_slice[..total_blocks].iter_mut() {
                if restart_interval > 0 {
                    if restart_countdown == 0 {
                        bit_reader.reset();
                        restart_countdown = restart_interval;
                    }
                    restart_countdown -= 1;
                }
                progressive::decode_dc_refine(bit_reader, coeffs, al)?;
            }
        } else if ah == 0 {
            let ac_table = ac_table.unwrap();
            for coeffs in coeff_slice[..total_blocks].iter_mut() {
                restart_check_ac!(bit_reader, eob_run, restart_countdown, restart_interval);
                progressive::decode_ac_first(
                    bit_reader,
                    ac_table,
                    coeffs,
                    ss,
                    se,
                    al,
                    &mut eob_run,
                )?;
            }
        } else {
            let ac_table = ac_table.unwrap();
            for coeffs in coeff_slice[..total_blocks].iter_mut() {
                restart_check_ac!(bit_reader, eob_run, restart_countdown, restart_interval);
                progressive::decode_ac_refine(
                    bit_reader,
                    ac_table,
                    coeffs,
                    ss,
                    se,
                    al,
                    &mut eob_run,
                )?;
            }
        }

        Ok(())
    }

    /// Resolve a Huffman table by index, returning an error if missing.
    fn resolve_table<'t>(
        tables: &'t [Option<HuffmanTable>; 4],
        index: u8,
        kind: &str,
    ) -> Result<&'t HuffmanTable> {
        tables[index as usize].as_ref().ok_or_else(|| {
            JpegError::CorruptData(format!("missing {} Huffman table {}", kind, index))
        })
    }

    /// Compute the MCU row range [start, end) needed for the vertical crop region.
    /// Returns (0, mcus_y) when no crop is set.
    fn mcu_row_range(
        &self,
        mcus_y: usize,
        block_size: usize,
        frame: &FrameHeader,
    ) -> (usize, usize) {
        let (crop_y, crop_h) = match (self.crop_y, self.crop_height) {
            (Some(y), Some(h)) => (y, h),
            _ => return (0, mcus_y),
        };

        let max_v = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);
        let mcu_pixel_h = max_v * block_size;

        let mcu_start = crop_y / mcu_pixel_h;
        let mcu_end = (crop_y + crop_h).div_ceil(mcu_pixel_h).min(mcus_y);

        (mcu_start, mcu_end)
    }

    /// Reassemble ICC profile from parsed APP2 chunks.
    fn icc_profile(&self) -> Option<Vec<u8>> {
        icc::reassemble_icc_profile(&self.metadata.icc_chunks)
    }

    /// Decode a lossless JPEG (SOF3).
    ///
    /// Lossless JPEG uses Huffman-coded differences + prediction instead of DCT.
    /// No quantization or IDCT is involved.
    fn decode_lossless_image(
        &self,
        frame: &FrameHeader,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        if self.metadata.is_arithmetic {
            self.decode_lossless_arithmetic(frame, width, height, icc_profile, exif_data)
        } else {
            self.decode_lossless_huffman(frame, width, height, icc_profile, exif_data)
        }
    }

    /// Decode lossless JPEG with Huffman entropy coding (SOF3).
    fn decode_lossless_huffman(
        &self,
        frame: &FrameHeader,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let scan = &self.metadata.scan;
        let precision = frame.precision;
        let psv = scan.spec_start; // Predictor selection value (Ss field)
        let pt = scan.succ_low; // Point transform (Al field)

        if !(1..=7).contains(&psv) {
            return Err(JpegError::Unsupported(format!(
                "lossless predictor {} (must be 1-7)",
                psv
            )));
        }

        let num_components = frame.components.len();

        // Resolve DC Huffman tables for each scan component
        let mut dc_tables: Vec<&HuffmanTable> = Vec::with_capacity(num_components);
        for i in 0..scan.components.len().min(num_components) {
            let dc_tbl_idx = scan.components[i].dc_table_index as usize;
            let dc_table = self.metadata.dc_huffman_tables[dc_tbl_idx]
                .as_ref()
                .ok_or_else(|| {
                    JpegError::CorruptData(format!("missing DC Huffman table {}", dc_tbl_idx))
                })?;
            dc_tables.push(dc_table);
        }

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut reader = BitReader::new(entropy_data);

        if num_components == 1 {
            // Single-component (grayscale) lossless decode
            let dc_table = dc_tables[0];
            let mut output = vec![0u16; width * height];
            let mut prev_row: Option<Vec<u16>> = None;

            for y in 0..height {
                let row_start = y * width;
                let mut diffs = Vec::with_capacity(width);
                for _ in 0..width {
                    let diff = huffman::decode_dc_coefficient(&mut reader, dc_table)?;
                    diffs.push(diff);
                }
                lossless::undifference_row(
                    &diffs,
                    prev_row.as_deref(),
                    &mut output[row_start..row_start + width],
                    psv,
                    precision,
                    pt,
                    y == 0,
                );
                prev_row = Some(output[row_start..row_start + width].to_vec());
            }

            self.lossless_output_grayscale(&output, width, height, pt, icc_profile, exif_data)
        } else if num_components == 3 {
            // Multi-component (color) lossless decode — interleaved scan
            let mut comp_planes: Vec<Vec<u16>> =
                (0..3).map(|_| vec![0u16; width * height]).collect();
            let mut prev_rows: Vec<Option<Vec<u16>>> = vec![None; 3];

            for y in 0..height {
                let row_start = y * width;
                let mut comp_diffs: Vec<Vec<i16>> =
                    (0..3).map(|_| Vec::with_capacity(width)).collect();

                // Interleaved: for each pixel, decode diff for each component
                for _ in 0..width {
                    for c in 0..3 {
                        let diff = huffman::decode_dc_coefficient(&mut reader, dc_tables[c])?;
                        comp_diffs[c].push(diff);
                    }
                }

                // Undifference each component
                for c in 0..3 {
                    lossless::undifference_row(
                        &comp_diffs[c],
                        prev_rows[c].as_deref(),
                        &mut comp_planes[c][row_start..row_start + width],
                        psv,
                        precision,
                        pt,
                        y == 0,
                    );
                    prev_rows[c] = Some(comp_planes[c][row_start..row_start + width].to_vec());
                }
            }

            self.lossless_output_color(&comp_planes, width, height, icc_profile, exif_data)
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported for lossless",
                num_components
            )))
        }
    }

    /// Decode lossless JPEG with arithmetic entropy coding (SOF11).
    fn decode_lossless_arithmetic(
        &self,
        frame: &FrameHeader,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        use crate::decode::arithmetic::ArithDecoder;

        let scan = &self.metadata.scan;
        let precision = frame.precision;
        let psv = scan.spec_start;
        let pt = scan.succ_low;

        if !(1..=7).contains(&psv) {
            return Err(JpegError::Unsupported(format!(
                "lossless predictor {} (must be 1-7)",
                psv
            )));
        }

        let num_components = frame.components.len();

        // Resolve DC table indices for each scan component
        let dc_tbl_indices: Vec<usize> = scan
            .components
            .iter()
            .take(num_components)
            .map(|sc| sc.dc_table_index as usize)
            .collect();

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut arith = ArithDecoder::new(entropy_data, 0);

        // Set conditioning parameters from DAC marker
        for i in 0..4 {
            let (l, u) = self.metadata.arith_dc_params[i];
            arith.set_dc_conditioning(i, l, u);
            arith.set_ac_conditioning(i, self.metadata.arith_ac_params[i]);
        }

        if num_components == 1 {
            let dc_tbl = dc_tbl_indices[0];
            let mut output = vec![0u16; width * height];
            let mut prev_row: Option<Vec<u16>> = None;

            for y in 0..height {
                let row_start = y * width;
                let mut diffs = Vec::with_capacity(width);
                for _ in 0..width {
                    // Save previous accumulated DC to extract the raw difference
                    let prev_dc: i32 = arith.last_dc_val[0];
                    let mut block: [i16; 64] = [0i16; 64];
                    arith.decode_dc_sequential(&mut block, 0, dc_tbl)?;
                    let diff: i16 = (arith.last_dc_val[0] - prev_dc) as i16;
                    diffs.push(diff);
                }
                lossless::undifference_row(
                    &diffs,
                    prev_row.as_deref(),
                    &mut output[row_start..row_start + width],
                    psv,
                    precision,
                    pt,
                    y == 0,
                );
                prev_row = Some(output[row_start..row_start + width].to_vec());
            }

            self.lossless_output_grayscale(&output, width, height, pt, icc_profile, exif_data)
        } else if num_components == 3 {
            let mut comp_planes: Vec<Vec<u16>> =
                (0..3).map(|_| vec![0u16; width * height]).collect();
            let mut prev_rows: Vec<Option<Vec<u16>>> = vec![None; 3];

            for y in 0..height {
                let row_start = y * width;
                let mut comp_diffs: Vec<Vec<i16>> =
                    (0..3).map(|_| Vec::with_capacity(width)).collect();

                // Interleaved: for each pixel, decode diff for each component
                for _ in 0..width {
                    for c in 0..3 {
                        let prev_dc: i32 = arith.last_dc_val[c];
                        let mut block: [i16; 64] = [0i16; 64];
                        arith.decode_dc_sequential(&mut block, c, dc_tbl_indices[c])?;
                        let diff: i16 = (arith.last_dc_val[c] - prev_dc) as i16;
                        comp_diffs[c].push(diff);
                    }
                }

                // Undifference each component
                for c in 0..3 {
                    lossless::undifference_row(
                        &comp_diffs[c],
                        prev_rows[c].as_deref(),
                        &mut comp_planes[c][row_start..row_start + width],
                        psv,
                        precision,
                        pt,
                        y == 0,
                    );
                    prev_rows[c] = Some(comp_planes[c][row_start..row_start + width].to_vec());
                }
            }

            self.lossless_output_color(&comp_planes, width, height, icc_profile, exif_data)
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported for lossless",
                num_components
            )))
        }
    }

    /// Convert decoded lossless grayscale samples to output Image.
    fn lossless_output_grayscale(
        &self,
        output: &[u16],
        width: usize,
        height: usize,
        pt: u8,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let out_format = self.output_format.unwrap_or(PixelFormat::Grayscale);
        let bpp = out_format.bytes_per_pixel();

        if out_format == PixelFormat::Grayscale {
            let mut data = Vec::with_capacity(width * height);
            for &sample in output {
                let val = if pt > 0 {
                    ((sample as u32) << pt) as u8
                } else {
                    sample as u8
                };
                data.push(val);
            }
            Ok(Image {
                width,
                height,
                pixel_format: PixelFormat::Grayscale,
                precision: 8,
                data,
                icc_profile,
                exif_data,
                comment: self.metadata.comment.clone(),
                density: self.metadata.density,
                saved_markers: self.metadata.saved_markers.clone(),
                warnings: Vec::new(),
            })
        } else {
            let mut data = Vec::with_capacity(width * height * bpp);
            for &sample in output {
                let val = if pt > 0 {
                    ((sample as u32) << pt) as u8
                } else {
                    sample as u8
                };
                match out_format {
                    PixelFormat::Rgb | PixelFormat::Bgr => {
                        data.push(val);
                        data.push(val);
                        data.push(val);
                    }
                    PixelFormat::Rgba
                    | PixelFormat::Bgra
                    | PixelFormat::Rgbx
                    | PixelFormat::Bgrx
                    | PixelFormat::Argb
                    | PixelFormat::Abgr => {
                        data.push(val);
                        data.push(val);
                        data.push(val);
                        data.push(255);
                    }
                    PixelFormat::Xrgb | PixelFormat::Xbgr => {
                        data.push(255);
                        data.push(val);
                        data.push(val);
                        data.push(val);
                    }
                    PixelFormat::Rgb565 => {
                        let packed: u16 = ((val as u16 >> 3) << 11)
                            | ((val as u16 >> 2) << 5)
                            | (val as u16 >> 3);
                        let bytes: [u8; 2] = packed.to_ne_bytes();
                        data.push(bytes[0]);
                        data.push(bytes[1]);
                    }
                    _ => unreachable!(),
                }
            }
            Ok(Image {
                width,
                height,
                pixel_format: out_format,
                precision: 8,
                data,
                icc_profile,
                exif_data,
                comment: self.metadata.comment.clone(),
                density: self.metadata.density,
                saved_markers: self.metadata.saved_markers.clone(),
                warnings: Vec::new(),
            })
        }
    }

    /// Convert decoded lossless YCbCr component planes to output Image.
    fn lossless_output_color(
        &self,
        comp_planes: &[Vec<u16>],
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let out_format = self.output_format.unwrap_or(PixelFormat::Rgb);
        let bpp = out_format.bytes_per_pixel();
        let mut data = Vec::with_capacity(width * height * bpp);

        for ((&y_pix, &cb_pix), &cr_pix) in comp_planes[0]
            .iter()
            .zip(comp_planes[1].iter())
            .zip(comp_planes[2].iter())
        {
            let y_val = y_pix as i32;
            let cb_val = cb_pix as i32;
            let cr_val = cr_pix as i32;

            // YCbCr to RGB (JFIF convention: Y,Cb,Cr centered at 128)
            let r = (y_val + ((cr_val - 128) * 359 + 128) / 256).clamp(0, 255) as u8;
            let g = (y_val - ((cb_val - 128) * 88 + (cr_val - 128) * 183 - 128) / 256).clamp(0, 255)
                as u8;
            let b = (y_val + ((cb_val - 128) * 454 + 128) / 256).clamp(0, 255) as u8;

            match out_format {
                PixelFormat::Rgb => {
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
                PixelFormat::Bgr => {
                    data.push(b);
                    data.push(g);
                    data.push(r);
                }
                PixelFormat::Rgba | PixelFormat::Rgbx => {
                    data.push(r);
                    data.push(g);
                    data.push(b);
                    data.push(255);
                }
                PixelFormat::Bgra | PixelFormat::Bgrx => {
                    data.push(b);
                    data.push(g);
                    data.push(r);
                    data.push(255);
                }
                PixelFormat::Xrgb | PixelFormat::Argb => {
                    data.push(255);
                    data.push(r);
                    data.push(g);
                    data.push(b);
                }
                PixelFormat::Xbgr | PixelFormat::Abgr => {
                    data.push(255);
                    data.push(b);
                    data.push(g);
                    data.push(r);
                }
                PixelFormat::Rgb565 => {
                    let packed: u16 =
                        ((r as u16 >> 3) << 11) | ((g as u16 >> 2) << 5) | (b as u16 >> 3);
                    let bytes: [u8; 2] = packed.to_ne_bytes();
                    data.push(bytes[0]);
                    data.push(bytes[1]);
                }
                _ => {
                    return Err(JpegError::Unsupported(
                        "cannot convert lossless 3-component to requested format".to_string(),
                    ));
                }
            }
        }

        Ok(Image {
            width,
            height,
            pixel_format: out_format,
            precision: 8,
            data,
            icc_profile,
            exif_data,
            comment: self.metadata.comment.clone(),
            density: self.metadata.density,
            saved_markers: self.metadata.saved_markers.clone(),
            warnings: Vec::new(),
        })
    }

    pub fn decode_image(&self) -> Result<Image> {
        let image: Image = self.decode_image_inner()?;
        if !self.marker_processors.is_empty() {
            for marker in &image.saved_markers {
                if let Some(processor) = self.marker_processors.get(&marker.code) {
                    processor(&marker.data);
                }
            }
        }
        // When stop_on_warning is enabled, any accumulated warning becomes fatal.
        if self.stop_on_warning && !image.warnings.is_empty() {
            let first_warning = &image.warnings[0];
            let detail = match first_warning {
                DecodeWarning::HuffmanError {
                    mcu_x,
                    mcu_y,
                    message,
                } => format!("Huffman error at MCU ({}, {}): {}", mcu_x, mcu_y, message),
                DecodeWarning::TruncatedData {
                    decoded_mcus,
                    total_mcus,
                } => format!("truncated: decoded {} of {} MCUs", decoded_mcus, total_mcus),
            };
            return Err(JpegError::CorruptData(format!(
                "stop_on_warning: {}",
                detail
            )));
        }
        Ok(image)
    }

    fn decode_image_inner(&self) -> Result<Image> {
        let frame = &self.metadata.frame;
        let width = frame.width as usize;
        let height = frame.height as usize;

        // Check pixel limit
        if let Some(max) = self.max_pixels {
            let total = width * height;
            if total > max {
                return Err(JpegError::Unsupported(format!(
                    "image {}x{} ({} pixels) exceeds limit of {}",
                    width, height, total, max
                )));
            }
        }

        // Enforce max_memory: reject if estimated decode memory exceeds limit.
        // Estimate = output_buffer + component_plane_buffers.
        if let Some(max_mem) = self.max_memory {
            let nc = frame.components.len();
            let out_bpp = self
                .output_format
                .unwrap_or(if nc == 1 {
                    PixelFormat::Grayscale
                } else {
                    PixelFormat::Rgb
                })
                .bytes_per_pixel();
            let total_estimated = width * height * out_bpp + width * height * nc;
            if total_estimated > max_mem {
                return Err(JpegError::Unsupported(format!(
                    "estimated decode memory {} bytes exceeds limit of {} bytes",
                    total_estimated, max_mem
                )));
            }
        }

        let icc_profile = self.icc_profile();
        let exif_data = self.metadata.exif_data.clone();

        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported)",
                frame.precision
            )));
        }

        let num_components = frame.components.len();
        let max_h = frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let max_v = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);

        let block_size = self.scale.block_size();
        let mcu_width = max_h * 8;
        let mcu_height = max_v * 8;
        let mcus_x = width.div_ceil(mcu_width);
        let mcus_y = height.div_ceil(mcu_height);
        // Scaled output dimensions
        let scaled_mcu_w = max_h * block_size;
        let scaled_mcu_h = max_v * block_size;
        let full_width = mcus_x * scaled_mcu_w;
        let full_height = mcus_y * scaled_mcu_h;
        // Final output dimensions (may be smaller than full due to MCU alignment)
        let out_width = self.scale.scale_dim(width);
        let out_height = self.scale.scale_dim(height);

        // Lossless JPEG (SOF3/SOF11) — different pipeline, no IDCT/quant
        if frame.is_lossless {
            return self.decode_lossless_image(frame, width, height, icc_profile, exif_data);
        }

        // Pre-resolve quant tables per component (once, not per-block)
        let quant_tables: Vec<&QuantTable> = frame
            .components
            .iter()
            .map(|comp| {
                self.metadata.quant_tables[comp.quant_table_index as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "missing quant table {}",
                            comp.quant_table_index
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        // Decode component planes — different paths for baseline vs progressive vs arithmetic
        let (component_planes, warnings) = if self.metadata.is_arithmetic && frame.is_progressive {
            self.decode_arithmetic_progressive_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                block_size,
            )?
        } else if self.metadata.is_arithmetic {
            self.decode_arithmetic_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                block_size,
            )?
        } else if frame.is_progressive {
            self.decode_progressive_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                block_size,
            )?
        } else {
            self.decode_baseline_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                block_size,
            )?
        };

        // Apply block smoothing if requested
        let (component_planes, warnings) = if self.block_smoothing {
            let mut planes = component_planes;
            for (ci, plane) in planes.iter_mut().enumerate() {
                let comp_w: usize =
                    mcus_x * frame.components[ci].horizontal_sampling as usize * block_size;
                let comp_h: usize =
                    mcus_y * frame.components[ci].vertical_sampling as usize * block_size;
                crate::decode::toggles::apply_block_smoothing(plane, comp_w, comp_h);
            }
            (planes, warnings)
        } else {
            (component_planes, warnings)
        };

        // Handle output colorspace override
        if let Some(cs) = self.output_colorspace {
            return crate::decode::toggles::decode_with_colorspace_override(
                cs,
                &component_planes,
                frame,
                out_width,
                out_height,
                mcus_x,
                block_size,
                icc_profile,
                exif_data,
                self.metadata.comment.clone(),
                self.metadata.density,
                self.metadata.saved_markers.clone(),
                warnings,
            );
        }

        // Upsample and color convert
        if num_components == 1 {
            let out_format = self.output_format.unwrap_or(PixelFormat::Grayscale);
            let comp_w = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;

            if out_format == PixelFormat::Grayscale {
                let mut data = Vec::with_capacity(out_width * out_height);
                for y in 0..out_height {
                    data.extend_from_slice(
                        &component_planes[0][y * comp_w..y * comp_w + out_width],
                    );
                }
                Ok(Image {
                    width: out_width,
                    height: out_height,
                    pixel_format: PixelFormat::Grayscale,
                    precision: 8,
                    data,
                    icc_profile: icc_profile.clone(),
                    exif_data: exif_data.clone(),
                    comment: self.metadata.comment.clone(),
                    density: self.metadata.density,
                    saved_markers: self.metadata.saved_markers.clone(),
                    warnings: warnings.clone(),
                })
            } else {
                // Expand grayscale to requested color format
                let bpp = out_format.bytes_per_pixel();
                let data_size = out_width * out_height * bpp;
                let mut data = Vec::with_capacity(data_size);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    data.set_len(data_size)
                };
                for y in 0..out_height {
                    let row = &component_planes[0][y * comp_w..y * comp_w + out_width];
                    let out_row = &mut data[y * out_width * bpp..(y + 1) * out_width * bpp];
                    // For dithered RGB565, use the dedicated row-level function.
                    if out_format == PixelFormat::Rgb565 && self.dither_565 {
                        crate::decode::color::gray_to_rgb565_dithered_row(
                            row, out_row, out_width, y,
                        );
                        continue;
                    }
                    for x in 0..out_width {
                        let v = row[x];
                        match out_format {
                            PixelFormat::Rgb | PixelFormat::Bgr => {
                                out_row[x * 3] = v;
                                out_row[x * 3 + 1] = v;
                                out_row[x * 3 + 2] = v;
                            }
                            PixelFormat::Rgba
                            | PixelFormat::Bgra
                            | PixelFormat::Rgbx
                            | PixelFormat::Bgrx
                            | PixelFormat::Argb
                            | PixelFormat::Abgr => {
                                out_row[x * 4] = v;
                                out_row[x * 4 + 1] = v;
                                out_row[x * 4 + 2] = v;
                                out_row[x * 4 + 3] = 255;
                            }
                            PixelFormat::Xrgb | PixelFormat::Xbgr => {
                                out_row[x * 4] = 255;
                                out_row[x * 4 + 1] = v;
                                out_row[x * 4 + 2] = v;
                                out_row[x * 4 + 3] = v;
                            }
                            PixelFormat::Rgb565 => {
                                // Grayscale v → pack as R=G=B=v (no dither)
                                let packed: u16 = ((v as u16 >> 3) << 11)
                                    | ((v as u16 >> 2) << 5)
                                    | (v as u16 >> 3);
                                let bytes: [u8; 2] = packed.to_ne_bytes();
                                out_row[x * 2] = bytes[0];
                                out_row[x * 2 + 1] = bytes[1];
                            }
                            PixelFormat::Grayscale | PixelFormat::Cmyk => unreachable!(),
                        }
                    }
                }
                Ok(Image {
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    precision: 8,
                    data,
                    icc_profile: icc_profile.clone(),
                    exif_data: exif_data.clone(),
                    comment: self.metadata.comment.clone(),
                    density: self.metadata.density,
                    saved_markers: self.metadata.saved_markers.clone(),
                    warnings: warnings.clone(),
                })
            }
        } else if num_components == 3 {
            let out_format = self.output_format.unwrap_or(PixelFormat::Rgb);
            if out_format == PixelFormat::Grayscale {
                return Err(JpegError::Unsupported(
                    "cannot convert color JPEG to grayscale".to_string(),
                ));
            }
            let bpp = out_format.bytes_per_pixel();

            let y_plane = &component_planes[0];
            let y_width = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;

            let cb_comp = &frame.components[1];
            let cb_w = mcus_x * cb_comp.horizontal_sampling as usize * block_size;
            let cb_h = mcus_y * cb_comp.vertical_sampling as usize * block_size;

            let h_factor = max_h / cb_comp.horizontal_sampling as usize;
            let v_factor = max_v / cb_comp.vertical_sampling as usize;

            // For 4:4:4, use component planes directly without clone.
            // For subsampled modes, upsample into separate buffers.
            let (cb_data, cr_data, cb_stride, cr_stride): (&[u8], &[u8], usize, usize);

            if h_factor == 1 && v_factor == 1 {
                // 4:4:4: no upsampling needed — reference planes directly
                cb_data = &component_planes[1];
                cr_data = &component_planes[2];
                cb_stride = cb_w;
                cr_stride = cb_w;
            } else {
                // Merged upsample path: combine upsample + color convert in one pass
                // for H2V1 (4:2:2) and H2V2 (4:2:0), avoiding intermediate chroma buffers.
                if self.merged_upsample
                    && out_format == PixelFormat::Rgb
                    && h_factor == 2
                    && (v_factor == 1 || v_factor == 2)
                {
                    let data_size: usize = out_width * out_height * bpp;
                    let mut data: Vec<u8> = Vec::with_capacity(data_size);
                    #[allow(clippy::uninit_vec)]
                    unsafe {
                        data.set_len(data_size)
                    };

                    if v_factor == 1 {
                        // H2V1 (4:2:2): one chroma row per Y row
                        for y in 0..out_height {
                            Self::merged_h2v1(
                                &y_plane[y * y_width..],
                                &component_planes[1][y * cb_w..],
                                &component_planes[2][y * cb_w..],
                                &mut data[y * out_width * bpp..],
                                out_width,
                            );
                        }
                    } else {
                        // H2V2 (4:2:0): one chroma row per 2 Y rows
                        let row_pairs: usize = out_height / 2;
                        for pair in 0..row_pairs {
                            let y0: usize = pair * 2;
                            let y1: usize = pair * 2 + 1;
                            let chroma_row: usize = pair;
                            let out0_start: usize = y0 * out_width * bpp;
                            let out1_start: usize = y1 * out_width * bpp;
                            // Split data into two non-overlapping mutable slices
                            let (top, bottom) = data.split_at_mut(out1_start);
                            Self::merged_h2v2(
                                &y_plane[y0 * y_width..],
                                &y_plane[y1 * y_width..],
                                &component_planes[1][chroma_row * cb_w..],
                                &component_planes[2][chroma_row * cb_w..],
                                &mut top[out0_start..],
                                bottom,
                                out_width,
                            );
                        }
                        // Handle odd height: last row uses H2V1 with last chroma row
                        if out_height & 1 != 0 {
                            let last_y: usize = out_height - 1;
                            let chroma_row: usize = last_y / 2;
                            Self::merged_h2v1(
                                &y_plane[last_y * y_width..],
                                &component_planes[1][chroma_row * cb_w..],
                                &component_planes[2][chroma_row * cb_w..],
                                &mut data[last_y * out_width * bpp..],
                                out_width,
                            );
                        }
                    }

                    return Ok(Image {
                        width: out_width,
                        height: out_height,
                        pixel_format: out_format,
                        precision: 8,
                        data,
                        icc_profile: icc_profile.clone(),
                        exif_data: exif_data.clone(),
                        comment: self.metadata.comment.clone(),
                        density: self.metadata.density,
                        saved_markers: self.metadata.saved_markers.clone(),
                        warnings: warnings.clone(),
                    });
                }

                // Allocate upsampled buffers
                let alloc_size = full_width * full_height;
                let mut cb_full = Vec::with_capacity(alloc_size);
                let mut cr_full = Vec::with_capacity(alloc_size);
                // SAFETY: all code paths below write every element before reading.
                unsafe {
                    cb_full.set_len(alloc_size);
                    cr_full.set_len(alloc_size);
                }

                if self.fast_upsample {
                    // Nearest-neighbor upsampling (fast path)
                    crate::decode::toggles::upsample_nearest(
                        &component_planes[1],
                        cb_w,
                        cb_h,
                        &mut cb_full,
                        full_width,
                        h_factor,
                        v_factor,
                    );
                    crate::decode::toggles::upsample_nearest(
                        &component_planes[2],
                        cb_w,
                        cb_h,
                        &mut cr_full,
                        full_width,
                        h_factor,
                        v_factor,
                    );
                } else if h_factor == 2 && v_factor == 1 {
                    for row in 0..cb_h {
                        self.fancy_upsample_h2v1(
                            &component_planes[1][row * cb_w..],
                            cb_w,
                            &mut cb_full[row * full_width..],
                        );
                        self.fancy_upsample_h2v1(
                            &component_planes[2][row * cb_w..],
                            cb_w,
                            &mut cr_full[row * full_width..],
                        );
                    }
                } else if h_factor == 2 && v_factor == 2 {
                    self.fancy_h2v2(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                    self.fancy_h2v2(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
                } else if h_factor == 1 && v_factor == 2 {
                    // S440: vertical-only 2x
                    self.fancy_h1v2(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                    self.fancy_h1v2(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
                } else if h_factor == 4 && v_factor == 1 {
                    // S411: horizontal-only 4x
                    for row in 0..cb_h {
                        Self::fancy_upsample_h4v1(
                            &component_planes[1][row * cb_w..row * cb_w + cb_w],
                            cb_w,
                            &mut cb_full[row * full_width..],
                        );
                        Self::fancy_upsample_h4v1(
                            &component_planes[2][row * cb_w..row * cb_w + cb_w],
                            cb_w,
                            &mut cr_full[row * full_width..],
                        );
                    }
                } else if h_factor == 4 && v_factor == 2 {
                    // 4:1:0: horizontal 4x + vertical 2x
                    self.fancy_h4v2(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                    self.fancy_h4v2(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
                } else if h_factor == 1 && v_factor == 4 {
                    // S441: vertical-only 4x upsampling
                    self.fancy_h1v4(&component_planes[1], cb_w, cb_h, &mut cb_full, full_width);
                    self.fancy_h1v4(&component_planes[2], cb_w, cb_h, &mut cr_full, full_width);
                } else {
                    // Generic fallback for non-standard sampling factors (e.g. 3x2, 3x1, 1x3, 4x2).
                    // Uses nearest-neighbor replication for any h/v factor combination.
                    upsample_generic_nearest(
                        &component_planes[1],
                        cb_w,
                        cb_h,
                        &mut cb_full,
                        full_width,
                        h_factor,
                        v_factor,
                    );
                    upsample_generic_nearest(
                        &component_planes[2],
                        cb_w,
                        cb_h,
                        &mut cr_full,
                        full_width,
                        h_factor,
                        v_factor,
                    );
                }

                // Rebind as immutable references for color conversion below.
                // We use a trick: leak the Vecs temporarily, do the conversion,
                // then reconstruct and drop them. But simpler: just use a nested scope.
                // Actually, let's just do the color conversion here and return.
                let data_size = out_width * out_height * bpp;
                let mut data = Vec::with_capacity(data_size);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    data.set_len(data_size)
                };
                for y in 0..out_height {
                    self.color_convert_row(
                        out_format,
                        &y_plane[y * y_width..],
                        &cb_full[y * full_width..],
                        &cr_full[y * full_width..],
                        &mut data[y * out_width * bpp..],
                        out_width,
                        y,
                    );
                }

                return Ok(Image {
                    width: out_width,
                    height: out_height,
                    pixel_format: out_format,
                    precision: 8,
                    data,
                    icc_profile: icc_profile.clone(),
                    exif_data: exif_data.clone(),
                    comment: self.metadata.comment.clone(),
                    density: self.metadata.density,
                    saved_markers: self.metadata.saved_markers.clone(),
                    warnings: warnings.clone(),
                });
            }

            // 4:4:4 path (no upsampling)
            let data_size = out_width * out_height * bpp;
            let mut data = Vec::with_capacity(data_size);
            #[allow(clippy::uninit_vec)]
            unsafe {
                data.set_len(data_size)
            };
            for y in 0..out_height {
                self.color_convert_row(
                    out_format,
                    &y_plane[y * y_width..],
                    &cb_data[y * cb_stride..],
                    &cr_data[y * cr_stride..],
                    &mut data[y * out_width * bpp..],
                    out_width,
                    y,
                );
            }

            Ok(Image {
                width: out_width,
                height: out_height,
                pixel_format: out_format,
                precision: 8,
                data,
                icc_profile: icc_profile.clone(),
                exif_data: exif_data.clone(),
                comment: self.metadata.comment.clone(),
                density: self.metadata.density,
                saved_markers: self.metadata.saved_markers.clone(),
                warnings: warnings.clone(),
            })
        } else if num_components == 4 {
            self.decode_4_component(
                &component_planes,
                frame,
                out_width,
                out_height,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                full_width,
                full_height,
                block_size,
                icc_profile,
                exif_data,
                warnings,
            )
        } else {
            Err(JpegError::Unsupported(format!(
                "{} components not yet supported",
                num_components
            )))
        }
    }

    /// Decode JPEG to raw downsampled component planes.
    ///
    /// Returns component planes at their native (potentially subsampled)
    /// resolution, without performing color conversion or upsampling.
    /// This matches libjpeg-turbo's `jpeg_read_raw_data()` functionality.
    pub fn decode_raw(self) -> Result<crate::api::raw_data::RawImage> {
        let frame = &self.metadata.frame;
        let width: usize = frame.width as usize;
        let height: usize = frame.height as usize;
        if frame.precision != 8 {
            return Err(JpegError::Unsupported(format!(
                "sample precision {} (only 8-bit supported)",
                frame.precision
            )));
        }
        let num_components: usize = frame.components.len();
        let max_h: usize = frame
            .components
            .iter()
            .map(|c| c.horizontal_sampling as usize)
            .max()
            .unwrap_or(1);
        let max_v: usize = frame
            .components
            .iter()
            .map(|c| c.vertical_sampling as usize)
            .max()
            .unwrap_or(1);
        let block_size: usize = 8;
        let mcu_width: usize = max_h * 8;
        let mcu_height: usize = max_v * 8;
        let mcus_x: usize = width.div_ceil(mcu_width);
        let mcus_y: usize = height.div_ceil(mcu_height);
        let quant_tables: Vec<&crate::common::quant_table::QuantTable> = frame
            .components
            .iter()
            .map(|comp| {
                self.metadata.quant_tables[comp.quant_table_index as usize]
                    .as_ref()
                    .ok_or_else(|| {
                        JpegError::CorruptData(format!(
                            "missing quant table {}",
                            comp.quant_table_index
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let (component_planes, _warnings) = if self.metadata.is_arithmetic && frame.is_progressive {
            self.decode_arithmetic_progressive_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                block_size,
            )?
        } else if self.metadata.is_arithmetic {
            self.decode_arithmetic_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                block_size,
            )?
        } else if frame.is_progressive {
            self.decode_progressive_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                max_h,
                max_v,
                block_size,
            )?
        } else {
            self.decode_baseline_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                block_size,
            )?
        };
        let mut plane_widths: Vec<usize> = Vec::with_capacity(num_components);
        let mut plane_heights: Vec<usize> = Vec::with_capacity(num_components);
        for comp in &frame.components {
            plane_widths.push(mcus_x * comp.horizontal_sampling as usize * block_size);
            plane_heights.push(mcus_y * comp.vertical_sampling as usize * block_size);
        }
        Ok(crate::api::raw_data::RawImage {
            planes: component_planes,
            plane_widths,
            plane_heights,
            width,
            height,
            num_components,
        })
    }

    /// Determine the JPEG color space from component count and Adobe marker.
    /// Follows the same heuristic as libjpeg-turbo (jdapimin.c).
    fn detect_color_space(&self) -> ColorSpace {
        let num_components = self.metadata.frame.components.len();
        match num_components {
            1 => ColorSpace::Grayscale,
            3 => {
                if self.metadata.saw_adobe_marker && self.metadata.adobe_transform == 0 {
                    ColorSpace::Rgb
                } else {
                    ColorSpace::YCbCr
                }
            }
            4 => {
                if self.metadata.saw_adobe_marker {
                    match self.metadata.adobe_transform {
                        0 => ColorSpace::Cmyk,
                        2 => ColorSpace::Ycck,
                        _ => ColorSpace::Ycck, // default for unknown Adobe transforms
                    }
                } else {
                    ColorSpace::Cmyk // no Adobe marker → assume CMYK
                }
            }
            _ => ColorSpace::YCbCr, // fallback
        }
    }

    /// Decode a 4-component (CMYK/YCCK) image.
    #[allow(clippy::too_many_arguments)]
    fn decode_4_component(
        &self,
        component_planes: &[Vec<u8>],
        frame: &FrameHeader,
        width: usize,
        height: usize,
        mcus_x: usize,
        mcus_y: usize,
        max_h: usize,
        max_v: usize,
        full_width: usize,
        full_height: usize,
        block_size: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
        warnings: Vec<DecodeWarning>,
    ) -> Result<Image> {
        let color_space = self.detect_color_space();
        let out_format = self.output_format.unwrap_or(PixelFormat::Cmyk);

        if out_format == PixelFormat::Grayscale {
            return Err(JpegError::Unsupported(
                "cannot convert CMYK/YCCK to grayscale".to_string(),
            ));
        }

        // Component 0 is always full-resolution (Y or C).
        let comp0_w = mcus_x * frame.components[0].horizontal_sampling as usize * block_size;

        // For YCCK, components 1-2 may be subsampled (chroma), component 3 (K) is full.
        // For CMYK, all components are typically the same resolution.
        let comp1 = &frame.components[1];
        let comp1_w = mcus_x * comp1.horizontal_sampling as usize * block_size;
        let comp1_h = mcus_y * comp1.vertical_sampling as usize * block_size;
        let comp3_w = mcus_x * frame.components[3].horizontal_sampling as usize * block_size;

        let h_factor = max_h / comp1.horizontal_sampling as usize;
        let v_factor = max_v / comp1.vertical_sampling as usize;

        // Upsample chroma if needed (for YCCK subsampled images)
        let (plane1, plane2, p1_stride, p2_stride): (&[u8], &[u8], usize, usize);

        if h_factor == 1 && v_factor == 1 {
            plane1 = &component_planes[1];
            plane2 = &component_planes[2];
            p1_stride = comp1_w;
            p2_stride = comp1_w;
        } else {
            let alloc_size = full_width * full_height;
            let mut p1_full = Vec::with_capacity(alloc_size);
            let mut p2_full = Vec::with_capacity(alloc_size);
            unsafe {
                p1_full.set_len(alloc_size);
                p2_full.set_len(alloc_size);
            }

            if h_factor == 2 && v_factor == 1 {
                for row in 0..comp1_h {
                    self.fancy_upsample_h2v1(
                        &component_planes[1][row * comp1_w..],
                        comp1_w,
                        &mut p1_full[row * full_width..],
                    );
                    self.fancy_upsample_h2v1(
                        &component_planes[2][row * comp1_w..],
                        comp1_w,
                        &mut p2_full[row * full_width..],
                    );
                }
            } else if h_factor == 2 && v_factor == 2 {
                self.fancy_h2v2(
                    &component_planes[1],
                    comp1_w,
                    comp1_h,
                    &mut p1_full,
                    full_width,
                );
                self.fancy_h2v2(
                    &component_planes[2],
                    comp1_w,
                    comp1_h,
                    &mut p2_full,
                    full_width,
                );
            } else {
                // Generic fallback for non-standard 4-component sampling factors.
                upsample_generic_nearest(
                    &component_planes[1],
                    comp1_w,
                    comp1_h,
                    &mut p1_full,
                    full_width,
                    h_factor,
                    v_factor,
                );
                upsample_generic_nearest(
                    &component_planes[2],
                    comp1_w,
                    comp1_h,
                    &mut p2_full,
                    full_width,
                    h_factor,
                    v_factor,
                );
            }

            return self.convert_4comp_output(
                color_space,
                out_format,
                &component_planes[0],
                comp0_w,
                &p1_full,
                full_width,
                &p2_full,
                full_width,
                &component_planes[3],
                comp3_w,
                width,
                height,
                icc_profile,
                exif_data,
                warnings,
            );
        }

        self.convert_4comp_output(
            color_space,
            out_format,
            &component_planes[0],
            comp0_w,
            plane1,
            p1_stride,
            plane2,
            p2_stride,
            &component_planes[3],
            comp3_w,
            width,
            height,
            icc_profile,
            exif_data,
            warnings,
        )
    }

    /// Color-convert 4 component planes to the output format.
    #[allow(clippy::too_many_arguments)]
    fn convert_4comp_output(
        &self,
        color_space: ColorSpace,
        out_format: PixelFormat,
        plane0: &[u8],
        p0_stride: usize,
        plane1: &[u8],
        p1_stride: usize,
        plane2: &[u8],
        p2_stride: usize,
        plane3: &[u8],
        p3_stride: usize,
        width: usize,
        height: usize,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
        warnings: Vec<DecodeWarning>,
    ) -> Result<Image> {
        use crate::decode::color;

        let bpp = out_format.bytes_per_pixel();
        let data_size = width * height * bpp;
        let mut data = Vec::with_capacity(data_size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(data_size)
        };

        for y in 0..height {
            let p0 = &plane0[y * p0_stride..];
            let p1 = &plane1[y * p1_stride..];
            let p2 = &plane2[y * p2_stride..];
            let p3 = &plane3[y * p3_stride..];
            let out = &mut data[y * width * bpp..];

            match (color_space, out_format) {
                // CMYK → CMYK: passthrough
                (ColorSpace::Cmyk, PixelFormat::Cmyk) => {
                    color::cmyk_passthrough_row(p0, p1, p2, p3, out, width);
                }
                // CMYK → RGB/RGBA/BGR/BGRA: direct conversion
                (ColorSpace::Cmyk, PixelFormat::Rgb) => {
                    color::cmyk_to_rgb_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Rgba) => {
                    color::cmyk_to_rgba_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Bgr) => {
                    color::cmyk_to_bgr_row(p0, p1, p2, p3, out, width);
                }
                (ColorSpace::Cmyk, PixelFormat::Bgra) => {
                    color::cmyk_to_bgra_row(p0, p1, p2, p3, out, width);
                }
                // YCCK → CMYK: YCbCr→RGB→invert→CMYK, K passthrough
                (ColorSpace::Ycck, PixelFormat::Cmyk) => {
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, out, width);
                }
                // YCCK → RGB: convert YCCK→CMYK first (into temp), then CMYK→RGB
                (ColorSpace::Ycck, PixelFormat::Rgb) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        out[x * 3] = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        out[x * 3 + 1] =
                            (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        out[x * 3 + 2] =
                            (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Rgba) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        out[x * 4] = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + 1] =
                            (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + 2] =
                            (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + 3] = 255;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgr) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        let r = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        let g = (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        let b = (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                        out[x * 3] = b;
                        out[x * 3 + 1] = g;
                        out[x * 3 + 2] = r;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgra) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        let r = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        let g = (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        let b = (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                        out[x * 4] = b;
                        out[x * 4 + 1] = g;
                        out[x * 4 + 2] = r;
                        out[x * 4 + 3] = 255;
                    }
                }
                // CMYK → 4bpp offset-based formats
                (
                    ColorSpace::Cmyk,
                    PixelFormat::Rgbx
                    | PixelFormat::Bgrx
                    | PixelFormat::Xrgb
                    | PixelFormat::Xbgr
                    | PixelFormat::Argb
                    | PixelFormat::Abgr,
                ) => {
                    let r_off: usize = out_format.red_offset().unwrap();
                    let g_off: usize = out_format.green_offset().unwrap();
                    let b_off: usize = out_format.blue_offset().unwrap();
                    // The remaining offset is 0+1+2+3=6 minus the other three
                    let pad_off: usize = 6 - r_off - g_off - b_off;
                    for x in 0..width {
                        let ki = 255 - p3[x] as u16;
                        let r = (((255 - p0[x] as u16) * ki + 127) / 255) as u8;
                        let g = (((255 - p1[x] as u16) * ki + 127) / 255) as u8;
                        let b = (((255 - p2[x] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + r_off] = r;
                        out[x * 4 + g_off] = g;
                        out[x * 4 + b_off] = b;
                        out[x * 4 + pad_off] = 255;
                    }
                }
                // YCCK → 4bpp offset-based formats
                (
                    ColorSpace::Ycck,
                    PixelFormat::Rgbx
                    | PixelFormat::Bgrx
                    | PixelFormat::Xrgb
                    | PixelFormat::Xbgr
                    | PixelFormat::Argb
                    | PixelFormat::Abgr,
                ) => {
                    let r_off: usize = out_format.red_offset().unwrap();
                    let g_off: usize = out_format.green_offset().unwrap();
                    let b_off: usize = out_format.blue_offset().unwrap();
                    let pad_off: usize = 6 - r_off - g_off - b_off;
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let ki = 255 - cmyk_buf[x * 4 + 3] as u16;
                        let r = (((255 - cmyk_buf[x * 4] as u16) * ki + 127) / 255) as u8;
                        let g = (((255 - cmyk_buf[x * 4 + 1] as u16) * ki + 127) / 255) as u8;
                        let b = (((255 - cmyk_buf[x * 4 + 2] as u16) * ki + 127) / 255) as u8;
                        out[x * 4 + r_off] = r;
                        out[x * 4 + g_off] = g;
                        out[x * 4 + b_off] = b;
                        out[x * 4 + pad_off] = 255;
                    }
                }
                _ => {
                    return Err(JpegError::Unsupported(format!(
                        "unsupported conversion: {:?} → {:?}",
                        color_space, out_format
                    )));
                }
            }
        }

        Ok(Image {
            width,
            height,
            pixel_format: out_format,
            precision: 8,
            data,
            icc_profile,
            exif_data,
            comment: self.metadata.comment.clone(),
            density: self.metadata.density,
            saved_markers: self.metadata.saved_markers.clone(),
            warnings,
        })
    }
}
