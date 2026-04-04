use crate::common::error::{DecodeWarning, JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::icc;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::entropy::{self, McuDecoder};
use crate::decode::huffman;
use crate::decode::idct_extended;
use crate::decode::idct_scaled;
use crate::decode::lossless;
use crate::decode::marker::{JpegMetadata, MarkerReader, ScanInfo};
use crate::decode::progressive;
use crate::simd::{self, SimdRoutines};

/// Generic nearest-neighbor upsampling for arbitrary h/v factor combinations.
///
/// Handles non-standard sampling factors like 3x2, 3x1, 1x3, 4x2 that lack
/// dedicated optimized paths. Each input sample is replicated h_factor times
/// horizontally and v_factor times vertically.
pub(crate) fn upsample_generic_nearest(
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
    /// Buffer width in blocks (rounded up to MCU alignment: mcus_x * h_samp).
    blocks_x: usize,
    /// Buffer height in blocks (rounded up to MCU alignment: mcus_y * v_samp).
    blocks_y: usize,
    h_samp: usize,
    v_samp: usize,
    comp_w: usize,
    block_size: usize,
    /// Actual number of encoded block columns for non-interleaved scans.
    /// = ceil(image_width * h_samp / (max_h * block_size))
    width_in_blocks: usize,
    /// Actual number of encoded block rows for non-interleaved scans.
    /// = ceil(image_height * v_samp / (max_v * block_size))
    height_in_blocks: usize,
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
        let mut metadata = reader.read_markers()?;
        // MJPEG frames may omit Huffman tables; provide standard defaults
        // (JPEG spec section K.3), matching C libjpeg-turbo's std_huff_tables().
        Self::fill_default_huffman_tables(&mut metadata);
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

    /// Fill in standard JPEG Huffman tables when no DHT markers were present.
    ///
    /// MJPEG frames typically omit DHT markers entirely, relying on the decoder
    /// to provide the standard tables from JPEG spec section K.3.
    /// Only fills when ALL table slots are `None` (no DHT was parsed at all).
    fn fill_default_huffman_tables(metadata: &mut JpegMetadata) {
        use crate::common::huffman_table::HuffmanTable;

        // Only fill defaults if no DHT markers were present at all.
        // If any table was defined (even if some slots are empty), respect the
        // original DHT data and do not override.
        let any_dc = metadata.dc_huffman_tables.iter().any(|t| t.is_some());
        let any_ac = metadata.ac_huffman_tables.iter().any(|t| t.is_some());
        if any_dc || any_ac {
            return;
        }

        // Standard DC luminance (table 0)
        #[rustfmt::skip]
        const BITS_DC_LUM: [u8; 17] = [
            0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
        ];
        #[rustfmt::skip]
        const VALS_DC_LUM: [u8; 12] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ];

        // Standard DC chrominance (table 1)
        #[rustfmt::skip]
        const BITS_DC_CHR: [u8; 17] = [
            0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
        ];
        #[rustfmt::skip]
        const VALS_DC_CHR: [u8; 12] = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
        ];

        // Standard AC luminance (table 0)
        #[rustfmt::skip]
        const BITS_AC_LUM: [u8; 17] = [
            0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d
        ];
        #[rustfmt::skip]
        const VALS_AC_LUM: [u8; 162] = [
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
            0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
            0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
            0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
            0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
            0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
            0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
            0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
            0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
            0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
            0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
            0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
            0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
            0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
            0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
            0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
            0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
            0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa,
        ];

        // Standard AC chrominance (table 1)
        #[rustfmt::skip]
        const BITS_AC_CHR: [u8; 17] = [
            0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77
        ];
        #[rustfmt::skip]
        const VALS_AC_CHR: [u8; 162] = [
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
            0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
            0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
            0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
            0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
            0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
            0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
            0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
            0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
            0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
            0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
            0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
            0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
            0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
            0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
            0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
            0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
            0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
            0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
            0xf9, 0xfa,
        ];

        // Fill missing DC tables
        if metadata.dc_huffman_tables[0].is_none() {
            if let Ok(tbl) = HuffmanTable::build(&BITS_DC_LUM, &VALS_DC_LUM) {
                metadata.dc_huffman_tables[0] = Some(tbl);
            }
        }
        if metadata.dc_huffman_tables[1].is_none() {
            if let Ok(tbl) = HuffmanTable::build(&BITS_DC_CHR, &VALS_DC_CHR) {
                metadata.dc_huffman_tables[1] = Some(tbl);
            }
        }

        // Fill missing AC tables
        if metadata.ac_huffman_tables[0].is_none() {
            if let Ok(tbl) = HuffmanTable::build(&BITS_AC_LUM, &VALS_AC_LUM) {
                metadata.ac_huffman_tables[0] = Some(tbl);
            }
        }
        if metadata.ac_huffman_tables[1].is_none() {
            if let Ok(tbl) = HuffmanTable::build(&BITS_AC_CHR, &VALS_AC_CHR) {
                metadata.ac_huffman_tables[1] = Some(tbl);
            }
        }

        // Also fill in ScanInfo Huffman tables for the first scan if needed
        for scan in &mut metadata.scans {
            for i in 0..4 {
                if scan.dc_huffman_tables[i].is_none() && metadata.dc_huffman_tables[i].is_some() {
                    scan.dc_huffman_tables[i] = metadata.dc_huffman_tables[i].clone();
                }
                if scan.ac_huffman_tables[i].is_none() && metadata.ac_huffman_tables[i].is_some() {
                    scan.ac_huffman_tables[i] = metadata.ac_huffman_tables[i].clone();
                }
            }
        }
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
        if fast {
            self.dct_method = DctMethod::IsFast;
        } else if self.dct_method == DctMethod::IsFast {
            self.dct_method = DctMethod::IsLow;
        }
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

    /// Select the IDCT function based on the configured DCT method.
    #[inline(always)]
    fn idct_fn(&self) -> fn(&[i16; 64], &[u16; 64], &mut [u8; 64]) {
        match self.dct_method {
            DctMethod::IsFast => self.routines.idct_ifast,
            DctMethod::Float => self.routines.idct_float,
            DctMethod::IsLow => self.routines.idct_islow,
        }
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
    /// Dispatches to ISLOW/IFAST/Float based on `self.dct_method`.
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
        // For ISLOW, use optimized strided SIMD paths when available.
        if matches!(self.dct_method, DctMethod::IsLow) {
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
            }
        }

        // Generic path: IDCT into temp buffer, then copy row-by-row.
        #[allow(unreachable_code)]
        {
            let idct = self.idct_fn();
            let mut tmp = [0u8; 64];
            idct(coeffs, quant, &mut tmp);
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
            16 => idct_extended::idct_16x16_strided(coeffs, quant, output, stride),
            15 => idct_extended::idct_15x15_strided(coeffs, quant, output, stride),
            14 => idct_extended::idct_14x14_strided(coeffs, quant, output, stride),
            13 => idct_extended::idct_13x13_strided(coeffs, quant, output, stride),
            12 => idct_extended::idct_12x12_strided(coeffs, quant, output, stride),
            11 => idct_extended::idct_11x11_strided(coeffs, quant, output, stride),
            10 => idct_extended::idct_10x10_strided(coeffs, quant, output, stride),
            9 => idct_extended::idct_9x9_strided(coeffs, quant, output, stride),
            8 => self.idct_islow_strided(coeffs, quant, output, stride),
            7 => idct_extended::idct_7x7_strided(coeffs, quant, output, stride),
            6 => idct_extended::idct_6x6_strided(coeffs, quant, output, stride),
            5 => idct_extended::idct_5x5_strided(coeffs, quant, output, stride),
            4 => idct_scaled::idct_4x4_strided(coeffs, quant, output, stride),
            3 => idct_extended::idct_3x3_strided(coeffs, quant, output, stride),
            2 => idct_scaled::idct_2x2_strided(coeffs, quant, output, stride),
            1 => idct_scaled::idct_1x1_strided(coeffs, quant, output, stride),
            _ => unreachable!("invalid block_size: {}", block_size),
        }
    }

    /// Compute per-component IDCT block size for scaled decode.
    ///
    /// Matches C libjpeg-turbo's `jpeg_calc_output_dimensions` (jdmaster.c):
    /// chroma components get a larger IDCT to absorb subsampling factors,
    /// eliminating spatial upsampling. For example, 4:2:0 at 1/2 scale uses
    /// 4x4 IDCT for Y but 8x8 IDCT for Cb/Cr, so all planes end up the same
    /// pixel dimensions — no upsample needed.
    fn compute_comp_block_size(
        min_block_size: usize,
        max_h: usize,
        max_v: usize,
        h_samp: usize,
        v_samp: usize,
    ) -> usize {
        let mut ssize: usize = min_block_size;
        while ssize < 8
            && (max_h * min_block_size).is_multiple_of(h_samp * ssize * 2)
            && (max_v * min_block_size).is_multiple_of(v_samp * ssize * 2)
        {
            ssize *= 2;
        }
        ssize
    }

    /// Compute per-component block sizes for all components in a frame.
    fn compute_all_comp_block_sizes(
        min_block_size: usize,
        max_h: usize,
        max_v: usize,
        frame: &FrameHeader,
    ) -> Vec<usize> {
        frame
            .components
            .iter()
            .map(|comp| {
                Self::compute_comp_block_size(
                    min_block_size,
                    max_h,
                    max_v,
                    comp.horizontal_sampling as usize,
                    comp.vertical_sampling as usize,
                )
            })
            .collect()
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

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_rgba_row(
                    y, cb, cr, out, width,
                );
            }
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

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_bgr_row(
                    y, cb, cr, out, width,
                );
            }
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

        #[cfg(all(target_arch = "x86_64", feature = "simd"))]
        {
            if is_x86_feature_detected!("avx2") {
                return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_bgra_row(
                    y, cb, cr, out, width,
                );
            }
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
            #[allow(unreachable_code)]
            PixelFormat::Rgbx => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_rgbx_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_rgbx_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 0, 1, 2, 3)
            }
            #[allow(unreachable_code)]
            PixelFormat::Bgrx => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_bgrx_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_bgrx_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 2, 1, 0, 3)
            }
            #[allow(unreachable_code)]
            PixelFormat::Xrgb => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_xrgb_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_xrgb_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 1, 2, 3, 0)
            }
            #[allow(unreachable_code)]
            PixelFormat::Xbgr => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_xbgr_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_xbgr_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 3, 2, 1, 0)
            }
            #[allow(unreachable_code)]
            PixelFormat::Argb => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_argb_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_argb_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
                crate::decode::color::ycbcr_to_generic_4bpp_row(y, cb, cr, out, width, 1, 2, 3, 0)
            }
            #[allow(unreachable_code)]
            PixelFormat::Abgr => {
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    return crate::simd::aarch64::color::neon_ycbcr_to_abgr_row(
                        y, cb, cr, out, width,
                    );
                }

                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        return crate::simd::x86_64::avx2_color::avx2_ycbcr_to_abgr_row(
                            y, cb, cr, out, width,
                        );
                    }
                }
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
        // For in_width <= 2, C's merged path uses box filter (no interpolation).
        // NEON/SIMD paths may not handle this edge case correctly, so use scalar.
        if in_width <= 2 {
            crate::decode::upsample::fancy_h2v1(input, in_width, output, 0);
            return;
        }

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

        // Fused H2V2: vertical + horizontal in one pass using >> 4 arithmetic.
        // Matches C libjpeg-turbo h2v2_fancy_upsample exactly, avoiding
        // double-rounding from the previous two-pass approach.
        #[allow(unreachable_code)]
        {
            crate::decode::upsample::fancy_h2v2(
                input,
                in_width,
                in_height,
                output,
                out_width,
                in_height * 2,
            );
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
            // Vertical triangle filter with ordered dither to avoid systematic
            // rounding bias (matches C jdsample.c h1v2_fancy_upsample):
            //   top row: bias=1, bottom row: bias=2
            for i in 0..in_width {
                out_top[i] = ((3 * cur_row[i] as u16 + above[i] as u16 + 1) >> 2) as u8;
                out_bot[i] = ((3 * cur_row[i] as u16 + below[i] as u16 + 2) >> 2) as u8;
            }
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
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        // Non-interleaved baseline: multiple SOS markers, each with a single component.
        // Dispatch to dedicated multi-scan path.
        if self.metadata.scans.len() > 1 {
            return self.decode_non_interleaved_baseline_planes(
                frame,
                quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                comp_block_sizes,
            );
        }

        let scan = &self.metadata.scan;
        let block_size: usize = comp_block_sizes[0]; // min (luma) block size for MCU row range

        // Determine MCU row range for IDCT
        let (mcu_y_start, mcu_y_end) = self.mcu_row_range(mcus_y, block_size, frame);

        // Allocate component planes (full MCU-aligned size, uninitialized).
        // SAFETY: The MCU decode loop + IDCT writes every pixel before reading.
        #[allow(clippy::uninit_vec)]
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
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
            block_size: usize,
        }
        let comp_layouts: Vec<CompLayout> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| CompLayout {
                comp_w: mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci],
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
                block_size: comp_block_sizes[ci],
            })
            .collect();

        let entropy_data = &self.raw_data[self.metadata.entropy_data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = McuDecoder::new(num_components);
        let mut mcu_count: u32 = 0;
        let mut coeffs = [0i16; 64];
        let mut warnings: Vec<DecodeWarning> = Vec::new();
        let total_mcus = mcus_x * mcus_y;

        // Fast path: non-lenient, no cropping — tight loop with minimal branching.
        // The lenient/crop path is below with full error recovery support.
        if !self.lenient && mcu_y_start == 0 && mcu_y_end == mcus_y {
            let restart_interval: u32 = self.metadata.restart_interval as u32;
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

                                let bs: usize = layout.block_size;
                                let block_x: usize = (mcu_x * layout.h_blocks + h) * bs;
                                let block_y: usize = (mcu_y * layout.v_blocks + v) * bs;
                                let dst_offset: usize = block_y * layout.comp_w + block_x;

                                unsafe {
                                    let dst: *mut u8 =
                                        component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                    self.idct_scaled_strided(
                                        &coeffs,
                                        qt_values,
                                        dst,
                                        layout.comp_w,
                                        bs,
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
                        && mcu_count.is_multiple_of(self.metadata.restart_interval as u32)
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
                                    let bs: usize = layout.block_size;
                                    let block_x: usize = (mcu_x * layout.h_blocks + h) * bs;
                                    let block_y: usize = (mcu_y * layout.v_blocks + v) * bs;
                                    let dst_offset: usize = block_y * layout.comp_w + block_x;

                                    unsafe {
                                        let dst: *mut u8 =
                                            component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                                        self.idct_scaled_strided(
                                            &coeffs,
                                            qt_values,
                                            dst,
                                            layout.comp_w,
                                            bs,
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

    /// Decode non-interleaved baseline JPEG (multiple SOS markers, one component per scan).
    ///
    /// Each SOS contains a single component with full DC+AC coefficients (ss=0, se=63).
    /// The MCU for a non-interleaved scan is a single 8x8 block, and blocks are
    /// iterated in raster order: blocks_x * blocks_y total blocks per scan.
    #[allow(clippy::too_many_arguments)]
    fn decode_non_interleaved_baseline_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        // Allocate component planes (full MCU-aligned size, zero-initialized).
        // Zero-init is needed because non-interleaved scans may encode fewer
        // blocks than the plane holds (padding blocks at right/bottom edges).
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w: usize =
                    mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h: usize = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
                let size: usize = comp_w * comp_h;
                vec![0u8; size]
            })
            .collect();

        // Process each scan independently
        for scan_info in &self.metadata.scans {
            let scan = &scan_info.header;

            // Each non-interleaved scan should have exactly 1 component
            if scan.components.len() != 1 {
                return Err(JpegError::CorruptData(format!(
                    "non-interleaved baseline scan has {} components, expected 1",
                    scan.components.len()
                )));
            }

            let scan_comp = &scan.components[0];

            // Find the frame component index for this scan's component
            let comp_idx: usize = frame
                .components
                .iter()
                .position(|fc| fc.id == scan_comp.component_id)
                .ok_or_else(|| {
                    JpegError::CorruptData(format!(
                        "scan references unknown component id {}",
                        scan_comp.component_id
                    ))
                })?;

            let comp = &frame.components[comp_idx];
            let h_samp: usize = comp.horizontal_sampling as usize;
            let v_samp: usize = comp.vertical_sampling as usize;
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

            // For non-interleaved scans, the number of encoded blocks is based on
            // the component's actual sample dimensions (JPEG spec ITU T.81 A.2.3):
            //   comp_samples = ceil(image_dim * h_samp / max_h)
            //   encoded_blocks = ceil(comp_samples / 8)
            let comp_width_samples: usize = (frame.width as usize * h_samp).div_ceil(max_h);
            let comp_height_samples: usize = (frame.height as usize * v_samp).div_ceil(max_v);
            let encoded_blocks_x: usize = comp_width_samples.div_ceil(8);
            let encoded_blocks_y: usize = comp_height_samples.div_ceil(8);

            // The plane is allocated based on the interleaved MCU grid,
            // which may have more blocks than the encoded data.
            let plane_blocks_x: usize = mcus_x * h_samp;
            let bs: usize = comp_block_sizes[comp_idx];
            let comp_w: usize = plane_blocks_x * bs;

            // Resolve Huffman tables for this scan
            let dc_table: &HuffmanTable =
                Self::resolve_table(&scan_info.dc_huffman_tables, scan_comp.dc_table_index, "DC")?;
            let ac_table: &HuffmanTable =
                Self::resolve_table(&scan_info.ac_huffman_tables, scan_comp.ac_table_index, "AC")?;

            let qt_values: &[u16; 64] = &quant_tables[comp_idx].values;

            let entropy_data: &[u8] = &self.raw_data[scan_info.data_offset..];
            let mut bit_reader: BitReader = BitReader::new(entropy_data);
            // Fresh DC prediction per scan (each non-interleaved scan starts at 0)
            let mut mcu_decoder: McuDecoder = McuDecoder::new(frame.components.len());
            let mut coeffs: [i16; 64] = [0i16; 64];

            let restart_interval: u32 = scan_info.restart_interval as u32;
            let mut mcu_count: u32 = 0;

            // In a non-interleaved scan, each MCU is a single block.
            // Iterate over encoded blocks (may be fewer than plane blocks
            // when image dimensions don't align with the MCU grid).
            for by in 0..encoded_blocks_y {
                for bx in 0..encoded_blocks_x {
                    // Restart interval handling
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        bit_reader.reset();
                        mcu_decoder.reset();
                    }

                    // Decode one 8x8 block
                    mcu_decoder.decode_block(
                        &mut bit_reader,
                        comp_idx,
                        dc_table,
                        ac_table,
                        &mut coeffs,
                    )?;

                    // IDCT and store into the component plane
                    let block_x: usize = bx * bs;
                    let block_y: usize = by * bs;
                    let dst_offset: usize = block_y * comp_w + block_x;

                    unsafe {
                        let dst: *mut u8 = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(&coeffs, qt_values, dst, comp_w, bs);
                    }

                    mcu_count += 1;
                }
            }
        }

        Ok((component_planes, Vec::new()))
    }

    /// Decode arithmetic-coded planes (SOF9 sequential).
    fn decode_arithmetic_planes(
        &self,
        frame: &FrameHeader,
        quant_tables: &[&QuantTable],
        _num_components: usize,
        mcus_x: usize,
        mcus_y: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        let scan = &self.metadata.scan;

        // Allocate component planes
        #[allow(clippy::uninit_vec)]
        let mut component_planes: Vec<Vec<u8>> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let comp_w = mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci];
                let comp_h = mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci];
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
            block_size: usize,
        }
        let comp_layouts: Vec<CompLayout> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| CompLayout {
                comp_w: mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci],
                h_blocks: comp.horizontal_sampling as usize,
                v_blocks: comp.vertical_sampling as usize,
                block_size: comp_block_sizes[ci],
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
                            let bs: usize = layout.block_size;
                            let bx = mcu_x * layout.h_blocks + h;
                            let by = mcu_y * layout.v_blocks + v;
                            let x_offset = bx * bs;
                            let y_offset = by * bs;

                            let plane = &mut component_planes[comp_idx];
                            let stride = layout.comp_w;

                            unsafe {
                                let out_ptr = plane.as_mut_ptr().add(y_offset * stride + x_offset);
                                self.idct_scaled_strided(&coeffs, qt_values, out_ptr, stride, bs);
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
        max_h: usize,
        max_v: usize,
        comp_block_sizes: &[usize],
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        use crate::decode::arithmetic::ArithDecoder;

        let img_w = frame.width as usize;
        let img_h = frame.height as usize;
        let dct_size: usize = 8;

        // Per-component coefficient buffers
        let comp_infos: Vec<CompInfo> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let h_samp = comp.horizontal_sampling as usize;
                let v_samp = comp.vertical_sampling as usize;
                let bs = comp_block_sizes[ci];
                CompInfo {
                    blocks_x: mcus_x * h_samp,
                    blocks_y: mcus_y * v_samp,
                    h_samp,
                    v_samp,
                    comp_w: mcus_x * h_samp * bs,
                    block_size: bs,
                    width_in_blocks: (img_w * h_samp).div_ceil(max_h * dct_size),
                    height_in_blocks: (img_h * v_samp).div_ceil(max_v * dct_size),
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
                let scan_bx = ci.width_in_blocks;
                let scan_by = ci.height_in_blocks;
                let stride = ci.blocks_x;

                for by in 0..scan_by {
                    for bx in 0..scan_bx {
                        let block_idx = by * stride + bx;
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
                let size = ci.comp_w * ci.blocks_y * ci.block_size;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        for (comp_idx, ci) in comp_infos.iter().enumerate() {
            let qt_values = &quant_tables[comp_idx].values;
            let bs: usize = ci.block_size;
            for by in 0..ci.blocks_y {
                for bx in 0..ci.blocks_x {
                    let block_idx = by * ci.blocks_x + bx;
                    let coeffs = &coeff_bufs[comp_idx][block_idx];

                    let px_x = bx * bs;
                    let px_y = by * bs;
                    let dst_offset = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(coeffs, qt_values, dst, ci.comp_w, bs);
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
        comp_block_sizes: &[usize],
        block_smoothing: bool,
    ) -> Result<(Vec<Vec<u8>>, Vec<DecodeWarning>)> {
        let img_w = frame.width as usize;
        let img_h = frame.height as usize;

        // Per-component coefficient buffers: blocks_x * blocks_y blocks of 64 coefficients.
        // width_in_blocks/height_in_blocks use DCT block size (8), not the scaled
        // output block size, because coefficient buffers are indexed by 8x8 DCT blocks.
        let dct_size: usize = 8;
        let comp_infos: Vec<CompInfo> = frame
            .components
            .iter()
            .enumerate()
            .map(|(ci, comp)| {
                let h_samp = comp.horizontal_sampling as usize;
                let v_samp = comp.vertical_sampling as usize;
                let bs = comp_block_sizes[ci];
                CompInfo {
                    blocks_x: mcus_x * h_samp,
                    blocks_y: mcus_y * v_samp,
                    h_samp,
                    v_samp,
                    comp_w: mcus_x * h_samp * bs,
                    block_size: bs,
                    width_in_blocks: (img_w * h_samp).div_ceil(max_h * dct_size),
                    height_in_blocks: (img_h * v_samp).div_ceil(max_v * dct_size),
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

        // Apply coefficient-level block smoothing before IDCT (if requested).
        // This matches C libjpeg-turbo's decompress_smooth_data() approach:
        // smooth the DCT coefficients, then run IDCT on the smoothed coefficients.
        if block_smoothing {
            let coef_bits_all: Vec<[i32; 10]> =
                crate::decode::toggles::compute_coef_bits(&self.metadata.scans, frame);
            for (comp_idx, ci) in comp_infos.iter().enumerate() {
                let cb: &[i32; 10] = &coef_bits_all[comp_idx];
                if crate::decode::toggles::smoothing_ok_for_component(cb, quant_tables[comp_idx]) {
                    crate::decode::toggles::apply_block_smoothing_coeffs(
                        &mut coeff_bufs[comp_idx],
                        ci.blocks_x,
                        ci.blocks_y,
                        cb,
                        quant_tables[comp_idx],
                    );
                }
            }
        }

        // IDCT all blocks into component planes
        #[allow(clippy::uninit_vec)]
        let mut component_planes: Vec<Vec<u8>> = comp_infos
            .iter()
            .map(|ci| {
                let size = ci.comp_w * ci.blocks_y * ci.block_size;
                let mut v = Vec::with_capacity(size);
                unsafe { v.set_len(size) };
                v
            })
            .collect();

        for (comp_idx, ci) in comp_infos.iter().enumerate() {
            let qt_values = &quant_tables[comp_idx].values;
            let bs: usize = ci.block_size;
            for by in 0..ci.blocks_y {
                for bx in 0..ci.blocks_x {
                    let block_idx = by * ci.blocks_x + bx;
                    let coeffs = &coeff_bufs[comp_idx][block_idx];

                    let px_x = bx * bs;
                    let px_y = by * bs;
                    let dst_offset = px_y * ci.comp_w + px_x;

                    unsafe {
                        let dst = component_planes[comp_idx].as_mut_ptr().add(dst_offset);
                        self.idct_scaled_strided(coeffs, qt_values, dst, ci.comp_w, bs);
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

        // Non-interleaved scans use width_in_blocks/height_in_blocks for iteration,
        // which may be smaller than blocks_x/blocks_y (the MCU-aligned buffer size).
        // Dummy blocks at the right/bottom edges only receive DC from interleaved scans.
        let coeff_slice = &mut coeff_bufs[comp_idx];
        let scan_blocks_x = ci.width_in_blocks;
        let scan_blocks_y = ci.height_in_blocks;
        let stride = ci.blocks_x; // buffer stride (MCU-aligned)

        if is_dc && ah == 0 {
            let dc_table = dc_table.unwrap();
            for by in 0..scan_blocks_y {
                for bx in 0..scan_blocks_x {
                    restart_check_dc!(bit_reader, dc_pred, restart_countdown, restart_interval);
                    let coeffs = &mut coeff_slice[by * stride + bx];
                    progressive::decode_dc_first(bit_reader, dc_table, &mut dc_pred, coeffs, al)?;
                }
            }
        } else if is_dc {
            for by in 0..scan_blocks_y {
                for bx in 0..scan_blocks_x {
                    if restart_interval > 0 {
                        if restart_countdown == 0 {
                            bit_reader.reset();
                            restart_countdown = restart_interval;
                        }
                        restart_countdown -= 1;
                    }
                    let coeffs = &mut coeff_slice[by * stride + bx];
                    progressive::decode_dc_refine(bit_reader, coeffs, al)?;
                }
            }
        } else if ah == 0 {
            let ac_table = ac_table.unwrap();
            for by in 0..scan_blocks_y {
                for bx in 0..scan_blocks_x {
                    restart_check_ac!(bit_reader, eob_run, restart_countdown, restart_interval);
                    let coeffs = &mut coeff_slice[by * stride + bx];
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
            }
        } else {
            let ac_table = ac_table.unwrap();
            for by in 0..scan_blocks_y {
                for bx in 0..scan_blocks_x {
                    restart_check_ac!(bit_reader, eob_run, restart_countdown, restart_interval);
                    let coeffs = &mut coeff_slice[by * stride + bx];
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

    /// Convert decoded lossless RGB component planes to output Image.
    ///
    /// Lossless JPEG stores raw RGB values with no color conversion (matching
    /// C libjpeg-turbo JCS_RGB behavior), so we output component values directly.
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

        for ((&r_pix, &g_pix), &b_pix) in comp_planes[0]
            .iter()
            .zip(comp_planes[1].iter())
            .zip(comp_planes[2].iter())
        {
            // Raw RGB: output component values directly (no color conversion)
            let r: u8 = r_pix.min(255) as u8;
            let g: u8 = g_pix.min(255) as u8;
            let b: u8 = b_pix.min(255) as u8;

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

    /// Decode a 12-bit JPEG by delegating to `decompress_12bit`, then scaling
    /// the 12-bit samples (0-4095) down to 8-bit (0-255). Converts to the
    /// requested output pixel format if one was set.
    fn decode_12bit_as_8bit(
        &self,
        icc_profile: Option<Vec<u8>>,
        exif_data: Option<Vec<u8>>,
    ) -> Result<Image> {
        let img12 = crate::api::precision::decompress_12bit(self.raw_data)?;
        let num_components: usize = img12.num_components;

        // Determine output format: default to Grayscale for 1-component,
        // RGB for 3-component, same as the 8-bit path.
        let default_format: PixelFormat = if num_components == 1 {
            PixelFormat::Grayscale
        } else {
            PixelFormat::Rgb
        };
        let out_format: PixelFormat = self.output_format.unwrap_or(default_format);

        // Scale 12-bit i16 samples to 8-bit u8: val * 255 / 4095.
        // This matches C djpeg's 12-to-8 bit downscaling.
        let width: usize = img12.width;
        let height: usize = img12.height;

        if num_components == 1 {
            // Grayscale: scale directly, ignore output format conversion
            // (only Grayscale makes sense for 1-component).
            let mut data: Vec<u8> = Vec::with_capacity(width * height);
            for &val in &img12.data {
                let clamped: i16 = val.clamp(0, 4095);
                data.push((clamped as u32 * 255 / 4095) as u8);
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
            // Color image: img12.data is interleaved RGB (3 values per pixel).
            // Convert to the requested output format.
            let bpp: usize = out_format.bytes_per_pixel();
            let mut data: Vec<u8> = vec![0u8; width * height * bpp];

            let r_off: Option<usize> = out_format.red_offset();
            let g_off: Option<usize> = out_format.green_offset();
            let b_off: Option<usize> = out_format.blue_offset();

            for i in 0..(width * height) {
                let src_idx: usize = i * 3;
                let r: u8 = (img12.data[src_idx].clamp(0, 4095) as u32 * 255 / 4095) as u8;
                let g: u8 = (img12.data[src_idx + 1].clamp(0, 4095) as u32 * 255 / 4095) as u8;
                let b: u8 = (img12.data[src_idx + 2].clamp(0, 4095) as u32 * 255 / 4095) as u8;
                let dst_idx: usize = i * bpp;

                match out_format {
                    PixelFormat::Rgb => {
                        data[dst_idx] = r;
                        data[dst_idx + 1] = g;
                        data[dst_idx + 2] = b;
                    }
                    PixelFormat::Grayscale => {
                        // Approximate luminance from RGB.
                        data[dst_idx] =
                            ((r as u32 * 77 + g as u32 * 150 + b as u32 * 29) >> 8) as u8;
                    }
                    _ => {
                        // Use offset-based mapping for all other RGB-derived formats.
                        if let (Some(ro), Some(go), Some(bo)) = (r_off, g_off, b_off) {
                            data[dst_idx + ro] = r;
                            data[dst_idx + go] = g;
                            data[dst_idx + bo] = b;
                            // Fill alpha/padding byte to 0xFF for 4-bpp formats.
                            if bpp == 4 {
                                let alpha_off: usize = 6 - ro - go - bo;
                                data[dst_idx + alpha_off] = 0xFF;
                            }
                        } else {
                            return Err(JpegError::Unsupported(format!(
                                "cannot convert 12-bit color JPEG to {:?}",
                                out_format
                            )));
                        }
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

        // Handle 12-bit JPEG transparently: decode via the 12-bit path, then
        // scale samples from 0-4095 to 0-255 so callers get standard 8-bit output.
        // This matches C djpeg behavior which handles 12-bit JPEGs automatically.
        if frame.precision == 12 {
            return self.decode_12bit_as_8bit(icc_profile, exif_data);
        }

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
        // Per-component IDCT block sizes: chroma components may use a larger
        // IDCT to absorb subsampling factors (matches C libjpeg-turbo).
        let comp_block_sizes: Vec<usize> =
            Self::compute_all_comp_block_sizes(block_size, max_h, max_v, frame);
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
                &comp_block_sizes,
            )?
        } else if self.metadata.is_arithmetic {
            self.decode_arithmetic_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                &comp_block_sizes,
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
                &comp_block_sizes,
                self.block_smoothing,
            )?
        } else {
            self.decode_baseline_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                &comp_block_sizes,
            )?
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
                &comp_block_sizes,
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
            let comp_w =
                mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];

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
            let y_width =
                mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];

            let cb_comp = &frame.components[1];
            let cr_comp = &frame.components[2];
            let cb_w = mcus_x * cb_comp.horizontal_sampling as usize * comp_block_sizes[1];
            let cb_h = mcus_y * cb_comp.vertical_sampling as usize * comp_block_sizes[1];
            let cr_w = mcus_x * cr_comp.horizontal_sampling as usize * comp_block_sizes[2];
            let cr_h = mcus_y * cr_comp.vertical_sampling as usize * comp_block_sizes[2];

            let y_height =
                mcus_y * frame.components[0].vertical_sampling as usize * comp_block_sizes[0];

            // Per-component effective upsample factors.
            // For scaled decode, chroma may use a larger IDCT that absorbs subsampling,
            // making the effective factor 1 (no upsample needed).
            let cb_h_factor: usize = y_width / cb_w;
            let cb_v_factor: usize = y_height / cb_h;
            let cr_h_factor: usize = y_width / cr_w;
            let cr_v_factor: usize = y_height / cr_h;

            // When both chroma components have the same factors, use the shared
            // factor variables that the existing optimized paths expect.
            let uniform_chroma: bool = cb_h_factor == cr_h_factor && cb_v_factor == cr_v_factor;
            let h_factor: usize = cb_h_factor;
            let v_factor: usize = cb_v_factor;

            // Actual chroma dimensions (may be smaller than MCU-aligned cb_w/cb_h).
            // C libjpeg-turbo uses downsampled_width/height for upsample, not
            // MCU-padded dimensions. Using MCU-padded values causes the upsample
            // to interpolate padding data, producing wrong edge pixels.
            let actual_cb_w: usize = out_width.div_ceil(cb_h_factor);
            let actual_cb_h: usize = out_height.div_ceil(cb_v_factor);
            let actual_cr_w: usize = out_width.div_ceil(cr_h_factor);
            let actual_cr_h: usize = out_height.div_ceil(cr_v_factor);

            // For 4:4:4, use component planes directly without clone.
            // For subsampled modes, upsample into separate buffers.
            let (cb_data, cr_data, cb_stride, cr_stride): (&[u8], &[u8], usize, usize);

            if cb_h_factor == 1 && cb_v_factor == 1 && cr_h_factor == 1 && cr_v_factor == 1 {
                // 4:4:4: no upsampling needed — reference planes directly
                cb_data = &component_planes[1];
                cr_data = &component_planes[2];
                cb_stride = cb_w;
                cr_stride = cr_w;
            } else {
                // Merged upsample path: combine upsample + color convert in one pass
                // for H2V1 (4:2:2) and H2V2 (4:2:0), avoiding intermediate chroma buffers.
                // Only available when both chroma components have the same sampling factors.
                if self.merged_upsample
                    && uniform_chroma
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

                // Row-streaming H2V2: skip full-plane allocation, process 2 rows at a time.
                // When actual_cb_w <= 2, C's merged upsample uses box filter for the
                // entire image (the NEON/SIMD fancy path doesn't kick in). Use box
                // filter (fast_upsample equivalent) to match C exactly.
                // Only available when both chroma components have the same sampling factors.
                if !self.fast_upsample
                    && uniform_chroma
                    && h_factor == 2
                    && v_factor == 2
                    && actual_cb_w > 2
                    && block_size == 8
                {
                    // Row-streaming H2V2: fuse upsample + color convert to avoid
                    // allocating full-size cb_full/cr_full buffers (~4MB for 1080p).
                    // Process 2 output rows at a time, keeping data in L1/L2 cache.
                    let data_size = out_width * out_height * bpp;
                    let mut data = Vec::with_capacity(data_size);
                    #[allow(clippy::uninit_vec)]
                    unsafe {
                        data.set_len(data_size)
                    };

                    // Small per-row upsample buffers (2 rows × full_width per component)
                    let mut cb_row_top = vec![0u8; full_width];
                    let mut cb_row_bot = vec![0u8; full_width];
                    let mut cr_row_top = vec![0u8; full_width];
                    let mut cr_row_bot = vec![0u8; full_width];

                    // Use actual chroma dimensions for upsample (not MCU-padded).
                    for cy in 0..actual_cb_h {
                        let cb_cur = &component_planes[1][cy * cb_w..cy * cb_w + actual_cb_w];
                        let cr_cur = &component_planes[2][cy * cb_w..cy * cb_w + actual_cb_w];
                        let cb_above = if cy > 0 {
                            &component_planes[1][(cy - 1) * cb_w..(cy - 1) * cb_w + actual_cb_w]
                        } else {
                            cb_cur
                        };
                        let cb_below = if cy + 1 < actual_cb_h {
                            &component_planes[1][(cy + 1) * cb_w..(cy + 1) * cb_w + actual_cb_w]
                        } else {
                            cb_cur
                        };
                        let cr_above = if cy > 0 {
                            &component_planes[2][(cy - 1) * cb_w..(cy - 1) * cb_w + actual_cb_w]
                        } else {
                            cr_cur
                        };
                        let cr_below = if cy + 1 < actual_cb_h {
                            &component_planes[2][(cy + 1) * cb_w..(cy + 1) * cb_w + actual_cb_w]
                        } else {
                            cr_cur
                        };

                        // Fused vertical+horizontal upsample for top output row
                        crate::decode::upsample::fancy_h2v2_row(
                            cb_cur,
                            cb_above,
                            &mut cb_row_top,
                            actual_cb_w,
                        );
                        crate::decode::upsample::fancy_h2v2_row(
                            cr_cur,
                            cr_above,
                            &mut cr_row_top,
                            actual_cb_w,
                        );

                        // Fused vertical+horizontal upsample for bottom output row
                        crate::decode::upsample::fancy_h2v2_row(
                            cb_cur,
                            cb_below,
                            &mut cb_row_bot,
                            actual_cb_w,
                        );
                        crate::decode::upsample::fancy_h2v2_row(
                            cr_cur,
                            cr_below,
                            &mut cr_row_bot,
                            actual_cb_w,
                        );

                        // Color convert both output rows immediately
                        let out_y_top = cy * 2;
                        let out_y_bot = cy * 2 + 1;
                        if out_y_top < out_height {
                            self.color_convert_row(
                                out_format,
                                &y_plane[out_y_top * y_width..],
                                &cb_row_top,
                                &cr_row_top,
                                &mut data[out_y_top * out_width * bpp..],
                                out_width,
                                out_y_top,
                            );
                        }
                        if out_y_bot < out_height {
                            self.color_convert_row(
                                out_format,
                                &y_plane[out_y_bot * y_width..],
                                &cb_row_bot,
                                &cr_row_bot,
                                &mut data[out_y_bot * out_width * bpp..],
                                out_width,
                                out_y_bot,
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

                // All remaining paths need full-plane cb_full/cr_full buffers.
                let alloc_size = full_width * full_height;
                let mut cb_full = Vec::with_capacity(alloc_size);
                let mut cr_full = Vec::with_capacity(alloc_size);
                #[allow(clippy::uninit_vec)]
                unsafe {
                    cb_full.set_len(alloc_size);
                    cr_full.set_len(alloc_size);
                }

                // Upsample each chroma component independently using its own factors.
                // This handles non-uniform chroma sampling (e.g. Cb=2x1, Cr=1x1)
                // where each component needs a different upsample strategy.
                for (comp_plane, comp_full, comp_w, comp_h, comp_hf, comp_vf, actual_w, actual_h) in [
                    (
                        &component_planes[1],
                        &mut cb_full,
                        cb_w,
                        cb_h,
                        cb_h_factor,
                        cb_v_factor,
                        actual_cb_w,
                        actual_cb_h,
                    ),
                    (
                        &component_planes[2],
                        &mut cr_full,
                        cr_w,
                        cr_h,
                        cr_h_factor,
                        cr_v_factor,
                        actual_cr_w,
                        actual_cr_h,
                    ),
                ] {
                    // C libjpeg-turbo uses box filter when:
                    // - fast_upsample requested, OR
                    // - actual chroma width <= 2 (fancy filter needs >= 3 columns), OR
                    // - min_DCT_scaled_size == 1 (jdsample.c line 478: jdmainct.c
                    //   doesn't support context rows at this size)
                    let use_box_filter: bool =
                        self.fast_upsample || actual_w <= 2 || block_size == 1;

                    if comp_hf == 1 && comp_vf == 1 {
                        // No upsampling needed for this component — copy directly.
                        for row in 0..full_height.min(comp_h) {
                            let src_start: usize = row * comp_w;
                            let dst_start: usize = row * full_width;
                            let copy_len: usize = full_width.min(comp_w);
                            comp_full[dst_start..dst_start + copy_len]
                                .copy_from_slice(&comp_plane[src_start..src_start + copy_len]);
                        }
                    } else if use_box_filter {
                        crate::decode::toggles::upsample_nearest(
                            comp_plane, comp_w, comp_h, comp_full, full_width, comp_hf, comp_vf,
                        );
                    } else if comp_hf == 2 && comp_vf == 1 {
                        // H2V1: horizontal-only 2x fancy upsample.
                        for row in 0..actual_h {
                            self.fancy_upsample_h2v1(
                                &comp_plane[row * comp_w..],
                                actual_w,
                                &mut comp_full[row * full_width..],
                            );
                        }
                    } else if comp_hf == 2 && comp_vf == 2 {
                        // H2V2: fused 2D triangle filter fancy upsample.
                        crate::decode::upsample::fancy_h2v2_strided(
                            comp_plane, actual_w, comp_w, actual_h, comp_full, full_width,
                        );
                    } else if comp_hf == 1 && comp_vf == 2 {
                        // H1V2: vertical-only 2x fancy upsample.
                        self.fancy_h1v2(comp_plane, comp_w, actual_h, comp_full, full_width);
                    } else {
                        // Generic fallback: nearest-neighbor for any factor combination
                        // (4x1, 4x2, 1x4, 3x2, etc.).
                        upsample_generic_nearest(
                            comp_plane, comp_w, comp_h, comp_full, full_width, comp_hf, comp_vf,
                        );
                    }
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
                &comp_block_sizes,
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
        // Raw data decode always uses full-size (8x8) IDCT for all components
        let comp_block_sizes: Vec<usize> = vec![block_size; num_components];
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
                &comp_block_sizes,
            )?
        } else if self.metadata.is_arithmetic {
            self.decode_arithmetic_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                &comp_block_sizes,
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
                &comp_block_sizes,
                false, // raw data decode: no block smoothing
            )?
        } else {
            self.decode_baseline_planes(
                frame,
                &quant_tables,
                num_components,
                mcus_x,
                mcus_y,
                &comp_block_sizes,
            )?
        };
        let mut plane_widths: Vec<usize> = Vec::with_capacity(num_components);
        let mut plane_heights: Vec<usize> = Vec::with_capacity(num_components);
        for (ci, comp) in frame.components.iter().enumerate() {
            plane_widths.push(mcus_x * comp.horizontal_sampling as usize * comp_block_sizes[ci]);
            plane_heights.push(mcus_y * comp.vertical_sampling as usize * comp_block_sizes[ci]);
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
        _max_h: usize,
        _max_v: usize,
        full_width: usize,
        full_height: usize,
        comp_block_sizes: &[usize],
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
        let comp0_w =
            mcus_x * frame.components[0].horizontal_sampling as usize * comp_block_sizes[0];

        // For YCCK, components 1-2 may be subsampled (chroma), component 3 (K) is full.
        // For CMYK, all components are typically the same resolution.
        let comp1 = &frame.components[1];
        let comp1_w = mcus_x * comp1.horizontal_sampling as usize * comp_block_sizes[1];
        let comp1_h = mcus_y * comp1.vertical_sampling as usize * comp_block_sizes[1];
        let comp3_w =
            mcus_x * frame.components[3].horizontal_sampling as usize * comp_block_sizes[3];

        let h_factor = comp0_w / comp1_w;
        let v_factor =
            (mcus_y * frame.components[0].vertical_sampling as usize * comp_block_sizes[0])
                / (mcus_y * comp1.vertical_sampling as usize * comp_block_sizes[1]);

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
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        out[x * 3] = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        out[x * 3 + 1] = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        out[x * 3 + 2] = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Rgba) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        out[x * 4] = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + 1] = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + 2] = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                        out[x * 4 + 3] = 255;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgr) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        let r = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        let g = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        let b = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
                        out[x * 3] = b;
                        out[x * 3 + 1] = g;
                        out[x * 3 + 2] = r;
                    }
                }
                (ColorSpace::Ycck, PixelFormat::Bgra) => {
                    let mut cmyk_buf = vec![0u8; width * 4];
                    color::ycck_to_cmyk_row(p0, p1, p2, p3, &mut cmyk_buf, width);
                    for x in 0..width {
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        let r = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        let g = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        let b = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
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
                        let kv = p3[x] as u16;
                        let r = ((p0[x] as u16 * kv + 127) / 255) as u8;
                        let g = ((p1[x] as u16 * kv + 127) / 255) as u8;
                        let b = ((p2[x] as u16 * kv + 127) / 255) as u8;
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
                        let kv = cmyk_buf[x * 4 + 3] as u16;
                        let r = ((cmyk_buf[x * 4] as u16 * kv + 127) / 255) as u8;
                        let g = ((cmyk_buf[x * 4 + 1] as u16 * kv + 127) / 255) as u8;
                        let b = ((cmyk_buf[x * 4 + 2] as u16 * kv + 127) / 255) as u8;
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
