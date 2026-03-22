use crate::api::quality;
use crate::common::error::Result;
use crate::common::types::{
    ColorSpace, DctMethod, PixelFormat, SavedMarker, ScanScript, Subsampling,
};
use crate::encode::pipeline as encoder;
use crate::encode::tables;

/// Configuration for DRI restart interval encoding.
#[derive(Debug, Clone, Copy)]
pub enum RestartConfig {
    /// Restart every N MCU blocks.
    Blocks(u16),
    /// Restart every N MCU rows.
    Rows(u16),
}

/// User-supplied Huffman table definition.
///
/// `bits[0]` is unused; `bits[1]..bits[16]` give the number of codes of each
/// bit length, matching the DHT marker format in ITU-T T.81 Annex C.
#[derive(Debug, Clone)]
pub struct HuffmanTableDef {
    /// Code-length counts. Index 0 is unused.
    pub bits: [u8; 17],
    /// Symbol values in order of increasing code length.
    pub values: Vec<u8>,
}

/// JPEG encoder with builder-pattern configuration.
pub struct Encoder<'a> {
    pixels: &'a [u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    optimize_huffman: bool,
    progressive: bool,
    arithmetic: bool,
    lossless: bool,
    lossless_predictor: u8,
    lossless_point_transform: u8,
    grayscale_from_color: bool,
    restart_interval: Option<RestartConfig>,
    icc_profile: Option<&'a [u8]>,
    exif_data: Option<&'a [u8]>,
    comment: Option<&'a str>,
    scan_script: Option<Vec<ScanScript>>,
    quality_factors: Option<[u8; 4]>,
    custom_quant_tables: [Option<[u16; 64]>; 4],
    custom_huffman_dc: [Option<HuffmanTableDef>; 4],
    custom_huffman_ac: [Option<HuffmanTableDef>; 4],
    dct_method: DctMethod,
    saved_markers: Vec<SavedMarker>,
    /// When true, constrain quantization table values to 1-255 for baseline JPEG compatibility.
    force_baseline: bool,
    /// When true, pixel rows are read bottom-to-top.
    bottom_up: bool,
    /// Explicit JPEG colorspace override. When `None`, auto-detected from pixel format.
    colorspace_override: Option<ColorSpace>,
    /// Linear scale factor for quantization (set via `linear_quality()`).
    /// When `Some`, overrides the quality-based scaling.
    linear_scale_factor: Option<u32>,
}

impl<'a> Encoder<'a> {
    /// Create a new encoder for the given pixel data.
    pub fn new(pixels: &'a [u8], width: usize, height: usize, pixel_format: PixelFormat) -> Self {
        Self {
            pixels,
            width,
            height,
            pixel_format,
            quality: 75,
            subsampling: Subsampling::S420,
            optimize_huffman: false,
            progressive: false,
            arithmetic: false,
            lossless: false,
            lossless_predictor: 1,
            lossless_point_transform: 0,
            grayscale_from_color: false,
            restart_interval: None,
            quality_factors: None,
            scan_script: None,
            icc_profile: None,
            exif_data: None,
            comment: None,
            custom_quant_tables: [None; 4],
            custom_huffman_dc: [None, None, None, None],
            custom_huffman_ac: [None, None, None, None],
            dct_method: DctMethod::IsLow,
            saved_markers: Vec::new(),
            force_baseline: false,
            bottom_up: false,
            colorspace_override: None,
            linear_scale_factor: None,
        }
    }

    /// Set JPEG quality (1-100, default 75).
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Set per-component quality for a specific quantization table slot (0-3).
    ///
    /// This allows different quality levels for each component. For example,
    /// table 0 controls luma quality and table 1 controls chroma quality.
    /// Slots without explicit quality factors fall back to the global quality.
    pub fn quality_factor(mut self, table_index: usize, quality: u8) -> Self {
        assert!(table_index < 4, "quality factor table index must be 0..3");
        let factors = self.quality_factors.get_or_insert([self.quality; 4]);
        factors[table_index] = quality;
        self
    }

    /// Set chroma subsampling (default S420).
    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.subsampling = subsampling;
        self
    }

    /// Enable 2-pass optimized Huffman tables.
    pub fn optimize_huffman(mut self, optimize: bool) -> Self {
        self.optimize_huffman = optimize;
        self
    }

    /// Enable progressive JPEG mode.
    pub fn progressive(mut self, progressive: bool) -> Self {
        self.progressive = progressive;
        self
    }

    /// Set a custom progressive scan script.
    ///
    /// When progressive mode is enabled, this script replaces the default
    /// `simple_progression()` scan order. Each `ScanScript` entry defines
    /// one scan pass with its component set and spectral/successive-approximation
    /// parameters.
    pub fn scan_script(mut self, script: Vec<ScanScript>) -> Self {
        self.scan_script = Some(script);
        self
    }

    /// Enable arithmetic entropy coding.
    pub fn arithmetic(mut self, arithmetic: bool) -> Self {
        self.arithmetic = arithmetic;
        self
    }

    /// Enable lossless JPEG mode (SOF3).
    pub fn lossless(mut self, lossless: bool) -> Self {
        self.lossless = lossless;
        self
    }

    /// Set the lossless predictor selection value (1-7).
    ///
    /// Only used when `lossless(true)` is set. Default is 1 (left neighbor).
    /// See ITU-T T.81 Table H.1 for predictor definitions.
    pub fn lossless_predictor(mut self, predictor: u8) -> Self {
        self.lossless_predictor = predictor;
        self
    }

    /// Set the lossless point transform value (0-15).
    ///
    /// Only used when `lossless(true)` is set. Default is 0 (no transform).
    /// Shifts pixel values right by this amount before encoding, reducing
    /// precision but improving compression.
    pub fn lossless_point_transform(mut self, point_transform: u8) -> Self {
        self.lossless_point_transform = point_transform;
        self
    }

    /// Convert color input to single-component grayscale by extracting Y (luminance).
    pub fn grayscale_from_color(mut self, v: bool) -> Self {
        self.grayscale_from_color = v;
        self
    }

    /// Set restart interval in MCU blocks.
    ///
    /// A restart marker (RST0..RST7) will be emitted every `n` MCU blocks,
    /// allowing partial error recovery during decoding.
    pub fn restart_blocks(mut self, n: u16) -> Self {
        self.restart_interval = Some(RestartConfig::Blocks(n));
        self
    }

    /// Set restart interval in MCU rows.
    ///
    /// A restart marker will be emitted after every `n` rows of MCUs.
    /// The actual interval in blocks is `n * mcus_per_row`.
    pub fn restart_rows(mut self, n: u16) -> Self {
        self.restart_interval = Some(RestartConfig::Rows(n));
        self
    }

    /// Embed an ICC color profile.
    pub fn icc_profile(mut self, data: &'a [u8]) -> Self {
        self.icc_profile = Some(data);
        self
    }

    /// Embed EXIF metadata (raw TIFF data).
    pub fn exif_data(mut self, data: &'a [u8]) -> Self {
        self.exif_data = Some(data);
        self
    }

    /// Set a COM (comment) marker in the JPEG output.
    pub fn comment(mut self, text: &'a str) -> Self {
        self.comment = Some(text);
        self
    }

    /// Add a saved marker (APP or COM) to the JPEG output.
    ///
    /// Multiple markers of the same type can be added; they will appear
    /// in the order added, after JFIF/ICC/EXIF but before DQT/SOF/SOS.
    pub fn saved_marker(mut self, marker: SavedMarker) -> Self {
        self.saved_markers.push(marker);
        self
    }

    /// Select the DCT algorithm for encoding (default: `DctMethod::IsLow`).
    ///
    /// - `IsLow`: accurate integer DCT (13-bit fixed-point)
    /// - `IsFast`: fast integer DCT with reduced accuracy (8-bit fixed-point)
    /// - `Float`: floating-point DCT
    pub fn dct_method(mut self, method: DctMethod) -> Self {
        self.dct_method = method;
        self
    }

    /// Constrain quantization table values to 1-255 for baseline JPEG compatibility.
    ///
    /// When true, any quantization value exceeding 255 is clamped to 255.
    /// This ensures the output JPEG is decodable by baseline-only decoders.
    /// Matches libjpeg-turbo's `force_baseline` parameter on `jpeg_set_quality()`.
    pub fn force_baseline(mut self, force: bool) -> Self {
        self.force_baseline = force;
        self
    }

    /// Read pixel rows bottom-to-top instead of top-to-bottom.
    ///
    /// When true, the encoder reverses the row order before encoding so that
    /// the last row in the buffer becomes the first row in the JPEG image.
    /// Matches libjpeg-turbo's `TJPARAM_BOTTOMUP`.
    pub fn bottom_up(mut self, bottom_up: bool) -> Self {
        self.bottom_up = bottom_up;
        self
    }

    /// Set an explicit JPEG colorspace, overriding automatic detection.
    ///
    /// By default, the encoder auto-selects the JPEG colorspace based on the
    /// input pixel format (e.g., RGB input -> YCbCr JPEG). This method lets
    /// you override that choice, e.g., to store RGB data directly without
    /// conversion to YCbCr. Matches libjpeg-turbo's `jpeg_set_colorspace()`.
    pub fn colorspace(mut self, cs: ColorSpace) -> Self {
        self.colorspace_override = Some(cs);
        self
    }

    /// Set quality using a linear scale factor instead of the 1-100 quality rating.
    ///
    /// The scale factor directly controls quantization table scaling as a percentage:
    /// - 100 means use the standard tables as-is (equivalent to quality 50)
    /// - 50 means halve the table values (equivalent to quality 75)
    /// - 200 means double the table values (equivalent to quality 25)
    ///
    /// Matches libjpeg-turbo's `jpeg_set_linear_quality()`.
    pub fn linear_quality(mut self, scale_factor: u32) -> Self {
        self.linear_scale_factor = Some(scale_factor);
        self
    }

    /// Set a custom quantization table for the given table slot (0-3).
    ///
    /// Index 0 is used for luma, index 1 for chroma. When set, the custom
    /// table overrides the quality-scaled standard table for that component.
    /// Values are raw quantization coefficients in zigzag order.
    pub fn quant_table(mut self, index: usize, table: [u16; 64]) -> Self {
        assert!(index < 4, "quantization table index must be 0..3");
        self.custom_quant_tables[index] = Some(table);
        self
    }

    /// Set a custom DC Huffman table for the given table slot (0-3).
    ///
    /// Index 0 is used for luma, index 1 for chroma. When set, the custom table
    /// overrides the standard DC Huffman table for that slot during encoding.
    pub fn huffman_dc_table(mut self, index: usize, table: HuffmanTableDef) -> Self {
        assert!(index < 4, "Huffman table index must be 0..3");
        self.custom_huffman_dc[index] = Some(table);
        self
    }

    /// Set a custom AC Huffman table for the given table slot (0-3).
    ///
    /// Index 0 is used for luma, index 1 for chroma. When set, the custom table
    /// overrides the standard AC Huffman table for that slot during encoding.
    pub fn huffman_ac_table(mut self, index: usize, table: HuffmanTableDef) -> Self {
        assert!(index < 4, "Huffman table index must be 0..3");
        self.custom_huffman_ac[index] = Some(table);
        self
    }

    /// Compute the restart interval in MCU blocks from the configured restart setting.
    fn compute_restart_interval(&self) -> u16 {
        match self.restart_interval {
            None => 0,
            Some(RestartConfig::Blocks(n)) => n,
            Some(RestartConfig::Rows(n)) => {
                let mcu_w = if self.pixel_format == PixelFormat::Grayscale {
                    8
                } else {
                    match self.subsampling {
                        Subsampling::S444
                        | Subsampling::S440
                        | Subsampling::S441
                        | Subsampling::Unknown => 8,
                        Subsampling::S422 | Subsampling::S420 => 16,
                        Subsampling::S411 => 32,
                    }
                };
                let mcus_x = ((self.width + mcu_w - 1) / mcu_w) as u16;
                n.saturating_mul(mcus_x)
            }
        }
    }

    /// Build effective quantization tables, merging per-component quality factors
    /// with any explicitly set custom quant tables. Custom quant tables take
    /// priority over quality factors for the same slot.
    fn effective_quant_tables(&self) -> [Option<[u16; 64]>; 4] {
        let mut result = self.custom_quant_tables;

        // Apply force_baseline clamping to explicit custom tables
        if self.force_baseline {
            for table in result.iter_mut().flatten() {
                for val in table.iter_mut() {
                    if *val > 255 {
                        *val = 255;
                    }
                }
            }
        }

        if let Some(factors) = self.quality_factors {
            // Standard base tables: slot 0 = luminance, slots 1-3 = chrominance
            let base_tables: [&[u8; 64]; 4] = [
                &tables::STD_LUMINANCE_QUANT_TABLE,
                &tables::STD_CHROMINANCE_QUANT_TABLE,
                &tables::STD_CHROMINANCE_QUANT_TABLE,
                &tables::STD_CHROMINANCE_QUANT_TABLE,
            ];
            for (i, base) in base_tables.iter().enumerate() {
                if result[i].is_none() {
                    let scale: u32 = quality::quality_scaling(factors[i]);
                    result[i] = Some(quality::scale_quant_table_linear(
                        base,
                        scale,
                        self.force_baseline,
                    ));
                }
            }
        }
        result
    }

    /// Returns true if any custom quantization table has been set, either
    /// explicitly or via per-component quality factors.
    fn has_custom_quant_tables(&self) -> bool {
        self.custom_quant_tables.iter().any(|t| t.is_some()) || self.quality_factors.is_some()
    }

    /// Returns true if any custom Huffman table has been set.
    fn has_custom_huffman_tables(&self) -> bool {
        self.custom_huffman_dc.iter().any(|t| t.is_some())
            || self.custom_huffman_ac.iter().any(|t| t.is_some())
    }

    /// Flip pixel rows vertically for bottom-up encoding.
    fn flip_rows(pixels: &[u8], width: usize, height: usize, bpp: usize) -> Vec<u8> {
        let row_bytes: usize = width * bpp;
        let mut flipped: Vec<u8> = Vec::with_capacity(pixels.len());
        for row in (0..height).rev() {
            let start: usize = row * row_bytes;
            flipped.extend_from_slice(&pixels[start..start + row_bytes]);
        }
        flipped
    }

    /// Extract Y (luminance) from color pixels using BT.601 coefficients.
    fn extract_luminance(pixels: &[u8], n: usize, pf: PixelFormat) -> Vec<u8> {
        let mut y = Vec::with_capacity(n);
        match pf {
            PixelFormat::Grayscale => y.extend_from_slice(&pixels[..n]),
            PixelFormat::Rgb => {
                for c in pixels[..n * 3].chunks_exact(3) {
                    y.push(
                        ((19595 * c[0] as u32 + 38470 * c[1] as u32 + 7471 * c[2] as u32 + 32768)
                            >> 16) as u8,
                    );
                }
            }
            PixelFormat::Rgba => {
                for c in pixels[..n * 4].chunks_exact(4) {
                    y.push(
                        ((19595 * c[0] as u32 + 38470 * c[1] as u32 + 7471 * c[2] as u32 + 32768)
                            >> 16) as u8,
                    );
                }
            }
            PixelFormat::Bgr => {
                for c in pixels[..n * 3].chunks_exact(3) {
                    y.push(
                        ((19595 * c[2] as u32 + 38470 * c[1] as u32 + 7471 * c[0] as u32 + 32768)
                            >> 16) as u8,
                    );
                }
            }
            PixelFormat::Bgra => {
                for c in pixels[..n * 4].chunks_exact(4) {
                    y.push(
                        ((19595 * c[2] as u32 + 38470 * c[1] as u32 + 7471 * c[0] as u32 + 32768)
                            >> 16) as u8,
                    );
                }
            }
            PixelFormat::Rgbx
            | PixelFormat::Xrgb
            | PixelFormat::Argb
            | PixelFormat::Bgrx
            | PixelFormat::Xbgr
            | PixelFormat::Abgr => {
                let r_off: usize = pf.red_offset().unwrap();
                let g_off: usize = pf.green_offset().unwrap();
                let b_off: usize = pf.blue_offset().unwrap();
                for c in pixels[..n * 4].chunks_exact(4) {
                    y.push(
                        ((19595 * c[r_off] as u32
                            + 38470 * c[g_off] as u32
                            + 7471 * c[b_off] as u32
                            + 32768)
                            >> 16) as u8,
                    );
                }
            }
            PixelFormat::Cmyk => y.resize(n, 128),
            PixelFormat::Rgb565 => y.resize(n, 128),
        }
        y
    }

    /// Determine the effective quality for encoding, accounting for linear_quality override.
    fn effective_quality(&self) -> u8 {
        if let Some(scale) = self.linear_scale_factor {
            // Reverse-map the linear scale factor to a quality value.
            // quality_scaling(q) produces the scale factor; we need the inverse.
            // For q < 50: scale = 5000 / q  =>  q = 5000 / scale
            // For q >= 50: scale = 200 - 2*q  =>  q = (200 - scale) / 2
            // At the boundary q=50, scale=100.
            if scale >= 100 {
                // q < 50 range
                let q: u32 = 5000 / scale.max(1);
                q.clamp(1, 100) as u8
            } else {
                // q >= 50 range
                let q: u32 = (200 - scale) / 2;
                q.clamp(1, 100) as u8
            }
        } else {
            self.quality
        }
    }

    /// Encode and return the JPEG byte stream.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let restart_interval = self.compute_restart_interval();

        // Handle bottom-up row flipping
        let flipped_buf: Vec<u8>;
        let input_pixels: &[u8] = if self.bottom_up {
            flipped_buf = Self::flip_rows(
                self.pixels,
                self.width,
                self.height,
                self.pixel_format.bytes_per_pixel(),
            );
            &flipped_buf
        } else {
            self.pixels
        };

        let (effective_pixels, effective_format);
        let gray_buf: Vec<u8>;
        if self.grayscale_from_color && self.pixel_format != PixelFormat::Grayscale {
            gray_buf =
                Self::extract_luminance(input_pixels, self.width * self.height, self.pixel_format);
            effective_pixels = &gray_buf[..];
            effective_format = PixelFormat::Grayscale;
        } else {
            effective_pixels = input_pixels;
            effective_format = self.pixel_format;
        }

        // Use the effective quality (handles linear_quality override)
        let quality: u8 = self.effective_quality();

        // Check if force_baseline or linear_quality requires custom quant tables
        let needs_custom_quant: bool = self.force_baseline
            || self.linear_scale_factor.is_some()
            || self.has_custom_quant_tables();

        let base = if self.lossless && self.arithmetic {
            encoder::compress_lossless_arithmetic(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.lossless_predictor,
                self.lossless_point_transform,
            )?
        } else if self.lossless {
            encoder::compress_lossless_extended(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.lossless_predictor,
                self.lossless_point_transform,
            )?
        } else if self.arithmetic && self.progressive {
            encoder::compress_arithmetic_progressive(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                self.subsampling,
            )?
        } else if self.arithmetic {
            encoder::compress_arithmetic(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                self.subsampling,
            )?
        } else if self.progressive {
            if let Some(ref script) = self.scan_script {
                encoder::compress_progressive_custom(
                    effective_pixels,
                    self.width,
                    self.height,
                    effective_format,
                    quality,
                    self.subsampling,
                    script,
                )?
            } else {
                encoder::compress_progressive(
                    effective_pixels,
                    self.width,
                    self.height,
                    effective_format,
                    quality,
                    self.subsampling,
                )?
            }
        } else if self.optimize_huffman {
            encoder::compress_optimized(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                self.subsampling,
            )?
        } else if self.has_custom_huffman_tables() {
            encoder::compress_custom_huffman(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                self.subsampling,
                &self.custom_huffman_dc,
                &self.custom_huffman_ac,
            )?
        } else if needs_custom_quant {
            let effective_tables = self.build_quant_tables(quality);
            encoder::compress_custom_quant(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                self.subsampling,
                &effective_tables,
            )?
        } else if restart_interval > 0 {
            encoder::compress_with_restart(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                self.subsampling,
                restart_interval,
            )?
        } else {
            encoder::compress(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                self.subsampling,
                self.dct_method,
            )?
        };

        let with_meta = if self.icc_profile.is_some() || self.exif_data.is_some() {
            encoder::inject_metadata(&base, self.icc_profile, self.exif_data)?
        } else {
            base
        };

        let with_comment: Vec<u8> = if let Some(text) = self.comment {
            encoder::inject_comment(&with_meta, text)
        } else {
            with_meta
        };

        if self.saved_markers.is_empty() {
            Ok(with_comment)
        } else {
            Ok(encoder::inject_saved_markers(
                &with_comment,
                &self.saved_markers,
            ))
        }
    }

    /// Build quantization tables for force_baseline / linear_quality scenarios.
    fn build_quant_tables(&self, quality: u8) -> [Option<[u16; 64]>; 4] {
        // Start with any explicit custom tables
        let mut result = self.custom_quant_tables;

        // Apply force_baseline clamping to explicit custom tables
        if self.force_baseline {
            for table in result.iter_mut().flatten() {
                for val in table.iter_mut() {
                    if *val > 255 {
                        *val = 255;
                    }
                }
            }
        }

        // For per-component quality factors
        if let Some(factors) = self.quality_factors {
            let base_tables: [&[u8; 64]; 4] = [
                &tables::STD_LUMINANCE_QUANT_TABLE,
                &tables::STD_CHROMINANCE_QUANT_TABLE,
                &tables::STD_CHROMINANCE_QUANT_TABLE,
                &tables::STD_CHROMINANCE_QUANT_TABLE,
            ];
            for (i, base) in base_tables.iter().enumerate() {
                if result[i].is_none() {
                    let scale: u32 = quality::quality_scaling(factors[i]);
                    result[i] = Some(quality::scale_quant_table_linear(
                        base,
                        scale,
                        self.force_baseline,
                    ));
                }
            }
            return result;
        }

        // For linear_quality or force_baseline without per-component factors
        let scale: u32 = if let Some(sf) = self.linear_scale_factor {
            sf
        } else {
            quality::quality_scaling(quality)
        };

        if result[0].is_none() {
            result[0] = Some(quality::scale_quant_table_linear(
                &tables::STD_LUMINANCE_QUANT_TABLE,
                scale,
                self.force_baseline,
            ));
        }
        if result[1].is_none() {
            result[1] = Some(quality::scale_quant_table_linear(
                &tables::STD_CHROMINANCE_QUANT_TABLE,
                scale,
                self.force_baseline,
            ));
        }

        result
    }
}
