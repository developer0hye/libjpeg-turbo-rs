use crate::common::error::Result;
use crate::common::types::{PixelFormat, Subsampling};
use crate::encode::pipeline as encoder;

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
    custom_quant_tables: [Option<[u16; 64]>; 4],
    custom_huffman_dc: [Option<HuffmanTableDef>; 4],
    custom_huffman_ac: [Option<HuffmanTableDef>; 4],
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
            icc_profile: None,
            exif_data: None,
            comment: None,
            custom_quant_tables: [None; 4],
            custom_huffman_dc: [None, None, None, None],
            custom_huffman_ac: [None, None, None, None],
        }
    }

    /// Set JPEG quality (1-100, default 75).
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality;
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
                        Subsampling::S444 | Subsampling::S440 => 8,
                        Subsampling::S422 | Subsampling::S420 => 16,
                        Subsampling::S411 => 32,
                    }
                };
                let mcus_x = ((self.width + mcu_w - 1) / mcu_w) as u16;
                n.saturating_mul(mcus_x)
            }
        }
    }

    /// Returns true if any custom quantization table has been set.
    fn has_custom_quant_tables(&self) -> bool {
        self.custom_quant_tables.iter().any(|t| t.is_some())
    }

    /// Returns true if any custom Huffman table has been set.
    fn has_custom_huffman_tables(&self) -> bool {
        self.custom_huffman_dc.iter().any(|t| t.is_some())
            || self.custom_huffman_ac.iter().any(|t| t.is_some())
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
            PixelFormat::Cmyk => y.resize(n, 128),
        }
        y
    }

    /// Encode and return the JPEG byte stream.
    pub fn encode(&self) -> Result<Vec<u8>> {
        let restart_interval = self.compute_restart_interval();

        let (effective_pixels, effective_format);
        let gray_buf: Vec<u8>;
        if self.grayscale_from_color && self.pixel_format != PixelFormat::Grayscale {
            gray_buf =
                Self::extract_luminance(self.pixels, self.width * self.height, self.pixel_format);
            effective_pixels = &gray_buf[..];
            effective_format = PixelFormat::Grayscale;
        } else {
            effective_pixels = self.pixels;
            effective_format = self.pixel_format;
        }

        let base = if self.lossless {
            encoder::compress_lossless_extended(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.lossless_predictor,
                self.lossless_point_transform,
            )?
        } else if self.arithmetic {
            encoder::compress_arithmetic(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.quality,
                self.subsampling,
            )?
        } else if self.progressive {
            encoder::compress_progressive(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.quality,
                self.subsampling,
            )?
        } else if self.optimize_huffman {
            encoder::compress_optimized(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.quality,
                self.subsampling,
            )?
        } else if self.has_custom_huffman_tables() {
            encoder::compress_custom_huffman(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.quality,
                self.subsampling,
                &self.custom_huffman_dc,
                &self.custom_huffman_ac,
            )?
        } else if self.has_custom_quant_tables() {
            encoder::compress_custom_quant(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.quality,
                self.subsampling,
                &self.custom_quant_tables,
            )?
        } else if restart_interval > 0 {
            encoder::compress_with_restart(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.quality,
                self.subsampling,
                restart_interval,
            )?
        } else {
            encoder::compress(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                self.quality,
                self.subsampling,
            )?
        };

        let with_meta = if self.icc_profile.is_some() || self.exif_data.is_some() {
            encoder::inject_metadata(&base, self.icc_profile, self.exif_data)?
        } else {
            base
        };

        if let Some(text) = self.comment {
            Ok(encoder::inject_comment(&with_meta, text))
        } else {
            Ok(with_meta)
        }
    }
}
