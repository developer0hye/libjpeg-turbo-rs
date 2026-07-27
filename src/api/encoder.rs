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
    xmp_data: Option<&'a [u8]>,
    iptc_data: Option<&'a [u8]>,
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
    /// Smoothing factor for pre-encode noise reduction (0-100). 0 = no smoothing.
    smoothing_factor: u8,
    /// When true, use a triangle/tent filter for chroma downsampling. Default is true.
    fancy_downsampling: bool,
    /// Custom JFIF version override. When `Some`, replaces the default 1.01.
    jfif_version: Option<(u8, u8)>,
    /// JFIF density override. When `Some`, patches the APP0 density fields.
    density: Option<(u8, u16, u16)>,
    /// Adobe APP14 marker control. `None` = auto. `Some(true)` = always. `Some(false)` = never.
    write_adobe_marker: Option<bool>,
    /// Custom per-component sampling factors as (h, v) pairs.
    /// When set, overrides the `subsampling` enum with explicit factors.
    /// The first component (Y) defines the max sampling factor; subsequent
    /// components (Cb, Cr) can use any factor from 1 to max_h/max_v.
    custom_sampling_factors: Option<Vec<(u8, u8)>>,
    /// When true, omit DQT and DHT markers from the output (abbreviated body-only stream).
    /// The resulting JPEG body can be decoded only when a decoder has preloaded the matching tables.
    suppress_tables: bool,
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
            xmp_data: None,
            iptc_data: None,
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
            smoothing_factor: 0,
            fancy_downsampling: false,
            jfif_version: None,
            density: None,
            write_adobe_marker: None,
            custom_sampling_factors: None,
            suppress_tables: false,
        }
    }

    /// Set JPEG quality (1-100, default 75).
    pub fn quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    /// Set per-component quality for a specific quantization table slot (0-3).
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
    pub fn lossless_predictor(mut self, predictor: u8) -> Self {
        self.lossless_predictor = predictor;
        self
    }

    /// Set the lossless point transform value (0-15).
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
    pub fn restart_blocks(mut self, n: u16) -> Self {
        self.restart_interval = Some(RestartConfig::Blocks(n));
        self
    }

    /// Set restart interval in MCU rows.
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

    /// Embed an XMP packet (APP1 `http://ns.adobe.com/xap/1.0/`).
    /// Packets larger than one APP1 segment error at encode time —
    /// Extended XMP writing is not implemented (issue #358).
    pub fn xmp_data(mut self, data: &'a [u8]) -> Self {
        self.xmp_data = Some(data);
        self
    }

    /// Embed an IPTC IIM payload in an APP13 Photoshop IRB (0x0404).
    pub fn iptc_data(mut self, data: &'a [u8]) -> Self {
        self.iptc_data = Some(data);
        self
    }

    /// Set a COM (comment) marker in the JPEG output.
    pub fn comment(mut self, text: &'a str) -> Self {
        self.comment = Some(text);
        self
    }

    /// Add a saved marker (APP or COM) to the JPEG output.
    pub fn saved_marker(mut self, marker: SavedMarker) -> Self {
        self.saved_markers.push(marker);
        self
    }

    /// Select the DCT algorithm for encoding.
    pub fn dct_method(mut self, method: DctMethod) -> Self {
        self.dct_method = method;
        self
    }

    /// Constrain quantization table values to 1-255 for baseline JPEG compatibility.
    pub fn force_baseline(mut self, force: bool) -> Self {
        self.force_baseline = force;
        self
    }

    /// Read pixel rows bottom-to-top instead of top-to-bottom.
    pub fn bottom_up(mut self, bottom_up: bool) -> Self {
        self.bottom_up = bottom_up;
        self
    }

    /// Set an explicit JPEG colorspace, overriding automatic detection.
    pub fn colorspace(mut self, cs: ColorSpace) -> Self {
        self.colorspace_override = Some(cs);
        self
    }

    /// Reset colorspace to auto-detection (like `jpeg_default_colorspace`).
    ///
    /// Clears any explicit colorspace override set by `colorspace()`.
    /// The encoder will infer the JPEG colorspace from the pixel format.
    pub fn reset_colorspace(mut self) -> Self {
        self.colorspace_override = None;
        self
    }

    /// Reset quantization tables to defaults (like `jpeg_default_qtables`).
    ///
    /// Clears any custom quantization tables and per-slot quality factors,
    /// forcing the encoder to regenerate the standard luminance and
    /// chrominance tables scaled by the current quality factor.
    ///
    /// `force_baseline` mirrors libjpeg's argument: when `true`, all
    /// quantization coefficients are clamped to 1..=255 so the output
    /// conforms to baseline JPEG (SOF0) decoder requirements. When
    /// `false`, the extended range (up to 32767) is allowed, which is
    /// useful at very low qualities.
    pub fn reset_quant_tables(mut self, force_baseline: bool) -> Self {
        self.custom_quant_tables = [None; 4];
        self.quality_factors = None;
        self.force_baseline = force_baseline;
        self
    }

    /// Set quality using a linear scale factor instead of the 1-100 quality rating.
    pub fn linear_quality(mut self, scale_factor: u32) -> Self {
        self.linear_scale_factor = Some(scale_factor);
        self
    }

    /// Set input smoothing factor (0-100, default 0).
    ///
    /// When greater than 0, applies a pre-encode smoothing filter to reduce
    /// noise artifacts at low quality settings. Matches libjpeg-turbo's `smoothing_factor`.
    pub fn smoothing_factor(mut self, factor: u8) -> Self {
        self.smoothing_factor = factor.min(100);
        self
    }

    /// Enable or disable fancy chroma downsampling (default: true).
    ///
    /// When true, uses a triangle/tent filter for chroma downsampling.
    /// When false, uses a simple box average.
    /// Matches libjpeg-turbo's `do_fancy_downsampling`.
    pub fn fancy_downsampling(mut self, fancy: bool) -> Self {
        self.fancy_downsampling = fancy;
        self
    }

    /// Set the JFIF version in the APP0 marker (default: 1.01).
    pub fn jfif_version(mut self, major: u8, minor: u8) -> Self {
        self.jfif_version = Some((major, minor));
        self
    }

    /// Set JFIF density (unit, x_density, y_density).
    ///
    /// Unit: 0 = unknown, 1 = DPI, 2 = DPCM. Patches the APP0 JFIF marker
    /// after encoding. Matches C libjpeg-turbo's `density_unit`/`X_density`/`Y_density`.
    pub fn density(mut self, unit: u8, x: u16, y: u16) -> Self {
        self.density = Some((unit, x, y));
        self
    }

    /// Control whether the Adobe APP14 marker is written.
    ///
    /// By default, the Adobe marker is written automatically for CMYK images
    /// and omitted for others. Matches libjpeg-turbo's `write_Adobe_marker`.
    pub fn write_adobe_marker(mut self, write: bool) -> Self {
        self.write_adobe_marker = Some(write);
        self
    }

    /// Set explicit per-component sampling factors, overriding the `subsampling` enum.
    ///
    /// `factors` is a list of `(h_sampling, v_sampling)` per component. For a
    /// 3-component YCbCr image, provide 3 entries:
    /// - `factors[0]` = Y (luminance) sampling factor
    /// - `factors[1]` = Cb sampling factor
    /// - `factors[2]` = Cr sampling factor
    ///
    /// The first component typically has the largest factors (e.g., `(3, 2)` for
    /// 3x2 sampling). Chroma components usually use `(1, 1)`.
    ///
    /// Valid factor values are 1..=4 for each dimension.
    pub fn sampling_factors(mut self, factors: Vec<(u8, u8)>) -> Self {
        self.custom_sampling_factors = Some(factors);
        self
    }

    /// Suppress DQT and DHT markers in the encoded output (abbreviated body-only stream).
    ///
    /// When `true`, the encoded JPEG omits all quantization and Huffman table markers.
    /// The resulting stream cannot be decoded on its own; a decoder must first load the
    /// matching tables via `read_header()` / `Decoder::new_with_tables()`.
    ///
    /// Matches libjpeg-turbo's abbreviated compressed data datastream (JPEG spec F.1.2.4).
    pub fn suppress_tables(mut self, suppress: bool) -> Self {
        self.suppress_tables = suppress;
        self
    }

    /// Produce a tables-only abbreviated datastream for this encoder's configuration.
    ///
    /// Returns `SOI + DQT(s) + DHT(s) + EOI` with no image data. The stream
    /// can be parsed by `read_header()` to preload tables into a decoder for
    /// subsequent decoding of body-only streams.
    ///
    /// Matches libjpeg-turbo's `jpeg_write_tables()` (JPEG spec F.1.2.4).
    pub fn write_tables(&self) -> Vec<u8> {
        crate::api::abbreviated::write_tables_for_encoder(self)
    }

    /// Set a custom quantization table for the given table slot (0-3).
    pub fn quant_table(mut self, index: usize, table: [u16; 64]) -> Self {
        assert!(index < 4, "quantization table index must be 0..3");
        self.custom_quant_tables[index] = Some(table);
        self
    }

    /// Set a custom DC Huffman table for the given table slot (0-3).
    pub fn huffman_dc_table(mut self, index: usize, table: HuffmanTableDef) -> Self {
        assert!(index < 4, "Huffman table index must be 0..3");
        self.custom_huffman_dc[index] = Some(table);
        self
    }

    /// Set a custom AC Huffman table for the given table slot (0-3).
    pub fn huffman_ac_table(mut self, index: usize, table: HuffmanTableDef) -> Self {
        assert!(index < 4, "Huffman table index must be 0..3");
        self.custom_huffman_ac[index] = Some(table);
        self
    }

    /// Compute the restart interval in MCUs.
    ///
    /// `effective_subsampling` is the subsampling actually used by the
    /// encode pipeline — typically `self.subsampling`, but when the caller
    /// provided `sampling_factors([(h,v),(1,1),(1,1)])` instead it is the
    /// `Subsampling` variant the factors map to (see `mapped_subsampling`).
    /// `restart_rows(n)` translates to `n * MCUs_per_row`, and the number
    /// of MCUs per row depends on the MCU width — which depends on the
    /// effective subsampling, NOT on the field default.
    fn compute_restart_interval(&self, effective_subsampling: Subsampling) -> u16 {
        match self.restart_interval {
            None => 0,
            Some(RestartConfig::Blocks(n)) => n,
            Some(RestartConfig::Rows(n)) => {
                // Lossless: 1 MCU = 1 pixel (no 8x8 blocks).
                let mcu_w: usize = if self.lossless {
                    1
                } else if self.pixel_format == PixelFormat::Grayscale {
                    8
                } else {
                    match effective_subsampling {
                        Subsampling::S444
                        | Subsampling::S440
                        | Subsampling::S441
                        | Subsampling::Unknown => 8,
                        Subsampling::S422 | Subsampling::S420 | Subsampling::S24 => 16,
                        Subsampling::S411 | Subsampling::S410 => 32,
                    }
                };
                let mcus_x: u16 = self.width.div_ceil(mcu_w) as u16;
                n.saturating_mul(mcus_x)
            }
        }
    }

    fn has_custom_quant_tables(&self) -> bool {
        self.custom_quant_tables.iter().any(|t| t.is_some()) || self.quality_factors.is_some()
    }

    fn has_custom_huffman_tables(&self) -> bool {
        self.custom_huffman_dc.iter().any(|t| t.is_some())
            || self.custom_huffman_ac.iter().any(|t| t.is_some())
    }

    fn flip_rows(pixels: &[u8], width: usize, height: usize, bpp: usize) -> Vec<u8> {
        let row_bytes: usize = width * bpp;
        let mut flipped: Vec<u8> = Vec::with_capacity(pixels.len());
        for row in (0..height).rev() {
            let start: usize = row * row_bytes;
            flipped.extend_from_slice(&pixels[start..start + row_bytes]);
        }
        flipped
    }

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

    fn effective_quality(&self) -> u8 {
        if let Some(scale) = self.linear_scale_factor {
            if scale >= 100 {
                let q: u32 = 5000 / scale.max(1);
                q.clamp(1, 100) as u8
            } else {
                let q: u32 = (200 - scale) / 2;
                q.clamp(1, 100) as u8
            }
        } else {
            self.quality
        }
    }

    fn apply_triangle_prefilter(
        pixels: &[u8],
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
        subsampling: Subsampling,
    ) -> Vec<u8> {
        if width <= 2 || height <= 2 {
            return pixels.to_vec();
        }
        let bpp: usize = pixel_format.bytes_per_pixel();
        let row_stride: usize = width * bpp;
        let mut output: Vec<u8> = pixels.to_vec();
        let needs_h: bool = matches!(
            subsampling,
            Subsampling::S420 | Subsampling::S422 | Subsampling::S411
        );
        let needs_v: bool = matches!(
            subsampling,
            Subsampling::S420 | Subsampling::S440 | Subsampling::S441
        );
        if needs_h && bpp >= 3 {
            for y in 0..height {
                for x in 1..width - 1 {
                    for c in 0..bpp {
                        let idx: usize = y * row_stride + x * bpp + c;
                        let left: u16 = pixels[idx - bpp] as u16;
                        let center: u16 = pixels[idx] as u16;
                        let right: u16 = pixels[idx + bpp] as u16;
                        output[idx] = ((left + 2 * center + right + 2) >> 2) as u8;
                    }
                }
            }
        }
        if needs_v && bpp >= 3 {
            let source: Vec<u8> = output.clone();
            for y in 1..height - 1 {
                for x in 0..width {
                    for c in 0..bpp {
                        let idx: usize = y * row_stride + x * bpp + c;
                        let top: u16 = source[idx - row_stride] as u16;
                        let center: u16 = source[idx] as u16;
                        let bottom: u16 = source[idx + row_stride] as u16;
                        output[idx] = ((top + 2 * center + bottom + 2) >> 2) as u8;
                    }
                }
            }
        }
        output
    }

    fn patch_jfif_density(mut data: Vec<u8>, unit: u8, x: u16, y: u16) -> Vec<u8> {
        // JFIF APP0 layout: SOI(2) + FF E0(2) + len(2) + "JFIF\0"(5) + ver(2) + unit(1) + xden(2) + yden(2)
        if data.len() > 17 && data[2] == 0xFF && data[3] == 0xE0 && data[6..11] == *b"JFIF\0" {
            data[13] = unit;
            data[14..16].copy_from_slice(&x.to_be_bytes());
            data[16..18].copy_from_slice(&y.to_be_bytes());
        }
        data
    }

    fn patch_jfif_version(mut data: Vec<u8>, major: u8, minor: u8) -> Vec<u8> {
        if data.len() > 12 && data[2] == 0xFF && data[3] == 0xE0 && &data[6..11] == b"JFIF\0" {
            data[11] = major;
            data[12] = minor;
        }
        data
    }

    fn find_adobe_marker(data: &[u8]) -> Option<usize> {
        let mut pos: usize = 2;
        while pos + 1 < data.len() {
            if data[pos] != 0xFF {
                break;
            }
            let code: u8 = data[pos + 1];
            if code == 0xDA || code == 0xD9 {
                break;
            }
            if code == 0xEE && pos + 9 < data.len() && &data[pos + 4..pos + 9] == b"Adobe" {
                return Some(pos);
            }
            if pos + 3 < data.len() {
                let seg_len: usize = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                pos += 2 + seg_len;
            } else {
                break;
            }
        }
        None
    }

    fn inject_adobe_marker(data: Vec<u8>, transform: u8) -> Vec<u8> {
        let insert_pos: usize = if data.len() >= 4 && data[2] == 0xFF && data[3] == 0xE0 {
            let app0_len: usize = u16::from_be_bytes([data[4], data[5]]) as usize;
            2 + 2 + app0_len
        } else {
            2
        };
        let mut out: Vec<u8> = Vec::with_capacity(data.len() + 16);
        out.extend_from_slice(&data[..insert_pos]);
        crate::encode::marker_writer::write_app14_adobe(&mut out, transform);
        out.extend_from_slice(&data[insert_pos..]);
        out
    }

    fn strip_adobe_marker(data: Vec<u8>) -> Vec<u8> {
        if let Some(offset) = Self::find_adobe_marker(&data) {
            let seg_len: usize = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
            let marker_total: usize = 2 + seg_len;
            let mut out: Vec<u8> = Vec::with_capacity(data.len() - marker_total);
            out.extend_from_slice(&data[..offset]);
            out.extend_from_slice(&data[offset + marker_total..]);
            out
        } else {
            data
        }
    }

    /// Encode and return the JPEG byte stream.
    pub fn encode(&self) -> Result<Vec<u8>> {
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

        // Smoothing is handled inside the pipeline (compress_optimized applies
        // fullsize_smooth + h2v2_smooth_downsample matching C libjpeg-turbo).
        let after_smooth: &[u8] = input_pixels;

        // Apply fancy downsampling pre-filter if enabled and subsampling is active.
        // Skip when grayscale_from_color: grayscale has no chroma, so no downsampling
        // prefilter should be applied (matches C cjpeg -grayscale behavior).
        let fancy_buf: Vec<u8>;
        let after_fancy: &[u8] = if self.fancy_downsampling
            && self.smoothing_factor == 0
            && !self.grayscale_from_color
            && self.pixel_format != PixelFormat::Grayscale
            && self.pixel_format != PixelFormat::Cmyk
            && self.subsampling != Subsampling::S444
        {
            fancy_buf = Self::apply_triangle_prefilter(
                after_smooth,
                self.width,
                self.height,
                self.pixel_format,
                self.subsampling,
            );
            &fancy_buf
        } else {
            after_smooth
        };

        let (effective_pixels, effective_format);
        let gray_buf: Vec<u8>;
        if self.grayscale_from_color && self.pixel_format != PixelFormat::Grayscale {
            // Use the SIMD-dispatched rgb_to_ycbcr_row to extract Y channel.
            // This matches C libjpeg-turbo's NEON rgb_gray_convert, ensuring
            // byte-identical output.  extract_luminance() uses scalar math which
            // can differ by ±1 from NEON due to intermediate rounding.
            if self.pixel_format == PixelFormat::Rgb {
                let enc_simd = crate::simd::detect_encoder();
                let n: usize = self.width * self.height;
                let mut y_plane: Vec<u8> = vec![0u8; n];
                let mut cb_dummy: Vec<u8> = vec![0u8; n];
                let mut cr_dummy: Vec<u8> = vec![0u8; n];
                for row in 0..self.height {
                    let src_off: usize = row * self.width * 3;
                    let dst_off: usize = row * self.width;
                    (enc_simd.rgb_to_ycbcr_row)(
                        &after_fancy[src_off..src_off + self.width * 3],
                        &mut y_plane[dst_off..dst_off + self.width],
                        &mut cb_dummy[dst_off..dst_off + self.width],
                        &mut cr_dummy[dst_off..dst_off + self.width],
                        self.width,
                    );
                }
                gray_buf = y_plane;
            } else {
                gray_buf = Self::extract_luminance(
                    after_fancy,
                    self.width * self.height,
                    self.pixel_format,
                );
            }
            effective_pixels = &gray_buf[..];
            effective_format = PixelFormat::Grayscale;
        } else {
            effective_pixels = after_fancy;
            effective_format = self.pixel_format;
        }

        let quality: u8 = self.effective_quality();

        // RGB-direct encoding: bypass color conversion entirely.
        // Matches C cjpeg `-rgb` (JCS_RGB colorspace).
        //
        // This used to be an early `return` that forwarded pixels, dimensions,
        // quality and the ICC profile and dropped everything else on the floor
        // — restart interval, custom tables, optimized Huffman, smoothing, the
        // DCT method, and the comment / EXIF / saved-marker injection below
        // (#343). It is now a flag consumed by the first arm of the mode
        // dispatch, which keeps its precedence while carrying the full option
        // set.
        let rgb_direct: bool = self.colorspace_override == Some(ColorSpace::Rgb)
            && effective_format == PixelFormat::Rgb;

        // Route through compress_custom_quant whenever the builder may need
        // non-baseline (16-bit) quantization values. C cjpeg defaults to
        // `force_baseline = FALSE`, allowing scaled quant values up to 32767
        // at low quality (e.g. q=1 produces 16*5000/100 = 800 for the luma DC
        // entry). The default `compress(...)` path uses
        // `quality_scale_quant_table` which clamps to 255, breaking parity
        // with cjpeg at low quality. Routing through `compress_custom_quant`
        // with the builder-resolved tables (which honour `force_baseline`)
        // keeps high-quality output bit-identical (no clamp triggers) and
        // produces 16-bit DQT markers at low quality just like cjpeg.
        let scaled_quant_could_exceed_255 = !self.force_baseline && {
            let q = self.quality_factors.map(|f| f[0]).unwrap_or(self.quality);
            // 5000/q > 255 iff q <= 19 (since 5000/19 = 263, 5000/20 = 250).
            // Smallest base entry is 8 (luma AC[3,3]); largest is 121 (chroma).
            // 121 * (5000/q)/100 > 255 iff 5000/q > 211 iff q <= 23.
            // Use a generous threshold to be safe.
            q < 50
        };
        let needs_custom_quant: bool = self.force_baseline
            || self.linear_scale_factor.is_some()
            || self.has_custom_quant_tables()
            || scaled_quant_could_exceed_255;

        // Smoothing needs the full-plane buffering that only the baseline
        // optimized path provides; the progressive, arithmetic and lossless
        // paths downsample per block from unpadded planes and have no
        // equivalent. Rather than accept the option and drop it — which is
        // what used to happen, and is the whole subject of #322 — say so.
        if self.smoothing_factor > 0 && (self.progressive || self.arithmetic || self.lossless) {
            return Err(crate::common::error::JpegError::Unsupported(
                "smoothing_factor is not supported with progressive, arithmetic \
                 or lossless encoding; it requires the full-plane path used by \
                 baseline encodes"
                    .to_string(),
            ));
        }

        // Resolved once so every arm can pass it. Only `Some` when the builder
        // actually needs non-default tables, so the default path is unchanged.
        let progressive_quant_tables: Option<[Option<[u16; 64]>; 4]> = if needs_custom_quant {
            Some(self.build_quant_tables(quality))
        } else {
            None
        };

        // Map a 3-component custom sampling factor list to a standard
        // YCbCr `Subsampling` variant when one matches. This lets
        // `Encoder::sampling_factors([(h,v),(1,1),(1,1)])` route through the
        // optimised / progressive / arithmetic / SOF1 paths instead of the
        // baseline-only `compress_custom_sampling` path. Required for
        // c_tjcomptest_lossy_full byte-parity at samp410 / samp24 (the only
        // standard JPEG subsamplings without dedicated `subsampling()` API
        // sugar).
        let mapped_subsampling: Option<Subsampling> =
            self.custom_sampling_factors.as_deref().and_then(|f| {
                if f.len() != 3 || f[1] != (1, 1) || f[2] != (1, 1) {
                    return None;
                }
                Some(match f[0] {
                    (1, 1) => Subsampling::S444,
                    (2, 1) => Subsampling::S422,
                    (1, 2) => Subsampling::S440,
                    (2, 2) => Subsampling::S420,
                    (4, 1) => Subsampling::S411,
                    (1, 4) => Subsampling::S441,
                    (4, 2) => Subsampling::S410,
                    (2, 4) => Subsampling::S24,
                    _ => return None,
                })
            });
        let use_custom_sampling: bool =
            self.custom_sampling_factors.is_some() && mapped_subsampling.is_none();
        let effective_subsampling: Subsampling = mapped_subsampling.unwrap_or(self.subsampling);
        // restart_interval depends on the *effective* MCU width, which can
        // differ from `self.subsampling` when the caller used
        // `sampling_factors([(h,v),(1,1),(1,1)])` instead of `subsampling()`.
        // Compute it after the mapping so e.g. samp410 (4x2) lands on
        // mcu_w=32, not the default S420 mcu_w=16.
        // RGB-direct puts every component at 1x1 (`jcparam.c:365-370`), so its
        // MCU is 8 pixels wide whatever `subsampling` says. A row-based
        // restart interval counted against a 16-wide MCU would put the markers
        // on the wrong rows.
        let restart_subsampling: Subsampling = if rgb_direct {
            Subsampling::S444
        } else {
            effective_subsampling
        };
        let restart_interval: u16 = self.compute_restart_interval(restart_subsampling);
        // For progressive: each scan recomputes restart_interval from
        // `restart_in_rows * MCUs_per_row(scan)` — interleaved DC scans use
        // the iMCU width while non-interleaved AC scans use the per-component
        // width_in_blocks. Pass the rows hint so the progressive encoder can
        // re-derive per-scan; non-row restart specs leave this at 0 and the
        // pre-computed `restart_interval` is used as-is for every scan.
        let restart_in_rows: u16 = match self.restart_interval {
            Some(RestartConfig::Rows(n)) => n,
            _ => 0,
        };

        // One params value carrying every baseline option, instead of an if/else
        // chain in which the first matching arm silently discarded whatever it
        // could not express. That chain lost `restart_blocks` behind either
        // table option, custom quant behind custom Huffman, and `dct_method`
        // behind both — 29 masked interactions in all (#322) — and the
        // RGB-direct arm lost all six (#343). The core decides internally
        // whether the two-pass optimized path is needed.
        let effective_quant_tables = self.build_quant_tables(quality);
        let baseline_params = {
            let mut params = encoder::CompressParams::new(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                effective_subsampling,
            )
            .dct_method(self.dct_method)
            .restart_interval(restart_interval)
            .optimize_huffman(self.optimize_huffman)
            .smoothing_factor(self.smoothing_factor);
            if needs_custom_quant {
                params = params.custom_quant(&effective_quant_tables);
            }
            if self.has_custom_huffman_tables() {
                params = params.custom_huffman(&self.custom_huffman_dc, &self.custom_huffman_ac);
            }
            params
        };

        let base = if rgb_direct && self.arithmetic && self.progressive && !self.lossless {
            // JCS_RGB arithmetic progressive (#345).
            encoder::compress_arithmetic_progressive_rgb_direct(&baseline_params, self.icc_profile)?
        } else if rgb_direct && self.arithmetic && !self.lossless {
            // JCS_RGB arithmetic (#345). `jcarith.c` codes coefficients and
            // never looks at the colorspace.
            encoder::compress_arithmetic_rgb_direct(&baseline_params, self.icc_profile)?
        } else if rgb_direct && self.progressive && !self.lossless {
            // JCS_RGB progressive (#345). Progressive coding is
            // colorspace-agnostic in C — the scan script comes from the
            // component count, not the colorspace — so `colorspace(Rgb)` and
            // `progressive` compose rather than one silently winning.
            encoder::compress_progressive_rgb_direct(&baseline_params, self.icc_profile)?
        } else if rgb_direct && !self.lossless {
            // Ahead of the remaining mode switches, as the early return it
            // replaces was. Lossless is excluded because the lossless arms
            // below already encode RGB as JCS_RGB — no colour conversion,
            // Adobe APP14, 'R','G','B' component IDs — so routing it here
            // would have replaced a lossless stream with a baseline one.
            encoder::compress_rgb_direct_with_params(&baseline_params, self.icc_profile)?
        } else if use_custom_sampling {
            let factors: &Vec<(u8, u8)> = self.custom_sampling_factors.as_ref().unwrap();
            encoder::compress_custom_sampling(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                factors,
            )?
        } else if self.lossless && self.arithmetic {
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
                restart_interval,
            )?
        } else if self.arithmetic && self.progressive {
            // Custom quantization tables reach these paths too; they used to be
            // discarded because only the baseline arms could carry them (#322).
            encoder::compress_arithmetic_progressive(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                effective_subsampling,
                self.dct_method,
                restart_interval,
                restart_in_rows,
                progressive_quant_tables.as_ref(),
            )?
        } else if self.arithmetic {
            encoder::compress_arithmetic(
                effective_pixels,
                self.width,
                self.height,
                effective_format,
                quality,
                effective_subsampling,
                self.dct_method,
                restart_interval,
                progressive_quant_tables.as_ref(),
            )?
        } else if self.progressive {
            if let Some(ref script) = self.scan_script {
                encoder::compress_progressive_custom_with_restart(
                    effective_pixels,
                    self.width,
                    self.height,
                    effective_format,
                    quality,
                    effective_subsampling,
                    script,
                    self.dct_method,
                    restart_interval,
                    restart_in_rows,
                    progressive_quant_tables.as_ref(),
                )?
            } else {
                encoder::compress_progressive_with_restart(
                    effective_pixels,
                    self.width,
                    self.height,
                    effective_format,
                    quality,
                    effective_subsampling,
                    self.dct_method,
                    restart_interval,
                    restart_in_rows,
                    progressive_quant_tables.as_ref(),
                )?
            }
        } else {
            encoder::compress_with_params(&baseline_params)?
        };

        // RGB-direct writes the ICC profile itself, right after the Adobe
        // marker, to keep cjpeg's marker order; injecting it again here would
        // emit two copies.
        let icc_to_inject: Option<&[u8]> = if rgb_direct { None } else { self.icc_profile };
        let with_meta = if icc_to_inject.is_some()
            || self.exif_data.is_some()
            || self.xmp_data.is_some()
            || self.iptc_data.is_some()
        {
            encoder::inject_metadata_full(
                &base,
                icc_to_inject,
                self.exif_data,
                self.xmp_data,
                self.iptc_data,
            )?
        } else {
            base
        };

        let with_comment: Vec<u8> = if let Some(text) = self.comment {
            encoder::inject_comment(&with_meta, text)
        } else {
            with_meta
        };

        let with_saved: Vec<u8> = if self.saved_markers.is_empty() {
            with_comment
        } else {
            encoder::inject_saved_markers(&with_comment, &self.saved_markers)
        };

        // Apply JFIF density override if configured
        let with_density: Vec<u8> = if let Some((unit, x, y)) = self.density {
            Self::patch_jfif_density(with_saved, unit, x, y)
        } else {
            with_saved
        };

        // Apply JFIF version override if configured
        let with_jfif: Vec<u8> = if let Some((major, minor)) = self.jfif_version {
            Self::patch_jfif_version(with_density, major, minor)
        } else {
            with_density
        };

        // Handle Adobe APP14 marker toggle
        let with_adobe: Vec<u8> = match self.write_adobe_marker {
            Some(true) => {
                if Self::find_adobe_marker(&with_jfif).is_none() {
                    let transform: u8 = if effective_format == PixelFormat::Cmyk {
                        0
                    } else {
                        1
                    };
                    Self::inject_adobe_marker(with_jfif, transform)
                } else {
                    with_jfif
                }
            }
            Some(false) => Self::strip_adobe_marker(with_jfif),
            None => with_jfif,
        };

        let with_tables = if self.suppress_tables {
            crate::api::abbreviated::strip_table_markers(with_adobe)
        } else {
            with_adobe
        };

        Ok(with_tables)
    }

    /// Expose effective quant tables for abbreviated stream generation.
    pub(crate) fn effective_quant_tables_for_abbrev(&self) -> [Option<[u16; 64]>; 4] {
        let quality = self.effective_quality();
        self.build_quant_tables(quality)
    }

    /// Expose custom Huffman DC tables for abbreviated stream generation.
    pub(crate) fn custom_huffman_dc_tables(&self) -> &[Option<HuffmanTableDef>; 4] {
        &self.custom_huffman_dc
    }

    /// Expose custom Huffman AC tables for abbreviated stream generation.
    pub(crate) fn custom_huffman_ac_tables(&self) -> &[Option<HuffmanTableDef>; 4] {
        &self.custom_huffman_ac
    }

    /// Whether the encoder uses arithmetic coding.
    pub(crate) fn is_arithmetic(&self) -> bool {
        self.arithmetic
    }

    fn build_quant_tables(&self, quality: u8) -> [Option<[u16; 64]>; 4] {
        let mut result = self.custom_quant_tables;
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

#[cfg(test)]
mod luminance_tests {
    use super::*;

    /// BT.601 reference, written independently of the implementation.
    fn bt601(red: u8, green: u8, blue: u8) -> u8 {
        ((19595 * red as u32 + 38470 * green as u32 + 7471 * blue as u32 + 32768) >> 16) as u8
    }

    fn interleave(format: PixelFormat, red: u8, green: u8, blue: u8) -> Vec<u8> {
        const PAD: u8 = 0xA5;
        match format {
            PixelFormat::Grayscale => vec![bt601(red, green, blue)],
            PixelFormat::Rgb => vec![red, green, blue],
            PixelFormat::Bgr => vec![blue, green, red],
            PixelFormat::Rgba | PixelFormat::Rgbx => vec![red, green, blue, PAD],
            PixelFormat::Bgra | PixelFormat::Bgrx => vec![blue, green, red, PAD],
            PixelFormat::Xrgb | PixelFormat::Argb => vec![PAD, red, green, blue],
            PixelFormat::Xbgr | PixelFormat::Abgr => vec![PAD, blue, green, red],
            other => panic!("unsupported in this test: {other:?}"),
        }
    }

    /// Covers `extract_luminance` directly rather than through `Encoder`.
    ///
    /// `Encoder` never reaches the `Rgb` or `Grayscale` arms — it routes plain
    /// `Rgb` through the SIMD `rgb_to_ycbcr_row` and skips the call entirely
    /// for `Grayscale` — so an integration test cannot exercise them, and
    /// mutation testing showed 16 mutants surviving in the `Rgb` arm alone
    /// (issue #325). A unit test is the only way to reach them.
    #[test]
    fn extract_luminance_matches_bt601_for_every_format() {
        let formats: &[PixelFormat] = &[
            PixelFormat::Grayscale,
            PixelFormat::Rgb,
            PixelFormat::Bgr,
            PixelFormat::Rgba,
            PixelFormat::Bgra,
            PixelFormat::Rgbx,
            PixelFormat::Bgrx,
            PixelFormat::Xrgb,
            PixelFormat::Xbgr,
            PixelFormat::Argb,
            PixelFormat::Abgr,
        ];
        // Primaries isolate each weight; the asymmetric mixes catch an R/B swap.
        let colours: &[(u8, u8, u8)] = &[
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (240, 12, 33),
            (17, 200, 93),
            (255, 255, 255),
            (0, 0, 0),
        ];

        for &format in formats {
            for &(red, green, blue) in colours {
                let pixel: Vec<u8> = interleave(format, red, green, blue);
                let got: Vec<u8> = Encoder::extract_luminance(&pixel, 1, format);
                assert_eq!(
                    got,
                    vec![bt601(red, green, blue)],
                    "{format:?} rgb({red},{green},{blue})"
                );
            }
        }
    }

    /// `restart_rows(n)` converts to MCU blocks as `n * MCUs_per_row`, so the
    /// MCU width per subsampling has to be right. Mutation testing found the
    /// `==` in that conversion invertible undetected (issue #325), and #322
    /// showed restart handling is fragile enough to be worth pinning.
    #[test]
    fn restart_rows_convert_to_blocks_using_the_mcu_width() {
        let pixels: Vec<u8> = vec![0u8; 64 * 64 * 3];
        // 64px wide: 8 MCUs across at mcu_w=8, 4 at 16, 2 at 32.
        let cases: &[(Subsampling, u16)] = &[
            (Subsampling::S444, 8),
            (Subsampling::S440, 8),
            (Subsampling::S441, 8),
            (Subsampling::S422, 4),
            (Subsampling::S420, 4),
            (Subsampling::S24, 4),
            (Subsampling::S411, 2),
            (Subsampling::S410, 2),
        ];
        for &(subsampling, mcus_across) in cases {
            let encoder = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
                .subsampling(subsampling)
                .restart_rows(3);
            assert_eq!(
                encoder.compute_restart_interval(subsampling),
                3 * mcus_across,
                "{subsampling:?}: restart_rows(3) should be 3 x {mcus_across} MCU blocks"
            );
        }
    }

    /// `restart_blocks` is already in MCU units and must pass through unchanged.
    #[test]
    fn restart_blocks_passes_through_unchanged() {
        let pixels: Vec<u8> = vec![0u8; 64 * 64 * 3];
        for n in [1u16, 7, 1000] {
            let encoder = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb).restart_blocks(n);
            assert_eq!(encoder.compute_restart_interval(Subsampling::S420), n);
        }
        // Unset means no restarts at all.
        let encoder = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb);
        assert_eq!(encoder.compute_restart_interval(Subsampling::S420), 0);
    }

    /// CMYK and RGB565 are not colour-converted here; both fill a neutral grey.
    #[test]
    fn extract_luminance_fills_neutral_for_unconverted_formats() {
        for format in [PixelFormat::Cmyk, PixelFormat::Rgb565] {
            let got: Vec<u8> = Encoder::extract_luminance(&[0u8; 16], 4, format);
            assert_eq!(got, vec![128u8; 4], "{format:?}");
        }
    }
}
