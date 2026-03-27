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
    /// Smoothing factor for pre-encode noise reduction (0-100). 0 = no smoothing.
    smoothing_factor: u8,
    /// When true, use a triangle/tent filter for chroma downsampling. Default is true.
    fancy_downsampling: bool,
    /// Custom JFIF version override. When `Some`, replaces the default 1.01.
    jfif_version: Option<(u8, u8)>,
    /// Adobe APP14 marker control. `None` = auto. `Some(true)` = always. `Some(false)` = never.
    write_adobe_marker: Option<bool>,
    /// Custom per-component sampling factors as (h, v) pairs.
    /// When set, overrides the `subsampling` enum with explicit factors.
    /// The first component (Y) defines the max sampling factor; subsequent
    /// components (Cb, Cr) can use any factor from 1 to max_h/max_v.
    custom_sampling_factors: Option<Vec<(u8, u8)>>,
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
            smoothing_factor: 0,
            fancy_downsampling: true,
            jfif_version: None,
            write_adobe_marker: None,
            custom_sampling_factors: None,
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
                let mcus_x = self.width.div_ceil(mcu_w) as u16;
                n.saturating_mul(mcus_x)
            }
        }
    }

    fn _effective_quant_tables(&self) -> [Option<[u16; 64]>; 4] {
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
        }
        result
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

    fn apply_smoothing(
        pixels: &[u8],
        width: usize,
        height: usize,
        bpp: usize,
        strength: u8,
    ) -> Vec<u8> {
        if strength == 0 || width <= 2 || height <= 2 {
            return pixels.to_vec();
        }
        let row_stride: usize = width * bpp;
        let mut output: Vec<u8> = pixels.to_vec();
        let neighbor_weight: u32 = (strength as u32 * 3) / 100 + 1;
        let center_weight: u32 = 256 - 8 * neighbor_weight;
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                for c in 0..bpp {
                    let idx: usize = y * row_stride + x * bpp + c;
                    let center: u32 = pixels[idx] as u32;
                    let top: u32 = pixels[idx - row_stride] as u32;
                    let bottom: u32 = pixels[idx + row_stride] as u32;
                    let left: u32 = pixels[idx - bpp] as u32;
                    let right: u32 = pixels[idx + bpp] as u32;
                    let tl: u32 = pixels[idx - row_stride - bpp] as u32;
                    let tr: u32 = pixels[idx - row_stride + bpp] as u32;
                    let bl: u32 = pixels[idx + row_stride - bpp] as u32;
                    let br: u32 = pixels[idx + row_stride + bpp] as u32;
                    let sum: u32 = center * center_weight
                        + (top + bottom + left + right + tl + tr + bl + br) * neighbor_weight;
                    output[idx] = (sum >> 8).min(255) as u8;
                }
            }
        }
        output
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
        let restart_interval = self.compute_restart_interval();

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

        // Apply smoothing filter if requested
        let smoothed_buf: Vec<u8>;
        let after_smooth: &[u8] = if self.smoothing_factor > 0 {
            smoothed_buf = Self::apply_smoothing(
                input_pixels,
                self.width,
                self.height,
                self.pixel_format.bytes_per_pixel(),
                self.smoothing_factor,
            );
            &smoothed_buf
        } else {
            input_pixels
        };

        // Apply fancy downsampling pre-filter if enabled and subsampling is active
        let fancy_buf: Vec<u8>;
        let after_fancy: &[u8] = if self.fancy_downsampling
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
            gray_buf =
                Self::extract_luminance(after_fancy, self.width * self.height, self.pixel_format);
            effective_pixels = &gray_buf[..];
            effective_format = PixelFormat::Grayscale;
        } else {
            effective_pixels = after_fancy;
            effective_format = self.pixel_format;
        }

        let quality: u8 = self.effective_quality();
        let needs_custom_quant: bool = self.force_baseline
            || self.linear_scale_factor.is_some()
            || self.has_custom_quant_tables();

        let base = if let Some(ref factors) = self.custom_sampling_factors {
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

        let with_saved: Vec<u8> = if self.saved_markers.is_empty() {
            with_comment
        } else {
            encoder::inject_saved_markers(&with_comment, &self.saved_markers)
        };

        // Apply JFIF version override if configured
        let with_jfif: Vec<u8> = if let Some((major, minor)) = self.jfif_version {
            Self::patch_jfif_version(with_saved, major, minor)
        } else {
            with_saved
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

        Ok(with_adobe)
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
