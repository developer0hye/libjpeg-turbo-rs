/// Full JPEG encoder pipeline.
///
/// Orchestrates color conversion, forward DCT, quantization, Huffman encoding,
/// and marker writing to produce a valid baseline JPEG file.
use crate::api::encoder::HuffmanTableDef;
use crate::common::error::{JpegError, Result};
use crate::common::types::{DctMethod, PixelFormat, SavedMarker, ScanScript, Subsampling};
use crate::encode::color;
use crate::encode::huffman_encode::{
    build_huff_table, local_drain_bits, local_put_bits, BitWriter, HuffTable, HuffmanEncoder,
};
use crate::encode::marker_writer;
use crate::encode::progressive::ProgressiveScan;
use crate::encode::tables;
use crate::simd::QuantDivisors;

/// Resolves the luma/chroma quantization tables for a component pair.
///
/// A custom slot wins; otherwise Annex K scaled by quality. Slot 0 is luma,
/// slot 1 chroma — the convention every entry point here shares.
fn resolve_quant_tables(
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
    quality: u8,
) -> ([u16; 64], [u16; 64]) {
    let luma: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let chroma: [u16; 64] = match custom_quant.and_then(|tables| tables[1]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality),
    };
    (luma, chroma)
}

/// Whether the fused SIMD extract+FDCT+quantize kernels may be used.
///
/// Those kernels hardcode the **islow** transform. The `ifast` and `float`
/// methods come with divisor tables scaled for their own transforms, so
/// feeding islow coefficients to them mis-scales every output by the AA&N
/// factor — which is how `-dct fast` ended up both lower quality and larger
/// than C's (#330). Callers that hold a `fdct_quantize_fn` must therefore ask
/// this before taking a SIMD shortcut.
fn may_use_islow_simd_kernel(
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> bool {
    let is_ifast: bool = std::ptr::eq(
        fdct_quantize_fn as *const (),
        crate::simd::scalar::scalar_fdct_ifast_quantize as *const (),
    );
    let is_float: bool = std::ptr::eq(
        fdct_quantize_fn as *const (),
        crate::simd::scalar::scalar_fdct_float_quantize as *const (),
    );
    !is_ifast && !is_float
}

/// Color conversion function: (pixels, y, cb, cr, width).
type ColorConvertRowFn = fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize);

/// Select the best available RGBA→YCbCr row conversion function.
fn select_rgba_to_ycbcr_fn() -> ColorConvertRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return crate::simd::aarch64::color_encode::neon_rgba_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    {
        return crate::simd::wasm32::color_encode::wasm_rgba_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx2") {
            return crate::simd::x86_64::avx2_color_encode::avx2_rgba_to_ycbcr_row;
        }
    }
    #[allow(unreachable_code)]
    color::rgba_to_ycbcr_row
}

/// Select the best available BGR→YCbCr row conversion function.
fn select_bgr_to_ycbcr_fn() -> ColorConvertRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return crate::simd::aarch64::color_encode::neon_bgr_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    {
        return crate::simd::wasm32::color_encode::wasm_bgr_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx2") {
            return crate::simd::x86_64::avx2_color_encode::avx2_bgr_to_ycbcr_row;
        }
    }
    #[allow(unreachable_code)]
    color::bgr_to_ycbcr_row_scalar
}

/// Select the best available BGRA→YCbCr row conversion function.
fn select_bgra_to_ycbcr_fn() -> ColorConvertRowFn {
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        return crate::simd::aarch64::color_encode::neon_bgra_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "wasm32", feature = "simd"))]
    {
        return crate::simd::wasm32::color_encode::wasm_bgra_to_ycbcr_row;
    }
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    {
        if is_x86_feature_detected!("avx2") {
            return crate::simd::x86_64::avx2_color_encode::avx2_bgra_to_ycbcr_row;
        }
    }
    #[allow(unreachable_code)]
    color::bgra_to_ycbcr_row_scalar
}

/// The full option set for a single-pass baseline encode.
///
/// This exists so that the baseline `compress_*` entry points are thin shims
/// over one implementation instead of near-copies of it. Historically each
/// variant carried only the options it named — `compress_with_restart` could
/// not express custom tables, `compress_custom_quant` could not express a
/// restart interval — so a fix or an optimization landed in whichever copy the
/// author happened to be editing. That produced real divergence: the
/// dummy-block contract was implemented in one branch and not the others
/// (#316), and CMYK silently discarded every option a variant could not pass
/// on (#313).
///
/// Options that are `None` / zero mean "not requested" and select the JPEG
/// default, so adding a field does not change any existing caller's output.
pub struct CompressParams<'a> {
    /// Raw pixel data in the format given by `pixel_format`.
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub pixel_format: PixelFormat,
    /// Quality factor 1-100. Ignored for components whose quantization table
    /// is supplied through `custom_quant`.
    pub quality: u8,
    pub subsampling: Subsampling,
    pub dct_method: DctMethod,
    /// MCUs between RST markers; 0 emits no DRI marker and no restarts.
    pub restart_interval: u16,
    /// Per-slot quantization tables. Slot 0 overrides luma, slot 1 chroma;
    /// unset slots fall back to the quality-scaled Annex K tables.
    pub custom_quant: Option<&'a [Option<[u16; 64]>; 4]>,
    /// Per-slot DC Huffman tables, same slot convention as `custom_quant`.
    pub custom_dc_huffman: Option<&'a [Option<HuffmanTableDef>; 4]>,
    /// Per-slot AC Huffman tables, same slot convention as `custom_quant`.
    pub custom_ac_huffman: Option<&'a [Option<HuffmanTableDef>; 4]>,
    /// Two-pass optimized Huffman coding. Computes tables from the actual
    /// symbol statistics, so any `custom_*_huffman` tables are superseded —
    /// matching libjpeg's `optimize_coding` semantics.
    pub optimize_huffman: bool,
    /// Input smoothing strength 0-100, as C's `smoothing_factor`.
    pub smoothing_factor: u8,
}

impl<'a> CompressParams<'a> {
    /// Construct with every optional knob at its JPEG default.
    pub fn new(
        pixels: &'a [u8],
        width: usize,
        height: usize,
        pixel_format: PixelFormat,
        quality: u8,
        subsampling: Subsampling,
    ) -> Self {
        Self {
            pixels,
            width,
            height,
            pixel_format,
            quality,
            subsampling,
            dct_method: DctMethod::IsLow,
            restart_interval: 0,
            custom_quant: None,
            custom_dc_huffman: None,
            custom_ac_huffman: None,
            optimize_huffman: false,
            smoothing_factor: 0,
        }
    }

    pub fn dct_method(mut self, dct_method: DctMethod) -> Self {
        self.dct_method = dct_method;
        self
    }

    pub fn restart_interval(mut self, restart_interval: u16) -> Self {
        self.restart_interval = restart_interval;
        self
    }

    pub fn custom_quant(mut self, custom_quant: &'a [Option<[u16; 64]>; 4]) -> Self {
        self.custom_quant = Some(custom_quant);
        self
    }

    pub fn custom_huffman(
        mut self,
        dc: &'a [Option<HuffmanTableDef>; 4],
        ac: &'a [Option<HuffmanTableDef>; 4],
    ) -> Self {
        self.custom_dc_huffman = Some(dc);
        self.custom_ac_huffman = Some(ac);
        self
    }

    pub fn optimize_huffman(mut self, optimize: bool) -> Self {
        self.optimize_huffman = optimize;
        self
    }

    pub fn smoothing_factor(mut self, factor: u8) -> Self {
        self.smoothing_factor = factor.min(100);
        self
    }
}

/// Resolved Huffman tables: the encoding tables plus the exact bits/values that
/// must be written into the DHT markers, so the two can never disagree.
struct ResolvedHuffman {
    dc_luma_bits: [u8; 17],
    dc_luma_values: Vec<u8>,
    ac_luma_bits: [u8; 17],
    ac_luma_values: Vec<u8>,
    dc_chroma_bits: [u8; 17],
    dc_chroma_values: Vec<u8>,
    ac_chroma_bits: [u8; 17],
    ac_chroma_values: Vec<u8>,
    dc_luma: HuffTable,
    ac_luma: HuffTable,
    dc_chroma: HuffTable,
    ac_chroma: HuffTable,
}

impl ResolvedHuffman {
    /// Custom slot 0 overrides luma, slot 1 chroma; unset slots use Annex K.
    fn resolve(
        custom_dc: Option<&[Option<HuffmanTableDef>; 4]>,
        custom_ac: Option<&[Option<HuffmanTableDef>; 4]>,
    ) -> Self {
        fn pick(
            custom: Option<&[Option<HuffmanTableDef>; 4]>,
            slot: usize,
            default_bits: &[u8; 17],
            default_values: &[u8],
        ) -> ([u8; 17], Vec<u8>) {
            match custom.and_then(|tables| tables[slot].as_ref()) {
                Some(table) => (table.bits, table.values.clone()),
                None => (*default_bits, default_values.to_vec()),
            }
        }

        let (dc_luma_bits, dc_luma_values) = pick(
            custom_dc,
            0,
            &tables::DC_LUMINANCE_BITS,
            &tables::DC_LUMINANCE_VALUES,
        );
        let (ac_luma_bits, ac_luma_values) = pick(
            custom_ac,
            0,
            &tables::AC_LUMINANCE_BITS,
            &tables::AC_LUMINANCE_VALUES,
        );
        let (dc_chroma_bits, dc_chroma_values) = pick(
            custom_dc,
            1,
            &tables::DC_CHROMINANCE_BITS,
            &tables::DC_CHROMINANCE_VALUES,
        );
        let (ac_chroma_bits, ac_chroma_values) = pick(
            custom_ac,
            1,
            &tables::AC_CHROMINANCE_BITS,
            &tables::AC_CHROMINANCE_VALUES,
        );

        Self {
            dc_luma: build_huff_table(&dc_luma_bits, &dc_luma_values),
            ac_luma: build_huff_table(&ac_luma_bits, &ac_luma_values),
            dc_chroma: build_huff_table(&dc_chroma_bits, &dc_chroma_values),
            ac_chroma: build_huff_table(&ac_chroma_bits, &ac_chroma_values),
            dc_luma_bits,
            dc_luma_values,
            ac_luma_bits,
            ac_luma_values,
            dc_chroma_bits,
            dc_chroma_values,
            ac_chroma_bits,
            ac_chroma_values,
        }
    }
}

/// Single-pass baseline encode — the one implementation behind `compress`,
/// `compress_with_restart`, `compress_custom_quant` and
/// `compress_custom_huffman`.
///
/// Two-pass optimized-Huffman encoding (`compress_optimized`) is a genuinely
/// different algorithm and still lives separately.
pub fn compress_with_params(params: &CompressParams<'_>) -> Result<Vec<u8>> {
    let CompressParams {
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        dct_method,
        restart_interval,
        custom_quant,
        custom_dc_huffman,
        custom_ac_huffman,
        optimize_huffman,
        smoothing_factor,
    } = *params;

    // Two-pass optimized Huffman, and smoothing, both need full-plane
    // buffering, so they live in the other implementation. Dispatching here
    // rather than in every caller is what stops the two from masking each
    // other's options (#322).
    if optimize_huffman || smoothing_factor > 0 {
        return compress_optimized_with_params(params);
    }

    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    // CMYK: 4-component path, no color conversion.
    //
    // NOTE(#313): `compress_cmyk` cannot express a restart interval, custom
    // tables or a DCT method, so all of those are silently dropped here. That
    // is pre-existing behaviour, deliberately preserved by this refactor —
    // changing it moves output bytes and needs its own C cross-validation.
    // Centralizing the drop in one place is what makes #313 a small fix.
    if pixel_format == PixelFormat::Cmyk {
        return compress_cmyk(params);
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Quantization tables: a custom slot wins, otherwise scale Annex K by quality.
    let luma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let chroma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[1]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality),
    };

    // Divisor tables scale quant values for the chosen FDCT method.
    // IsLow: multiply by 8 (islow leaves factor-of-8 in output).
    // IsFast: multiply by AA&N scale factors (ifast_raw leaves AA&N-scaled output).
    let luma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&luma_quant)
    } else {
        scale_quant_for_fdct(&luma_quant)
    };
    let chroma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&chroma_quant)
    } else {
        scale_quant_for_fdct(&chroma_quant)
    };

    // Huffman tables: custom slots win, otherwise Annex K. Destructured so the
    // encoding tables keep the names the MCU loops already use, and so the
    // bits/values that go into the DHT markers travel with them.
    let ResolvedHuffman {
        dc_luma: dc_luma_table,
        ac_luma: ac_luma_table,
        dc_chroma: dc_chroma_table,
        ac_chroma: ac_chroma_table,
        dc_luma_bits,
        dc_luma_values,
        ac_luma_bits,
        ac_luma_values,
        dc_chroma_bits,
        dc_chroma_values,
        ac_chroma_bits,
        ac_chroma_values,
    } = ResolvedHuffman::resolve(custom_dc_huffman, custom_ac_huffman);

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Determine MCU dimensions based on subsampling
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);

    // Dispatch FDCT+quantize based on DCT method.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };

    // Entropy encode all MCUs
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    // Restart state, shared by both encode strategies below. `mcu_count` runs
    // across MCU rows, so a restart can fall anywhere inside a row.
    let restart_mcu_interval: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut restart_marker_index: u8 = 0;

    /// Emit an RST marker and reset the DC predictors when the MCU index lands
    /// on an interval boundary. Never fires before the first MCU.
    macro_rules! maybe_emit_restart {
        () => {
            if restart_mcu_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(restart_mcu_interval)
            {
                bit_writer.flush_restart();
                bit_writer.write_restart_marker(restart_marker_index);
                restart_marker_index = restart_marker_index.wrapping_add(1);
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
            }
        };
    }

    // Single-pass fused approach: convert MCU rows on-the-fly instead
    // of pre-allocating full-size planes. Keeps data in L1/L2 cache between
    // color conversion and encoding.
    // Select format-specific color conversion function + BPP for the fast path.
    let fused_color_fn: Option<(ColorConvertRowFn, usize)> = if is_grayscale {
        None
    } else {
        match pixel_format {
            PixelFormat::Rgb => Some((enc_simd.rgb_to_ycbcr_row, 3)),
            PixelFormat::Rgba => Some((select_rgba_to_ycbcr_fn(), 4)),
            PixelFormat::Bgr => Some((select_bgr_to_ycbcr_fn(), 3)),
            PixelFormat::Bgra => Some((select_bgra_to_ycbcr_fn(), 4)),
            _ => None,
        }
    };
    if let Some((color_convert_fn, bpp)) = fused_color_fn {
        // Pad buffer width to MCU-aligned, matching C libjpeg-turbo's behavior.
        // C allocates coefficient buffers padded to MCU boundaries and pads input
        // with expand_right_edge up to width_in_blocks * DCTSIZE per component.
        // Blocks beyond width_in_blocks are left as zeros in C (never FDCT'd).
        let padded_w: usize = mcus_x * mcu_w;
        let padded_h: usize = mcu_h;
        let row_buf_size: usize = padded_w * padded_h;
        let mut y_buf: Vec<u8> = vec![0u8; row_buf_size];
        let mut cb_buf: Vec<u8> = vec![0u8; row_buf_size];
        let mut cr_buf: Vec<u8> = vec![0u8; row_buf_size];

        // For 420 on x86_64: pre-allocate half-resolution chroma buffers.
        // After color conversion, we downsample full-res Cb/Cr into these compact
        // buffers so that FDCT reads from stride=half_w instead of stride=padded_w.
        //
        // Both the downsample below and the encode fast path further down call
        // `#[target_feature(enable = "avx2")]` helpers, so a single capability
        // flag gates every step. Deriving the fast path's guard from buffer
        // emptiness instead would let a non-AVX2 x86_64 CPU reach AVX2
        // intrinsics with never-downsampled (all-zero) chroma — issue #315.
        #[cfg(target_arch = "x86_64")]
        // Also gated on the DCT method: this path calls the islow AVX2 kernels
        // directly, while ifast/float carry divisors scaled for their own
        // transforms (#330).
        let use_avx2_420: bool = subsampling == Subsampling::S420
            && is_x86_feature_detected!("avx2")
            && may_use_islow_simd_kernel(fdct_quantize_fn);
        #[cfg(target_arch = "x86_64")]
        let half_w: usize = padded_w / 2;
        #[cfg(target_arch = "x86_64")]
        let half_h: usize = padded_h / 2;
        #[cfg(target_arch = "x86_64")]
        let mut cb_half: Vec<u8> = if use_avx2_420 {
            vec![0u8; half_w * half_h]
        } else {
            Vec::new()
        };
        #[cfg(target_arch = "x86_64")]
        let mut cr_half: Vec<u8> = if use_avx2_420 {
            vec![0u8; half_w * half_h]
        } else {
            Vec::new()
        };

        for mcu_row in 0..mcus_y {
            let y0: usize = mcu_row * mcu_h;
            let rows_available: usize = (height - y0).min(mcu_h);

            // Convert this MCU row's pixel data to YCbCr
            for row in 0..rows_available {
                let src_row: usize = y0 + row;
                let src_offset: usize = src_row * width * bpp;
                let dst_offset: usize = row * padded_w;
                color_convert_fn(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_buf[dst_offset..dst_offset + width],
                    &mut cb_buf[dst_offset..dst_offset + width],
                    &mut cr_buf[dst_offset..dst_offset + width],
                    width,
                );
                // Pad right edge by replicating last pixel to MCU-aligned width,
                // matching C libjpeg-turbo's expand_right_edge behavior.
                if width < padded_w {
                    let last_y: u8 = y_buf[dst_offset + width - 1];
                    let last_cb: u8 = cb_buf[dst_offset + width - 1];
                    let last_cr: u8 = cr_buf[dst_offset + width - 1];
                    for x in width..padded_w {
                        y_buf[dst_offset + x] = last_y;
                        cb_buf[dst_offset + x] = last_cb;
                        cr_buf[dst_offset + x] = last_cr;
                    }
                }
            }
            // Pad remaining rows to match C libjpeg-turbo's behavior:
            // Y component: replicate last real row (jccoefct.c expand_bottom_edge)
            // Cb/Cr components: replicate last complete row group so that chroma
            // downsampling produces the same result as C's two-phase approach
            // (jcprepct.c pads to row group, downsamples, then replicates the
            // downsampled output in jccoefct.c).
            let last_row_offset: usize = (rows_available - 1) * padded_w;

            // Y: simple last-row replication (matches C's luma behavior)
            for row in rows_available..padded_h {
                let dst_offset: usize = row * padded_w;
                y_buf.copy_within(last_row_offset..last_row_offset + padded_w, dst_offset);
            }

            // Cb/Cr: row-group replication for correct chroma downsampling
            let max_v: usize = subsampling.sampling_factors().1 as usize;
            let row_group_end: usize = rows_available.div_ceil(max_v).min(padded_h / max_v) * max_v;

            // Phase 1: complete the last row group (replicate last real row)
            for row in rows_available..row_group_end.min(padded_h) {
                let dst_offset: usize = row * padded_w;
                cb_buf.copy_within(last_row_offset..last_row_offset + padded_w, dst_offset);
                cr_buf.copy_within(last_row_offset..last_row_offset + padded_w, dst_offset);
            }

            // Phase 2: replicate the last complete row group
            if row_group_end < padded_h {
                let group_start: usize = row_group_end - max_v;
                for row in row_group_end..padded_h {
                    let src_row: usize = group_start + (row - row_group_end) % max_v;
                    let dst_offset: usize = row * padded_w;
                    let src_offset: usize = src_row * padded_w;
                    cb_buf.copy_within(src_offset..src_offset + padded_w, dst_offset);
                    cr_buf.copy_within(src_offset..src_offset + padded_w, dst_offset);
                }
            }

            // For 420: downsample full-res Cb/Cr to compact half-res buffers.
            // This allows FDCT to read from stride=half_w instead of fused
            // downsample+FDCT from stride=padded_w, improving cache locality.
            #[cfg(target_arch = "x86_64")]
            if use_avx2_420 {
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v2_plane(
                        &cb_buf,
                        padded_w,
                        padded_h,
                        &mut cb_half,
                        half_w,
                    );
                    crate::simd::x86_64::avx2_downsample_h2v2_plane(
                        &cr_buf,
                        padded_w,
                        padded_h,
                        &mut cr_half,
                        half_w,
                    );
                }
            }

            // Encode all MCUs in this row.
            // For the last MCU column, C libjpeg-turbo creates "dummy" blocks
            // for components that extend beyond width_in_blocks: all AC=0, DC
            // copied from the previous block (jccoefct.c lines 184-191).
            let (h_samp, v_samp) = subsampling.sampling_factors();
            let y_width_in_blocks: usize = width.div_ceil(8);
            let y_height_in_blocks: usize = height.div_ceil(8);
            let y_mcu_width: usize = h_samp as usize;
            let y_mcu_height: usize = v_samp as usize;
            let y_last_col_width: usize = {
                let rem: usize = y_width_in_blocks % y_mcu_width;
                if rem == 0 {
                    y_mcu_width
                } else {
                    rem
                }
            };
            let y_last_row_height: usize = {
                let rem: usize = y_height_in_blocks % y_mcu_height;
                if rem == 0 {
                    y_mcu_height
                } else {
                    rem
                }
            };
            let is_last_mcu_row: bool = mcu_row == mcus_y - 1;
            let eff_row_height: usize = if is_last_mcu_row {
                y_last_row_height
            } else {
                y_mcu_height
            };

            // 420 fast path: row-level hoisted bit buffer + inline FDCT+Huffman.
            // One begin_block/end_block per MCU row (not per MCU), eliminating
            // ~120 ensure_capacity checks per row for 1920-wide images.
            //
            // It FDCTs every block of every MCU unconditionally, so it is only
            // valid where no dummy blocks are needed: interior MCU rows
            // (`eff_row_height == y_mcu_height`) *and* images whose last MCU
            // column is full (`y_last_col_width == y_mcu_width`). C zeroes
            // dummy blocks and copies the previous block's DC rather than
            // transforming replicated edge pixels (jccoefct.c:292-312), so
            // running this path over a partial last column produced output that
            // diverged from cjpeg for every width with `ceil(width/8)` odd —
            // issue #314. Partial geometries fall through to the generic path
            // below, which handles dummies via `encode_color_mcu_with_dummies`.
            // Where the generic loop below must pick up. The fast path covers
            // the interior columns; a partial final column falls through.
            let mut generic_start_col: usize = 0;

            #[cfg(target_arch = "x86_64")]
            // Restarts are excluded because this path hoists one bit-buffer
            // region across the whole MCU row; an RST marker mid-row would have
            // to break out of it. Restart encodes take the generic path below.
            if use_avx2_420 && restart_mcu_interval == 0 && eff_row_height == y_mcu_height {
                // Every block of every MCU is FDCT'd unconditionally here, so
                // only columns with no dummy blocks qualify. When the last MCU
                // column is partial it is excluded and handled generically
                // rather than disqualifying the whole row — that costs one
                // column instead of `mcus_x` of them (#317). C zeroes dummy
                // blocks and copies the previous DC instead of transforming
                // replicated edge pixels (jccoefct.c:292-312), which is what
                // #314 got wrong.
                let fast_cols: usize = if y_last_col_width == y_mcu_width {
                    mcus_x
                } else {
                    mcus_x - 1
                };

                if fast_cols > 0 {
                    unsafe {
                        // Reserve capacity for the columns this path will encode
                        let (mut pb, mut fb, mut buf) = bit_writer.begin_block(3072 * fast_cols);

                        for mcu_col in 0..fast_cols {
                            let x0: usize = mcu_col * mcu_w;
                            let cx0: usize = mcu_col * (mcu_w / 2);

                            // FDCT + quantize 6 blocks (4Y + Cb + Cr)
                            let mut q: [[i16; 64]; 6] = [[0i16; 64]; 6];
                            let y_ptr: *const u8 = y_buf.as_ptr().add(x0);
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr,
                                padded_w,
                                &luma_divisors,
                                &mut q[0],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr.add(8),
                                padded_w,
                                &luma_divisors,
                                &mut q[1],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr.add(8 * padded_w),
                                padded_w,
                                &luma_divisors,
                                &mut q[2],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                y_ptr.add(8 * padded_w + 8),
                                padded_w,
                                &luma_divisors,
                                &mut q[3],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                cb_half.as_ptr().add(cx0),
                                half_w,
                                &chroma_divisors,
                                &mut q[4],
                            );
                            crate::simd::x86_64::avx2_extract_fdct_quantize(
                                cr_half.as_ptr().add(cx0),
                                half_w,
                                &chroma_divisors,
                                &mut q[5],
                            );

                            // Huffman encode 6 blocks with row-hoisted state
                            for block in q.iter().take(4) {
                                HuffmanEncoder::encode_block_hoisted(
                                    &mut pb,
                                    &mut fb,
                                    &mut buf,
                                    block,
                                    &mut prev_dc_y,
                                    &dc_luma_table,
                                    &ac_luma_table,
                                );
                            }
                            HuffmanEncoder::encode_block_hoisted(
                                &mut pb,
                                &mut fb,
                                &mut buf,
                                &q[4],
                                &mut prev_dc_cb,
                                &dc_chroma_table,
                                &ac_chroma_table,
                            );
                            HuffmanEncoder::encode_block_hoisted(
                                &mut pb,
                                &mut fb,
                                &mut buf,
                                &q[5],
                                &mut prev_dc_cr,
                                &dc_chroma_table,
                                &ac_chroma_table,
                            );
                        }

                        bit_writer.end_block(pb, fb, buf);
                    }
                    // Only reachable with restarts disabled, but keep the counter
                    // meaningful for every path.
                    mcu_count += fast_cols as u32;
                }

                if fast_cols == mcus_x {
                    continue; // whole row handled; skip the generic loop
                }
                generic_start_col = fast_cols;
            }

            // Generic path for non-420, edge MCU rows, restarts, non-x86_64,
            // and the trailing partial column left by the fast path above.
            for mcu_col in generic_start_col..mcus_x {
                maybe_emit_restart!();

                let x0: usize = mcu_col * mcu_w;
                let is_last_mcu_col: bool = mcu_col == mcus_x - 1;
                let eff_col_width: usize = if is_last_mcu_col {
                    y_last_col_width
                } else {
                    y_mcu_width
                };

                let need_dummies: bool =
                    eff_col_width < y_mcu_width || eff_row_height < y_mcu_height;

                if need_dummies {
                    encode_color_mcu_with_dummies(
                        &y_buf,
                        &cb_buf,
                        &cr_buf,
                        padded_w,
                        padded_h,
                        x0,
                        0,
                        subsampling,
                        &luma_divisors,
                        &chroma_divisors,
                        &dc_luma_table,
                        &ac_luma_table,
                        &dc_chroma_table,
                        &ac_chroma_table,
                        &mut bit_writer,
                        &mut prev_dc_y,
                        &mut prev_dc_cb,
                        &mut prev_dc_cr,
                        fdct_quantize_fn,
                        eff_col_width,
                        eff_row_height,
                    );
                } else {
                    encode_color_mcu(
                        &y_buf,
                        &cb_buf,
                        &cr_buf,
                        padded_w,
                        padded_h,
                        x0,
                        0,
                        subsampling,
                        &luma_divisors,
                        &chroma_divisors,
                        &dc_luma_table,
                        &ac_luma_table,
                        &dc_chroma_table,
                        &ac_chroma_table,
                        &mut bit_writer,
                        &mut prev_dc_y,
                        &mut prev_dc_cb,
                        &mut prev_dc_cr,
                        fdct_quantize_fn,
                    );
                }

                mcu_count += 1;
            }
        }
    } else {
        // Fallback: full-plane color conversion for non-RGB formats and grayscale
        let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
            pixels,
            width,
            height,
            pixel_format,
            enc_simd.rgb_to_ycbcr_row,
        )?;

        // Pad all planes to MCU-aligned dimensions so all blocks (including edge
        // blocks) go through the NEON fused FDCT+quantize path instead of the
        // scalar fallback.  This matches C libjpeg-turbo's expand_right_edge
        // behavior and ensures byte-identical output.
        let padded_w: usize = mcus_x * mcu_w;
        let padded_h: usize = mcus_y * mcu_h;

        fn pad_plane(
            plane: &[u8],
            src_w: usize,
            src_h: usize,
            dst_w: usize,
            dst_h: usize,
        ) -> Vec<u8> {
            if src_w == dst_w && src_h == dst_h {
                return plane.to_vec();
            }
            let mut padded: Vec<u8> = vec![0u8; dst_w * dst_h];
            for row in 0..src_h {
                let src_start: usize = row * src_w;
                let dst_start: usize = row * dst_w;
                padded[dst_start..dst_start + src_w]
                    .copy_from_slice(&plane[src_start..src_start + src_w]);
                if src_w < dst_w {
                    let last_val: u8 = plane[src_start + src_w - 1];
                    for x in src_w..dst_w {
                        padded[dst_start + x] = last_val;
                    }
                }
            }
            if src_h < dst_h {
                let last_row: Vec<u8> = padded[(src_h - 1) * dst_w..src_h * dst_w].to_vec();
                for row in src_h..dst_h {
                    let dst_start: usize = row * dst_w;
                    padded[dst_start..dst_start + dst_w].copy_from_slice(&last_row);
                }
            }
            padded
        }

        /// Pad a chroma plane using row-group replication to match C libjpeg-turbo's
        /// two-phase approach (jcprepct.c + jccoefct.c).
        fn pad_chroma_plane(
            plane: &[u8],
            src_w: usize,
            src_h: usize,
            dst_w: usize,
            dst_h: usize,
            max_v: usize,
        ) -> Vec<u8> {
            if src_w == dst_w && src_h == dst_h {
                return plane.to_vec();
            }
            let mut padded: Vec<u8> = vec![0u8; dst_w * dst_h];
            for row in 0..src_h {
                let src_start: usize = row * src_w;
                let dst_start: usize = row * dst_w;
                padded[dst_start..dst_start + src_w]
                    .copy_from_slice(&plane[src_start..src_start + src_w]);
                if src_w < dst_w {
                    let last_val: u8 = plane[src_start + src_w - 1];
                    for x in src_w..dst_w {
                        padded[dst_start + x] = last_val;
                    }
                }
            }
            if src_h < dst_h {
                let row_group_end: usize = src_h.div_ceil(max_v).min(dst_h / max_v) * max_v;
                let last_row: Vec<u8> = padded[(src_h - 1) * dst_w..src_h * dst_w].to_vec();
                // Phase 1: pad to row group boundary
                for row in src_h..row_group_end.min(dst_h) {
                    let dst_start: usize = row * dst_w;
                    padded[dst_start..dst_start + dst_w].copy_from_slice(&last_row);
                }
                // Phase 2: replicate last complete row group
                if row_group_end < dst_h {
                    let group_start: usize = row_group_end - max_v;
                    for row in row_group_end..dst_h {
                        let src_row: usize = group_start + (row - row_group_end) % max_v;
                        let dst_start: usize = row * dst_w;
                        let src_start: usize = src_row * dst_w;
                        let src_data: Vec<u8> = padded[src_start..src_start + dst_w].to_vec();
                        padded[dst_start..dst_start + dst_w].copy_from_slice(&src_data);
                    }
                }
            }
            padded
        }

        let (_, v_samp) = subsampling.sampling_factors();
        let fb_max_v: usize = v_samp as usize;
        let y_plane_padded: Vec<u8> = pad_plane(&y_plane, width, height, padded_w, padded_h);
        let cb_plane_padded: Vec<u8> =
            pad_chroma_plane(&cb_plane, width, height, padded_w, padded_h, fb_max_v);
        let cr_plane_padded: Vec<u8> =
            pad_chroma_plane(&cr_plane, width, height, padded_w, padded_h, fb_max_v);

        for mcu_row in 0..mcus_y {
            for mcu_col in 0..mcus_x {
                maybe_emit_restart!();

                let x0: usize = mcu_col * mcu_w;
                let y0: usize = mcu_row * mcu_h;

                if is_grayscale {
                    encode_single_block(
                        &y_plane_padded,
                        padded_w,
                        padded_h,
                        x0,
                        y0,
                        &luma_divisors,
                        &dc_luma_table,
                        &ac_luma_table,
                        &mut bit_writer,
                        &mut prev_dc_y,
                        fdct_quantize_fn,
                    );
                } else {
                    encode_color_mcu(
                        &y_plane_padded,
                        &cb_plane_padded,
                        &cr_plane_padded,
                        padded_w,
                        padded_h,
                        x0,
                        y0,
                        subsampling,
                        &luma_divisors,
                        &chroma_divisors,
                        &dc_luma_table,
                        &ac_luma_table,
                        &dc_chroma_table,
                        &ac_chroma_table,
                        &mut bit_writer,
                        &mut prev_dc_y,
                        &mut prev_dc_cb,
                        &mut prev_dc_cr,
                        fdct_quantize_fn,
                    );
                }

                mcu_count += 1;
            }
        }
    }

    bit_writer.flush();

    // Assemble output: markers + entropy data + EOI
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // Frame header. A quantization value above 255 needs 16-bit DQT entries,
    // which baseline (SOF0) forbids, so those streams are extended sequential
    // (SOF1). Only reachable through custom quant tables — the quality-scaled
    // Annex K tables clamp at 255.
    let needs_sof1: bool = luma_quant.iter().any(|&value| value > 255)
        || (!is_grayscale && chroma_quant.iter().any(|&value| value > 255));
    let write_frame_header = if needs_sof1 {
        marker_writer::write_sof1
    } else {
        marker_writer::write_sof0
    };
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        write_frame_header(&mut output, width as u16, height as u16, &components);
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![
            (1, h_samp, v_samp, 0), // Y
            (2, 1, 1, 1),           // Cb
            (3, 1, 1, 1),           // Cr
        ];
        write_frame_header(&mut output, width as u16, height as u16, &components);
    }

    // Huffman tables — the same bits/values the encoding tables were built from.
    marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);
    marker_writer::write_dht(&mut output, 1, 0, &ac_luma_bits, &ac_luma_values);
    if !is_grayscale {
        marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);
        marker_writer::write_dht(&mut output, 1, 1, &ac_chroma_bits, &ac_chroma_values);
    }

    // Restart interval. Omitted entirely when zero, matching C.
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // Scan header
    if is_grayscale {
        let scan_components = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components = vec![
            (1, 0, 0), // Y: DC table 0, AC table 0
            (2, 1, 1), // Cb: DC table 1, AC table 1
            (3, 1, 1), // Cr: DC table 1, AC table 1
        ];
        marker_writer::write_sos(&mut output, &scan_components);
    }

    // Entropy-coded data
    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress raw pixel data into a JPEG byte stream.
///
/// # Arguments
/// * `pixels` - Raw pixel data in the format specified by `pixel_format`
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pixel_format` - Pixel format of the input data
/// * `quality` - JPEG quality factor (1-100, where 100 is best quality)
/// * `subsampling` - Chroma subsampling mode
/// * `dct_method` - Forward DCT algorithm
///
/// # Returns
/// A `Vec<u8>` containing the complete JPEG file data.
pub fn compress(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
) -> Result<Vec<u8>> {
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .dct_method(dct_method),
    )
}

/// Compress raw pixel data into a JPEG byte stream using user-supplied Huffman tables.
///
/// Custom DC/AC table at index 0 overrides the standard luminance Huffman table.
/// Custom DC/AC table at index 1 overrides the standard chrominance Huffman table.
/// Unset slots fall back to the standard tables from Annex K.
#[allow(clippy::too_many_arguments)]
pub fn compress_custom_huffman(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    custom_dc: &[Option<HuffmanTableDef>; 4],
    custom_ac: &[Option<HuffmanTableDef>; 4],
) -> Result<Vec<u8>> {
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .custom_huffman(custom_dc, custom_ac),
    )
}

/// Compress raw pixel data into a JPEG byte stream using custom quantization tables.
///
/// When `custom_quant[0]` is `Some`, it overrides the quality-scaled luminance table.
/// When `custom_quant[1]` is `Some`, it overrides the quality-scaled chrominance table.
/// Unset slots fall back to the standard quality-scaled tables.
pub fn compress_custom_quant(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    custom_quant: &[Option<[u16; 64]>; 4],
) -> Result<Vec<u8>> {
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .custom_quant(custom_quant),
    )
}

/// Compress raw pixel data into a JPEG byte stream with DRI restart markers.
///
/// `restart_interval` is the number of MCU blocks between restart markers.
/// When non-zero, a DRI marker is written in the header and RST markers
/// are inserted into the entropy-coded data at the specified interval.
#[allow(clippy::too_many_arguments)]
pub fn compress_with_restart(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    restart_interval: u16,
    dct_method: DctMethod,
) -> Result<Vec<u8>> {
    compress_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .dct_method(dct_method)
            .restart_interval(restart_interval),
    )
}

/// Compress with optional ICC profile and EXIF metadata.
///
/// Inserts APP1 (EXIF) and APP2 (ICC) markers after the APP0 JFIF marker.
#[allow(clippy::too_many_arguments)]
pub fn compress_with_metadata(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let base = compress(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        DctMethod::IsLow,
    )?;
    inject_metadata(&base, icc_profile, exif_data)
}

/// Insert APP1 (EXIF) and APP2 (ICC) markers into an existing JPEG byte stream.
pub fn inject_metadata(
    base: &[u8],
    icc_profile: Option<&[u8]>,
    exif_data: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if icc_profile.is_none() && exif_data.is_none() {
        return Ok(base.to_vec());
    }

    // Find insertion point after all leading APP markers (APP0/JFIF, APP14/Adobe).
    // ICC (APP2) and EXIF (APP1) are inserted after these application markers
    // but before SOF/DHT/DRI/SOS.
    let mut insert_pos: usize = 2; // After SOI
    while insert_pos + 3 < base.len()
        && base[insert_pos] == 0xFF
        && (base[insert_pos + 1] & 0xF0) == 0xE0
    {
        let app_len = u16::from_be_bytes([base[insert_pos + 2], base[insert_pos + 3]]) as usize;
        insert_pos += 2 + app_len;
    }

    let extra_cap =
        icc_profile.map_or(0, |p| p.len() + 100) + exif_data.map_or(0, |e| e.len() + 20);
    let mut out = Vec::with_capacity(base.len() + extra_cap);
    out.extend_from_slice(&base[..insert_pos]);
    if let Some(exif) = exif_data {
        marker_writer::write_app1_exif(&mut out, exif);
    }
    if let Some(icc) = icc_profile {
        marker_writer::write_app2_icc(&mut out, icc);
    }
    out.extend_from_slice(&base[insert_pos..]);
    Ok(out)
}

/// Inject a COM (comment) marker into an existing JPEG byte stream, after APP0.
pub fn inject_comment(base: &[u8], text: &str) -> Vec<u8> {
    // Find insertion point after APP0 JFIF marker (SOI + APP0)
    let insert_pos = if base.len() >= 4 && base[2] == 0xFF && base[3] == 0xE0 {
        let app0_len = u16::from_be_bytes([base[4], base[5]]) as usize;
        2 + 2 + app0_len // SOI(2) + APP0 marker(2) + APP0 data
    } else {
        2 // After SOI only
    };

    let mut out = Vec::with_capacity(base.len() + text.len() + 6);
    out.extend_from_slice(&base[..insert_pos]);
    marker_writer::write_com(&mut out, text);
    out.extend_from_slice(&base[insert_pos..]);
    out
}

/// Inject saved markers (APP/COM) into an existing JPEG byte stream.
///
/// Markers are inserted after SOI + APP0 (and any existing metadata markers),
/// preserving the same insertion point pattern as `inject_metadata`/`inject_comment`.
pub fn inject_saved_markers(base: &[u8], markers: &[SavedMarker]) -> Vec<u8> {
    if markers.is_empty() {
        return base.to_vec();
    }

    // Find insertion point after APP0 JFIF marker (SOI + APP0)
    let insert_pos: usize = if base.len() >= 4 && base[2] == 0xFF && base[3] == 0xE0 {
        let app0_len: usize = u16::from_be_bytes([base[4], base[5]]) as usize;
        2 + 2 + app0_len
    } else {
        2
    };

    let extra: usize = markers.iter().map(|m| m.data.len() + 4).sum();
    let mut out: Vec<u8> = Vec::with_capacity(base.len() + extra);
    out.extend_from_slice(&base[..insert_pos]);
    for marker in markers {
        marker_writer::write_marker(&mut out, marker.code, &marker.data);
    }
    out.extend_from_slice(&base[insert_pos..]);
    out
}

/// A Huffman table in the form DHT wants it: the per-length code counts and
/// the value list, before either is turned into an encoding table.
type HuffmanTableSpec = ([u8; 17], Vec<u8>);

/// A colorspace encoded plane-by-plane with no colour conversion: CMYK
/// (`JCS_CMYK`) and RGB-direct (`JCS_RGB`).
///
/// Both write an Adobe APP14 and no JFIF, give every component the same
/// quantization and Huffman slot, and name their components with ASCII
/// initials (`jcparam.c:365-390`). They differ only in how many components
/// there are and which of them carry the sampling factors, so they share one
/// encoder rather than two copies of it — the copies are how #313's five
/// dropped options and #343's six got there in the first place.
struct DirectPlanarSpec {
    /// Component IDs in scan order: `b"RGB"` or `b"CMYK"`.
    component_ids: &'static [u8],
    /// `(h, v)` sampling factors per component, in scan order.
    sampling: Vec<(usize, usize)>,
    /// ICC profile to emit as APP2 right after the Adobe marker.
    icc_profile: Option<Vec<u8>>,
}

impl DirectPlanarSpec {
    /// TurboJPEG's CMYK layout (`turbojpeg.c:418-427`): components 0 and 3
    /// carry the sampling factors, 1 and 2 stay at 1x1.
    fn cmyk(h_samp: usize, v_samp: usize) -> Self {
        Self {
            component_ids: b"CMYK",
            sampling: vec![(h_samp, v_samp), (1, 1), (1, 1), (h_samp, v_samp)],
            icc_profile: None,
        }
    }

    /// `JCS_RGB` (`jcparam.c:365-370`): three components, all at 1x1.
    /// Subsampling is not expressible — every component is already maximal.
    fn rgb_direct(icc_profile: Option<&[u8]>) -> Self {
        Self {
            component_ids: b"RGB",
            sampling: vec![(1, 1), (1, 1), (1, 1)],
            icc_profile: icc_profile.map(<[u8]>::to_vec),
        }
    }

    fn components(&self) -> usize {
        self.component_ids.len()
    }

    fn max_sampling(&self) -> (usize, usize) {
        self.sampling
            .iter()
            .fold((1, 1), |(h, v), &(ch, cv)| (h.max(ch), v.max(cv)))
    }

    /// How far component `index` is downsampled from the maximum, as the
    /// `(h, v)` factor the block-gather helpers take.
    fn downsample_factor(&self, index: usize) -> (usize, usize) {
        let (max_h, max_v) = self.max_sampling();
        let (h, v) = self.sampling[index];
        (max_h / h, max_v / v)
    }
}

/// Geometry of a direct-planar scan: image and MCU-padded dimensions plus the
/// MCU grid derived from the maximum sampling factors.
struct PlanarLayout {
    width: usize,
    height: usize,
    padded_width: usize,
    padded_height: usize,
    mcus_x: usize,
    mcus_y: usize,
    restart_interval: u16,
}

/// What [`scan_planar_blocks`] hands back to its caller, in scan order.
enum PlanarScanEvent<'a> {
    /// An MCU boundary where the restart interval elapsed. DC predictors reset
    /// for every component; the single-pass caller also emits the marker.
    Restart,
    /// One quantized, zigzagged block belonging to `component`.
    Block {
        component: usize,
        coefficients: &'a [i16; 64],
    },
}

/// Walk the MCU grid, producing every block in scan order.
///
/// Both the direct-write path and the optimized-Huffman path drive this one
/// walk, so their block streams cannot drift: the statistics that pick the
/// optimal tables are gathered from exactly the blocks that will be written
/// with them. Getting that wrong is silent — the file still decodes, just with
/// tables fitted to a slightly different distribution.
fn scan_planar_blocks(
    planes: &[Vec<u8>],
    smoothed_halved: &[Option<Vec<u8>>],
    spec: &DirectPlanarSpec,
    layout: &PlanarLayout,
    divisors: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    on_event: &mut dyn FnMut(PlanarScanEvent),
) {
    let PlanarLayout {
        width,
        height,
        padded_width,
        padded_height,
        mcus_x,
        mcus_y,
        restart_interval,
    } = *layout;
    let (max_h, max_v) = spec.max_sampling();
    let mcu_w: usize = max_h * 8;
    let mcu_h: usize = max_v * 8;
    let restart_mcu_interval: u32 = restart_interval as u32;

    let mut prev_dc: Vec<i16> = vec![0i16; spec.components()];
    let mut mcu_count: u32 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            if restart_mcu_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(restart_mcu_interval)
            {
                prev_dc.fill(0);
                on_event(PlanarScanEvent::Restart);
            }
            mcu_count += 1;
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;

            for component in 0..spec.components() {
                let (h_samp, v_samp) = spec.sampling[component];
                let (h_factor, v_factor) = spec.downsample_factor(component);
                // A component's blocks tile the MCU at its own resolution, so
                // each covers `8 * factor` source samples.
                let block_w: usize = 8 * h_factor;
                let block_h: usize = 8 * v_factor;

                for dy in 0..v_samp {
                    for dx in 0..h_samp {
                        let block_x: usize = x0 + dx * block_w;
                        let block_y: usize = y0 + dy * block_h;

                        // The dummy-block test uses the *original* dimensions
                        // while the reads use the padded plane: a block past
                        // the image edge is a dummy (`jccoefct.c:178-199`), but
                        // one that merely straddles the edge is real and must
                        // read the padding C generated, not a clamp.
                        if block_x >= width || block_y >= height {
                            let mut dummy = [0i16; 64];
                            dummy[0] = prev_dc[component];
                            on_event(PlanarScanEvent::Block {
                                component,
                                coefficients: &dummy,
                            });
                            continue;
                        }

                        let coefficients: [i16; 64] =
                            if let Some(halved) = &smoothed_halved[component] {
                                // Pre-smoothed at half resolution, so the block
                                // is a plain gather at halved coordinates.
                                gather_block(
                                    halved,
                                    padded_width / 2,
                                    padded_height / 2,
                                    block_x / 2,
                                    block_y / 2,
                                    divisors,
                                    fdct_quantize_fn,
                                )
                            } else if h_factor == 1 && v_factor == 1 {
                                gather_block(
                                    &planes[component],
                                    padded_width,
                                    padded_height,
                                    block_x,
                                    block_y,
                                    divisors,
                                    fdct_quantize_fn,
                                )
                            } else {
                                gather_downsampled_block(
                                    &planes[component],
                                    padded_width,
                                    padded_height,
                                    block_x,
                                    block_y,
                                    h_factor,
                                    v_factor,
                                    divisors,
                                    fdct_quantize_fn,
                                )
                            };
                        prev_dc[component] = coefficients[0];
                        on_event(PlanarScanEvent::Block {
                            component,
                            coefficients: &coefficients,
                        });
                    }
                }
            }
        }
    }
}

/// Compress CMYK pixel data as a 4-component JPEG with Adobe APP14 marker.
///
/// Honors `subsampling` by writing the SOF sampling factors that
/// libjpeg-turbo's `tj3Compress8` uses for CMYK: components 0 and 3
/// (C and K) get the luma sampling factors, components 1 and 2 (M and Y)
/// stay at (1, 1). Per-MCU layout therefore emits `h_samp * v_samp` C
/// blocks, 1 M block (downsampled), 1 Y block (downsampled), then
/// `h_samp * v_samp` K blocks. No color conversion — CMYK samples are
/// encoded directly. Matches the SOF subsamp inference path so
/// `tj3DecompressHeader` reports the requested `TJSAMP_*` value back.
fn compress_cmyk(params: &CompressParams<'_>) -> Result<Vec<u8>> {
    let (h_samp_u8, v_samp_u8) = params.subsampling.sampling_factors();
    let (h_samp, v_samp) = (h_samp_u8 as usize, v_samp_u8 as usize);

    // JPEG spec § B.2.3 caps an MCU at 10 blocks. CMYK applies the luma
    // sampling factors to comp 0 AND comp 3, so per-MCU block count is
    // `2 * h_samp * v_samp + 2`. S410 / S24 (h*v = 8) blow that to 18 and
    // produce streams that conforming decoders reject. tjunittest skips
    // these combinations (line 727) so it never tripped, but our public
    // `compress()` API would silently emit invalid JPEGs.
    let blocks_per_mcu: usize = 2 * h_samp * v_samp + 2;
    if blocks_per_mcu > 10 {
        return Err(JpegError::Unsupported(format!(
            "CMYK with subsampling {:?} would emit {} blocks per MCU; JPEG spec § B.2.3 caps at 10. \
             Use a less aggressive subsampling for CMYK input.",
            params.subsampling, blocks_per_mcu
        )));
    }

    compress_direct_planar(params, &DirectPlanarSpec::cmyk(h_samp, v_samp))
}

/// Encode a colorspace that is stored plane-by-plane with no colour
/// conversion: CMYK (`JCS_CMYK`) or RGB-direct (`JCS_RGB`).
///
/// Every option in `params` applies. That is the whole point of this function
/// existing: both colorspaces used to sit behind an early return into a
/// narrower signature, and each dropped the options it could not express —
/// five of them for CMYK (#313), six for RGB-direct (#343), all silently.
fn compress_direct_planar(params: &CompressParams<'_>, spec: &DirectPlanarSpec) -> Result<Vec<u8>> {
    let CompressParams {
        pixels,
        width,
        height,
        quality,
        dct_method,
        restart_interval,
        custom_quant,
        custom_dc_huffman,
        custom_ac_huffman,
        optimize_huffman,
        smoothing_factor,
        ..
    } = *params;
    let components: usize = spec.components();
    let (max_h, max_v) = spec.max_sampling();

    let quant_table: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&quant_table)
    } else {
        scale_quant_for_fdct(&quant_table)
    };

    let ResolvedHuffman {
        dc_luma: default_dc_table,
        ac_luma: default_ac_table,
        dc_luma_bits,
        dc_luma_values,
        ac_luma_bits,
        ac_luma_values,
        ..
    } = ResolvedHuffman::resolve(custom_dc_huffman, custom_ac_huffman);

    // De-interleave into one plane per component at full resolution. Sub-
    // sampling happens per block during the scan so the SIMD downsample
    // helpers run, except under smoothing where the filter needs the whole
    // neighbourhood up front.
    let num_pixels: usize = width * height;
    let mut planes: Vec<Vec<u8>> = vec![vec![0u8; num_pixels]; components];
    for pixel in 0..num_pixels {
        for (component, plane) in planes.iter_mut().enumerate() {
            plane[pixel] = pixels[pixel * components + component];
        }
    }

    let mcu_w: usize = max_h * 8;
    let mcu_h: usize = max_v * 8;
    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);
    let padded_w: usize = mcus_x * mcu_w;
    let padded_h: usize = mcus_y * mcu_h;

    // Pad to the MCU grid the way C does (#340), rather than letting the
    // per-block edge path clamp.
    //
    // C pads twice by different rules: the input side completes a row group
    // (`jcprepct.c:171-178`), the output side fills the iMCU by repeating the
    // last *downsampled* row (`:197-205`). Carried back to full resolution the
    // second rule differs per component — one sampled at the maximum
    // downsamples 1:1, so repeating its last output row is repeating its last
    // input row, while one subsampled `v` ways means repeating the last
    // complete group of `v` input rows. A single rule cannot serve both, and a
    // plain clamp serves neither once `v > 1`.
    //
    // Smoothing changes this again: it needs context rows, which moves the
    // whole prep controller onto `pre_process_context` (`jcprepct.c:220-299`),
    // and that routine has no output-side padding at all — every component
    // falls back to a plain last-row repeat, including the ones C declines to
    // smooth. `need_context_rows` is pipeline-wide, not per component.
    let smoothing_on: bool = smoothing_factor > 0;
    let pad_to_mcu_grid = |plane: &[u8], row_group_height: usize| {
        pad_plane_to_mcu_grid(plane, width, height, padded_w, padded_h, row_group_height)
    };
    let smooth_full_size = |plane: &[u8]| {
        fullsize_smooth_plane(
            &pad_to_mcu_grid(plane, 1),
            padded_w,
            padded_h,
            smoothing_factor,
        )
    };

    // Which components get smoothed follows `jcsample.c:506-553`, not the
    // colorspace: a component at the maximum takes `fullsize_smooth_downsample`,
    // one halved in both axes takes `h2v2_smooth_downsample`, and any other
    // ratio clears `smoothok` and falls back to the plain downsample with a
    // JTRC_SMOOTH_NOTIMPL trace — which is what the unsmoothed path does.
    let mut smoothed_halved: Vec<Option<Vec<u8>>> = vec![None; components];
    let mut prepared: Vec<Vec<u8>> = Vec::with_capacity(components);
    for (component, plane) in planes.iter().enumerate() {
        let (h_factor, v_factor) = spec.downsample_factor(component);
        let at_maximum: bool = h_factor == 1 && v_factor == 1;
        if smoothing_on && h_factor == 2 && v_factor == 2 {
            smoothed_halved[component] = Some(h2v2_smooth_downsample_plane(
                &pad_to_mcu_grid(plane, 1),
                padded_w,
                padded_h,
                smoothing_factor,
            ));
        }
        prepared.push(match (smoothing_on, at_maximum) {
            (true, true) => smooth_full_size(plane),
            (true, false) => pad_to_mcu_grid(plane, 1),
            (false, _) => pad_to_mcu_grid(plane, v_factor),
        });
    }
    let planes: Vec<Vec<u8>> = prepared;

    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };

    let layout = PlanarLayout {
        width,
        height,
        padded_width: padded_w,
        padded_height: padded_h,
        mcus_x,
        mcus_y,
        restart_interval,
    };

    // Optimized Huffman runs the scan twice, as C's `optimize_coding` does
    // (`jcmaster.c`): once to count symbols, once to emit them. Re-deriving
    // the coefficients costs a second FDCT pass but keeps memory flat, which
    // matters more here — a four-component image buffers a third more blocks
    // than a three-component one.
    let optimized_tables: Option<(HuffmanTableSpec, HuffmanTableSpec)> = if optimize_huffman {
        let mut dc_freq = [0u32; 257];
        let mut ac_freq = [0u32; 257];
        let mut prev_dc: Vec<i16> = vec![0i16; components];
        scan_planar_blocks(
            &planes,
            &smoothed_halved,
            spec,
            &layout,
            &divisors,
            fdct_quantize_fn,
            &mut |event| match event {
                PlanarScanEvent::Restart => prev_dc.fill(0),
                PlanarScanEvent::Block {
                    component,
                    coefficients,
                } => {
                    let diff: i16 = coefficients[0] - prev_dc[component];
                    prev_dc[component] = coefficients[0];
                    crate::encode::huff_opt::gather_dc_symbol(diff, &mut dc_freq);
                    crate::encode::huff_opt::gather_ac_symbols(coefficients, &mut ac_freq);
                }
            },
        );
        Some((
            crate::encode::huff_opt::gen_optimal_table(&dc_freq),
            crate::encode::huff_opt::gen_optimal_table(&ac_freq),
        ))
    } else {
        None
    };

    // Every component shares table slot 0 (`jcparam.c:365-390`), so there is
    // one DC and one AC table regardless of which mode produced them.
    let (dc_bits, dc_values, ac_bits, ac_values): (&[u8; 17], &[u8], &[u8; 17], &[u8]) =
        match &optimized_tables {
            Some(((dc_bits, dc_values), (ac_bits, ac_values))) => {
                (dc_bits, dc_values, ac_bits, ac_values)
            }
            None => (
                &dc_luma_bits,
                &dc_luma_values,
                &ac_luma_bits,
                &ac_luma_values,
            ),
        };
    let dc_table: HuffTable = match &optimized_tables {
        Some(_) => build_huff_table(dc_bits, dc_values),
        None => default_dc_table,
    };
    let ac_table: HuffTable = match &optimized_tables {
        Some(_) => build_huff_table(ac_bits, ac_values),
        None => default_ac_table,
    };

    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc: Vec<i16> = vec![0i16; components];
    let mut restart_marker_index: u8 = 0;
    scan_planar_blocks(
        &planes,
        &smoothed_halved,
        spec,
        &layout,
        &divisors,
        fdct_quantize_fn,
        &mut |event| match event {
            // Every component resets together at a restart, as C does.
            PlanarScanEvent::Restart => {
                bit_writer.flush_restart();
                bit_writer.write_restart_marker(restart_marker_index);
                restart_marker_index = restart_marker_index.wrapping_add(1);
                prev_dc.fill(0);
            }
            PlanarScanEvent::Block {
                component,
                coefficients,
            } => HuffmanEncoder::encode_block(
                &mut bit_writer,
                coefficients,
                &mut prev_dc[component],
                &dc_table,
                &ac_table,
            ),
        },
    );

    bit_writer.flush();

    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    // No JFIF APP0 (#339). `jpeg_set_colorspace` clears `write_JFIF_header` and
    // re-enables it only for JCS_GRAYSCALE and JCS_YCbCr (`jcparam.c:357-392`);
    // JCS_CMYK and JCS_RGB set `write_Adobe_marker` alone. JFIF is defined for
    // grayscale and YCbCr only, so an APP0 here asserts something untrue about
    // the data — and cost 18 bytes in every CMYK file we wrote.
    marker_writer::write_app14_adobe(&mut output, 0);

    // ICC profile immediately after APP14, matching C cjpeg's marker order.
    if let Some(icc) = &spec.icc_profile {
        marker_writer::write_app2_icc(&mut output, icc);
    }

    marker_writer::write_dqt(&mut output, 0, &quant_table);

    // Component IDs are the ASCII initials libjpeg writes (#339):
    // 'C','M','Y','K' or 'R','G','B' (`jcparam.c:365-390`).
    let sof_components: Vec<(u8, u8, u8, u8)> = spec
        .component_ids
        .iter()
        .zip(spec.sampling.iter())
        .map(|(&id, &(h, v))| (id, h as u8, v as u8, 0))
        .collect();
    // A quantization value above 255 needs 16-bit DQT entries, which baseline
    // (SOF0) forbids, so those streams are extended sequential (SOF1) — what
    // `cjpeg -rgb -quality 1` writes, warning "quantization tables are too
    // coarse for baseline JPEG" as it does. Reachable through custom tables or
    // a low quality with `force_baseline` off.
    let needs_sof1: bool = quant_table.iter().any(|&value| value > 255);
    let write_frame_header = if needs_sof1 {
        marker_writer::write_sof1
    } else {
        marker_writer::write_sof0
    };
    write_frame_header(&mut output, width as u16, height as u16, &sof_components);

    marker_writer::write_dht(&mut output, 0, 0, dc_bits, dc_values);
    marker_writer::write_dht(&mut output, 1, 0, ac_bits, ac_values);

    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // SOS references the same IDs the SOF declared.
    let scan_components: Vec<(u8, u8, u8)> = spec
        .component_ids
        .iter()
        .map(|&id| (id, 0u8, 0u8))
        .collect();
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress RGB pixels directly without color conversion (JCS_RGB / `cjpeg -rgb`).
///
/// Component IDs follow C libjpeg-turbo convention: R=82('R'), G=71('G'), B=66('B').
/// All 3 components use 1x1 sampling and the same luminance quantization table.
/// Produces Adobe APP14 marker with transform=0 (no JFIF APP0).
///
/// This signature carries only quality and the DCT method. Anything else the
/// caller set — restart interval, custom tables, optimized Huffman, smoothing —
/// has to reach [`compress_rgb_direct_with_params`], which is what `Encoder`
/// uses. Routing through here instead is what made all six silently vanish
/// (#343).
pub fn compress_rgb_direct(
    pixels: &[u8],
    width: usize,
    height: usize,
    quality: u8,
    dct_method: DctMethod,
    icc_profile: Option<&[u8]>,
) -> Result<Vec<u8>> {
    compress_rgb_direct_with_params(
        &CompressParams::new(
            pixels,
            width,
            height,
            PixelFormat::Rgb,
            quality,
            Subsampling::S444,
        )
        .dct_method(dct_method),
        icc_profile,
    )
}

/// Compress RGB pixels as `JCS_RGB`, honouring every option in `params`.
///
/// `subsampling` is the one thing that cannot apply: `jpeg_set_colorspace`
/// puts all three components at 1x1 (`jcparam.c:365-370`), so there is nothing
/// to subsample. C is the same — `cjpeg -rgb -sample 2x2` writes 1x1.
pub fn compress_rgb_direct_with_params(
    params: &CompressParams<'_>,
    icc_profile: Option<&[u8]>,
) -> Result<Vec<u8>> {
    if params.width == 0 || params.height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if params.width > 65535 || params.height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            params.width, params.height
        )));
    }
    let expected_size: usize = params.width * params.height * 3;
    if params.pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: params.pixels.len(),
        });
    }

    compress_direct_planar(params, &DirectPlanarSpec::rgb_direct(icc_profile))
}

/// Compress as lossless JPEG (SOF3).
///
/// Uses predictor 1 (left) and no point transform.
/// Produces exact pixel-identical output when decoded.
/// Currently supports grayscale only; use `compress_lossless_extended` for color.
pub fn compress_lossless(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
) -> Result<Vec<u8>> {
    compress_lossless_extended(pixels, width, height, pixel_format, 1, 0, 0)
}

/// Compress as lossless JPEG (SOF3) with configurable predictor and point transform.
///
/// # Arguments
/// * `predictor` - Predictor selection value (1-7), as defined in ITU-T T.81 Table H.1
/// * `point_transform` - Point transform value (0-15), right-shifts pixel data before encoding
///
/// Supports grayscale (1-component) and RGB (3-component interleaved).
/// For RGB, the encoder converts to YCbCr before encoding (JFIF convention).
pub fn compress_lossless_extended(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    compress_lossless_extended_precision(
        pixels,
        width,
        height,
        pixel_format,
        predictor,
        point_transform,
        restart_interval,
        8,
    )
}

/// Like `compress_lossless_extended` but with an explicit sample precision
/// (2..=8). The precision field controls the SOF3 marker and the lossless
/// predictor arithmetic; the source samples are still `u8` (8-bit values).
#[allow(clippy::too_many_arguments)]
pub fn compress_lossless_extended_precision(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
    precision: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }

    if !(2..=8).contains(&precision) {
        return Err(JpegError::Unsupported(format!(
            "lossless precision must be 2-8 for 8-bit samples, got {}",
            precision
        )));
    }

    if point_transform >= precision {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0-{} for {}-bit precision, got {}",
            precision - 1,
            precision,
            point_transform
        )));
    }

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    match pixel_format {
        PixelFormat::Grayscale => compress_lossless_grayscale(
            pixels,
            width,
            height,
            predictor,
            point_transform,
            restart_interval,
            precision,
        ),
        PixelFormat::Rgb => compress_lossless_rgb(
            pixels,
            width,
            height,
            predictor,
            point_transform,
            restart_interval,
            precision,
        ),
        _ => Err(JpegError::Unsupported(format!(
            "lossless encoding does not support {:?}, use Grayscale or Rgb",
            pixel_format
        ))),
    }
}

/// Compute the lossless difference for a single sample.
///
/// Returns the **raw signed difference** `(sample >> Pt) - prediction`,
/// matching libjpeg-turbo `jclossls.c` (`*diff_buf++ = samp - PREDICTOR;`).
/// The lossless JPEG bitstream (ITU-T T.81 Annex H.1.2.2) classifies the
/// diff by its raw 16-bit signed magnitude, NOT by the P-bit modular value.
/// Folding to the P-bit modular range produces a bitstream that decodes to
/// the same pixels (the decoder reconstructs modulo 2^P) but is NOT
/// byte-identical to C cjpeg, because the magnitude category (and thus
/// the optimised Huffman table) differs.
///
/// For 8-bit (P=8) samples the diff is in [-255, +255]; for higher
/// precision it is in [-(2^P - 1), +(2^P - 1)]. Both fit in i16 for
/// P <= 15. The 16-bit precision path lives in `src/api/precision.rs`.
#[allow(clippy::too_many_arguments)]
fn lossless_diff(
    pixel: i32,
    x: usize,
    y: usize,
    plane: &[u8],
    width: usize,
    predictor: u8,
    point_transform: u8,
    precision: u8,
) -> i16 {
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1);

    // Apply point transform: shift right before encoding
    let sample: i32 = pixel >> point_transform as i32;

    let prediction: i32 = if y == 0 && x == 0 {
        initial_pred
    } else if y == 0 {
        // First row: predictor is always "left" (ra) regardless of psv
        (plane[y * width + x - 1] as i32) >> point_transform as i32
    } else if x == 0 {
        // First column: predictor is always "above" (rb) regardless of psv
        (plane[(y - 1) * width + x] as i32) >> point_transform as i32
    } else {
        let ra: i32 = (plane[y * width + x - 1] as i32) >> point_transform as i32;
        let rb: i32 = (plane[(y - 1) * width + x] as i32) >> point_transform as i32;
        let rc: i32 = (plane[(y - 1) * width + x - 1] as i32) >> point_transform as i32;
        crate::decode::lossless::predict(predictor, ra, rb, rc)
    };

    // Raw signed difference (no modular fold). See doc comment.
    (sample - prediction) as i16
}

/// Encode a single-component (grayscale) lossless JPEG.
fn compress_lossless_grayscale(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
    precision: u8,
) -> Result<Vec<u8>> {
    let num_pixels: usize = width * height;
    let ri: u32 = restart_interval as u32;
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1);

    // Collect all diffs for 2-pass optimized Huffman encoding.
    let mut all_diffs: Vec<i16> = Vec::with_capacity(num_pixels);
    let mut mcu_count: u32 = 0;
    let mut in_restart_row: bool = false;

    for y in 0..height {
        for x in 0..width {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                in_restart_row = true;
            }
            let pixel: i32 = pixels[y * width + x] as i32;
            // After restart, use "first row" prediction: x=0 → initial_pred,
            // x>0 → left neighbor (PSV=1 fallback, matching decoder behavior).
            let signed_diff: i16 = if in_restart_row {
                let sample: i32 = pixel >> point_transform as i32;
                if x == 0 {
                    (sample - initial_pred) as i16
                } else {
                    let left: i32 = pixels[y * width + x - 1] as i32 >> point_transform as i32;
                    (sample - left) as i16
                }
            } else {
                lossless_diff(
                    pixel,
                    x,
                    y,
                    pixels,
                    width,
                    predictor,
                    point_transform,
                    precision,
                )
            };
            all_diffs.push(signed_diff);
            mcu_count += 1;
        }
        in_restart_row = false;
    }

    // Pass 1: gather DC symbol frequencies for optimal Huffman table.
    use crate::encode::huff_opt;
    let mut dc_freq: [u32; 257] = [0u32; 257];
    for &diff in &all_diffs {
        huff_opt::gather_dc_symbol(diff, &mut dc_freq);
    }
    dc_freq[256] = 1;
    let (opt_bits, opt_values) = huff_opt::gen_optimal_table(&dc_freq);
    let dc_table: HuffTable = build_huff_table(&opt_bits, &opt_values);

    // Pass 2: entropy encode with optimal table + restart markers.
    let mut bit_writer: BitWriter = BitWriter::new(num_pixels);
    let mut restart_idx: u8 = 0;
    mcu_count = 0;

    for &diff in &all_diffs {
        if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
            bit_writer.flush();
            bit_writer.write_restart_marker(restart_idx);
            restart_idx = (restart_idx + 1) & 7;
        }
        HuffmanEncoder::encode_dc_only(&mut bit_writer, diff, &dc_table);
        mcu_count += 1;
    }
    bit_writer.flush();

    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 256);

    marker_writer::write_soi(&mut output);

    // JFIF APP0 marker (matching C cjpeg grayscale lossless)
    marker_writer::write_app0_jfif(&mut output);

    // SOF3 with 1 component
    let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
    marker_writer::write_sof3(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // Optimized DC Huffman table (after SOF3, matching C)
    marker_writer::write_dht(&mut output, 0, 0, &opt_bits, &opt_values);

    // DRI (restart interval)
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    let scan_components: Vec<(u8, u8)> = vec![(1, 0)];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode a 3-component RGB interleaved lossless JPEG.
///
/// Stores raw RGB component values with no color conversion, matching
/// C libjpeg-turbo behavior for lossless JPEG (JCS_RGB, no YCbCr conversion).
fn compress_lossless_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
    restart_interval: u16,
    precision: u8,
) -> Result<Vec<u8>> {
    let num_pixels: usize = width * height;
    let ri: u32 = restart_interval as u32;
    let initial_pred: i32 = 1 << (precision as i32 - point_transform as i32 - 1);

    // Split interleaved RGB into separate planes (no color conversion)
    let mut r_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut g_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut b_plane: Vec<u8> = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        r_plane[i] = pixels[i * 3];
        g_plane[i] = pixels[i * 3 + 1];
        b_plane[i] = pixels[i * 3 + 2];
    }

    let planes: [&[u8]; 3] = [&r_plane, &g_plane, &b_plane];

    // Collect all lossless diffs first for 2-pass optimized Huffman encoding.
    // One MCU = one pixel (all 3 interleaved components).
    let mut all_diffs: Vec<i16> = Vec::with_capacity(num_pixels * 3);
    let mut mcu_count: u32 = 0;
    let mut in_restart_row: bool = false;

    for y in 0..height {
        for x in 0..width {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                in_restart_row = true;
            }
            for plane in &planes {
                let pixel: i32 = plane[y * width + x] as i32;
                // After restart, use "first row" prediction: x=0 → initial_pred,
                // x>0 → left neighbor (PSV=1 fallback, matching decoder).
                let signed_diff: i16 = if in_restart_row {
                    let sample: i32 = pixel >> point_transform as i32;
                    if x == 0 {
                        (sample - initial_pred) as i16
                    } else {
                        let left: i32 = plane[y * width + x - 1] as i32 >> point_transform as i32;
                        (sample - left) as i16
                    }
                } else {
                    lossless_diff(
                        pixel,
                        x,
                        y,
                        plane,
                        width,
                        predictor,
                        point_transform,
                        precision,
                    )
                };
                all_diffs.push(signed_diff);
            }
            mcu_count += 1;
        }
        in_restart_row = false;
    }

    // Pass 1: gather DC symbol frequencies for optimal Huffman table.
    use crate::encode::huff_opt;
    let mut dc_freq: [u32; 257] = [0u32; 257];
    for &diff in &all_diffs {
        huff_opt::gather_dc_symbol(diff, &mut dc_freq);
    }
    dc_freq[256] = 1; // pseudo-symbol (Annex K.2)
    let (opt_bits, opt_values) = huff_opt::gen_optimal_table(&dc_freq);
    let dc_table: HuffTable = build_huff_table(&opt_bits, &opt_values);

    // Pass 2: entropy encode with optimal table + restart markers.
    // Restart markers are emitted between MCUs (1 MCU = 3 component data units).
    let mut bit_writer: BitWriter = BitWriter::new(num_pixels * 3);
    let mut restart_idx: u8 = 0;
    let mut diff_idx: usize = 0;
    mcu_count = 0;

    for _y in 0..height {
        for _x in 0..width {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                bit_writer.flush();
                bit_writer.write_restart_marker(restart_idx);
                restart_idx = (restart_idx + 1) & 7;
            }
            for _ in 0..3 {
                HuffmanEncoder::encode_dc_only(&mut bit_writer, all_diffs[diff_idx], &dc_table);
                diff_idx += 1;
            }
            mcu_count += 1;
        }
    }
    bit_writer.flush();

    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 512);

    marker_writer::write_soi(&mut output);

    // Adobe APP14 with transform=0 to signal RGB colorspace (matching C cjpeg).
    // C cjpeg does NOT emit JFIF APP0 for RGB lossless — only APP14.
    marker_writer::write_app14_adobe(&mut output, 0);

    // SOF3 with 3 components: R(id='R'), G(id='G'), B(id='B'), all 1x1, qt=0.
    // C libjpeg-turbo uses ASCII component IDs for RGB colorspace lossless.
    let components: Vec<(u8, u8, u8, u8)> = vec![
        (b'R', 1, 1, 0), // R: id=0x52, h=1, v=1, qt=0
        (b'G', 1, 1, 0), // G: id=0x47, h=1, v=1, qt=0
        (b'B', 1, 1, 0), // B: id=0x42, h=1, v=1, qt=0
    ];
    marker_writer::write_sof3(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // Optimized DC Huffman table 0 for all 3 components (after SOF3, matching C)
    marker_writer::write_dht(&mut output, 0, 0, &opt_bits, &opt_values);

    // DRI (restart interval)
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // SOS with 3 components: all use DC table 0 (matching SOF3 ASCII IDs)
    let scan_components: Vec<(u8, u8)> = vec![
        (b'R', 0), // R -> DC table 0
        (b'G', 0), // G -> DC table 0
        (b'B', 0), // B -> DC table 0
    ];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress as lossless JPEG with arithmetic entropy coding (SOF11).
///
/// Same predictor-based pipeline as SOF3 but uses ArithEncoder instead of
/// Huffman coding. Writes SOF11 (0xCB) marker and DAC conditioning parameters.
pub fn compress_lossless_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    if !(1..=7).contains(&predictor) {
        return Err(JpegError::Unsupported(format!(
            "lossless predictor must be 1-7, got {}",
            predictor
        )));
    }

    if point_transform >= 8 {
        return Err(JpegError::Unsupported(format!(
            "point transform must be 0-7 for 8-bit precision, got {}",
            point_transform
        )));
    }

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    match pixel_format {
        PixelFormat::Grayscale => compress_lossless_arithmetic_grayscale(
            pixels,
            width,
            height,
            predictor,
            point_transform,
        ),
        PixelFormat::Rgb => {
            compress_lossless_arithmetic_rgb(pixels, width, height, predictor, point_transform)
        }
        _ => Err(JpegError::Unsupported(format!(
            "lossless arithmetic encoding does not support {:?}, use Grayscale or Rgb",
            pixel_format
        ))),
    }
}

/// Encode a single-component (grayscale) lossless JPEG with arithmetic coding.
fn compress_lossless_arithmetic_grayscale(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    let precision: u8 = 8;

    let mut arith_enc: ArithEncoder = ArithEncoder::new(width * height);

    // Encode each pixel's difference as a DC coefficient
    for y in 0..height {
        for x in 0..width {
            let pixel: i32 = pixels[y * width + x] as i32;
            let signed_diff: i16 = lossless_diff(
                pixel,
                x,
                y,
                pixels,
                width,
                predictor,
                point_transform,
                precision,
            );
            // Pack the difference into block[0] and encode as DC-only
            let mut block: [i16; 64] = [0i16; 64];
            block[0] = signed_diff.wrapping_add(arith_enc.last_dc_val[0] as i16);
            arith_enc.encode_dc_sequential(&block, 0, 0);
        }
    }

    arith_enc.finish();

    let mut output: Vec<u8> = Vec::with_capacity(arith_enc.data().len() + 256);

    marker_writer::write_soi(&mut output);

    // SOF11 with 1 component
    let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
    marker_writer::write_sof11(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // DAC marker for DC table 0
    let dc_params: [(u8, u8); 2] = [(0u8, 1u8), (0, 1)];
    let ac_params: [u8; 2] = [5u8, 5];
    marker_writer::write_dac(&mut output, 1, &dc_params, 0, &ac_params);

    // SOS for lossless scan
    let scan_components: Vec<(u8, u8)> = vec![(1, 0)];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(arith_enc.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode a 3-component RGB interleaved lossless JPEG with arithmetic coding.
///
/// Stores raw RGB component values with no color conversion, matching
/// C libjpeg-turbo behavior for lossless JPEG (JCS_RGB, no YCbCr conversion).
fn compress_lossless_arithmetic_rgb(
    pixels: &[u8],
    width: usize,
    height: usize,
    predictor: u8,
    point_transform: u8,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    let precision: u8 = 8;
    let num_pixels: usize = width * height;

    // Split interleaved RGB into separate planes (no color conversion)
    let mut r_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut g_plane: Vec<u8> = vec![0u8; num_pixels];
    let mut b_plane: Vec<u8> = vec![0u8; num_pixels];

    for i in 0..num_pixels {
        r_plane[i] = pixels[i * 3];
        g_plane[i] = pixels[i * 3 + 1];
        b_plane[i] = pixels[i * 3 + 2];
    }

    let planes: [&[u8]; 3] = [&r_plane, &g_plane, &b_plane];
    // All components use DC table 0 (no chrominance table)
    let dc_tbls: [usize; 3] = [0, 0, 0];

    let mut arith_enc: ArithEncoder = ArithEncoder::new(num_pixels * 3);

    // Interleaved encoding: for each pixel, encode diff for Y, Cb, Cr
    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let pixel: i32 = planes[c][y * width + x] as i32;
                let signed_diff: i16 = lossless_diff(
                    pixel,
                    x,
                    y,
                    planes[c],
                    width,
                    predictor,
                    point_transform,
                    precision,
                );
                // Pack the difference into block[0] and encode as DC-only
                let mut block: [i16; 64] = [0i16; 64];
                block[0] = signed_diff.wrapping_add(arith_enc.last_dc_val[c] as i16);
                arith_enc.encode_dc_sequential(&block, c, dc_tbls[c]);
            }
        }
    }

    arith_enc.finish();

    let mut output: Vec<u8> = Vec::with_capacity(arith_enc.data().len() + 512);

    marker_writer::write_soi(&mut output);

    // SOF11 with 3 components: R(id=1), G(id=2), B(id=3), all 1x1, qt=0
    let components: Vec<(u8, u8, u8, u8)> = vec![
        (1, 1, 1, 0), // R
        (2, 1, 1, 0), // G
        (3, 1, 1, 0), // B
    ];
    marker_writer::write_sof11(
        &mut output,
        width as u16,
        height as u16,
        precision,
        &components,
    );

    // DAC marker for DC table 0 only
    let dc_params: [(u8, u8); 2] = [(0u8, 1u8), (0, 1)];
    let ac_params: [u8; 2] = [5u8, 5];
    marker_writer::write_dac(&mut output, 1, &dc_params, 0, &ac_params);

    // SOS with 3 components: all use DC table 0
    let scan_components: Vec<(u8, u8)> = vec![
        (1, 0), // R -> DC table 0
        (2, 0), // G -> DC table 0
        (3, 0), // B -> DC table 0
    ];
    marker_writer::write_sos_lossless(&mut output, &scan_components, predictor, point_transform);

    output.extend_from_slice(arith_enc.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Per-component block layout for progressive encoding.
struct CompLayout {
    blocks_x: usize,
    blocks_y: usize,
    h_blocks: usize,
    v_blocks: usize,
}

/// Compress as progressive JPEG (SOF2, multi-scan).
///
/// Buffers all DCT coefficients, then encodes across multiple scans
/// following the default `simple_progression()` scan script.
pub fn compress_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
) -> Result<Vec<u8>> {
    compress_progressive_with_restart(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        dct_method,
        0,
        0,
        None,
    )
}

/// Compress as progressive JPEG (SOF2) with an explicit restart interval.
///
/// `restart_interval` is the number of MCUs between restart markers
/// (0 disables restart marker insertion). `restart_in_rows` is the
/// per-row restart hint that, when non-zero, takes precedence: every
/// scan recomputes its restart_interval as `restart_in_rows * MCUs_per_row`
/// based on whether that scan is interleaved or non-interleaved. This
/// mirrors `jcmaster.c`, where DC interleaved scans use the iMCU width and
/// non-interleaved AC scans use the per-component `width_in_blocks` to
/// derive the per-scan restart distance — required for byte-parity with
/// `cjpeg -r N -p`.
#[allow(clippy::too_many_arguments)]
pub fn compress_progressive_with_restart(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
    use crate::encode::progressive::simple_progression;

    let is_grayscale = pixel_format == PixelFormat::Grayscale;
    let num_components = if is_grayscale { 1 } else { 3 };
    let scans = simple_progression(num_components);

    compress_progressive_with_scans(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        &scans,
        dct_method,
        restart_interval,
        restart_in_rows,
        custom_quant,
    )
}

/// Compress as progressive JPEG (SOF2) with a user-supplied scan script.
///
/// Same as `compress_progressive` but uses the provided `ScanScript` entries
/// instead of the default `simple_progression()` scan order.
#[allow(clippy::too_many_arguments)]
pub fn compress_progressive_custom(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    script: &[ScanScript],
    dct_method: DctMethod,
) -> Result<Vec<u8>> {
    compress_progressive_custom_with_restart(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        script,
        dct_method,
        0,
        0,
        None,
    )
}

/// Same as `compress_progressive_custom` but with an explicit restart interval.
#[allow(clippy::too_many_arguments)]
pub fn compress_progressive_custom_with_restart(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    script: &[ScanScript],
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
    let scans: Vec<ProgressiveScan> = script
        .iter()
        .map(|s| ProgressiveScan {
            component_indices: s.components.iter().map(|&c| c as usize).collect(),
            ss: s.ss,
            se: s.se,
            ah: s.ah,
            al: s.al,
        })
        .collect();

    compress_progressive_with_scans(
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        &scans,
        dct_method,
        restart_interval,
        restart_in_rows,
        custom_quant,
    )
}

/// Shared progressive encoding logic used by both default and custom scan scripts.
#[allow(clippy::too_many_arguments)]
fn compress_progressive_with_scans(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    scans: &[ProgressiveScan],
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };
    let use_simd_fdct: bool = dct_method == DctMethod::IsLow;

    let (luma_quant, chroma_quant) = resolve_quant_tables(custom_quant, quality);
    let luma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&luma_quant)
    } else {
        scale_quant_for_fdct(&luma_quant)
    };
    let chroma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&chroma_quant)
    } else {
        scale_quant_for_fdct(&chroma_quant)
    };

    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);

    let (h_samp, v_samp) = if is_grayscale {
        (1usize, 1usize)
    } else {
        let (h, v) = subsampling.sampling_factors();
        (h as usize, v as usize)
    };

    let comp_layouts: Vec<CompLayout> = if is_grayscale {
        vec![CompLayout {
            blocks_x: mcus_x,
            blocks_y: mcus_y,
            h_blocks: 1,
            v_blocks: 1,
        }]
    } else {
        vec![
            CompLayout {
                blocks_x: mcus_x * h_samp,
                blocks_y: mcus_y * v_samp,
                h_blocks: h_samp,
                v_blocks: v_samp,
            },
            CompLayout {
                blocks_x: mcus_x,
                blocks_y: mcus_y,
                h_blocks: 1,
                v_blocks: 1,
            },
            CompLayout {
                blocks_x: mcus_x,
                blocks_y: mcus_y,
                h_blocks: 1,
                v_blocks: 1,
            },
        ]
    };

    // Buffer all quantized coefficients per component
    let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_layouts
        .iter()
        .map(|cl| vec![[0i16; 64]; cl.blocks_x * cl.blocks_y])
        .collect();

    // Per-component actual block counts (width_in_blocks × height_in_blocks).
    // For non-interleaved AC scans, C libjpeg-turbo only encodes this many blocks,
    // not the MCU-padded count. Must match decoder expectations from SOF2 dimensions.
    let comp_wib: Vec<usize> = if is_grayscale {
        vec![width.div_ceil(8)]
    } else {
        vec![
            width.div_ceil(8),          // Y
            width.div_ceil(h_samp * 8), // Cb
            width.div_ceil(h_samp * 8), // Cr
        ]
    };
    let comp_hib: Vec<usize> = if is_grayscale {
        vec![height.div_ceil(8)]
    } else {
        vec![
            height.div_ceil(8),          // Y
            height.div_ceil(v_samp * 8), // Cb
            height.div_ceil(v_samp * 8), // Cr
        ]
    };

    // FDCT + quantize all blocks into coefficient buffers.
    // For blocks beyond width_in_blocks or height_in_blocks, C libjpeg-turbo
    // creates "dummy" blocks (all AC=0, DC=previous block's DC) instead of
    // FDCT'ing edge-replicated pixels (jccoefct.c lines 184-200).
    let y_wib: usize = comp_wib[0];
    let y_hib: usize = comp_hib[0];
    let mut prev_dc_y_prog: i16 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            let x0: usize = mcu_x * mcu_w;
            let y0: usize = mcu_y * mcu_h;

            if is_grayscale {
                let bx: usize = mcu_x;
                let by: usize = mcu_y;
                if is_y_dummy(x0, y0, y_wib, y_hib) {
                    coeff_bufs[0][by * mcus_x + bx][0] = prev_dc_y_prog;
                } else {
                    progressive_fdct_y_block(
                        &y_plane,
                        width,
                        height,
                        x0,
                        y0,
                        &luma_divisors,
                        fdct_quantize_fn,
                        &mut coeff_bufs[0][by * mcus_x + bx],
                        use_simd_fdct,
                    );
                    prev_dc_y_prog = coeff_bufs[0][by * mcus_x + bx][0];
                }
            } else {
                // Y blocks
                let blocks_x: usize = comp_layouts[0].blocks_x;
                for bv in 0..v_samp {
                    for bh in 0..h_samp {
                        let bx: usize = mcu_x * h_samp + bh;
                        let by: usize = mcu_y * v_samp + bv;
                        if is_y_dummy(x0 + bh * 8, y0 + bv * 8, y_wib, y_hib) {
                            coeff_bufs[0][by * blocks_x + bx][0] = prev_dc_y_prog;
                        } else {
                            progressive_fdct_y_block(
                                &y_plane,
                                width,
                                height,
                                x0 + bh * 8,
                                y0 + bv * 8,
                                &luma_divisors,
                                fdct_quantize_fn,
                                &mut coeff_bufs[0][by * blocks_x + bx],
                                use_simd_fdct,
                            );
                            prev_dc_y_prog = coeff_bufs[0][by * blocks_x + bx][0];
                        }
                    }
                }
                // Cb/Cr blocks
                for (comp_idx, plane) in [(1usize, &cb_plane), (2usize, &cr_plane)] {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    progressive_fdct_chroma_block(
                        plane,
                        width,
                        height,
                        x0,
                        y0,
                        h_samp,
                        v_samp,
                        &chroma_divisors,
                        fdct_quantize_fn,
                        &mut coeff_bufs[comp_idx][by * mcus_x + bx],
                        use_simd_fdct,
                    );
                }
            }
        }
    }

    // Assemble output
    let mut output = Vec::with_capacity(width * height * 2);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // SOF2 (progressive)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof2(&mut output, width as u16, height as u16, &components);
    } else {
        let components = vec![
            (1, h_samp as u8, v_samp as u8, 0),
            (2, 1, 1, 1),
            (3, 1, 1, 1),
        ];
        marker_writer::write_sof2(&mut output, width as u16, height as u16, &components);
    }

    // Single BitWriter reused across all scans (reset instead of reallocate).
    let mut bit_writer: BitWriter = BitWriter::new(width * height / 4);

    // Pre-allocate precomp buffers outside the scan loop (clear+reuse per scan).
    let max_blocks: usize = comp_layouts
        .iter()
        .map(|cl| cl.blocks_x * cl.blocks_y)
        .max()
        .unwrap_or(0);
    let mut precomp_zerobits: Vec<u64> = Vec::with_capacity(max_blocks);
    let mut precomp_values: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_diffs: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_absvals: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_signs: Vec<[u16; 64]> = Vec::with_capacity(max_blocks);
    let mut precomp_eob: Vec<usize> = Vec::with_capacity(max_blocks);

    // Encode each scan with per-scan optimized Huffman tables.
    // DC first scans (ss=0, se=0, ah=0): gather DC frequencies, generate optimal
    // table, write DHT, encode. DC refine scans (ah>0): no DHT, just encode.
    // AC scans (ss>0): gather AC frequencies, generate optimal table, write DHT, encode.
    //
    // Track the last-emitted DRI value across scans. C `jcmarker.c::write_scan_header`
    // only emits DRI when `restart_interval` differs from the previous scan's value
    // (initial 0); we mirror that to avoid duplicate DRI markers.
    let mut last_ri: u16 = 0;
    for scan in scans {
        // Per-scan restart_interval: when `restart_in_rows` is set, derive
        // it from the scan's MCUs_per_row. Interleaved DC scans use the iMCU
        // width (mcus_x); non-interleaved AC scans (single component) use
        // that component's width_in_blocks. Otherwise inherit the
        // user-provided MCU count unchanged.
        let restart_interval: u16 = if restart_in_rows > 0 {
            let mcus_per_row: usize = if scan.component_indices.len() > 1 {
                mcus_x
            } else {
                comp_wib[scan.component_indices[0]]
            };
            (restart_in_rows as usize)
                .saturating_mul(mcus_per_row)
                .min(65535) as u16
        } else {
            restart_interval
        };
        let is_dc_scan: bool = scan.ss == 0 && scan.se == 0;
        let is_first_scan: bool = scan.ah == 0;

        // Stack-allocate SOS component list (max 3 components in JPEG).
        let mut sos_comps: [(u8, u8, u8); 3] = [(0, 0, 0); 3];
        let sos_len: usize = scan.component_indices.len();
        for (idx, &ci) in scan.component_indices.iter().enumerate() {
            let comp_id: u8 = (ci + 1) as u8;
            let tbl_idx: u8 = if ci == 0 { 0 } else { 1 };
            let dc_tbl: u8 = if is_dc_scan { tbl_idx } else { 0 };
            let ac_tbl: u8 = if is_dc_scan { 0 } else { tbl_idx };
            sos_comps[idx] = (comp_id, dc_tbl, ac_tbl);
        }
        let sos_slice: &[(u8, u8, u8)] = &sos_comps[..sos_len];

        if is_dc_scan && is_first_scan {
            // DC first scan: gather DC symbol frequencies, generate optimal tables,
            // write DHT markers before SOS.
            let mut dc_luma_freq = [0u32; 257];
            let mut dc_chroma_freq = [0u32; 257];
            // Seed pseudo-symbol to ensure valid table even if no symbols appear
            dc_luma_freq[256] = 1;
            dc_chroma_freq[256] = 1;

            let mut prev_dc: [i16; 4] = [0i16; 4];
            let ri_dc_gather: u32 = restart_interval as u32;
            let mut mcu_idx_gather: u32 = 0;
            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if ri_dc_gather > 0
                        && mcu_idx_gather > 0
                        && mcu_idx_gather.is_multiple_of(ri_dc_gather)
                    {
                        // DC predictor reset at the restart boundary —
                        // mirror the encode loop so the diff symbol
                        // category histogram matches what's actually
                        // emitted under restart.
                        prev_dc = [0i16; 4];
                    }
                    for (scan_ci, &ci) in scan.component_indices.iter().enumerate() {
                        let layout = &comp_layouts[ci];
                        let freq = if ci == 0 {
                            &mut dc_luma_freq
                        } else {
                            &mut dc_chroma_freq
                        };
                        for bv in 0..layout.v_blocks {
                            for bh in 0..layout.h_blocks {
                                let bx: usize = mcu_x * layout.h_blocks + bh;
                                let by: usize = mcu_y * layout.v_blocks + bv;
                                let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];
                                let dc: i16 = block[0] >> scan.al;
                                let diff: i16 = dc.wrapping_sub(prev_dc[scan_ci]);
                                prev_dc[scan_ci] = dc;
                                crate::encode::huff_opt::gather_dc_symbol(diff, freq);
                            }
                        }
                    }
                    mcu_idx_gather = mcu_idx_gather.wrapping_add(1);
                }
            }

            let (dc_luma_bits, dc_luma_values) =
                crate::encode::huff_opt::gen_optimal_table(&dc_luma_freq);
            marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);

            if !is_grayscale {
                let (dc_chroma_bits, dc_chroma_values) =
                    crate::encode::huff_opt::gen_optimal_table(&dc_chroma_freq);
                marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);

                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                let dc_luma_table: HuffTable = build_huff_table(&dc_luma_bits, &dc_luma_values);
                let dc_chroma_table: HuffTable =
                    build_huff_table(&dc_chroma_bits, &dc_chroma_values);
                encode_progressive_dc_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &dc_luma_table,
                    &dc_chroma_table,
                    &mut output,
                    restart_interval,
                );
            } else {
                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                let dc_luma_table: HuffTable = build_huff_table(&dc_luma_bits, &dc_luma_values);
                let dc_chroma_table: HuffTable =
                    build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
                encode_progressive_dc_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &dc_luma_table,
                    &dc_chroma_table,
                    &mut output,
                    restart_interval,
                );
            }
        } else if is_dc_scan {
            // DC refinement scan (ah > 0): no DHT needed, just write SOS and encode.
            let dc_luma_table: HuffTable =
                build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
            let dc_chroma_table: HuffTable =
                build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
            if restart_interval != last_ri {
                if restart_interval > 0 {
                    marker_writer::write_dri(&mut output, restart_interval);
                }
                last_ri = restart_interval;
            }
            marker_writer::write_sos_progressive(
                &mut output,
                sos_slice,
                scan.ss,
                scan.se,
                scan.ah,
                scan.al,
            );
            encode_progressive_dc_scan(
                &coeff_bufs,
                &comp_layouts,
                scan,
                mcus_x,
                mcus_y,
                &dc_luma_table,
                &dc_chroma_table,
                &mut output,
                restart_interval,
            );
        } else {
            // AC scan (ss > 0): fused gather+encode with precomputed block data.
            // Eliminates actual_blocks Vec copy by iterating coeff_bufs with stride.
            let ci: usize = scan.component_indices[0];
            let mut ac_freq = [0u32; 257];
            ac_freq[256] = 1;
            let wib: usize = comp_wib[ci];
            let hib: usize = comp_hib[ci];
            let layout = &comp_layouts[ci];
            let stride: usize = layout.blocks_x;
            let num_blocks: usize = wib * hib;
            let ss_enc: usize = scan.ss as usize;
            let se_enc: usize = scan.se as usize;
            let band_len: usize = se_enc - ss_enc + 1;

            if scan.ah == 0 {
                // AC first scan: gather frequencies + precompute per-block data
                precomp_zerobits.clear();
                precomp_values.clear();
                precomp_diffs.clear();

                let mut eobrun_gather: u32 = 0;
                let ri_gather: u32 = restart_interval as u32;

                for by in 0..hib {
                    for bx in 0..wib {
                        // Restart boundary forces a flush of any pending
                        // EOBRUN — the encode loop emits EOBRUN before the
                        // RST marker, so the frequency gather must do the
                        // same or the optimised Huffman tables won't match
                        // the actual encoded stream.
                        let blk_idx: usize = by * wib + bx;
                        if ri_gather > 0
                            && blk_idx > 0
                            && (blk_idx as u32).is_multiple_of(ri_gather)
                            && eobrun_gather > 0
                        {
                            emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                            eobrun_gather = 0;
                        }

                        let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];

                        let mut zerobits: u64 = 0;
                        let mut values = [0u16; 64];
                        let mut diffs = [0u16; 64];

                        prepare_ac_first_coeffs(
                            block,
                            ss_enc,
                            band_len,
                            scan.al,
                            &mut zerobits,
                            &mut values,
                            &mut diffs,
                        );

                        precomp_zerobits.push(zerobits);
                        precomp_values.push(values);
                        precomp_diffs.push(diffs);

                        // Gather frequencies with EOBRUN batching
                        if zerobits == 0 {
                            eobrun_gather += 1;
                            if eobrun_gather == 0x7FFF {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                            }
                            continue;
                        }

                        if eobrun_gather > 0 {
                            emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                            eobrun_gather = 0;
                        }

                        let mut prev_pos: usize = 0;
                        let mut bits: u64 = zerobits;
                        while bits != 0 {
                            let pos: usize = bits.trailing_zeros() as usize;
                            bits &= bits - 1;

                            let mut zero_run: usize = pos - prev_pos;
                            while zero_run >= 16 {
                                ac_freq[0xF0] += 1;
                                zero_run -= 16;
                            }
                            let nbits: u8 = 16 - values[pos].leading_zeros() as u8;
                            let symbol: usize = (zero_run << 4) | (nbits as usize);
                            ac_freq[symbol] += 1;
                            prev_pos = pos + 1;
                        }

                        if prev_pos < band_len {
                            eobrun_gather += 1;
                            if eobrun_gather == 0x7FFF {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                            }
                        }
                    }
                }
                if eobrun_gather > 0 {
                    emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                }

                // Generate optimal table, write DHT + (DRI) + SOS
                let (ac_bits, ac_values) = crate::encode::huff_opt::gen_optimal_table(&ac_freq);
                let table_id: u8 = if ci == 0 { 0 } else { 1 };
                marker_writer::write_dht(&mut output, 1, table_id, &ac_bits, &ac_values);
                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                // Encode from precomputed data
                let ac_table: HuffTable = build_huff_table(&ac_bits, &ac_values);
                bit_writer.reset();
                let mut eobrun: u32 = 0;
                let ri_ac: u32 = restart_interval as u32;
                let mut rst_count: u8 = 0;

                for blk_idx in 0..num_blocks {
                    if ri_ac > 0 && blk_idx > 0 && (blk_idx as u32).is_multiple_of(ri_ac) {
                        // Flush pending EOBRUN, byte-pad bits, emit RST marker,
                        // reset EOBRUN per C jcphuff.c::emit_restart.
                        if eobrun > 0 {
                            emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                        }
                        bit_writer.flush_restart();
                        bit_writer.write_restart_marker(rst_count);
                        rst_count = (rst_count + 1) & 7;
                    }

                    let zerobits: u64 = precomp_zerobits[blk_idx];

                    if zerobits == 0 {
                        eobrun += 1;
                        if eobrun == 0x7FFF {
                            emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                        }
                        continue;
                    }

                    if eobrun > 0 {
                        emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                    }

                    let values = &precomp_values[blk_idx];
                    let diffs = &precomp_diffs[blk_idx];

                    // Pre-compute nbits for non-zero positions
                    let mut nbits_arr = [0u8; 64];
                    {
                        let mut bits: u64 = zerobits;
                        while bits != 0 {
                            let pos: usize = bits.trailing_zeros() as usize;
                            bits &= bits - 1;
                            nbits_arr[pos] = 16 - values[pos].leading_zeros() as u8;
                        }
                    }

                    let mut prev_pos: usize = 0;
                    let mut bits: u64 = zerobits;
                    while bits != 0 {
                        let pos: usize = bits.trailing_zeros() as usize;
                        bits &= bits - 1;

                        let mut zero_run: usize = pos - prev_pos;
                        while zero_run >= 16 {
                            bit_writer
                                .put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
                            zero_run -= 16;
                        }

                        let nbits: u8 = nbits_arr[pos];
                        let symbol: usize = (zero_run << 4) | (nbits as usize);
                        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
                        let huff_size: u8 = ac_table.ehufsi[symbol];
                        let mag_masked: u32 = diffs[pos] as u32 & ((1u32 << nbits) - 1);
                        let combined: u32 = (huff_code << nbits) | mag_masked;
                        bit_writer.put_bits(combined, huff_size + nbits);
                        prev_pos = pos + 1;
                    }

                    if prev_pos < band_len {
                        eobrun += 1;
                        if eobrun == 0x7FFF {
                            emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                        }
                    }
                }

                if eobrun > 0 {
                    emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                }
            } else {
                // AC refine scan: gather frequencies + precompute per-block data
                precomp_absvals.clear();
                precomp_signs.clear();
                precomp_eob.clear();

                let mut eobrun_gather: u32 = 0;
                let mut be: usize = 0;
                let ri_gather: u32 = restart_interval as u32;

                for by in 0..hib {
                    for bx in 0..wib {
                        // Restart boundary: flush any pending EOBRUN/BE so
                        // the gathered frequencies match what the encode
                        // loop will actually emit.
                        let blk_idx: usize = by * wib + bx;
                        if ri_gather > 0
                            && blk_idx > 0
                            && (blk_idx as u32).is_multiple_of(ri_gather)
                            && eobrun_gather > 0
                        {
                            emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                            eobrun_gather = 0;
                            be = 0;
                        }

                        let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];

                        let mut absvals = [0u16; 64];
                        let mut sign_bits = [0u16; 64];
                        let mut eob_pos: usize = 0;

                        prepare_ac_refine_coeffs(
                            block,
                            ss_enc,
                            band_len,
                            scan.al,
                            &mut absvals,
                            &mut sign_bits,
                            &mut eob_pos,
                        );

                        precomp_absvals.push(absvals);
                        precomp_signs.push(sign_bits);
                        precomp_eob.push(eob_pos);

                        // Gather frequencies with EOBRUN batching
                        let mut r: usize = 0;
                        let mut br: usize = 0;
                        let mut idx: usize = 0;

                        while idx < band_len {
                            let temp: u16 = absvals[idx];

                            if temp == 0 {
                                r += 1;
                                idx += 1;
                                continue;
                            }

                            while r > 15 && idx < eob_pos {
                                if eobrun_gather > 0 {
                                    emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                    eobrun_gather = 0;
                                    be = 0;
                                }
                                ac_freq[0xF0] += 1;
                                r -= 16;
                                br = 0;
                            }

                            if temp > 1 {
                                br += 1;
                                idx += 1;
                                continue;
                            }

                            if eobrun_gather > 0 {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                                be = 0;
                            }
                            let symbol: usize = (r << 4) | 1;
                            ac_freq[symbol] += 1;
                            r = 0;
                            br = 0;
                            idx += 1;
                        }

                        if r > 0 || br > 0 {
                            eobrun_gather += 1;
                            be += br;
                            if eobrun_gather == 0x7FFF || be > (MAX_CORR_BITS - 64 + 1) {
                                emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                                eobrun_gather = 0;
                                be = 0;
                            }
                        }
                    }
                }
                if eobrun_gather > 0 {
                    emit_eobrun_freq(eobrun_gather, &mut ac_freq);
                }

                // Generate optimal table, write DHT + (DRI) + SOS
                let (ac_bits, ac_values) = crate::encode::huff_opt::gen_optimal_table(&ac_freq);
                let table_id: u8 = if ci == 0 { 0 } else { 1 };
                marker_writer::write_dht(&mut output, 1, table_id, &ac_bits, &ac_values);
                if restart_interval != last_ri {
                    if restart_interval > 0 {
                        marker_writer::write_dri(&mut output, restart_interval);
                    }
                    last_ri = restart_interval;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    sos_slice,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                // Encode from precomputed data
                let ac_table: HuffTable = build_huff_table(&ac_bits, &ac_values);
                bit_writer.reset();
                let mut eobrun: u32 = 0;
                let mut corr_buffer: Vec<u8> = Vec::with_capacity(MAX_CORR_BITS);
                let ri_ac: u32 = restart_interval as u32;
                let mut rst_count: u8 = 0;

                for blk_idx in 0..num_blocks {
                    if ri_ac > 0 && blk_idx > 0 && (blk_idx as u32).is_multiple_of(ri_ac) {
                        // Flush pending EOBRUN+corr, byte-pad bits, emit RST.
                        // Per C jcphuff.c::emit_restart: clear EOBRUN AND BE
                        // (correction-bit count) on every restart.
                        if eobrun > 0 {
                            emit_eobrun_with_corr(
                                &ac_table,
                                &mut bit_writer,
                                &mut eobrun,
                                &mut corr_buffer,
                            );
                        }
                        bit_writer.flush_restart();
                        bit_writer.write_restart_marker(rst_count);
                        rst_count = (rst_count + 1) & 7;
                        corr_buffer.clear();
                    }

                    let absvals = &precomp_absvals[blk_idx];
                    let sign_bits = &precomp_signs[blk_idx];
                    let eob_pos: usize = precomp_eob[blk_idx];

                    let mut r: usize = 0;
                    let mut br_bits: [u8; 64] = [0u8; 64];
                    let mut br: usize = 0;
                    let mut idx: usize = 0;

                    while idx < band_len {
                        let temp: u16 = absvals[idx];

                        if temp == 0 {
                            r += 1;
                            idx += 1;
                            continue;
                        }

                        while r > 15 && idx < eob_pos {
                            emit_eobrun_with_corr(
                                &ac_table,
                                &mut bit_writer,
                                &mut eobrun,
                                &mut corr_buffer,
                            );
                            bit_writer
                                .put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
                            r -= 16;
                            emit_buffered_bits(&mut bit_writer, &br_bits[..br]);
                            br = 0;
                        }

                        if temp > 1 {
                            br_bits[br] = (temp & 1) as u8;
                            br += 1;
                            idx += 1;
                            continue;
                        }

                        // Newly nonzero (temp == 1)
                        emit_eobrun_with_corr(
                            &ac_table,
                            &mut bit_writer,
                            &mut eobrun,
                            &mut corr_buffer,
                        );

                        let symbol: usize = (r << 4) | 1;
                        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
                        let huff_size: u8 = ac_table.ehufsi[symbol];
                        let combined: u32 = (huff_code << 1) | sign_bits[idx] as u32;
                        bit_writer.put_bits(combined, huff_size + 1);

                        emit_buffered_bits(&mut bit_writer, &br_bits[..br]);
                        br = 0;
                        r = 0;
                        idx += 1;
                    }

                    if r > 0 || br > 0 {
                        eobrun += 1;
                        corr_buffer.extend_from_slice(&br_bits[..br]);
                        if eobrun == 0x7FFF || corr_buffer.len() > (MAX_CORR_BITS - 64 + 1) {
                            emit_eobrun_with_corr(
                                &ac_table,
                                &mut bit_writer,
                                &mut eobrun,
                                &mut corr_buffer,
                            );
                        }
                    }
                }

                if eobrun > 0 {
                    emit_eobrun_with_corr(
                        &ac_table,
                        &mut bit_writer,
                        &mut eobrun,
                        &mut corr_buffer,
                    );
                }
            }
            bit_writer.flush();
            output.extend_from_slice(bit_writer.data());
        }
    }

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Prepare AC first-scan coefficients: compute zerobits/values/diffs.
///
/// Dispatches to SSE2-vectorized path on x86_64, scalar fallback elsewhere.
#[inline]
fn prepare_ac_first_coeffs(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    zerobits: &mut u64,
    values: &mut [u16; 64],
    diffs: &mut [u16; 64],
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        prepare_ac_first_sse2(block, ss, band_len, al, zerobits, values, diffs);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        *zerobits = 0;
        for i in 0..band_len {
            let coeff: i16 = block[ss + i];
            if coeff == 0 {
                continue;
            }
            // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
            let coeff: i32 = coeff as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            if temp == 0 {
                continue;
            }
            values[i] = temp;
            diffs[i] = (sign_mask ^ (abs_coeff >> al)) as u16;
            *zerobits |= 1u64 << i;
        }
    }
}

/// SSE2-vectorized AC first-scan coefficient preparation.
///
/// Processes 8 i16 coefficients per iteration: abs via sign-mask,
/// point-transform shift, bitmap via cmpgt+movemask.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prepare_ac_first_sse2(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    zerobits: &mut u64,
    values: &mut [u16; 64],
    diffs: &mut [u16; 64],
) {
    use core::arch::x86_64::*;

    *zerobits = 0;
    let shift_amt: __m128i = _mm_cvtsi64_si128(al as i64);
    let zeros: __m128i = _mm_setzero_si128();

    let mut i: usize = 0;
    while i + 8 <= band_len {
        let raw: __m128i = _mm_loadu_si128(block.as_ptr().add(ss + i) as *const __m128i);

        // abs(coeff) via sign-mask
        let sign: __m128i = _mm_srai_epi16(raw, 15);
        let abs_val: __m128i = _mm_sub_epi16(_mm_xor_si128(raw, sign), sign);

        // Point-transform shift: temp = abs_val >> al
        let temp: __m128i = _mm_sra_epi16(abs_val, shift_amt);

        // Store values
        _mm_storeu_si128(values.as_mut_ptr().add(i) as *mut __m128i, temp);

        // Compute diffs: sign_mask ^ (abs_coeff >> al)
        let diff: __m128i = _mm_xor_si128(sign, temp);
        _mm_storeu_si128(diffs.as_mut_ptr().add(i) as *mut __m128i, diff);

        // Build bitmap: nonzero positions
        let nz: __m128i = _mm_cmpgt_epi16(temp, zeros);
        let packed: __m128i = _mm_packs_epi16(nz, zeros);
        let mask: u32 = _mm_movemask_epi8(packed) as u32;
        *zerobits |= (mask as u64 & 0xFF) << i;

        i += 8;
    }

    // Scalar tail for remaining coefficients
    while i < band_len {
        let coeff: i16 = *block.get_unchecked(ss + i);
        if coeff != 0 {
            // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
            let coeff: i32 = coeff as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            if temp != 0 {
                *values.get_unchecked_mut(i) = temp;
                *diffs.get_unchecked_mut(i) = (sign_mask ^ (abs_coeff >> al)) as u16;
                *zerobits |= 1u64 << i;
            }
        }
        i += 1;
    }
}

/// Prepare AC refine-scan coefficients: compute absvals/sign_bits/eob_pos.
///
/// Dispatches to SSE2-vectorized path on x86_64, scalar fallback elsewhere.
#[inline]
fn prepare_ac_refine_coeffs(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    absvals: &mut [u16; 64],
    sign_bits: &mut [u16; 64],
    eob_pos: &mut usize,
) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        prepare_ac_refine_sse2(block, ss, band_len, al, absvals, sign_bits, eob_pos);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        *eob_pos = 0;
        for i in 0..band_len {
            let coeff: i32 = block[ss + i] as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            absvals[i] = temp;
            sign_bits[i] = (sign_mask as u16).wrapping_add(1);
            if temp == 1 {
                *eob_pos = i + 1;
            }
        }
    }
}

/// SSE2-vectorized AC refine-scan coefficient preparation.
///
/// Processes 8 i16 coefficients per iteration: abs via sign-mask,
/// point-transform shift, sign extraction, eob_pos tracking.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prepare_ac_refine_sse2(
    block: &[i16; 64],
    ss: usize,
    band_len: usize,
    al: u8,
    absvals: &mut [u16; 64],
    sign_bits: &mut [u16; 64],
    eob_pos: &mut usize,
) {
    use core::arch::x86_64::*;

    *eob_pos = 0;
    let shift_amt: __m128i = _mm_cvtsi64_si128(al as i64);
    let ones: __m128i = _mm_set1_epi16(1);

    let mut i: usize = 0;
    while i + 8 <= band_len {
        let raw: __m128i = _mm_loadu_si128(block.as_ptr().add(ss + i) as *const __m128i);

        // abs(coeff)
        let sign: __m128i = _mm_srai_epi16(raw, 15);
        let abs_val: __m128i = _mm_sub_epi16(_mm_xor_si128(raw, sign), sign);

        // temp = abs_val >> al
        let temp: __m128i = _mm_sra_epi16(abs_val, shift_amt);
        _mm_storeu_si128(absvals.as_mut_ptr().add(i) as *mut __m128i, temp);

        // sign_bits = (sign_mask as u16) + 1 = 0 for negative, 1 for positive/zero
        let sign_out: __m128i = _mm_add_epi16(sign, ones);
        _mm_storeu_si128(sign_bits.as_mut_ptr().add(i) as *mut __m128i, sign_out);

        // Track eob_pos: find positions where temp == 1
        let eq_one: __m128i = _mm_cmpeq_epi16(temp, ones);
        let mask: u32 = _mm_movemask_epi8(_mm_packs_epi16(eq_one, _mm_setzero_si128())) as u32;
        if mask != 0 {
            // Highest set bit position in the 8-bit mask
            let highest: u32 = 7 - (mask as u8).leading_zeros();
            let pos: usize = i + highest as usize + 1;
            if pos > *eob_pos {
                *eob_pos = pos;
            }
        }

        i += 8;
    }

    // Scalar tail
    while i < band_len {
        let coeff: i32 = *block.get_unchecked(ss + i) as i32;
        let sign_mask: i32 = coeff >> 31;
        let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
        let temp: u16 = (abs_coeff >> al) as u16;
        *absvals.get_unchecked_mut(i) = temp;
        *sign_bits.get_unchecked_mut(i) = (sign_mask as u16).wrapping_add(1);
        if temp == 1 {
            *eob_pos = i + 1;
        }
        i += 1;
    }
}

/// Gather AC symbol frequencies for a progressive AC scan (ah==0, first scan).
///
/// Mirrors the zero-run / EOB logic from `encode_ac_first_block` to produce
/// accurate symbol frequency counts for optimal Huffman table generation.
/// `ss` and `se` are the spectral band limits (1..=63); `al` is the point transform.
#[allow(dead_code)]
fn gather_progressive_ac_freq(blocks: &[[i16; 64]], ss: u8, se: u8, al: u8, freq: &mut [u32; 257]) {
    let ss_usize: usize = ss as usize;
    let se_usize: usize = se as usize;
    let band_len: usize = se_usize - ss_usize + 1;
    let mut eobrun: u32 = 0;

    for block in blocks.iter() {
        let mut zerobits: u64 = 0;
        let mut values = [0u16; 64];

        for i in 0..band_len {
            let coeff: i16 = block[ss_usize + i];
            if coeff == 0 {
                continue;
            }
            // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
            let coeff: i32 = coeff as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            if temp == 0 {
                continue;
            }
            values[i] = temp;
            zerobits |= 1u64 << i;
        }

        if zerobits == 0 {
            // Accumulate EOBRUN instead of emitting individual EOB
            eobrun += 1;
            if eobrun == 0x7FFF {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
            }
            continue;
        }

        // Flush pending EOBRUN before encoding nonzero coefficients
        if eobrun > 0 {
            emit_eobrun_freq(eobrun, freq);
            eobrun = 0;
        }

        let mut prev_pos: usize = 0;
        let mut bits = zerobits;
        while bits != 0 {
            let pos: usize = bits.trailing_zeros() as usize;
            bits &= bits - 1;

            let mut zero_run: usize = pos - prev_pos;
            while zero_run >= 16 {
                freq[0xF0] += 1; // ZRL
                zero_run -= 16;
            }
            let nbits: u8 = 16 - values[pos].leading_zeros() as u8;
            let symbol: usize = (zero_run << 4) | (nbits as usize);
            freq[symbol] += 1;
            prev_pos = pos + 1;
        }

        if prev_pos < band_len {
            // Trailing zeros → start EOBRUN
            eobrun += 1;
            if eobrun == 0x7FFF {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
            }
        }
    }

    // Flush any remaining EOBRUN at end of scan
    if eobrun > 0 {
        emit_eobrun_freq(eobrun, freq);
    }
}

/// Emit EOBRUN symbol frequency: nbits = JPEG_NBITS(eobrun) - 1, symbol = nbits << 4.
/// Matches C libjpeg-turbo's emit_eobrun in jcphuff.c.
fn emit_eobrun_freq(eobrun: u32, freq: &mut [u32; 257]) {
    let nbits: u8 = (32 - eobrun.leading_zeros()) as u8 - 1; // JPEG_NBITS_NONZERO - 1
    let symbol: usize = (nbits as usize) << 4;
    freq[symbol] += 1;
}

/// Gather AC symbol frequencies for a progressive AC refinement scan (ah > 0).
///
/// Mirrors the symbol-emission logic from `encode_ac_refine_block` with cross-block
/// EOBRUN batching: only ZRL (0xF0), EOB (batched via EOBRUN), and `(run, 1)`
/// symbols are counted. EOBRUN batching affects which EOB symbol (nbits << 4)
/// is emitted, so frequencies must match the encoder exactly.
#[allow(dead_code)]
fn gather_progressive_ac_refine_freq(
    blocks: &[[i16; 64]],
    ss: u8,
    se: u8,
    al: u8,
    freq: &mut [u32; 257],
) {
    let ss_usize: usize = ss as usize;
    let se_usize: usize = se as usize;
    let band_len: usize = se_usize - ss_usize + 1;

    let mut eobrun: u32 = 0;
    let mut be: usize = 0; // count of cross-block buffered correction bits

    for block in blocks.iter() {
        let mut absvals = [0u16; 64];
        let mut eob_pos: usize = 0;

        for i in 0..band_len {
            let coeff: i32 = block[ss_usize + i] as i32;
            let sign_mask: i32 = coeff >> 31;
            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
            let temp: u16 = (abs_coeff >> al) as u16;
            absvals[i] = temp;
            if temp == 1 {
                eob_pos = i + 1;
            }
        }

        let mut r: usize = 0;
        let mut br: usize = 0; // this block's correction bit count
        let mut idx: usize = 0;

        while idx < band_len {
            let temp: u16 = absvals[idx];

            if temp == 0 {
                r += 1;
                idx += 1;
                continue;
            }

            while r > 15 && idx < eob_pos {
                // Flush EOBRUN before ZRL
                if eobrun > 0 {
                    emit_eobrun_freq(eobrun, freq);
                    eobrun = 0;
                    be = 0;
                }
                freq[0xF0] += 1;
                r -= 16;
                br = 0;
            }

            if temp > 1 {
                br += 1;
                idx += 1;
                continue;
            }

            // Newly nonzero: flush EOBRUN before emitting symbol
            if eobrun > 0 {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
                be = 0;
            }
            let symbol: usize = (r << 4) | 1;
            freq[symbol] += 1;
            r = 0;
            br = 0;
            idx += 1;
        }

        // Trailing zeroes or correction bits → accumulate EOBRUN
        if r > 0 || br > 0 {
            eobrun += 1;
            be += br;
            if eobrun == 0x7FFF || be > (MAX_CORR_BITS - 64 + 1) {
                emit_eobrun_freq(eobrun, freq);
                eobrun = 0;
                be = 0;
            }
        }
    }

    // Flush trailing EOBRUN
    if eobrun > 0 {
        emit_eobrun_freq(eobrun, freq);
    }
}

/// Compress with arithmetic entropy coding (SOF9).
///
/// Uses the QM-coder binary arithmetic encoder instead of Huffman coding.
#[allow(clippy::too_many_arguments)]
pub fn compress_arithmetic(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
    restart_interval: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    let enc_simd = crate::simd::detect_encoder();

    // Generate quantization tables. ifast pre-applies AA&N scaling so its
    // divisors fold the AA&N constants in (paired with `fdct_ifast_raw`);
    // islow/float keep the simple `quant * 8` divisors. Float routes the
    // per-coefficient `1 / (q · aan_row · aan_col · 8)` value through
    // `QuantDivisors::float_divisors`.
    let (luma_quant, chroma_quant) = resolve_quant_tables(custom_quant, quality);
    let luma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&luma_quant)
    } else {
        scale_quant_for_fdct(&luma_quant)
    };
    let chroma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&chroma_quant)
    } else {
        scale_quant_for_fdct(&chroma_quant)
    };

    // MCU dimensions
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);
    let padded_w: usize = mcus_x * mcu_w;
    let padded_h: usize = mcus_y * mcu_h;

    // Color convert with MCU-aligned padding
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr_padded(
        pixels,
        width,
        height,
        padded_w,
        padded_h,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
        mcu_h / 8,
    )?;

    let original_width: usize = width;
    let original_height: usize = height;
    let width: usize = padded_w;
    let height: usize = padded_h;

    // Dummy block detection
    let y_width_in_blocks: usize = original_width.div_ceil(8);
    let y_height_in_blocks: usize = original_height.div_ceil(8);

    // FDCT + quantize all blocks. Dispatch on dct_method so `gather_block`
    // routes through the matching scalar routine instead of always running
    // the SIMD islow kernel internally — see the `is_ifast`/`is_float`
    // pointer-compare bypass at the top of `gather_block`.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };
    let mut all_blocks: Vec<[i16; 64]> = Vec::new();
    let mut prev_dc_y_gather: i16 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0 = mcu_col * mcu_w;
            let y0 = mcu_row * mcu_h;

            if is_grayscale {
                let q = gather_block(
                    &y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    &luma_divisors,
                    fdct_quantize_fn,
                );
                all_blocks.push(q);
            } else {
                match subsampling {
                    Subsampling::S444 | Subsampling::Unknown => {
                        for (plane, divisors) in [
                            (&y_plane, &luma_divisors),
                            (&cb_plane, &chroma_divisors),
                            (&cr_plane, &chroma_divisors),
                        ] {
                            let q = gather_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S422 => {
                        for dx in [0usize, 8] {
                            if is_y_dummy(x0 + dx, y0, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                1,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S420 => {
                        for (dx, dy) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
                            if is_y_dummy(x0 + dx, y0 + dy, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S440 => {
                        for dy in [0usize, 8] {
                            if is_y_dummy(x0, y0 + dy, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0,
                                    y0 + dy,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                1,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S411 => {
                        for dx in [0usize, 8, 16, 24] {
                            if is_y_dummy(x0 + dx, y0, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                            } else {
                                let q = gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                prev_dc_y_gather = q[0];
                                all_blocks.push(q);
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                4,
                                1,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S441 => {
                        // 4 Y blocks vertically
                        for dy in [0usize, 8, 16, 24] {
                            if is_y_dummy(x0, y0 + dy, y_width_in_blocks, y_height_in_blocks) {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y_gather;
                                all_blocks.push(dummy);
                                continue;
                            }
                            let mut block = [0i16; 64];
                            extract_block(&y_plane, width, height, x0, y0 + dy, &mut block);
                            let mut q = [0i16; 64];
                            fdct_quantize_fn(&mut block, &luma_divisors, &mut q);
                            prev_dc_y_gather = q[0];
                            all_blocks.push(q);
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let mut block = [0i16; 64];
                            downsample_chroma_block(plane, width, height, x0, y0, 1, 4, &mut block);
                            let mut q = [0i16; 64];
                            fdct_quantize_fn(&mut block, &chroma_divisors, &mut q);
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S410 => {
                        // 4 Y horizontal × 2 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8] {
                            for dx in [0usize, 8, 16, 24] {
                                if is_y_dummy(
                                    x0 + dx,
                                    y0 + dy,
                                    y_width_in_blocks,
                                    y_height_in_blocks,
                                ) {
                                    let mut dummy = [0i16; 64];
                                    dummy[0] = prev_dc_y_gather;
                                    all_blocks.push(dummy);
                                } else {
                                    let q = gather_block(
                                        &y_plane,
                                        width,
                                        height,
                                        x0 + dx,
                                        y0 + dy,
                                        &luma_divisors,
                                        fdct_quantize_fn,
                                    );
                                    prev_dc_y_gather = q[0];
                                    all_blocks.push(q);
                                }
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                4,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                    Subsampling::S24 => {
                        // 2 Y horizontal × 4 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8, 16, 24] {
                            for dx in [0usize, 8] {
                                if is_y_dummy(
                                    x0 + dx,
                                    y0 + dy,
                                    y_width_in_blocks,
                                    y_height_in_blocks,
                                ) {
                                    let mut dummy = [0i16; 64];
                                    dummy[0] = prev_dc_y_gather;
                                    all_blocks.push(dummy);
                                } else {
                                    let q = gather_block(
                                        &y_plane,
                                        width,
                                        height,
                                        x0 + dx,
                                        y0 + dy,
                                        &luma_divisors,
                                        fdct_quantize_fn,
                                    );
                                    prev_dc_y_gather = q[0];
                                    all_blocks.push(q);
                                }
                            }
                        }
                        for plane in [&cb_plane, &cr_plane] {
                            let q = gather_downsampled_block(
                                plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                4,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            );
                            all_blocks.push(q);
                        }
                    }
                }
            }
        }
    }

    // Arithmetic encode all blocks
    let mut arith_enc = ArithEncoder::new(width * height);
    let mut block_idx = 0;
    let ri_arith: u32 = restart_interval as u32;
    let mut mcu_count_arith: u32 = 0;
    let mut rst_count_arith: u8 = 0;

    for _mcu_row in 0..mcus_y {
        for _mcu_col in 0..mcus_x {
            if ri_arith > 0 && mcu_count_arith > 0 && mcu_count_arith.is_multiple_of(ri_arith) {
                // ArithEncoder::emit_restart finishes the coder, writes
                // FF/RST0+n, then re-inits coder + DC/AC stats per
                // jcarith.c::emit_restart.
                arith_enc.emit_restart(rst_count_arith);
                rst_count_arith = (rst_count_arith + 1) & 7;
            }
            if is_grayscale {
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 0, 0);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], 0);
                block_idx += 1;
            } else {
                let y_blocks = match subsampling {
                    Subsampling::S444 | Subsampling::Unknown => 1,
                    Subsampling::S422 => 2,
                    Subsampling::S420 => 4,
                    Subsampling::S440 => 2,
                    Subsampling::S411 | Subsampling::S441 => 4,
                    Subsampling::S410 | Subsampling::S24 => 8,
                };
                for _ in 0..y_blocks {
                    arith_enc.encode_dc_sequential(&all_blocks[block_idx], 0, 0);
                    arith_enc.encode_ac_sequential(&all_blocks[block_idx], 0);
                    block_idx += 1;
                }
                // Cb
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 1, 1);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], 1);
                block_idx += 1;
                // Cr
                arith_enc.encode_dc_sequential(&all_blocks[block_idx], 2, 1);
                arith_enc.encode_ac_sequential(&all_blocks[block_idx], 1);
                block_idx += 1;
            }
            mcu_count_arith = mcu_count_arith.wrapping_add(1);
        }
    }

    arith_enc.finish();

    // Assemble output
    let mut output = Vec::with_capacity(arith_enc.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // SOF9 (arithmetic sequential)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof9(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof9(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
    }

    // DAC marker
    let dc_params = [(0u8, 1u8), (0, 1)];
    let ac_params = [5u8, 5];
    let num_dc = if is_grayscale { 1 } else { 2 };
    let num_ac = if is_grayscale { 1 } else { 2 };
    marker_writer::write_dac(&mut output, num_dc, &dc_params, num_ac, &ac_params);

    // DRI marker — emitted from `write_scan_header` in C
    // (jcmarker.c::emit_dri), i.e. between DAC and SOS.
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // SOS
    if is_grayscale {
        let scan_components = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components = vec![(1, 0, 0), (2, 1, 1), (3, 1, 1)];
        marker_writer::write_sos(&mut output, &scan_components);
    }

    // Entropy-coded data
    output.extend_from_slice(arith_enc.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Compress with arithmetic progressive encoding (SOF10).
///
/// Combines progressive multi-scan encoding with arithmetic entropy coding.
/// Buffers all DCT coefficients, then encodes across multiple scans using
/// a standard scan progression script with ArithEncoder.
#[allow(clippy::too_many_arguments)]
pub fn compress_arithmetic_progressive(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
    restart_interval: u16,
    restart_in_rows: u16,
    custom_quant: Option<&[Option<[u16; 64]>; 4]>,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;
    use crate::encode::progressive::simple_progression;

    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale: bool = pixel_format == PixelFormat::Grayscale;
    let num_components: usize = if is_grayscale { 1 } else { 3 };

    let enc_simd = crate::simd::detect_encoder();

    let (luma_quant, chroma_quant) = resolve_quant_tables(custom_quant, quality);
    let luma_divisors: QuantDivisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&luma_quant)
    } else {
        scale_quant_for_fdct(&luma_quant)
    };
    let chroma_divisors: QuantDivisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&chroma_quant)
    } else {
        scale_quant_for_fdct(&chroma_quant)
    };

    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    let (mcu_w, mcu_h): (usize, usize) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);

    // Compute per-component block dimensions
    let (h_samp, v_samp): (usize, usize) = if is_grayscale {
        (1, 1)
    } else {
        let (h, v) = subsampling.sampling_factors();
        (h as usize, v as usize)
    };

    let comp_layouts: Vec<CompLayout> = if is_grayscale {
        vec![CompLayout {
            blocks_x: mcus_x,
            blocks_y: mcus_y,
            h_blocks: 1,
            v_blocks: 1,
        }]
    } else {
        vec![
            CompLayout {
                blocks_x: mcus_x * h_samp,
                blocks_y: mcus_y * v_samp,
                h_blocks: h_samp,
                v_blocks: v_samp,
            },
            CompLayout {
                blocks_x: mcus_x,
                blocks_y: mcus_y,
                h_blocks: 1,
                v_blocks: 1,
            },
            CompLayout {
                blocks_x: mcus_x,
                blocks_y: mcus_y,
                h_blocks: 1,
                v_blocks: 1,
            },
        ]
    };

    // Buffer all quantized coefficients per component
    let mut coeff_bufs: Vec<Vec<[i16; 64]>> = comp_layouts
        .iter()
        .map(|cl| vec![[0i16; 64]; cl.blocks_x * cl.blocks_y])
        .collect();

    // FDCT + quantize all blocks into coefficient buffers.
    // Track per-component prev_dc to honor the C `jccoefct.c:178-199` dummy
    // block rule: Y blocks whose 8×8 origin sits outside the original image
    // must encode as DC=prev_dc, AC=0 (so the DC diff is zero) instead of
    // the replicated-edge content `extract_block` would otherwise produce.
    // Failing to do this drops a coefficient on the wrong side of every
    // arithmetic conditioning bucket and breaks byte-parity with cjpeg for
    // any subsampled MCU (samp422/420/440/411/441/410/24) on images whose
    // dimensions don't divide evenly by the MCU stride.
    // Dispatch on dct_method so the in-place FDCT routes through the matching
    // scalar routine for IFAST/Float (instead of always running the SIMD islow
    // kernel). Same pattern as compress_arithmetic / compress_optimized.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };
    let y_width_in_blocks: usize = width.div_ceil(8);
    let y_height_in_blocks: usize = height.div_ceil(8);
    // Per-component actual block counts (`width_in_blocks * height_in_blocks`
    // in the libjpeg-turbo sense). Non-interleaved AC scans iterate this
    // count, NOT the MCU-padded `comp_layouts[*].blocks_x * blocks_y`.
    // Including the right/bottom dummy blocks in a single-component scan
    // makes the encoder run extra "EOB at start" emissions that cjpeg never
    // emits, so the entropy stream length and any downstream adaptive bin
    // state diverge from C output.
    let comp_wib: Vec<usize> = if is_grayscale {
        vec![width.div_ceil(8)]
    } else {
        vec![
            width.div_ceil(8),
            width.div_ceil(h_samp * 8),
            width.div_ceil(h_samp * 8),
        ]
    };
    let comp_hib: Vec<usize> = if is_grayscale {
        vec![height.div_ceil(8)]
    } else {
        vec![
            height.div_ceil(8),
            height.div_ceil(v_samp * 8),
            height.div_ceil(v_samp * 8),
        ]
    };
    let mut prev_dc_y_gather: i16 = 0;
    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            let x0: usize = mcu_x * mcu_w;
            let y0: usize = mcu_y * mcu_h;

            if is_grayscale {
                let bx: usize = mcu_x;
                let by: usize = mcu_y;
                let mut block = [0i16; 64];
                extract_block(&y_plane, width, height, x0, y0, &mut block);
                fdct_quantize_fn(
                    &mut block,
                    &luma_divisors,
                    &mut coeff_bufs[0][by * mcus_x + bx],
                );
            } else {
                // Y blocks
                for bv in 0..v_samp {
                    for bh in 0..h_samp {
                        let bx: usize = mcu_x * h_samp + bh;
                        let by: usize = mcu_y * v_samp + bv;
                        let blocks_x: usize = comp_layouts[0].blocks_x;
                        if is_y_dummy(
                            x0 + bh * 8,
                            y0 + bv * 8,
                            y_width_in_blocks,
                            y_height_in_blocks,
                        ) {
                            let mut dummy = [0i16; 64];
                            dummy[0] = prev_dc_y_gather;
                            coeff_bufs[0][by * blocks_x + bx] = dummy;
                            continue;
                        }
                        let mut block = [0i16; 64];
                        extract_block(
                            &y_plane,
                            width,
                            height,
                            x0 + bh * 8,
                            y0 + bv * 8,
                            &mut block,
                        );
                        fdct_quantize_fn(
                            &mut block,
                            &luma_divisors,
                            &mut coeff_bufs[0][by * blocks_x + bx],
                        );
                        prev_dc_y_gather = coeff_bufs[0][by * blocks_x + bx][0];
                    }
                }
                // Cb block — use real h_samp/v_samp; clamping to {1,2}
                // would mismatch the SOF's 1/4-chroma claim for S411/S441/
                // S410/S24 and corrupt the decoded image (P2-11).
                {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    let mut block = [0i16; 64];
                    if h_samp == 1 && v_samp == 1 {
                        extract_block(&cb_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cb_plane, width, height, x0, y0, h_samp, v_samp, &mut block,
                        );
                    }
                    fdct_quantize_fn(
                        &mut block,
                        &chroma_divisors,
                        &mut coeff_bufs[1][by * mcus_x + bx],
                    );
                }
                // Cr block — same fix as Cb above.
                {
                    let bx: usize = mcu_x;
                    let by: usize = mcu_y;
                    let mut block = [0i16; 64];
                    if h_samp == 1 && v_samp == 1 {
                        extract_block(&cr_plane, width, height, x0, y0, &mut block);
                    } else {
                        downsample_chroma_block(
                            &cr_plane, width, height, x0, y0, h_samp, v_samp, &mut block,
                        );
                    }
                    fdct_quantize_fn(
                        &mut block,
                        &chroma_divisors,
                        &mut coeff_bufs[2][by * mcus_x + bx],
                    );
                }
            }
        }
    }

    // Generate scan progression
    let scans = simple_progression(num_components);

    // Assemble output
    let mut output: Vec<u8> = Vec::with_capacity(width * height * 2);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // SOF10 (arithmetic progressive)
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof10(&mut output, width as u16, height as u16, &components);
    } else {
        let components = vec![
            (1, h_samp as u8, v_samp as u8, 0),
            (2, 1, 1, 1),
            (3, 1, 1, 1),
        ];
        marker_writer::write_sof10(&mut output, width as u16, height as u16, &components);
    }

    // Default arithmetic conditioning parameters (C `jcparam.c`):
    //   DC: L=0 U=1 → packed byte 0x10
    //   AC: Kx=5
    use crate::decode::arithmetic::NUM_ARITH_TBLS;
    let mut dc_params_full: [(u8, u8); NUM_ARITH_TBLS] = [(0u8, 1u8); NUM_ARITH_TBLS];
    let mut ac_params_full: [u8; NUM_ARITH_TBLS] = [5u8; NUM_ARITH_TBLS];
    let _ = (&mut dc_params_full, &mut ac_params_full);

    // Encode each scan with arithmetic coding
    let mut arith_enc: ArithEncoder = ArithEncoder::new(width * height / 4);
    // Track last-emitted DRI to suppress redundant markers when the per-scan
    // restart_interval doesn't change — matches `jcmarker.c::write_scan_header`.
    let mut last_ri: u16 = 0;

    for scan in &scans {
        // Reset encoder state for each scan
        arith_enc.reset();

        // Per-scan restart_interval: when `restart_in_rows` is set, derive
        // it from this scan's MCUs_per_row. Interleaved DC scans use the
        // iMCU width (mcus_x); non-interleaved AC scans (single component)
        // use that component's width_in_blocks. Otherwise inherit the
        // user-provided MCU count unchanged.
        let scan_ri: u16 = if restart_in_rows > 0 {
            let mcus_per_row: usize = if scan.component_indices.len() > 1 {
                mcus_x
            } else {
                comp_wib[scan.component_indices[0]]
            };
            (restart_in_rows as usize)
                .saturating_mul(mcus_per_row)
                .min(65535) as u16
        } else {
            restart_interval
        };

        // Build SOS component list.
        // Per JPEG spec: DC-only scans (Ss=0) set Ta=0, AC-only scans (Ss>0) set Td=0.
        let is_dc_scan_arith: bool = scan.ss == 0;
        let sos_comps: Vec<(u8, u8, u8)> = scan
            .component_indices
            .iter()
            .map(|&ci| {
                let comp_id: u8 = (ci + 1) as u8;
                let tbl_idx: u8 = if ci == 0 { 0 } else { 1 };
                let dc_tbl: u8 = if is_dc_scan_arith { tbl_idx } else { 0 };
                let ac_tbl: u8 = if is_dc_scan_arith { 0 } else { tbl_idx };
                (comp_id, dc_tbl, ac_tbl)
            })
            .collect();

        // C jcmarker.c::emit_dac (called from write_scan_header) — emit a
        // DAC per scan with only the conditioning entries used by THIS scan:
        // DC tables when the scan starts at Ss=0 with Ah=0 (initial DC), AC
        // tables when Se>0. We replicate that here so each progressive scan
        // header carries the minimal-and-correct DAC, matching cjpeg byte
        // for byte even though "duplicate DAC for repeated tables" is per-
        // spec wasted bytes.
        let mut dc_in_use: [bool; NUM_ARITH_TBLS] = [false; NUM_ARITH_TBLS];
        let mut ac_in_use: [bool; NUM_ARITH_TBLS] = [false; NUM_ARITH_TBLS];
        let need_dc: bool = scan.ss == 0 && scan.ah == 0;
        let need_ac: bool = scan.se > 0;
        for (comp_id, dc_tbl, ac_tbl) in &sos_comps {
            let _ = comp_id;
            if need_dc {
                dc_in_use[(*dc_tbl as usize).min(NUM_ARITH_TBLS - 1)] = true;
            }
            if need_ac {
                ac_in_use[(*ac_tbl as usize).min(NUM_ARITH_TBLS - 1)] = true;
            }
        }
        marker_writer::write_dac_selected(
            &mut output,
            &dc_in_use,
            &dc_params_full,
            &ac_in_use,
            &ac_params_full,
        );

        if scan_ri != last_ri {
            if scan_ri > 0 {
                marker_writer::write_dri(&mut output, scan_ri);
            }
            last_ri = scan_ri;
        }

        marker_writer::write_sos_progressive(
            &mut output,
            &sos_comps,
            scan.ss,
            scan.se,
            scan.ah,
            scan.al,
        );

        let is_dc_scan: bool = scan.ss == 0 && scan.se == 0;

        if is_dc_scan {
            if scan.ah == 0 {
                // DC first scan
                encode_arith_dc_first_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &mut arith_enc,
                    scan_ri,
                );
            } else {
                // DC refine scan
                encode_arith_dc_refine_scan(
                    &coeff_bufs,
                    &comp_layouts,
                    scan,
                    mcus_x,
                    mcus_y,
                    &mut arith_enc,
                    scan_ri,
                );
            }
        } else if scan.ah == 0 {
            // AC first scan
            encode_arith_ac_first_scan(
                &coeff_bufs,
                &comp_layouts,
                &comp_wib,
                &comp_hib,
                scan,
                &mut arith_enc,
                scan_ri,
            );
        } else {
            // AC refine scan
            encode_arith_ac_refine_scan(
                &coeff_bufs,
                &comp_layouts,
                &comp_wib,
                &comp_hib,
                scan,
                &mut arith_enc,
                scan_ri,
            );
        }

        arith_enc.finish();
        output.extend_from_slice(arith_enc.data());
    }

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Encode arithmetic DC first scan (Ah=0) across all MCUs.
#[allow(clippy::too_many_arguments)]
fn encode_arith_dc_first_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
) {
    let al: u8 = scan.al;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
            for &ci in &scan.component_indices {
                let layout: &CompLayout = &comp_layouts[ci];
                let dc_tbl: usize = if ci == 0 { 0 } else { 1 };

                for bv in 0..layout.v_blocks {
                    for bh in 0..layout.h_blocks {
                        let bx: usize = mcu_x * layout.h_blocks + bh;
                        let by: usize = mcu_y * layout.v_blocks + bv;
                        let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                        arith_enc.encode_dc_first(block, ci, dc_tbl, al);
                    }
                }
            }
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}

/// Encode arithmetic DC refine scan (Ah!=0) across all MCUs.
#[allow(clippy::too_many_arguments)]
fn encode_arith_dc_refine_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
) {
    let al: u8 = scan.al;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
            for &ci in &scan.component_indices {
                let layout: &CompLayout = &comp_layouts[ci];

                for bv in 0..layout.v_blocks {
                    for bh in 0..layout.h_blocks {
                        let bx: usize = mcu_x * layout.h_blocks + bh;
                        let by: usize = mcu_y * layout.v_blocks + bv;
                        let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                        arith_enc.encode_dc_refine(block, al);
                    }
                }
            }
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}

/// Encode arithmetic AC first scan (Ah=0, single component).
///
/// Iterates the component's raster blocks (`width_in_blocks * height_in_blocks`)
/// rather than MCU-padded `blocks_x * blocks_y`. C `jccoefct.c::start_iMCU_row`
/// drops MCU-multi-block layout for non-interleaved scans (`comps_in_scan == 1`)
/// — each "MCU" is a single block, and `MCUs_per_row` equals the component's
/// `width_in_blocks` rather than `MCUs_per_row(image)`. Iterating the padded
/// `blocks_x` would re-encode the right-edge dummy fillers (which have AC=0)
/// as extra "EOB at start" emissions cjpeg never produces, drifting the
/// arithmetic coder state and breaking byte-parity.
#[allow(clippy::too_many_arguments)]
fn encode_arith_ac_first_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    comp_wib: &[usize],
    comp_hib: &[usize],
    scan: &crate::encode::progressive::ProgressiveScan,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
) {
    let ci: usize = scan.component_indices[0]; // AC scans are single-component
    let layout: &CompLayout = &comp_layouts[ci];
    let ac_tbl: usize = if ci == 0 { 0 } else { 1 };
    let wib: usize = comp_wib[ci];
    let hib: usize = comp_hib[ci];
    let stride: usize = layout.blocks_x;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for by in 0..hib {
        for bx in 0..wib {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
            let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];
            arith_enc.encode_ac_first(block, ac_tbl, scan.ss, scan.se, scan.al);
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}

/// Encode arithmetic AC refine scan (Ah!=0, single component).
///
/// Same raster iteration as `encode_arith_ac_first_scan` — see that comment.
#[allow(clippy::too_many_arguments)]
fn encode_arith_ac_refine_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    comp_wib: &[usize],
    comp_hib: &[usize],
    scan: &crate::encode::progressive::ProgressiveScan,
    arith_enc: &mut crate::encode::arithmetic::ArithEncoder,
    restart_interval: u16,
) {
    let ci: usize = scan.component_indices[0]; // AC scans are single-component
    let layout: &CompLayout = &comp_layouts[ci];
    let ac_tbl: usize = if ci == 0 { 0 } else { 1 };
    let wib: usize = comp_wib[ci];
    let hib: usize = comp_hib[ci];
    let stride: usize = layout.blocks_x;
    let ri: u32 = restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut rst_count: u8 = 0;

    for by in 0..hib {
        for bx in 0..wib {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(rst_count);
                rst_count = (rst_count + 1) & 7;
            }
            let block: &[i16; 64] = &coeff_bufs[ci][by * stride + bx];
            arith_enc.encode_ac_refine(block, ac_tbl, scan.ss, scan.se, scan.al, scan.ah);
            mcu_count = mcu_count.wrapping_add(1);
        }
    }
}

/// Encode a progressive DC scan directly into the output Vec.
///
/// Uses hoisted BitWriter state (local_put_bits) to avoid struct field
/// store-reload overhead per block. Writes directly into the output Vec,
/// eliminating the intermediate BitWriter allocation and final extend_from_slice.
#[allow(clippy::too_many_arguments)]
fn encode_progressive_dc_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    mcus_x: usize,
    mcus_y: usize,
    dc_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    output: &mut Vec<u8>,
    restart_interval: u16,
) {
    let al: u8 = scan.al;
    let ah: u8 = scan.ah;
    let mut prev_dc: [i16; 4] = [0i16; 4];

    // Reserve capacity: worst-case ~32 bits per block per component, plus a
    // small per-MCU cushion that absorbs the worst-case 8-byte bit drain and
    // 2-byte RST marker every restart_interval MCUs.
    let total_blocks: usize = scan
        .component_indices
        .iter()
        .map(|&ci| {
            let layout = &comp_layouts[ci];
            mcus_x * mcus_y * layout.h_blocks * layout.v_blocks
        })
        .sum();
    let total_mcus: usize = mcus_x * mcus_y;
    let restart_overhead: usize = if restart_interval > 0 {
        total_mcus.div_ceil(restart_interval as usize) * 16
    } else {
        0
    };
    let reserve: usize = total_blocks * 4 + restart_overhead + 64;
    output.reserve(reserve);

    let ri: u32 = restart_interval as u32;

    unsafe {
        let base: usize = output.len();
        let mut pb: u64 = 0;
        let mut fb: i32 = 64;
        let mut buf: *mut u8 = output.as_mut_ptr().add(base);
        let mut mcu_count: u32 = 0;
        let mut rst_count: u8 = 0;

        for mcu_y in 0..mcus_y {
            for mcu_x in 0..mcus_x {
                if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                    // Drain partial bits with 1-padding, write RST marker.
                    local_drain_bits(&mut pb, &mut fb, &mut buf);
                    // Reserve room for the 2-byte marker plus next-MCU worst case.
                    let written: usize = buf.offset_from(output.as_ptr().add(base)) as usize;
                    if written + 80 > reserve {
                        output.set_len(base + written);
                        output.reserve(reserve);
                        buf = output.as_mut_ptr().add(base + written);
                    }
                    *buf = 0xFF;
                    *buf.add(1) = 0xD0 + (rst_count & 7);
                    buf = buf.add(2);
                    pb = 0;
                    fb = 64;
                    prev_dc = [0i16; 4];
                    rst_count = (rst_count + 1) & 7;
                }

                for (scan_ci, &ci) in scan.component_indices.iter().enumerate() {
                    let layout = &comp_layouts[ci];
                    let dc_table: &HuffTable = if ci == 0 {
                        dc_luma_table
                    } else {
                        dc_chroma_table
                    };

                    for bv in 0..layout.v_blocks {
                        for bh in 0..layout.h_blocks {
                            let bx: usize = mcu_x * layout.h_blocks + bh;
                            let by: usize = mcu_y * layout.v_blocks + bv;
                            let block: &[i16; 64] = &coeff_bufs[ci][by * layout.blocks_x + bx];

                            // Ensure capacity: 16 bytes per block worst-case
                            let written: usize =
                                buf.offset_from(output.as_ptr().add(base)) as usize;
                            if written + 64 > reserve {
                                output.set_len(base + written);
                                output.reserve(reserve);
                                buf = output.as_mut_ptr().add(base + written);
                            }

                            if ah == 0 {
                                let dc: i16 = block[0] >> al;
                                let diff: i16 = dc.wrapping_sub(prev_dc[scan_ci]);
                                prev_dc[scan_ci] = dc;

                                if diff == 0 {
                                    local_put_bits(
                                        &mut pb,
                                        &mut fb,
                                        &mut buf,
                                        dc_table.ehufco[0] as u32,
                                        dc_table.ehufsi[0],
                                    );
                                } else {
                                    let abs_diff: u16 = diff.unsigned_abs();
                                    let category: u8 = 16 - abs_diff.leading_zeros() as u8;
                                    let magnitude: u16 =
                                        if diff > 0 { diff as u16 } else { !abs_diff };
                                    let huff_code: u32 = dc_table.ehufco[category as usize] as u32;
                                    let huff_size: u8 = dc_table.ehufsi[category as usize];
                                    let mag_masked: u32 =
                                        magnitude as u32 & ((1u32 << category) - 1);
                                    let combined: u32 = (huff_code << category) | mag_masked;
                                    local_put_bits(
                                        &mut pb,
                                        &mut fb,
                                        &mut buf,
                                        combined,
                                        huff_size + category,
                                    );
                                }
                            } else {
                                let bit: u32 = ((block[0] >> al) & 1) as u32;
                                local_put_bits(&mut pb, &mut fb, &mut buf, bit, 1);
                            }
                        }
                    }
                }
                mcu_count = mcu_count.wrapping_add(1);
            }
        }

        local_drain_bits(&mut pb, &mut fb, &mut buf);
        let final_len: usize = buf.offset_from(output.as_ptr().add(base)) as usize;
        output.set_len(base + final_len);
    }
}

/// Encode a progressive AC scan (single component).
///
/// Iterates all blocks in flat raster order within the component buffer.
#[allow(dead_code, clippy::too_many_arguments)]
fn encode_progressive_ac_scan(
    coeff_bufs: &[Vec<[i16; 64]>],
    comp_layouts: &[CompLayout],
    scan: &crate::encode::progressive::ProgressiveScan,
    _mcus_x: usize,
    _mcus_y: usize,
    ac_luma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
) {
    let ci = scan.component_indices[0]; // AC scans are single-component
    let _layout = &comp_layouts[ci];
    let ac_table = if ci == 0 {
        ac_luma_table
    } else {
        ac_chroma_table
    };
    let ss = scan.ss as usize;
    let se = scan.se as usize;
    let al = scan.al;
    let ah = scan.ah;

    // Non-interleaved AC scans iterate blocks in raster order within the component.
    let blocks: &[[i16; 64]] = &coeff_bufs[ci];
    if ah == 0 {
        let mut eobrun: u32 = 0;
        for block in blocks.iter() {
            encode_ac_first_block(block, ss, se, al, ac_table, writer, &mut eobrun);
        }
        if eobrun > 0 {
            emit_eobrun(ac_table, writer, &mut eobrun);
        }
    } else {
        let mut eobrun: u32 = 0;
        let mut corr_buffer: Vec<u8> = Vec::with_capacity(MAX_CORR_BITS);
        for block in blocks.iter() {
            encode_ac_refine_block(
                block,
                ss,
                se,
                al,
                ac_table,
                writer,
                &mut eobrun,
                &mut corr_buffer,
            );
        }
        if eobrun > 0 {
            emit_eobrun_with_corr(ac_table, writer, &mut eobrun, &mut corr_buffer);
        }
    }
}

/// Encode one block for AC first scan (ah==0).
///
/// Pre-computes values and bitmap to skip zero runs via CTZ, matching
/// C's jcphuff.c prepare+encode pattern. Combines Huffman code + magnitude
/// into single put_bits calls.
pub(crate) fn encode_ac_first_block(
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    eobrun: &mut u32,
) {
    let band_len: usize = se - ss + 1;

    let mut values = [0u16; 64];
    let mut diffs = [0u16; 64];
    let mut zerobits: u64 = 0;

    for i in 0..band_len {
        let coeff: i16 = block[ss + i];
        if coeff == 0 {
            continue;
        }
        // i32 widen: see api/coefficient.rs note (i16::MIN abs overflow).
        let coeff: i32 = coeff as i32;
        let sign_mask: i32 = coeff >> 31;
        let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
        let temp: u16 = (abs_coeff >> al) as u16;
        if temp == 0 {
            continue;
        }
        values[i] = temp;
        diffs[i] = (sign_mask ^ (abs_coeff >> al)) as u16;
        zerobits |= 1u64 << i;
    }

    if zerobits == 0 {
        // Accumulate EOBRUN
        *eobrun += 1;
        if *eobrun == 0x7FFF {
            emit_eobrun(ac_table, writer, eobrun);
        }
        return;
    }

    // Flush pending EOBRUN before encoding nonzero coefficients
    if *eobrun > 0 {
        emit_eobrun(ac_table, writer, eobrun);
    }

    let mut nbits_arr = [0u8; 64];
    {
        let mut bits: u64 = zerobits;
        while bits != 0 {
            let pos: usize = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            nbits_arr[pos] = 16 - values[pos].leading_zeros() as u8;
        }
    }

    let mut prev_pos: usize = 0;

    while zerobits != 0 {
        let pos: usize = zerobits.trailing_zeros() as usize;
        zerobits &= zerobits - 1;

        let mut zero_run: usize = pos - prev_pos;
        while zero_run >= 16 {
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            zero_run -= 16;
        }

        let nbits: u8 = nbits_arr[pos];
        let symbol: usize = (zero_run << 4) | (nbits as usize);
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let mag_masked: u32 = diffs[pos] as u32 & ((1u32 << nbits) - 1);
        let combined: u32 = (huff_code << nbits) | mag_masked;
        writer.put_bits(combined, huff_size + nbits);
        prev_pos = pos + 1;
    }

    if prev_pos < band_len {
        // Trailing zeros → accumulate EOBRUN
        *eobrun += 1;
        if *eobrun == 0x7FFF {
            emit_eobrun(ac_table, writer, eobrun);
        }
    }
}

/// Emit buffered EOBRUN to the bitstream. Matches C's emit_eobrun in jcphuff.c.
pub(crate) fn emit_eobrun(ac_table: &HuffTable, writer: &mut BitWriter, eobrun: &mut u32) {
    if *eobrun == 0 {
        return;
    }
    let nbits: u8 = (32 - (*eobrun).leading_zeros()) as u8 - 1;
    let symbol: usize = (nbits as usize) << 4;
    let huff_code: u32 = ac_table.ehufco[symbol] as u32;
    let huff_size: u8 = ac_table.ehufsi[symbol];
    if nbits > 0 {
        let combined: u32 = (huff_code << nbits) | (*eobrun & ((1u32 << nbits) - 1));
        writer.put_bits(combined, huff_size + nbits);
    } else {
        writer.put_bits(huff_code, huff_size);
    }
    *eobrun = 0;
}

/// Maximum number of correction bits buffered across blocks for AC refine EOBRUN.
/// Matches C libjpeg-turbo's MAX_CORR_BITS in jcphuff.c.
pub(crate) const MAX_CORR_BITS: usize = 1000;

/// Emit buffered correction bits from a byte slice.
/// Each byte holds a single bit value (0 or 1).
/// Matches C libjpeg-turbo's emit_buffered_bits in jcphuff.c.
#[inline]
fn emit_buffered_bits(writer: &mut BitWriter, bits: &[u8]) {
    for &bit in bits {
        writer.put_bits(bit as u32, 1);
    }
}

/// Emit pending EOBRUN symbol and all buffered correction bits.
/// Used by AC refine scans where correction bits must be associated with the
/// EOBRUN symbol. Matches C libjpeg-turbo's emit_eobrun in jcphuff.c when
/// combined with the correction bit buffer (entropy->bit_buffer / entropy->BE).
pub(crate) fn emit_eobrun_with_corr(
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    eobrun: &mut u32,
    corr_buffer: &mut Vec<u8>,
) {
    if *eobrun == 0 {
        return;
    }
    let nbits: u8 = (32 - (*eobrun).leading_zeros()) as u8 - 1;
    let symbol: usize = (nbits as usize) << 4;
    let huff_code: u32 = ac_table.ehufco[symbol] as u32;
    let huff_size: u8 = ac_table.ehufsi[symbol];
    if nbits > 0 {
        let combined: u32 = (huff_code << nbits) | (*eobrun & ((1u32 << nbits) - 1));
        writer.put_bits(combined, huff_size + nbits);
    } else {
        writer.put_bits(huff_code, huff_size);
    }
    *eobrun = 0;

    // Emit all buffered correction bits
    emit_buffered_bits(writer, corr_buffer);
    corr_buffer.clear();
}

/// Encode one block for AC successive approximation refinement scan (ah!=0).
///
/// Ported from libjpeg-turbo jcphuff.c `encode_mcu_AC_refine`.
/// Per ITU-T T.81 Figure G.7, previously-nonzero coefficients emit correction
/// bits that must be associated with the next Huffman symbol (ZRL, EOB, or
/// newly-nonzero code).
///
/// EOBRUN is batched across blocks with correction bits buffered in
/// `corr_buffer` (matching C's `entropy->bit_buffer` / `entropy->BE`).
/// Per-block correction bits (BR) are kept in a local array and flushed
/// after each Huffman symbol, while cross-block bits (BE) accumulate in
/// `corr_buffer` and are flushed only when the EOBRUN is emitted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_ac_refine_block(
    block: &[i16; 64],
    ss: usize,
    se: usize,
    al: u8,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    eobrun: &mut u32,
    corr_buffer: &mut Vec<u8>,
) {
    let band_len: usize = se - ss + 1;

    let mut absvals = [0u16; 64];
    let mut sign_bits = [0u16; 64];
    let mut eob_pos: usize = 0;

    for i in 0..band_len {
        let coeff: i32 = block[ss + i] as i32;
        let sign_mask: i32 = coeff >> 31;
        let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
        let temp: u16 = (abs_coeff >> al) as u16;
        absvals[i] = temp;
        sign_bits[i] = (sign_mask as u16).wrapping_add(1);
        if temp == 1 {
            eob_pos = i + 1;
        }
    }

    let mut r: usize = 0;
    // BR: this block's correction bits (separate from cross-block BE in corr_buffer)
    let mut br_bits: [u8; 64] = [0u8; 64];
    let mut br: usize = 0;
    let mut idx: usize = 0;

    while idx < band_len {
        let temp: u16 = absvals[idx];

        if temp == 0 {
            r += 1;
            idx += 1;
            continue;
        }

        // Emit ZRLs for zero runs > 15, but not if they can be folded into EOB
        while r > 15 && idx < eob_pos {
            // Flush pending EOBRUN + BE correction bits
            emit_eobrun_with_corr(ac_table, writer, eobrun, corr_buffer);
            // Emit ZRL symbol
            writer.put_bits(ac_table.ehufco[0xF0] as u32, ac_table.ehufsi[0xF0]);
            r -= 16;
            // Emit this block's buffered correction bits (BR)
            emit_buffered_bits(writer, &br_bits[..br]);
            br = 0;
        }

        if temp > 1 {
            // Previously nonzero: buffer correction bit
            br_bits[br] = (temp & 1) as u8;
            br += 1;
            idx += 1;
            continue;
        }

        // Newly nonzero (temp == 1): flush EOBRUN, emit symbol + sign bit
        emit_eobrun_with_corr(ac_table, writer, eobrun, corr_buffer);

        let symbol: usize = (r << 4) | 1;
        let huff_code: u32 = ac_table.ehufco[symbol] as u32;
        let huff_size: u8 = ac_table.ehufsi[symbol];
        let combined: u32 = (huff_code << 1) | sign_bits[idx] as u32;
        writer.put_bits(combined, huff_size + 1);

        // Emit this block's buffered correction bits (BR)
        emit_buffered_bits(writer, &br_bits[..br]);
        br = 0;
        r = 0;
        idx += 1;
    }

    // Trailing zeroes or correction bits → accumulate EOBRUN
    if r > 0 || br > 0 {
        *eobrun += 1;
        // Append this block's correction bits (BR) to cross-block buffer (BE)
        corr_buffer.extend_from_slice(&br_bits[..br]);
        // Force flush to prevent overflow of EOBRUN counter or correction buffer
        if *eobrun == 0x7FFF || corr_buffer.len() > (MAX_CORR_BITS - 64 + 1) {
            emit_eobrun_with_corr(ac_table, writer, eobrun, corr_buffer);
        }
    }
}

/// FDCT+quantize a Y block. Uses fused extract+FDCT on aarch64 for interior blocks.
#[inline]
#[allow(clippy::too_many_arguments)]
fn progressive_fdct_y_block(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    bx: usize,
    by: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    output: &mut [i16; 64],
    use_simd_fdct: bool,
) {
    #[cfg(target_arch = "aarch64")]
    {
        if use_simd_fdct && bx + 8 <= plane_w && by + 8 <= plane_h {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(by * plane_w + bx),
                    plane_w,
                    quant,
                    output,
                );
            }
            return;
        }
    }
    let _ = use_simd_fdct;
    let mut block = [0i16; 64];
    extract_block(plane, plane_w, plane_h, bx, by, &mut block);
    fdct_quantize_fn(&mut block, quant, output);
}

/// FDCT+quantize a chroma block with optional downsampling.
#[inline]
#[allow(clippy::too_many_arguments)]
fn progressive_fdct_chroma_block(
    plane: &[u8],
    plane_w: usize,
    plane_h: usize,
    x0: usize,
    y0: usize,
    h_samp: usize,
    v_samp: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    output: &mut [i16; 64],
    use_simd_fdct: bool,
) {
    // Use the real sampling ratios from the caller. Clamping to {1,2} would
    // silently corrupt 4-pixel chroma factors (S411/S441/S410/S24): the
    // emitted SOF says "1/4 chroma resolution" but the buffer would carry
    // 1/2-resolution coefficients packed at the wrong positions, producing
    // ~max-150 pixel divergence in the decoded image (P2-11).
    let hf: usize = h_samp;
    let vf: usize = v_samp;

    if hf == 1 && vf == 1 {
        progressive_fdct_y_block(
            plane,
            plane_w,
            plane_h,
            x0,
            y0,
            quant,
            fdct_quantize_fn,
            output,
            use_simd_fdct,
        );
        return;
    }

    #[cfg(target_arch = "aarch64")]
    {
        let src_w: usize = hf * 8;
        let src_h: usize = vf * 8;
        if use_simd_fdct && x0 + src_w <= plane_w && y0 + src_h <= plane_h {
            unsafe {
                let ptr: *const u8 = plane.as_ptr().add(y0 * plane_w + x0);
                if hf == 2 && vf == 2 {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else if hf == 2 && vf == 1 {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else {
                    let mut block = [0i16; 64];
                    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
                    fdct_quantize_fn(&mut block, quant, output);
                }
            }
            return;
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        let src_w: usize = hf * 8;
        let src_h: usize = vf * 8;
        if use_simd_fdct && x0 + src_w <= plane_w && y0 + src_h <= plane_h {
            unsafe {
                let ptr: *const u8 = plane.as_ptr().add(y0 * plane_w + x0);
                if hf == 2 && vf == 2 {
                    crate::simd::wasm32::wasm_downsample_h2v2_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else if hf == 2 && vf == 1 {
                    crate::simd::wasm32::wasm_downsample_h2v1_fdct_quantize(
                        ptr, plane_w, quant, output,
                    );
                } else {
                    let mut block = [0i16; 64];
                    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
                    fdct_quantize_fn(&mut block, quant, output);
                }
            }
            return;
        }
    }

    let _ = use_simd_fdct;
    let mut block = [0i16; 64];
    downsample_chroma_block(plane, plane_w, plane_h, x0, y0, hf, vf, &mut block);
    fdct_quantize_fn(&mut block, quant, output);
}

/// Find highest set bit position (1-indexed). Returns 0 for val=0.
/// Port of C libjpeg-turbo's `flss` from jcdctmgr.c.
fn flss(val: u16) -> i32 {
    if val == 0 {
        return 0;
    }
    16 - val.leading_zeros() as i32
}

/// Compute adaptive-precision reciprocal for exact SIMD quantization.
/// Port of C libjpeg-turbo's `compute_reciprocal` from jcdctmgr.c.
///
/// Returns (reciprocal, correction, scale, shift).
/// - NEON uses (reciprocal, correction, shift) with per-element variable shift.
/// - AVX2 uses (reciprocal, correction, scale) with two `pmulhuw` ops (matching C).
pub fn compute_reciprocal(divisor: u16) -> (u16, u16, u16, i16) {
    if divisor <= 1 {
        // scale=1 for the identity case (matches C: dtbl[DCTSIZE2*2] = 1)
        return (1, 0, 1, -(std::mem::size_of::<i16>() as i16 * 8));
    }

    let b: i32 = flss(divisor) - 1;
    let r: i32 = 16 + b; // adaptive precision

    let fq: u32 = (1u32 << r) / divisor as u32;
    let fr: u32 = (1u32 << r) % divisor as u32;

    let mut recip: u32 = fq;
    let mut corr: u16 = divisor / 2;
    let mut r: i32 = r;

    if fr == 0 {
        // Divisor is power of two: fq is one bit too large, adjust
        recip >>= 1;
        r -= 1;
    } else if fr <= (divisor as u32 / 2) {
        // Fractional part < 0.5: round down, bump correction
        corr += 1;
    } else {
        // Fractional part > 0.5: round up
        recip += 1;
    }

    let shift: i16 = (r - 16) as i16;
    // Scale for AVX2: replaces per-element variable shift with a second mulhi.
    // scale = 1 << (32 - r), so mulhi(x, scale) == x >> (r - 16) == x >> shift.
    // Matches C: dtbl[DCTSIZE2 * 2] = (DCTELEM)(1 << (sizeof(DCTELEM)*8*2 - r))
    let scale: u16 = (1u32 << (32 - r)) as u16;
    (recip as u16, corr, scale, shift)
}

/// Scale quantization table for the IFAST FDCT using AA&N scale factors.
///
/// Computes `DESCALE(quant[i] * aanscales[i], CONST_BITS - 3)` where
/// `CONST_BITS = 14`, matching C libjpeg-turbo's `jcdctmgr.c` ifast divisor
/// computation exactly. Paired with `fdct_ifast_raw` (no AA&N rescaling).
fn scale_quant_for_ifast(quant_table: &[u16; 64]) -> QuantDivisors {
    use crate::encode::fdct::AANSCALES;
    let mut divisors = [0u16; 64];
    let mut reciprocals = [0u16; 64];
    let mut corrections = [0u16; 64];
    let mut shifts = [0i16; 64];
    let mut scales = [0u16; 64];
    for i in 0..64 {
        // DESCALE(quant * aanscale, 14 - 3) = (quant * aanscale + 1024) >> 11
        let product: i64 = quant_table[i] as i64 * AANSCALES[i] as i64;
        let d: u16 = ((product + (1i64 << 10)) >> 11) as u16;
        divisors[i] = d;
        let (recip, corr, scale, shift) = compute_reciprocal(d);
        reciprocals[i] = recip;
        corrections[i] = corr;
        scales[i] = scale;
        shifts[i] = shift;
    }
    let float_divisors = compute_float_divisors(quant_table);
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    let mut divisors_zigzag = [0u16; 64];
    let mut reciprocals_zigzag = [0u16; 64];
    let mut corrections_zigzag = [0u16; 64];
    let mut shifts_zigzag = [0i16; 64];
    let mut scales_zigzag = [0u16; 64];
    let mut float_divisors_zigzag = [0.0f32; 64];
    for zz in 0..64 {
        divisors_zigzag[zz] = divisors[zigzag[zz]];
        reciprocals_zigzag[zz] = reciprocals[zigzag[zz]];
        corrections_zigzag[zz] = corrections[zigzag[zz]];
        shifts_zigzag[zz] = shifts[zigzag[zz]];
        scales_zigzag[zz] = scales[zigzag[zz]];
        float_divisors_zigzag[zz] = float_divisors[zigzag[zz]];
    }
    QuantDivisors {
        divisors,
        reciprocals,
        corrections,
        shifts,
        scales,
        divisors_zigzag,
        reciprocals_zigzag,
        corrections_zigzag,
        shifts_zigzag,
        scales_zigzag,
        float_divisors,
        float_divisors_zigzag,
    }
}

/// C `jcdctmgr.c` lines 346–365: float divisor =
/// `1 / (quant[i] * aanscalefactor[row] * aanscalefactor[col] * 8)`.
fn compute_float_divisors(quant_table: &[u16; 64]) -> [f32; 64] {
    const AANSCALEFACTOR: [f64; 8] = [
        1.0,
        1.387039845,
        1.306562965,
        1.175875602,
        1.0,
        0.785694958,
        0.541196100,
        0.275899379,
    ];
    let mut out = [0.0f32; 64];
    #[allow(clippy::needless_range_loop)]
    for row in 0..8 {
        for col in 0..8 {
            let i: usize = row * 8 + col;
            let denom: f64 =
                quant_table[i] as f64 * AANSCALEFACTOR[row] * AANSCALEFACTOR[col] * 8.0;
            out[i] = (1.0 / denom) as f32;
        }
    }
    out
}

/// Scale quantization table values by 8 to create divisor table for the islow FDCT.
///
/// Uses C libjpeg-turbo's adaptive-precision reciprocal algorithm for exact
/// SIMD quantization (no rounding errors vs true integer division).
fn scale_quant_for_fdct(quant_table: &[u16; 64]) -> QuantDivisors {
    let mut divisors = [0u16; 64];
    let mut reciprocals = [0u16; 64];
    let mut corrections = [0u16; 64];
    let mut shifts = [0i16; 64];
    let mut scales = [0u16; 64];
    for i in 0..64 {
        let d: u16 = (quant_table[i] as u32 * 8) as u16;
        divisors[i] = d;
        let (recip, corr, scale, shift) = compute_reciprocal(d);
        reciprocals[i] = recip;
        corrections[i] = corr;
        scales[i] = scale;
        shifts[i] = shift;
    }
    let float_divisors = compute_float_divisors(quant_table);
    let zigzag = &crate::encode::tables::ZIGZAG_ORDER;
    let mut divisors_zigzag = [0u16; 64];
    let mut reciprocals_zigzag = [0u16; 64];
    let mut corrections_zigzag = [0u16; 64];
    let mut shifts_zigzag = [0i16; 64];
    let mut scales_zigzag = [0u16; 64];
    let mut float_divisors_zigzag = [0.0f32; 64];
    for zz in 0..64 {
        divisors_zigzag[zz] = divisors[zigzag[zz]];
        reciprocals_zigzag[zz] = reciprocals[zigzag[zz]];
        corrections_zigzag[zz] = corrections[zigzag[zz]];
        shifts_zigzag[zz] = shifts[zigzag[zz]];
        scales_zigzag[zz] = scales[zigzag[zz]];
        float_divisors_zigzag[zz] = float_divisors[zigzag[zz]];
    }
    QuantDivisors {
        divisors,
        reciprocals,
        corrections,
        shifts,
        scales,
        divisors_zigzag,
        reciprocals_zigzag,
        corrections_zigzag,
        shifts_zigzag,
        scales_zigzag,
        float_divisors,
        float_divisors_zigzag,
    }
}

/// Convert input pixels to Y, Cb, Cr planes.
#[allow(clippy::type_complexity)]
/// Convert pixels to YCbCr planes with MCU-aligned padding.
///
/// Returns `(y_plane, cb_plane, cr_plane, padded_w, padded_h)` where planes are
/// padded to `padded_w × padded_h` with replicated-last-pixel/row matching C
/// libjpeg-turbo's `expand_right_edge` behavior.  All blocks (including edge)
/// are interior to the padded dimensions, so the NEON fused FDCT+quantize path
/// is always taken, ensuring byte-identical output with C.
#[allow(clippy::too_many_arguments)]
fn convert_to_ycbcr_padded(
    pixels: &[u8],
    width: usize,
    height: usize,
    padded_w: usize,
    padded_h: usize,
    pixel_format: PixelFormat,
    rgb_to_ycbcr_row_fn: fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize),
    max_v_samp: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let plane_size: usize = padded_w * padded_h;
    let mut y_plane: Vec<u8> = vec![0u8; plane_size];
    let mut cb_plane: Vec<u8> = vec![0u8; plane_size];
    let mut cr_plane: Vec<u8> = vec![0u8; plane_size];

    let bpp: usize = pixel_format.bytes_per_pixel();

    match pixel_format {
        PixelFormat::Grayscale => {
            for row in 0..height {
                let src_start: usize = row * width;
                let dst_start: usize = row * padded_w;
                y_plane[dst_start..dst_start + width]
                    .copy_from_slice(&pixels[src_start..src_start + width]);
                // Right-edge padding
                if width < padded_w {
                    let last_val: u8 = pixels[src_start + width - 1];
                    for x in width..padded_w {
                        y_plane[dst_start + x] = last_val;
                    }
                }
            }
        }
        PixelFormat::Rgb => {
            for row in 0..height {
                let src_offset: usize = row * width * bpp;
                let dst_offset: usize = row * padded_w;
                rgb_to_ycbcr_row_fn(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
                // Right-edge padding
                if width < padded_w {
                    let last_y: u8 = y_plane[dst_offset + width - 1];
                    let last_cb: u8 = cb_plane[dst_offset + width - 1];
                    let last_cr: u8 = cr_plane[dst_offset + width - 1];
                    for x in width..padded_w {
                        y_plane[dst_offset + x] = last_y;
                        cb_plane[dst_offset + x] = last_cb;
                        cr_plane[dst_offset + x] = last_cr;
                    }
                }
            }
        }
        _ => {
            // Non-RGB formats: use convert_to_ycbcr then pad
            let (y_raw, cb_raw, cr_raw) =
                convert_to_ycbcr(pixels, width, height, pixel_format, rgb_to_ycbcr_row_fn)?;
            for row in 0..height {
                let src_start: usize = row * width;
                let dst_start: usize = row * padded_w;
                y_plane[dst_start..dst_start + width]
                    .copy_from_slice(&y_raw[src_start..src_start + width]);
                cb_plane[dst_start..dst_start + width]
                    .copy_from_slice(&cb_raw[src_start..src_start + width]);
                cr_plane[dst_start..dst_start + width]
                    .copy_from_slice(&cr_raw[src_start..src_start + width]);
                if width < padded_w {
                    let last_y: u8 = y_raw[src_start + width - 1];
                    let last_cb: u8 = cb_raw[src_start + width - 1];
                    let last_cr: u8 = cr_raw[src_start + width - 1];
                    for x in width..padded_w {
                        y_plane[dst_start + x] = last_y;
                        cb_plane[dst_start + x] = last_cb;
                        cr_plane[dst_start + x] = last_cr;
                    }
                }
            }
        }
    }

    // Bottom-edge padding: Y uses last-row replication, Cb/Cr use row-group
    // replication to match C libjpeg-turbo's two-phase approach.
    if height < padded_h {
        let last_row: Vec<u8> = y_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        for row in height..padded_h {
            let dst: usize = row * padded_w;
            y_plane[dst..dst + padded_w].copy_from_slice(&last_row);
        }

        // Chroma: row-group replication
        let row_group_end: usize =
            height.div_ceil(max_v_samp).min(padded_h / max_v_samp) * max_v_samp;
        let last_cb: Vec<u8> = cb_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        let last_cr: Vec<u8> = cr_plane[(height - 1) * padded_w..height * padded_w].to_vec();
        // Phase 1: pad to row group boundary
        for row in height..row_group_end.min(padded_h) {
            let dst: usize = row * padded_w;
            cb_plane[dst..dst + padded_w].copy_from_slice(&last_cb);
            cr_plane[dst..dst + padded_w].copy_from_slice(&last_cr);
        }
        // Phase 2: replicate last complete row group
        if row_group_end < padded_h {
            let group_start: usize = row_group_end - max_v_samp;
            for row in row_group_end..padded_h {
                let src_row: usize = group_start + (row - row_group_end) % max_v_samp;
                let dst: usize = row * padded_w;
                let src: usize = src_row * padded_w;
                let cb_src: Vec<u8> = cb_plane[src..src + padded_w].to_vec();
                let cr_src: Vec<u8> = cr_plane[src..src + padded_w].to_vec();
                cb_plane[dst..dst + padded_w].copy_from_slice(&cb_src);
                cr_plane[dst..dst + padded_w].copy_from_slice(&cr_src);
            }
        }
    }

    Ok((y_plane, cb_plane, cr_plane))
}

#[allow(clippy::type_complexity)]
fn convert_to_ycbcr(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    rgb_to_ycbcr_row_fn: fn(&[u8], &mut [u8], &mut [u8], &mut [u8], usize),
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let plane_size = width * height;
    let mut y_plane = vec![0u8; plane_size];
    let mut cb_plane = vec![0u8; plane_size];
    let mut cr_plane = vec![0u8; plane_size];

    let bpp = pixel_format.bytes_per_pixel();

    match pixel_format {
        PixelFormat::Grayscale => {
            y_plane.copy_from_slice(&pixels[..plane_size]);
            // Cb and Cr stay at 0 (won't be used for grayscale)
        }
        PixelFormat::Rgb => {
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                rgb_to_ycbcr_row_fn(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Rgba => {
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    crate::simd::aarch64::color_encode::neon_rgba_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "wasm32", feature = "simd"))]
                {
                    crate::simd::wasm32::color_encode::wasm_rgba_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        crate::simd::x86_64::avx2_color_encode::avx2_rgba_to_ycbcr_row(
                            &pixels[src_offset..src_offset + width * bpp],
                            &mut y_plane[dst_offset..dst_offset + width],
                            &mut cb_plane[dst_offset..dst_offset + width],
                            &mut cr_plane[dst_offset..dst_offset + width],
                            width,
                        );
                        continue;
                    }
                }
                #[allow(unreachable_code)]
                color::rgba_to_ycbcr_row(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Bgr => {
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    crate::simd::aarch64::color_encode::neon_bgr_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "wasm32", feature = "simd"))]
                {
                    crate::simd::wasm32::color_encode::wasm_bgr_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        crate::simd::x86_64::avx2_color_encode::avx2_bgr_to_ycbcr_row(
                            &pixels[src_offset..src_offset + width * bpp],
                            &mut y_plane[dst_offset..dst_offset + width],
                            &mut cb_plane[dst_offset..dst_offset + width],
                            &mut cr_plane[dst_offset..dst_offset + width],
                            width,
                        );
                        continue;
                    }
                }
                #[allow(unreachable_code)]
                color::bgr_to_ycbcr_row_scalar(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Bgra => {
            for row in 0..height {
                let src_offset = row * width * bpp;
                let dst_offset = row * width;
                #[cfg(all(target_arch = "aarch64", feature = "simd"))]
                {
                    crate::simd::aarch64::color_encode::neon_bgra_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "wasm32", feature = "simd"))]
                {
                    crate::simd::wasm32::color_encode::wasm_bgra_to_ycbcr_row(
                        &pixels[src_offset..src_offset + width * bpp],
                        &mut y_plane[dst_offset..dst_offset + width],
                        &mut cb_plane[dst_offset..dst_offset + width],
                        &mut cr_plane[dst_offset..dst_offset + width],
                        width,
                    );
                    continue;
                }
                #[cfg(all(target_arch = "x86_64", feature = "simd"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        crate::simd::x86_64::avx2_color_encode::avx2_bgra_to_ycbcr_row(
                            &pixels[src_offset..src_offset + width * bpp],
                            &mut y_plane[dst_offset..dst_offset + width],
                            &mut cb_plane[dst_offset..dst_offset + width],
                            &mut cr_plane[dst_offset..dst_offset + width],
                            width,
                        );
                        continue;
                    }
                }
                #[allow(unreachable_code)]
                color::bgra_to_ycbcr_row_scalar(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                );
            }
        }
        PixelFormat::Rgbx
        | PixelFormat::Bgrx
        | PixelFormat::Xrgb
        | PixelFormat::Xbgr
        | PixelFormat::Argb
        | PixelFormat::Abgr => {
            let r_off: usize = pixel_format.red_offset().unwrap();
            let g_off: usize = pixel_format.green_offset().unwrap();
            let b_off: usize = pixel_format.blue_offset().unwrap();
            for row in 0..height {
                let src_offset: usize = row * width * bpp;
                let dst_offset: usize = row * width;
                color::generic_to_ycbcr_row(
                    &pixels[src_offset..src_offset + width * bpp],
                    &mut y_plane[dst_offset..dst_offset + width],
                    &mut cb_plane[dst_offset..dst_offset + width],
                    &mut cr_plane[dst_offset..dst_offset + width],
                    width,
                    bpp,
                    r_off,
                    g_off,
                    b_off,
                );
            }
        }
        PixelFormat::Cmyk => {
            return Err(JpegError::Unsupported(
                "CMYK pixel format not supported for encoding".to_string(),
            ));
        }
        PixelFormat::Rgb565 => {
            return Err(JpegError::Unsupported(
                "Rgb565 pixel format is decode-only and not supported for encoding".to_string(),
            ));
        }
    }

    Ok((y_plane, cb_plane, cr_plane))
}

/// Extract an 8x8 block from a plane with edge padding.
///
/// Replicates the last column/row when the block extends beyond the image boundary.
fn extract_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    // SIMD fast path for interior blocks (no bounds checking needed)
    if block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            extract_block_neon(plane, plane_width, block_x, block_y, block);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("sse2") {
                // SAFETY: SSE2 availability checked above, interior block bounds verified.
                unsafe {
                    extract_block_sse2(plane, plane_width, block_x, block_y, block);
                }
                return;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                extract_block_wasm(plane, plane_width, block_x, block_y, block);
            }
            return;
        }
    }

    // Scalar fallback for border blocks
    for row in 0..8 {
        let src_y: usize = (block_y + row).min(plane_height - 1);
        for col in 0..8 {
            let src_x: usize = (block_x + col).min(plane_width - 1);
            block[row * 8 + col] = plane[src_y * plane_width + src_x] as i16 - 128;
        }
    }
}

/// NEON-accelerated block extraction with level-shift for interior blocks.
///
/// Loads 8 bytes per row, widens to i16, subtracts 128. No bounds checking.
#[cfg(target_arch = "aarch64")]
fn extract_block_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use std::arch::aarch64::*;
    unsafe {
        let level_shift: int16x8_t = vdupq_n_s16(128);

        for row in 0..8 {
            let src_ptr: *const u8 = plane.as_ptr().add((block_y + row) * plane_width + block_x);
            let pixels: uint8x8_t = vld1_u8(src_ptr);
            let wide: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(pixels));
            let shifted: int16x8_t = vsubq_s16(wide, level_shift);
            vst1q_s16(block.as_mut_ptr().add(row * 8), shifted);
        }
    }
}

/// SSE2-accelerated block extraction with level-shift for interior blocks.
///
/// Loads 8 bytes per row, widens to i16, subtracts 128. No bounds checking.
///
/// # Safety
/// Requires SSE2. Caller must ensure `block_x + 8 <= plane_width` and
/// `block_y + 8 <= plane_height` (interior block bounds).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn extract_block_sse2(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    let level_shift: __m128i = _mm_set1_epi16(128);
    let zeros: __m128i = _mm_setzero_si128();

    for row in 0..8 {
        let src_ptr: *const u8 = plane.as_ptr().add((block_y + row) * plane_width + block_x);
        // Load 8 bytes (only low 64 bits used)
        let pixels: __m128i = _mm_loadl_epi64(src_ptr as *const __m128i);
        // Zero-extend u8 → i16
        let wide: __m128i = _mm_unpacklo_epi8(pixels, zeros);
        // Level-shift: subtract 128
        let shifted: __m128i = _mm_sub_epi16(wide, level_shift);
        _mm_storeu_si128(block.as_mut_ptr().add(row * 8) as *mut __m128i, shifted);
    }
}

/// WASM simd128-accelerated block extraction with level-shift for interior blocks.
///
/// Loads 8 bytes per row, widens to i16, subtracts 128. No bounds checking.
///
/// # Safety
/// Requires simd128. Caller must ensure `block_x + 8 <= plane_width` and
/// `block_y + 8 <= plane_height` (interior block bounds).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[target_feature(enable = "simd128")]
unsafe fn extract_block_wasm(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::wasm32::*;

    let level_shift: v128 = i16x8_splat(128);

    for row in 0..8 {
        let src_ptr: *const u8 = plane.as_ptr().add((block_y + row) * plane_width + block_x);
        let pixels: v128 = v128_load64_zero(src_ptr as *const u64);
        let wide: v128 = u16x8_extend_low_u8x16(pixels);
        let shifted: v128 = i16x8_sub(wide, level_shift);
        v128_store(block.as_mut_ptr().add(row * 8) as *mut v128, shifted);
    }
}

/// Downsample a chroma plane region using a box filter.
///
/// For 4:2:2: averages 2x1 pixel groups horizontally.
/// For 4:2:0: averages 2x2 pixel groups.
#[allow(clippy::too_many_arguments)]
fn downsample_chroma_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    block: &mut [i16; 64],
) {
    // SIMD fast path for interior blocks (no bounds checking needed)
    {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if block_x + src_w <= plane_width && block_y + src_h <= plane_height {
            #[cfg(target_arch = "aarch64")]
            {
                if h_factor == 2 && v_factor == 2 {
                    downsample_chroma_block_h2v2_neon(plane, plane_width, block_x, block_y, block);
                    return;
                }
                if h_factor == 2 && v_factor == 1 {
                    downsample_chroma_block_h2v1_neon(plane, plane_width, block_x, block_y, block);
                    return;
                }
            }
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("ssse3") {
                    if h_factor == 2 && v_factor == 2 {
                        // SAFETY: SSSE3 availability checked above, interior block bounds verified.
                        unsafe {
                            downsample_chroma_block_h2v2_ssse3(
                                plane,
                                plane_width,
                                block_x,
                                block_y,
                                block,
                            );
                        }
                        return;
                    }
                    if h_factor == 2 && v_factor == 1 {
                        unsafe {
                            downsample_chroma_block_h2v1_ssse3(
                                plane,
                                plane_width,
                                block_x,
                                block_y,
                                block,
                            );
                        }
                        return;
                    }
                }
            }
        }
    }

    // Scalar fallback: alternating bias matching C libjpeg-turbo jcsample.c
    let divisor: u32 = (h_factor * v_factor) as u32;
    let use_alt: bool = h_factor == 2 && (v_factor == 1 || v_factor == 2);

    // Vertical edge handling follows C's two-phase model (jcprepct.c then
    // jccoefct.c): pad the source up to a complete row *group*, downsample,
    // and then replicate the resulting **downsampled** row for everything
    // below the image. Clamping the source row instead is only equivalent
    // when the final row group is incomplete.
    //
    // With `v_factor == 2` and an even height the final group is complete, so
    // C replicates `avg(last_two_rows)` while a source clamp yields
    // `last_row` alone — which is why progressive diverged from cjpeg at every
    // even height that is not a multiple of the MCU height, 1920x1080
    // included (#324). Odd heights agreed by accident: their last group is
    // incomplete, so both models replicate the same single row.
    //
    // Horizontally there is no such phase — C's `expand_right_edge` replicates
    // source *pixels* — so column clamping below is already correct.
    let chroma_rows: usize = plane_height.div_ceil(v_factor);
    let first_chroma_row: usize = block_y / v_factor;

    for row in 0..8 {
        let source_row_base: usize =
            (first_chroma_row + row).min(chroma_rows.saturating_sub(1)) * v_factor;
        let mut bias: u32 = if h_factor == 2 && v_factor == 1 {
            0
        } else if h_factor == 2 && v_factor == 2 {
            1
        } else {
            divisor / 2
        };
        let toggle: u32 = if h_factor == 2 && v_factor == 1 { 1 } else { 3 };
        for col in 0..8 {
            let mut sum: u32 = 0;
            for dy in 0..v_factor {
                for dx in 0..h_factor {
                    let sx = (block_x + col * h_factor + dx).min(plane_width - 1);
                    let sy = (source_row_base + dy).min(plane_height - 1);
                    sum += plane[sy * plane_width + sx] as u32;
                }
            }
            let avg = (sum + bias) / divisor;
            block[row * 8 + col] = avg as i16 - 128;
            if use_alt {
                bias ^= toggle;
            }
        }
    }
}

/// NEON-accelerated H2V2 downsample + level-shift for interior chroma blocks.
///
/// Processes 16x16 source pixels → 8x8 output using vpadalq_u8 pairwise add.
/// Each 2x2 block is averaged and level-shifted (-128) in NEON registers.
#[cfg(target_arch = "aarch64")]
fn downsample_chroma_block_h2v2_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use std::arch::aarch64::*;
    unsafe {
        // Rounding bias of 2 for divide-by-4 (matches scalar: (sum + 2) / 4)
        let bias: uint16x8_t = vreinterpretq_u16_u32(vdupq_n_u32(0x00020001));
        let level_shift: int16x8_t = vdupq_n_s16(128);

        for row in 0..8 {
            let sy: usize = block_y + row * 2;
            let r0_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);
            let r1_ptr: *const u8 = plane.as_ptr().add((sy + 1) * plane_width + block_x);

            let r0: uint8x16_t = vld1q_u8(r0_ptr);
            let r1: uint8x16_t = vld1q_u8(r1_ptr);

            // Pairwise-add adjacent u8 pairs from both rows into u16 sums
            let mut sum: uint16x8_t = vpadalq_u8(bias, r0);
            sum = vpadalq_u8(sum, r1);

            // Divide by 4 and narrow to u8
            let avg_u8: uint8x8_t = vshrn_n_u16(sum, 2);

            // Widen to i16 and level-shift (-128)
            let avg_i16: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(avg_u8));
            let shifted: int16x8_t = vsubq_s16(avg_i16, level_shift);

            vst1q_s16(block.as_mut_ptr().add(row * 8), shifted);
        }
    }
}

/// NEON-accelerated H2V1 downsample + level-shift for interior chroma blocks.
#[cfg(target_arch = "aarch64")]
fn downsample_chroma_block_h2v1_neon(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use std::arch::aarch64::*;
    unsafe {
        // Rounding bias of 1 for divide-by-2 (matches scalar: (sum + 1) / 2)
        let bias: uint16x8_t = vreinterpretq_u16_u32(vdupq_n_u32(0x00010000));
        let level_shift: int16x8_t = vdupq_n_s16(128);

        for row in 0..8 {
            let sy: usize = block_y + row;
            let r_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);

            let r: uint8x16_t = vld1q_u8(r_ptr);
            let sum: uint16x8_t = vpadalq_u8(bias, r);
            let avg_u8: uint8x8_t = vshrn_n_u16(sum, 1);
            let avg_i16: int16x8_t = vreinterpretq_s16_u16(vmovl_u8(avg_u8));
            let shifted: int16x8_t = vsubq_s16(avg_i16, level_shift);

            vst1q_s16(block.as_mut_ptr().add(row * 8), shifted);
        }
    }
}

/// SSSE3-accelerated H2V2 downsample + level-shift for interior chroma blocks.
///
/// Processes 16x16 source pixels → 8x8 output using maddubs pairwise add.
/// Each 2x2 block is averaged and level-shifted (-128).
///
/// # Safety
/// Requires SSSE3. Caller must ensure `block_x + 16 <= plane_width` and
/// `block_y + 16 <= plane_height` (interior block bounds).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn downsample_chroma_block_h2v2_ssse3(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    // maddubs(data, ones) computes pairwise sum of adjacent u8 pairs → i16
    let ones: __m128i = _mm_set1_epi8(1);
    let bias: __m128i = _mm_set_epi16(2, 1, 2, 1, 2, 1, 2, 1); // rounding for divide-by-4
    let level_shift: __m128i = _mm_set1_epi16(128);

    for row in 0..8 {
        let sy: usize = block_y + row * 2;
        let r0_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);
        let r1_ptr: *const u8 = plane.as_ptr().add((sy + 1) * plane_width + block_x);

        let r0: __m128i = _mm_loadu_si128(r0_ptr as *const __m128i);
        let r1: __m128i = _mm_loadu_si128(r1_ptr as *const __m128i);

        // Pairwise add: sum adjacent u8 pairs from each row → i16
        let sum0: __m128i = _mm_maddubs_epi16(r0, ones);
        let sum1: __m128i = _mm_maddubs_epi16(r1, ones);

        // Sum both rows + bias, divide by 4
        let total: __m128i = _mm_add_epi16(_mm_add_epi16(sum0, sum1), bias);
        let avg: __m128i = _mm_srai_epi16::<2>(total);

        // Level-shift (-128) and store
        let shifted: __m128i = _mm_sub_epi16(avg, level_shift);
        _mm_storeu_si128(block.as_mut_ptr().add(row * 8) as *mut __m128i, shifted);
    }
}

/// SSSE3-accelerated H2V1 downsample + level-shift for interior chroma blocks.
///
/// # Safety
/// Requires SSSE3. Caller must ensure `block_x + 16 <= plane_width` and
/// `block_y + 8 <= plane_height` (interior block bounds).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
unsafe fn downsample_chroma_block_h2v1_ssse3(
    plane: &[u8],
    plane_width: usize,
    block_x: usize,
    block_y: usize,
    block: &mut [i16; 64],
) {
    use core::arch::x86_64::*;

    let ones: __m128i = _mm_set1_epi8(1);
    let bias: __m128i = _mm_set_epi16(1, 0, 1, 0, 1, 0, 1, 0); // rounding for divide-by-2
    let level_shift: __m128i = _mm_set1_epi16(128);

    for row in 0..8 {
        let sy: usize = block_y + row;
        let r_ptr: *const u8 = plane.as_ptr().add(sy * plane_width + block_x);

        let r: __m128i = _mm_loadu_si128(r_ptr as *const __m128i);
        let sum: __m128i = _mm_add_epi16(_mm_maddubs_epi16(r, ones), bias);
        let avg: __m128i = _mm_srai_epi16::<1>(sum);
        let shifted: __m128i = _mm_sub_epi16(avg, level_shift);
        _mm_storeu_si128(block.as_mut_ptr().add(row * 8) as *mut __m128i, shifted);
    }
}

/// Apply fullsize smooth filter to a component plane, matching C's `fullsize_smooth_downsample`.
fn fullsize_smooth_plane(
    plane: &[u8],
    width: usize,
    height: usize,
    smoothing_factor: u8,
) -> Vec<u8> {
    let sf: i64 = smoothing_factor as i64;
    let memberscale: i64 = 65536 - sf * 512;
    let neighscale: i64 = sf * 64;
    let mut output: Vec<u8> = vec![0u8; plane.len()];
    for row in 0..height {
        let above_row: usize = if row == 0 { 0 } else { row - 1 };
        let below_row: usize = if row + 1 >= height { row } else { row + 1 };
        let inp: &[u8] = &plane[row * width..];
        let abv: &[u8] = &plane[above_row * width..];
        let blw: &[u8] = &plane[below_row * width..];
        let out: &mut [u8] = &mut output[row * width..];
        if width <= 1 {
            if width == 1 {
                let membersum: i64 = inp[0] as i64;
                let colsum: i64 = abv[0] as i64 + blw[0] as i64 + inp[0] as i64;
                let neighsum: i64 = colsum + (colsum - membersum) + colsum;
                let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
                out[0] = result.clamp(0, 255) as u8;
            }
            continue;
        }
        let mut colsum: i64 = abv[0] as i64 + blw[0] as i64 + inp[0] as i64;
        let membersum: i64 = inp[0] as i64;
        let mut nextcolsum: i64 = abv[1] as i64 + blw[1] as i64 + inp[1] as i64;
        let neighsum: i64 = colsum + (colsum - membersum) + nextcolsum;
        let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
        out[0] = result.clamp(0, 255) as u8;
        let mut lastcolsum: i64 = colsum;
        colsum = nextcolsum;
        for col in 1..width - 1 {
            let membersum: i64 = inp[col] as i64;
            nextcolsum = abv[col + 1] as i64 + blw[col + 1] as i64 + inp[col + 1] as i64;
            let neighsum: i64 = lastcolsum + (colsum - membersum) + nextcolsum;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[col] = result.clamp(0, 255) as u8;
            lastcolsum = colsum;
            colsum = nextcolsum;
        }
        let col: usize = width - 1;
        let membersum: i64 = inp[col] as i64;
        let neighsum: i64 = lastcolsum + (colsum - membersum) + colsum;
        let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
        out[col] = result.clamp(0, 255) as u8;
    }
    output
}

/// Smooth-downsample a chroma plane from full to half resolution,
/// matching C's `h2v2_smooth_downsample` (jcsample.c lines 308-387).
fn h2v2_smooth_downsample_plane(
    plane: &[u8],
    in_width: usize,
    in_height: usize,
    smoothing_factor: u8,
) -> Vec<u8> {
    let sf: i64 = smoothing_factor as i64;
    let memberscale: i64 = 16384 - sf * 80;
    let neighscale: i64 = sf * 16;
    let out_width: usize = in_width / 2;
    let out_height: usize = in_height / 2;
    let mut output: Vec<u8> = vec![0u8; out_width * out_height];
    for out_row in 0..out_height {
        let in_row: usize = out_row * 2;
        let above_row: usize = if in_row == 0 { 0 } else { in_row - 1 };
        let below_row: usize = (in_row + 2).min(in_height - 1);
        let r1_row: usize = (in_row + 1).min(in_height - 1);
        let r0: &[u8] = &plane[in_row * in_width..];
        let r1: &[u8] = &plane[r1_row * in_width..];
        let abv: &[u8] = &plane[above_row * in_width..];
        let blw: &[u8] = &plane[below_row * in_width..];
        let out: &mut [u8] = &mut output[out_row * out_width..];
        if out_width == 0 {
            continue;
        }
        {
            let r: usize = 2usize.min(in_width - 1);
            let membersum: i64 = r0[0] as i64 + r0[1] as i64 + r1[0] as i64 + r1[1] as i64;
            let mut neighsum: i64 = abv[0] as i64
                + abv[1] as i64
                + blw[0] as i64
                + blw[1] as i64
                + r0[0] as i64
                + r0[r] as i64
                + r1[0] as i64
                + r1[r] as i64;
            neighsum += neighsum;
            neighsum += abv[0] as i64 + abv[r] as i64 + blw[0] as i64 + blw[r] as i64;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[0] = result.clamp(0, 255) as u8;
        }
        let middle_end: usize = out_width.saturating_sub(1);
        for c_base in (2..middle_end * 2).step_by(2) {
            let membersum: i64 = r0[c_base] as i64
                + r0[c_base + 1] as i64
                + r1[c_base] as i64
                + r1[c_base + 1] as i64;
            let mut neighsum: i64 = abv[c_base] as i64
                + abv[c_base + 1] as i64
                + blw[c_base] as i64
                + blw[c_base + 1] as i64
                + r0[c_base - 1] as i64
                + r0[c_base + 2] as i64
                + r1[c_base - 1] as i64
                + r1[c_base + 2] as i64;
            neighsum += neighsum;
            neighsum += abv[c_base - 1] as i64
                + abv[c_base + 2] as i64
                + blw[c_base - 1] as i64
                + blw[c_base + 2] as i64;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[c_base / 2] = result.clamp(0, 255) as u8;
        }
        if out_width > 1 {
            let out_col: usize = out_width - 1;
            let c: usize = out_col * 2;
            let c1: usize = (c + 1).min(in_width - 1);
            let membersum: i64 = r0[c] as i64 + r0[c1] as i64 + r1[c] as i64 + r1[c1] as i64;
            let mut neighsum: i64 = abv[c] as i64
                + abv[c1] as i64
                + blw[c] as i64
                + blw[c1] as i64
                + r0[c - 1] as i64
                + r0[c1] as i64
                + r1[c - 1] as i64
                + r1[c1] as i64;
            neighsum += neighsum;
            neighsum += abv[c - 1] as i64 + abv[c1] as i64 + blw[c - 1] as i64 + blw[c1] as i64;
            let result: i64 = (membersum * memberscale + neighsum * neighscale + 32768) >> 16;
            out[out_col] = result.clamp(0, 255) as u8;
        }
    }
    output
}

/// Expand a component plane to the MCU grid exactly as C's prep controller
/// does, so the blocks that straddle the image edge see the same samples.
///
/// Horizontally this is `expand_right_edge` (`jcsample.c`): repeat the last
/// column. Vertically C pads twice — once on the input rows, to complete a row
/// group of `max_v` rows (`jcprepct.c:171-178`), and once on the *downsampled*
/// output, to fill the iMCU height by repeating the last output row
/// (`jcprepct.c:197-205`).
///
/// `row_group_height` is what carries that second pass back to full
/// resolution. A component sampled at the maximum downsamples 1:1, so
/// repeating its last output row is just repeating its last input row: pass 1.
/// A component subsampled `v` ways has one output row per `v` input rows, so
/// repeating the last output row means repeating the last complete *group* of
/// `v` input rows: pass `v`. Getting this backwards is invisible until the
/// height is a multiple of `v` but not of the MCU height, where group repeat
/// gives rows 16,17,16,17 and a plain clamp gives 17,17,17,17.
fn pad_plane_to_mcu_grid(
    plane: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    row_group_height: usize,
) -> Vec<u8> {
    if src_width == dst_width && src_height == dst_height {
        return plane.to_vec();
    }
    let mut padded: Vec<u8> = vec![0u8; dst_width * dst_height];
    for row in 0..src_height {
        let src_start: usize = row * src_width;
        let dst_start: usize = row * dst_width;
        padded[dst_start..dst_start + src_width]
            .copy_from_slice(&plane[src_start..src_start + src_width]);
        let last: u8 = plane[src_start + src_width - 1];
        padded[dst_start + src_width..dst_start + dst_width].fill(last);
    }
    if src_height < dst_height {
        let row_group_end: usize = src_height.div_ceil(row_group_height) * row_group_height;
        // Phase 1: repeat the last real row up to the row-group boundary.
        let last_row: Vec<u8> =
            padded[(src_height - 1) * dst_width..src_height * dst_width].to_vec();
        for row in src_height..row_group_end.min(dst_height) {
            let dst_start: usize = row * dst_width;
            padded[dst_start..dst_start + dst_width].copy_from_slice(&last_row);
        }
        // Phase 2: repeat the last complete row group.
        if row_group_end < dst_height {
            let group_start: usize = row_group_end - row_group_height;
            for row in row_group_end..dst_height {
                let src_row: usize = group_start + (row - row_group_end) % row_group_height;
                let src_start: usize = src_row * dst_width;
                let source: Vec<u8> = padded[src_start..src_start + dst_width].to_vec();
                let dst_start: usize = row * dst_width;
                padded[dst_start..dst_start + dst_width].copy_from_slice(&source);
            }
        }
    }
    padded
}

/// Encode a single 8x8 block through the DCT -> quantize -> Huffman pipeline.
#[allow(clippy::too_many_arguments)]
fn encode_single_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant_table: &QuantDivisors,
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // C jccoefct.c:178-199 — when a block is entirely outside the image
    // (subsampled MCU stride exceeds image dimensions), emit a dummy block:
    // DC = previous block's DC (so DC diff = 0), AC all zero. This matches
    // upstream's "Create a row of dummy blocks at the bottom of the image"
    // path and keeps the DC frequency distribution byte-identical to cjpeg
    // for non-444 subsamplings whose MCU height/width does not divide the
    // image dimensions evenly.
    if block_x >= plane_width || block_y >= plane_height {
        let mut dummy = [0i16; 64];
        dummy[0] = *prev_dc;
        HuffmanEncoder::encode_block(writer, &dummy, prev_dc, dc_table, ac_table);
        return;
    }

    let mut quantized = [0i16; 64];

    // The fused SIMD path uses islow FDCT internally. Skip it for ifast/float
    // so the caller-provided fdct_quantize_fn (with correct divisors) is used.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // Fused path for interior blocks: load u8 → FDCT → quantize → zigzag
    // without intermediate [i16; 64] buffer between extract and FDCT.
    if use_fused_simd && block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(block_y * plane_width + block_x),
                    plane_width,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                crate::simd::wasm32::wasm_extract_fdct_quantize(
                    plane.as_ptr().add(block_y * plane_width + block_x),
                    plane_width,
                    quant_table,
                    &mut quantized,
                );
            }
            HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
            return;
        }
    }

    // Border blocks: pad to a local 8×8 buffer with replicated-last-pixel,
    // then use the NEON/AVX2 fused path.  This ensures byte-identical output
    // with C libjpeg-turbo's expand_right_edge + NEON convsamp/fdct path.
    let is_edge: bool = block_x + 8 > plane_width || block_y + 8 > plane_height;
    if is_edge {
        let mut local_buf = [0u8; 64]; // 8×8 padded block
        for row in 0..8usize {
            let src_y: usize = (block_y + row).min(plane_height - 1);
            for col in 0..8usize {
                let src_x: usize = (block_x + col).min(plane_width - 1);
                local_buf[row * 8 + col] = plane[src_y * plane_width + src_x];
            }
        }

        if use_fused_simd {
            #[cfg(target_arch = "aarch64")]
            {
                unsafe {
                    crate::simd::aarch64::neon_extract_fdct_quantize(
                        local_buf.as_ptr(),
                        8,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        crate::simd::x86_64::avx2_extract_fdct_quantize(
                            local_buf.as_ptr(),
                            8,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                    return;
                }
            }
        }
    }

    // Generic path: extract block + caller-provided FDCT+quantize.
    // Used for ifast, float, and non-SIMD fallback.
    let mut block = [0i16; 64];
    extract_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        &mut block,
    );
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);
    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
}

/// Encode a full color MCU (multiple Y blocks + Cb + Cr blocks).
#[allow(clippy::too_many_arguments)]
fn encode_color_mcu(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    subsampling: Subsampling,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    match subsampling {
        Subsampling::S444 | Subsampling::Unknown => {
            // 1 Y + 1 Cb + 1 Cr = 3 blocks, MCU-level hoisting saves 2 begin/end pairs
            #[cfg(target_arch = "x86_64")]
            {
                encode_mcu_444_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    cb_plane,
                    width,
                    height,
                    x0,
                    y0,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S422 => {
            // 2 Y + 1 Cb + 1 Cr = 4 blocks, MCU-level hoisting saves 3 begin/end pairs
            #[cfg(target_arch = "x86_64")]
            {
                encode_mcu_422_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + 8,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_downsampled_chroma_block(
                    cb_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    1,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_downsampled_chroma_block(
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    1,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S420 => {
            // 4 Y blocks (2x2 arrangement) + 1 downsampled Cb + 1 downsampled Cr
            // Optimized path: do all FDCT+quantize first, then all Huffman encoding
            // with a single hoisted begin_block/end_block per MCU (saves 5 pairs).
            #[cfg(target_arch = "x86_64")]
            {
                encode_mcu_420_x86_64(
                    y_plane,
                    cb_plane,
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    chroma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_y,
                    prev_dc_cb,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                // Y blocks in order: top-left, top-right, bottom-left, bottom-right
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + 8,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0 + 8,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + 8,
                    y0 + 8,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
                // Downsample chroma: 2x2 box filter
                encode_downsampled_chroma_block(
                    cb_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    2,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_downsampled_chroma_block(
                    cr_plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    2,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S440 => {
            // 2 Y blocks vertically: (x0, y0) and (x0, y0+8)
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
                fdct_quantize_fn,
            );
            encode_single_block(
                y_plane,
                width,
                height,
                x0,
                y0 + 8,
                luma_quant,
                dc_luma_table,
                ac_luma_table,
                writer,
                prev_dc_y,
                fdct_quantize_fn,
            );
            // Cb/Cr downsampled 1x2
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                1,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
                fdct_quantize_fn,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                1,
                2,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
                fdct_quantize_fn,
            );
        }
        Subsampling::S411 => {
            // 4 Y blocks horizontally
            for i in 0..4 {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0 + i * 8,
                    y0,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
            // Cb/Cr downsampled 4x1
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                4,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
                fdct_quantize_fn,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                4,
                1,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
                fdct_quantize_fn,
            );
        }
        Subsampling::S441 => {
            // 4 Y blocks vertically
            for i in 0..4 {
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    x0,
                    y0 + i * 8,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
            // Cb/Cr downsampled 1x4
            encode_downsampled_chroma_block(
                cb_plane,
                width,
                height,
                x0,
                y0,
                1,
                4,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cb,
                fdct_quantize_fn,
            );
            encode_downsampled_chroma_block(
                cr_plane,
                width,
                height,
                x0,
                y0,
                1,
                4,
                chroma_quant,
                dc_chroma_table,
                ac_chroma_table,
                writer,
                prev_dc_cr,
                fdct_quantize_fn,
            );
        }
        Subsampling::S410 => {
            // 4 Y horizontal × 2 vertical = 8 luma blocks per MCU
            for dy in [0usize, 8] {
                for dx in [0usize, 8, 16, 24] {
                    encode_single_block(
                        y_plane,
                        width,
                        height,
                        x0 + dx,
                        y0 + dy,
                        luma_quant,
                        dc_luma_table,
                        ac_luma_table,
                        writer,
                        prev_dc_y,
                        fdct_quantize_fn,
                    );
                }
            }
            // Cb/Cr downsampled 4x2
            for (plane, prev_dc) in [(cb_plane, &mut *prev_dc_cb), (cr_plane, &mut *prev_dc_cr)] {
                encode_downsampled_chroma_block(
                    plane,
                    width,
                    height,
                    x0,
                    y0,
                    4,
                    2,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc,
                    fdct_quantize_fn,
                );
            }
        }
        Subsampling::S24 => {
            // 2 Y horizontal × 4 vertical = 8 luma blocks per MCU
            for dy in [0usize, 8, 16, 24] {
                for dx in [0usize, 8] {
                    encode_single_block(
                        y_plane,
                        width,
                        height,
                        x0 + dx,
                        y0 + dy,
                        luma_quant,
                        dc_luma_table,
                        ac_luma_table,
                        writer,
                        prev_dc_y,
                        fdct_quantize_fn,
                    );
                }
            }
            // Cb/Cr downsampled 2x4
            for (plane, prev_dc) in [(cb_plane, &mut *prev_dc_cb), (cr_plane, &mut *prev_dc_cr)] {
                encode_downsampled_chroma_block(
                    plane,
                    width,
                    height,
                    x0,
                    y0,
                    2,
                    4,
                    chroma_quant,
                    dc_chroma_table,
                    ac_chroma_table,
                    writer,
                    prev_dc,
                    fdct_quantize_fn,
                );
            }
        }
    }
}

/// Check if a Y block at the given pixel position is a dummy block
/// (beyond the real image boundary in either dimension).
/// C libjpeg-turbo creates dummy blocks with AC=0, DC=prev for these positions.
fn is_y_dummy(block_x_px: usize, block_y_px: usize, y_wib: usize, y_hib: usize) -> bool {
    block_x_px / 8 >= y_wib || block_y_px / 8 >= y_hib
}

/// Encode a dummy block (AC=0, DC=previous block's DC) matching C jccoefct.c.
fn encode_dummy_block(
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
) {
    let mut dummy: [i16; 64] = [0i16; 64];
    dummy[0] = *prev_dc;
    HuffmanEncoder::encode_block(writer, &dummy, prev_dc, dc_table, ac_table);
}

/// Encode a color MCU with dummy Y blocks for the last MCU column.
///
/// C libjpeg-turbo creates "dummy" blocks beyond `width_in_blocks`: all AC=0,
/// DC = previous block's DC (jccoefct.c lines 184-191). This produces smaller
/// output than FDCT'ing the padded pixel data.
#[allow(clippy::too_many_arguments)]
fn encode_color_mcu_with_dummies(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    subsampling: Subsampling,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    eff_col_width: usize,
    eff_row_height: usize,
) {
    let (h_samp, v_samp) = subsampling.sampling_factors();
    let y_mcu_width: usize = h_samp as usize;
    let y_mcu_height: usize = v_samp as usize;

    // Encode Y blocks: real blocks where vy < eff_row_height && hx < eff_col_width,
    // dummy blocks elsewhere (AC=0, DC=prev_dc, matching C jccoefct.c lines 184-199).
    for vy in 0..y_mcu_height {
        let is_dummy_row: bool = vy >= eff_row_height;
        for hx in 0..y_mcu_width {
            let is_dummy_col: bool = hx >= eff_col_width;
            if is_dummy_row || is_dummy_col {
                // Dummy block: AC=0, DC=previous block's DC
                let mut dummy = [0i16; 64];
                dummy[0] = *prev_dc_y;
                HuffmanEncoder::encode_block(
                    writer,
                    &dummy,
                    prev_dc_y,
                    dc_luma_table,
                    ac_luma_table,
                );
            } else {
                let bx: usize = x0 + hx * 8;
                let by: usize = y0 + vy * 8;
                encode_single_block(
                    y_plane,
                    width,
                    height,
                    bx,
                    by,
                    luma_quant,
                    dc_luma_table,
                    ac_luma_table,
                    writer,
                    prev_dc_y,
                    fdct_quantize_fn,
                );
            }
        }
    }

    // Chroma blocks: always encode normally (chroma MCU_width=1 for S422/S420)
    encode_downsampled_chroma_block(
        cb_plane,
        width,
        height,
        x0,
        y0,
        h_samp as usize,
        v_samp as usize,
        chroma_quant,
        dc_chroma_table,
        ac_chroma_table,
        writer,
        prev_dc_cb,
        fdct_quantize_fn,
    );
    encode_downsampled_chroma_block(
        cr_plane,
        width,
        height,
        x0,
        y0,
        h_samp as usize,
        v_samp as usize,
        chroma_quant,
        dc_chroma_table,
        ac_chroma_table,
        writer,
        prev_dc_cr,
        fdct_quantize_fn,
    );
}

/// Helper: FDCT+quantize a single block (interior: fused SIMD, border: scalar fallback).
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fdct_quantize_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    out: &mut [i16; 64],
) {
    if block_x + 8 <= plane_width
        && block_y + 8 <= plane_height
        && is_x86_feature_detected!("avx2")
        && may_use_islow_simd_kernel(fdct_quantize_fn)
    {
        unsafe {
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                plane.as_ptr().add(block_y * plane_width + block_x),
                plane_width,
                quant,
                out,
            );
        }
    } else {
        let mut block = [0i16; 64];
        extract_block(
            plane,
            plane_width,
            plane_height,
            block_x,
            block_y,
            &mut block,
        );
        fdct_quantize_fn(&mut block, quant, out);
    }
}

/// Helper: FDCT+quantize a downsampled H2V1 chroma block.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn fdct_quantize_chroma_h2v1(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
    out: &mut [i16; 64],
) {
    // Fused path: downsample + FDCT + quantize in one pass (AVX2)
    if block_x + 16 <= plane_width
        && block_y + 8 <= plane_height
        && is_x86_feature_detected!("avx2")
        && may_use_islow_simd_kernel(fdct_quantize_fn)
    {
        unsafe {
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                plane.as_ptr().add(block_y * plane_width + block_x),
                plane_width,
                quant,
                out,
            );
        }
        return;
    }
    // Separate downsample + FDCT (SSSE3 downsample only)
    if block_x + 16 <= plane_width
        && block_y + 8 <= plane_height
        && is_x86_feature_detected!("ssse3")
        && may_use_islow_simd_kernel(fdct_quantize_fn)
    {
        let mut block = [0i16; 64];
        unsafe {
            downsample_chroma_block_h2v1_ssse3(plane, plane_width, block_x, block_y, &mut block);
        }
        fdct_quantize_fn(&mut block, quant, out);
    } else {
        let mut block = [0i16; 64];
        downsample_chroma_block(
            plane,
            plane_width,
            plane_height,
            block_x,
            block_y,
            2,
            1,
            &mut block,
        );
        fdct_quantize_fn(&mut block, quant, out);
    }
}

/// Optimized 4:4:4 MCU encoding with MCU-level BitWriter hoisting.
///
/// 3 blocks (Y + Cb + Cr), saves 2 begin/end pairs per MCU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn encode_mcu_444_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 3] = [[0i16; 64]; 3];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool =
        is_x86_feature_detected!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);
    let interior: bool = x0 + 8 <= width && y0 + 8 <= height;

    if interior && has_avx2 {
        unsafe {
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_plane.as_ptr().add(y0 * width + x0),
                width,
                luma_quant,
                &mut q[0],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[2],
            );
        }
    } else {
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[0],
        );
        fdct_quantize_block(
            cb_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[1],
        );
        fdct_quantize_block(
            cr_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[2],
        );
    }

    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(1536);
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[0],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[1],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[2],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Optimized 4:2:2 MCU encoding with MCU-level BitWriter hoisting.
///
/// 4 blocks (2 Y + Cb + Cr), saves 3 begin/end pairs per MCU.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn encode_mcu_422_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 4] = [[0i16; 64]; 4];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool =
        is_x86_feature_detected!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);
    // Interior check: 2 Y blocks (16 wide) + H2V1 chroma (16 wide, 8 tall)
    let interior: bool = x0 + 16 <= width && y0 + 8 <= height;

    if interior && has_avx2 {
        unsafe {
            let y_ptr: *const u8 = y_plane.as_ptr().add(y0 * width + x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, width, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                width,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[3],
            );
        }
    } else {
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[0],
        );
        fdct_quantize_block(
            y_plane,
            width,
            height,
            x0 + 8,
            y0,
            luma_quant,
            fdct_quantize_fn,
            &mut q[1],
        );
        fdct_quantize_chroma_h2v1(
            cb_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[2],
        );
        fdct_quantize_chroma_h2v1(
            cr_plane,
            width,
            height,
            x0,
            y0,
            chroma_quant,
            fdct_quantize_fn,
            &mut q[3],
        );
    }

    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(2048);
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[0],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[1],
            prev_dc_y,
            dc_luma_table,
            ac_luma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[2],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[3],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Optimized 4:2:0 MCU encoding with MCU-level BitWriter hoisting.
///
/// Does all FDCT+quantize for 6 blocks first, then all Huffman encoding in one
/// hoisted begin_block/end_block pair. Saves 5 begin/end pairs per MCU.
/// 6 blocks × 128 bytes = 768 bytes of quantized data fits in L1.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
fn encode_mcu_420_x86_64(
    y_plane: &[u8],
    cb_plane: &[u8],
    cr_plane: &[u8],
    width: usize,
    height: usize,
    x0: usize,
    y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // Phase 1: FDCT + quantize all 6 blocks (4 Y + 1 Cb + 1 Cr)
    // Cache feature detection once per MCU (not per block).
    let mut q: [[i16; 64]; 6] = [[0i16; 64]; 6];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool =
        is_x86_feature_detected!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);

    // Check if all 4 Y blocks and both chroma blocks are interior (common case).
    // For 1080p with 16x16 MCUs, only edge MCUs fail this check.
    let interior: bool = x0 + 16 <= width && y0 + 16 <= height;

    if interior && has_avx2 {
        // Fast path: all blocks are interior, use fused SIMD for everything
        unsafe {
            let y_ptr: *const u8 = y_plane.as_ptr().add(y0 * width + x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, width, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                width,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * width),
                width,
                luma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * width + 8),
                width,
                luma_quant,
                &mut q[3],
            );
            crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                cb_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[4],
            );
            crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                cr_plane.as_ptr().add(y0 * width + x0),
                width,
                chroma_quant,
                &mut q[5],
            );
        }
    } else {
        // Slow path: handle edge MCUs with per-block bounds checking
        let y_offsets: [(usize, usize); 4] =
            [(x0, y0), (x0 + 8, y0), (x0, y0 + 8), (x0 + 8, y0 + 8)];
        for (idx, &(bx, by)) in y_offsets.iter().enumerate() {
            if has_avx2 && bx + 8 <= width && by + 8 <= height {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        y_plane.as_ptr().add(by * width + bx),
                        width,
                        luma_quant,
                        &mut q[idx],
                    );
                }
            } else {
                let mut block = [0i16; 64];
                extract_block(y_plane, width, height, bx, by, &mut block);
                fdct_quantize_fn(&mut block, luma_quant, &mut q[idx]);
            }
        }
        if has_avx2 && x0 + 16 <= width && y0 + 16 <= height {
            unsafe {
                crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                    cb_plane.as_ptr().add(y0 * width + x0),
                    width,
                    chroma_quant,
                    &mut q[4],
                );
            }
        } else {
            let mut block = [0i16; 64];
            downsample_chroma_block(cb_plane, width, height, x0, y0, 2, 2, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[4]);
        }
        if has_avx2 && x0 + 16 <= width && y0 + 16 <= height {
            unsafe {
                crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                    cr_plane.as_ptr().add(y0 * width + x0),
                    width,
                    chroma_quant,
                    &mut q[5],
                );
            }
        } else {
            let mut block = [0i16; 64];
            downsample_chroma_block(cr_plane, width, height, x0, y0, 2, 2, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[5]);
        }
    }

    // Phase 2: Huffman encode all 6 blocks with MCU-level hoisted state.
    // 3072 bytes = 6 blocks × 512 bytes worst-case per block.
    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(3072);

        // 4 Y blocks
        for block in q.iter().take(4) {
            HuffmanEncoder::encode_block_hoisted(
                &mut pb,
                &mut fb,
                &mut buf,
                block,
                prev_dc_y,
                dc_luma_table,
                ac_luma_table,
            );
        }
        // Cb
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[4],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        // Cr
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[5],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );

        writer.end_block(pb, fb, buf);
    }
}

/// Encode one 420 MCU using pre-downsampled half-resolution chroma buffers.
///
/// Y blocks are read from full-resolution `y_plane` (stride = `y_stride`).
/// Cb/Cr blocks are read from half-resolution buffers (stride = `chroma_stride`).
/// Since chroma is already downsampled, we use `avx2_extract_fdct_quantize`
/// instead of the heavier `avx2_downsample_h2v2_fdct_quantize`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments, dead_code)]
fn encode_mcu_420_half_chroma(
    y_plane: &[u8],
    y_stride: usize,
    cb_half: &[u8],
    cr_half: &[u8],
    chroma_stride: usize,
    y_x0: usize,
    y_y0: usize,
    chroma_x0: usize,
    chroma_y0: usize,
    luma_quant: &QuantDivisors,
    chroma_quant: &QuantDivisors,
    dc_luma_table: &HuffTable,
    ac_luma_table: &HuffTable,
    dc_chroma_table: &HuffTable,
    ac_chroma_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc_y: &mut i16,
    prev_dc_cb: &mut i16,
    prev_dc_cr: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    let mut q: [[i16; 64]; 6] = [[0i16; 64]; 6];
    // The AVX2 kernels below are islow-only; ifast/float carry divisors
    // scaled for their own transforms (#330).
    let has_avx2: bool =
        is_x86_feature_detected!("avx2") && may_use_islow_simd_kernel(fdct_quantize_fn);

    // Check if all blocks are interior (common case for non-edge MCUs)
    let y_interior: bool = y_x0 + 16 <= y_stride && y_y0 + 16 <= 16;
    let c_interior: bool = chroma_x0 + 8 <= chroma_stride && chroma_y0 + 8 <= 8;

    if y_interior && c_interior && has_avx2 {
        unsafe {
            // 4 Y blocks from full-res plane
            let y_ptr: *const u8 = y_plane.as_ptr().add(y_y0 * y_stride + y_x0);
            crate::simd::x86_64::avx2_extract_fdct_quantize(y_ptr, y_stride, luma_quant, &mut q[0]);
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8),
                y_stride,
                luma_quant,
                &mut q[1],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * y_stride),
                y_stride,
                luma_quant,
                &mut q[2],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                y_ptr.add(8 * y_stride + 8),
                y_stride,
                luma_quant,
                &mut q[3],
            );
            // Cb/Cr from half-res plane (already downsampled, just extract 8×8)
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cb_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                chroma_stride,
                chroma_quant,
                &mut q[4],
            );
            crate::simd::x86_64::avx2_extract_fdct_quantize(
                cr_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                chroma_stride,
                chroma_quant,
                &mut q[5],
            );
        }
    } else {
        // Fallback: scalar extract for edge MCUs
        let y_offsets: [(usize, usize); 4] = [
            (y_x0, y_y0),
            (y_x0 + 8, y_y0),
            (y_x0, y_y0 + 8),
            (y_x0 + 8, y_y0 + 8),
        ];
        for (idx, &(bx, by)) in y_offsets.iter().enumerate() {
            if has_avx2 && bx + 8 <= y_stride && by + 8 <= 16 {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        y_plane.as_ptr().add(by * y_stride + bx),
                        y_stride,
                        luma_quant,
                        &mut q[idx],
                    );
                }
            } else {
                let mut block = [0i16; 64];
                extract_block(y_plane, y_stride, 16, bx, by, &mut block);
                fdct_quantize_fn(&mut block, luma_quant, &mut q[idx]);
            }
        }
        // Chroma from half-res
        if has_avx2 && chroma_x0 + 8 <= chroma_stride && chroma_y0 + 8 <= 8 {
            unsafe {
                crate::simd::x86_64::avx2_extract_fdct_quantize(
                    cb_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                    chroma_stride,
                    chroma_quant,
                    &mut q[4],
                );
                crate::simd::x86_64::avx2_extract_fdct_quantize(
                    cr_half.as_ptr().add(chroma_y0 * chroma_stride + chroma_x0),
                    chroma_stride,
                    chroma_quant,
                    &mut q[5],
                );
            }
        } else {
            let mut block = [0i16; 64];
            extract_block(cb_half, chroma_stride, 8, chroma_x0, chroma_y0, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[4]);
            extract_block(cr_half, chroma_stride, 8, chroma_x0, chroma_y0, &mut block);
            fdct_quantize_fn(&mut block, chroma_quant, &mut q[5]);
        }
    }

    // Huffman encode all 6 blocks with MCU-level hoisted state
    unsafe {
        let (mut pb, mut fb, mut buf) = writer.begin_block(3072);
        for block in q.iter().take(4) {
            HuffmanEncoder::encode_block_hoisted(
                &mut pb,
                &mut fb,
                &mut buf,
                block,
                prev_dc_y,
                dc_luma_table,
                ac_luma_table,
            );
        }
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[4],
            prev_dc_cb,
            dc_chroma_table,
            ac_chroma_table,
        );
        HuffmanEncoder::encode_block_hoisted(
            &mut pb,
            &mut fb,
            &mut buf,
            &q[5],
            prev_dc_cr,
            dc_chroma_table,
            ac_chroma_table,
        );
        writer.end_block(pb, fb, buf);
    }
}

/// Encode a downsampled chroma block through the full pipeline.
#[allow(clippy::too_many_arguments)]
fn encode_downsampled_chroma_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    quant_table: &QuantDivisors,
    dc_table: &HuffTable,
    ac_table: &HuffTable,
    writer: &mut BitWriter,
    prev_dc: &mut i16,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) {
    // The fused SIMD paths use islow FDCT; skip for ifast/float.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // Fused NEON path: downsample + FDCT + quantize + zigzag in one pass,
    // eliminating the intermediate [i16; 64] downsampled block.
    #[cfg(target_arch = "aarch64")]
    if use_fused_simd {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if block_x + src_w <= plane_width && block_y + src_h <= plane_height {
            let plane_ptr: *const u8 =
                unsafe { plane.as_ptr().add(block_y * plane_width + block_x) };
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        plane_ptr,
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        plane_ptr,
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // x86_64 fused path: AVX2 downsample+FDCT+quantize+zigzag
    #[cfg(target_arch = "x86_64")]
    if use_fused_simd {
        let src_w: usize = 8 * h_factor;
        let src_h: usize = 8 * v_factor;
        if is_x86_feature_detected!("avx2")
            && block_x + src_w <= plane_width
            && block_y + src_h <= plane_height
        {
            // Fused downsample+FDCT+quantize for H2V2
            if h_factor == 2 && v_factor == 2 {
                let mut quantized = [0i16; 64];
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            // Fused downsample+FDCT+quantize for H2V1
            if h_factor == 2 && v_factor == 1 {
                let mut quantized = [0i16; 64];
                unsafe {
                    crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
    }

    // Edge block: pad source area locally and use NEON/AVX2 fused path.
    // This matches C libjpeg-turbo's expand_right_edge + SIMD downsample behavior.
    let src_w: usize = 8 * h_factor;
    let src_h: usize = 8 * v_factor;
    let mut local_buf = vec![0u8; src_w * src_h];
    for row in 0..src_h {
        let src_y: usize = (block_y + row).min(plane_height - 1);
        for col in 0..src_w {
            let src_x: usize = (block_x + col).min(plane_width - 1);
            local_buf[row * src_w + col] = plane[src_y * plane_width + src_x];
        }
    }

    // Try NEON/AVX2 fused downsample+FDCT+quantize on the padded local buffer
    if use_fused_simd {
        #[cfg(target_arch = "aarch64")]
        {
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                return;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut quantized = [0i16; 64];
                if h_factor == 2 && v_factor == 2 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                            local_buf.as_ptr(),
                            src_w,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                    return;
                }
                if h_factor == 2 && v_factor == 1 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                            local_buf.as_ptr(),
                            src_w,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
                    return;
                }
            }
        }
    }

    // Scalar fallback (non-SIMD platforms): downsample from padded buffer
    let mut block = [0i16; 64];
    downsample_chroma_block(
        &local_buf, src_w, src_h, 0, 0, h_factor, v_factor, &mut block,
    );

    let mut quantized = [0i16; 64];
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);

    HuffmanEncoder::encode_block(writer, &quantized, prev_dc, dc_table, ac_table);
}

/// Compress with optimized Huffman tables (2-pass encoding).
///
/// Pass 1: FDCT + quantize all blocks, gather symbol frequencies.
/// Two-pass optimized-Huffman encode: pass 1 gathers symbol statistics, pass 2
/// generates optimal tables and encodes with them. Produces smaller output
/// than `compress()` at the cost of an extra pass.
///
/// Public shim over [`compress_optimized_with_params`].
#[allow(clippy::too_many_arguments)]
pub fn compress_optimized(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    smoothing_factor: u8,
    dct_method: DctMethod,
    restart_interval: u16,
) -> Result<Vec<u8>> {
    compress_optimized_with_params(
        &CompressParams::new(pixels, width, height, pixel_format, quality, subsampling)
            .dct_method(dct_method)
            .restart_interval(restart_interval)
            .smoothing_factor(smoothing_factor)
            .optimize_huffman(true),
    )
}

/// Two-pass optimized-Huffman encode, and the only path that applies
/// `smoothing_factor` — it needs full-plane buffering.
///
/// Custom Huffman tables are deliberately ignored: `optimize_coding` derives
/// tables from the actual symbol statistics, which is the point of the pass,
/// and matches libjpeg when both are supplied.
pub fn compress_optimized_with_params(params: &CompressParams<'_>) -> Result<Vec<u8>> {
    let CompressParams {
        pixels,
        width,
        height,
        pixel_format,
        quality,
        subsampling,
        dct_method,
        restart_interval,
        custom_quant,
        custom_dc_huffman,
        custom_ac_huffman,
        optimize_huffman,
        smoothing_factor,
    } = *params;
    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp = pixel_format.bytes_per_pixel();
    let expected_size = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    // CMYK owns its own two-pass mode (#313). Neither `optimize_coding` nor
    // `smoothing_factor` is colorspace-gated in C — `jcmaster.c` and
    // `jcsample.c` both work per component — so rejecting four-component input
    // here made two builder options fail outright on it.
    if pixel_format == PixelFormat::Cmyk {
        return compress_cmyk(params);
    }

    let is_grayscale = pixel_format == PixelFormat::Grayscale;

    // Generate quantization tables. The divisor table layout depends on the
    // chosen FDCT — ifast pre-applies AA&N scaling so its divisors fold the
    // AA&N constants in (paired with `fdct_ifast_raw`); islow/float keep
    // the simple `quant * 8` divisors, with the float path routing through
    // the embedded `float_divisors` field via `scalar_fdct_float_quantize`.
    let luma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[0]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality),
    };
    let chroma_quant: [u16; 64] = match custom_quant.and_then(|tables| tables[1]) {
        Some(table) => table,
        None => tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality),
    };
    let luma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&luma_quant)
    } else {
        scale_quant_for_fdct(&luma_quant)
    };
    let chroma_divisors = if dct_method == DctMethod::IsFast {
        scale_quant_for_ifast(&chroma_quant)
    } else {
        scale_quant_for_fdct(&chroma_quant)
    };

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // FDCT dispatch: SIMD fused islow for the default path, scalar ifast/float
    // for the legacy paths. Only the islow scalar form is byte-equivalent to
    // the NEON/AVX2 fused kernels, so the per-block SIMD shortcuts inside
    // `gather_block` must be skipped when `dct_method != IsLow`.
    let fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]) = match dct_method {
        DctMethod::IsLow => enc_simd.fdct_quantize,
        DctMethod::IsFast => crate::simd::scalar::scalar_fdct_ifast_quantize,
        DctMethod::Float => crate::simd::scalar::scalar_fdct_float_quantize,
    };

    // Determine MCU dimensions
    let (mcu_w, mcu_h) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };

    let mcus_x = width.div_ceil(mcu_w);
    let mcus_y = height.div_ceil(mcu_h);
    let padded_w: usize = mcus_x * mcu_w;
    let padded_h: usize = mcus_y * mcu_h;

    // Color convert with MCU-aligned padding (matches C expand_right_edge)
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr_padded(
        pixels,
        width,
        height,
        padded_w,
        padded_h,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
        mcu_h / 8,
    )?;

    // Apply smoothing to component planes when smoothing_factor > 0.
    //
    // C selects `fullsize_smooth_downsample` for every component sampled at
    // the maximum factors (`jcsample.c:506-513`), which for a single-component
    // image is the grayscale plane itself — `cjpeg -grayscale -smooth 50`
    // demonstrably differs from `-smooth 0`. Excluding grayscale here made
    // `Encoder::smoothing_factor` a silent no-op for it (#327).
    let y_plane: Vec<u8> = if smoothing_factor > 0 {
        fullsize_smooth_plane(&y_plane, padded_w, padded_h, smoothing_factor)
    } else {
        y_plane
    };
    let use_smooth_chroma: bool =
        smoothing_factor > 0 && !is_grayscale && subsampling == Subsampling::S420;
    let cb_smooth: Vec<u8>;
    let cr_smooth: Vec<u8>;
    if use_smooth_chroma {
        cb_smooth = h2v2_smooth_downsample_plane(&cb_plane, padded_w, padded_h, smoothing_factor);
        cr_smooth = h2v2_smooth_downsample_plane(&cr_plane, padded_w, padded_h, smoothing_factor);
    } else {
        cb_smooth = Vec::new();
        cr_smooth = Vec::new();
    }

    // Shadow width/height with padded values so all encode loops use padded planes.
    // The planes are already padded to padded_w × padded_h by convert_to_ycbcr_padded.
    let original_width: usize = width;
    let original_height: usize = height;
    let width: usize = padded_w;
    let height: usize = padded_h;

    // Dummy block detection: C creates dummy blocks (AC=0, DC=prev) for Y blocks
    // beyond width_in_blocks/height_in_blocks (jccoefct.c lines 184-199).
    let y_width_in_blocks: usize = original_width.div_ceil(8);
    let y_height_in_blocks: usize = original_height.div_ceil(8);

    // === Pass 1: FDCT + quantize all blocks, gather symbol frequencies ===
    use crate::encode::huff_opt;

    // Frequency arrays: DC lum, DC chr, AC lum, AC chr
    let mut dc_luma_freq = [0u32; 257];
    let mut dc_chroma_freq = [0u32; 257];
    let mut ac_luma_freq = [0u32; 257];
    let mut ac_chroma_freq = [0u32; 257];

    // Buffer all quantized blocks for pass 2
    let mut all_blocks: Vec<[i16; 64]> = Vec::new();

    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    let ri_gather: u32 = restart_interval as u32;
    let mut mcu_count_gather: u32 = 0;

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            // Reset DC predictors at restart boundary so the gathered DC
            // diff symbol categories match what pass 2 will actually emit.
            // Without this the optimised Huffman tables would diverge from
            // cjpeg whenever `-r N` is set.
            if ri_gather > 0 && mcu_count_gather > 0 && mcu_count_gather.is_multiple_of(ri_gather) {
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
            }
            let x0 = mcu_col * mcu_w;
            let y0 = mcu_row * mcu_h;

            if is_grayscale {
                let q = gather_block(
                    &y_plane,
                    width,
                    height,
                    x0,
                    y0,
                    &luma_divisors,
                    fdct_quantize_fn,
                );
                let diff = q[0] - prev_dc_y;
                prev_dc_y = q[0];
                huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                huff_opt::gather_ac_symbols(&q, &mut ac_luma_freq);
                all_blocks.push(q);
            } else {
                match subsampling {
                    Subsampling::S444 | Subsampling::Unknown => {
                        // 1 Y + 1 Cb + 1 Cr
                        let yq = gather_block(
                            &y_plane,
                            width,
                            height,
                            x0,
                            y0,
                            &luma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = yq[0] - prev_dc_y;
                        prev_dc_y = yq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                        huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                        all_blocks.push(yq);

                        let cbq = gather_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S422 => {
                        // 2 Y blocks + 1 Cb + 1 Cr
                        // y_width_in_blocks = ceil(original_width / 8)
                        let y_wib: usize = original_width.div_ceil(8);
                        for dx in [0usize, 8] {
                            let block_col: usize = (x0 + dx) / 8;
                            let yq = if block_col >= y_wib {
                                // Dummy block: AC=0, DC=prev (C jccoefct.c lines 184-191)
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y;
                                dummy
                            } else {
                                gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                )
                            };
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            1,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            1,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S420 => {
                        // 4 Y blocks + 1 Cb + 1 Cr
                        for (dx, dy) in [(0, 0), (8, 0), (0, 8), (8, 8)] {
                            let block_col: usize = (x0 + dx) / 8;
                            let block_row: usize = (y0 + dy) / 8;
                            let yq = if block_col >= y_width_in_blocks
                                || block_row >= y_height_in_blocks
                            {
                                let mut dummy = [0i16; 64];
                                dummy[0] = prev_dc_y;
                                dummy
                            } else {
                                gather_block(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                )
                            };
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = if use_smooth_chroma {
                            gather_block(
                                &cb_smooth,
                                width / 2,
                                height / 2,
                                x0 / 2,
                                y0 / 2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        } else {
                            gather_downsampled_block(
                                &cb_plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        };
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = if use_smooth_chroma {
                            gather_block(
                                &cr_smooth,
                                width / 2,
                                height / 2,
                                x0 / 2,
                                y0 / 2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        } else {
                            gather_downsampled_block(
                                &cr_plane,
                                width,
                                height,
                                x0,
                                y0,
                                2,
                                2,
                                &chroma_divisors,
                                fdct_quantize_fn,
                            )
                        };
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S440 => {
                        for dy in [0usize, 8] {
                            let yq = gather_block_or_dummy(
                                &y_plane,
                                width,
                                height,
                                x0,
                                y0 + dy,
                                original_width,
                                original_height,
                                prev_dc_y,
                                &luma_divisors,
                                fdct_quantize_fn,
                            );
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            1,
                            2,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            1,
                            2,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S411 => {
                        // 4 Y blocks horizontally
                        for dx in [0usize, 8, 16, 24] {
                            let yq = gather_block_or_dummy(
                                &y_plane,
                                width,
                                height,
                                x0 + dx,
                                y0,
                                original_width,
                                original_height,
                                prev_dc_y,
                                &luma_divisors,
                                fdct_quantize_fn,
                            );
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            4,
                            1,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            4,
                            1,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S441 => {
                        // 4 Y blocks vertically
                        for dy in [0usize, 8, 16, 24] {
                            let yq = gather_block_or_dummy(
                                &y_plane,
                                width,
                                height,
                                x0,
                                y0 + dy,
                                original_width,
                                original_height,
                                prev_dc_y,
                                &luma_divisors,
                                fdct_quantize_fn,
                            );
                            let diff = yq[0] - prev_dc_y;
                            prev_dc_y = yq[0];
                            huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                            huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                            all_blocks.push(yq);
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            1,
                            4,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            1,
                            4,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S410 => {
                        // 4 Y horizontal × 2 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8] {
                            for dx in [0usize, 8, 16, 24] {
                                let yq = gather_block_or_dummy(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    original_width,
                                    original_height,
                                    prev_dc_y,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                let diff = yq[0] - prev_dc_y;
                                prev_dc_y = yq[0];
                                huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                                huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                                all_blocks.push(yq);
                            }
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            4,
                            2,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            4,
                            2,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                    Subsampling::S24 => {
                        // 2 Y horizontal × 4 vertical = 8 luma blocks per MCU
                        for dy in [0usize, 8, 16, 24] {
                            for dx in [0usize, 8] {
                                let yq = gather_block_or_dummy(
                                    &y_plane,
                                    width,
                                    height,
                                    x0 + dx,
                                    y0 + dy,
                                    original_width,
                                    original_height,
                                    prev_dc_y,
                                    &luma_divisors,
                                    fdct_quantize_fn,
                                );
                                let diff = yq[0] - prev_dc_y;
                                prev_dc_y = yq[0];
                                huff_opt::gather_dc_symbol(diff, &mut dc_luma_freq);
                                huff_opt::gather_ac_symbols(&yq, &mut ac_luma_freq);
                                all_blocks.push(yq);
                            }
                        }
                        let cbq = gather_downsampled_block(
                            &cb_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            4,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = cbq[0] - prev_dc_cb;
                        prev_dc_cb = cbq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&cbq, &mut ac_chroma_freq);
                        all_blocks.push(cbq);

                        let crq = gather_downsampled_block(
                            &cr_plane,
                            width,
                            height,
                            x0,
                            y0,
                            2,
                            4,
                            &chroma_divisors,
                            fdct_quantize_fn,
                        );
                        let diff = crq[0] - prev_dc_cr;
                        prev_dc_cr = crq[0];
                        huff_opt::gather_dc_symbol(diff, &mut dc_chroma_freq);
                        huff_opt::gather_ac_symbols(&crq, &mut ac_chroma_freq);
                        all_blocks.push(crq);
                    }
                }
            }
            mcu_count_gather = mcu_count_gather.wrapping_add(1);
        }
    }

    // Add pseudo-symbol (required for optimal table generation)
    dc_luma_freq[256] = 1;
    ac_luma_freq[256] = 1;
    dc_chroma_freq[256] = 1;
    ac_chroma_freq[256] = 1;

    // Huffman tables: derived from the gathered statistics when optimization
    // was asked for, otherwise the caller's custom tables (or Annex K).
    //
    // Reaching this function does not by itself imply optimization — a
    // `smoothing_factor` alone routes here too, because smoothing needs the
    // full-plane buffering this path provides. Unconditionally deriving
    // optimal tables would then silently override custom Huffman tables that
    // the caller supplied alongside smoothing (#322).
    let resolved: ResolvedHuffman = if optimize_huffman {
        let (dc_luma_bits, dc_luma_values) = huff_opt::gen_optimal_table(&dc_luma_freq);
        let (ac_luma_bits, ac_luma_values) = huff_opt::gen_optimal_table(&ac_luma_freq);
        let (dc_chroma_bits, dc_chroma_values) = huff_opt::gen_optimal_table(&dc_chroma_freq);
        let (ac_chroma_bits, ac_chroma_values) = huff_opt::gen_optimal_table(&ac_chroma_freq);
        ResolvedHuffman {
            dc_luma: build_huff_table(&dc_luma_bits, &dc_luma_values),
            ac_luma: build_huff_table(&ac_luma_bits, &ac_luma_values),
            dc_chroma: build_huff_table(&dc_chroma_bits, &dc_chroma_values),
            ac_chroma: build_huff_table(&ac_chroma_bits, &ac_chroma_values),
            dc_luma_bits,
            dc_luma_values,
            ac_luma_bits,
            ac_luma_values,
            dc_chroma_bits,
            dc_chroma_values,
            ac_chroma_bits,
            ac_chroma_values,
        }
    } else {
        ResolvedHuffman::resolve(custom_dc_huffman, custom_ac_huffman)
    };
    let ResolvedHuffman {
        dc_luma: dc_luma_table,
        ac_luma: ac_luma_table,
        dc_chroma: dc_chroma_table,
        ac_chroma: ac_chroma_table,
        dc_luma_bits,
        dc_luma_values,
        ac_luma_bits,
        ac_luma_values,
        dc_chroma_bits,
        dc_chroma_values,
        ac_chroma_bits,
        ac_chroma_values,
    } = resolved;

    // === Pass 2: Encode all buffered blocks with optimal tables ===
    let mut bit_writer = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    let mut block_idx = 0;
    let ri_enc: u32 = restart_interval as u32;
    let mut mcu_count_enc: u32 = 0;
    let mut rst_count_enc: u8 = 0;

    for _mcu_row in 0..mcus_y {
        for _mcu_col in 0..mcus_x {
            if ri_enc > 0 && mcu_count_enc > 0 && mcu_count_enc.is_multiple_of(ri_enc) {
                // Insert RST marker, reset DC predictors per
                // C jchuff.c::flush_packet.
                bit_writer.flush_restart();
                bit_writer.write_restart_marker(rst_count_enc);
                rst_count_enc = (rst_count_enc + 1) & 7;
                prev_dc_y = 0;
                prev_dc_cb = 0;
                prev_dc_cr = 0;
            }
            if is_grayscale {
                HuffmanEncoder::encode_block(
                    &mut bit_writer,
                    &all_blocks[block_idx],
                    &mut prev_dc_y,
                    &dc_luma_table,
                    &ac_luma_table,
                );
                block_idx += 1;
            } else {
                match subsampling {
                    Subsampling::S444 | Subsampling::Unknown => {
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_y,
                            &dc_luma_table,
                            &ac_luma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S422 => {
                        for _ in 0..2 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S420 => {
                        for _ in 0..4 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S440 => {
                        for _ in 0..2 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S411 | Subsampling::S441 => {
                        for _ in 0..4 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                    Subsampling::S410 | Subsampling::S24 => {
                        // 8 Y blocks + 1 Cb + 1 Cr per MCU (h*v = 4*2 or 2*4)
                        for _ in 0..8 {
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &all_blocks[block_idx],
                                &mut prev_dc_y,
                                &dc_luma_table,
                                &ac_luma_table,
                            );
                            block_idx += 1;
                        }
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cb,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                        HuffmanEncoder::encode_block(
                            &mut bit_writer,
                            &all_blocks[block_idx],
                            &mut prev_dc_cr,
                            &dc_chroma_table,
                            &ac_chroma_table,
                        );
                        block_idx += 1;
                    }
                }
            }
            mcu_count_enc = mcu_count_enc.wrapping_add(1);
        }
    }

    bit_writer.flush();

    // Assemble output with optimal DHT markers
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // Frame header
    if is_grayscale {
        let components = vec![(1, 1, 1, 0)];
        marker_writer::write_sof0(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
    } else {
        let (h_samp, v_samp) = subsampling.sampling_factors();
        let components = vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof0(
            &mut output,
            original_width as u16,
            original_height as u16,
            &components,
        );
    }

    // Write optimal Huffman tables
    marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);
    marker_writer::write_dht(&mut output, 1, 0, &ac_luma_bits, &ac_luma_values);
    if !is_grayscale {
        marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);
        marker_writer::write_dht(&mut output, 1, 1, &ac_chroma_bits, &ac_chroma_values);
    }

    // DRI marker — emitted from `write_scan_header` in C
    // (jcmarker.c::emit_dri), i.e. right before SOS in the only scan.
    if restart_interval > 0 {
        marker_writer::write_dri(&mut output, restart_interval);
    }

    // Scan header
    if is_grayscale {
        let scan_components = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components = vec![(1, 0, 0), (2, 1, 1), (3, 1, 1)];
        marker_writer::write_sos(&mut output, &scan_components);
    }

    // Entropy-coded data
    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// FDCT + quantize a single block, return the quantized coefficients.
/// Like `gather_block` but emits a dummy block (DC = prev_dc, AC = 0) when
/// the block is entirely outside the original (un-padded) image dimensions.
///
/// Mirrors C `jccoefct.c` lines 178-199: for the right- and bottom-edge MCUs
/// of subsampled formats (samp422, 420, 440, 411, 441, 410, 24), some Y
/// blocks may sit fully outside the image. C handles these by zeroing the AC
/// coefficients and copying the previous block's DC into [0][0], producing a
/// DC-diff of zero. Replicating this is essential for byte-parity with cjpeg
/// since the dummy DC=0 entries change the optimised Huffman frequency
/// distribution (and thus the resulting per-image DHT).
#[allow(clippy::too_many_arguments)]
fn gather_block_or_dummy(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    orig_width: usize,
    orig_height: usize,
    prev_dc: i16,
    quant_table: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> [i16; 64] {
    if block_x >= orig_width || block_y >= orig_height {
        let mut dummy = [0i16; 64];
        dummy[0] = prev_dc;
        return dummy;
    }
    gather_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        quant_table,
        fdct_quantize_fn,
    )
}

fn gather_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    quant_table: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> [i16; 64] {
    let mut quantized = [0i16; 64];

    // The fused NEON/AVX2 kernels here always run the islow FDCT internally.
    // For ifast / float methods the caller supplies a scalar `fdct_quantize_fn`
    // that pairs the matching FDCT with its method-specific divisors, so the
    // SIMD shortcuts must be bypassed to avoid silently downgrading to islow.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // NEON/AVX2 fused path for interior blocks
    if use_fused_simd && block_x + 8 <= plane_width && block_y + 8 <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    plane.as_ptr().add(block_y * plane_width + block_x),
                    plane_width,
                    quant_table,
                    &mut quantized,
                );
            }
            return quantized;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
    }

    // Edge blocks: pad to 8×8 then use NEON/AVX2
    let is_edge: bool = block_x + 8 > plane_width || block_y + 8 > plane_height;
    if use_fused_simd && is_edge {
        let mut local_buf = [0u8; 64];
        for row in 0..8usize {
            let src_y = (block_y + row).min(plane_height - 1);
            for col in 0..8usize {
                let src_x = (block_x + col).min(plane_width - 1);
                local_buf[row * 8 + col] = plane[src_y * plane_width + src_x];
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                crate::simd::aarch64::neon_extract_fdct_quantize(
                    local_buf.as_ptr(),
                    8,
                    quant_table,
                    &mut quantized,
                );
            }
            return quantized;
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    crate::simd::x86_64::avx2_extract_fdct_quantize(
                        local_buf.as_ptr(),
                        8,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
    }

    // Fallback: extract_block (with SSE2 for interior) + fdct_quantize
    let mut block = [0i16; 64];
    extract_block(
        plane,
        plane_width,
        plane_height,
        block_x,
        block_y,
        &mut block,
    );
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);
    quantized
}

/// FDCT + quantize a downsampled chroma block, return the quantized coefficients.
#[allow(clippy::too_many_arguments)]
fn gather_downsampled_block(
    plane: &[u8],
    plane_width: usize,
    plane_height: usize,
    block_x: usize,
    block_y: usize,
    h_factor: usize,
    v_factor: usize,
    quant_table: &QuantDivisors,
    fdct_quantize_fn: fn(&mut [i16; 64], &QuantDivisors, &mut [i16; 64]),
) -> [i16; 64] {
    let src_w: usize = 8 * h_factor;
    let src_h: usize = 8 * v_factor;

    // The fused downsample+FDCT NEON/AVX2 kernels here use islow internally;
    // bypass them when the caller asked for ifast/float so the supplied
    // `fdct_quantize_fn` (with method-matching divisors) is used instead.
    let use_fused_simd: bool = may_use_islow_simd_kernel(fdct_quantize_fn);

    // NEON/AVX2 fused downsample+FDCT+quantize for interior blocks
    if use_fused_simd && block_x + src_w <= plane_width && block_y + src_h <= plane_height {
        #[cfg(target_arch = "aarch64")]
        {
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        plane.as_ptr().add(block_y * plane_width + block_x),
                        plane_width,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut quantized = [0i16; 64];
                if h_factor == 2 && v_factor == 2 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                            plane.as_ptr().add(block_y * plane_width + block_x),
                            plane_width,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    return quantized;
                }
                if h_factor == 2 && v_factor == 1 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                            plane.as_ptr().add(block_y * plane_width + block_x),
                            plane_width,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    return quantized;
                }
            }
        }
    }

    // Edge block: pad source area locally and use NEON/AVX2
    let mut local_buf = vec![0u8; src_w * src_h];
    for row in 0..src_h {
        let src_y = (block_y + row).min(plane_height - 1);
        for col in 0..src_w {
            let src_x = (block_x + col).min(plane_width - 1);
            local_buf[row * src_w + col] = plane[src_y * plane_width + src_x];
        }
    }
    if use_fused_simd {
        #[cfg(target_arch = "aarch64")]
        {
            let mut quantized = [0i16; 64];
            if h_factor == 2 && v_factor == 2 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v2_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
            if h_factor == 2 && v_factor == 1 {
                unsafe {
                    crate::simd::aarch64::neon_downsample_h2v1_fdct_quantize(
                        local_buf.as_ptr(),
                        src_w,
                        quant_table,
                        &mut quantized,
                    );
                }
                return quantized;
            }
        }
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let mut quantized = [0i16; 64];
                if h_factor == 2 && v_factor == 2 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v2_fdct_quantize(
                            local_buf.as_ptr(),
                            src_w,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    return quantized;
                }
                if h_factor == 2 && v_factor == 1 {
                    unsafe {
                        crate::simd::x86_64::avx2_downsample_h2v1_fdct_quantize(
                            local_buf.as_ptr(),
                            src_w,
                            quant_table,
                            &mut quantized,
                        );
                    }
                    return quantized;
                }
            }
        }
    }

    // Scalar fallback
    let mut block = [0i16; 64];
    downsample_chroma_block(
        &local_buf, src_w, src_h, 0, 0, h_factor, v_factor, &mut block,
    );
    let mut quantized = [0i16; 64];
    fdct_quantize_fn(&mut block, quant_table, &mut quantized);
    quantized
}

/// Compress JPEG from raw downsampled component planes.
///
/// Bypasses color conversion and chroma downsampling — the caller provides
/// data already in the YCbCr color space at the correct subsampled dimensions.
/// This matches libjpeg-turbo's `jpeg_write_raw_data()` functionality.
#[allow(clippy::too_many_arguments)]
pub fn compress_raw(
    planes: &[&[u8]],
    plane_widths: &[usize],
    plane_heights: &[usize],
    image_width: usize,
    image_height: usize,
    quality: u8,
    subsampling: Subsampling,
) -> Result<Vec<u8>> {
    if image_width == 0 || image_height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if planes.len() != plane_widths.len() || planes.len() != plane_heights.len() {
        return Err(JpegError::CorruptData(
            "planes, plane_widths, and plane_heights must have the same length".to_string(),
        ));
    }
    let is_grayscale: bool = planes.len() == 1;
    if is_grayscale && subsampling != Subsampling::S444 {
        return Err(JpegError::CorruptData(format!(
            "1 plane (grayscale) is only valid with S444 subsampling, got {:?}",
            subsampling
        )));
    }
    if !is_grayscale && planes.len() != 3 {
        return Err(JpegError::CorruptData(format!(
            "expected 1 (grayscale) or 3 (YCbCr) planes, got {}",
            planes.len()
        )));
    }
    let (h_samp, v_samp): (u8, u8) = subsampling.sampling_factors();
    if !is_grayscale {
        let expected_cb_w: usize = image_width.div_ceil(h_samp as usize);
        let expected_cb_h: usize = image_height.div_ceil(v_samp as usize);
        if plane_widths[0] != image_width || plane_heights[0] != image_height {
            return Err(JpegError::CorruptData(format!(
                "Y plane dimensions {}x{} do not match image dimensions {}x{}",
                plane_widths[0], plane_heights[0], image_width, image_height
            )));
        }
        for comp_idx in 1..3 {
            let comp_name: &str = if comp_idx == 1 { "Cb" } else { "Cr" };
            if plane_widths[comp_idx] != expected_cb_w || plane_heights[comp_idx] != expected_cb_h {
                return Err(JpegError::CorruptData(format!(
                    "{} plane dimensions {}x{} do not match expected {}x{} for {:?} subsampling",
                    comp_name,
                    plane_widths[comp_idx],
                    plane_heights[comp_idx],
                    expected_cb_w,
                    expected_cb_h,
                    subsampling
                )));
            }
        }
    }
    for (i, plane) in planes.iter().enumerate() {
        let expected_size: usize = plane_widths[i] * plane_heights[i];
        if plane.len() < expected_size {
            return Err(JpegError::BufferTooSmall {
                need: expected_size,
                got: plane.len(),
            });
        }
    }
    let luma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors: QuantDivisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors: QuantDivisors = scale_quant_for_fdct(&chroma_quant);
    let dc_luma_table: HuffTable =
        build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table: HuffTable =
        build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table: HuffTable =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table: HuffTable =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);
    let (mcu_w, mcu_h): (usize, usize) = if is_grayscale {
        (8, 8)
    } else {
        match subsampling {
            Subsampling::S444 | Subsampling::Unknown => (8, 8),
            Subsampling::S422 => (16, 8),
            Subsampling::S420 => (16, 16),
            Subsampling::S440 => (8, 16),
            Subsampling::S411 => (32, 8),
            Subsampling::S441 => (8, 32),
            Subsampling::S410 => (32, 16),
            Subsampling::S24 => (16, 32),
        }
    };
    let mcus_x: usize = image_width.div_ceil(mcu_w);
    let mcus_y: usize = image_height.div_ceil(mcu_h);
    let enc_simd = crate::simd::detect_encoder();
    let fdct_quantize_fn = enc_simd.fdct_quantize;
    let mut bit_writer: BitWriter = BitWriter::new(image_width * image_height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;
    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;
            if is_grayscale {
                encode_single_block(
                    planes[0],
                    plane_widths[0],
                    plane_heights[0],
                    x0,
                    y0,
                    &luma_divisors,
                    &dc_luma_table,
                    &ac_luma_table,
                    &mut bit_writer,
                    &mut prev_dc_y,
                    fdct_quantize_fn,
                );
            } else {
                let h: usize = h_samp as usize;
                let v: usize = v_samp as usize;
                for vy in 0..v {
                    for hx in 0..h {
                        encode_single_block(
                            planes[0],
                            plane_widths[0],
                            plane_heights[0],
                            x0 + hx * 8,
                            y0 + vy * 8,
                            &luma_divisors,
                            &dc_luma_table,
                            &ac_luma_table,
                            &mut bit_writer,
                            &mut prev_dc_y,
                            fdct_quantize_fn,
                        );
                    }
                }
                let chroma_x: usize = x0 / h;
                let chroma_y: usize = y0 / v;
                encode_single_block(
                    planes[1],
                    plane_widths[1],
                    plane_heights[1],
                    chroma_x,
                    chroma_y,
                    &chroma_divisors,
                    &dc_chroma_table,
                    &ac_chroma_table,
                    &mut bit_writer,
                    &mut prev_dc_cb,
                    fdct_quantize_fn,
                );
                encode_single_block(
                    planes[2],
                    plane_widths[2],
                    plane_heights[2],
                    chroma_x,
                    chroma_y,
                    &chroma_divisors,
                    &dc_chroma_table,
                    &ac_chroma_table,
                    &mut bit_writer,
                    &mut prev_dc_cr,
                    fdct_quantize_fn,
                );
            }
        }
    }
    bit_writer.flush();
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);
    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }
    if is_grayscale {
        let components: Vec<(u8, u8, u8, u8)> = vec![(1, 1, 1, 0)];
        marker_writer::write_sof0(
            &mut output,
            image_width as u16,
            image_height as u16,
            &components,
        );
    } else {
        let components: Vec<(u8, u8, u8, u8)> =
            vec![(1, h_samp, v_samp, 0), (2, 1, 1, 1), (3, 1, 1, 1)];
        marker_writer::write_sof0(
            &mut output,
            image_width as u16,
            image_height as u16,
            &components,
        );
    }
    marker_writer::write_dht(
        &mut output,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );
    marker_writer::write_dht(
        &mut output,
        1,
        0,
        &tables::AC_LUMINANCE_BITS,
        &tables::AC_LUMINANCE_VALUES,
    );
    if !is_grayscale {
        marker_writer::write_dht(
            &mut output,
            0,
            1,
            &tables::DC_CHROMINANCE_BITS,
            &tables::DC_CHROMINANCE_VALUES,
        );
        marker_writer::write_dht(
            &mut output,
            1,
            1,
            &tables::AC_CHROMINANCE_BITS,
            &tables::AC_CHROMINANCE_VALUES,
        );
    }
    if is_grayscale {
        marker_writer::write_sos(&mut output, &[(1, 0, 0)]);
    } else {
        marker_writer::write_sos(&mut output, &[(1, 0, 0), (2, 1, 1), (3, 1, 1)]);
    }
    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);
    Ok(output)
}

/// Compress raw pixel data into a JPEG byte stream using explicit per-component
/// sampling factors instead of the predefined `Subsampling` enum.
///
/// This supports non-standard sampling configurations such as 3x2, 3x1, 1x3,
/// and 4x2 that are not covered by the standard Subsampling enum values.
///
/// # Arguments
/// * `pixels` - Raw pixel data in the format specified by `pixel_format`
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `pixel_format` - Pixel format of the input data
/// * `quality` - JPEG quality factor (1-100)
/// * `factors` - Per-component `(h_sampling, v_sampling)` factors
///
/// # Returns
/// A `Vec<u8>` containing the complete JPEG file data.
pub fn compress_custom_sampling(
    pixels: &[u8],
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    quality: u8,
    factors: &[(u8, u8)],
) -> Result<Vec<u8>> {
    // Validate inputs
    if width == 0 || height == 0 {
        return Err(JpegError::CorruptData(
            "image dimensions must be non-zero".to_string(),
        ));
    }
    if width > 65535 || height > 65535 {
        return Err(JpegError::CorruptData(format!(
            "JPEG dimensions must be <= 65535, got {}x{}",
            width, height
        )));
    }

    let bpp: usize = pixel_format.bytes_per_pixel();
    let expected_size: usize = width * height * bpp;
    if pixels.len() < expected_size {
        return Err(JpegError::BufferTooSmall {
            need: expected_size,
            got: pixels.len(),
        });
    }

    let is_grayscale: bool = pixel_format == PixelFormat::Grayscale;

    // Validate factor count matches component count
    let num_components: usize = if is_grayscale { 1 } else { 3 };
    if factors.len() != num_components {
        return Err(JpegError::CorruptData(format!(
            "expected {} sampling factors for {}, got {}",
            num_components,
            if is_grayscale { "grayscale" } else { "YCbCr" },
            factors.len()
        )));
    }

    // Validate factor values (1..=4)
    for (i, &(h, v)) in factors.iter().enumerate() {
        if h == 0 || h > 4 || v == 0 || v > 4 {
            return Err(JpegError::CorruptData(format!(
                "sampling factor ({}, {}) for component {} is out of range 1..=4",
                h, v, i
            )));
        }
    }

    // Max sampling factors determine MCU size
    let max_h: u8 = factors.iter().map(|&(h, _)| h).max().unwrap_or(1);
    let max_v: u8 = factors.iter().map(|&(_, v)| v).max().unwrap_or(1);

    // Validate that max_h and max_v are from component 0 (Y) for standard JPEG structure,
    // or at least that all factor ratios are valid integers.
    for (i, &(h, v)) in factors.iter().enumerate() {
        if !max_h.is_multiple_of(h) || !max_v.is_multiple_of(v) {
            return Err(JpegError::CorruptData(format!(
                "component {} sampling factors ({}, {}) must evenly divide max factors ({}, {})",
                i, h, v, max_h, max_v
            )));
        }
    }

    // MCU dimensions in pixels
    let mcu_w: usize = max_h as usize * 8;
    let mcu_h: usize = max_v as usize * 8;
    let mcus_x: usize = width.div_ceil(mcu_w);
    let mcus_y: usize = height.div_ceil(mcu_h);

    // Generate scaled quantization tables
    let luma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_LUMINANCE_QUANT_TABLE, quality);
    let chroma_quant: [u16; 64] =
        tables::quality_scale_quant_table(&tables::STD_CHROMINANCE_QUANT_TABLE, quality);
    let luma_divisors: QuantDivisors = scale_quant_for_fdct(&luma_quant);
    let chroma_divisors: QuantDivisors = scale_quant_for_fdct(&chroma_quant);

    // Build Huffman tables
    let dc_luma_table: HuffTable =
        build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table: HuffTable =
        build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table: HuffTable =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table: HuffTable =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    // SIMD dispatch — used for both color conversion and FDCT+quantize
    let enc_simd = crate::simd::detect_encoder();

    // Color convert
    let (y_plane, cb_plane, cr_plane) = convert_to_ycbcr(
        pixels,
        width,
        height,
        pixel_format,
        enc_simd.rgb_to_ycbcr_row,
    )?;

    // FDCT function
    let fdct_quantize_fn = enc_simd.fdct_quantize;

    // Entropy encode all MCUs
    let mut bit_writer: BitWriter = BitWriter::new(width * height);
    let mut prev_dc_y: i16 = 0;
    let mut prev_dc_cb: i16 = 0;
    let mut prev_dc_cr: i16 = 0;

    let y_h: u8 = factors[0].0;
    let y_v: u8 = factors[0].1;

    // Per-component width/height in blocks for dummy block detection.
    // C libjpeg-turbo creates dummy blocks (DC=prev, AC=0) beyond these.
    let y_wib: usize = width.div_ceil(8);
    let y_hib: usize = height.div_ceil(8);

    for mcu_row in 0..mcus_y {
        for mcu_col in 0..mcus_x {
            let x0: usize = mcu_col * mcu_w;
            let y0: usize = mcu_row * mcu_h;

            if is_grayscale {
                // Grayscale: h_i x v_i blocks of Y
                for bv in 0..y_v as usize {
                    for bh in 0..y_h as usize {
                        let bx: usize = x0 + bh * 8;
                        let by: usize = y0 + bv * 8;
                        if is_y_dummy(bx, by, y_wib, y_hib) {
                            encode_dummy_block(
                                &dc_luma_table,
                                &ac_luma_table,
                                &mut bit_writer,
                                &mut prev_dc_y,
                            );
                        } else {
                            encode_single_block(
                                &y_plane,
                                width,
                                height,
                                bx,
                                by,
                                &luma_divisors,
                                &dc_luma_table,
                                &ac_luma_table,
                                &mut bit_writer,
                                &mut prev_dc_y,
                                fdct_quantize_fn,
                            );
                        }
                    }
                }
            } else {
                // Y blocks: y_h x y_v blocks per MCU (row-major order)
                for bv in 0..y_v as usize {
                    for bh in 0..y_h as usize {
                        let bx: usize = x0 + bh * 8;
                        let by: usize = y0 + bv * 8;
                        if is_y_dummy(bx, by, y_wib, y_hib) {
                            encode_dummy_block(
                                &dc_luma_table,
                                &ac_luma_table,
                                &mut bit_writer,
                                &mut prev_dc_y,
                            );
                        } else {
                            encode_single_block(
                                &y_plane,
                                width,
                                height,
                                bx,
                                by,
                                &luma_divisors,
                                &dc_luma_table,
                                &ac_luma_table,
                                &mut bit_writer,
                                &mut prev_dc_y,
                                fdct_quantize_fn,
                            );
                        }
                    }
                }

                // Chroma components (Cb, Cr): each has factors[1] and factors[2]
                let cb_h: u8 = factors[1].0;
                let cb_v: u8 = factors[1].1;
                let h_downsample: usize = max_h as usize / cb_h as usize;
                let v_downsample: usize = max_v as usize / cb_v as usize;
                let cb_wib: usize = width.div_ceil(h_downsample * 8);
                let cb_hib: usize = height.div_ceil(v_downsample * 8);

                // Cb blocks
                for bv in 0..cb_v as usize {
                    for bh in 0..cb_h as usize {
                        let bx: usize = x0 / h_downsample + bh * 8;
                        let by: usize = y0 / v_downsample + bv * 8;
                        if bx / 8 >= cb_wib || by / 8 >= cb_hib {
                            encode_dummy_block(
                                &dc_chroma_table,
                                &ac_chroma_table,
                                &mut bit_writer,
                                &mut prev_dc_cb,
                            );
                        } else {
                            encode_downsampled_chroma_block(
                                &cb_plane,
                                width,
                                height,
                                x0 + bh * 8 * h_downsample,
                                y0 + bv * 8 * v_downsample,
                                h_downsample,
                                v_downsample,
                                &chroma_divisors,
                                &dc_chroma_table,
                                &ac_chroma_table,
                                &mut bit_writer,
                                &mut prev_dc_cb,
                                fdct_quantize_fn,
                            );
                        }
                    }
                }

                let cr_h: u8 = factors[2].0;
                let cr_v: u8 = factors[2].1;
                let h_downsample_cr: usize = max_h as usize / cr_h as usize;
                let v_downsample_cr: usize = max_v as usize / cr_v as usize;
                let cr_wib: usize = width.div_ceil(h_downsample_cr * 8);
                let cr_hib: usize = height.div_ceil(v_downsample_cr * 8);

                // Cr blocks
                for bv in 0..cr_v as usize {
                    for bh in 0..cr_h as usize {
                        let bx: usize = x0 / h_downsample_cr + bh * 8;
                        let by: usize = y0 / v_downsample_cr + bv * 8;
                        if bx / 8 >= cr_wib || by / 8 >= cr_hib {
                            encode_dummy_block(
                                &dc_chroma_table,
                                &ac_chroma_table,
                                &mut bit_writer,
                                &mut prev_dc_cr,
                            );
                        } else {
                            encode_downsampled_chroma_block(
                                &cr_plane,
                                width,
                                height,
                                x0 + bh * 8 * h_downsample_cr,
                                y0 + bv * 8 * v_downsample_cr,
                                h_downsample_cr,
                                v_downsample_cr,
                                &chroma_divisors,
                                &dc_chroma_table,
                                &ac_chroma_table,
                                &mut bit_writer,
                                &mut prev_dc_cr,
                                fdct_quantize_fn,
                            );
                        }
                    }
                }
            }
        }
    }

    bit_writer.flush();

    // Assemble output
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    marker_writer::write_dqt(&mut output, 0, &luma_quant);
    if !is_grayscale {
        marker_writer::write_dqt(&mut output, 1, &chroma_quant);
    }

    // Frame header with explicit sampling factors
    if is_grayscale {
        let components: Vec<(u8, u8, u8, u8)> = vec![(1, y_h, y_v, 0)];
        marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);
    } else {
        let components: Vec<(u8, u8, u8, u8)> = vec![
            (1, factors[0].0, factors[0].1, 0), // Y
            (2, factors[1].0, factors[1].1, 1), // Cb
            (3, factors[2].0, factors[2].1, 1), // Cr
        ];
        marker_writer::write_sof0(&mut output, width as u16, height as u16, &components);
    }

    // Huffman tables
    marker_writer::write_dht(
        &mut output,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );
    marker_writer::write_dht(
        &mut output,
        1,
        0,
        &tables::AC_LUMINANCE_BITS,
        &tables::AC_LUMINANCE_VALUES,
    );
    if !is_grayscale {
        marker_writer::write_dht(
            &mut output,
            0,
            1,
            &tables::DC_CHROMINANCE_BITS,
            &tables::DC_CHROMINANCE_VALUES,
        );
        marker_writer::write_dht(
            &mut output,
            1,
            1,
            &tables::AC_CHROMINANCE_BITS,
            &tables::AC_CHROMINANCE_VALUES,
        );
    }

    // Scan header
    if is_grayscale {
        let scan_components: Vec<(u8, u8, u8)> = vec![(1, 0, 0)];
        marker_writer::write_sos(&mut output, &scan_components);
    } else {
        let scan_components: Vec<(u8, u8, u8)> = vec![
            (1, 0, 0), // Y: DC table 0, AC table 0
            (2, 1, 1), // Cb: DC table 1, AC table 1
            (3, 1, 1), // Cr: DC table 1, AC table 1
        ];
        marker_writer::write_sos(&mut output, &scan_components);
    }

    // Entropy-coded data
    output.extend_from_slice(bit_writer.data());

    marker_writer::write_eoi(&mut output);

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compress_grayscale_1x1() {
        // Minimal 1x1 grayscale image
        let pixels = [128u8];
        let result = compress(
            &pixels,
            1,
            1,
            PixelFormat::Grayscale,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
        let jpeg = result.unwrap();
        // Check SOI marker
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        // Check EOI marker
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn compress_rgb_8x8() {
        // Red 8x8 image
        let mut pixels = vec![0u8; 8 * 8 * 3];
        for i in 0..64 {
            pixels[i * 3] = 255; // R
            pixels[i * 3 + 1] = 0; // G
            pixels[i * 3 + 2] = 0; // B
        }
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
        let jpeg = result.unwrap();
        assert_eq!(jpeg[0], 0xFF);
        assert_eq!(jpeg[1], 0xD8);
        assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
        assert_eq!(jpeg[jpeg.len() - 1], 0xD9);
    }

    #[test]
    fn compress_rgb_422() {
        // 16x8 green image with 4:2:2 subsampling
        let mut pixels = vec![0u8; 16 * 8 * 3];
        for i in 0..(16 * 8) {
            pixels[i * 3] = 0;
            pixels[i * 3 + 1] = 255;
            pixels[i * 3 + 2] = 0;
        }
        let result = compress(
            &pixels,
            16,
            8,
            PixelFormat::Rgb,
            75,
            Subsampling::S422,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rgb_420() {
        // 16x16 blue image with 4:2:0 subsampling
        let mut pixels = vec![0u8; 16 * 16 * 3];
        for i in 0..(16 * 16) {
            pixels[i * 3] = 0;
            pixels[i * 3 + 1] = 0;
            pixels[i * 3 + 2] = 255;
        }
        let result = compress(
            &pixels,
            16,
            16,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_non_multiple_of_8() {
        // 10x6 image (not a multiple of 8 in either dimension)
        let pixels = vec![128u8; 10 * 6 * 3];
        let result = compress(
            &pixels,
            10,
            6,
            PixelFormat::Rgb,
            50,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_non_multiple_of_16_420() {
        // 13x11 image with 4:2:0 (MCU = 16x16)
        let pixels = vec![200u8; 13 * 11 * 3];
        let result = compress(
            &pixels,
            13,
            11,
            PixelFormat::Rgb,
            90,
            Subsampling::S420,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rgba_input() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgba,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_bgr_input() {
        let pixels = vec![128u8; 8 * 8 * 3];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Bgr,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_bgra_input() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Bgra,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn compress_rejects_zero_dimensions() {
        let pixels = vec![128u8; 64];
        let result = compress(
            &pixels,
            0,
            8,
            PixelFormat::Grayscale,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_err());
    }

    #[test]
    fn compress_rejects_buffer_too_small() {
        let pixels = vec![128u8; 10];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_err());
    }

    #[test]
    fn compress_quality_extremes() {
        let pixels = vec![128u8; 8 * 8 * 3];
        // Quality 1 (worst)
        let result1 = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            1,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result1.is_ok());
        // Quality 100 (best)
        let result100 = compress(
            &pixels,
            8,
            8,
            PixelFormat::Rgb,
            100,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result100.is_ok());
        // Higher quality should generally produce larger output
        assert!(result100.unwrap().len() >= result1.unwrap().len());
    }

    #[test]
    fn roundtrip_grayscale() {
        // Encode a grayscale image and decode it back
        let width = 8;
        let height = 8;
        let pixels: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let jpeg = compress(
            &pixels,
            width,
            height,
            PixelFormat::Grayscale,
            100,
            Subsampling::S444,
            DctMethod::IsLow,
        )
        .unwrap();

        // Decode using our own decoder
        let image = crate::api::high_level::decompress(&jpeg).unwrap();
        assert_eq!(image.width, width);
        assert_eq!(image.height, height);
        assert_eq!(image.pixel_format, PixelFormat::Grayscale);

        // At quality 100, the roundtrip should be close (within ~2 for 8-bit)
        for i in 0..64 {
            let diff = (image.data[i] as i16 - pixels[i] as i16).unsigned_abs();
            assert!(
                diff <= 3,
                "pixel {i}: expected ~{}, got {} (diff {})",
                pixels[i],
                image.data[i],
                diff
            );
        }
    }

    #[test]
    fn roundtrip_rgb_444() {
        let width = 8;
        let height = 8;
        // Uniform mid-gray
        let pixels = vec![128u8; width * height * 3];
        let jpeg = compress(
            &pixels,
            width,
            height,
            PixelFormat::Rgb,
            100,
            Subsampling::S444,
            DctMethod::IsLow,
        )
        .unwrap();

        let image = crate::api::high_level::decompress(&jpeg).unwrap();
        assert_eq!(image.width, width);
        assert_eq!(image.height, height);

        // Color conversion (RGB -> YCbCr -> RGB) introduces rounding errors.
        // At quality 100 with uniform input, allow a modest tolerance.
        for i in 0..image.data.len() {
            let diff = (image.data[i] as i16 - 128).unsigned_abs();
            assert!(
                diff <= 8,
                "byte {i}: expected ~128, got {} (diff {})",
                image.data[i],
                diff
            );
        }
    }

    #[test]
    fn compress_cmyk_produces_valid_jpeg() {
        let pixels = vec![128u8; 8 * 8 * 4];
        let result = compress(
            &pixels,
            8,
            8,
            PixelFormat::Cmyk,
            75,
            Subsampling::S444,
            DctMethod::IsLow,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn extract_block_edge_padding() {
        // 4x4 plane: values 0..15
        let plane: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
        let mut block = [0i16; 64];
        extract_block(&plane, 4, 4, 0, 0, &mut block);

        // Row 0, col 0 should be plane[0] - 128 = 0 - 128 = -128
        assert_eq!(block[0], -128);
        // Row 0, col 3 should be plane[3] - 128 = 48 - 128 = -80
        assert_eq!(block[3], 48 - 128);
        // Row 0, col 4..7 should replicate col 3 (plane[3] = 48)
        assert_eq!(block[4], 48 - 128);
        assert_eq!(block[7], 48 - 128);
        // Row 4..7 should replicate row 3
        assert_eq!(block[4 * 8], block[3 * 8]);
    }
}
