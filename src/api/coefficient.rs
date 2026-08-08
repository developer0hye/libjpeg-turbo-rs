// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
/// Coefficient-level JPEG access for lossless transforms.
///
/// Provides read_coefficients() / write_coefficients() / transform() API
/// similar to libjpeg-turbo's jpegtran workflow.
use crate::common::error::{JpegError, Result};
use crate::common::quant_table::NATURAL_ORDER;
use crate::common::types::{MarkerSaveConfig, SavedMarker};
use crate::decode::marker::{JpegMetadata, MarkerReader};
use crate::encode::huffman_encode::{build_huff_table, BitWriter, HuffTable, HuffmanEncoder};
use crate::encode::marker_writer;
use crate::encode::pipeline as encoder_pipeline;
use crate::encode::tables;
use crate::transform::spatial;
use crate::transform::{TransformOp, TransformOptions};
#[allow(unused_imports)]
use alloc::{format, vec};
#[allow(unused_imports)]
use alloc::{string::ToString, vec::Vec};

/// Per-component DCT coefficient data.
#[derive(Debug, Clone)]
pub struct ComponentCoefficients {
    /// Quantized DCT blocks in zigzag order, each block is 64 coefficients.
    pub blocks: Vec<[i16; 64]>,
    /// Width in blocks.
    pub blocks_x: usize,
    /// Height in blocks.
    pub blocks_y: usize,
    /// Horizontal sampling factor.
    pub h_sampling: u8,
    /// Vertical sampling factor.
    pub v_sampling: u8,
    /// Quantization table index.
    pub quant_table_index: u8,
    /// Component identifier from the source JPEG (1=Y, 2=Cb, 3=Cr per JFIF).
    pub component_id: u8,
}

/// Complete coefficient representation of a JPEG image.
#[derive(Debug, Clone)]
pub struct JpegCoefficients {
    /// Image width in pixels.
    pub width: u16,
    /// Image height in pixels.
    pub height: u16,
    /// Sample data precision in bits per component (8 for baseline,
    /// 12 for extended sequential / `monkey12`-style sources).
    /// Re-emitted as the SOF byte at offset 4 of the SOF segment so
    /// transcoded output preserves the source precision instead of
    /// silently downgrading to 8-bit. `0` is treated as 8 for
    /// backward compatibility with callers constructed before this
    /// field existed.
    pub data_precision: u8,
    /// Per-component coefficient data.
    pub components: Vec<ComponentCoefficients>,
    /// Quantization tables (up to 4, in zigzag order).
    pub quant_tables: Vec<[u16; 64]>,
    /// Restart interval from the source JPEG (0 = no restart markers).
    pub restart_interval: u16,
    /// JFIF density units from source (0=aspect ratio, 1=DPI, 2=DPCM).
    pub density_unit: u8,
    /// JFIF X density from source.
    pub x_density: u16,
    /// JFIF Y density from source.
    pub y_density: u16,
    /// Whether the source contained a JFIF APP0 marker. This is independent
    /// of `adobe_transform`: legal streams can contain both APP0 and APP14.
    pub saw_jfif_marker: bool,
    /// Adobe APP14 color-transform byte from the source JPEG, if an
    /// Adobe marker was present. `None` means no APP14 was seen.
    /// Re-emitting the same value on transcode preserves the original
    /// colorspace classification (RGB vs YCbCr vs YCCK vs CMYK).
    pub adobe_transform: Option<u8>,
}

impl JpegCoefficients {
    /// Effective sample precision (bits per component): the stored
    /// `data_precision`, or 8 when the field was left zeroed by an
    /// older caller that pre-dates the precision plumb.
    #[inline]
    pub fn effective_precision(&self) -> u8 {
        if self.data_precision == 0 {
            8
        } else {
            self.data_precision
        }
    }
}

fn has_rgb_component_ids(coeffs: &JpegCoefficients) -> bool {
    coeffs.components.len() == 3
        && coeffs
            .components
            .iter()
            .map(|component| component.component_id)
            .eq(*b"RGB")
}

fn uses_single_rgb_coding_table(coeffs: &JpegCoefficients) -> bool {
    has_rgb_component_ids(coeffs)
        && coeffs
            .components
            .iter()
            .all(|component| component.quant_table_index == 0)
}

fn coding_table_for_component(coeffs: &JpegCoefficients, component_index: usize) -> usize {
    if component_index == 0 || uses_single_rgb_coding_table(coeffs) {
        0
    } else {
        1
    }
}

fn write_coefficient_colorspace_marker(output: &mut Vec<u8>, coeffs: &JpegCoefficients) {
    if coeffs.saw_jfif_marker
        || (coeffs.adobe_transform.is_none() && !has_rgb_component_ids(coeffs))
    {
        marker_writer::write_app0_jfif_with_density(
            output,
            coeffs.density_unit,
            coeffs.x_density,
            coeffs.y_density,
        );
    }
    if let Some(transform) = coeffs.adobe_transform {
        marker_writer::write_app14_adobe(output, transform);
    } else if has_rgb_component_ids(coeffs) {
        // Markerless ASCII R/G/B streams are classified as RGB by libjpeg.
        // jpegtran emits Adobe transform 0 when rewriting such coefficients;
        // emitting JFIF here would instead make decoders treat them as YCbCr.
        marker_writer::write_app14_adobe(output, 0);
    }
}

/// Per-component info extracted for re-encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncoderComponentInfo {
    /// Horizontal sampling factor.
    pub h_sampling: u8,
    /// Vertical sampling factor.
    pub v_sampling: u8,
    /// Quantization table index.
    pub quant_table_index: u8,
}

/// Critical JPEG parameters extracted from decoded coefficients for re-encoding.
///
/// Matches the subset of `jpeg_compress_struct` fields that
/// `jpeg_copy_critical_parameters()` copies between a decompressor and compressor.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Number of components.
    pub num_components: usize,
    /// Per-component sampling and quantization info.
    pub component_info: Vec<EncoderComponentInfo>,
    /// Quantization tables (in zigzag order).
    pub quant_tables: Vec<[u16; 64]>,
}

/// Copy critical JPEG parameters from decoded coefficients for re-encoding.
///
/// Extracts dimensions, sampling factors, and quantization tables from
/// `JpegCoefficients` into an `EncoderConfig` suitable for driving a new
/// encoding pass. Matches libjpeg-turbo's `jpeg_copy_critical_parameters()`.
pub fn copy_critical_parameters(coeffs: &JpegCoefficients) -> EncoderConfig {
    let component_info: Vec<EncoderComponentInfo> = coeffs
        .components
        .iter()
        .map(|comp| EncoderComponentInfo {
            h_sampling: comp.h_sampling,
            v_sampling: comp.v_sampling,
            quant_table_index: comp.quant_table_index,
        })
        .collect();

    EncoderConfig {
        width: coeffs.width as usize,
        height: coeffs.height as usize,
        num_components: coeffs.components.len(),
        component_info,
        quant_tables: coeffs.quant_tables.clone(),
    }
}

/// Read DCT coefficients from a JPEG byte stream.
///
/// Decodes entropy data to recover quantized DCT coefficients
/// without performing IDCT or color conversion.
pub fn read_coefficients(data: &[u8]) -> Result<JpegCoefficients> {
    let mut reader = MarkerReader::new(data);
    let metadata = reader.read_markers()?;
    // Default frame-dimension guard (issue #355 review HIGH-1): this
    // entry point has no limits API, so the permissive defaults bound
    // the header bomb before block buffers are sized from the SOF.
    crate::common::types::DecodeLimits::default().check_frame(
        metadata.frame.width as usize,
        metadata.frame.height as usize,
    )?;

    let frame = &metadata.frame;

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

    let mcu_w = max_h * 8;
    let mcu_h = max_v * 8;
    let mcus_x = (frame.width as usize).div_ceil(mcu_w);
    let mcus_y = (frame.height as usize).div_ceil(mcu_h);

    // Collect quant tables in natural (row-major) order for write_dqt
    // compatibility. The four DQT slots may be sparse (e.g. only slot 1
    // defined, or a gap at slot 2) — the writers emit `quant_tables[i]`
    // as slot `i`, so compacting the slots requires remapping every
    // component's slot reference into the dense index space or the
    // re-encoded SOF would reference a table the output never defines
    // (Fuzz Smoke runs 29679993066..30064906856, P4-34).
    let mut slot_to_dense: [Option<u8>; 4] = [None; 4];
    let mut quant_tables: Vec<[u16; 64]> = Vec::new();
    for (slot, qt) in metadata.quant_tables.iter().enumerate() {
        if let Some(q) = qt.as_ref() {
            slot_to_dense[slot] = Some(quant_tables.len() as u8);
            quant_tables.push(q.values);
        }
    }

    // Allocate component coefficient buffers
    let mut comp_data: Vec<ComponentCoefficients> = frame
        .components
        .iter()
        .map(|comp| {
            let bx = mcus_x * comp.horizontal_sampling as usize;
            let by = mcus_y * comp.vertical_sampling as usize;
            ComponentCoefficients {
                blocks: vec![[0i16; 64]; bx * by],
                blocks_x: bx,
                blocks_y: by,
                h_sampling: comp.horizontal_sampling,
                v_sampling: comp.vertical_sampling,
                // A reference to an undefined slot falls back to dense
                // index 0 — the coefficient pass never dequantizes, and
                // djpeg rejects the C-side equivalent anyway, so any
                // defined table keeps the output self-consistent.
                quant_table_index: slot_to_dense[comp.quant_table_index as usize].unwrap_or(0),
                component_id: comp.id,
            }
        })
        .collect();

    if frame.is_progressive && metadata.is_arithmetic {
        // SOF10: arithmetic progressive — use arithmetic decoder with progressive scans.
        decode_arithmetic_progressive_coefficients(
            data,
            &metadata,
            &mut comp_data,
            mcus_x,
            mcus_y,
        )?;
    } else if frame.is_progressive {
        decode_progressive_coefficients(data, &metadata, &mut comp_data, mcus_x, mcus_y)?;
    } else if metadata.is_arithmetic {
        decode_arithmetic_coefficients(data, &metadata, &mut comp_data, mcus_x, mcus_y)?;
    } else {
        decode_baseline_coefficients(data, &metadata, &mut comp_data, mcus_x, mcus_y)?;
    }

    // Decoder stores blocks in natural (row-major) order;
    // convert to zigzag order for encoder compatibility.
    convert_all_to_zigzag(&mut comp_data);

    let density: &crate::common::types::DensityInfo = &metadata.density;
    let density_unit: u8 = match density.unit {
        crate::common::types::DensityUnit::Unknown => 0,
        crate::common::types::DensityUnit::Dpi => 1,
        crate::common::types::DensityUnit::Dpcm => 2,
    };

    Ok(JpegCoefficients {
        width: frame.width,
        height: frame.height,
        data_precision: frame.precision,
        components: comp_data,
        quant_tables,
        restart_interval: metadata.restart_interval,
        density_unit,
        x_density: density.x,
        y_density: density.y,
        saw_jfif_marker: metadata.saw_jfif_marker,
        adobe_transform: if metadata.saw_adobe_marker {
            Some(metadata.adobe_transform)
        } else {
            None
        },
    })
}

/// Write DCT coefficients to a JPEG byte stream.
///
/// Encodes quantized DCT coefficients using Huffman coding,
/// producing a valid baseline JPEG file.
pub fn write_coefficients(coeffs: &JpegCoefficients) -> Result<Vec<u8>> {
    let num_components = coeffs.components.len();
    let is_grayscale = num_components == 1;
    let uses_secondary_table: bool = !is_grayscale && !uses_single_rgb_coding_table(coeffs);

    // Standard ITU-T T.81 Annex K Huffman tables only define DC
    // categories 0..=11; for `data_precision > 8` the source can produce
    // DC categories 12..=15 that would silently encode as 0-bit codes
    // and corrupt the stream. The optimised path
    // (`write_coefficients_optimized`) is the supported route for
    // 12-bit transcode (FFI sets `optimize_coding=1` automatically when
    // `data_precision == 12`); refusing here keeps the failure loud.
    // Tracked under `docs/LAST_MILE.md` → P0-4 (12-bit transcode).
    let precision: u8 = coeffs.effective_precision();
    if precision > 8 {
        return Err(JpegError::Unsupported(format!(
            "write_coefficients: 12-bit (precision={precision}) transcode not yet \
             supported on the non-optimized path; route through \
             write_coefficients_optimized (set optimize_coding=1)"
        )));
    }

    // Build Huffman tables
    let dc_luma_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    let max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let interleaved_mcus_x =
        coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let interleaved_mcus_y =
        coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    // Compute actual data block counts per component (not MCU-padded).
    // Blocks beyond these are "dummy" blocks: DC copied from last real block,
    // AC all zeros. Matches C libjpeg-turbo jccoefct.c:184-191.
    let data_blocks_x: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.width as usize * c.h_sampling as usize).div_ceil(max_h * 8))
        .collect();
    let data_blocks_y: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.height as usize * c.v_sampling as usize).div_ceil(max_v * 8))
        .collect();
    let (mcus_x, mcus_y) = if is_grayscale {
        (data_blocks_x[0], data_blocks_y[0])
    } else {
        (interleaved_mcus_x, interleaved_mcus_y)
    };

    // Entropy encode with optional restart markers
    let mut bit_writer = BitWriter::new(coeffs.width as usize * coeffs.height as usize);
    let mut prev_dc = vec![0i16; num_components];
    let dummy_block: [i16; 64] = [0i16; 64];
    let ri: u32 = coeffs.restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut restart_idx: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            // Insert restart marker between MCUs when interval is set
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                bit_writer.flush();
                bit_writer.write_restart_marker(restart_idx);
                restart_idx = (restart_idx + 1) & 7;
                // Reset DC predictions after restart
                for dc in prev_dc.iter_mut() {
                    *dc = 0;
                }
            }

            for (ci, comp) in coeffs.components.iter().enumerate() {
                let dc_table = if coding_table_for_component(coeffs, ci) == 0 {
                    &dc_luma_table
                } else {
                    &dc_chroma_table
                };
                let ac_table = if coding_table_for_component(coeffs, ci) == 0 {
                    &ac_luma_table
                } else {
                    &ac_chroma_table
                };

                let h_blocks = if is_grayscale {
                    1
                } else {
                    comp.h_sampling as usize
                };
                let v_blocks = if is_grayscale {
                    1
                } else {
                    comp.v_sampling as usize
                };
                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let bx = mcu_x * h_blocks + h;
                        let by = mcu_y * v_blocks + v;
                        let is_dummy: bool = bx >= data_blocks_x[ci] || by >= data_blocks_y[ci];

                        if is_dummy {
                            let mut dblock: [i16; 64] = dummy_block;
                            dblock[0] = prev_dc[ci];
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &dblock,
                                &mut prev_dc[ci],
                                dc_table,
                                ac_table,
                            );
                        } else {
                            let block_idx = by * comp.blocks_x + bx;
                            let block = &comp.blocks[block_idx];
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                block,
                                &mut prev_dc[ci],
                                dc_table,
                                ac_table,
                            );
                        }
                    }
                }
            }
            mcu_count += 1;
        }
    }

    bit_writer.flush();

    // Assemble output
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    // Preserve source JFIF density (matches C jpegtran behavior)
    write_coefficient_colorspace_marker(&mut output, coeffs);

    // Quantization tables
    for (i, qt) in coeffs.quant_tables.iter().enumerate() {
        marker_writer::write_dqt(&mut output, i as u8, qt);
    }

    // Frame header — use SOF1 (extended sequential) when quant tables need 16-bit
    // precision (values > 255). The 12-bit precision case is rejected
    // earlier in the guard at the top of this function, so `precision`
    // here is always ≤ 8 and only quant-table magnitude matters.
    // Matching C jpegtran behavior in `references/libjpeg-turbo/src/jcparam.c`.
    let needs_extended: bool = coeffs
        .quant_tables
        .iter()
        .any(|qt| qt.iter().any(|&v| v > 255));
    let components: Vec<(u8, u8, u8, u8)> = coeffs
        .components
        .iter()
        .map(|c| {
            (
                c.component_id,
                c.h_sampling,
                c.v_sampling,
                c.quant_table_index,
            )
        })
        .collect();
    output.push(0xFF);
    output.push(if needs_extended { 0xC1 } else { 0xC0 });
    let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    output.extend_from_slice(&sof_len.to_be_bytes());
    output.push(precision); // sample precision (8 default; >8 forces SOF1)
    output.extend_from_slice(&coeffs.height.to_be_bytes());
    output.extend_from_slice(&coeffs.width.to_be_bytes());
    output.push(components.len() as u8);
    for &(id, h_samp, v_samp, quant_tbl_id) in &components {
        output.push(id);
        output.push((h_samp << 4) | v_samp);
        output.push(quant_tbl_id);
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
    if uses_secondary_table {
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

    // DRI (restart interval) — after DHT, before SOS (matching C jpegtran order)
    if coeffs.restart_interval > 0 {
        marker_writer::write_dri(&mut output, coeffs.restart_interval);
    }

    // Scan header — preserve source component IDs
    let scan_components: Vec<(u8, u8, u8)> = coeffs
        .components
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let tbl = coding_table_for_component(coeffs, i) as u8;
            (c.component_id, tbl, tbl)
        })
        .collect();
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Apply a lossless transform to a JPEG image.
///
/// Delegates to [`transform_jpeg_with_options`] with default options, so
/// metadata (EXIF/ICC/COM markers) is preserved
/// ([`MarkerCopyMode::All`](crate::MarkerCopyMode::All), matching both [`TransformOptions::default`]
/// and C TurboJPEG's `tjTransform` without `TJXOPT_COPYNONE`). Pass
/// [`MarkerCopyMode::None`](crate::MarkerCopyMode::None) via [`transform_jpeg_with_options`] to strip
/// markers instead.
/// One axis of upstream's edge trimming.
///
/// Returns the largest whole number of iMCUs that fits in `extent`, or `extent`
/// unchanged when fewer than one iMCU fits — mirroring the `MCU_cols > 0` /
/// `MCU_rows > 0` guards in `trim_right_edge` / `trim_bottom_edge`
/// (transupp.c:1570-1592). Dropping the guard turns "there is nothing to trim"
/// into "trim everything", which is how P4-117 rejected valid 4:4:1 images.
fn trim_to_whole_imcus(extent: usize, imcu: usize, trim: bool) -> usize {
    if !trim || imcu == 0 {
        return extent;
    }
    let whole_imcus: usize = extent / imcu;
    if whole_imcus > 0 {
        whole_imcus * imcu
    } else {
        extent
    }
}

pub fn transform_jpeg(data: &[u8], op: TransformOp) -> Result<Vec<u8>> {
    transform_jpeg_with_options(
        data,
        &TransformOptions {
            op,
            ..Default::default()
        },
    )
}

/// Apply a lossless transform with full TJXOPT-compatible options.
///
/// Supports all 9 flags from libjpeg-turbo: perfect, trim, crop, grayscale,
/// no_output, progressive, arithmetic, optimize, and copy_markers.
pub fn transform_jpeg_with_options(data: &[u8], options: &TransformOptions) -> Result<Vec<u8>> {
    // Read saved markers from the source based on copy_markers mode.
    let saved_markers: Vec<SavedMarker> = match options.copy_markers {
        crate::transform::MarkerCopyMode::All => {
            let mut reader: MarkerReader<'_> = MarkerReader::new(data);
            reader.set_marker_save_config(MarkerSaveConfig::All);
            let meta: JpegMetadata = reader.read_markers()?;
            // Filter out JFIF APP0 since write_coefficients writes its own.
            meta.saved_markers
                .into_iter()
                .filter(|m| m.code != 0xE0)
                .collect()
        }
        crate::transform::MarkerCopyMode::IccOnly => {
            let mut reader: MarkerReader<'_> = MarkerReader::new(data);
            reader.set_marker_save_config(MarkerSaveConfig::All);
            let meta: JpegMetadata = reader.read_markers()?;
            // Keep only APP2 markers that contain ICC profile data.
            meta.saved_markers
                .into_iter()
                .filter(|m| m.code == 0xE2)
                .collect()
        }
        crate::transform::MarkerCopyMode::None => Vec::new(),
    };

    let mut coeffs = read_coefficients(data)?;
    let op: TransformOp = options.op;

    // Determine iMCU dimensions from the coefficient data.
    let max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let imcu_w: usize = max_h * 8;
    let imcu_h: usize = max_v * 8;

    // For transforms that swap dimensions, use swapped iMCU sizes for alignment checks.
    let swaps_dims: bool = matches!(
        op,
        TransformOp::Transpose | TransformOp::Transverse | TransformOp::Rot90 | TransformOp::Rot270
    );

    // Check which dimension(s) need to be iMCU-aligned for this transform.
    let needs_width_aligned: bool = matches!(
        op,
        TransformOp::HFlip
            | TransformOp::Transverse
            | TransformOp::Rot90
            | TransformOp::Rot180
            | TransformOp::Rot270
    );
    let needs_height_aligned: bool = matches!(
        op,
        TransformOp::VFlip
            | TransformOp::Transverse
            | TransformOp::Rot90
            | TransformOp::Rot180
            | TransformOp::Rot270
    );

    let width_aligned: bool = (coeffs.width as usize).is_multiple_of(imcu_w);
    let height_aligned: bool = (coeffs.height as usize).is_multiple_of(imcu_h);

    let has_partial_width: bool = needs_width_aligned && !width_aligned;
    let has_partial_height: bool = needs_height_aligned && !height_aligned;

    // PERFECT: fail if partial iMCU blocks exist for this transform.
    if options.perfect && (has_partial_width || has_partial_height) {
        return Err(JpegError::CorruptData(format!(
            "perfect transform requested but image {}x{} is not iMCU-aligned (iMCU={}x{})",
            coeffs.width, coeffs.height, imcu_w, imcu_h
        )));
    }

    // TRIM: discard partial iMCU blocks at edges.
    // For dimension-swapping transforms (rot90, rot270), C jpegtran only
    // trims selectively: rot90 trims what becomes the output width (source
    // height), rot270 trims what becomes the output height (source width).
    if options.trim && (has_partial_width || has_partial_height) {
        let trim_width: bool = match op {
            // ROT90: source height → output width, only trim source height
            TransformOp::Rot90 => false,
            _ => has_partial_width,
        };
        let trim_height: bool = match op {
            // ROT270: source width → output height, only trim source width
            TransformOp::Rot270 => false,
            _ => has_partial_height,
        };

        // Trim to whole iMCUs, but only when at least one whole iMCU exists on
        // that axis. Upstream guards each edge the same way — `trim_right_edge`
        // and `trim_bottom_edge` (transupp.c:1570-1592) both begin
        // `if (MCU_cols > 0 && ...)` / `if (MCU_rows > 0 && ...)`, so an image
        // narrower or shorter than one iMCU is simply left alone.
        //
        // P4-117: without the guard, a 4:4:1 (h=1, v=4) image only 27 rows tall
        // computes `(27 / 32) * 32 == 0` and the whole transform was rejected
        // with "trim would remove all image data". C returns the untrimmed
        // 35x27 for `-trim -flip vertical` on exactly that input, and there is
        // no error path in upstream's trim at all.
        let trimmed_w: usize = trim_to_whole_imcus(coeffs.width as usize, imcu_w, trim_width);
        let trimmed_h: usize = trim_to_whole_imcus(coeffs.height as usize, imcu_h, trim_height);

        coeffs.width = trimmed_w as u16;
        coeffs.height = trimmed_h as u16;

        // Trim coefficient arrays for each component.
        for comp in &mut coeffs.components {
            let new_bx: usize = trimmed_w.div_ceil(8) * comp.h_sampling as usize / max_h;
            let new_by: usize = trimmed_h.div_ceil(8) * comp.v_sampling as usize / max_v;

            // Only need to rebuild if we actually trimmed columns or rows.
            if new_bx < comp.blocks_x || new_by < comp.blocks_y {
                let mut new_blocks: Vec<[i16; 64]> = Vec::with_capacity(new_bx * new_by);
                for by in 0..new_by {
                    for bx in 0..new_bx {
                        let old_idx: usize = by * comp.blocks_x + bx;
                        new_blocks.push(comp.blocks[old_idx]);
                    }
                }
                comp.blocks = new_blocks;
                comp.blocks_x = new_bx;
                comp.blocks_y = new_by;
            }
        }
    }

    // GRAYSCALE: drop all non-Y components.
    // When the original Y component had h_sampling > 1 or v_sampling > 1,
    // the block grid was MCU-padded. Rearrange blocks to a 1x1 raster layout
    // by stripping padding blocks at row/column edges.
    if options.grayscale && coeffs.components.len() > 1 {
        let orig_bx: usize = coeffs.components[0].blocks_x;
        let orig_by: usize = coeffs.components[0].blocks_y;

        // Target: ceil(width/8) x ceil(height/8) blocks in simple raster order
        let target_bx: usize = (coeffs.width as usize).div_ceil(8);
        let target_by: usize = (coeffs.height as usize).div_ceil(8);

        // Strip-padding rebuild assumes the source grid covers at least as
        // many blocks as the target raster. With degenerate sampling factors
        // (e.g. Y=h2v1 alongside Cb=h2v3), max_v is dominated by a non-Y
        // component, so the Y MCU grid can be smaller than ceil(H/8)×ceil(W/8)
        // and the strip loop would read past `comp.blocks`. C jpegtran rejects
        // the same input with "Unsupported color conversion request"; match
        // that rather than fabricating zero blocks.
        if orig_bx < target_bx || orig_by < target_by {
            return Err(JpegError::Unsupported(format!(
                "grayscale conversion requires Y component grid {}x{} \
                 to cover image raster {}x{}; sampling factors are incompatible",
                orig_bx, orig_by, target_bx, target_by
            )));
        }

        if orig_bx != target_bx || orig_by != target_by {
            // Rearrange: blocks are in raster order within MCU-padded grid,
            // just strip the extra columns/rows.
            let mut new_blocks: Vec<[i16; 64]> = Vec::with_capacity(target_bx * target_by);
            for by in 0..target_by {
                for bx in 0..target_bx {
                    new_blocks.push(coeffs.components[0].blocks[by * orig_bx + bx]);
                }
            }
            coeffs.components[0].blocks = new_blocks;
            coeffs.components[0].blocks_x = target_bx;
            coeffs.components[0].blocks_y = target_by;
        }

        coeffs.components.truncate(1);
        coeffs.components[0].h_sampling = 1;
        coeffs.components[0].v_sampling = 1;
        if coeffs.quant_tables.len() > 1 {
            coeffs.quant_tables.truncate(1);
        }
        coeffs.components[0].quant_table_index = 0;
    }

    // Recompute max sampling factors after grayscale may have changed them.
    let max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);

    // Apply spatial transform (reuses existing logic from transform_jpeg).
    // C libjpeg-turbo only transforms blocks within full MCU columns/rows,
    // leaving partial edge MCU blocks untouched. comp_w/comp_h are the
    // "mirrorable" region sizes per component.
    if op != TransformOp::None {
        // Blocks are stored in zigzag order; apply the op through its
        // zigzag-composed permutation map instead of converting the whole
        // coefficient corpus to natural order and back (issue #308).
        // Bind the `MAP_*` consts directly (not via `zigzag_map(op)`) so
        // each branch reads a statically-known table; `map_op` is only
        // for the generic fallback branch.
        let map_op: &spatial::ZigzagMap = spatial::zigzag_map(op);
        let map_transpose: &spatial::ZigzagMap = &spatial::MAP_TRANSPOSE;
        let map_rot90: &spatial::ZigzagMap = &spatial::MAP_ROT90;
        let map_rot270: &spatial::ZigzagMap = &spatial::MAP_ROT270;
        let map_flip_h: &spatial::ZigzagMap = &spatial::MAP_HFLIP;
        let map_flip_v: &spatial::ZigzagMap = &spatial::MAP_VFLIP;
        let map_transverse: &spatial::ZigzagMap = &spatial::MAP_TRANSVERSE;
        let map_rot180: &spatial::ZigzagMap = &spatial::MAP_ROT180;

        // Full MCU columns/rows (partial edge MCUs excluded from transform).
        let mcu_cols: usize = coeffs.width as usize / (max_h * 8);
        let mcu_rows: usize = coeffs.height as usize / (max_v * 8);

        for comp in &mut coeffs.components {
            let old_bx: usize = comp.blocks_x;
            let old_by: usize = comp.blocks_y;
            // Mirrorable region: only full MCU blocks participate in transform.
            let comp_w: usize = mcu_cols * comp.h_sampling as usize;
            let comp_h: usize = mcu_rows * comp.v_sampling as usize;
            // Dimension-swapping ops (Transpose/Rot90/Rot270/Transverse)
            // write every destination block below, so seeding the output
            // with a copy of the source would be pure wasted memory
            // traffic (~2×128 bytes per block). Only the in-place mirror
            // ops (HFlip/VFlip/Rot180) need the "edge blocks stay
            // untouched" pre-copy.
            let mut new_blocks: Vec<[i16; 64]> = if swaps_dims {
                vec![[0i16; 64]; old_bx * old_by]
            } else {
                comp.blocks.clone()
            };

            if matches!(op, TransformOp::Transpose) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = bx * old_by + by;
                        map_transpose.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::Rot90) {
                let new_bx: usize = old_by;
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        if by < comp_h {
                            let dst_idx: usize = bx * new_bx + (comp_h - 1 - by);
                            map_rot90.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        } else {
                            let dst_idx: usize = bx * new_bx + by;
                            map_transpose.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        }
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::Rot270) {
                let new_bx: usize = old_by;
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        if bx < comp_w {
                            let dst_idx: usize = (comp_w - 1 - bx) * new_bx + by;
                            map_rot270.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        } else {
                            let dst_idx: usize = bx * new_bx + by;
                            map_transpose.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        }
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::Transverse) {
                let new_bx: usize = old_by;
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let in_h: bool = by < comp_h;
                        let in_w: bool = bx < comp_w;
                        if in_h && in_w {
                            let dst_idx: usize = (comp_w - 1 - bx) * new_bx + (comp_h - 1 - by);
                            map_transverse.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        } else if !in_h && in_w {
                            let dst_idx: usize = (comp_w - 1 - bx) * new_bx + by;
                            map_rot270.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        } else if in_h && !in_w {
                            let dst_idx: usize = bx * new_bx + (comp_h - 1 - by);
                            map_rot90.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        } else {
                            let dst_idx: usize = bx * new_bx + by;
                            map_transpose.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        }
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::HFlip) {
                // Only flip within the mirrorable region (comp_w blocks).
                // Edge blocks beyond comp_w are left untouched.
                for by in 0..old_by {
                    for bx in 0..comp_w {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = by * old_bx + (comp_w - 1 - bx);
                        map_flip_h.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
            } else if matches!(op, TransformOp::VFlip) {
                // Only flip within the mirrorable region (comp_h rows).
                for by in 0..comp_h {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = (comp_h - 1 - by) * old_bx + bx;
                        map_flip_v.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
            } else if matches!(op, TransformOp::Rot180) {
                // 4-zone approach matching C transupp.c do_rot_180:
                // Zone 1 (bx<comp_w, by<comp_h): full 180° (both axes mirror)
                // Zone 2 (bx>=comp_w, by<comp_h): only vertical mirror
                // Zone 3 (bx<comp_w, by>=comp_h): only horizontal mirror
                // Zone 4 (bx>=comp_w, by>=comp_h): copy verbatim (already done)
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        if by < comp_h && bx < comp_w {
                            // Zone 1: full 180° rotation
                            let dst_idx: usize = (comp_h - 1 - by) * old_bx + (comp_w - 1 - bx);
                            map_rot180.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        } else if by < comp_h {
                            // Zone 2: only vertical mirror (right edge)
                            let dst_idx: usize = (comp_h - 1 - by) * old_bx + bx;
                            map_flip_v.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        } else if bx < comp_w {
                            // Zone 3: only horizontal mirror (bottom edge)
                            let dst_idx: usize = by * old_bx + (comp_w - 1 - bx);
                            map_flip_h.apply(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                        }
                        // Zone 4: already copied verbatim from copy_from_slice
                    }
                }
            } else {
                for (i, new_block) in new_blocks.iter_mut().enumerate() {
                    map_op.apply(&comp.blocks[i], new_block);
                }
            }

            comp.blocks = new_blocks;
        }

        if swaps_dims {
            core::mem::swap(&mut coeffs.width, &mut coeffs.height);
            for comp in &mut coeffs.components {
                core::mem::swap(&mut comp.h_sampling, &mut comp.v_sampling);
            }
            for qt in &mut coeffs.quant_tables {
                transpose_quant_table(qt);
            }
        }
    }

    // CROP: crop coefficient arrays to the specified region.
    // Applied AFTER spatial transform (crop coordinates are in output space).
    // Matches C jpegtran semantics: X/Y are rounded DOWN to iMCU boundaries,
    // and output dimensions are extended to fully cover the requested region.
    if let Some(crop) = &options.crop {
        // Recompute max sampling factors from post-transform state
        // (dimension-swapping transforms swap h/v sampling factors).
        let post_max_h: usize = coeffs
            .components
            .iter()
            .map(|c| c.h_sampling as usize)
            .max()
            .unwrap_or(1);
        let post_max_v: usize = coeffs
            .components
            .iter()
            .map(|c| c.v_sampling as usize)
            .max()
            .unwrap_or(1);
        let imcu_w: usize = post_max_h * 8;
        let imcu_h: usize = post_max_v * 8;

        // Crop coordinates are in *post-transform* image space (see comment
        // above). A swap-dim spatial op (Rot90 / Rot270 / Transpose /
        // Transverse) can leave the requested crop origin past the
        // post-transform width or height — e.g. a 2048x16 source rotated 90°
        // becomes 16x2048, so a crop with x=2032 from the original frame is
        // outside the new width=16. The downstream `coeffs.width - (crop.x -
        // remainder_x)` then underflows. Found via fuzz_transform_options
        // round-5 (CI run 25218344069) at coefficient.rs:907. Reject up
        // front rather than wrap silently.
        if crop.x >= coeffs.width as usize || crop.y >= coeffs.height as usize {
            return Err(JpegError::Unsupported(format!(
                "crop origin (x={}, y={}) lies outside post-transform image \
                 ({}x{}); crop coordinates must be in output space",
                crop.x, crop.y, coeffs.width, coeffs.height,
            )));
        }

        let remainder_x: usize = crop.x % imcu_w;
        let remainder_y: usize = crop.y % imcu_h;

        // Block-level offsets (rounded down to iMCU boundary)
        let crop_x_blocks: usize = crop.x / imcu_w * post_max_h;
        let crop_y_blocks: usize = crop.y / imcu_h * post_max_v;

        // Extend output size to cover the full requested region from the
        // MCU-aligned start position.
        let out_w: usize =
            (crop.width + remainder_x).min(coeffs.width as usize - (crop.x - remainder_x));
        let out_h: usize =
            (crop.height + remainder_y).min(coeffs.height as usize - (crop.y - remainder_y));
        // Compute block dimensions in iMCU units first, then multiply by sampling
        // factor. This guarantees block counts are always multiples of max_h/max_v,
        // matching C libjpeg-turbo's transupp.c:1805-1822 approach.
        let crop_w_blocks: usize = out_w.div_ceil(imcu_w) * post_max_h;
        let crop_h_blocks: usize = out_h.div_ceil(imcu_h) * post_max_v;

        coeffs.width = out_w as u16;
        coeffs.height = out_h as u16;

        for comp in &mut coeffs.components {
            let comp_crop_x: usize = crop_x_blocks * comp.h_sampling as usize / post_max_h;
            let comp_crop_y: usize = crop_y_blocks * comp.v_sampling as usize / post_max_v;
            let comp_crop_w: usize = crop_w_blocks * comp.h_sampling as usize / post_max_h;
            let comp_crop_h: usize = crop_h_blocks * comp.v_sampling as usize / post_max_v;

            let new_bx: usize = comp_crop_w.min(comp.blocks_x - comp_crop_x);
            let new_by: usize = comp_crop_h.min(comp.blocks_y - comp_crop_y);

            let mut new_blocks: Vec<[i16; 64]> = Vec::with_capacity(new_bx * new_by);
            for by in 0..new_by {
                for bx in 0..new_bx {
                    let old_idx: usize = (comp_crop_y + by) * comp.blocks_x + (comp_crop_x + bx);
                    new_blocks.push(comp.blocks[old_idx]);
                }
            }
            comp.blocks = new_blocks;
            comp.blocks_x = new_bx;
            comp.blocks_y = new_by;
        }
    }

    // CUSTOM_FILTER: invoke user callback on each block after spatial transform.
    if let Some(ref filter) = options.custom_filter {
        for (ci, comp) in coeffs.components.iter_mut().enumerate() {
            let blocks_x: usize = comp.blocks_x;
            for by in 0..comp.blocks_y {
                for bx in 0..blocks_x {
                    let block_idx: usize = by * blocks_x + bx;
                    filter(&mut comp.blocks[block_idx], ci, bx, by);
                }
            }
        }
    }

    // NO_OUTPUT: skip writing, return empty.
    if options.no_output {
        return Ok(Vec::new());
    }

    // Apply restart interval: preserve source RI unless user explicitly overrides.
    // Matches C jpegtran behavior — source restart interval flows through
    // transforms unchanged. Only overwrite when explicitly requested.
    //
    // When `restart_in_rows == true`, the user-supplied value is in MCU rows.
    // For sequential/optimized writers the DRI is a single scan-wide value,
    // so it is precomputed against the output interleaved MCU grid. The
    // progressive writer recomputes per-scan DRI from `progressive_restart_rows`
    // below (matches C `per_scan_setup` which updates `cinfo->restart_interval`
    // based on each scan's `MCUs_per_row` — interleaved scans and
    // non-interleaved AC scans use different row counts).
    let progressive_restart_rows: Option<u16> =
        if options.progressive && options.restart_interval > 0 && options.restart_in_rows {
            Some(options.restart_interval)
        } else {
            None
        };

    if options.restart_interval > 0 {
        if options.restart_in_rows {
            let max_h: usize = coeffs
                .components
                .iter()
                .map(|c| c.h_sampling as usize)
                .max()
                .unwrap_or(1);
            let output_mcus_per_row: usize = (coeffs.width as usize).div_ceil(max_h * 8);
            let dri: usize = options.restart_interval as usize * output_mcus_per_row;
            coeffs.restart_interval = dri.min(u16::MAX as usize) as u16;
        } else {
            coeffs.restart_interval = options.restart_interval;
        }
    }
    // Source RI carried over from the input JPEG can become invalid after
    // dimension-swapping transforms with trim, since the output MCU grid is
    // fundamentally different. Clear to avoid producing truncated entropy data.
    let swaps_dimensions: bool = matches!(
        options.op,
        crate::transform::TransformOp::Rot90
            | crate::transform::TransformOp::Rot270
            | crate::transform::TransformOp::Transpose
            | crate::transform::TransformOp::Transverse
    );
    if swaps_dimensions && options.trim && options.restart_interval == 0 {
        coeffs.restart_interval = 0;
    }

    // Write output with the appropriate encoding. C jpegtran -progressive
    // implies -optimize (per-scan Huffman tables).
    //
    // The sampling-factor gate `max_{h,v} ∈ {1,2,4}` matches the eight standard
    // TJSAMP factors (444/422/440/420/411/441/410/24) — the set verified by
    // `tests/regression_progressive_4pixel_chroma_transform.rs` (P3-4 closure)
    // and the full `c_tjtrantest_full` matrix. Non-standard 3x sampling
    // (max_h or max_v = 3) is unverified against `jpegtran -progressive` and
    // is tracked under P3-6; it falls back to optimized baseline until that
    // entry closes.
    //
    // The `data_blocks_{x,y} ≤ comp.blocks_{x,y}` check guards malformed
    // coefficient buffers where the stored block grid is smaller than the
    // image dimensions imply — well-formed coefficients from
    // `read_coefficients` always satisfy it.
    let progressive_safe: bool = options.progressive && {
        let max_h: usize = coeffs
            .components
            .iter()
            .map(|c| c.h_sampling as usize)
            .max()
            .unwrap_or(1);
        let max_v: usize = coeffs
            .components
            .iter()
            .map(|c| c.v_sampling as usize)
            .max()
            .unwrap_or(1);
        let standard_factors: bool =
            max_h.is_power_of_two() && max_v.is_power_of_two() && max_h <= 4 && max_v <= 4;
        standard_factors
            && coeffs.components.iter().all(|comp| {
                let dbx: usize =
                    (coeffs.width as usize * comp.h_sampling as usize).div_ceil(max_h * 8);
                let dby: usize =
                    (coeffs.height as usize * comp.v_sampling as usize).div_ceil(max_v * 8);
                dbx <= comp.blocks_x && dby <= comp.blocks_y
            })
    };
    // 12-bit precision (e.g. `monkey12.jpg` transcode) and coefficient
    // buffers with out-of-range baseline symbols MUST go through the
    // optimized Huffman writer. The non-optimized path uses the standard
    // Annex K tables, which do not define every possible DC/AC category;
    // using it anyway would encode zero-bit Huffman symbols and produce
    // a JPEG that downstream C tools reject. This applies to any source
    // — baseline JPEGs with custom DHT tables can also carry categories
    // beyond the Annex K range (DC > 11, AC > 10).
    let force_optimize: bool =
        coeffs.effective_precision() > 8 || needs_optimized_baseline_huffman(&coeffs);
    let output: Vec<u8> = if options.arithmetic && progressive_safe {
        write_coefficients_progressive_arithmetic(&coeffs, progressive_restart_rows)?
    } else if options.arithmetic {
        write_coefficients_arithmetic(&coeffs)?
    } else if progressive_safe {
        write_coefficients_progressive(&coeffs, progressive_restart_rows)?
    } else if options.optimize || force_optimize {
        write_coefficients_optimized(&coeffs)?
    } else {
        write_coefficients(&coeffs)?
    };

    // Inject saved markers from the source if copy_markers is enabled.
    if !saved_markers.is_empty() {
        Ok(encoder_pipeline::inject_saved_markers(
            &output,
            &saved_markers,
        ))
    } else {
        Ok(output)
    }
}

fn huffman_category(value: i16) -> u8 {
    let magnitude: u16 = value.unsigned_abs();
    if magnitude == 0 {
        0
    } else {
        (16 - magnitude.leading_zeros()) as u8
    }
}

fn needs_optimized_baseline_huffman(coeffs: &JpegCoefficients) -> bool {
    if coeffs.components.is_empty() {
        return false;
    }

    let max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let mcus_x: usize = coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let mcus_y: usize = coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    let data_blocks_x: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.width as usize * c.h_sampling as usize).div_ceil(max_h * 8))
        .collect();
    let data_blocks_y: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.height as usize * c.v_sampling as usize).div_ceil(max_v * 8))
        .collect();

    let mut prev_dc: Vec<i16> = vec![0; coeffs.components.len()];
    let ri: u32 = coeffs.restart_interval as u32;
    let mut mcu_count: u32 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                prev_dc.fill(0);
            }

            for (ci, comp) in coeffs.components.iter().enumerate() {
                for v in 0..comp.v_sampling as usize {
                    for h in 0..comp.h_sampling as usize {
                        let bx: usize = mcu_x * comp.h_sampling as usize + h;
                        let by: usize = mcu_y * comp.v_sampling as usize + v;
                        let is_dummy: bool = bx >= data_blocks_x[ci] || by >= data_blocks_y[ci];

                        if is_dummy {
                            continue;
                        }

                        let block: &[i16; 64] = &comp.blocks[by * comp.blocks_x + bx];
                        let dc_diff: i16 = block[0].wrapping_sub(prev_dc[ci]);
                        prev_dc[ci] = block[0];

                        if huffman_category(dc_diff) > 11 {
                            return true;
                        }

                        if block[1..]
                            .iter()
                            .any(|&coef| coef != 0 && huffman_category(coef) > 10)
                        {
                            return true;
                        }
                    }
                }
            }
            mcu_count += 1;
        }
    }

    false
}

/// Write DCT coefficients with optimized Huffman tables (2-pass encoding).
///
/// Pass 1 gathers symbol frequencies from the coefficient data, then
/// generates optimal Huffman tables. Pass 2 encodes with those tables.
pub fn write_coefficients_optimized(coeffs: &JpegCoefficients) -> Result<Vec<u8>> {
    use crate::encode::huff_opt;

    let num_components: usize = coeffs.components.len();
    let is_grayscale: bool = num_components == 1;
    let uses_secondary_table: bool = !is_grayscale && !uses_single_rgb_coding_table(coeffs);

    let opt_max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let opt_max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let interleaved_mcus_x: usize =
        coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let interleaved_mcus_y: usize =
        coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    let opt_data_bx: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.width as usize * c.h_sampling as usize).div_ceil(opt_max_h * 8))
        .collect();
    let opt_data_by: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.height as usize * c.v_sampling as usize).div_ceil(opt_max_v * 8))
        .collect();
    let (mcus_x, mcus_y): (usize, usize) = if is_grayscale {
        (opt_data_bx[0], opt_data_by[0])
    } else {
        (interleaved_mcus_x, interleaved_mcus_y)
    };

    // === Pass 1: gather symbol frequencies ===
    let mut dc_luma_freq = [0u32; 257];
    let mut dc_chroma_freq = [0u32; 257];
    let mut ac_luma_freq = [0u32; 257];
    let mut ac_chroma_freq = [0u32; 257];

    let mut prev_dc: Vec<i16> = vec![0i16; num_components];
    let opt_dummy: [i16; 64] = [0i16; 64];
    let p1_ri: u32 = coeffs.restart_interval as u32;
    let mut p1_mcu_count: u32 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            // Reset DC predictions at restart boundaries (matching Pass 2)
            if p1_ri > 0 && p1_mcu_count > 0 && p1_mcu_count.is_multiple_of(p1_ri) {
                for dc in prev_dc.iter_mut() {
                    *dc = 0;
                }
            }

            for (ci, comp) in coeffs.components.iter().enumerate() {
                let dc_freq: &mut [u32; 257] = if coding_table_for_component(coeffs, ci) == 0 {
                    &mut dc_luma_freq
                } else {
                    &mut dc_chroma_freq
                };
                let ac_freq: &mut [u32; 257] = if coding_table_for_component(coeffs, ci) == 0 {
                    &mut ac_luma_freq
                } else {
                    &mut ac_chroma_freq
                };

                let h_blocks: usize = if is_grayscale {
                    1
                } else {
                    comp.h_sampling as usize
                };
                let v_blocks: usize = if is_grayscale {
                    1
                } else {
                    comp.v_sampling as usize
                };
                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let bx: usize = mcu_x * h_blocks + h;
                        let by: usize = mcu_y * v_blocks + v;
                        let is_dummy: bool = bx >= opt_data_bx[ci] || by >= opt_data_by[ci];

                        let block: &[i16; 64] = if is_dummy {
                            &opt_dummy
                        } else {
                            let block_idx: usize = by * comp.blocks_x + bx;
                            &comp.blocks[block_idx]
                        };

                        let dc_val: i16 = if is_dummy { prev_dc[ci] } else { block[0] };
                        // wrapping_sub: corrupt/adversarial input can pair DCs
                        // whose difference exceeds i16; wrap matches the
                        // baseline-encoder convention (huffman_encode.rs:461/495)
                        // and gather_dc_symbol's leading-zeros classification.
                        let diff: i16 = dc_val.wrapping_sub(prev_dc[ci]);
                        prev_dc[ci] = dc_val;
                        // Magnitude category 16 cannot be expressed in a DHT
                        // symbol (4-bit size field); in i16 storage only a
                        // value/diff of -32768 produces it. C's scalar encoder
                        // rejects it with ERREXIT(JERR_BAD_DCT_COEF)
                        // (jchuff.c); its SIMD path silently emits an
                        // undecodable stream instead — match the scalar
                        // contract (Fuzz Smoke run 30064906856, P4-35).
                        //
                        // Deliberate leniency vs C: C computes the DC diff in
                        // int and ERREXITs when the *pre-wrap* magnitude needs
                        // category 16 (e.g. 32767 - (-2) = 32769). Our wrapped
                        // diff stays representable, pass 2 wraps identically,
                        // and the decoder's own predictor wrap recovers the
                        // same i16 DC values — the output is valid and
                        // decodable, so we transcode where C refuses. Only the
                        // wrapped value -32768 (true category 16) must reject.
                        if diff == i16::MIN || block[1..].contains(&i16::MIN) {
                            return Err(JpegError::CorruptData(
                                "DCT coefficient out of range for Huffman coding".to_string(),
                            ));
                        }
                        huff_opt::gather_dc_symbol(diff, dc_freq);
                        huff_opt::gather_ac_symbols(block, ac_freq);
                    }
                }
            }
            p1_mcu_count += 1;
        }
    }

    // Add pseudo-symbol (required by Annex K.2 optimal table generation).
    dc_luma_freq[256] = 1;
    ac_luma_freq[256] = 1;
    dc_chroma_freq[256] = 1;
    ac_chroma_freq[256] = 1;

    // Generate optimal tables.
    let (dc_luma_bits, dc_luma_values) = huff_opt::gen_optimal_table(&dc_luma_freq);
    let (ac_luma_bits, ac_luma_values) = huff_opt::gen_optimal_table(&ac_luma_freq);
    let (dc_chroma_bits, dc_chroma_values) = huff_opt::gen_optimal_table(&dc_chroma_freq);
    let (ac_chroma_bits, ac_chroma_values) = huff_opt::gen_optimal_table(&ac_chroma_freq);

    // Build encoding tables from optimal bits/values.
    let dc_luma_table = build_huff_table(&dc_luma_bits, &dc_luma_values);
    let ac_luma_table = build_huff_table(&ac_luma_bits, &ac_luma_values);
    let dc_chroma_table = build_huff_table(&dc_chroma_bits, &dc_chroma_values);
    let ac_chroma_table = build_huff_table(&ac_chroma_bits, &ac_chroma_values);

    // === Pass 2: entropy encode with optimal tables ===
    let mut bit_writer = BitWriter::new(coeffs.width as usize * coeffs.height as usize);
    let mut prev_dc_pass2: Vec<i16> = vec![0i16; num_components];
    let opt_ri: u32 = coeffs.restart_interval as u32;
    let mut opt_mcu_count: u32 = 0;
    let mut opt_restart_idx: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if opt_ri > 0 && opt_mcu_count > 0 && opt_mcu_count.is_multiple_of(opt_ri) {
                bit_writer.flush();
                bit_writer.write_restart_marker(opt_restart_idx);
                opt_restart_idx = (opt_restart_idx + 1) & 7;
                for dc in prev_dc_pass2.iter_mut() {
                    *dc = 0;
                }
            }

            for (ci, comp) in coeffs.components.iter().enumerate() {
                let dc_table = if coding_table_for_component(coeffs, ci) == 0 {
                    &dc_luma_table
                } else {
                    &dc_chroma_table
                };
                let ac_table = if coding_table_for_component(coeffs, ci) == 0 {
                    &ac_luma_table
                } else {
                    &ac_chroma_table
                };

                let h_blocks: usize = if is_grayscale {
                    1
                } else {
                    comp.h_sampling as usize
                };
                let v_blocks: usize = if is_grayscale {
                    1
                } else {
                    comp.v_sampling as usize
                };
                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let bx: usize = mcu_x * h_blocks + h;
                        let by: usize = mcu_y * v_blocks + v;
                        let is_dummy: bool = bx >= opt_data_bx[ci] || by >= opt_data_by[ci];

                        if is_dummy {
                            let mut dblock: [i16; 64] = opt_dummy;
                            dblock[0] = prev_dc_pass2[ci];
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                &dblock,
                                &mut prev_dc_pass2[ci],
                                dc_table,
                                ac_table,
                            );
                        } else {
                            let block_idx: usize = by * comp.blocks_x + bx;
                            let block: &[i16; 64] = &comp.blocks[block_idx];
                            HuffmanEncoder::encode_block(
                                &mut bit_writer,
                                block,
                                &mut prev_dc_pass2[ci],
                                dc_table,
                                ac_table,
                            );
                        }
                    }
                }
            }
            opt_mcu_count += 1;
        }
    }

    bit_writer.flush();

    // === Assemble output ===
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    write_coefficient_colorspace_marker(&mut output, coeffs);

    // Quantization tables.
    for (i, qt) in coeffs.quant_tables.iter().enumerate() {
        marker_writer::write_dqt(&mut output, i as u8, qt);
    }

    // Frame header — SOF1 for 16-bit quant tables OR sample precision > 8
    // (e.g. 12-bit `monkey12.jpg` transcode), SOF0 otherwise.
    let opt_precision: u8 = coeffs.effective_precision();
    let opt_needs_ext: bool = opt_precision > 8
        || coeffs
            .quant_tables
            .iter()
            .any(|qt| qt.iter().any(|&v| v > 255));
    let opt_comps: Vec<(u8, u8, u8, u8)> = coeffs
        .components
        .iter()
        .map(|c| {
            (
                c.component_id,
                c.h_sampling,
                c.v_sampling,
                c.quant_table_index,
            )
        })
        .collect();
    output.push(0xFF);
    output.push(if opt_needs_ext { 0xC1 } else { 0xC0 });
    let opt_sof_len: u16 = 2 + 1 + 2 + 2 + 1 + (opt_comps.len() as u16 * 3);
    output.extend_from_slice(&opt_sof_len.to_be_bytes());
    output.push(opt_precision);
    output.extend_from_slice(&coeffs.height.to_be_bytes());
    output.extend_from_slice(&coeffs.width.to_be_bytes());
    output.push(opt_comps.len() as u8);
    for &(id, h_samp, v_samp, quant_tbl_id) in &opt_comps {
        output.push(id);
        output.push((h_samp << 4) | v_samp);
        output.push(quant_tbl_id);
    }

    // Optimized Huffman tables.
    marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);
    marker_writer::write_dht(&mut output, 1, 0, &ac_luma_bits, &ac_luma_values);
    if uses_secondary_table {
        marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);
        marker_writer::write_dht(&mut output, 1, 1, &ac_chroma_bits, &ac_chroma_values);
    }

    // DRI (restart interval)
    if coeffs.restart_interval > 0 {
        marker_writer::write_dri(&mut output, coeffs.restart_interval);
    }

    // Scan header — preserve source component IDs.
    let scan_components: Vec<(u8, u8, u8)> = coeffs
        .components
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let tbl: u8 = coding_table_for_component(coeffs, i) as u8;
            (c.component_id, tbl, tbl)
        })
        .collect();
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Write DCT coefficients as progressive JPEG (SOF2, multi-scan) with
/// per-scan optimized Huffman tables.
///
/// Matches C `jpegtran -progressive` behavior, which implies `-optimize`.
/// Uses the default libjpeg-turbo scan progression (simple_progression).
///
/// `restart_rows` selects the restart accounting mode:
/// - `Some(rows)` — row mode (`jpegtran -restart N`): the DRI is recomputed
///   per scan as `rows * MCUs_per_row_of_scan`, where `MCUs_per_row` is the
///   interleaved MCU count for multi-component scans and `width_in_blocks`
///   for non-interleaved AC scans (matches C `per_scan_setup`).
/// - `None` — byte mode (`-restart Nb`) or source-preserved RI: `coeffs.restart_interval`
///   is used uniformly for every scan.
pub fn write_coefficients_progressive(
    coeffs: &JpegCoefficients,
    restart_rows: Option<u16>,
) -> Result<Vec<u8>> {
    use crate::encode::huff_opt;
    use crate::encode::progressive::{generic_progression, simple_progression};

    let num_components: usize = coeffs.components.len();
    let is_grayscale: bool = num_components == 1;
    let uses_secondary_table: bool = !is_grayscale && !uses_single_rgb_coding_table(coeffs);

    let max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let interleaved_mcus_x: usize =
        coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let interleaved_mcus_y: usize =
        coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    // Per-component actual block counts for non-interleaved AC scans.
    // C libjpeg-turbo only encodes width_in_blocks × height_in_blocks data
    // units for non-interleaved scans, not the MCU-padded count.
    let data_blocks_x: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.width as usize * c.h_sampling as usize).div_ceil(max_h * 8))
        .collect();
    let data_blocks_y: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.height as usize * c.v_sampling as usize).div_ceil(max_v * 8))
        .collect();
    let (mcus_x, mcus_y): (usize, usize) = if is_grayscale {
        (data_blocks_x[0], data_blocks_y[0])
    } else {
        (interleaved_mcus_x, interleaved_mcus_y)
    };

    let scans = if uses_single_rgb_coding_table(coeffs) {
        generic_progression(num_components)
    } else {
        simple_progression(num_components)
    };

    // === Assemble output header ===
    let mut output: Vec<u8> = Vec::with_capacity(coeffs.width as usize * coeffs.height as usize);

    marker_writer::write_soi(&mut output);
    write_coefficient_colorspace_marker(&mut output, coeffs);

    for (i, qt) in coeffs.quant_tables.iter().enumerate() {
        marker_writer::write_dqt(&mut output, i as u8, qt);
    }

    // SOF2 (progressive)
    let components: Vec<(u8, u8, u8, u8)> = coeffs
        .components
        .iter()
        .map(|c| {
            (
                c.component_id,
                c.h_sampling,
                c.v_sampling,
                c.quant_table_index,
            )
        })
        .collect();
    marker_writer::write_sof2_with_precision(
        &mut output,
        coeffs.width,
        coeffs.height,
        &components,
        coeffs.effective_precision(),
    );

    let mut bit_writer: BitWriter =
        BitWriter::new(coeffs.width as usize * coeffs.height as usize / 4);

    // DRI is emitted per-scan, after DHT and before SOS — only when the
    // restart interval changes from the previous scan (matches C jcmarker.c
    // `write_scan_header`). `saved_ri` starts at 0 so the first scan emits
    // DRI whenever the image has restart markers.
    let mut saved_ri: u16 = 0;

    // Compute the per-scan DRI used for RST emission and stream markers.
    // Row mode follows C `per_scan_setup`: interleaved (multi-component)
    // scans use the interleaved MCU grid; non-interleaved scans use that
    // component's `width_in_blocks`. Byte mode applies `coeffs.restart_interval`
    // uniformly.
    let per_scan_ri = |scan_ci: &[usize]| -> u16 {
        match restart_rows {
            Some(rows) => {
                let mcus_per_row: usize = if scan_ci.len() == 1 {
                    // Non-interleaved: MCU row count = component's width in blocks.
                    let ci: usize = scan_ci[0];
                    let comp = &coeffs.components[ci];
                    (coeffs.width as usize * comp.h_sampling as usize).div_ceil(max_h * 8)
                } else {
                    // Interleaved: width divided by max horizontal sampling × 8.
                    (coeffs.width as usize).div_ceil(max_h * 8)
                };
                let dri: usize = rows as usize * mcus_per_row;
                dri.min(u16::MAX as usize) as u16
            }
            None => coeffs.restart_interval,
        }
    };

    // === Encode each scan with per-scan optimized Huffman tables ===
    for scan in &scans {
        let is_dc_scan: bool = scan.ss == 0 && scan.se == 0;
        let is_first: bool = scan.ah == 0;
        let scan_ri: u16 = per_scan_ri(&scan.component_indices);

        // Build SOS component list preserving source component IDs.
        // DC refine scans (ah>0) use no Huffman table — set Td=0 to match C.
        let scan_comps: Vec<(u8, u8, u8)> = scan
            .component_indices
            .iter()
            .map(|&ci| {
                let tbl: u8 = coding_table_for_component(coeffs, ci) as u8;
                let dc_tbl: u8 = if is_dc_scan && is_first { tbl } else { 0 };
                let ac_tbl: u8 = if is_dc_scan { 0 } else { tbl };
                (coeffs.components[ci].component_id, dc_tbl, ac_tbl)
            })
            .collect();

        if is_dc_scan && is_first {
            // === DC FIRST scan ===
            // Pass 1: gather DC symbol frequencies.
            let mut dc_luma_freq = [0u32; 257];
            let mut dc_chroma_freq = [0u32; 257];
            dc_luma_freq[256] = 1;
            dc_chroma_freq[256] = 1;

            let mut prev_dc: Vec<i16> = vec![0i16; scan.component_indices.len()];
            let ri: u32 = scan_ri as u32;
            let mut restarts_to_go: u32 = ri;

            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if ri > 0 && restarts_to_go == 0 {
                        for dc in prev_dc.iter_mut() {
                            *dc = 0;
                        }
                        restarts_to_go = ri;
                    }
                    for (scan_ci, &ci) in scan.component_indices.iter().enumerate() {
                        let comp = &coeffs.components[ci];
                        let freq: &mut [u32; 257] = if coding_table_for_component(coeffs, ci) == 0 {
                            &mut dc_luma_freq
                        } else {
                            &mut dc_chroma_freq
                        };
                        let h_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.h_sampling as usize
                        };
                        let v_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.v_sampling as usize
                        };
                        for v in 0..v_blocks {
                            for h in 0..h_blocks {
                                let bx: usize = mcu_x * h_blocks + h;
                                let by: usize = mcu_y * v_blocks + v;
                                let is_dummy: bool =
                                    bx >= data_blocks_x[ci] || by >= data_blocks_y[ci];
                                let dc: i16 = if is_dummy {
                                    prev_dc[scan_ci]
                                } else {
                                    let block: &[i16; 64] = &comp.blocks[by * comp.blocks_x + bx];
                                    block[0] >> scan.al
                                };
                                let diff: i16 = dc.wrapping_sub(prev_dc[scan_ci]);
                                prev_dc[scan_ci] = dc;
                                // DC diff of -32768 needs magnitude category 16,
                                // which no DHT symbol can express — C's scalar
                                // encoder ERREXITs (JERR_BAD_DCT_COEF); match it
                                // (P4-35).
                                if diff == i16::MIN {
                                    return Err(JpegError::CorruptData(
                                        "DCT coefficient out of range for Huffman coding"
                                            .to_string(),
                                    ));
                                }
                                huff_opt::gather_dc_symbol(diff, freq);
                            }
                        }
                    }
                    if ri > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }

            // Generate optimal tables and write DHT markers.
            let (dc_luma_bits, dc_luma_values) = huff_opt::gen_optimal_table(&dc_luma_freq);
            let dc_luma_table: HuffTable = build_huff_table(&dc_luma_bits, &dc_luma_values);
            marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);

            let dc_chroma_table: HuffTable = if uses_secondary_table {
                let (bits, vals) = huff_opt::gen_optimal_table(&dc_chroma_freq);
                marker_writer::write_dht(&mut output, 0, 1, &bits, &vals);
                build_huff_table(&bits, &vals)
            } else {
                // Unused for grayscale.
                build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES)
            };

            if scan_ri != saved_ri {
                marker_writer::write_dri(&mut output, scan_ri);
                saved_ri = scan_ri;
            }
            marker_writer::write_sos_progressive(
                &mut output,
                &scan_comps,
                scan.ss,
                scan.se,
                scan.ah,
                scan.al,
            );

            // Pass 2: encode DC first scan.
            bit_writer.reset();
            let mut enc_prev_dc: Vec<i16> = vec![0i16; scan.component_indices.len()];
            let ri: u32 = scan_ri as u32;
            let mut restarts_to_go: u32 = ri;
            let mut next_restart_num: u8 = 0;

            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if ri > 0 && restarts_to_go == 0 {
                        bit_writer.flush_restart();
                        bit_writer.write_restart_marker(next_restart_num);
                        next_restart_num = (next_restart_num + 1) & 7;
                        for dc in enc_prev_dc.iter_mut() {
                            *dc = 0;
                        }
                        restarts_to_go = ri;
                    }
                    for (scan_ci, &ci) in scan.component_indices.iter().enumerate() {
                        let comp = &coeffs.components[ci];
                        let dc_table: &HuffTable = if coding_table_for_component(coeffs, ci) == 0 {
                            &dc_luma_table
                        } else {
                            &dc_chroma_table
                        };
                        let h_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.h_sampling as usize
                        };
                        let v_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.v_sampling as usize
                        };
                        for v in 0..v_blocks {
                            for h in 0..h_blocks {
                                let bx: usize = mcu_x * h_blocks + h;
                                let by: usize = mcu_y * v_blocks + v;
                                let is_dummy: bool =
                                    bx >= data_blocks_x[ci] || by >= data_blocks_y[ci];
                                let dc: i16 = if is_dummy {
                                    enc_prev_dc[scan_ci]
                                } else {
                                    let block: &[i16; 64] = &comp.blocks[by * comp.blocks_x + bx];
                                    block[0] >> scan.al
                                };
                                let diff: i16 = dc.wrapping_sub(enc_prev_dc[scan_ci]);
                                enc_prev_dc[scan_ci] = dc;
                                HuffmanEncoder::encode_dc_only(&mut bit_writer, diff, dc_table);
                            }
                        }
                    }
                    if ri > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }

            bit_writer.flush();
            output.extend_from_slice(bit_writer.data());
        } else if is_dc_scan {
            // === DC REFINE scan ===
            // No Huffman table needed — just raw bits.
            if scan_ri != saved_ri {
                marker_writer::write_dri(&mut output, scan_ri);
                saved_ri = scan_ri;
            }
            marker_writer::write_sos_progressive(
                &mut output,
                &scan_comps,
                scan.ss,
                scan.se,
                scan.ah,
                scan.al,
            );

            bit_writer.reset();
            // Track last real DC for dummy block refine bits.
            let mut refine_prev_dc: Vec<i16> = vec![0i16; num_components];
            let ri: u32 = scan_ri as u32;
            let mut restarts_to_go: u32 = ri;
            let mut next_restart_num: u8 = 0;

            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if ri > 0 && restarts_to_go == 0 {
                        bit_writer.flush_restart();
                        bit_writer.write_restart_marker(next_restart_num);
                        next_restart_num = (next_restart_num + 1) & 7;
                        restarts_to_go = ri;
                    }
                    for &ci in &scan.component_indices {
                        let comp = &coeffs.components[ci];
                        let h_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.h_sampling as usize
                        };
                        let v_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.v_sampling as usize
                        };
                        for v in 0..v_blocks {
                            for h in 0..h_blocks {
                                let bx: usize = mcu_x * h_blocks + h;
                                let by: usize = mcu_y * v_blocks + v;
                                let is_dummy: bool =
                                    bx >= data_blocks_x[ci] || by >= data_blocks_y[ci];
                                let dc_val: i16 = if is_dummy {
                                    refine_prev_dc[ci]
                                } else {
                                    let block: &[i16; 64] = &comp.blocks[by * comp.blocks_x + bx];
                                    refine_prev_dc[ci] = block[0];
                                    block[0]
                                };
                                let bit: u32 = ((dc_val >> scan.al) & 1) as u32;
                                bit_writer.put_bits(bit, 1);
                            }
                        }
                    }
                    if ri > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }

            bit_writer.flush();
            output.extend_from_slice(bit_writer.data());
        } else {
            // === AC scan (single component, non-interleaved) ===
            let ci: usize = scan.component_indices[0];
            let comp = &coeffs.components[ci];
            let wib: usize = data_blocks_x[ci].min(comp.blocks_x);
            let hib: usize = data_blocks_y[ci].min(comp.blocks_y);
            let stride: usize = comp.blocks_x;
            let ss: usize = scan.ss as usize;
            let se: usize = scan.se as usize;
            let al: u8 = scan.al;
            let band_len: usize = se - ss + 1;

            if is_first {
                // --- AC first scan ---
                // Pass 1: gather AC symbol frequencies.
                let mut ac_freq = [0u32; 257];
                ac_freq[256] = 1;
                let mut eobrun_gather: u32 = 0;
                let ri: u32 = scan_ri as u32;
                let mut restarts_to_go: u32 = ri;

                for by in 0..hib {
                    for bx in 0..wib {
                        if ri > 0 && restarts_to_go == 0 {
                            if eobrun_gather > 0 {
                                let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                                ac_freq[(nbits as usize) << 4] += 1;
                                eobrun_gather = 0;
                            }
                            restarts_to_go = ri;
                        }
                        let block: &[i16; 64] = &comp.blocks[by * stride + bx];

                        let mut zerobits: u64 = 0;
                        let mut values = [0u16; 64];

                        for i in 0..band_len {
                            let coeff: i16 = block[ss + i];
                            if coeff == 0 {
                                continue;
                            }
                            // i32 widen to handle adversarial coeff = i16::MIN:
                            // |i16::MIN| = 32768 doesn't fit in i16 (the
                            // branchless abs `(c ^ -1) - -1 = ~c + 1`
                            // overflowed). Found via fuzz_transform_options
                            // round-3 (CI run 25215431132) at coefficient.rs:1696.
                            let coeff: i32 = coeff as i32;
                            let sign_mask: i32 = coeff >> 31;
                            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
                            let temp: u16 = (abs_coeff >> al) as u16;
                            if temp == 0 {
                                continue;
                            }
                            // temp = 32768 (coeff = i16::MIN with al = 0) needs
                            // magnitude category 16, which no DHT symbol can
                            // express — C's scalar encoder ERREXITs
                            // (JERR_BAD_DCT_COEF); match it (P4-35).
                            if temp >= 0x8000 {
                                return Err(JpegError::CorruptData(
                                    "DCT coefficient out of range for Huffman coding".to_string(),
                                ));
                            }
                            values[i] = temp;
                            zerobits |= 1u64 << i;
                        }

                        if zerobits == 0 {
                            eobrun_gather += 1;
                            if eobrun_gather == 0x7FFF {
                                let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                                ac_freq[(nbits as usize) << 4] += 1;
                                eobrun_gather = 0;
                            }
                            if ri > 0 {
                                restarts_to_go -= 1;
                            }
                            continue;
                        }

                        if eobrun_gather > 0 {
                            let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                            ac_freq[(nbits as usize) << 4] += 1;
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
                                let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                                ac_freq[(nbits as usize) << 4] += 1;
                                eobrun_gather = 0;
                            }
                        }
                        if ri > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
                if eobrun_gather > 0 {
                    let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                    ac_freq[(nbits as usize) << 4] += 1;
                }

                // Generate optimal table, write DHT + SOS.
                let (ac_bits, ac_values) = huff_opt::gen_optimal_table(&ac_freq);
                let table_id: u8 = coding_table_for_component(coeffs, ci) as u8;
                marker_writer::write_dht(&mut output, 1, table_id, &ac_bits, &ac_values);
                if scan_ri != saved_ri {
                    marker_writer::write_dri(&mut output, scan_ri);
                    saved_ri = scan_ri;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    &scan_comps,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                // Pass 2: encode AC first scan.
                let ac_table: HuffTable = build_huff_table(&ac_bits, &ac_values);
                bit_writer.reset();
                let mut eobrun: u32 = 0;
                let ri: u32 = scan_ri as u32;
                let mut restarts_to_go: u32 = ri;
                let mut next_restart_num: u8 = 0;

                for by in 0..hib {
                    for bx in 0..wib {
                        if ri > 0 && restarts_to_go == 0 {
                            if eobrun > 0 {
                                encoder_pipeline::emit_eobrun(
                                    &ac_table,
                                    &mut bit_writer,
                                    &mut eobrun,
                                );
                            }
                            bit_writer.flush_restart();
                            bit_writer.write_restart_marker(next_restart_num);
                            next_restart_num = (next_restart_num + 1) & 7;
                            restarts_to_go = ri;
                        }
                        let block: &[i16; 64] = &comp.blocks[by * stride + bx];
                        encoder_pipeline::encode_ac_first_block(
                            block,
                            ss,
                            se,
                            al,
                            &ac_table,
                            &mut bit_writer,
                            &mut eobrun,
                        );
                        if ri > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
                if eobrun > 0 {
                    encoder_pipeline::emit_eobrun(&ac_table, &mut bit_writer, &mut eobrun);
                }

                bit_writer.flush();
                output.extend_from_slice(bit_writer.data());
            } else {
                // --- AC refine scan ---
                // Pass 1: gather AC refine symbol frequencies.
                let mut ac_freq = [0u32; 257];
                ac_freq[256] = 1;
                let mut eobrun_gather: u32 = 0;
                let mut be: usize = 0;
                let ri: u32 = scan_ri as u32;
                let mut restarts_to_go: u32 = ri;

                for by in 0..hib {
                    for bx in 0..wib {
                        if ri > 0 && restarts_to_go == 0 {
                            if eobrun_gather > 0 {
                                let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                                ac_freq[(nbits as usize) << 4] += 1;
                            }
                            // Match C `emit_restart` (jcphuff.c:444-446): EOBRUN
                            // and BE correction buffer are always cleared at an
                            // RST boundary regardless of whether EOBRUN was
                            // pending. Keeps `be` accounting robust even if the
                            // gather/encode invariant `be>0 ⇒ eobrun>0` ever
                            // loosens.
                            eobrun_gather = 0;
                            be = 0;
                            restarts_to_go = ri;
                        }
                        let block: &[i16; 64] = &comp.blocks[by * stride + bx];

                        let mut absvals = [0u16; 64];
                        let mut eob_pos: usize = 0;

                        for i in 0..band_len {
                            let coeff: i32 = block[ss + i] as i32;
                            let sign_mask: i32 = coeff >> 31;
                            let abs_coeff: i32 = (coeff ^ sign_mask) - sign_mask;
                            let temp: u16 = (abs_coeff >> al) as u16;
                            absvals[i] = temp;
                            if temp == 1 {
                                eob_pos = i + 1;
                            }
                        }

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
                                    let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                                    ac_freq[(nbits as usize) << 4] += 1;
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
                                let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                                ac_freq[(nbits as usize) << 4] += 1;
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
                            if eobrun_gather == 0x7FFF
                                || be > (encoder_pipeline::MAX_CORR_BITS - 64 + 1)
                            {
                                let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                                ac_freq[(nbits as usize) << 4] += 1;
                                eobrun_gather = 0;
                                be = 0;
                            }
                        }
                        if ri > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
                if eobrun_gather > 0 {
                    let nbits: u8 = (32 - eobrun_gather.leading_zeros()) as u8 - 1;
                    ac_freq[(nbits as usize) << 4] += 1;
                }

                // Generate optimal table, write DHT + SOS.
                let (ac_bits, ac_values) = huff_opt::gen_optimal_table(&ac_freq);
                let table_id: u8 = coding_table_for_component(coeffs, ci) as u8;
                marker_writer::write_dht(&mut output, 1, table_id, &ac_bits, &ac_values);
                if scan_ri != saved_ri {
                    marker_writer::write_dri(&mut output, scan_ri);
                    saved_ri = scan_ri;
                }
                marker_writer::write_sos_progressive(
                    &mut output,
                    &scan_comps,
                    scan.ss,
                    scan.se,
                    scan.ah,
                    scan.al,
                );

                // Pass 2: encode AC refine scan.
                let ac_table: HuffTable = build_huff_table(&ac_bits, &ac_values);
                bit_writer.reset();
                let mut eobrun: u32 = 0;
                let mut corr_buffer: Vec<u8> = Vec::with_capacity(encoder_pipeline::MAX_CORR_BITS);
                let ri: u32 = scan_ri as u32;
                let mut restarts_to_go: u32 = ri;
                let mut next_restart_num: u8 = 0;

                for by in 0..hib {
                    for bx in 0..wib {
                        if ri > 0 && restarts_to_go == 0 {
                            if eobrun > 0 {
                                encoder_pipeline::emit_eobrun_with_corr(
                                    &ac_table,
                                    &mut bit_writer,
                                    &mut eobrun,
                                    &mut corr_buffer,
                                );
                            }
                            bit_writer.flush_restart();
                            bit_writer.write_restart_marker(next_restart_num);
                            next_restart_num = (next_restart_num + 1) & 7;
                            restarts_to_go = ri;
                        }
                        let block: &[i16; 64] = &comp.blocks[by * stride + bx];
                        encoder_pipeline::encode_ac_refine_block(
                            block,
                            ss,
                            se,
                            al,
                            &ac_table,
                            &mut bit_writer,
                            &mut eobrun,
                            &mut corr_buffer,
                        );
                        if ri > 0 {
                            restarts_to_go -= 1;
                        }
                    }
                }
                if eobrun > 0 {
                    encoder_pipeline::emit_eobrun_with_corr(
                        &ac_table,
                        &mut bit_writer,
                        &mut eobrun,
                        &mut corr_buffer,
                    );
                }

                bit_writer.flush();
                output.extend_from_slice(bit_writer.data());
            }
        }
    }

    marker_writer::write_eoi(&mut output);
    Ok(output)
}

/// Write DCT coefficients with arithmetic entropy coding (SOF9).
///
/// Re-encodes coefficient blocks using the JPEG arithmetic coder, matching
/// jpegtran's arithmetic output mode for non-progressive transforms.
pub fn write_coefficients_arithmetic(coeffs: &JpegCoefficients) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;

    let num_components: usize = coeffs.components.len();
    let is_grayscale: bool = num_components == 1;
    let num_arith_tables: usize = if is_grayscale || uses_single_rgb_coding_table(coeffs) {
        1
    } else {
        2
    };

    let max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let interleaved_mcus_x: usize =
        coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let interleaved_mcus_y: usize =
        coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    let data_blocks_x: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.width as usize * c.h_sampling as usize).div_ceil(max_h * 8))
        .collect();
    let data_blocks_y: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.height as usize * c.v_sampling as usize).div_ceil(max_v * 8))
        .collect();
    let (mcus_x, mcus_y): (usize, usize) = if is_grayscale {
        (data_blocks_x[0], data_blocks_y[0])
    } else {
        (interleaved_mcus_x, interleaved_mcus_y)
    };

    let mut arith_enc: ArithEncoder =
        ArithEncoder::new(coeffs.width as usize * coeffs.height as usize);
    let mut prev_dc: Vec<i16> = vec![0; num_components];
    let ri: u32 = coeffs.restart_interval as u32;
    let mut mcu_count: u32 = 0;
    let mut restart_idx: u8 = 0;

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            // Insert RST marker between MCU groups when restart_interval is set.
            // Mirrors libjpeg-turbo `jcarith.c` arithmetic restart: flush the
            // current entropy state byte-aligned, push `FF Dn`, reset coder
            // and DC predictors, then continue with the next group.
            if ri > 0 && mcu_count > 0 && mcu_count.is_multiple_of(ri) {
                arith_enc.emit_restart(restart_idx);
                restart_idx = restart_idx.wrapping_add(1) & 7;
                prev_dc.iter_mut().for_each(|v| *v = 0);
            }
            for (ci, comp) in coeffs.components.iter().enumerate() {
                let dc_tbl: usize = coding_table_for_component(coeffs, ci);
                let ac_tbl: usize = coding_table_for_component(coeffs, ci);

                let h_blocks: usize = if is_grayscale {
                    1
                } else {
                    comp.h_sampling as usize
                };
                let v_blocks: usize = if is_grayscale {
                    1
                } else {
                    comp.v_sampling as usize
                };
                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        let bx: usize = mcu_x * h_blocks + h;
                        let by: usize = mcu_y * v_blocks + v;

                        let mut dummy = [0i16; 64];
                        let block: &[i16; 64] =
                            if bx >= data_blocks_x[ci] || by >= data_blocks_y[ci] {
                                dummy[0] = prev_dc[ci];
                                &dummy
                            } else {
                                let real_block: &[i16; 64] = &comp.blocks[by * comp.blocks_x + bx];
                                prev_dc[ci] = real_block[0];
                                real_block
                            };

                        arith_enc.encode_dc_sequential(block, ci, dc_tbl);
                        arith_enc.encode_ac_sequential(block, ac_tbl);
                    }
                }
            }
            mcu_count += 1;
        }
    }

    arith_enc.finish();

    let mut output: Vec<u8> = Vec::with_capacity(arith_enc.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    write_coefficient_colorspace_marker(&mut output, coeffs);

    for (i, qt) in coeffs.quant_tables.iter().enumerate() {
        marker_writer::write_dqt(&mut output, i as u8, qt);
    }

    let components: Vec<(u8, u8, u8, u8)> = coeffs
        .components
        .iter()
        .map(|c| {
            (
                c.component_id,
                c.h_sampling,
                c.v_sampling,
                c.quant_table_index,
            )
        })
        .collect();
    marker_writer::write_sof9_with_precision(
        &mut output,
        coeffs.width,
        coeffs.height,
        &components,
        coeffs.effective_precision(),
    );

    let dc_params = [(0u8, 1u8); crate::decode::arithmetic::NUM_ARITH_TBLS];
    let ac_params = [5u8; crate::decode::arithmetic::NUM_ARITH_TBLS];
    let mut dc_in_use = [false; crate::decode::arithmetic::NUM_ARITH_TBLS];
    let mut ac_in_use = [false; crate::decode::arithmetic::NUM_ARITH_TBLS];
    for table in 0..num_arith_tables {
        dc_in_use[table] = true;
        ac_in_use[table] = true;
    }
    marker_writer::write_dac_selected(&mut output, &dc_in_use, &dc_params, &ac_in_use, &ac_params);

    if coeffs.restart_interval > 0 {
        marker_writer::write_dri(&mut output, coeffs.restart_interval);
    }

    let scan_components: Vec<(u8, u8, u8)> = coeffs
        .components
        .iter()
        .enumerate()
        .map(|(ci, c)| {
            let tbl: u8 = coding_table_for_component(coeffs, ci) as u8;
            (c.component_id, tbl, tbl)
        })
        .collect();
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(arith_enc.data());
    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Write DCT coefficients with arithmetic progressive entropy coding (SOF10).
///
/// Uses the default libjpeg-turbo progressive scan script and emits the
/// arithmetic conditioning marker for each scan, matching libjpeg marker order.
///
/// `restart_rows` selects the restart accounting mode (mirrors
/// `write_coefficients_progressive`):
/// - `Some(rows)` — row mode (`jpegtran -restart N`): the DRI is recomputed
///   per scan as `rows * MCUs_per_row_of_scan`. Multi-component scans use
///   the interleaved MCU grid; non-interleaved AC scans use that
///   component's `width_in_blocks`.
/// - `None` — byte mode (`-restart Nb`) or source-preserved RI:
///   `coeffs.restart_interval` applies uniformly across all scans.
pub fn write_coefficients_progressive_arithmetic(
    coeffs: &JpegCoefficients,
    restart_rows: Option<u16>,
) -> Result<Vec<u8>> {
    use crate::encode::arithmetic::ArithEncoder;
    use crate::encode::progressive::{generic_progression, simple_progression};

    let num_components: usize = coeffs.components.len();
    let is_grayscale: bool = num_components == 1;

    let max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let interleaved_mcus_x: usize =
        coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let interleaved_mcus_y: usize =
        coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    let data_blocks_x: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.width as usize * c.h_sampling as usize).div_ceil(max_h * 8))
        .collect();
    let data_blocks_y: Vec<usize> = coeffs
        .components
        .iter()
        .map(|c| (coeffs.height as usize * c.v_sampling as usize).div_ceil(max_v * 8))
        .collect();
    let (mcus_x, mcus_y): (usize, usize) = if is_grayscale {
        (data_blocks_x[0], data_blocks_y[0])
    } else {
        (interleaved_mcus_x, interleaved_mcus_y)
    };

    let scans = if uses_single_rgb_coding_table(coeffs) {
        generic_progression(num_components)
    } else {
        simple_progression(num_components)
    };
    let dc_params = [(0u8, 1u8); crate::decode::arithmetic::NUM_ARITH_TBLS];
    let ac_params = [5u8; crate::decode::arithmetic::NUM_ARITH_TBLS];

    let mut output: Vec<u8> = Vec::with_capacity(coeffs.width as usize * coeffs.height as usize);

    marker_writer::write_soi(&mut output);
    write_coefficient_colorspace_marker(&mut output, coeffs);

    for (i, qt) in coeffs.quant_tables.iter().enumerate() {
        marker_writer::write_dqt(&mut output, i as u8, qt);
    }

    let components: Vec<(u8, u8, u8, u8)> = coeffs
        .components
        .iter()
        .map(|c| {
            (
                c.component_id,
                c.h_sampling,
                c.v_sampling,
                c.quant_table_index,
            )
        })
        .collect();
    marker_writer::write_sof10_with_precision(
        &mut output,
        coeffs.width,
        coeffs.height,
        &components,
        coeffs.effective_precision(),
    );

    let mut arith_enc: ArithEncoder =
        ArithEncoder::new(coeffs.width as usize * coeffs.height as usize / 4);

    // Per-scan DRI tracking. Mirrors the Huffman progressive writer
    // (`write_coefficients_progressive`): row mode recomputes the
    // restart interval per scan as `rows * MCUs_per_row_of_scan` where
    // multi-component scans use the interleaved MCU grid and
    // non-interleaved AC scans use that component's width in blocks.
    // Byte mode applies `coeffs.restart_interval` uniformly.
    let per_scan_ri = |scan_ci: &[usize]| -> u16 {
        match restart_rows {
            Some(rows) => {
                let mcus_per_row: usize = if scan_ci.len() == 1 {
                    let ci: usize = scan_ci[0];
                    let comp = &coeffs.components[ci];
                    (coeffs.width as usize * comp.h_sampling as usize).div_ceil(max_h * 8)
                } else {
                    (coeffs.width as usize).div_ceil(max_h * 8)
                };
                let dri: usize = rows as usize * mcus_per_row;
                dri.min(u16::MAX as usize) as u16
            }
            None => coeffs.restart_interval,
        }
    };

    // Track the last-emitted DRI so we only re-emit the marker when the
    // value changes between scans (matches C `jcmarker.c::write_scan_header`).
    let mut saved_ri: u16 = 0;

    for scan in &scans {
        arith_enc.reset();

        let is_dc_scan: bool = scan.ss == 0 && scan.se == 0;
        let is_first: bool = scan.ah == 0;
        let scan_ri: u16 = per_scan_ri(&scan.component_indices);

        let scan_components: Vec<(u8, u8, u8)> = scan
            .component_indices
            .iter()
            .map(|&ci| {
                let tbl: u8 = coding_table_for_component(coeffs, ci) as u8;
                let dc_tbl: u8 = if is_dc_scan && is_first { tbl } else { 0 };
                let ac_tbl: u8 = if scan.se > 0 { tbl } else { 0 };
                (coeffs.components[ci].component_id, dc_tbl, ac_tbl)
            })
            .collect();

        let mut dc_in_use = [false; crate::decode::arithmetic::NUM_ARITH_TBLS];
        let mut ac_in_use = [false; crate::decode::arithmetic::NUM_ARITH_TBLS];
        if is_dc_scan && is_first {
            for &ci in &scan.component_indices {
                dc_in_use[coding_table_for_component(coeffs, ci)] = true;
            }
        }
        if scan.se > 0 {
            for &ci in &scan.component_indices {
                ac_in_use[coding_table_for_component(coeffs, ci)] = true;
            }
        }

        marker_writer::write_dac_selected(
            &mut output,
            &dc_in_use,
            &dc_params,
            &ac_in_use,
            &ac_params,
        );
        // DRI is emitted only when the per-scan restart interval changes
        // from what the previous scan installed. `saved_ri == 0` initially,
        // so the first scan with restart markers re-emits DRI.
        if scan_ri != saved_ri {
            marker_writer::write_dri(&mut output, scan_ri);
            saved_ri = scan_ri;
        }
        marker_writer::write_sos_progressive(
            &mut output,
            &scan_components,
            scan.ss,
            scan.se,
            scan.ah,
            scan.al,
        );

        let ri: u32 = scan_ri as u32;
        let mut restarts_to_go: u32 = ri;
        let mut next_restart_num: u8 = 0;

        if is_dc_scan && is_first {
            let mut prev_dc: Vec<i16> = vec![0; num_components];

            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    // Mirrors libjpeg-turbo `jcarith.c::encode_mcu_DC_first`:
                    // when the per-scan restart counter reaches zero,
                    // flush byte-aligned, push FF Dn, reset coder state
                    // and DC predictors, then continue the next group.
                    if ri > 0 && restarts_to_go == 0 {
                        arith_enc.emit_restart(next_restart_num);
                        next_restart_num = (next_restart_num + 1) & 7;
                        prev_dc.iter_mut().for_each(|v| *v = 0);
                        restarts_to_go = ri;
                    }
                    for &ci in &scan.component_indices {
                        let comp = &coeffs.components[ci];
                        let dc_tbl: usize = coding_table_for_component(coeffs, ci);

                        let h_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.h_sampling as usize
                        };
                        let v_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.v_sampling as usize
                        };
                        for v in 0..v_blocks {
                            for h in 0..h_blocks {
                                let bx: usize = mcu_x * h_blocks + h;
                                let by: usize = mcu_y * v_blocks + v;

                                let mut dummy = [0i16; 64];
                                let block: &[i16; 64] =
                                    if bx >= data_blocks_x[ci] || by >= data_blocks_y[ci] {
                                        dummy[0] = prev_dc[ci];
                                        &dummy
                                    } else {
                                        let real_block: &[i16; 64] =
                                            &comp.blocks[by * comp.blocks_x + bx];
                                        prev_dc[ci] = real_block[0];
                                        real_block
                                    };

                                arith_enc.encode_dc_first(block, ci, dc_tbl, scan.al);
                            }
                        }
                    }
                    if ri > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }
        } else if is_dc_scan {
            let mut prev_dc: Vec<i16> = vec![0; num_components];

            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if ri > 0 && restarts_to_go == 0 {
                        arith_enc.emit_restart(next_restart_num);
                        next_restart_num = (next_restart_num + 1) & 7;
                        prev_dc.iter_mut().for_each(|v| *v = 0);
                        restarts_to_go = ri;
                    }
                    for &ci in &scan.component_indices {
                        let comp = &coeffs.components[ci];

                        let h_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.h_sampling as usize
                        };
                        let v_blocks: usize = if scan.component_indices.len() == 1 {
                            1
                        } else {
                            comp.v_sampling as usize
                        };
                        for v in 0..v_blocks {
                            for h in 0..h_blocks {
                                let bx: usize = mcu_x * h_blocks + h;
                                let by: usize = mcu_y * v_blocks + v;

                                let mut dummy = [0i16; 64];
                                let block: &[i16; 64] =
                                    if bx >= data_blocks_x[ci] || by >= data_blocks_y[ci] {
                                        dummy[0] = prev_dc[ci];
                                        &dummy
                                    } else {
                                        let real_block: &[i16; 64] =
                                            &comp.blocks[by * comp.blocks_x + bx];
                                        prev_dc[ci] = real_block[0];
                                        real_block
                                    };

                                arith_enc.encode_dc_refine(block, scan.al);
                            }
                        }
                    }
                    if ri > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }
        } else if is_first {
            let ci: usize = scan.component_indices[0];
            let comp = &coeffs.components[ci];
            let ac_tbl: usize = coding_table_for_component(coeffs, ci);
            let wib: usize = data_blocks_x[ci].min(comp.blocks_x);
            let hib: usize = data_blocks_y[ci].min(comp.blocks_y);

            // Non-interleaved AC scan: each block is one MCU per the
            // JPEG spec, so the restart counter applies per block.
            for by in 0..hib {
                for bx in 0..wib {
                    if ri > 0 && restarts_to_go == 0 {
                        arith_enc.emit_restart(next_restart_num);
                        next_restart_num = (next_restart_num + 1) & 7;
                        restarts_to_go = ri;
                    }
                    let block: &[i16; 64] = &comp.blocks[by * comp.blocks_x + bx];
                    arith_enc.encode_ac_first(block, ac_tbl, scan.ss, scan.se, scan.al);
                    if ri > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }
        } else {
            let ci: usize = scan.component_indices[0];
            let comp = &coeffs.components[ci];
            let ac_tbl: usize = coding_table_for_component(coeffs, ci);
            let wib: usize = data_blocks_x[ci].min(comp.blocks_x);
            let hib: usize = data_blocks_y[ci].min(comp.blocks_y);

            for by in 0..hib {
                for bx in 0..wib {
                    if ri > 0 && restarts_to_go == 0 {
                        arith_enc.emit_restart(next_restart_num);
                        next_restart_num = (next_restart_num + 1) & 7;
                        restarts_to_go = ri;
                    }
                    let block: &[i16; 64] = &comp.blocks[by * comp.blocks_x + bx];
                    arith_enc.encode_ac_refine(block, ac_tbl, scan.ss, scan.se, scan.al, scan.ah);
                    if ri > 0 {
                        restarts_to_go -= 1;
                    }
                }
            }
        }

        arith_enc.finish();
        output.extend_from_slice(arith_enc.data());
    }

    marker_writer::write_eoi(&mut output);
    Ok(output)
}

/// Transpose a quantization table (8x8 matrix) in-place.
/// Required for dimension-swapping transforms (transpose, rot90, rot270, transverse)
/// so that each coefficient position uses the correct quantization value.
fn transpose_quant_table(qt: &mut [u16; 64]) {
    let mut transposed: [u16; 64] = [0u16; 64];
    for row in 0..8 {
        for col in 0..8 {
            transposed[col * 8 + row] = qt[row * 8 + col];
        }
    }
    *qt = transposed;
}

/// Convert a block from natural (row-major) order to zigzag order.
fn natural_to_zigzag(natural: &[i16; 64]) -> [i16; 64] {
    let mut zigzag = [0i16; 64];
    for i in 0..64 {
        zigzag[NATURAL_ORDER[i]] = natural[i];
    }
    zigzag
}

/// Convert all blocks in comp_data from natural to zigzag order.
fn convert_all_to_zigzag(comp_data: &mut [ComponentCoefficients]) {
    for comp in comp_data.iter_mut() {
        for block in &mut comp.blocks {
            *block = natural_to_zigzag(block);
        }
    }
}

// --- Internal decode helpers ---

fn decode_baseline_coefficients(
    data: &[u8],
    metadata: &JpegMetadata,
    comp_data: &mut [ComponentCoefficients],
    mcus_x: usize,
    mcus_y: usize,
) -> Result<()> {
    use crate::decode::bitstream::BitReader;
    use crate::decode::entropy;

    let frame = &metadata.frame;
    let scan = &metadata.scan;

    // A scan containing the sole frame component is non-interleaved: one
    // entropy-coded data unit is one block, regardless of the H/V sampling
    // values retained in the SOF. Decode the actual component block grid so
    // restart boundaries and partial edge MCUs match jpeg_read_coefficients().
    if frame.components.len() == 1 {
        let scan_info = metadata.scans.first().ok_or_else(|| {
            JpegError::CorruptData("single-component JPEG has no scan data".to_string())
        })?;
        let mcu_plan = entropy::resolve_mcu_plan(
            frame,
            &scan_info.header,
            &scan_info.dc_huffman_tables,
            &scan_info.ac_huffman_tables,
        )?;
        if mcu_plan.len() != 1 {
            return Err(JpegError::CorruptData(format!(
                "single-component scan has {} entropy plans",
                mcu_plan.len()
            )));
        }

        let comp = &frame.components[0];
        let max_h = comp.horizontal_sampling as usize;
        let max_v = comp.vertical_sampling as usize;
        let blocks_x = (frame.width as usize * max_h).div_ceil(max_h * 8);
        let blocks_y = (frame.height as usize * max_v).div_ceil(max_v * 8);
        let stride = comp_data[0].blocks_x;
        let plan = &mcu_plan[0];
        let entropy_data = &data[scan_info.data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);
        let mut mcu_decoder = entropy::McuDecoder::new(1);
        let mut coeffs = [0i16; 64];
        let mut mcu_count: u32 = 0;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if scan_info.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count.is_multiple_of(scan_info.restart_interval as u32)
                {
                    bit_reader.reset();
                    mcu_decoder.reset();
                }
                mcu_decoder.decode_block(
                    &mut bit_reader,
                    plan.comp_idx,
                    plan.dc_table,
                    plan.ac_table,
                    &mut coeffs,
                )?;
                comp_data[0].blocks[by * stride + bx] = coeffs;
                mcu_count += 1;
            }
        }
        return Ok(());
    }

    let mcu_plan = entropy::resolve_mcu_plan(
        frame,
        scan,
        &metadata.dc_huffman_tables,
        &metadata.ac_huffman_tables,
    )?;

    // Baseline single-scan JPEG must reference every frame component in
    // its SOS — otherwise the per-component indexing below would walk off
    // the plan. Malformed streams surface a clean CorruptData error
    // rather than panicking on `mcu_plan[comp_idx]`.
    if mcu_plan.len() != comp_data.len() {
        return Err(crate::common::error::JpegError::CorruptData(format!(
            "baseline SOS covers {} components but frame has {}",
            mcu_plan.len(),
            comp_data.len()
        )));
    }

    let entropy_data = &data[metadata.entropy_data_offset..];
    let mut bit_reader = BitReader::new(entropy_data);
    let mut mcu_decoder = entropy::McuDecoder::new(frame.components.len());
    let mut mcu_count: u32 = 0;
    let mut coeffs = [0i16; 64];

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if metadata.restart_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(metadata.restart_interval as u32)
            {
                bit_reader.reset();
                mcu_decoder.reset();
            }

            for (comp_idx, comp) in comp_data.iter_mut().enumerate() {
                let plan = &mcu_plan[comp_idx];

                let h_blocks = frame.components[comp_idx].horizontal_sampling as usize;
                let v_blocks = frame.components[comp_idx].vertical_sampling as usize;

                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        mcu_decoder.decode_block(
                            &mut bit_reader,
                            plan.comp_idx,
                            plan.dc_table,
                            plan.ac_table,
                            &mut coeffs,
                        )?;

                        let bx = mcu_x * h_blocks + h;
                        let by = mcu_y * v_blocks + v;
                        let block_idx = by * comp.blocks_x + bx;
                        comp.blocks[block_idx] = coeffs;
                    }
                }
            }

            mcu_count += 1;
        }
    }

    Ok(())
}

fn decode_arithmetic_coefficients(
    data: &[u8],
    metadata: &JpegMetadata,
    comp_data: &mut [ComponentCoefficients],
    mcus_x: usize,
    mcus_y: usize,
) -> Result<()> {
    use crate::decode::arithmetic::ArithDecoder;

    let frame = &metadata.frame;
    let scan = &metadata.scan;

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

    let entropy_data = &data[metadata.entropy_data_offset..];
    let mut arith = ArithDecoder::new(entropy_data, 0);

    for i in 0..crate::decode::arithmetic::NUM_ARITH_TBLS {
        let (l, u) = metadata.arith_dc_params[i];
        arith.set_dc_conditioning(i, l, u);
        arith.set_ac_conditioning(i, metadata.arith_ac_params[i]);
    }

    // Pre-extract layout info to avoid borrow conflicts
    let layouts: Vec<(usize, usize, usize)> = comp_data
        .iter()
        .map(|c| (c.h_sampling as usize, c.v_sampling as usize, c.blocks_x))
        .collect();
    let mut coeffs: [i16; 64];

    let restart_interval = metadata.restart_interval as u32;
    let mut restarts_to_go = restart_interval;

    if frame.components.len() == 1 && scan_comps.len() == 1 {
        let (comp_idx, dc_tbl, ac_tbl) = scan_comps[0];
        let blocks_x = (frame.width as usize).div_ceil(8);
        let blocks_y = (frame.height as usize).div_ceil(8);
        let stride = comp_data[comp_idx].blocks_x;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                if restart_interval > 0 && restarts_to_go == 0 {
                    arith.process_restart();
                    restarts_to_go = restart_interval;
                }
                coeffs = [0i16; 64];
                arith.decode_dc_sequential(&mut coeffs, comp_idx, dc_tbl)?;
                arith.decode_ac_sequential(&mut coeffs, ac_tbl)?;
                comp_data[comp_idx].blocks[by * stride + bx] = coeffs;
                if restart_interval > 0 {
                    restarts_to_go -= 1;
                }
            }
        }
        return Ok(());
    }

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if restart_interval > 0 && restarts_to_go == 0 {
                arith.process_restart();
                restarts_to_go = restart_interval;
            }
            for &(comp_idx, dc_tbl, ac_tbl) in &scan_comps {
                let (h_blocks, v_blocks, blocks_x) = layouts[comp_idx];

                for v in 0..v_blocks {
                    for h in 0..h_blocks {
                        coeffs = [0i16; 64];
                        arith.decode_dc_sequential(&mut coeffs, comp_idx, dc_tbl)?;
                        arith.decode_ac_sequential(&mut coeffs, ac_tbl)?;

                        let bx = mcu_x * h_blocks + h;
                        let by = mcu_y * v_blocks + v;
                        let block_idx = by * blocks_x + bx;
                        comp_data[comp_idx].blocks[block_idx] = coeffs;
                    }
                }
            }
            if restart_interval > 0 {
                restarts_to_go -= 1;
            }
        }
    }

    Ok(())
}

/// Decode SOF10 (arithmetic progressive) coefficients.
fn decode_arithmetic_progressive_coefficients(
    data: &[u8],
    metadata: &JpegMetadata,
    comp_data: &mut [ComponentCoefficients],
    _mcus_x: usize,
    _mcus_y: usize,
) -> Result<()> {
    use crate::decode::arithmetic::ArithDecoder;

    let frame = &metadata.frame;

    for scan_info in &metadata.scans {
        let scan = &scan_info.header;
        let ss: u8 = scan.spec_start;
        let se: u8 = scan.spec_end;
        let ah: u8 = scan.succ_high;
        let al: u8 = scan.succ_low;
        let is_dc: bool = ss == 0 && se == 0;

        let entropy_data: &[u8] = &data[scan_info.data_offset..];
        let mut arith: ArithDecoder<'_> = ArithDecoder::new(entropy_data, 0);

        // Set arithmetic conditioning parameters (16 slots per NUM_ARITH_TBLS).
        for i in 0..crate::decode::arithmetic::NUM_ARITH_TBLS {
            let (l, u) = metadata.arith_dc_params[i];
            arith.set_dc_conditioning(i, l, u);
            arith.set_ac_conditioning(i, metadata.arith_ac_params[i]);
        }

        let scan_comp_indices: Vec<usize> = scan
            .components
            .iter()
            .map(|sc| {
                frame
                    .components
                    .iter()
                    .position(|fc| fc.id == sc.component_id)
                    .unwrap_or(0)
            })
            .collect();

        if scan.components.len() > 1 {
            // Interleaved DC scan — iterate MCU by MCU
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
            let mcu_w: usize = max_h * 8;
            let mcu_h: usize = max_v * 8;
            let mcus_x: usize = (frame.width as usize).div_ceil(mcu_w);
            let mcus_y: usize = (frame.height as usize).div_ceil(mcu_h);

            for _mcu_y in 0..mcus_y {
                for _mcu_x in 0..mcus_x {
                    for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                        let h_samp: usize = frame.components[comp_idx].horizontal_sampling as usize;
                        let v_samp: usize = frame.components[comp_idx].vertical_sampling as usize;
                        let blocks_x: usize = comp_data[comp_idx].blocks_x;
                        let dc_tbl: usize = scan.components[si].dc_table_index as usize;

                        for v in 0..v_samp {
                            for h in 0..h_samp {
                                let bx: usize = _mcu_x * h_samp + h;
                                let by: usize = _mcu_y * v_samp + v;
                                let block_idx: usize = by * blocks_x + bx;
                                let block: &mut [i16; 64] =
                                    &mut comp_data[comp_idx].blocks[block_idx];

                                if is_dc && ah == 0 {
                                    arith
                                        .decode_dc_first_progressive(block, comp_idx, dc_tbl, al)?;
                                } else if is_dc {
                                    arith.decode_dc_refine_progressive(block, al)?;
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // Non-interleaved scan (single component)
            let comp_idx: usize = scan_comp_indices[0];
            // Use actual data block counts for non-interleaved scans, not MCU-padded.
            // The JPEG bitstream contains exactly width_in_blocks * height_in_blocks
            // data units per component (C libjpeg-turbo jdinput.c:119-124,175-176).
            let arith_max_h: usize = frame
                .components
                .iter()
                .map(|c| c.horizontal_sampling as usize)
                .max()
                .unwrap_or(1);
            let arith_max_v: usize = frame
                .components
                .iter()
                .map(|c| c.vertical_sampling as usize)
                .max()
                .unwrap_or(1);
            let h_samp: usize = comp_data[comp_idx].h_sampling as usize;
            let v_samp: usize = comp_data[comp_idx].v_sampling as usize;
            let comp_blocks_x: usize = (frame.width as usize * h_samp).div_ceil(arith_max_h * 8);
            let comp_blocks_y: usize = (frame.height as usize * v_samp).div_ceil(arith_max_v * 8);
            let stride_x: usize = comp_data[comp_idx].blocks_x;
            let dc_tbl: usize = scan.components[0].dc_table_index as usize;
            let ac_tbl: usize = scan.components[0].ac_table_index as usize;

            for by in 0..comp_blocks_y {
                for bx in 0..comp_blocks_x {
                    let block_idx: usize = by * stride_x + bx;
                    let block: &mut [i16; 64] = &mut comp_data[comp_idx].blocks[block_idx];

                    if is_dc {
                        if ah == 0 {
                            arith.decode_dc_first_progressive(block, comp_idx, dc_tbl, al)?;
                        } else {
                            arith.decode_dc_refine_progressive(block, al)?;
                        }
                    } else if ah == 0 {
                        arith.decode_ac_first_progressive(block, ac_tbl, ss, se, al)?;
                    } else {
                        arith.decode_ac_refine_progressive(block, ac_tbl, ss, se, al)?;
                    }
                }
            }
        }
    }

    Ok(())
}

fn decode_progressive_coefficients(
    data: &[u8],
    metadata: &JpegMetadata,
    comp_data: &mut [ComponentCoefficients],
    mcus_x: usize,
    mcus_y: usize,
) -> Result<()> {
    use crate::decode::bitstream::BitReader;
    use crate::decode::progressive;

    let frame = &metadata.frame;
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

    // Per-block highest nonzero AC zigzag index (issue #352: bounds the
    // refinement EOB-run walk to the block's spectral extent).
    let mut ac_max_k: Vec<Vec<u8>> = comp_data
        .iter()
        .map(|cd| vec![0u8; cd.blocks.len()])
        .collect();

    for scan_info in &metadata.scans {
        let scan = &scan_info.header;
        let ss = scan.spec_start;
        let se = scan.spec_end;
        let ah = scan.succ_high;
        let al = scan.succ_low;
        let is_dc = ss == 0 && se == 0;

        let entropy_data = &data[scan_info.data_offset..];
        let mut bit_reader = BitReader::new(entropy_data);

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
            // Interleaved scan (DC only)
            let mut dc_preds = [0i16; 4];
            let mut mcu_count: u32 = 0;

            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if scan_info.restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(scan_info.restart_interval as u32)
                    {
                        bit_reader.reset();
                        dc_preds = [0i16; 4];
                    }

                    for (si, &comp_idx) in scan_comp_indices.iter().enumerate() {
                        let h_samp = comp_data[comp_idx].h_sampling as usize;
                        let v_samp = comp_data[comp_idx].v_sampling as usize;
                        let blocks_x = comp_data[comp_idx].blocks_x;
                        let scan_comp = &scan.components[si];

                        let dc_table = scan_info.dc_huffman_tables
                            [scan_comp.dc_table_index as usize]
                            .as_ref()
                            .ok_or_else(|| {
                                JpegError::CorruptData(format!(
                                    "missing DC table {}",
                                    scan_comp.dc_table_index
                                ))
                            })?;

                        for v in 0..v_samp {
                            for h in 0..h_samp {
                                let bx = mcu_x * h_samp + h;
                                let by = mcu_y * v_samp + v;
                                let block_idx = by * blocks_x + bx;
                                let coeffs = &mut comp_data[comp_idx].blocks[block_idx];

                                if is_dc {
                                    if ah == 0 {
                                        progressive::decode_dc_first(
                                            &mut bit_reader,
                                            dc_table,
                                            &mut dc_preds[comp_idx],
                                            coeffs,
                                            al,
                                        )?;
                                    } else {
                                        progressive::decode_dc_refine(&mut bit_reader, coeffs, al)?;
                                    }
                                }
                            }
                        }
                    }

                    mcu_count += 1;
                }
            }
        } else {
            // Non-interleaved scan
            let comp_idx = scan_comp_indices[0];
            let scan_comp = &scan.components[0];
            // Use actual data block counts for non-interleaved scans, not MCU-padded.
            // The JPEG bitstream contains exactly width_in_blocks * height_in_blocks
            // data units per component (C libjpeg-turbo jdinput.c:119-124,175-176).
            let h_samp: usize = comp_data[comp_idx].h_sampling as usize;
            let v_samp: usize = comp_data[comp_idx].v_sampling as usize;
            let comp_blocks_x: usize = (frame.width as usize * h_samp).div_ceil(max_h * 8);
            let comp_blocks_y: usize = (frame.height as usize * v_samp).div_ceil(max_v * 8);
            let stride_x: usize = comp_data[comp_idx].blocks_x;
            let mut dc_pred = 0i16;
            let mut eob_run = 0u16;
            let mut mcu_count: u32 = 0;

            let dc_table = if is_dc {
                Some(
                    scan_info.dc_huffman_tables[scan_comp.dc_table_index as usize]
                        .as_ref()
                        .ok_or_else(|| {
                            JpegError::CorruptData(format!(
                                "missing DC table {}",
                                scan_comp.dc_table_index
                            ))
                        })?,
                )
            } else {
                None
            };
            let ac_table = if !is_dc || se > 0 {
                Some(
                    scan_info.ac_huffman_tables[scan_comp.ac_table_index as usize]
                        .as_ref()
                        .ok_or_else(|| {
                            JpegError::CorruptData(format!(
                                "missing AC table {}",
                                scan_comp.ac_table_index
                            ))
                        })?,
                )
            } else {
                None
            };

            let restart_interval = scan_info.restart_interval as u32;

            for by in 0..comp_blocks_y {
                for bx in 0..comp_blocks_x {
                    if restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(restart_interval)
                    {
                        bit_reader.reset();
                        dc_pred = 0;
                        eob_run = 0;
                    }

                    let block_idx = by * stride_x + bx;
                    let coeffs = &mut comp_data[comp_idx].blocks[block_idx];

                    if is_dc {
                        if ah == 0 {
                            progressive::decode_dc_first(
                                &mut bit_reader,
                                dc_table.unwrap(),
                                &mut dc_pred,
                                coeffs,
                                al,
                            )?;
                        } else {
                            progressive::decode_dc_refine(&mut bit_reader, coeffs, al)?;
                        }
                    } else if ah == 0 {
                        progressive::decode_ac_first_tracked(
                            &mut bit_reader,
                            ac_table.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                            &mut ac_max_k[comp_idx][block_idx],
                        )?;
                    } else {
                        progressive::decode_ac_refine_tracked(
                            &mut bit_reader,
                            ac_table.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                            &mut ac_max_k[comp_idx][block_idx],
                        )?;
                    }

                    mcu_count += 1;
                }
            }
        }
    }

    Ok(())
}
