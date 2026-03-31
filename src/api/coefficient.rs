/// Coefficient-level JPEG access for lossless transforms.
///
/// Provides read_coefficients() / write_coefficients() / transform() API
/// similar to libjpeg-turbo's jpegtran workflow.
use crate::common::error::{JpegError, Result};
use crate::common::quant_table::NATURAL_ORDER;
use crate::common::types::{MarkerSaveConfig, SavedMarker};
use crate::decode::marker::{JpegMetadata, MarkerReader};
use crate::encode::huffman_encode::{build_huff_table, BitWriter, HuffmanEncoder};
use crate::encode::marker_writer;
use crate::encode::pipeline as encoder_pipeline;
use crate::encode::tables;
use crate::transform::spatial;
use crate::transform::{TransformOp, TransformOptions};

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
}

/// Complete coefficient representation of a JPEG image.
#[derive(Debug, Clone)]
pub struct JpegCoefficients {
    /// Image width in pixels.
    pub width: u16,
    /// Image height in pixels.
    pub height: u16,
    /// Per-component coefficient data.
    pub components: Vec<ComponentCoefficients>,
    /// Quantization tables (up to 4, in zigzag order).
    pub quant_tables: Vec<[u16; 64]>,
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

    // Collect quant tables in natural (row-major) order for write_dqt compatibility
    let quant_tables: Vec<[u16; 64]> = metadata
        .quant_tables
        .iter()
        .filter_map(|qt| qt.as_ref().map(|q| q.values))
        .collect();

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
                quant_table_index: comp.quant_table_index,
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

    Ok(JpegCoefficients {
        width: frame.width,
        height: frame.height,
        components: comp_data,
        quant_tables,
    })
}

/// Write DCT coefficients to a JPEG byte stream.
///
/// Encodes quantized DCT coefficients using Huffman coding,
/// producing a valid baseline JPEG file.
pub fn write_coefficients(coeffs: &JpegCoefficients) -> Result<Vec<u8>> {
    let num_components = coeffs.components.len();
    let is_grayscale = num_components == 1;

    // Build Huffman tables
    let dc_luma_table = build_huff_table(&tables::DC_LUMINANCE_BITS, &tables::DC_LUMINANCE_VALUES);
    let ac_luma_table = build_huff_table(&tables::AC_LUMINANCE_BITS, &tables::AC_LUMINANCE_VALUES);
    let dc_chroma_table =
        build_huff_table(&tables::DC_CHROMINANCE_BITS, &tables::DC_CHROMINANCE_VALUES);
    let ac_chroma_table =
        build_huff_table(&tables::AC_CHROMINANCE_BITS, &tables::AC_CHROMINANCE_VALUES);

    let _max_h = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let _max_v = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let mcus_x = coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let mcus_y = coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    // Entropy encode
    let mut bit_writer = BitWriter::new(coeffs.width as usize * coeffs.height as usize);
    let mut prev_dc = vec![0i16; num_components];

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for (ci, comp) in coeffs.components.iter().enumerate() {
                let dc_table = if ci == 0 {
                    &dc_luma_table
                } else {
                    &dc_chroma_table
                };
                let ac_table = if ci == 0 {
                    &ac_luma_table
                } else {
                    &ac_chroma_table
                };

                for v in 0..comp.v_sampling as usize {
                    for h in 0..comp.h_sampling as usize {
                        let bx = mcu_x * comp.h_sampling as usize + h;
                        let by = mcu_y * comp.v_sampling as usize + v;
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
    }

    bit_writer.flush();

    // Assemble output
    let mut output = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables
    for (i, qt) in coeffs.quant_tables.iter().enumerate() {
        marker_writer::write_dqt(&mut output, i as u8, qt);
    }

    // Frame header (SOF0)
    let components: Vec<(u8, u8, u8, u8)> = coeffs
        .components
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                (i + 1) as u8,
                c.h_sampling,
                c.v_sampling,
                c.quant_table_index,
            )
        })
        .collect();
    marker_writer::write_sof0(&mut output, coeffs.width, coeffs.height, &components);

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
    let scan_components: Vec<(u8, u8, u8)> = coeffs
        .components
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let tbl = if i == 0 { 0u8 } else { 1u8 };
            ((i + 1) as u8, tbl, tbl)
        })
        .collect();
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());
    marker_writer::write_eoi(&mut output);

    Ok(output)
}

/// Apply a lossless transform to a JPEG image.
///
/// Reads DCT coefficients, applies the spatial transform, adjusts
/// dimensions, and writes back to JPEG. No quality loss.
pub fn transform_jpeg(data: &[u8], op: TransformOp) -> Result<Vec<u8>> {
    let mut coeffs = read_coefficients(data)?;

    if op == TransformOp::None {
        return write_coefficients(&coeffs);
    }

    let transform_fn: fn(&[i16; 64], &mut [i16; 64]) = match op {
        TransformOp::None => spatial::do_nothing,
        TransformOp::HFlip => spatial::do_flip_h,
        TransformOp::VFlip => spatial::do_flip_v,
        TransformOp::Transpose => spatial::do_transpose,
        TransformOp::Transverse => spatial::do_transverse,
        TransformOp::Rot90 => spatial::do_rot_90,
        TransformOp::Rot180 => spatial::do_rot_180,
        TransformOp::Rot270 => spatial::do_rot_270,
    };

    let swaps_dims = matches!(
        op,
        TransformOp::Transpose | TransformOp::Transverse | TransformOp::Rot90 | TransformOp::Rot270
    );

    // Spatial transforms operate on natural (row-major) order coefficients.
    // Blocks are stored in zigzag order, so convert before/after transform.
    convert_all_to_natural(&mut coeffs.components);

    // Transform each component
    for comp in &mut coeffs.components {
        let old_bx = comp.blocks_x;
        let old_by = comp.blocks_y;
        let mut new_blocks = vec![[0i16; 64]; old_bx * old_by];

        if matches!(op, TransformOp::Transpose) {
            // Transpose: swap block (bx, by) → (by, bx)
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx: usize = by * old_bx + bx;
                    let dst_idx: usize = bx * old_by + by;
                    transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                }
            }
            comp.blocks_x = old_by;
            comp.blocks_y = old_bx;
        } else if matches!(op, TransformOp::Rot90) {
            // Rot90 CW = transpose grid + reverse columns
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx: usize = by * old_bx + bx;
                    let dst_idx: usize = bx * old_by + (old_by - 1 - by);
                    transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                }
            }
            comp.blocks_x = old_by;
            comp.blocks_y = old_bx;
        } else if matches!(op, TransformOp::Rot270) {
            // Rot270 CW = transpose grid + reverse rows
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx: usize = by * old_bx + bx;
                    let dst_idx: usize = (old_bx - 1 - bx) * old_by + by;
                    transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                }
            }
            comp.blocks_x = old_by;
            comp.blocks_y = old_bx;
        } else if matches!(op, TransformOp::Transverse) {
            // Transverse = transpose grid + reverse both rows and columns
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx: usize = by * old_bx + bx;
                    let dst_idx: usize = (old_bx - 1 - bx) * old_by + (old_by - 1 - by);
                    transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                }
            }
            comp.blocks_x = old_by;
            comp.blocks_y = old_bx;
        } else if matches!(op, TransformOp::HFlip) {
            // Horizontal flip: reverse block columns
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx = by * old_bx + bx;
                    let dst_idx = by * old_bx + (old_bx - 1 - bx);
                    transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                }
            }
        } else if matches!(op, TransformOp::VFlip) {
            // Vertical flip: reverse block rows
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx = by * old_bx + bx;
                    let dst_idx = (old_by - 1 - by) * old_bx + bx;
                    transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                }
            }
        } else if matches!(op, TransformOp::Rot180) {
            // Rotate 180: reverse both rows and columns
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx = by * old_bx + bx;
                    let dst_idx = (old_by - 1 - by) * old_bx + (old_bx - 1 - bx);
                    transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                }
            }
        } else {
            // Per-block only transform (shouldn't reach here for None)
            for (i, new_block) in new_blocks.iter_mut().enumerate() {
                transform_fn(&comp.blocks[i], new_block);
            }
        }

        comp.blocks = new_blocks;
    }

    // Convert back to zigzag order for encoder.
    convert_all_to_zigzag(&mut coeffs.components);

    // Swap dimensions and transpose quant tables for dimension-swapping ops.
    // When DCT blocks are transposed, the quant table must also be transposed
    // so that each coefficient position uses the correct quantization value.
    if swaps_dims {
        std::mem::swap(&mut coeffs.width, &mut coeffs.height);
        for comp in &mut coeffs.components {
            std::mem::swap(&mut comp.h_sampling, &mut comp.v_sampling);
        }
        for qt in &mut coeffs.quant_tables {
            transpose_quant_table(qt);
        }
    }

    write_coefficients(&coeffs)
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

        let trimmed_w: usize = if trim_width {
            (coeffs.width as usize / imcu_w) * imcu_w
        } else {
            coeffs.width as usize
        };
        let trimmed_h: usize = if trim_height {
            (coeffs.height as usize / imcu_h) * imcu_h
        } else {
            coeffs.height as usize
        };

        if trimmed_w == 0 || trimmed_h == 0 {
            return Err(JpegError::CorruptData(
                "trim would remove all image data".to_string(),
            ));
        }

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

    // CROP: crop coefficient arrays to the specified region.
    // Matches C jpegtran semantics: X/Y are rounded DOWN to iMCU boundaries,
    // and output dimensions are extended to fully cover the requested region.
    if let Some(crop) = &options.crop {
        let imcu_w: usize = max_h as usize * 8;
        let imcu_h: usize = max_v as usize * 8;
        let remainder_x: usize = crop.x % imcu_w;
        let remainder_y: usize = crop.y % imcu_h;

        // Block-level offsets (rounded down to iMCU boundary)
        let crop_x_blocks: usize = crop.x / imcu_w * max_h as usize;
        let crop_y_blocks: usize = crop.y / imcu_h * max_v as usize;

        // Extend output size to cover the full requested region from the
        // MCU-aligned start position.
        let out_w: usize =
            (crop.width + remainder_x).min(coeffs.width as usize - (crop.x - remainder_x));
        let out_h: usize =
            (crop.height + remainder_y).min(coeffs.height as usize - (crop.y - remainder_y));
        let crop_w_blocks: usize = out_w.div_ceil(8);
        let crop_h_blocks: usize = out_h.div_ceil(8);

        coeffs.width = out_w as u16;
        coeffs.height = out_h as u16;

        for comp in &mut coeffs.components {
            let comp_crop_x: usize = crop_x_blocks * comp.h_sampling as usize / max_h;
            let comp_crop_y: usize = crop_y_blocks * comp.v_sampling as usize / max_v;
            let comp_crop_w: usize = crop_w_blocks * comp.h_sampling as usize / max_h;
            let comp_crop_h: usize = crop_h_blocks * comp.v_sampling as usize / max_v;

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

    // GRAYSCALE: drop all non-Y components.
    if options.grayscale && coeffs.components.len() > 1 {
        coeffs.components.truncate(1);
        // Normalize sampling factors to 1x1 for single-component.
        coeffs.components[0].h_sampling = 1;
        coeffs.components[0].v_sampling = 1;
        // Keep only the first quant table.
        if coeffs.quant_tables.len() > 1 {
            coeffs.quant_tables.truncate(1);
        }
        coeffs.components[0].quant_table_index = 0;
    }

    // Spatial transforms operate on natural (row-major) order coefficients.
    // Blocks are stored in zigzag order, so convert before/after transform.
    if op != TransformOp::None {
        convert_all_to_natural(&mut coeffs.components);
    }

    // Apply spatial transform (reuses existing logic from transform_jpeg).
    if op != TransformOp::None {
        let transform_fn: fn(&[i16; 64], &mut [i16; 64]) = match op {
            TransformOp::None => spatial::do_nothing,
            TransformOp::HFlip => spatial::do_flip_h,
            TransformOp::VFlip => spatial::do_flip_v,
            TransformOp::Transpose => spatial::do_transpose,
            TransformOp::Transverse => spatial::do_transverse,
            TransformOp::Rot90 => spatial::do_rot_90,
            TransformOp::Rot180 => spatial::do_rot_180,
            TransformOp::Rot270 => spatial::do_rot_270,
        };

        for comp in &mut coeffs.components {
            let old_bx: usize = comp.blocks_x;
            let old_by: usize = comp.blocks_y;
            let mut new_blocks: Vec<[i16; 64]> = vec![[0i16; 64]; old_bx * old_by];

            if matches!(op, TransformOp::Transpose) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = bx * old_by + by;
                        transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::Rot90) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = bx * old_by + (old_by - 1 - by);
                        transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::Rot270) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = (old_bx - 1 - bx) * old_by + by;
                        transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::Transverse) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = (old_bx - 1 - bx) * old_by + (old_by - 1 - by);
                        transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
                comp.blocks_x = old_by;
                comp.blocks_y = old_bx;
            } else if matches!(op, TransformOp::HFlip) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = by * old_bx + (old_bx - 1 - bx);
                        transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
            } else if matches!(op, TransformOp::VFlip) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = (old_by - 1 - by) * old_bx + bx;
                        transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
            } else if matches!(op, TransformOp::Rot180) {
                for by in 0..old_by {
                    for bx in 0..old_bx {
                        let src_idx: usize = by * old_bx + bx;
                        let dst_idx: usize = (old_by - 1 - by) * old_bx + (old_bx - 1 - bx);
                        transform_fn(&comp.blocks[src_idx], &mut new_blocks[dst_idx]);
                    }
                }
            } else {
                for (i, new_block) in new_blocks.iter_mut().enumerate() {
                    transform_fn(&comp.blocks[i], new_block);
                }
            }

            comp.blocks = new_blocks;
        }

        if swaps_dims {
            std::mem::swap(&mut coeffs.width, &mut coeffs.height);
            for comp in &mut coeffs.components {
                std::mem::swap(&mut comp.h_sampling, &mut comp.v_sampling);
            }
            for qt in &mut coeffs.quant_tables {
                transpose_quant_table(qt);
            }
        }

        // Convert back to zigzag order for encoder.
        convert_all_to_zigzag(&mut coeffs.components);
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

    // Write output with the appropriate encoding.
    let output: Vec<u8> = if options.optimize {
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

/// Write DCT coefficients with optimized Huffman tables (2-pass encoding).
///
/// Pass 1 gathers symbol frequencies from the coefficient data, then
/// generates optimal Huffman tables. Pass 2 encodes with those tables.
fn write_coefficients_optimized(coeffs: &JpegCoefficients) -> Result<Vec<u8>> {
    use crate::encode::huff_opt;

    let num_components: usize = coeffs.components.len();
    let is_grayscale: bool = num_components == 1;

    let _max_h: usize = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let _max_v: usize = coeffs
        .components
        .iter()
        .map(|c| c.v_sampling as usize)
        .max()
        .unwrap_or(1);
    let mcus_x: usize = coeffs.components[0].blocks_x / coeffs.components[0].h_sampling as usize;
    let mcus_y: usize = coeffs.components[0].blocks_y / coeffs.components[0].v_sampling as usize;

    // === Pass 1: gather symbol frequencies ===
    let mut dc_luma_freq = [0u32; 257];
    let mut dc_chroma_freq = [0u32; 257];
    let mut ac_luma_freq = [0u32; 257];
    let mut ac_chroma_freq = [0u32; 257];

    let mut prev_dc: Vec<i16> = vec![0i16; num_components];

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for (ci, comp) in coeffs.components.iter().enumerate() {
                let dc_freq: &mut [u32; 257] = if ci == 0 {
                    &mut dc_luma_freq
                } else {
                    &mut dc_chroma_freq
                };
                let ac_freq: &mut [u32; 257] = if ci == 0 {
                    &mut ac_luma_freq
                } else {
                    &mut ac_chroma_freq
                };

                for v in 0..comp.v_sampling as usize {
                    for h in 0..comp.h_sampling as usize {
                        let bx: usize = mcu_x * comp.h_sampling as usize + h;
                        let by: usize = mcu_y * comp.v_sampling as usize + v;
                        let block_idx: usize = by * comp.blocks_x + bx;
                        let block: &[i16; 64] = &comp.blocks[block_idx];

                        let diff: i16 = block[0] - prev_dc[ci];
                        prev_dc[ci] = block[0];
                        huff_opt::gather_dc_symbol(diff, dc_freq);
                        huff_opt::gather_ac_symbols(block, ac_freq);
                    }
                }
            }
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

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            for (ci, comp) in coeffs.components.iter().enumerate() {
                let dc_table = if ci == 0 {
                    &dc_luma_table
                } else {
                    &dc_chroma_table
                };
                let ac_table = if ci == 0 {
                    &ac_luma_table
                } else {
                    &ac_chroma_table
                };

                for v in 0..comp.v_sampling as usize {
                    for h in 0..comp.h_sampling as usize {
                        let bx: usize = mcu_x * comp.h_sampling as usize + h;
                        let by: usize = mcu_y * comp.v_sampling as usize + v;
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
    }

    bit_writer.flush();

    // === Assemble output ===
    let mut output: Vec<u8> = Vec::with_capacity(bit_writer.data().len() + 1024);

    marker_writer::write_soi(&mut output);
    marker_writer::write_app0_jfif(&mut output);

    // Quantization tables.
    for (i, qt) in coeffs.quant_tables.iter().enumerate() {
        marker_writer::write_dqt(&mut output, i as u8, qt);
    }

    // Frame header (SOF0).
    let components: Vec<(u8, u8, u8, u8)> = coeffs
        .components
        .iter()
        .enumerate()
        .map(|(i, c)| {
            (
                (i + 1) as u8,
                c.h_sampling,
                c.v_sampling,
                c.quant_table_index,
            )
        })
        .collect();
    marker_writer::write_sof0(&mut output, coeffs.width, coeffs.height, &components);

    // Optimized Huffman tables.
    marker_writer::write_dht(&mut output, 0, 0, &dc_luma_bits, &dc_luma_values);
    marker_writer::write_dht(&mut output, 1, 0, &ac_luma_bits, &ac_luma_values);
    if !is_grayscale {
        marker_writer::write_dht(&mut output, 0, 1, &dc_chroma_bits, &dc_chroma_values);
        marker_writer::write_dht(&mut output, 1, 1, &ac_chroma_bits, &ac_chroma_values);
    }

    // Scan header.
    let scan_components: Vec<(u8, u8, u8)> = coeffs
        .components
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let tbl: u8 = if i == 0 { 0u8 } else { 1u8 };
            ((i + 1) as u8, tbl, tbl)
        })
        .collect();
    marker_writer::write_sos(&mut output, &scan_components);

    output.extend_from_slice(bit_writer.data());
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

/// Convert a block from zigzag order to natural (row-major) order.
fn zigzag_to_natural(zigzag: &[i16; 64]) -> [i16; 64] {
    let mut natural = [0i16; 64];
    for i in 0..64 {
        natural[i] = zigzag[NATURAL_ORDER[i]];
    }
    natural
}

/// Convert all blocks in comp_data from natural to zigzag order.
fn convert_all_to_zigzag(comp_data: &mut [ComponentCoefficients]) {
    for comp in comp_data.iter_mut() {
        for block in &mut comp.blocks {
            *block = natural_to_zigzag(block);
        }
    }
}

/// Convert all blocks in comp_data from zigzag to natural order.
fn convert_all_to_natural(comp_data: &mut [ComponentCoefficients]) {
    for comp in comp_data.iter_mut() {
        for block in &mut comp.blocks {
            *block = zigzag_to_natural(block);
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

    let mcu_plan = entropy::resolve_mcu_plan(
        frame,
        scan,
        &metadata.dc_huffman_tables,
        &metadata.ac_huffman_tables,
    )?;

    let entropy_data = &data[metadata.entropy_data_offset..];
    let mut bit_reader = BitReader::new(entropy_data);
    let mut mcu_decoder = entropy::McuDecoder::new(frame.components.len());
    let mut mcu_count: u16 = 0;
    let mut coeffs = [0i16; 64];

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
            if metadata.restart_interval > 0
                && mcu_count > 0
                && mcu_count.is_multiple_of(metadata.restart_interval)
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

    for i in 0..4 {
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

    for mcu_y in 0..mcus_y {
        for mcu_x in 0..mcus_x {
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

        // Set arithmetic conditioning parameters
        for i in 0..4 {
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
            let comp_blocks_x: usize = comp_data[comp_idx].blocks_x;
            let comp_blocks_y: usize = comp_data[comp_idx].blocks_y;
            let dc_tbl: usize = scan.components[0].dc_table_index as usize;
            let ac_tbl: usize = scan.components[0].ac_table_index as usize;

            for by in 0..comp_blocks_y {
                for bx in 0..comp_blocks_x {
                    let block_idx: usize = by * comp_blocks_x + bx;
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
    let _max_h = frame
        .components
        .iter()
        .map(|c| c.horizontal_sampling as usize)
        .max()
        .unwrap_or(1);
    let _max_v = frame
        .components
        .iter()
        .map(|c| c.vertical_sampling as usize)
        .max()
        .unwrap_or(1);

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
            let mut mcu_count: u16 = 0;

            for mcu_y in 0..mcus_y {
                for mcu_x in 0..mcus_x {
                    if scan_info.restart_interval > 0
                        && mcu_count > 0
                        && mcu_count.is_multiple_of(scan_info.restart_interval)
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
            let comp_blocks_x = comp_data[comp_idx].blocks_x;
            let comp_blocks_y = comp_data[comp_idx].blocks_y;
            let mut dc_pred = 0i16;
            let mut eob_run = 0u16;
            let mut mcu_count: u16 = 0;

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

            let restart_interval = scan_info.restart_interval;

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

                    let block_idx = by * comp_blocks_x + bx;
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
                        progressive::decode_ac_first(
                            &mut bit_reader,
                            ac_table.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                        )?;
                    } else {
                        progressive::decode_ac_refine(
                            &mut bit_reader,
                            ac_table.unwrap(),
                            coeffs,
                            ss,
                            se,
                            al,
                            &mut eob_run,
                        )?;
                    }

                    mcu_count += 1;
                }
            }
        }
    }

    Ok(())
}
