/// Coefficient-level JPEG access for lossless transforms.
///
/// Provides read_coefficients() / write_coefficients() / transform() API
/// similar to libjpeg-turbo's jpegtran workflow.
use crate::common::error::{JpegError, Result};
use crate::common::quant_table::NATURAL_ORDER;
use crate::decode::marker::{JpegMetadata, MarkerReader};
use crate::encode::huffman_encode::{build_huff_table, BitWriter, HuffmanEncoder};
use crate::encode::marker_writer;
use crate::encode::tables;
use crate::transform::spatial;
use crate::transform::TransformOp;

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
    let mcus_x = (frame.width as usize + mcu_w - 1) / mcu_w;
    let mcus_y = (frame.height as usize + mcu_h - 1) / mcu_h;

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

    if frame.is_progressive {
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

    let max_h = coeffs
        .components
        .iter()
        .map(|c| c.h_sampling as usize)
        .max()
        .unwrap_or(1);
    let max_v = coeffs
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

    // Transform each component
    for comp in &mut coeffs.components {
        let old_bx = comp.blocks_x;
        let old_by = comp.blocks_y;
        let mut new_blocks = vec![[0i16; 64]; old_bx * old_by];

        if swaps_dims {
            // Transpose block grid + apply per-block transform
            for by in 0..old_by {
                for bx in 0..old_bx {
                    let src_idx = by * old_bx + bx;
                    let dst_idx = bx * old_by + by;
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
            for i in 0..comp.blocks.len() {
                transform_fn(&comp.blocks[i], &mut new_blocks[i]);
            }
        }

        comp.blocks = new_blocks;
    }

    // Swap dimensions if needed
    if swaps_dims {
        std::mem::swap(&mut coeffs.width, &mut coeffs.height);
        for comp in &mut coeffs.components {
            std::mem::swap(&mut comp.h_sampling, &mut comp.v_sampling);
        }
    }

    write_coefficients(&coeffs)
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
                && mcu_count % metadata.restart_interval == 0
            {
                bit_reader.reset();
                mcu_decoder.reset();
            }

            let mut plan_idx = 0;
            for (comp_idx, comp) in comp_data.iter_mut().enumerate() {
                let plan = &mcu_plan[plan_idx];
                plan_idx += 1;

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
    let mut coeffs = [0i16; 64];

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
                        && mcu_count % scan_info.restart_interval == 0
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
                    if restart_interval > 0 && mcu_count > 0 && mcu_count % restart_interval == 0 {
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
