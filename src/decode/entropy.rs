use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::huffman;

/// Pre-resolved per-component decode plan for the MCU loop.
/// Computed once before decoding starts to avoid repeated lookups and
/// `format!()` allocations in the hot path.
pub struct McuComponentPlan<'a> {
    /// Index into the frame's component list (for DC prediction).
    pub comp_idx: usize,
    /// Number of 8×8 blocks per MCU for this component (h_sampling × v_sampling).
    pub num_blocks: usize,
    pub dc_table: &'a HuffmanTable,
    pub ac_table: &'a HuffmanTable,
}

/// Resolve scan components → Huffman tables and block counts once.
/// Returns an error if any referenced component or table is missing.
pub fn resolve_mcu_plan<'a>(
    frame: &FrameHeader,
    scan: &ScanHeader,
    dc_tables: &'a [Option<HuffmanTable>; 4],
    ac_tables: &'a [Option<HuffmanTable>; 4],
) -> Result<Vec<McuComponentPlan<'a>>> {
    let mut plan = Vec::with_capacity(scan.components.len());

    for scan_comp in &scan.components {
        let (comp_idx, comp) = frame
            .components
            .iter()
            .enumerate()
            .find(|(_, c)| c.id == scan_comp.component_id)
            .ok_or_else(|| {
                JpegError::CorruptData(format!(
                    "scan references unknown component id {}",
                    scan_comp.component_id
                ))
            })?;

        let dc_table = dc_tables[scan_comp.dc_table_index as usize]
            .as_ref()
            .ok_or_else(|| {
                JpegError::CorruptData(format!(
                    "missing DC Huffman table {}",
                    scan_comp.dc_table_index
                ))
            })?;

        let ac_table = ac_tables[scan_comp.ac_table_index as usize]
            .as_ref()
            .ok_or_else(|| {
                JpegError::CorruptData(format!(
                    "missing AC Huffman table {}",
                    scan_comp.ac_table_index
                ))
            })?;

        let num_blocks = (comp.horizontal_sampling as usize) * (comp.vertical_sampling as usize);

        plan.push(McuComponentPlan {
            comp_idx,
            num_blocks,
            dc_table,
            ac_table,
        });
    }

    Ok(plan)
}

/// Decodes MCUs from entropy-coded data. Manages DC prediction per component.
pub struct McuDecoder {
    dc_pred: [i16; 4],
}

impl McuDecoder {
    pub fn new(num_components: usize) -> Self {
        debug_assert!(num_components <= 4);
        let _ = num_components;
        Self { dc_pred: [0; 4] }
    }

    pub fn dc_prediction(&self, component_index: usize) -> i16 {
        self.dc_pred[component_index]
    }

    pub fn reset(&mut self) {
        self.dc_pred = [0; 4];
    }

    /// Decode one 8x8 block of DCT coefficients (in natural/row-major order).
    #[inline]
    pub fn decode_block(
        &mut self,
        reader: &mut BitReader,
        component_index: usize,
        dc_table: &HuffmanTable,
        ac_table: &HuffmanTable,
        coeffs: &mut [i16; 64],
    ) -> Result<()> {
        *coeffs = [0i16; 64];

        let dc_diff = huffman::decode_dc_coefficient(reader, dc_table)?;
        let dc_pred = &mut self.dc_pred[component_index];
        *dc_pred += dc_diff;
        coeffs[0] = *dc_pred;

        huffman::decode_ac_coefficients(reader, ac_table, coeffs)?;

        Ok(())
    }

    /// Decode a complete MCU using a pre-resolved plan.
    /// No allocations or lookups happen here — just decode.
    #[inline]
    pub fn decode_mcu_fast(
        &mut self,
        reader: &mut BitReader,
        plan: &[McuComponentPlan],
        blocks: &mut Vec<[i16; 64]>,
    ) -> Result<()> {
        blocks.clear();

        for comp_plan in plan {
            for _ in 0..comp_plan.num_blocks {
                let mut coeffs = [0i16; 64];
                self.decode_block(
                    reader,
                    comp_plan.comp_idx,
                    comp_plan.dc_table,
                    comp_plan.ac_table,
                    &mut coeffs,
                )?;
                blocks.push(coeffs);
            }
        }

        Ok(())
    }

    /// Decode a complete MCU (legacy API, resolves tables per call).
    pub fn decode_mcu(
        &mut self,
        reader: &mut BitReader,
        frame: &FrameHeader,
        scan: &ScanHeader,
        dc_tables: &[Option<HuffmanTable>; 4],
        ac_tables: &[Option<HuffmanTable>; 4],
        blocks: &mut Vec<[i16; 64]>,
    ) -> Result<()> {
        let plan = resolve_mcu_plan(frame, scan, dc_tables, ac_tables)?;
        self.decode_mcu_fast(reader, &plan, blocks)
    }
}
