use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::types::*;
use crate::decode::bitstream::BitReader;
use crate::decode::huffman;

/// Decodes MCUs from entropy-coded data. Manages DC prediction per component.
pub struct McuDecoder {
    dc_pred: Vec<i16>,
}

impl McuDecoder {
    pub fn new(num_components: usize) -> Self {
        Self {
            dc_pred: vec![0; num_components],
        }
    }

    pub fn dc_prediction(&self, component_index: usize) -> i16 {
        self.dc_pred[component_index]
    }

    pub fn reset(&mut self) {
        for pred in self.dc_pred.iter_mut() {
            *pred = 0;
        }
    }

    /// Decode one 8x8 block of DCT coefficients (in zigzag order).
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
        self.dc_pred[component_index] += dc_diff;
        coeffs[0] = self.dc_pred[component_index];

        huffman::decode_ac_coefficients(reader, ac_table, coeffs)?;

        Ok(())
    }

    /// Decode a complete MCU.
    pub fn decode_mcu(
        &mut self,
        reader: &mut BitReader,
        frame: &FrameHeader,
        scan: &ScanHeader,
        dc_tables: &[Option<HuffmanTable>; 4],
        ac_tables: &[Option<HuffmanTable>; 4],
        blocks: &mut Vec<[i16; 64]>,
    ) -> Result<()> {
        blocks.clear();

        for scan_comp in &scan.components {
            let (comp_idx, comp) = frame
                .components
                .iter()
                .enumerate()
                .find(|(_, c)| c.id == scan_comp.component_id)
                .ok_or(JpegError::CorruptData(format!(
                    "scan references unknown component id {}",
                    scan_comp.component_id
                )))?;

            let dc_table = dc_tables[scan_comp.dc_table_index as usize]
                .as_ref()
                .ok_or(JpegError::CorruptData(format!(
                    "missing DC Huffman table {}",
                    scan_comp.dc_table_index
                )))?;

            let ac_table = ac_tables[scan_comp.ac_table_index as usize]
                .as_ref()
                .ok_or(JpegError::CorruptData(format!(
                    "missing AC Huffman table {}",
                    scan_comp.ac_table_index
                )))?;

            let num_blocks =
                (comp.horizontal_sampling as usize) * (comp.vertical_sampling as usize);

            for _ in 0..num_blocks {
                let mut coeffs = [0i16; 64];
                self.decode_block(reader, comp_idx, dc_table, ac_table, &mut coeffs)?;
                blocks.push(coeffs);
            }
        }

        Ok(())
    }
}
