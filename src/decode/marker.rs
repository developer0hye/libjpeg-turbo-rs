use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;

// JPEG marker codes
const SOI: u8 = 0xD8;
const EOI: u8 = 0xD9;
const SOF0: u8 = 0xC0;
const DHT: u8 = 0xC4;
const DQT: u8 = 0xDB;
const SOS: u8 = 0xDA;
const DRI: u8 = 0xDD;
const COM: u8 = 0xFE;

/// All metadata parsed from JPEG markers before the entropy-coded data.
#[derive(Debug)]
pub struct JpegMetadata {
    pub frame: FrameHeader,
    pub scan: ScanHeader,
    pub quant_tables: [Option<QuantTable>; 4],
    pub dc_huffman_tables: [Option<HuffmanTable>; 4],
    pub ac_huffman_tables: [Option<HuffmanTable>; 4],
    pub restart_interval: u16,
    /// Byte offset where entropy-coded data begins.
    pub entropy_data_offset: usize,
}

/// Reads and parses JPEG markers from a byte slice.
pub struct MarkerReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> MarkerReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    /// Parse all markers up to and including SOS.
    pub fn read_markers(&mut self) -> Result<JpegMetadata> {
        self.expect_marker(SOI)?;

        let mut frame: Option<FrameHeader> = None;
        let mut scan: Option<ScanHeader> = None;
        let mut quant_tables: [Option<QuantTable>; 4] = [None, None, None, None];
        let mut dc_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut ac_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut restart_interval: u16 = 0;

        loop {
            let marker = self.read_marker()?;
            match marker {
                SOF0 => {
                    frame = Some(self.read_sof()?);
                }
                DQT => {
                    self.read_dqt(&mut quant_tables)?;
                }
                DHT => {
                    self.read_dht(&mut dc_huffman_tables, &mut ac_huffman_tables)?;
                }
                DRI => {
                    restart_interval = self.read_dri()?;
                }
                SOS => {
                    scan = Some(self.read_sos()?);
                    break;
                }
                EOI => {
                    return Err(JpegError::CorruptData("unexpected EOI before SOS".into()));
                }
                // Skip APPn and COM markers
                m if (0xE0..=0xEF).contains(&m) || m == COM => {
                    self.skip_marker_segment()?;
                }
                // Skip other markers with length
                m if m != 0x00 && m != 0xFF => {
                    self.skip_marker_segment()?;
                }
                m => {
                    return Err(JpegError::InvalidMarker(m));
                }
            }
        }

        let frame = frame.ok_or(JpegError::CorruptData("missing SOF marker".into()))?;
        let scan = scan.ok_or(JpegError::CorruptData("missing SOS marker".into()))?;

        Ok(JpegMetadata {
            frame,
            scan,
            quant_tables,
            dc_huffman_tables,
            ac_huffman_tables,
            restart_interval,
            entropy_data_offset: self.pos,
        })
    }

    fn expect_marker(&mut self, expected: u8) -> Result<()> {
        if self.pos + 1 >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        if self.data[self.pos] != 0xFF || self.data[self.pos + 1] != expected {
            return Err(JpegError::UnexpectedMarker(
                self.data.get(self.pos + 1).copied().unwrap_or(0),
            ));
        }
        self.pos += 2;
        Ok(())
    }

    fn read_marker(&mut self) -> Result<u8> {
        while self.pos < self.data.len() && self.data[self.pos] == 0xFF {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let marker = self.data[self.pos];
        self.pos += 1;
        if marker == 0x00 {
            return Err(JpegError::InvalidMarker(0x00));
        }
        Ok(marker)
    }

    fn read_u8(&mut self) -> Result<u8> {
        if self.pos >= self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let val = self.data[self.pos];
        self.pos += 1;
        Ok(val)
    }

    fn read_u16_be(&mut self) -> Result<u16> {
        let hi = self.read_u8()? as u16;
        let lo = self.read_u8()? as u16;
        Ok((hi << 8) | lo)
    }

    fn skip_marker_segment(&mut self) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        if length < 2 {
            return Err(JpegError::CorruptData("marker segment length < 2".into()));
        }
        let skip = length - 2;
        if self.pos + skip > self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        self.pos += skip;
        Ok(())
    }

    fn read_sof(&mut self) -> Result<FrameHeader> {
        let length = self.read_u16_be()? as usize;
        let start = self.pos;

        let precision = self.read_u8()?;
        let height = self.read_u16_be()?;
        let width = self.read_u16_be()?;
        let num_components = self.read_u8()? as usize;

        let mut components = Vec::with_capacity(num_components);
        for _ in 0..num_components {
            let id = self.read_u8()?;
            let sampling = self.read_u8()?;
            let quant_table_index = self.read_u8()?;
            components.push(ComponentInfo {
                id,
                horizontal_sampling: sampling >> 4,
                vertical_sampling: sampling & 0x0F,
                quant_table_index,
            });
        }

        let consumed = self.pos - start;
        if consumed != length - 2 {
            self.pos = start + length - 2;
        }

        Ok(FrameHeader {
            precision,
            height,
            width,
            components,
        })
    }

    fn read_dqt(&mut self, tables: &mut [Option<QuantTable>; 4]) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        let end = self.pos + length - 2;

        while self.pos < end {
            let info = self.read_u8()?;
            let precision = info >> 4;
            let table_id = (info & 0x0F) as usize;

            if table_id >= 4 {
                return Err(JpegError::CorruptData(format!(
                    "quantization table id {} out of range",
                    table_id
                )));
            }

            let mut zigzag = [0u16; 64];
            if precision == 0 {
                for entry in zigzag.iter_mut() {
                    *entry = self.read_u8()? as u16;
                }
            } else {
                for entry in zigzag.iter_mut() {
                    *entry = self.read_u16_be()?;
                }
            }

            tables[table_id] = Some(QuantTable::from_zigzag(&zigzag));
        }

        Ok(())
    }

    fn read_dht(
        &mut self,
        dc_tables: &mut [Option<HuffmanTable>; 4],
        ac_tables: &mut [Option<HuffmanTable>; 4],
    ) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        let end = self.pos + length - 2;

        while self.pos < end {
            let info = self.read_u8()?;
            let table_class = info >> 4;
            let table_id = (info & 0x0F) as usize;

            if table_id >= 4 {
                return Err(JpegError::CorruptData(format!(
                    "Huffman table id {} out of range",
                    table_id
                )));
            }

            let mut bits = [0u8; 17];
            for i in 1..=16 {
                bits[i] = self.read_u8()?;
            }

            let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
            let mut values = Vec::with_capacity(total);
            for _ in 0..total {
                values.push(self.read_u8()?);
            }

            let table = HuffmanTable::build(&bits, &values)?;

            if table_class == 0 {
                dc_tables[table_id] = Some(table);
            } else {
                ac_tables[table_id] = Some(table);
            }
        }

        Ok(())
    }

    fn read_dri(&mut self) -> Result<u16> {
        let _length = self.read_u16_be()?;
        self.read_u16_be()
    }

    fn read_sos(&mut self) -> Result<ScanHeader> {
        let _length = self.read_u16_be()?;
        let num_components = self.read_u8()? as usize;

        let mut components = Vec::with_capacity(num_components);
        for _ in 0..num_components {
            let component_id = self.read_u8()?;
            let tables = self.read_u8()?;
            components.push(ScanComponentSelector {
                component_id,
                dc_table_index: tables >> 4,
                ac_table_index: tables & 0x0F,
            });
        }

        // Skip: Ss, Se, Ah|Al
        let _ss = self.read_u8()?;
        let _se = self.read_u8()?;
        let _ahl = self.read_u8()?;

        Ok(ScanHeader { components })
    }
}
