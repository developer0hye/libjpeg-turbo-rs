use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;

// JPEG marker codes
const SOI: u8 = 0xD8;
const EOI: u8 = 0xD9;
const SOF0: u8 = 0xC0;
const SOF2: u8 = 0xC2;
const DHT: u8 = 0xC4;
const DQT: u8 = 0xDB;
const SOS: u8 = 0xDA;
const DRI: u8 = 0xDD;
const COM: u8 = 0xFE;

/// Per-scan info with Huffman table snapshot (needed because tables can be
/// redefined between scans in progressive JPEG).
#[derive(Debug, Clone)]
pub struct ScanInfo {
    pub header: ScanHeader,
    /// Byte offset where this scan's entropy-coded data begins.
    pub data_offset: usize,
    pub dc_huffman_tables: [Option<HuffmanTable>; 4],
    pub ac_huffman_tables: [Option<HuffmanTable>; 4],
    pub restart_interval: u16,
}

/// All metadata parsed from JPEG markers.
///
/// For baseline: one scan, entropy_data_offset points to its data.
/// For progressive: multiple scans with separate offsets and table snapshots.
#[derive(Debug)]
pub struct JpegMetadata {
    pub frame: FrameHeader,
    /// First scan header (used by baseline path).
    pub scan: ScanHeader,
    pub quant_tables: [Option<QuantTable>; 4],
    pub dc_huffman_tables: [Option<HuffmanTable>; 4],
    pub ac_huffman_tables: [Option<HuffmanTable>; 4],
    pub restart_interval: u16,
    /// Byte offset where the first scan's entropy-coded data begins.
    pub entropy_data_offset: usize,
    /// For progressive: all scans with table snapshots.
    pub scans: Vec<ScanInfo>,
    /// True if an Adobe APP14 marker was found.
    pub saw_adobe_marker: bool,
    /// Adobe color transform code (0 = CMYK/RGB, 1 = YCbCr, 2 = YCCK).
    pub adobe_transform: u8,
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

    /// Parse all markers. For baseline, stops after first SOS.
    /// For progressive, reads all SOS markers until EOI.
    pub fn read_markers(&mut self) -> Result<JpegMetadata> {
        self.expect_marker(SOI)?;

        let mut frame: Option<FrameHeader> = None;
        let mut quant_tables: [Option<QuantTable>; 4] = [None, None, None, None];
        let mut dc_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut ac_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
        let mut restart_interval: u16 = 0;
        let mut scans: Vec<ScanInfo> = Vec::new();
        let mut saw_adobe_marker: bool = false;
        let mut adobe_transform: u8 = 0;

        loop {
            let marker = self.read_marker()?;
            match marker {
                SOF0 => {
                    frame = Some(self.read_sof(false)?);
                }
                SOF2 => {
                    frame = Some(self.read_sof(true)?);
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
                    let header = self.read_sos()?;
                    let offset = self.pos;
                    scans.push(ScanInfo {
                        header,
                        data_offset: offset,
                        dc_huffman_tables: dc_huffman_tables.clone(),
                        ac_huffman_tables: ac_huffman_tables.clone(),
                        restart_interval,
                    });

                    let is_progressive = frame.as_ref().map_or(false, |f| f.is_progressive);
                    if !is_progressive {
                        // Baseline: single scan, stop here
                        break;
                    }

                    // Progressive: skip entropy data to find next marker
                    self.skip_entropy_data();
                }
                EOI => {
                    break;
                }
                // APP14 (Adobe marker) — parse for color transform info
                0xEE => {
                    self.read_app14(&mut saw_adobe_marker, &mut adobe_transform)?;
                }
                // Skip other APPn and COM markers
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
        if scans.is_empty() {
            return Err(JpegError::CorruptData("missing SOS marker".into()));
        }

        let first_scan = scans[0].header.clone();
        let first_offset = scans[0].data_offset;

        Ok(JpegMetadata {
            frame,
            scan: first_scan,
            quant_tables,
            dc_huffman_tables,
            ac_huffman_tables,
            restart_interval,
            entropy_data_offset: first_offset,
            scans,
            saw_adobe_marker,
            adobe_transform,
        })
    }

    /// Skip past entropy-coded data to find the next marker.
    /// Entropy data ends at an unescaped 0xFF byte followed by a non-zero, non-RST marker.
    fn skip_entropy_data(&mut self) {
        while self.pos < self.data.len() {
            if self.data[self.pos] != 0xFF {
                self.pos += 1;
                continue;
            }
            // Found 0xFF — check next byte
            if self.pos + 1 >= self.data.len() {
                self.pos += 1;
                return;
            }
            let next = self.data[self.pos + 1];
            if next == 0x00 {
                // Byte-stuffed 0xFF data — skip both bytes
                self.pos += 2;
            } else if (0xD0..=0xD7).contains(&next) {
                // Restart marker — skip it and continue scanning entropy data
                self.pos += 2;
            } else {
                // Real marker — leave pos at 0xFF so read_marker can find it
                // read_marker skips 0xFF prefix bytes, so point to the 0xFF
                self.pos += 1; // skip past 0xFF, read_marker will read the marker byte
                return;
            }
        }
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

    /// Parse Adobe APP14 marker to extract color transform.
    /// Transform values: 0 = CMYK or RGB, 1 = YCbCr, 2 = YCCK.
    fn read_app14(&mut self, saw_adobe: &mut bool, transform: &mut u8) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        if length < 2 {
            return Err(JpegError::CorruptData("APP14 segment length < 2".into()));
        }
        let end = self.pos + length - 2;

        // Adobe APP14 marker starts with "Adobe" (5 bytes) and is at least 12 bytes
        if length >= 14
            && self.pos + 12 <= self.data.len()
            && &self.data[self.pos..self.pos + 5] == b"Adobe"
        {
            // Skip "Adobe" (5) + version (2) + flags0 (2) + flags1 (2) = 11 bytes
            *transform = self.data[self.pos + 11];
            *saw_adobe = true;
        }

        self.pos = end;
        Ok(())
    }

    fn read_sof(&mut self, is_progressive: bool) -> Result<FrameHeader> {
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
            is_progressive,
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

        let ss = self.read_u8()?;
        let se = self.read_u8()?;
        let ahl = self.read_u8()?;
        let ah = ahl >> 4;
        let al = ahl & 0x0F;

        Ok(ScanHeader {
            components,
            spec_start: ss,
            spec_end: se,
            succ_high: ah,
            succ_low: al,
        })
    }
}
