use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::common::types::*;

/// "ICC_PROFILE\0" identifier (12 bytes) in APP2 markers.
const ICC_PROFILE_HEADER: &[u8; 12] = b"ICC_PROFILE\0";

/// "Exif\0\0" identifier (6 bytes) in APP1 markers.
const EXIF_HEADER: &[u8; 6] = b"Exif\0\0";

// JPEG marker codes
const SOI: u8 = 0xD8;
const EOI: u8 = 0xD9;
const SOF0: u8 = 0xC0;
const SOF1: u8 = 0xC1; // Extended sequential, Huffman-coded
const SOF2: u8 = 0xC2;
const SOF3: u8 = 0xC3; // Lossless, Huffman-coded
const SOF9: u8 = 0xC9; // Arithmetic sequential
const SOF10: u8 = 0xCA; // Arithmetic progressive
const SOF11: u8 = 0xCB; // Lossless, arithmetic-coded
const DHT: u8 = 0xC4;
const DAC: u8 = 0xCC; // Define arithmetic conditioning
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
    /// ICC profile chunks from APP2 markers (reassembled via `common::icc`).
    pub icc_chunks: Vec<IccChunk>,
    /// Raw EXIF TIFF data from the first APP1 marker (after "Exif\0\0" header).
    pub exif_data: Option<Vec<u8>>,
    /// COM marker text, if present.
    pub comment: Option<String>,
    /// Pixel density from JFIF header.
    pub density: DensityInfo,
    /// True if using arithmetic entropy coding (SOF9/SOF10).
    pub is_arithmetic: bool,
    /// DAC conditioning: DC parameters (L, U) per table.
    pub arith_dc_params: [(u8, u8); 4],
    /// DAC conditioning: AC parameter (Kx) per table.
    pub arith_ac_params: [u8; 4],
    /// Saved APP/COM markers according to the marker save configuration.
    pub saved_markers: Vec<SavedMarker>,
}

/// Reads and parses JPEG markers from a byte slice.
pub struct MarkerReader<'a> {
    data: &'a [u8],
    pos: usize,
    marker_save_config: MarkerSaveConfig,
}

impl<'a> MarkerReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            marker_save_config: MarkerSaveConfig::None,
        }
    }

    /// Set the marker save configuration.
    pub fn set_marker_save_config(&mut self, config: MarkerSaveConfig) {
        self.marker_save_config = config;
    }

    /// Check whether a given marker code should be saved according to config.
    fn should_save_marker(&self, code: u8) -> bool {
        match &self.marker_save_config {
            MarkerSaveConfig::None => false,
            MarkerSaveConfig::All => (0xE0..=0xEF).contains(&code) || code == COM,
            MarkerSaveConfig::AppOnly => (0xE0..=0xEF).contains(&code),
            MarkerSaveConfig::Specific(codes) => codes.contains(&code),
        }
    }

    /// Read a marker segment's raw data without advancing pos.
    /// Returns the data portion (after the 2-byte length field).
    fn peek_marker_data(&self) -> Option<Vec<u8>> {
        if self.pos + 2 > self.data.len() {
            return None;
        }
        let length = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]) as usize;
        if length < 2 || self.pos + length > self.data.len() {
            return None;
        }
        Some(self.data[self.pos + 2..self.pos + length].to_vec())
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
        let mut icc_chunks: Vec<IccChunk> = Vec::new();
        let mut exif_data: Option<Vec<u8>> = None;
        let mut is_arithmetic = false;
        let mut arith_dc_params: [(u8, u8); 4] = [(0, 1); 4];
        let mut arith_ac_params: [u8; 4] = [5; 4];
        let mut comment: Option<String> = None;
        let mut density: DensityInfo = DensityInfo::default();
        let mut saved_markers: Vec<SavedMarker> = Vec::new();

        loop {
            let marker = self.read_marker()?;
            match marker {
                SOF0 | SOF1 => {
                    // SOF0 = baseline, SOF1 = extended sequential (e.g., 16-bit DQT)
                    // Both are sequential DCT with Huffman coding, decoded identically.
                    frame = Some(self.read_sof(false, false)?);
                }
                SOF2 => {
                    frame = Some(self.read_sof(true, false)?);
                }
                SOF3 => {
                    frame = Some(self.read_sof(false, true)?);
                }
                SOF9 => {
                    // Arithmetic sequential
                    frame = Some(self.read_sof(false, false)?);
                    is_arithmetic = true;
                }
                SOF10 => {
                    // Arithmetic progressive
                    frame = Some(self.read_sof(true, false)?);
                    is_arithmetic = true;
                }
                SOF11 => {
                    // Lossless, arithmetic-coded
                    frame = Some(self.read_sof(false, true)?);
                    is_arithmetic = true;
                }
                DAC => {
                    self.read_dac(&mut arith_dc_params, &mut arith_ac_params)?;
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

                    let is_progressive = frame.as_ref().is_some_and(|f| f.is_progressive);
                    // Non-interleaved baseline: SOS has fewer components than
                    // the frame.  Continue reading to find remaining SOS markers.
                    let scan_comp_count = scans.last().unwrap().header.components.len();
                    let is_non_interleaved_baseline = !is_progressive
                        && frame
                            .as_ref()
                            .is_some_and(|f| scan_comp_count < f.components.len());
                    if !is_progressive && !is_non_interleaved_baseline {
                        // Interleaved baseline: single scan, stop here
                        break;
                    }

                    // Progressive or non-interleaved baseline: skip entropy
                    // data to find next marker
                    self.skip_entropy_data();
                }
                EOI => {
                    break;
                }
                // APP1 (EXIF) — parse for EXIF metadata
                0xE1 => {
                    if self.should_save_marker(0xE1) {
                        if let Some(raw) = self.peek_marker_data() {
                            saved_markers.push(SavedMarker {
                                code: 0xE1,
                                data: raw,
                            });
                        }
                    }
                    self.read_app1(&mut exif_data)?;
                }
                // APP2 (ICC profile) — parse for ICC profile chunks
                0xE2 => {
                    if self.should_save_marker(0xE2) {
                        if let Some(raw) = self.peek_marker_data() {
                            saved_markers.push(SavedMarker {
                                code: 0xE2,
                                data: raw,
                            });
                        }
                    }
                    self.read_app2(&mut icc_chunks)?;
                }
                // APP14 (Adobe marker) — parse for color transform info
                0xEE => {
                    if self.should_save_marker(0xEE) {
                        if let Some(raw) = self.peek_marker_data() {
                            saved_markers.push(SavedMarker {
                                code: 0xEE,
                                data: raw,
                            });
                        }
                    }
                    self.read_app14(&mut saw_adobe_marker, &mut adobe_transform)?;
                }
                // APP0 (JFIF) — parse for density info
                0xE0 => {
                    if self.should_save_marker(0xE0) {
                        if let Some(raw) = self.peek_marker_data() {
                            saved_markers.push(SavedMarker {
                                code: 0xE0,
                                data: raw,
                            });
                        }
                    }
                    self.read_app0(&mut density)?;
                }
                // COM marker — parse comment text
                COM => {
                    if self.should_save_marker(COM) {
                        if let Some(raw) = self.peek_marker_data() {
                            saved_markers.push(SavedMarker {
                                code: COM,
                                data: raw,
                            });
                        }
                    }
                    self.read_com(&mut comment)?;
                }
                // Other APPn markers — save if configured, then skip
                m if (0xE3..=0xEF).contains(&m) => {
                    if self.should_save_marker(m) {
                        if let Some(raw) = self.peek_marker_data() {
                            saved_markers.push(SavedMarker { code: m, data: raw });
                        }
                    }
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
            icc_chunks,
            exif_data,
            comment,
            density,
            is_arithmetic,
            arith_dc_params,
            arith_ac_params,
            saved_markers,
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

    /// Parse APP0 (JFIF) marker to extract pixel density info.
    fn read_app0(&mut self, density: &mut DensityInfo) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        if length < 2 {
            return Err(JpegError::CorruptData("APP0 segment length < 2".into()));
        }
        let end = self.pos + length - 2;

        // JFIF header: "JFIF\0" (5 bytes) + version (2) + units (1) + density (4) = 12 bytes min payload
        if length >= 16
            && self.pos + 12 <= self.data.len()
            && &self.data[self.pos..self.pos + 5] == b"JFIF\0"
        {
            let unit_byte = self.data[self.pos + 7];
            let x_density = u16::from_be_bytes([self.data[self.pos + 8], self.data[self.pos + 9]]);
            let y_density =
                u16::from_be_bytes([self.data[self.pos + 10], self.data[self.pos + 11]]);
            density.unit = match unit_byte {
                1 => DensityUnit::Dpi,
                2 => DensityUnit::Dpcm,
                _ => DensityUnit::Unknown,
            };
            density.x = x_density;
            density.y = y_density;
        }

        self.pos = end;
        Ok(())
    }

    /// Parse COM marker to extract comment text.
    fn read_com(&mut self, comment: &mut Option<String>) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        if length < 2 {
            return Err(JpegError::CorruptData("COM segment length < 2".into()));
        }
        let text_len = length - 2;
        if self.pos + text_len > self.data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let data = &self.data[self.pos..self.pos + text_len];
        self.pos += text_len;
        *comment = Some(String::from_utf8_lossy(data).into_owned());
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

    /// Parse APP1 marker for EXIF data.
    /// Only the first EXIF APP1 is stored; subsequent ones are skipped.
    fn read_app1(&mut self, exif_data: &mut Option<Vec<u8>>) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        if length < 2 {
            return Err(JpegError::CorruptData("APP1 segment length < 2".into()));
        }
        let end = self.pos + length - 2;

        // "Exif\0\0" header is 6 bytes; only store first EXIF APP1
        if exif_data.is_none()
            && length >= 8
            && self.pos + 6 <= self.data.len()
            && &self.data[self.pos..self.pos + 6] == EXIF_HEADER
        {
            let data_start = self.pos + 6;
            let data_len = end.saturating_sub(data_start);
            *exif_data = Some(self.data[data_start..data_start + data_len].to_vec());
        }

        self.pos = end;
        Ok(())
    }

    /// Parse APP2 marker for ICC profile data.
    /// ICC profile chunks have a 14-byte overhead: "ICC_PROFILE\0" (12) + seq_no (1) + num_markers (1).
    fn read_app2(&mut self, icc_chunks: &mut Vec<IccChunk>) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        if length < 2 {
            return Err(JpegError::CorruptData("APP2 segment length < 2".into()));
        }
        let end = self.pos + length - 2;

        // ICC_PROFILE header: 12 bytes identifier + 1 seq_no + 1 num_markers = 14 bytes overhead
        if length >= 16
            && self.pos + 14 <= self.data.len()
            && &self.data[self.pos..self.pos + 12] == ICC_PROFILE_HEADER
        {
            let seq_no = self.data[self.pos + 12];
            let num_markers = self.data[self.pos + 13];
            let data_start = self.pos + 14;
            let data_len = end.saturating_sub(data_start);
            let data = self.data[data_start..data_start + data_len].to_vec();
            icc_chunks.push(IccChunk {
                seq_no,
                num_markers,
                data,
            });
        }

        self.pos = end;
        Ok(())
    }

    fn read_sof(&mut self, is_progressive: bool, is_lossless: bool) -> Result<FrameHeader> {
        let length = self.read_u16_be()? as usize;
        let start = self.pos;

        let precision = self.read_u8()?;
        let height = self.read_u16_be()?;
        let width = self.read_u16_be()?;
        let num_components = self.read_u8()? as usize;

        if width == 0 {
            return Err(JpegError::CorruptData("SOF width must not be 0".into()));
        }
        if num_components == 0 || num_components > 4 {
            return Err(JpegError::CorruptData(format!(
                "SOF component count must be 1-4, got {}",
                num_components
            )));
        }

        let mut components = Vec::with_capacity(num_components);
        for _ in 0..num_components {
            let id = self.read_u8()?;
            let sampling = self.read_u8()?;
            let h_samp = sampling >> 4;
            let v_samp = sampling & 0x0F;
            if h_samp == 0 || h_samp > 4 || v_samp == 0 || v_samp > 4 {
                return Err(JpegError::CorruptData(format!(
                    "sampling factor must be 1-4, got {}x{}",
                    h_samp, v_samp
                )));
            }
            let quant_table_index = self.read_u8()?;
            components.push(ComponentInfo {
                id,
                horizontal_sampling: h_samp,
                vertical_sampling: v_samp,
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
            is_lossless,
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
            for b in &mut bits[1..=16] {
                *b = self.read_u8()?;
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

    /// Parse DAC (Define Arithmetic Conditioning) marker.
    fn read_dac(&mut self, dc_params: &mut [(u8, u8); 4], ac_params: &mut [u8; 4]) -> Result<()> {
        let length = self.read_u16_be()? as usize;
        let end = self.pos + length - 2;

        while self.pos < end {
            let tc_tb = self.read_u8()?;
            let tc = tc_tb >> 4; // table class: 0=DC, 1=AC
            let tb = (tc_tb & 0x0F) as usize; // table index
            let val = self.read_u8()?;

            if tb >= 4 {
                continue;
            }
            if tc == 0 {
                // DC: val = L | (U << 4)
                let l = val & 0x0F;
                let u = val >> 4;
                dc_params[tb] = (l, u);
            } else {
                // AC: val = Kx
                ac_params[tb] = val;
            }
        }
        Ok(())
    }

    fn read_sos(&mut self) -> Result<ScanHeader> {
        let _length = self.read_u16_be()?;
        let num_components = self.read_u8()? as usize;

        if num_components == 0 || num_components > 4 {
            return Err(JpegError::CorruptData(format!(
                "SOS component count must be 1-4, got {}",
                num_components
            )));
        }

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
