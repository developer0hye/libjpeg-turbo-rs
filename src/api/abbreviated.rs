//! Abbreviated (tables-only / body-only) JPEG datastream support.
//!
//! Implements JPEG spec F.1.2.4 abbreviated datastreams:
//! - Tables-only stream: SOI + DQT + DHT + (DAC if arithmetic) + EOI
//! - Body-only stream: full JPEG with DQT/DHT stripped
//! - Inter-session table reuse via `TablesOnlyState`

use crate::common::error::{JpegError, Result};
use crate::common::huffman_table::HuffmanTable;
use crate::common::quant_table::QuantTable;
use crate::encode::marker_writer;
use crate::encode::tables as encode_tables;

// ===========================================================================
// Tables-only stream generation
// ===========================================================================

/// Build a tables-only abbreviated datastream from an `Encoder`'s configuration.
///
/// Produces: SOI + DQT(luma) + DQT(chroma) + DHT(DC0) + DHT(AC0) + DHT(DC1) + DHT(AC1) + EOI.
/// Matches libjpeg-turbo's `jpeg_write_tables()`.
pub(crate) fn write_tables_for_encoder(encoder: &crate::api::encoder::Encoder<'_>) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(2048);
    marker_writer::write_soi(&mut buf);

    // Quantization tables
    let quant_tables = encoder.effective_quant_tables_for_abbrev();
    // Determine how many quant table slots are populated
    // (slot 0 = luma, slot 1 = chroma; standard JPEG uses 0 and 1)
    for (idx, maybe_table) in quant_tables.iter().enumerate() {
        if let Some(table) = maybe_table {
            marker_writer::write_dqt(&mut buf, idx as u8, table);
        }
    }
    // If no custom quant tables, write standard luma + chroma for the configured quality
    if quant_tables.iter().all(|t| t.is_none()) {
        // This case shouldn't happen since build_quant_tables always fills slot 0 and 1,
        // but handle defensively with standard tables at quality 75.
        let scale = crate::api::quality::quality_scaling(75);
        let luma = crate::api::quality::scale_quant_table_linear(
            &encode_tables::STD_LUMINANCE_QUANT_TABLE,
            scale,
            false,
        );
        let chroma = crate::api::quality::scale_quant_table_linear(
            &encode_tables::STD_CHROMINANCE_QUANT_TABLE,
            scale,
            false,
        );
        marker_writer::write_dqt(&mut buf, 0, &luma);
        marker_writer::write_dqt(&mut buf, 1, &chroma);
    }

    // Huffman tables: use custom if set, otherwise standard tables
    let dc_tables = encoder.custom_huffman_dc_tables();
    let ac_tables = encoder.custom_huffman_ac_tables();

    let any_custom_huff =
        dc_tables.iter().any(|t| t.is_some()) || ac_tables.iter().any(|t| t.is_some());

    if any_custom_huff {
        for (idx, maybe_table) in dc_tables.iter().enumerate() {
            if let Some(table) = maybe_table {
                marker_writer::write_dht(&mut buf, 0, idx as u8, &table.bits, &table.values);
            }
        }
        for (idx, maybe_table) in ac_tables.iter().enumerate() {
            if let Some(table) = maybe_table {
                marker_writer::write_dht(&mut buf, 1, idx as u8, &table.bits, &table.values);
            }
        }
    } else {
        // Standard Huffman tables
        marker_writer::write_dht(
            &mut buf,
            0,
            0,
            &encode_tables::DC_LUMINANCE_BITS,
            &encode_tables::DC_LUMINANCE_VALUES,
        );
        marker_writer::write_dht(
            &mut buf,
            1,
            0,
            &encode_tables::AC_LUMINANCE_BITS,
            &encode_tables::AC_LUMINANCE_VALUES,
        );
        marker_writer::write_dht(
            &mut buf,
            0,
            1,
            &encode_tables::DC_CHROMINANCE_BITS,
            &encode_tables::DC_CHROMINANCE_VALUES,
        );
        marker_writer::write_dht(
            &mut buf,
            1,
            1,
            &encode_tables::AC_CHROMINANCE_BITS,
            &encode_tables::AC_CHROMINANCE_VALUES,
        );
    }

    // If arithmetic coding, write DAC marker
    if encoder.is_arithmetic() {
        // Default arithmetic conditioning parameters (matches C libjpeg-turbo defaults)
        let dc_in_use = [true, true, false, false];
        let dc_params = [(0u8, 1u8), (0u8, 1u8), (0u8, 1u8), (0u8, 1u8)];
        let ac_in_use = [true, true, false, false];
        let ac_params = [5u8, 5u8, 5u8, 5u8];
        marker_writer::write_dac_selected(&mut buf, &dc_in_use, &dc_params, &ac_in_use, &ac_params);
    }

    marker_writer::write_eoi(&mut buf);
    buf
}

// ===========================================================================
// Table marker stripping (body-only stream generation)
// ===========================================================================

/// Strip DQT (0xDB) and DHT (0xC4) markers from a JPEG byte stream.
///
/// Used to produce abbreviated body-only streams for inter-session table reuse.
/// APP markers, SOF, SOS, and entropy data are preserved as-is.
pub(crate) fn strip_table_markers(data: Vec<u8>) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(data.len());
    let mut pos: usize = 0;

    while pos < data.len() {
        if pos + 1 >= data.len() {
            out.push(data[pos]);
            break;
        }

        // Look for 0xFF marker
        if data[pos] != 0xFF {
            out.push(data[pos]);
            pos += 1;
            continue;
        }

        let code = data[pos + 1];
        match code {
            // DQT and DHT: skip this marker segment entirely
            0xDB | 0xC4 => {
                if pos + 3 < data.len() {
                    let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                    pos += 2 + seg_len;
                } else {
                    // Truncated segment; preserve remaining bytes
                    out.extend_from_slice(&data[pos..]);
                    break;
                }
            }
            // SOS (0xDA): once we hit SOS, copy everything verbatim to preserve entropy data
            0xDA => {
                out.extend_from_slice(&data[pos..]);
                break;
            }
            // All other markers: copy marker + segment verbatim
            _ => {
                out.push(data[pos]);
                out.push(data[pos + 1]);
                pos += 2;

                // Markers with no length: SOI(0xD8), EOI(0xD9), RST(0xD0-0xD7)
                let no_length = code == 0xD8 || code == 0xD9 || (0xD0..=0xD7).contains(&code);

                if !no_length && pos + 2 <= data.len() {
                    let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                    let copy_end = (pos + seg_len).min(data.len());
                    out.extend_from_slice(&data[pos..copy_end]);
                    pos = copy_end;
                }
            }
        }
    }
    out
}

// ===========================================================================
// TablesOnlyState: parsed tables from an abbreviated tables-only stream
// ===========================================================================

/// Tables parsed from an abbreviated tables-only JPEG stream.
///
/// Produced by `read_header()` when the stream contains DQT/DHT/DAC but no SOF.
/// Use `Decoder::new_with_tables()` to decode a body-only stream with these tables.
#[derive(Debug, Clone)]
pub struct TablesOnlyState {
    /// Quantization tables indexed 0-3 (in zigzag → natural order via `QuantTable`).
    pub(crate) quant_tables: [Option<QuantTable>; 4],
    /// DC Huffman tables indexed 0-3.
    pub(crate) dc_huffman_tables: [Option<HuffmanTable>; 4],
    /// AC Huffman tables indexed 0-3.
    pub(crate) ac_huffman_tables: [Option<HuffmanTable>; 4],
    /// Whether arithmetic coding conditioning tables were present.
    pub(crate) is_arithmetic: bool,
    /// DAC DC conditioning parameters (L, U) per slot.
    pub(crate) arith_dc_params: [(u8, u8); 4],
    /// DAC AC conditioning parameter (Kx) per slot.
    pub(crate) arith_ac_params: [u8; 4],
}

impl TablesOnlyState {
    /// Returns true if quantization table `idx` was populated.
    pub fn has_quant_table(&self, idx: usize) -> bool {
        idx < 4 && self.quant_tables[idx].is_some()
    }

    /// Returns true if DC Huffman table `idx` was populated.
    pub fn has_dc_huffman(&self, idx: usize) -> bool {
        idx < 4 && self.dc_huffman_tables[idx].is_some()
    }

    /// Returns true if AC Huffman table `idx` was populated.
    pub fn has_ac_huffman(&self, idx: usize) -> bool {
        idx < 4 && self.ac_huffman_tables[idx].is_some()
    }
}

// ===========================================================================
// HeaderResult enum
// ===========================================================================

/// Result of parsing a JPEG stream header.
///
/// A tables-only abbreviated stream (per JPEG spec F.1.2.4) contains DQT/DHT/DAC
/// but no SOF marker. A full or body-only stream returns `Image` with a `Decoder`
/// ready to produce pixels.
pub enum HeaderResult<'a> {
    /// A tables-only abbreviated stream was found. Contains parsed table state.
    TablesOnly(Box<TablesOnlyState>),
    /// A full JPEG or body-only stream with SOF was found. Contains a ready decoder.
    Image(Box<crate::decode::pipeline::Decoder<'a>>),
}

// ===========================================================================
// read_header(): entry point for abbreviated datastream detection
// ===========================================================================

/// Parse a JPEG stream header, distinguishing full JPEGs from tables-only abbreviated streams.
///
/// - **Tables-only stream** (SOI + DQT/DHT/DAC + EOI, no SOF): returns `HeaderResult::TablesOnly`.
/// - **Full or body-only stream** (has SOF marker): returns `HeaderResult::Image` with a `Decoder`.
///
/// This matches libjpeg-turbo's `jpeg_read_header()` with `JPEG_SUSPENDED` / `JPEG_HEADER_TABLES_ONLY`
/// return codes.
///
/// # Errors
///
/// Returns `Err` only for genuinely corrupt data (invalid marker, unexpected EOF, etc.).
/// A tables-only stream is **not** an error.
pub fn read_header(data: &[u8]) -> Result<HeaderResult<'_>> {
    // Attempt to parse as a tables-only stream first.
    // A tables-only stream: SOI + [DQT/DHT/DAC markers] + EOI (no SOF/SOS).
    if let Some(state) = try_parse_tables_only(data)? {
        return Ok(HeaderResult::TablesOnly(Box::new(state)));
    }

    // Otherwise it's a full JPEG (or body-only with SOF). Use the normal decoder.
    let decoder = crate::decode::pipeline::Decoder::new(data)?;
    Ok(HeaderResult::Image(Box::new(decoder)))
}

/// Try to parse a tables-only abbreviated stream.
///
/// Returns `Ok(Some(TablesOnlyState))` when the stream is tables-only (no SOF before EOI).
/// Returns `Ok(None)` when the stream contains a SOF marker (full JPEG).
/// Returns `Err` only for corrupt/invalid data.
fn try_parse_tables_only(data: &[u8]) -> Result<Option<TablesOnlyState>> {
    if data.len() < 4 {
        return Err(JpegError::UnexpectedEof);
    }
    // Must start with SOI
    if data[0] != 0xFF || data[1] != 0xD8 {
        return Err(JpegError::UnexpectedMarker(data[1]));
    }

    let mut pos: usize = 2;
    let mut quant_tables: [Option<QuantTable>; 4] = [None, None, None, None];
    let mut dc_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
    let mut ac_huffman_tables: [Option<HuffmanTable>; 4] = [None, None, None, None];
    let mut is_arithmetic = false;
    let mut arith_dc_params: [(u8, u8); 4] = [(0, 1); 4];
    let mut arith_ac_params: [u8; 4] = [5; 4];
    let mut saw_table = false;

    loop {
        // Skip fill bytes (0xFF padding)
        while pos < data.len() && data[pos] == 0xFF {
            pos += 1;
        }
        if pos >= data.len() {
            return Err(JpegError::UnexpectedEof);
        }

        let code = data[pos];
        pos += 1;

        match code {
            // SOF markers: this is a full JPEG, not tables-only
            0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC9 | 0xCA | 0xCB => {
                return Ok(None);
            }
            // EOI: tables-only stream ends here
            0xD9 => {
                if saw_table {
                    return Ok(Some(TablesOnlyState {
                        quant_tables,
                        dc_huffman_tables,
                        ac_huffman_tables,
                        is_arithmetic,
                        arith_dc_params,
                        arith_ac_params,
                    }));
                }
                // EOI without any table markers: not a tables-only stream
                return Ok(None);
            }
            // DQT
            0xDB => {
                parse_dqt(data, &mut pos, &mut quant_tables)?;
                saw_table = true;
            }
            // DHT
            0xC4 => {
                parse_dht(
                    data,
                    &mut pos,
                    &mut dc_huffman_tables,
                    &mut ac_huffman_tables,
                )?;
                saw_table = true;
            }
            // DAC
            0xCC => {
                parse_dac(data, &mut pos, &mut arith_dc_params, &mut arith_ac_params)?;
                is_arithmetic = true;
                saw_table = true;
            }
            // SOI (nested? shouldn't happen but skip)
            0xD8 => {}
            // SOS: this is a body stream, not tables-only
            0xDA => {
                return Ok(None);
            }
            // All other markers with length: skip
            _ => {
                if pos + 2 > data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                if seg_len < 2 {
                    return Err(JpegError::CorruptData("marker segment length < 2".into()));
                }
                if pos + seg_len > data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                pos += seg_len;
            }
        }
    }
}

/// Parse a DQT marker segment starting at `*pos` (points to the 2-byte length field).
fn parse_dqt(
    data: &[u8],
    pos: &mut usize,
    quant_tables: &mut [Option<QuantTable>; 4],
) -> Result<()> {
    if *pos + 2 > data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    let length = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if length < 2 {
        return Err(JpegError::CorruptData("DQT segment length < 2".into()));
    }
    let end = *pos + length;
    if end > data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    *pos += 2;

    while *pos < end {
        let info = data[*pos];
        *pos += 1;
        let precision = info >> 4;
        let table_id = (info & 0x0F) as usize;
        if table_id >= 4 {
            return Err(JpegError::CorruptData("DQT table id out of range".into()));
        }
        let mut zigzag = [0u16; 64];
        if precision == 0 {
            for entry in zigzag.iter_mut() {
                if *pos >= data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                *entry = data[*pos] as u16;
                *pos += 1;
            }
        } else {
            for entry in zigzag.iter_mut() {
                if *pos + 2 > data.len() {
                    return Err(JpegError::UnexpectedEof);
                }
                *entry = u16::from_be_bytes([data[*pos], data[*pos + 1]]);
                *pos += 2;
            }
        }
        quant_tables[table_id] = Some(QuantTable::from_zigzag(&zigzag));
    }
    Ok(())
}

/// Parse a DHT marker segment starting at `*pos` (points to the 2-byte length field).
fn parse_dht(
    data: &[u8],
    pos: &mut usize,
    dc_tables: &mut [Option<HuffmanTable>; 4],
    ac_tables: &mut [Option<HuffmanTable>; 4],
) -> Result<()> {
    if *pos + 2 > data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    let length = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if length < 2 {
        return Err(JpegError::CorruptData("DHT segment length < 2".into()));
    }
    let end = *pos + length;
    if end > data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    *pos += 2;

    while *pos < end {
        let info = data[*pos];
        *pos += 1;
        let table_class = info >> 4;
        let table_id = (info & 0x0F) as usize;
        if table_id >= 4 {
            return Err(JpegError::CorruptData("DHT table id out of range".into()));
        }
        let mut bits = [0u8; 17];
        for b in &mut bits[1..=16] {
            if *pos >= data.len() {
                return Err(JpegError::UnexpectedEof);
            }
            *b = data[*pos];
            *pos += 1;
        }
        let total: usize = bits[1..=16].iter().map(|&b| b as usize).sum();
        if *pos + total > data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let values = data[*pos..*pos + total].to_vec();
        *pos += total;

        let table = HuffmanTable::build(&bits, &values)?;
        if table_class == 0 {
            dc_tables[table_id] = Some(table);
        } else {
            ac_tables[table_id] = Some(table);
        }
    }
    Ok(())
}

/// Parse a DAC marker segment starting at `*pos` (points to the 2-byte length field).
fn parse_dac(
    data: &[u8],
    pos: &mut usize,
    dc_params: &mut [(u8, u8); 4],
    ac_params: &mut [u8; 4],
) -> Result<()> {
    if *pos + 2 > data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    let length = u16::from_be_bytes([data[*pos], data[*pos + 1]]) as usize;
    if length < 2 {
        return Err(JpegError::CorruptData("DAC segment length < 2".into()));
    }
    let end = *pos + length;
    if end > data.len() {
        return Err(JpegError::UnexpectedEof);
    }
    *pos += 2;

    while *pos < end {
        if *pos + 2 > data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let tc_tb = data[*pos];
        let val = data[*pos + 1];
        *pos += 2;

        let tc = tc_tb >> 4;
        let tb = (tc_tb & 0x0F) as usize;
        if tb >= 4 {
            continue;
        }
        if tc == 0 {
            dc_params[tb] = (val & 0x0F, val >> 4);
        } else {
            ac_params[tb] = val;
        }
    }
    Ok(())
}
