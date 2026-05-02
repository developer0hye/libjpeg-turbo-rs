/// JPEG marker writing for the encoder.
///
/// Writes the required JFIF/JPEG markers to produce a valid baseline JPEG file.
use crate::encode::tables::ZIGZAG_ORDER;

/// Write SOI (Start Of Image) marker: 0xFFD8.
pub fn write_soi(buf: &mut Vec<u8>) {
    buf.push(0xFF);
    buf.push(0xD8);
}

/// Write APP0 JFIF marker.
///
/// Produces a standard JFIF 1.01 header with 72 DPI resolution.
pub fn write_app0_jfif(buf: &mut Vec<u8>) {
    buf.push(0xFF);
    buf.push(0xE0); // APP0

    let length: u16 = 16; // 2 (length) + 5 (JFIF\0) + 2 (version) + 1 (units) + 4 (density) + 2 (thumbnail)
    buf.extend_from_slice(&length.to_be_bytes());

    // JFIF identifier
    buf.extend_from_slice(b"JFIF\0");

    // Version 1.01
    buf.push(1); // major
    buf.push(1); // minor

    // Units: 0 = no units (aspect ratio only)
    buf.push(0);

    // X density: 1 (aspect ratio)
    buf.extend_from_slice(&1u16.to_be_bytes());
    // Y density: 1
    buf.extend_from_slice(&1u16.to_be_bytes());

    // Thumbnail: 0x0
    buf.push(0);
    buf.push(0);
}

/// Write DQT (Define Quantization Table) marker.
///
/// Writes the quantization table in zigzag order. Uses 8-bit precision
/// if all values fit in a byte, otherwise 16-bit precision.
pub fn write_dqt(buf: &mut Vec<u8>, table_id: u8, table: &[u16; 64]) {
    buf.push(0xFF);
    buf.push(0xDB); // DQT

    let is_16bit = table.iter().any(|&v| v > 255);
    let precision = if is_16bit { 1u8 } else { 0u8 };
    let table_bytes = if is_16bit { 128 } else { 64 };
    let length: u16 = 2 + 1 + table_bytes; // length field + Pq/Tq byte + table data
    buf.extend_from_slice(&length.to_be_bytes());

    // Pq (precision, upper nibble) | Tq (table ID, lower nibble)
    buf.push((precision << 4) | (table_id & 0x0F));

    // Write table values in zigzag order
    for &natural_idx in &ZIGZAG_ORDER[..64] {
        if is_16bit {
            buf.extend_from_slice(&table[natural_idx].to_be_bytes());
        } else {
            buf.push(table[natural_idx] as u8);
        }
    }
}

/// Write SOF0 (Start Of Frame, Baseline DCT) marker.
///
/// `components` is a slice of (component_id, h_sampling_factor, v_sampling_factor, quant_table_id).
/// Always 8-bit precision — SOF0 is by definition baseline 8-bit; if the
/// caller has 12-bit data it must select SOF1 (extended sequential) instead.
pub fn write_sof0(buf: &mut Vec<u8>, width: u16, height: u16, components: &[(u8, u8, u8, u8)]) {
    buf.push(0xFF);
    buf.push(0xC0); // SOF0

    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());

    // Sample precision: 8 bits (SOF0 is baseline only)
    buf.push(8);

    // Image dimensions
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(&width.to_be_bytes());

    // Number of components
    buf.push(components.len() as u8);

    for &(id, h_samp, v_samp, quant_tbl_id) in components {
        buf.push(id);
        buf.push((h_samp << 4) | v_samp);
        buf.push(quant_tbl_id);
    }
}

/// Write DHT (Define Huffman Table) marker.
///
/// `class` is 0 for DC, 1 for AC.
/// `id` is the table identifier (0 or 1).
/// `bits[0]` is unused; `bits[1]..bits[16]` give the count of codes per length.
/// `values` are the symbol values.
pub fn write_dht(buf: &mut Vec<u8>, class: u8, id: u8, bits: &[u8; 17], values: &[u8]) {
    buf.push(0xFF);
    buf.push(0xC4); // DHT

    let num_symbols: usize = bits[1..].iter().map(|&b| b as usize).sum();
    let length: u16 = (2 + 1 + 16 + num_symbols) as u16;
    buf.extend_from_slice(&length.to_be_bytes());

    // Tc (table class, upper nibble) | Th (table ID, lower nibble)
    buf.push((class << 4) | (id & 0x0F));

    // Write bits[1..16] (16 bytes of code length counts)
    buf.extend_from_slice(&bits[1..17]);

    // Write symbol values
    buf.extend_from_slice(&values[..num_symbols]);
}

/// Write SOS (Start Of Scan) marker.
///
/// `components` is a slice of (component_id, dc_table_id, ac_table_id).
pub fn write_sos(buf: &mut Vec<u8>, components: &[(u8, u8, u8)]) {
    buf.push(0xFF);
    buf.push(0xDA); // SOS

    let length: u16 = 2 + 1 + (components.len() as u16 * 2) + 3;
    buf.extend_from_slice(&length.to_be_bytes());

    // Number of components in scan
    buf.push(components.len() as u8);

    for &(id, dc_tbl, ac_tbl) in components {
        buf.push(id);
        buf.push((dc_tbl << 4) | ac_tbl);
    }

    // Spectral selection: Ss=0, Se=63
    buf.push(0); // Ss
    buf.push(63); // Se
                  // Successive approximation: Ah=0, Al=0
    buf.push(0); // Ah << 4 | Al
}

/// Write a generic SOF marker with explicit marker code and sample
/// precision. Used by the SOF1/SOF2/SOF9/SOF10 helpers below to share a
/// single body, since the 12-bit transcode path needs to inject a
/// precision other than the 8-bit default.
fn write_sof_generic(
    buf: &mut Vec<u8>,
    marker_code: u8,
    width: u16,
    height: u16,
    components: &[(u8, u8, u8, u8)],
    precision: u8,
) {
    buf.push(0xFF);
    buf.push(marker_code);

    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());

    buf.push(precision);
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(&width.to_be_bytes());
    buf.push(components.len() as u8);

    for &(id, h_samp, v_samp, quant_tbl_id) in components {
        buf.push(id);
        buf.push((h_samp << 4) | v_samp);
        buf.push(quant_tbl_id);
    }
}

/// Write SOF1 (Start Of Frame, Extended Sequential DCT) marker, 8-bit precision.
///
/// Same structure as SOF0 but uses marker code 0xC1. C cjpeg emits SOF1
/// instead of SOF0 when any quantization table value exceeds 255 (16-bit
/// quantization), since baseline (SOF0) requires 8-bit quant tables.
pub fn write_sof1(buf: &mut Vec<u8>, width: u16, height: u16, components: &[(u8, u8, u8, u8)]) {
    write_sof_generic(buf, 0xC1, width, height, components, 8);
}

/// Write SOF2 (Start Of Frame, Progressive DCT) marker, 8-bit precision.
///
/// Same structure as SOF0 but uses marker code 0xC2.
pub fn write_sof2(buf: &mut Vec<u8>, width: u16, height: u16, components: &[(u8, u8, u8, u8)]) {
    write_sof_generic(buf, 0xC2, width, height, components, 8);
}

/// Write SOF2 with caller-supplied sample precision (8 or 12).
///
/// Used by 12-bit progressive transcode in
/// `api::coefficient::write_coefficients_progressive`.
pub fn write_sof2_with_precision(
    buf: &mut Vec<u8>,
    width: u16,
    height: u16,
    components: &[(u8, u8, u8, u8)],
    precision: u8,
) {
    write_sof_generic(buf, 0xC2, width, height, components, precision);
}

/// Write SOF9 (Start Of Frame, Arithmetic Sequential) marker, 8-bit precision.
///
/// Same structure as SOF0 but uses marker code 0xC9.
pub fn write_sof9(buf: &mut Vec<u8>, width: u16, height: u16, components: &[(u8, u8, u8, u8)]) {
    write_sof_generic(buf, 0xC9, width, height, components, 8);
}

/// Write SOF9 with caller-supplied sample precision (8 or 12).
///
/// Used by 12-bit arithmetic transcode in
/// `api::coefficient::write_coefficients_arithmetic`.
pub fn write_sof9_with_precision(
    buf: &mut Vec<u8>,
    width: u16,
    height: u16,
    components: &[(u8, u8, u8, u8)],
    precision: u8,
) {
    write_sof_generic(buf, 0xC9, width, height, components, precision);
}

/// Write SOF10 (Start Of Frame, Arithmetic Progressive DCT) marker, 8-bit precision.
///
/// Same structure as SOF2 but uses marker code 0xCA for arithmetic coding.
pub fn write_sof10(buf: &mut Vec<u8>, width: u16, height: u16, components: &[(u8, u8, u8, u8)]) {
    write_sof_generic(buf, 0xCA, width, height, components, 8);
}

/// Write SOF10 with caller-supplied sample precision (8 or 12).
///
/// Used by 12-bit arithmetic-progressive transcode in
/// `api::coefficient::write_coefficients_progressive_arithmetic`.
pub fn write_sof10_with_precision(
    buf: &mut Vec<u8>,
    width: u16,
    height: u16,
    components: &[(u8, u8, u8, u8)],
    precision: u8,
) {
    write_sof_generic(buf, 0xCA, width, height, components, precision);
}

/// Write SOS marker for progressive scan with spectral selection and successive approximation.
///
/// Mirrors C `jcmarker.c::write_scan_header` table-hiding logic: zeroes
/// `Td` / `Ta` in the per-component byte for scans that do not use the
/// corresponding tables. Specifically:
///
/// - `Td = 0` when `ss != 0` (no DC table needed for AC-only scans)
/// - `Td = 0` when `ah != 0` (refinement scans don't reference Huffman tables)
/// - `Ta = 0` when `se != 63` (no AC table needed for DC-only scans)
/// - `Ta = 0` when `ah != 0 && ss == 0` (DC refinement)
///
/// Without this normalisation, our progressive SOS bytes diverge from cjpeg
/// in the bytes where the upstream encoder hides table indices that the
/// decoder will never use.
pub fn write_sos_progressive(
    buf: &mut Vec<u8>,
    components: &[(u8, u8, u8)],
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
) {
    buf.push(0xFF);
    buf.push(0xDA); // SOS

    let length: u16 = 2 + 1 + (components.len() as u16 * 2) + 3;
    buf.extend_from_slice(&length.to_be_bytes());

    buf.push(components.len() as u8);

    let hide_dc: bool = ss != 0 || ah != 0;
    let hide_ac: bool = se != 63 || (ah != 0 && ss == 0);

    for &(id, dc_tbl, ac_tbl) in components {
        let td: u8 = if hide_dc { 0 } else { dc_tbl };
        let ta: u8 = if hide_ac { 0 } else { ac_tbl };
        buf.push(id);
        buf.push((td << 4) | ta);
    }

    buf.push(ss);
    buf.push(se);
    buf.push((ah << 4) | al);
}

/// Write DAC (Define Arithmetic Conditioning) marker.
///
/// `dc_params`: (L, U) per DC table. `ac_params`: Kx per AC table.
/// Accepts up to 16 slots per ITU-T T.81 `NUM_ARITH_TBLS` (jpeglib.h).
pub fn write_dac(
    buf: &mut Vec<u8>,
    num_dc: usize,
    dc_params: &[(u8, u8)],
    num_ac: usize,
    ac_params: &[u8],
) {
    use crate::decode::arithmetic::NUM_ARITH_TBLS;

    let mut dc_in_use = [false; NUM_ARITH_TBLS];
    let mut ac_in_use = [false; NUM_ARITH_TBLS];
    let mut full_dc_params = [(0u8, 1u8); NUM_ARITH_TBLS];
    let mut full_ac_params = [5u8; NUM_ARITH_TBLS];

    for i in 0..num_dc.min(NUM_ARITH_TBLS) {
        dc_in_use[i] = true;
        full_dc_params[i] = dc_params[i];
    }
    for i in 0..num_ac.min(NUM_ARITH_TBLS) {
        ac_in_use[i] = true;
        full_ac_params[i] = ac_params[i];
    }

    write_dac_selected(
        buf,
        &dc_in_use,
        &full_dc_params,
        &ac_in_use,
        &full_ac_params,
    );
}

/// Write DAC (Define Arithmetic Conditioning) marker for an arbitrary subset of
/// DC/AC arithmetic tables.
///
/// Entries are emitted in libjpeg-turbo order (Spec F.2.4.3, `jcmarker.c::emit_dac`):
/// DC0, AC0, DC1, AC1, ... DC15, AC15. Tc/Tb packs Tc (0=DC,1=AC) in the
/// high nibble and Tb (table index 0..=15) in the low nibble.
pub fn write_dac_selected(
    buf: &mut Vec<u8>,
    dc_in_use: &[bool; crate::decode::arithmetic::NUM_ARITH_TBLS],
    dc_params: &[(u8, u8); crate::decode::arithmetic::NUM_ARITH_TBLS],
    ac_in_use: &[bool; crate::decode::arithmetic::NUM_ARITH_TBLS],
    ac_params: &[u8; crate::decode::arithmetic::NUM_ARITH_TBLS],
) {
    let num_entries: usize = dc_in_use.iter().filter(|&&used| used).count()
        + ac_in_use.iter().filter(|&&used| used).count();
    if num_entries == 0 {
        return;
    }

    buf.push(0xFF);
    buf.push(0xCC); // DAC

    let length: u16 = 2 + (num_entries as u16 * 2);
    buf.extend_from_slice(&length.to_be_bytes());

    for i in 0..crate::decode::arithmetic::NUM_ARITH_TBLS {
        if dc_in_use[i] {
            let (l, u) = dc_params[i];
            buf.push(i as u8); // Tc=0 (DC), Tb=i (0..=15)
            buf.push((u << 4) | l);
        }
        if ac_in_use[i] {
            buf.push(0x10 | i as u8); // Tc=1 (AC), Tb=i (0..=15)
            buf.push(ac_params[i]);
        }
    }
}

/// Write APP1 EXIF marker. `tiff_data` is raw TIFF-format EXIF data (after "Exif\0\0" header).
pub fn write_app1_exif(buf: &mut Vec<u8>, tiff_data: &[u8]) {
    let header = b"Exif\0\0";
    let marker_len: u16 = (2 + header.len() + tiff_data.len()) as u16;

    buf.push(0xFF);
    buf.push(0xE1); // APP1
    buf.extend_from_slice(&marker_len.to_be_bytes());
    buf.extend_from_slice(header);
    buf.extend_from_slice(tiff_data);
}

/// Write APP2 ICC profile markers. Splits profile into chunks of max 65519 bytes.
pub fn write_app2_icc(buf: &mut Vec<u8>, profile: &[u8]) {
    const ICC_OVERHEAD: usize = 14; // "ICC_PROFILE\0" + seq_no + num_markers
    const MAX_DATA: usize = 65533 - ICC_OVERHEAD; // 65519

    let num_markers = profile.len().div_ceil(MAX_DATA);
    let mut offset = 0;

    for seq in 1..=num_markers {
        let chunk_len = (profile.len() - offset).min(MAX_DATA);
        let marker_len: u16 = (ICC_OVERHEAD + chunk_len) as u16 + 2;

        buf.push(0xFF);
        buf.push(0xE2); // APP2
        buf.extend_from_slice(&marker_len.to_be_bytes());
        buf.extend_from_slice(b"ICC_PROFILE\0");
        buf.push(seq as u8);
        buf.push(num_markers as u8);
        buf.extend_from_slice(&profile[offset..offset + chunk_len]);

        offset += chunk_len;
    }
}

/// Write Adobe APP14 marker for CMYK/YCCK color space identification.
/// transform: 0 = CMYK, 1 = YCbCr, 2 = YCCK
pub fn write_app14_adobe(buf: &mut Vec<u8>, transform: u8) {
    buf.push(0xFF);
    buf.push(0xEE); // APP14
    let length: u16 = 14; // 2 (length) + 5 (Adobe) + 2 (version) + 2 (flags0) + 2 (flags1) + 1 (transform)
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(b"Adobe"); // identifier
    buf.extend_from_slice(&100u16.to_be_bytes()); // version
    buf.extend_from_slice(&0u16.to_be_bytes()); // flags0
    buf.extend_from_slice(&0u16.to_be_bytes()); // flags1
    buf.push(transform); // color transform
}

/// Write SOF3 (lossless, Huffman-coded) frame header.
pub fn write_sof3(
    buf: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    buf.push(0xFF);
    buf.push(0xC3); // SOF3
    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());
    buf.push(precision);
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(&width.to_be_bytes());
    buf.push(components.len() as u8);
    for &(id, h_samp, v_samp, qt_idx) in components {
        buf.push(id);
        buf.push((h_samp << 4) | v_samp);
        buf.push(qt_idx);
    }
}

/// Write SOF11 (lossless, arithmetic-coded) frame header.
pub fn write_sof11(
    buf: &mut Vec<u8>,
    width: u16,
    height: u16,
    precision: u8,
    components: &[(u8, u8, u8, u8)],
) {
    buf.push(0xFF);
    buf.push(0xCB); // SOF11
    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());
    buf.push(precision);
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(&width.to_be_bytes());
    buf.push(components.len() as u8);
    for &(id, h_samp, v_samp, qt_idx) in components {
        buf.push(id);
        buf.push((h_samp << 4) | v_samp);
        buf.push(qt_idx);
    }
}

/// Write SOS for lossless scan. Ss=predictor (1-7), Se=0, Ah=0, Al=point_transform.
pub fn write_sos_lossless(
    buf: &mut Vec<u8>,
    components: &[(u8, u8)],
    predictor: u8,
    point_transform: u8,
) {
    buf.push(0xFF);
    buf.push(0xDA); // SOS
    let length: u16 = 2 + 1 + (components.len() as u16 * 2) + 3;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.push(components.len() as u8);
    for &(id, dc_tbl) in components {
        buf.push(id);
        buf.push(dc_tbl << 4); // DC table only, AC unused
    }
    buf.push(predictor); // Ss = predictor selection (1-7)
    buf.push(0); // Se = 0
    buf.push(point_transform & 0x0F); // Ah=0, Al=point_transform
}

/// Write COM (comment) marker.
pub fn write_com(buf: &mut Vec<u8>, text: &str) {
    buf.push(0xFF);
    buf.push(0xFE);
    let length: u16 = (2 + text.len()) as u16;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(text.as_bytes());
}

/// Write DRI (restart interval) marker.
pub fn write_dri(buf: &mut Vec<u8>, interval: u16) {
    buf.push(0xFF);
    buf.push(0xDD);
    buf.extend_from_slice(&4u16.to_be_bytes());
    buf.extend_from_slice(&interval.to_be_bytes());
}

/// Write arbitrary marker with data.
pub fn write_marker(buf: &mut Vec<u8>, code: u8, data: &[u8]) {
    buf.push(0xFF);
    buf.push(code);
    let length: u16 = (2 + data.len()) as u16;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(data);
}

/// Write APP0 JFIF marker with custom density.
pub fn write_app0_jfif_with_density(buf: &mut Vec<u8>, unit: u8, x_density: u16, y_density: u16) {
    buf.push(0xFF);
    buf.push(0xE0); // APP0

    let length: u16 = 16;
    buf.extend_from_slice(&length.to_be_bytes());

    // JFIF identifier
    buf.extend_from_slice(b"JFIF\0");

    // Version 1.01
    buf.push(1);
    buf.push(1);

    // Units
    buf.push(unit);

    // X density
    buf.extend_from_slice(&x_density.to_be_bytes());
    // Y density
    buf.extend_from_slice(&y_density.to_be_bytes());

    // Thumbnail: 0x0
    buf.push(0);
    buf.push(0);
}

/// Write EOI (End Of Image) marker: 0xFFD9.
pub fn write_eoi(buf: &mut Vec<u8>) {
    buf.push(0xFF);
    buf.push(0xD9);
}

/// Write a tables-only JPEG stream (like `jpeg_write_tables`).
///
/// Produces SOI + DQT (luminance + chrominance) + DHT (standard DC/AC) + EOI.
/// No image data — only table definitions for table-sharing workflows.
pub fn write_tables_only(quality: u8) -> Vec<u8> {
    use crate::api::quality;
    use crate::encode::tables;

    let scale: u32 = quality::quality_scaling(quality);
    let luma_qt: [u16; 64] =
        quality::scale_quant_table_linear(&tables::STD_LUMINANCE_QUANT_TABLE, scale, false);
    let chroma_qt: [u16; 64] =
        quality::scale_quant_table_linear(&tables::STD_CHROMINANCE_QUANT_TABLE, scale, false);

    let mut buf: Vec<u8> = Vec::with_capacity(1024);
    write_soi(&mut buf);
    write_dqt(&mut buf, 0, &luma_qt);
    write_dqt(&mut buf, 1, &chroma_qt);

    // Standard DC/AC Huffman tables
    write_dht(
        &mut buf,
        0,
        0,
        &tables::DC_LUMINANCE_BITS,
        &tables::DC_LUMINANCE_VALUES,
    );
    write_dht(
        &mut buf,
        1,
        0,
        &tables::AC_LUMINANCE_BITS,
        &tables::AC_LUMINANCE_VALUES,
    );
    write_dht(
        &mut buf,
        0,
        1,
        &tables::DC_CHROMINANCE_BITS,
        &tables::DC_CHROMINANCE_VALUES,
    );
    write_dht(
        &mut buf,
        1,
        1,
        &tables::AC_CHROMINANCE_BITS,
        &tables::AC_CHROMINANCE_VALUES,
    );

    write_eoi(&mut buf);
    buf
}

/// Streaming marker writer for building marker segments byte-by-byte.
pub struct MarkerStreamWriter {
    marker_type: u8,
    data: Vec<u8>,
}

impl MarkerStreamWriter {
    /// Create a new streaming marker writer for the given marker type.
    pub fn new(marker_type: u8) -> Self {
        Self {
            marker_type,
            data: Vec::new(),
        }
    }

    /// Write a single byte to the marker data.
    pub fn write_byte(&mut self, byte: u8) {
        self.data.push(byte);
    }

    /// Write multiple bytes to the marker data.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    /// Finish writing and return the complete marker segment.
    pub fn finish(self) -> Vec<u8> {
        let length: u16 = (2 + self.data.len()) as u16;
        let mut segment: Vec<u8> = Vec::with_capacity(4 + self.data.len());
        segment.push(0xFF);
        segment.push(self.marker_type);
        segment.extend_from_slice(&length.to_be_bytes());
        segment.extend_from_slice(&self.data);
        segment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soi_marker() {
        let mut buf = Vec::new();
        write_soi(&mut buf);
        assert_eq!(buf, [0xFF, 0xD8]);
    }

    #[test]
    fn eoi_marker() {
        let mut buf = Vec::new();
        write_eoi(&mut buf);
        assert_eq!(buf, [0xFF, 0xD9]);
    }

    #[test]
    fn app0_jfif_marker() {
        let mut buf = Vec::new();
        write_app0_jfif(&mut buf);
        // Starts with FF E0
        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[1], 0xE0);
        // Length
        let length = u16::from_be_bytes([buf[2], buf[3]]);
        assert_eq!(length, 16);
        // JFIF identifier
        assert_eq!(&buf[4..9], b"JFIF\0");
        // Version 1.01
        assert_eq!(buf[9], 1);
        assert_eq!(buf[10], 1);
        // Total size
        assert_eq!(buf.len(), 18); // 2 (marker) + 16 (data)
    }

    #[test]
    fn dqt_8bit_precision() {
        let table = [16u16; 64];
        let mut buf = Vec::new();
        write_dqt(&mut buf, 0, &table);
        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[1], 0xDB);
        let length = u16::from_be_bytes([buf[2], buf[3]]);
        assert_eq!(length, 67); // 2 + 1 + 64
                                // Precision = 0 (8-bit), table ID = 0
        assert_eq!(buf[4], 0x00);
        // All table values should be 16
        for i in 0..64 {
            assert_eq!(buf[5 + i], 16);
        }
    }

    #[test]
    fn dqt_16bit_precision() {
        let mut table = [16u16; 64];
        table[0] = 300; // Force 16-bit precision
        let mut buf = Vec::new();
        write_dqt(&mut buf, 1, &table);
        assert_eq!(buf[4], 0x11); // Precision=1, ID=1
        let length = u16::from_be_bytes([buf[2], buf[3]]);
        assert_eq!(length, 131); // 2 + 1 + 128
    }

    #[test]
    fn sof0_marker_rgb() {
        let mut buf = Vec::new();
        let components = vec![
            (1, 2, 2, 0), // Y: 2x2, quant table 0
            (2, 1, 1, 1), // Cb: 1x1, quant table 1
            (3, 1, 1, 1), // Cr: 1x1, quant table 1
        ];
        write_sof0(&mut buf, 640, 480, &components);
        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[1], 0xC0);
        // Precision
        assert_eq!(buf[4], 8);
        // Height = 480
        assert_eq!(u16::from_be_bytes([buf[5], buf[6]]), 480);
        // Width = 640
        assert_eq!(u16::from_be_bytes([buf[7], buf[8]]), 640);
        // Num components
        assert_eq!(buf[9], 3);
    }

    #[test]
    fn sos_marker() {
        let mut buf = Vec::new();
        let components = vec![
            (1, 0, 0), // Y: DC table 0, AC table 0
            (2, 1, 1), // Cb: DC table 1, AC table 1
            (3, 1, 1), // Cr: DC table 1, AC table 1
        ];
        write_sos(&mut buf, &components);
        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[1], 0xDA);
        // Num components
        assert_eq!(buf[4], 3);
        // Ss = 0, Se = 63
        let offset = 4 + 1 + 3 * 2;
        assert_eq!(buf[offset], 0);
        assert_eq!(buf[offset + 1], 63);
        assert_eq!(buf[offset + 2], 0);
    }

    #[test]
    fn dht_marker() {
        let bits: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
        let values: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
        let mut buf = Vec::new();
        write_dht(&mut buf, 0, 0, &bits, &values);
        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[1], 0xC4);
        // Class=0, ID=0
        assert_eq!(buf[4], 0x00);
        // Length = 2 + 1 + 16 + 12 = 31
        let length = u16::from_be_bytes([buf[2], buf[3]]);
        assert_eq!(length, 31);
    }

    /// Round-trip a DAC segment with Tb=12 through writer + decoder.
    ///
    /// Spec: ITU-T T.81 B.2.4.3 / jpeglib.h NUM_ARITH_TBLS = 16 allows
    /// Tb up to 15. Before A4-3, writer's dc_in_use/ac_in_use were
    /// length-4, silently dropping any slot > 3.
    #[test]
    fn dac_round_trip_high_table_indices() {
        use crate::decode::arithmetic::NUM_ARITH_TBLS;
        use crate::decode::marker::MarkerReader;

        // Select DC slot 5 (L=2, U=6) and AC slot 12 (Kx=9). Other
        // slots must remain at their default conditioning.
        let mut dc_in_use = [false; NUM_ARITH_TBLS];
        let mut ac_in_use = [false; NUM_ARITH_TBLS];
        let mut dc_params = [(0u8, 1u8); NUM_ARITH_TBLS];
        let mut ac_params = [5u8; NUM_ARITH_TBLS];
        dc_in_use[5] = true;
        dc_params[5] = (2, 6);
        ac_in_use[12] = true;
        ac_params[12] = 9;

        let mut buf: Vec<u8> = Vec::new();
        write_dac_selected(&mut buf, &dc_in_use, &dc_params, &ac_in_use, &ac_params);

        // DAC = FF CC, length = 2 + 2 entries * 2 bytes = 6, then entries.
        assert_eq!(buf[0], 0xFF);
        assert_eq!(buf[1], 0xCC);
        let length = u16::from_be_bytes([buf[2], buf[3]]);
        assert_eq!(length, 6, "one DC + one AC entry");
        // Emission order (libjpeg-turbo): DC0..15 for each slot interleaved
        // with AC. Here only DC5 and AC12 are set, so the bytes are:
        //   Tc/Tb = 0x05 (DC, Tb=5), val = (U<<4)|L = 0x62
        //   Tc/Tb = 0x1C (AC, Tb=12), val = 9
        assert_eq!(buf[4], 0x05);
        assert_eq!(buf[5], 0x62);
        assert_eq!(buf[6], 0x1C);
        assert_eq!(buf[7], 0x09);

        // Now parse it back through the decoder's read_dac — drop the
        // FF CC marker bytes, pass the raw segment (length + entries).
        let segment: &[u8] = &buf[2..];
        let mut reader = MarkerReader::new(segment);
        let mut rt_dc: [(u8, u8); NUM_ARITH_TBLS] = [(0, 1); NUM_ARITH_TBLS];
        let mut rt_ac: [u8; NUM_ARITH_TBLS] = [5; NUM_ARITH_TBLS];
        reader
            .read_dac_public(&mut rt_dc, &mut rt_ac)
            .expect("DAC must round-trip");

        assert_eq!(rt_dc[5], (2, 6), "DC Tb=5 conditioning preserved");
        assert_eq!(rt_ac[12], 9, "AC Tb=12 Kx preserved");
        // Untouched slots keep defaults.
        assert_eq!(rt_dc[0], (0, 1));
        assert_eq!(rt_ac[0], 5);
    }
}
