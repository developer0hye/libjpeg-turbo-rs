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

    // Units: 1 = dots per inch
    buf.push(1);

    // X density: 72
    buf.extend_from_slice(&72u16.to_be_bytes());
    // Y density: 72
    buf.extend_from_slice(&72u16.to_be_bytes());

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
    for zigzag_pos in 0..64 {
        let natural_idx = ZIGZAG_ORDER[zigzag_pos];
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
pub fn write_sof0(buf: &mut Vec<u8>, width: u16, height: u16, components: &[(u8, u8, u8, u8)]) {
    buf.push(0xFF);
    buf.push(0xC0); // SOF0

    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());

    // Sample precision: 8 bits
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

/// Write SOF2 (Start Of Frame, Progressive DCT) marker.
///
/// Same structure as SOF0 but uses marker code 0xC2.
pub fn write_sof2(buf: &mut Vec<u8>, width: u16, height: u16, components: &[(u8, u8, u8, u8)]) {
    buf.push(0xFF);
    buf.push(0xC2); // SOF2

    let length: u16 = 2 + 1 + 2 + 2 + 1 + (components.len() as u16 * 3);
    buf.extend_from_slice(&length.to_be_bytes());

    buf.push(8); // 8-bit precision
    buf.extend_from_slice(&height.to_be_bytes());
    buf.extend_from_slice(&width.to_be_bytes());
    buf.push(components.len() as u8);

    for &(id, h_samp, v_samp, quant_tbl_id) in components {
        buf.push(id);
        buf.push((h_samp << 4) | v_samp);
        buf.push(quant_tbl_id);
    }
}

/// Write SOS marker for progressive scan with spectral selection and successive approximation.
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

    for &(id, dc_tbl, ac_tbl) in components {
        buf.push(id);
        buf.push((dc_tbl << 4) | ac_tbl);
    }

    buf.push(ss);
    buf.push(se);
    buf.push((ah << 4) | al);
}

/// Write EOI (End Of Image) marker: 0xFFD9.
pub fn write_eoi(buf: &mut Vec<u8>) {
    buf.push(0xFF);
    buf.push(0xD9);
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
}
