//! `JpegCoefficients::quant_tables` are in natural (row-major) order, while
//! `ComponentCoefficients::blocks` are in zigzag order. The field's doc said
//! zigzag until a fuzz helper multiplied the two index-for-index (found
//! reviewing #690); this pins the order the code has always used, which
//! `write_coefficients` relies on when it writes them back as DQT.

use libjpeg_turbo_rs::{compress, read_coefficients, PixelFormat, Subsampling};

/// Zigzag index → natural (row-major) position: libjpeg's
/// `jpeg_natural_order`.
const ZIGZAG_TO_NATURAL: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// The first DQT segment's table 0, as stored: zigzag order, 8-bit values.
fn first_dqt_table(jpeg: &[u8]) -> [u16; 64] {
    let dqt: usize = jpeg
        .windows(2)
        .position(|marker| marker == [0xFF, 0xDB])
        .expect("a DQT segment");
    assert_eq!(jpeg[dqt + 4], 0x00, "8-bit table 0 first");
    let mut table: [u16; 64] = [0; 64];
    for (k, value) in table.iter_mut().enumerate() {
        *value = u16::from(jpeg[dqt + 5 + k]);
    }
    table
}

#[test]
fn quant_tables_are_in_natural_order() {
    let (width, height): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = (0..width * height * 3)
        .map(|i| (i * 7 % 256) as u8)
        .collect();
    let jpeg: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .expect("compress");
    let stored: [u16; 64] = first_dqt_table(&jpeg);
    let coefficients = read_coefficients(&jpeg).expect("read_coefficients");
    let table: &[u16; 64] = &coefficients.quant_tables[0];
    for (k, &value) in stored.iter().enumerate() {
        assert_eq!(
            table[ZIGZAG_TO_NATURAL[k]], value,
            "zigzag index {k} belongs at natural position {}",
            ZIGZAG_TO_NATURAL[k]
        );
    }
    // The standard luminance table differs between the two orders, so the
    // check above cannot pass for a zigzag-ordered field.
    assert_ne!(table[..], stored[..]);
}
