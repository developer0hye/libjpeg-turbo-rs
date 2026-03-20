use libjpeg_turbo_rs::common::quant_table::QuantTable;
use libjpeg_turbo_rs::decode::dequant;

#[test]
fn dequantize_multiplies_by_table() {
    let mut coeffs = [0i16; 64];
    coeffs[0] = 10;
    coeffs[1] = 5;

    let mut qt_zigzag = [1u16; 64];
    qt_zigzag[0] = 16;
    qt_zigzag[1] = 11;

    let table = QuantTable::from_zigzag(&qt_zigzag);
    let result = dequant::dequantize_block(&coeffs, &table);

    assert_eq!(result[0], 160); // 10 * 16
    assert_eq!(result[1], 55); // 5 * 11
}

#[test]
fn dequantize_preserves_zeros() {
    let coeffs = [0i16; 64];
    let qt_zigzag = [8u16; 64];
    let table = QuantTable::from_zigzag(&qt_zigzag);

    let result = dequant::dequantize_block(&coeffs, &table);
    for &val in result.iter() {
        assert_eq!(val, 0);
    }
}
