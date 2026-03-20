use libjpeg_turbo_rs::common::huffman_table::HuffmanTable;
use libjpeg_turbo_rs::decode::bitstream::BitReader;
use libjpeg_turbo_rs::decode::huffman;

#[test]
fn decode_dc_coefficient_positive() {
    // Category 3, value +5: Huffman code "0" + extra bits "101"
    let bits: [u8; 17] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![3];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    // 0_101_0000 = 0x50
    let data = [0x50u8, 0x00];
    let mut reader = BitReader::new(&data);

    let dc_value = huffman::decode_dc_coefficient(&mut reader, &table).unwrap();
    assert_eq!(dc_value, 5);
}

#[test]
fn decode_dc_negative_value() {
    // Category 2, value -3: code "0" + extra bits "00"
    let bits: [u8; 17] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![2];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    let data = [0x00u8, 0x00];
    let mut reader = BitReader::new(&data);

    let dc_value = huffman::decode_dc_coefficient(&mut reader, &table).unwrap();
    assert_eq!(dc_value, -3);
}

#[test]
fn decode_dc_zero() {
    let bits: [u8; 17] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![0];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    let data = [0x00u8, 0x00];
    let mut reader = BitReader::new(&data);

    let dc_value = huffman::decode_dc_coefficient(&mut reader, &table).unwrap();
    assert_eq!(dc_value, 0);
}
