use libjpeg_turbo_rs::decode::bitstream::BitReader;

#[test]
fn read_bits_basic() {
    let data = [0xB4u8]; // 10110100
    let mut reader = BitReader::new(&data);
    assert_eq!(reader.read_bits(1), 1);
    assert_eq!(reader.read_bits(3), 0b011);
    assert_eq!(reader.read_bits(4), 0b0100);
}

#[test]
fn read_bits_across_bytes() {
    let data = [0xAB, 0xCD];
    let mut reader = BitReader::new(&data);
    assert_eq!(reader.read_bits(4), 0b1010);
    assert_eq!(reader.read_bits(8), 0b1011_1100);
    assert_eq!(reader.read_bits(4), 0b1101);
}

#[test]
fn byte_stuffing_ff00_is_transparent() {
    let data = [0xFF, 0x00, 0x80];
    let mut reader = BitReader::new(&data);
    assert_eq!(reader.read_bits(8), 0xFF);
    assert_eq!(reader.read_bits(1), 1);
}

#[test]
fn peek_bits_does_not_consume() {
    let data = [0xB4u8];
    let mut reader = BitReader::new(&data);
    assert_eq!(reader.peek_bits(4), 0b1011);
    assert_eq!(reader.peek_bits(4), 0b1011);
    assert_eq!(reader.read_bits(4), 0b1011);
    assert_eq!(reader.read_bits(4), 0b0100);
}
