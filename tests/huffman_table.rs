use libjpeg_turbo_rs::common::huffman_table::HuffmanTable;

#[test]
fn build_dc_luminance_table() {
    let bits: [u8; 17] = [0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    let table = HuffmanTable::build(&bits, &values).unwrap();
    assert_eq!(table.num_symbols(), 12);
}

#[test]
fn lookup_known_codes() {
    let bits: [u8; 17] = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![0x00, 0x01, 0x02];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    let (symbol, length) = table.lookup(0b0000_0000_0000_0000).unwrap();
    assert_eq!(symbol, 0x00);
    assert_eq!(length, 1);

    let (symbol, length) = table.lookup(0b1000_0000_0000_0000).unwrap();
    assert_eq!(symbol, 0x01);
    assert_eq!(length, 2);

    let (symbol, length) = table.lookup(0b1100_0000_0000_0000).unwrap();
    assert_eq!(symbol, 0x02);
    assert_eq!(length, 3);
}

#[test]
fn lookup_fast_known_codes() {
    let bits: [u8; 17] = [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let values: Vec<u8> = vec![0x00, 0x01, 0x02];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    assert_eq!(table.lookup_fast(0b0000_0000_0000_0000), (0x00, 1));
    assert_eq!(table.lookup_fast(0b1000_0000_0000_0000), (0x01, 2));
    assert_eq!(table.lookup_fast(0b1100_0000_0000_0000), (0x02, 3));
}

#[test]
fn lookup_fast_misses_long_codes() {
    let mut bits: [u8; 17] = [0; 17];
    bits[10] = 1;
    let values: Vec<u8> = vec![0xAB];
    let table = HuffmanTable::build(&bits, &values).unwrap();

    assert_eq!(table.lookup_fast(0), (0, 0));

    let (symbol, length) = table.lookup(0).unwrap();
    assert_eq!(symbol, 0xAB);
    assert_eq!(length, 10);
}
