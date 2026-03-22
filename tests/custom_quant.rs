use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn custom_quant_table_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let table = [16u16; 64]; // flat quant table
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, table)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn custom_quant_table_affects_output() {
    let pixels = vec![128u8; 16 * 16 * 3];

    let default_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .encode()
        .unwrap();

    let custom_table = [1u16; 64]; // very fine quantization
    let custom_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, custom_table)
        .encode()
        .unwrap();

    // Different quant tables should produce different output
    assert_ne!(default_jpeg, custom_jpeg);
}

#[test]
fn custom_quant_table_chroma() {
    let pixels = vec![128u8; 16 * 16 * 3];

    let default_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();

    let chroma_table = [2u16; 64]; // custom chroma quantization
    let custom_jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .quant_table(1, chroma_table)
        .encode()
        .unwrap();

    // Custom chroma table should produce different output
    assert_ne!(default_jpeg, custom_jpeg);
}

#[test]
fn custom_quant_table_both_luma_and_chroma() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let luma_table = [8u16; 64];
    let chroma_table = [32u16; 64];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, luma_table)
        .quant_table(1, chroma_table)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}
