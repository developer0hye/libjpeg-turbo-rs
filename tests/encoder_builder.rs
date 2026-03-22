use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat, Subsampling};

#[test]
fn encoder_basic_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn encoder_progressive() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(85)
        .progressive(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_with_metadata() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let icc = vec![0x42u8; 100];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Rgb)
        .quality(75)
        .icc_profile(&icc)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(icc.as_slice()));
}

#[test]
fn encoder_arithmetic() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .arithmetic(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_optimized_huffman() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .optimize_huffman(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}

#[test]
fn encoder_lossless() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

#[test]
fn existing_compress_still_works() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = libjpeg_turbo_rs::compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444)
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
}
