use wasm_bindgen_test::*;
wasm_bindgen_test_configure!(run_in_browser);

use libjpeg_turbo_wasm::{
    decode, decode_to, encode, jpeg_dimensions, DecodedImage, PixelFormat, Subsampling, WasmEncoder,
};

const TEST_JPEG: &[u8] = include_bytes!("../../../tests/fixtures/blue_16x16_420.jpg");

#[wasm_bindgen_test]
fn decode_rgb() {
    let img: DecodedImage = decode(TEST_JPEG).unwrap();
    assert_eq!(img.width(), 16);
    assert_eq!(img.height(), 16);
    assert_eq!(img.format(), PixelFormat::Rgb);
    assert_eq!(img.data().length(), 16 * 16 * 3);
    assert_eq!(img.bytes_per_pixel(), 3);
}

#[wasm_bindgen_test]
fn decode_to_rgba() {
    let img: DecodedImage = decode_to(TEST_JPEG, PixelFormat::Rgba).unwrap();
    assert_eq!(img.width(), 16);
    assert_eq!(img.height(), 16);
    assert_eq!(img.format(), PixelFormat::Rgba);
    assert_eq!(img.data().length(), 16 * 16 * 4);
    assert_eq!(img.bytes_per_pixel(), 4);
}

#[wasm_bindgen_test]
fn decode_to_grayscale() {
    let img: DecodedImage = decode_to(TEST_JPEG, PixelFormat::Grayscale).unwrap();
    assert_eq!(img.format(), PixelFormat::Grayscale);
    assert_eq!(img.data().length(), 16 * 16);
    assert_eq!(img.bytes_per_pixel(), 1);
}

#[wasm_bindgen_test]
fn encode_roundtrip() {
    let img: DecodedImage = decode(TEST_JPEG).unwrap();
    let pixels: Vec<u8> = img.data().to_vec();
    let w: u32 = img.width();
    let h: u32 = img.height();

    let jpeg_out: js_sys::Uint8Array =
        encode(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    assert!(jpeg_out.length() > 0);

    // Decode the re-encoded JPEG to verify validity
    let img2: DecodedImage = decode(&jpeg_out.to_vec()).unwrap();
    assert_eq!(img2.width(), w);
    assert_eq!(img2.height(), h);
}

#[wasm_bindgen_test]
fn encoder_builder() {
    let img: DecodedImage = decode(TEST_JPEG).unwrap();
    let pixels: Vec<u8> = img.data().to_vec();
    let w: u32 = img.width();
    let h: u32 = img.height();

    let mut enc: WasmEncoder = WasmEncoder::new(&pixels, w, h, PixelFormat::Rgb);
    enc.quality(85);
    enc.subsampling(Subsampling::S444);
    enc.optimize_huffman(true);

    let result: js_sys::Uint8Array = enc.encode().unwrap();
    assert!(result.length() > 0);

    // Verify the output is valid JPEG
    let img2: DecodedImage = decode(&result.to_vec()).unwrap();
    assert_eq!(img2.width(), w);
    assert_eq!(img2.height(), h);
}

#[wasm_bindgen_test]
fn encoder_progressive() {
    let img: DecodedImage = decode(TEST_JPEG).unwrap();
    let pixels: Vec<u8> = img.data().to_vec();

    let mut enc: WasmEncoder =
        WasmEncoder::new(&pixels, img.width(), img.height(), PixelFormat::Rgb);
    enc.quality(75);
    enc.progressive(true);

    let result: js_sys::Uint8Array = enc.encode().unwrap();
    assert!(result.length() > 0);
}

#[wasm_bindgen_test]
fn jpeg_dimensions_from_header() {
    let dims: js_sys::Uint32Array = jpeg_dimensions(TEST_JPEG).unwrap();
    assert_eq!(dims.get_index(0), 16);
    assert_eq!(dims.get_index(1), 16);
}

#[wasm_bindgen_test]
fn data_ptr_and_len() {
    let img: DecodedImage = decode(TEST_JPEG).unwrap();
    assert!(img.data_ptr() > 0);
    assert_eq!(img.data_len(), 16 * 16 * 3);
}

#[wasm_bindgen_test]
fn decode_invalid_data() {
    let bad_data: &[u8] = &[0xFF, 0x00, 0x42];
    let result = decode(bad_data);
    assert!(result.is_err());
}
