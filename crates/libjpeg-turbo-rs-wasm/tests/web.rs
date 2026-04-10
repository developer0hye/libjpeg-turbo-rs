use wasm_bindgen_test::*;

use libjpeg_turbo_rs_wasm::{
    decode, decode_to, encode, jpeg_dimensions, DecodedImage, PixelFormat, Subsampling, WasmEncoder,
};

// --- Fixtures ---
const TEST_JPEG: &[u8] = include_bytes!("../../../tests/fixtures/blue_16x16_420.jpg");
const PROG_JPEG: &[u8] = include_bytes!("../../../tests/fixtures/blue_16x16_420_prog.jpg");
const CHECKER_640: &[u8] = include_bytes!("../../../tests/fixtures/checker_640x480_420.jpg");
const CHECKER_PROG: &[u8] = include_bytes!("../../../tests/fixtures/checker_640x480_420_prog.jpg");

// Various sizes and subsampling from cjpeg cross-validation fixtures
const ODD_1X1_420: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_1x1_420.jpg");
const ODD_15X9_420: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_15x9_landscape_420.jpg");
const ODD_15X9_422: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_15x9_landscape_422.jpg");
const ODD_15X9_444: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_15x9_landscape_444.jpg");
const ODD_16X15_420: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_16x15_even_odd_420.jpg");
const ODD_15X16_420: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_15x16_odd_even_420.jpg");
const STRIP_100X1_420: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_100x1_strip_420.jpg");
const STRIP_100X1_422: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_100x1_strip_422.jpg");
const STRIP_100X1_444: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_100x1_strip_444.jpg");
const ODD_127X63_420: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_127x63_2to1_420.jpg");
const ODD_127X63_422: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_127x63_2to1_422.jpg");
const ODD_127X63_444: &[u8] = include_bytes!("../../../tests/fixtures/cjpeg_127x63_2to1_444.jpg");

// ===================== Basic API tests =====================

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
    assert_eq!(img.format(), PixelFormat::Rgba);
    assert_eq!(img.data().length(), 16 * 16 * 4);
    assert_eq!(img.bytes_per_pixel(), 4);
}

#[wasm_bindgen_test]
fn decode_to_grayscale_from_color_errors() {
    let result = decode_to(TEST_JPEG, PixelFormat::Grayscale);
    assert!(result.is_err());
}

#[wasm_bindgen_test]
fn decode_invalid_data() {
    let result = decode(&[0xFF, 0x00, 0x42]);
    assert!(result.is_err());
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

// ===================== Decode fixture tests =====================

fn assert_decode_ok(data: &[u8], expected_w: u32, expected_h: u32, label: &str) {
    let img: DecodedImage =
        decode(data).unwrap_or_else(|e| panic!("{label}: decode failed: {:?}", e));
    assert_eq!(img.width(), expected_w, "{label}: width");
    assert_eq!(img.height(), expected_h, "{label}: height");
    assert_eq!(
        img.data().length(),
        expected_w * expected_h * 3,
        "{label}: data size"
    );
}

fn assert_decode_rgba_ok(data: &[u8], expected_w: u32, expected_h: u32, label: &str) {
    let img: DecodedImage = decode_to(data, PixelFormat::Rgba)
        .unwrap_or_else(|e| panic!("{label}: decode RGBA failed: {:?}", e));
    assert_eq!(img.width(), expected_w, "{label}: width");
    assert_eq!(img.height(), expected_h, "{label}: height");
    assert_eq!(
        img.data().length(),
        expected_w * expected_h * 4,
        "{label}: data size"
    );
}

#[wasm_bindgen_test]
fn decode_progressive() {
    assert_decode_ok(PROG_JPEG, 16, 16, "blue_16x16_prog");
}

#[wasm_bindgen_test]
fn decode_640x480() {
    assert_decode_ok(CHECKER_640, 640, 480, "checker_640x480_420");
    assert_decode_rgba_ok(CHECKER_640, 640, 480, "checker_640x480_420_rgba");
}

#[wasm_bindgen_test]
fn decode_640x480_progressive() {
    assert_decode_ok(CHECKER_PROG, 640, 480, "checker_640x480_420_prog");
}

#[wasm_bindgen_test]
fn decode_1x1() {
    assert_decode_ok(ODD_1X1_420, 1, 1, "1x1_420");
}

#[wasm_bindgen_test]
fn decode_odd_15x9() {
    assert_decode_ok(ODD_15X9_420, 15, 9, "15x9_420");
    assert_decode_ok(ODD_15X9_422, 15, 9, "15x9_422");
    assert_decode_ok(ODD_15X9_444, 15, 9, "15x9_444");
}

#[wasm_bindgen_test]
fn decode_odd_even_boundaries() {
    assert_decode_ok(ODD_16X15_420, 16, 15, "16x15_420");
    assert_decode_ok(ODD_15X16_420, 15, 16, "15x16_420");
}

#[wasm_bindgen_test]
fn decode_strip_100x1() {
    assert_decode_ok(STRIP_100X1_420, 100, 1, "100x1_420");
    assert_decode_ok(STRIP_100X1_422, 100, 1, "100x1_422");
    assert_decode_ok(STRIP_100X1_444, 100, 1, "100x1_444");
}

#[wasm_bindgen_test]
fn decode_127x63() {
    assert_decode_ok(ODD_127X63_420, 127, 63, "127x63_420");
    assert_decode_ok(ODD_127X63_422, 127, 63, "127x63_422");
    assert_decode_ok(ODD_127X63_444, 127, 63, "127x63_444");
}

// ===================== Encode→Decode roundtrip tests =====================

fn roundtrip_test(w: u32, h: u32, format: PixelFormat, ss: Subsampling, label: &str) {
    let bpp: u32 = match format {
        PixelFormat::Rgb => 3,
        PixelFormat::Rgba => 4,
        _ => 3,
    };
    let mut pixels: Vec<u8> = vec![0u8; (w * h * bpp) as usize];
    for i in 0..(w * h) as usize {
        pixels[i * bpp as usize] = ((i * 7) & 255) as u8;
        pixels[i * bpp as usize + 1] = ((i * 13) & 255) as u8;
        pixels[i * bpp as usize + 2] = ((i * 29) & 255) as u8;
        if bpp == 4 {
            pixels[i * 4 + 3] = 255;
        }
    }

    let jpeg: js_sys::Uint8Array = encode(&pixels, w, h, format, 90, ss)
        .unwrap_or_else(|e| panic!("{label}: encode failed: {:?}", e));
    assert!(jpeg.length() > 0, "{label}: empty JPEG");

    let img: DecodedImage = decode(&jpeg.to_vec())
        .unwrap_or_else(|e| panic!("{label}: decode of encoded JPEG failed: {:?}", e));
    assert_eq!(img.width(), w, "{label}: roundtrip width");
    assert_eq!(img.height(), h, "{label}: roundtrip height");
}

#[wasm_bindgen_test]
fn roundtrip_small_sizes() {
    roundtrip_test(1, 1, PixelFormat::Rgb, Subsampling::S444, "1x1_444");
    roundtrip_test(3, 3, PixelFormat::Rgb, Subsampling::S444, "3x3_444");
    roundtrip_test(7, 7, PixelFormat::Rgb, Subsampling::S420, "7x7_420");
    roundtrip_test(8, 8, PixelFormat::Rgb, Subsampling::S420, "8x8_420");
    roundtrip_test(9, 9, PixelFormat::Rgb, Subsampling::S422, "9x9_422");
}

#[wasm_bindgen_test]
fn roundtrip_odd_sizes() {
    roundtrip_test(15, 9, PixelFormat::Rgb, Subsampling::S420, "15x9_420");
    roundtrip_test(17, 33, PixelFormat::Rgb, Subsampling::S444, "17x33_444");
    roundtrip_test(33, 17, PixelFormat::Rgb, Subsampling::S422, "33x17_422");
    roundtrip_test(101, 101, PixelFormat::Rgb, Subsampling::S420, "101x101_420");
    roundtrip_test(127, 63, PixelFormat::Rgb, Subsampling::S420, "127x63_420");
}

#[wasm_bindgen_test]
fn roundtrip_standard_sizes() {
    roundtrip_test(640, 480, PixelFormat::Rgb, Subsampling::S420, "640x480_420");
    roundtrip_test(640, 480, PixelFormat::Rgb, Subsampling::S444, "640x480_444");
    roundtrip_test(640, 480, PixelFormat::Rgb, Subsampling::S422, "640x480_422");
}

#[wasm_bindgen_test]
fn roundtrip_odd_standard_sizes() {
    roundtrip_test(641, 481, PixelFormat::Rgb, Subsampling::S420, "641x481_420");
    roundtrip_test(641, 481, PixelFormat::Rgb, Subsampling::S444, "641x481_444");
    roundtrip_test(641, 481, PixelFormat::Rgb, Subsampling::S422, "641x481_422");
}

#[wasm_bindgen_test]
fn roundtrip_rgba() {
    roundtrip_test(
        100,
        100,
        PixelFormat::Rgba,
        Subsampling::S420,
        "100x100_rgba_420",
    );
    roundtrip_test(
        641,
        481,
        PixelFormat::Rgba,
        Subsampling::S420,
        "641x481_rgba_420",
    );
}

#[wasm_bindgen_test]
fn roundtrip_strip() {
    roundtrip_test(1, 1000, PixelFormat::Rgb, Subsampling::S444, "1x1000_444");
    roundtrip_test(1000, 1, PixelFormat::Rgb, Subsampling::S444, "1000x1_444");
    roundtrip_test(100, 1, PixelFormat::Rgb, Subsampling::S420, "100x1_420");
}

// ===================== Encoder builder tests =====================

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
fn encode_roundtrip() {
    let img: DecodedImage = decode(TEST_JPEG).unwrap();
    let pixels: Vec<u8> = img.data().to_vec();
    let w: u32 = img.width();
    let h: u32 = img.height();

    let jpeg_out: js_sys::Uint8Array =
        encode(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S420).unwrap();
    assert!(jpeg_out.length() > 0);

    let img2: DecodedImage = decode(&jpeg_out.to_vec()).unwrap();
    assert_eq!(img2.width(), w);
    assert_eq!(img2.height(), h);
}

// ===================== Cross-format decode consistency =====================

#[wasm_bindgen_test]
fn decode_rgb_vs_rgba_consistency() {
    let rgb: DecodedImage = decode(CHECKER_640).unwrap();
    let rgba: DecodedImage = decode_to(CHECKER_640, PixelFormat::Rgba).unwrap();

    assert_eq!(rgb.width(), rgba.width());
    assert_eq!(rgb.height(), rgba.height());

    // Verify RGB values match between formats (sample first 100 pixels).
    // Allow ±1 tolerance: RGB and RGBA use different SIMD color conversion
    // paths (RGB interleaves via scalar store, RGBA via shuffle), and the
    // intermediate i16 rounding can differ by 1 LSB.
    let rgb_data: Vec<u8> = rgb.data().to_vec();
    let rgba_data: Vec<u8> = rgba.data().to_vec();
    for i in 0..100 {
        let dr: i32 = rgb_data[i * 3] as i32 - rgba_data[i * 4] as i32;
        let dg: i32 = rgb_data[i * 3 + 1] as i32 - rgba_data[i * 4 + 1] as i32;
        let db: i32 = rgb_data[i * 3 + 2] as i32 - rgba_data[i * 4 + 2] as i32;
        assert!(
            dr.abs() <= 2 && dg.abs() <= 2 && db.abs() <= 2,
            "pixel {i}: R={dr} G={dg} B={db} (rgb={},{},{} rgba={},{},{})",
            rgb_data[i * 3],
            rgb_data[i * 3 + 1],
            rgb_data[i * 3 + 2],
            rgba_data[i * 4],
            rgba_data[i * 4 + 1],
            rgba_data[i * 4 + 2],
        );
        assert_eq!(rgba_data[i * 4 + 3], 255, "Alpha not 255 at pixel {i}");
    }
}
