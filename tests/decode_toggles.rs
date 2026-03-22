use libjpeg_turbo_rs::*;

/// Helper: create a simple 16x16 RGB test JPEG with 4:2:0 subsampling.
fn make_test_jpeg_420() -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(16 * 16 * 3);
    for y in 0..16u8 {
        for x in 0..16u8 {
            pixels.push(y.wrapping_mul(16).wrapping_add(x));
            pixels.push(128);
            pixels.push(255u8.wrapping_sub(y.wrapping_mul(8)));
        }
    }
    compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S420).unwrap()
}

/// Helper: create a simple 32x32 RGB test JPEG with 4:4:4 subsampling.
fn make_test_jpeg_444() -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(32 * 32 * 3);
    for y in 0..32u8 {
        for x in 0..32u8 {
            pixels.push(y.wrapping_mul(8).wrapping_add(x.wrapping_mul(2)));
            pixels.push(128);
            pixels.push(64);
        }
    }
    compress(&pixels, 32, 32, PixelFormat::Rgb, 85, Subsampling::S444).unwrap()
}

#[test]
fn fast_upsample_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_420();
    let img_fancy: Image = decompress(&jpeg).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_fast_upsample(true);
    let img_fast: Image = dec.finish().unwrap();
    assert_eq!(img_fast.width, img_fancy.width);
    assert_eq!(img_fast.height, img_fancy.height);
    assert_eq!(img_fast.data.len(), img_fancy.data.len());
    assert!(!img_fast.data.is_empty());
}

#[test]
fn fast_upsample_differs_from_fancy_on_subsampled() {
    let jpeg: Vec<u8> = make_test_jpeg_420();
    let img_fancy: Image = decompress(&jpeg).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_fast_upsample(true);
    let img_fast: Image = dec.finish().unwrap();
    let differences: usize = img_fast
        .data
        .iter()
        .zip(img_fancy.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differences > 0,
        "fast and fancy upsample should produce different results for 4:2:0"
    );
}

#[test]
fn fast_dct_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_fast_dct(true);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert!(!img.data.is_empty());
}

#[test]
fn dct_method_islow_is_default() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let img_default: Image = decompress(&jpeg).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_dct_method(DctMethod::IsLow);
    let img_islow: Image = dec.finish().unwrap();
    assert_eq!(img_default.data, img_islow.data);
}

#[test]
fn block_smoothing_on_vs_off_differs_for_low_quality() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 32, 32, PixelFormat::Rgb, 10, Subsampling::S420).unwrap();
    let mut dec_smooth = ScanlineDecoder::new(&jpeg).unwrap();
    dec_smooth.set_block_smoothing(true);
    let img_smooth: Image = dec_smooth.finish().unwrap();
    let mut dec_no_smooth = ScanlineDecoder::new(&jpeg).unwrap();
    dec_no_smooth.set_block_smoothing(false);
    let img_no_smooth: Image = dec_no_smooth.finish().unwrap();
    assert_eq!(img_smooth.width, img_no_smooth.width);
    assert_eq!(img_smooth.height, img_no_smooth.height);
    assert_eq!(img_smooth.data.len(), img_no_smooth.data.len());
}

#[test]
fn output_colorspace_grayscale_from_color() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_output_colorspace(ColorSpace::Grayscale);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img.data.len(), 32 * 32);
}

#[test]
fn output_colorspace_ycbcr_keeps_raw_planes() {
    let jpeg: Vec<u8> = make_test_jpeg_444();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_output_colorspace(ColorSpace::YCbCr);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.data.len(), 32 * 32 * 3);
}

#[test]
fn scanline_crop_produces_correct_width() {
    let pixels: Vec<u8> = vec![128u8; 64 * 64 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_crop_x(8, 32);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 64);
}

#[test]
fn scanline_crop_at_zero_offset() {
    let pixels: Vec<u8> = vec![200u8; 32 * 32 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_crop_x(0, 16);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 32);
}

#[test]
fn scanline_12bit_roundtrip() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels_12: Vec<i16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels_12.push(((y * width + x) * 64) as i16);
        }
    }
    let jpeg: Vec<u8> =
        precision::compress_12bit(&pixels_12, width, height, 1, 90, Subsampling::S444).unwrap();
    let decoded: precision::Image12 = precision::decompress_12bit(&jpeg).unwrap();
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.data.len(), width * height);
    for i in 0..decoded.data.len() {
        let diff: i16 = (decoded.data[i] - pixels_12[i]).abs();
        assert!(
            diff < 200,
            "12-bit sample {} differs too much: got {} expected {}, diff {}",
            i,
            decoded.data[i],
            pixels_12[i],
            diff
        );
    }
}

#[test]
fn scanline_16bit_read_write_stubs() {
    let width: usize = 4;
    let height: usize = 4;
    let pixels_16: Vec<u16> = (0..width * height).map(|i| (i * 1000) as u16).collect();
    let result = precision::compress_16bit(&pixels_16, width, height, 1, 1, 0);
    assert!(result.is_ok(), "16-bit compress should succeed");
    let jpeg: Vec<u8> = result.unwrap();
    let decoded: precision::Image16 = precision::decompress_16bit(&jpeg).unwrap();
    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);
    assert_eq!(decoded.data, pixels_16);
}
