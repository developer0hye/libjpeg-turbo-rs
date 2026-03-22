use libjpeg_turbo_rs::common::types::Subsampling;
use libjpeg_turbo_rs::precision::{
    compress_12bit, compress_16bit, decompress_12bit, decompress_16bit, Image12, Image16,
};

#[test]
fn roundtrip_12bit_grayscale_quality100() {
    let width: usize = 16;
    let height: usize = 16;
    let nc: usize = 1;
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 16) as i16);
        }
    }
    let jpeg = compress_12bit(&pixels, width, height, nc, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data.len(), width * height);
    let max_diff: i16 = pixels
        .iter()
        .zip(img.data.iter())
        .map(|(a, b)| (*a - *b).abs())
        .max()
        .unwrap_or(0);
    assert!(
        max_diff <= 8,
        "12-bit q100 roundtrip max diff {} exceeds tolerance",
        max_diff
    );
}

#[test]
fn roundtrip_12bit_grayscale_lower_quality() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        pixels.push((i as i16 * 50) % 4096);
    }
    let jpeg = compress_12bit(&pixels, width, height, 1, 50, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    for &val in &img.data {
        assert!(val >= 0 && val <= 4095, "12-bit value {} out of range", val);
    }
}

#[test]
fn roundtrip_12bit_three_component() {
    let width: usize = 16;
    let height: usize = 16;
    let nc: usize = 3;
    let mut pixels: Vec<i16> = Vec::with_capacity(width * height * nc);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 16) as i16);
            pixels.push((x * 256) as i16);
            pixels.push((y * 256) as i16);
        }
    }
    let jpeg = compress_12bit(&pixels, width, height, nc, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data.len(), width * height * nc);
}

#[test]
fn verify_12bit_sof_precision() {
    let pixels: Vec<i16> = vec![2048i16; 64];
    let jpeg = compress_12bit(&pixels, 8, 8, 1, 90, Subsampling::S444).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]);
    assert!(sof_pos.is_some(), "SOF0 marker not found");
    assert_eq!(jpeg[sof_pos.unwrap() + 4], 12, "SOF precision should be 12");
}

#[test]
fn roundtrip_16bit_lossless_grayscale() {
    let width: usize = 16;
    let height: usize = 16;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 256) as u16);
        }
    }
    let jpeg = compress_16bit(&pixels, width, height, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.data, pixels, "16-bit lossless must be exact");
}

#[test]
fn roundtrip_16bit_lossless_three_component() {
    let width: usize = 16;
    let height: usize = 16;
    let nc: usize = 3;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height * nc);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((y * width + x) * 256) as u16);
            pixels.push(((x * 512) % 65536) as u16);
            pixels.push(((y * 512) % 65536) as u16);
        }
    }
    let jpeg = compress_16bit(&pixels, width, height, nc, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(img.num_components, nc);
    assert_eq!(img.data, pixels, "16-bit lossless 3-comp must be exact");
}

#[test]
fn roundtrip_16bit_lossless_all_predictors() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        pixels.push((i as u16).wrapping_mul(1000));
    }
    for predictor in 1u8..=7 {
        let jpeg = compress_16bit(&pixels, width, height, 1, predictor, 0)
            .unwrap_or_else(|e| panic!("predictor {} failed: {}", predictor, e));
        let img = decompress_16bit(&jpeg)
            .unwrap_or_else(|e| panic!("decompress predictor {} failed: {}", predictor, e));
        assert_eq!(
            img.data, pixels,
            "predictor {} roundtrip must be exact",
            predictor
        );
    }
}

#[test]
fn roundtrip_16bit_lossless_with_point_transform() {
    let width: usize = 8;
    let height: usize = 8;
    let mut pixels: Vec<u16> = Vec::with_capacity(width * height);
    for i in 0..(width * height) {
        pixels.push((i as u16).wrapping_mul(1024) & 0xFFFC);
    }
    let jpeg = compress_16bit(&pixels, width, height, 1, 1, 2).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    for (orig, decoded) in pixels.iter().zip(img.data.iter()) {
        let expected = (orig >> 2) << 2;
        assert_eq!(
            *decoded, expected,
            "pt=2: orig={}, expected={}, got={}",
            orig, expected, decoded
        );
    }
}

#[test]
fn verify_16bit_sof_precision() {
    let pixels: Vec<u16> = vec![32768u16; 64];
    let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0).unwrap();
    let sof_pos = jpeg.windows(2).position(|w| w == [0xFF, 0xC3]);
    assert!(sof_pos.is_some(), "SOF3 marker not found");
    assert_eq!(jpeg[sof_pos.unwrap() + 4], 16, "SOF precision should be 16");
}

#[test]
fn error_16bit_invalid_predictor() {
    let pixels: Vec<u16> = vec![100u16; 64];
    assert!(compress_16bit(&pixels, 8, 8, 1, 0, 0).is_err());
    assert!(compress_16bit(&pixels, 8, 8, 1, 8, 0).is_err());
}

#[test]
fn roundtrip_12bit_edge_values() {
    let mut pixels: Vec<i16> = vec![0i16; 64];
    pixels[0] = 0;
    pixels[1] = 4095;
    pixels[2] = 2048;
    let jpeg = compress_12bit(&pixels, 8, 8, 1, 100, Subsampling::S444).unwrap();
    let img = decompress_12bit(&jpeg).unwrap();
    for &val in &img.data {
        assert!(val >= 0 && val <= 4095, "12-bit value {} out of range", val);
    }
}

#[test]
fn roundtrip_16bit_full_range() {
    let mut pixels: Vec<u16> = Vec::with_capacity(64);
    pixels.push(0);
    pixels.push(65535);
    pixels.push(32768);
    pixels.push(1);
    pixels.push(65534);
    for i in 5..64 {
        pixels.push((i as u16).wrapping_mul(997));
    }
    let jpeg = compress_16bit(&pixels, 8, 8, 1, 1, 0).unwrap();
    let img = decompress_16bit(&jpeg).unwrap();
    assert_eq!(
        img.data, pixels,
        "16-bit full-range roundtrip must be exact"
    );
}
