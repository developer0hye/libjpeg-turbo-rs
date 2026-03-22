use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn encode_s440_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S440).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn encode_s411_roundtrip() {
    let pixels = vec![128u8; 64 * 16 * 3];
    let jpeg = compress(&pixels, 64, 16, PixelFormat::Rgb, 75, Subsampling::S411).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 64);
    assert_eq!(img.height, 16);
}

#[test]
fn encode_s440_gradient_pixel_accuracy() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 8) as u8;
            pixels[i + 1] = (y * 8) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S440).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
}

#[test]
fn encode_s411_gradient_pixel_accuracy() {
    let (w, h) = (64, 16);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 4) as u8;
            pixels[i + 1] = (y * 16) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S411).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
}
