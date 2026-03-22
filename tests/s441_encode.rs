use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn encode_s441_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn encode_s441_gradient() {
    let (w, h) = (16, 32);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = (x * 16) as u8;
            pixels[i + 1] = (y * 8) as u8;
            pixels[i + 2] = 128;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 95, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data.len(), w * h * 3);
}

#[test]
fn encode_s441_non_mcu_aligned() {
    // Image height not a multiple of 32 (MCU height for S441)
    let (w, h) = (8, 20);
    let pixels = vec![100u8; w * h * 3];
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 80, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
}

#[test]
fn encode_s441_large_image() {
    // Larger image to exercise multi-MCU rows
    let (w, h) = (24, 64);
    let mut pixels = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            pixels[i] = ((x * 10) % 256) as u8;
            pixels[i + 1] = ((y * 4) % 256) as u8;
            pixels[i + 2] = 200;
        }
    }
    let jpeg = compress(&pixels, w, h, PixelFormat::Rgb, 90, Subsampling::S441).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
    assert_eq!(img.data.len(), w * h * 3);
}
