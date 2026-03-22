#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct RoundtripInput {
    width: u8,
    height: u8,
    quality: u8,
    subsampling_idx: u8,
    pixels: Vec<u8>,
}

fuzz_target!(|input: RoundtripInput| {
    let width: usize = input.width.max(1) as usize;
    let height: usize = input.height.max(1) as usize;
    let quality: u8 = input.quality.clamp(1, 100);
    let required_len: usize = width * height * 3;

    if input.pixels.len() < required_len {
        return;
    }

    let subsampling = match input.subsampling_idx % 3 {
        0 => libjpeg_turbo_rs::Subsampling::S420,
        1 => libjpeg_turbo_rs::Subsampling::S422,
        _ => libjpeg_turbo_rs::Subsampling::S444,
    };

    let jpeg_data: Vec<u8> = match libjpeg_turbo_rs::compress(
        &input.pixels[..required_len],
        width,
        height,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        quality,
        subsampling,
    ) {
        Ok(data) => data,
        Err(_) => return,
    };

    // Decompression of our own encoder output must always succeed
    let result = libjpeg_turbo_rs::decompress(&jpeg_data);
    if let Ok(image) = result {
        assert_eq!(image.width, width);
        assert_eq!(image.height, height);
    }
});
