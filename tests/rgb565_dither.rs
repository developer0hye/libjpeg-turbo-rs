/// Test RGB565 dithered decode support.
///
/// Verifies that decoding a JPEG to RGB565 with dithering enabled produces
/// different output than without dithering (because the ordered dither pattern
/// adds noise to reduce quantization banding).
use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

/// Helper: create a JPEG with smooth gradients (to make dithering visible).
fn make_gradient_jpeg() -> Vec<u8> {
    let width: usize = 64;
    let height: usize = 64;
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * 3;
            // Smooth gradient that will show quantization banding in RGB565.
            let val: u8 = ((x * 4) % 256) as u8;
            pixels[idx] = val;
            pixels[idx + 1] = val;
            pixels[idx + 2] = val;
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        95,
        Subsampling::S444,
    )
    .unwrap()
}

#[test]
fn rgb565_decode_without_dither() {
    let jpeg: Vec<u8> = make_gradient_jpeg();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(PixelFormat::Rgb565);
    let image = decoder.decode_image().unwrap();

    assert_eq!(image.width, 64);
    assert_eq!(image.height, 64);
    assert_eq!(image.pixel_format, PixelFormat::Rgb565);
    assert_eq!(image.data.len(), 64 * 64 * 2);
}

#[test]
fn rgb565_decode_with_dither_produces_different_output() {
    let jpeg: Vec<u8> = make_gradient_jpeg();

    // Decode without dithering
    let mut decoder_nodither = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder_nodither.set_output_format(PixelFormat::Rgb565);
    let image_nodither = decoder_nodither.decode_image().unwrap();

    // Decode with dithering
    let mut decoder_dither = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder_dither.set_output_format(PixelFormat::Rgb565);
    decoder_dither.set_dither_565(true);
    let image_dither = decoder_dither.decode_image().unwrap();

    assert_eq!(image_dither.width, 64);
    assert_eq!(image_dither.height, 64);
    assert_eq!(image_dither.pixel_format, PixelFormat::Rgb565);
    assert_eq!(image_dither.data.len(), 64 * 64 * 2);

    // The dithered output should differ from undithered because the dither pattern
    // perturbs RGB values before truncation.
    assert_ne!(
        image_nodither.data, image_dither.data,
        "dithered and undithered RGB565 output should differ for gradient images"
    );
}

#[test]
fn rgb565_dither_output_is_deterministic() {
    let jpeg: Vec<u8> = make_gradient_jpeg();

    let mut decoder1 = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder1.set_output_format(PixelFormat::Rgb565);
    decoder1.set_dither_565(true);
    let image1 = decoder1.decode_image().unwrap();

    let mut decoder2 = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder2.set_output_format(PixelFormat::Rgb565);
    decoder2.set_dither_565(true);
    let image2 = decoder2.decode_image().unwrap();

    assert_eq!(
        image1.data, image2.data,
        "dithered output should be deterministic (same input -> same output)"
    );
}

#[test]
fn rgb565_dither_values_are_valid() {
    let jpeg: Vec<u8> = make_gradient_jpeg();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(PixelFormat::Rgb565);
    decoder.set_dither_565(true);
    let image = decoder.decode_image().unwrap();

    // Every pair of bytes should form a valid RGB565 pixel (always true for u16,
    // but verify the data length is correct).
    assert_eq!(image.data.len() % 2, 0);
    let pixel_count: usize = image.data.len() / 2;
    assert_eq!(pixel_count, 64 * 64);
}
