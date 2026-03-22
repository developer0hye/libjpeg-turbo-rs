/// Test 4:1:0 (H=4, V=2) subsampling decode support.
///
/// 4:1:0 is a rare subsampling mode where luma uses 4x2 sampling and chroma
/// uses 1x1 sampling (upsampling factor 4 horizontally, 2 vertically).
/// This test constructs a JPEG with 4:1:0 sampling factors by directly
/// manipulating the SOF header, then verifies successful decode.

/// Helper: create a minimal valid JPEG with custom sampling factors in the SOF0 header.
/// We start from a standard 4:2:0 JPEG and patch the SOF0 component definitions.
fn make_jpeg_with_410_sampling() -> Vec<u8> {
    // Start with a real JPEG and manually construct one with 4:1:0 sampling.
    // We use the encoder to create a baseline image, then patch the SOF marker.
    use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

    // Encode a 32x32 image at 4:2:0 (H=2, V=2 for luma).
    let width: usize = 32;
    let height: usize = 32;
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let idx: usize = (y * width + x) * 3;
            pixels[idx] = (x * 8) as u8; // R gradient
            pixels[idx + 1] = (y * 8) as u8; // G gradient
            pixels[idx + 2] = 128; // B constant
        }
    }

    let mut jpeg: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap();

    // Find SOF0 marker (0xFF 0xC0) and patch sampling factors.
    // SOF0 layout: FF C0 [len:2] [precision:1] [height:2] [width:2] [nf:1]
    //   then per component: [id:1] [h_samp<<4 | v_samp:1] [qt:1]
    let sof_pos: usize = find_marker(&jpeg, 0xC0).expect("SOF0 marker not found");
    // Skip marker (2 bytes) + length (2 bytes) + precision (1) + height (2) + width (2) + nf (1) = 10 bytes
    let comp_start: usize = sof_pos + 2 + 2 + 1 + 2 + 2 + 1;

    // Component 0 (Y): change from 2x2 to 4x2
    // Current: sampling byte = (2<<4)|2 = 0x22
    // Desired: sampling byte = (4<<4)|2 = 0x42
    assert_eq!(
        jpeg[comp_start + 1],
        0x22,
        "expected Y component with 2x2 sampling"
    );
    jpeg[comp_start + 1] = 0x42; // 4x2 for Y

    // Components 1,2 (Cb, Cr) stay at 1x1 (0x11) -- no change needed.

    jpeg
}

/// Find a JPEG marker by code. Returns offset of the marker (at 0xFF byte).
fn find_marker(data: &[u8], code: u8) -> Option<usize> {
    let mut i: usize = 0;
    while i + 1 < data.len() {
        if data[i] == 0xFF && data[i + 1] == code {
            return Some(i);
        }
        i += 1;
    }
    None
}

#[test]
fn decode_410_subsampling_produces_correct_dimensions() {
    let jpeg: Vec<u8> = make_jpeg_with_410_sampling();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    let image = decoder.decode_image().unwrap();

    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.pixel_format, libjpeg_turbo_rs::PixelFormat::Rgb);
    assert_eq!(image.data.len(), 32 * 32 * 3);
}

#[test]
fn decode_410_subsampling_produces_plausible_pixels() {
    let jpeg: Vec<u8> = make_jpeg_with_410_sampling();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    let image = decoder.decode_image().unwrap();

    // Verify pixels are not all zeros (i.e., decode actually produced content).
    let nonzero: usize = image.data.iter().filter(|&&b| b != 0).count();
    assert!(
        nonzero > image.data.len() / 4,
        "expected non-trivial pixel data, got {} nonzero out of {}",
        nonzero,
        image.data.len()
    );
}

#[test]
fn decode_410_subsampling_with_fast_upsample() {
    let jpeg: Vec<u8> = make_jpeg_with_410_sampling();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.set_output_format(libjpeg_turbo_rs::PixelFormat::Rgb);
    decoder.set_fast_upsample(true);
    let image = decoder.decode_image().unwrap();

    assert_eq!(image.width, 32);
    assert_eq!(image.height, 32);
    assert_eq!(image.data.len(), 32 * 32 * 3);
}
