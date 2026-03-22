use libjpeg_turbo_rs::*;

#[test]
fn scanline_decode_read_all() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    assert_eq!(dec.header().width, 16);
    let mut row: Vec<u8> = vec![0u8; 16 * 3];
    for _ in 0..16 {
        dec.read_scanline(&mut row).unwrap();
    }
    assert_eq!(dec.output_scanline(), 16);
}

#[test]
fn scanline_decode_skip() {
    let pixels: Vec<u8> = vec![128u8; 32 * 32 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    let skipped: usize = dec.skip_scanlines(10).unwrap();
    assert_eq!(skipped, 10);
    assert_eq!(dec.output_scanline(), 10);
}

#[test]
fn scanline_decode_finish_returns_image() {
    let pixels: Vec<u8> = vec![200u8; 8 * 8 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 8, 8, PixelFormat::Rgb, 90, Subsampling::S444).unwrap();
    let dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 8);
    assert_eq!(img.height, 8);
    assert_eq!(img.pixel_format, PixelFormat::Rgb);
}

#[test]
fn scanline_decode_read_past_end_fails() {
    let pixels: Vec<u8> = vec![128u8; 4 * 4 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 4, 4, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    let mut row: Vec<u8> = vec![0u8; 4 * 3];
    // Read all 4 scanlines
    for _ in 0..4 {
        dec.read_scanline(&mut row).unwrap();
    }
    // One more should fail
    let result: Result<()> = dec.read_scanline(&mut row);
    assert!(result.is_err());
}

#[test]
fn scanline_decode_set_output_format() {
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_output_format(PixelFormat::Rgba);
    let mut row: Vec<u8> = vec![0u8; 8 * 4]; // RGBA = 4 bytes per pixel
    dec.read_scanline(&mut row).unwrap();
    assert_eq!(dec.output_scanline(), 1);
}

#[test]
fn scanline_encode_roundtrip() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(16, 16, PixelFormat::Rgb);
    enc.set_quality(75);
    let row: Vec<u8> = vec![128u8; 16 * 3];
    for _ in 0..16 {
        enc.write_scanline(&row).unwrap();
    }
    let jpeg: Vec<u8> = enc.finish().unwrap();
    let img: Image = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn scanline_encode_incomplete_fails() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(8, 8, PixelFormat::Rgb);
    let row: Vec<u8> = vec![128u8; 8 * 3];
    enc.write_scanline(&row).unwrap(); // only 1 of 8
    let result: std::result::Result<Vec<u8>, JpegError> = enc.finish();
    assert!(result.is_err());
}

#[test]
fn scanline_encode_write_past_end_fails() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(4, 4, PixelFormat::Rgb);
    let row: Vec<u8> = vec![128u8; 4 * 3];
    for _ in 0..4 {
        enc.write_scanline(&row).unwrap();
    }
    // One more should fail
    let result: Result<()> = enc.write_scanline(&row);
    assert!(result.is_err());
}

#[test]
fn scanline_encode_next_scanline_tracks() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(8, 8, PixelFormat::Rgb);
    assert_eq!(enc.next_scanline(), 0);
    let row: Vec<u8> = vec![128u8; 8 * 3];
    enc.write_scanline(&row).unwrap();
    assert_eq!(enc.next_scanline(), 1);
    enc.write_scanline(&row).unwrap();
    assert_eq!(enc.next_scanline(), 2);
}

#[test]
fn scanline_encode_set_subsampling() {
    let mut enc: ScanlineEncoder = ScanlineEncoder::new(16, 16, PixelFormat::Rgb);
    enc.set_quality(80);
    enc.set_subsampling(Subsampling::S422);
    let row: Vec<u8> = vec![100u8; 16 * 3];
    for _ in 0..16 {
        enc.write_scanline(&row).unwrap();
    }
    let jpeg: Vec<u8> = enc.finish().unwrap();
    let img: Image = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn scanline_decode_skip_then_read() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    dec.skip_scanlines(8).unwrap();
    assert_eq!(dec.output_scanline(), 8);
    let mut row: Vec<u8> = vec![0u8; 16 * 3];
    // Should be able to read the remaining 8 lines
    for _ in 0..8 {
        dec.read_scanline(&mut row).unwrap();
    }
    assert_eq!(dec.output_scanline(), 16);
}

#[test]
fn scanline_skip_clamped_to_remaining() {
    let pixels: Vec<u8> = vec![128u8; 8 * 8 * 3];
    let jpeg: Vec<u8> = compress(&pixels, 8, 8, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let mut dec: ScanlineDecoder = ScanlineDecoder::new(&jpeg).unwrap();
    // Try to skip more than available
    let skipped: usize = dec.skip_scanlines(100).unwrap();
    assert_eq!(skipped, 8);
    assert_eq!(dec.output_scanline(), 8);
}
