//! Size probe: decode one JPEG to RGB with zune-jpeg directly.
//! See probe_none.rs for how the sizes are read.

use zune_jpeg::zune_core::bytestream::ZCursor;
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;

fn main() {
    let path: String = std::env::args()
        .nth(1)
        .expect("usage: probe-zune <file.jpg>");
    let jpeg: Vec<u8> = std::fs::read(&path).expect("read input");
    let options: DecoderOptions =
        DecoderOptions::default().jpeg_set_out_colorspace(ColorSpace::RGB);
    let mut decoder = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(&jpeg[..]), options);
    let pixels: Vec<u8> = decoder.decode().expect("decode");
    let (width, height) = decoder.dimensions().expect("dimensions after decode");
    let checksum: u64 = pixels.iter().map(|b| *b as u64).sum();
    println!("{width}x{height} {checksum}");
}
