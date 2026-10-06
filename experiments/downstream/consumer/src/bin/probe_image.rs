//! Size probe: decode one JPEG with image's built-in decoder
//! (`image::codecs::jpeg`, zune-jpeg inside). See probe_none.rs for how the
//! sizes are read.

use image::ImageDecoder;

fn main() {
    let path: String = std::env::args()
        .nth(1)
        .expect("usage: probe-image <file.jpg>");
    let jpeg: Vec<u8> = std::fs::read(&path).expect("read input");
    let decoder = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&jpeg[..]))
        .expect("image decoder");
    let (width, height) = decoder.dimensions();
    let mut pixels: Vec<u8> = vec![0; decoder.total_bytes() as usize];
    decoder.read_image(&mut pixels).expect("read_image");
    let checksum: u64 = pixels.iter().map(|b| *b as u64).sum();
    println!("{width}x{height} {checksum}");
}
