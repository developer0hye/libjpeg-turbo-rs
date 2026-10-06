//! Size probe: decode one JPEG through the candidate's image adapter
//! (`libjpeg-turbo-rs-image`), i.e. candidate + image's core traits.
//! See probe_none.rs for how the sizes are read.

use image::ImageDecoder;

fn main() {
    let path: String = std::env::args()
        .nth(1)
        .expect("usage: probe-adapter <file.jpg>");
    let jpeg: Vec<u8> = std::fs::read(&path).expect("read input");
    let decoder: libjpeg_turbo_rs_image::JpegDecoder =
        libjpeg_turbo_rs_image::JpegDecoder::new(&jpeg).expect("decode");
    let (width, height) = decoder.dimensions();
    let mut pixels: Vec<u8> = vec![0; decoder.total_bytes() as usize];
    decoder.read_image(&mut pixels).expect("read_image");
    let checksum: u64 = pixels.iter().map(|b| *b as u64).sum();
    println!("{width}x{height} {checksum}");
}
