//! Size probe: decode one JPEG to RGB with the candidate libjpeg-turbo-rs.
//! See probe_none.rs for how the sizes are read.

fn main() {
    let path: String = std::env::args()
        .nth(1)
        .expect("usage: probe-candidate <file.jpg>");
    let jpeg: Vec<u8> = std::fs::read(&path).expect("read input");
    let image: ljt_candidate::Image =
        ljt_candidate::decompress_to(&jpeg, ljt_candidate::PixelFormat::Rgb).expect("decode");
    let checksum: u64 = image.data.iter().map(|b| *b as u64).sum();
    println!("{}x{} {checksum}", image.width, image.height);
}
