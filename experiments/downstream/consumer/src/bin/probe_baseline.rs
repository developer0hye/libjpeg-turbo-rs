//! Size probe: decode one JPEG to RGB with the published baseline
//! (libjpeg-turbo-rs 0.8.0). See probe_none.rs for how the sizes are read.

fn main() {
    let path: String = std::env::args()
        .nth(1)
        .expect("usage: probe-baseline <file.jpg>");
    let jpeg: Vec<u8> = std::fs::read(&path).expect("read input");
    let image: ljt_baseline::Image =
        ljt_baseline::decompress_to(&jpeg, ljt_baseline::PixelFormat::Rgb).expect("decode");
    let checksum: u64 = image.data.iter().map(|b| *b as u64).sum();
    println!("{}x{} {checksum}", image.width, image.height);
}
