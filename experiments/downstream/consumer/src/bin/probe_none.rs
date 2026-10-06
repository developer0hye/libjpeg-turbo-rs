//! Size probe: the same program shape as the other probes with no JPEG code,
//! so `size(probe-X) - size(probe-none)` is backend X's contribution to a
//! stock release binary (std, argument parsing and I/O cancel out).

fn main() {
    let path: String = std::env::args().nth(1).expect("usage: probe-none <file>");
    let bytes: Vec<u8> = std::fs::read(&path).expect("read input");
    let checksum: u64 = bytes.iter().map(|b| *b as u64).sum();
    println!("{} {checksum}", bytes.len());
}
