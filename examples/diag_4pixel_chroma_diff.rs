//! P2-11 diagnostic: characterize the byte-divergence between Rust encode
//! and cjpeg for 4-pixel chroma subsampling factors (TJSAMP_411, _441,
//! _410, _24) in baseline and progressive modes.
//!
//! Reads `references/libjpeg-turbo/testimages/testorig_full.ppm` (227x149,
//! intentionally not a multiple of 4 so right-edge expansion is exercised),
//! encodes via both Rust and stock `cjpeg` for the matrix:
//!
//!   subsampling ∈ {S411, S441, S410, S24}
//!   progressive ∈ {false, true}
//!   quality     = default (75)
//!   optimize    = false (per the test matrix gating)
//!   arithmetic  = false
//!
//! Reports per case: byte-equal? first diverging byte offset, total bytes,
//! decoded pixel max/mean diff via stock `djpeg`. This pinpoints whether
//! the divergence is purely entropy-coding (decoded pixels match) or
//! a real downsample/quantize/DCT divergence (decoded pixels differ).
//!
//! Run with:
//!   cargo run --release --example diag_4pixel_chroma_diff

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{Decoder, Encoder, PixelFormat, Subsampling};

fn locate_cjpeg() -> Option<PathBuf> {
    for path in [
        "/opt/homebrew/bin/cjpeg",
        "/usr/local/bin/cjpeg",
        "/usr/bin/cjpeg",
    ] {
        let p: PathBuf = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn locate_djpeg() -> Option<PathBuf> {
    for path in [
        "/opt/homebrew/bin/djpeg",
        "/usr/local/bin/djpeg",
        "/usr/bin/djpeg",
    ] {
        let p: PathBuf = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn parse_ppm(path: &Path) -> (usize, usize, Vec<u8>) {
    let bytes: Vec<u8> = std::fs::read(path).expect("read ppm");
    // Skip P6 header: 3 whitespace-separated tokens then exactly one whitespace
    // (per Netpbm spec). Walk by hand to avoid pulling in a parser crate.
    let mut i: usize = 0;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 && i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let start: usize = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            tokens.push(String::from_utf8(bytes[start..i].to_vec()).expect("ascii"));
        }
    }
    assert_eq!(tokens[0], "P6", "expected P6 PPM");
    let width: usize = tokens[1].parse().expect("width");
    let height: usize = tokens[2].parse().expect("height");
    let maxval: usize = tokens[3].parse().expect("maxval");
    assert_eq!(maxval, 255, "expected 8-bit PPM");
    // Per spec, exactly ONE whitespace separator after maxval, then raster.
    i += 1;
    let pixels: Vec<u8> = bytes[i..].to_vec();
    (width, height, pixels)
}

fn encode_rust(
    pixels: &[u8],
    width: usize,
    height: usize,
    subsampling: Subsampling,
    progressive: bool,
) -> Vec<u8> {
    let enc: Encoder<'_> = Encoder::new(pixels, width, height, PixelFormat::Rgb)
        .fancy_downsampling(false)
        .subsampling(subsampling)
        .progressive(progressive);
    enc.encode().expect("Rust encode")
}

fn encode_cjpeg(cjpeg: &Path, ppm_path: &Path, samp_arg: &str, progressive: bool) -> Vec<u8> {
    let mut cmd: Command = Command::new(cjpeg);
    if progressive {
        cmd.arg("-p");
    }
    cmd.arg("-sa").arg(samp_arg).arg(ppm_path);
    let out = cmd.output().expect("run cjpeg");
    assert!(out.status.success(), "cjpeg failed: {:?}", out);
    out.stdout
}

fn decode_djpeg_to_ppm(djpeg: &Path, jpeg: &[u8]) -> Vec<u8> {
    use std::io::Write;
    use std::process::Stdio;
    let mut child = Command::new(djpeg)
        .arg("-pnm")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn djpeg");
    child
        .stdin
        .as_mut()
        .expect("djpeg stdin")
        .write_all(jpeg)
        .expect("write to djpeg stdin");
    let out = child.wait_with_output().expect("djpeg output");
    assert!(out.status.success(), "djpeg failed: {:?}", out);
    out.stdout
}

fn pixel_stats(a: &[u8], b: &[u8]) -> (u32, f64) {
    assert_eq!(a.len(), b.len(), "decoded sizes differ");
    let mut max_d: u32 = 0;
    let mut sum_d: u64 = 0;
    for i in 0..a.len() {
        let d: u32 = (a[i] as i32 - b[i] as i32).unsigned_abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d as u64;
    }
    let mean: f64 = sum_d as f64 / a.len() as f64;
    (max_d, mean)
}

fn first_byte_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    let n: usize = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            return Some(i);
        }
    }
    if a.len() != b.len() {
        Some(n)
    } else {
        None
    }
}

fn count_byte_diffs(a: &[u8], b: &[u8]) -> usize {
    let n: usize = a.len().min(b.len());
    let mut k: usize = 0;
    for i in 0..n {
        if a[i] != b[i] {
            k += 1;
        }
    }
    k + a.len().abs_diff(b.len())
}

fn main() {
    let cjpeg: PathBuf = match locate_cjpeg() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found in /opt/homebrew/bin, /usr/local/bin, /usr/bin");
            return;
        }
    };
    let djpeg: PathBuf = match locate_djpeg() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let ppm_path: PathBuf = PathBuf::from("references/libjpeg-turbo/testimages/testorig_full.ppm");
    if !ppm_path.exists() {
        eprintln!("SKIP: {:?} not found", ppm_path);
        return;
    }

    let (width, height, pixels) = parse_ppm(&ppm_path);
    println!(
        "fixture: {:?}  ({}x{}, {} bytes raw RGB)",
        ppm_path,
        width,
        height,
        pixels.len()
    );

    // (label, Subsampling, cjpeg -sa argument)
    let subs: [(&str, Subsampling, &str); 4] = [
        ("S411", Subsampling::S411, "4x1"),
        ("S441", Subsampling::S441, "1x4"),
        ("S410", Subsampling::S410, "4x2"),
        ("S24", Subsampling::S24, "2x4"),
    ];

    println!(
        "\n{:>4}  {:>10}  {:>5}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10}",
        "samp", "mode", "match", "rust_bytes", "c_bytes", "first_d", "diff_n", "px_max", "px_mean"
    );
    println!("{}", "-".repeat(96));

    for (sub_label, sub, samp_arg) in subs {
        for (mode_label, progressive) in [("baseline", false), ("progressive", true)] {
            let rust_jpg: Vec<u8> = encode_rust(&pixels, width, height, sub, progressive);
            let c_jpg: Vec<u8> = encode_cjpeg(&cjpeg, &ppm_path, samp_arg, progressive);

            let byte_match: bool = rust_jpg == c_jpg;
            let first_d: String = match first_byte_diff(&rust_jpg, &c_jpg) {
                Some(o) => format!("0x{:x}", o),
                None => "-".to_string(),
            };
            let diff_n: usize = count_byte_diffs(&rust_jpg, &c_jpg);

            // Decoded-pixel comparison via djpeg.
            let rust_pnm: Vec<u8> = decode_djpeg_to_ppm(&djpeg, &rust_jpg);
            let c_pnm: Vec<u8> = decode_djpeg_to_ppm(&djpeg, &c_jpg);
            // Skip the PPM header to get raw RGB pixels.
            let (_, _, rust_px) = parse_ppm_from_bytes(&rust_pnm);
            let (_, _, c_px) = parse_ppm_from_bytes(&c_pnm);
            let (px_max, px_mean) = pixel_stats(&rust_px, &c_px);

            println!(
                "{:>4}  {:>10}  {:>5}  {:>10}  {:>10}  {:>8}  {:>8}  {:>10}  {:>10.4}",
                sub_label,
                mode_label,
                if byte_match { "Y" } else { "N" },
                rust_jpg.len(),
                c_jpg.len(),
                first_d,
                diff_n,
                px_max,
                px_mean,
            );

            // Cross-decode: feed Rust output to OUR decoder and to djpeg —
            // both must produce the same pixels (byte-identical or close).
            let rust_self_decode = Decoder::decode(&rust_jpg).expect("Rust self-decode");
            let (rust_self_max, rust_self_mean) = pixel_stats(&rust_self_decode.data, &rust_px);
            if rust_self_max > 1 {
                println!(
                    "       NOTE: Rust self-decode of own output diverges from djpeg by max={} mean={:.4}",
                    rust_self_max, rust_self_mean
                );
            }
        }
    }
}

fn parse_ppm_from_bytes(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut i: usize = 0;
    let mut tokens: Vec<String> = Vec::new();
    while tokens.len() < 4 && i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let start: usize = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if start < i {
            tokens.push(String::from_utf8(bytes[start..i].to_vec()).expect("ascii"));
        }
    }
    assert_eq!(tokens[0], "P6");
    let w: usize = tokens[1].parse().expect("w");
    let h: usize = tokens[2].parse().expect("h");
    i += 1;
    (w, h, bytes[i..].to_vec())
}
