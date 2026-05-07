//! P3-4 diagnostic: characterise byte-divergence between Rust progressive
//! transform output and stock `jpegtran -progressive -copy all <op>` for
//! 4-pixel chroma subsampling factors (S411 / S441 / S410 / S24).
//!
//! Mirror of `diag_4pixel_chroma_diff.rs` (encoder-side P2-11 closure) but
//! for the *transform* writer at `src/api/coefficient.rs::write_coefficients_progressive`.
//!
//! Pipeline (each row in the output table):
//!   1. Source baseline JPEG is built either by Rust (`Encoder::encode`) or
//!      by stock `cjpeg`, at the target sampling factor and baseline
//!      Huffman.
//!   2. Rust runs `transform_jpeg_with_options(..., progressive = true,
//!      copy_markers = All)`.
//!   3. Stock `jpegtran -progressive -copy all <op-flag>` runs against the
//!      same baseline.
//!   4. Bytes are compared. On mismatch, decoded pixels (via `djpeg`) are
//!      compared to determine whether divergence is purely entropy-coding
//!      or a real coefficient mismatch.
//!
//! Run with:
//!   cargo run --release --example diag_4pixel_chroma_transform_diff

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, Encoder, MarkerCopyMode, PixelFormat, Subsampling, TransformOp,
    TransformOptions,
};

fn locate(name: &str) -> Option<PathBuf> {
    for dir in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
        let p: PathBuf = PathBuf::from(dir).join(name);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn parse_ppm(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
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
    let w: usize = tokens[1].parse().unwrap();
    let h: usize = tokens[2].parse().unwrap();
    i += 1;
    (w, h, bytes[i..].to_vec())
}

fn jpegtran_op_flag(op: TransformOp) -> Vec<&'static str> {
    match op {
        TransformOp::None => vec![],
        TransformOp::HFlip => vec!["-flip", "horizontal"],
        TransformOp::VFlip => vec!["-flip", "vertical"],
        TransformOp::Transpose => vec!["-transpose"],
        TransformOp::Transverse => vec!["-transverse"],
        TransformOp::Rot90 => vec!["-rotate", "90"],
        TransformOp::Rot180 => vec!["-rotate", "180"],
        TransformOp::Rot270 => vec!["-rotate", "270"],
    }
}

fn op_label(op: TransformOp) -> &'static str {
    match op {
        TransformOp::None => "none",
        TransformOp::HFlip => "hflip",
        TransformOp::VFlip => "vflip",
        TransformOp::Transpose => "trans",
        TransformOp::Transverse => "trnv",
        TransformOp::Rot90 => "r90",
        TransformOp::Rot180 => "r180",
        TransformOp::Rot270 => "r270",
    }
}

fn run_jpegtran(jpegtran: &Path, op: TransformOp, input: &[u8]) -> Vec<u8> {
    let mut cmd = Command::new(jpegtran);
    cmd.arg("-progressive").arg("-copy").arg("all");
    for arg in jpegtran_op_flag(op) {
        cmd.arg(arg);
    }
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn jpegtran");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(input)
        .expect("write stdin");
    let out = child.wait_with_output().expect("jpegtran output");
    assert!(out.status.success(), "jpegtran failed: {:?}", out);
    out.stdout
}

fn run_djpeg(djpeg: &Path, jpeg: &[u8]) -> Vec<u8> {
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
        .unwrap()
        .write_all(jpeg)
        .expect("write");
    let out = child.wait_with_output().expect("djpeg out");
    assert!(out.status.success(), "djpeg failed: {:?}", out);
    out.stdout
}

fn pixel_stats(a: &[u8], b: &[u8]) -> (u32, f64) {
    if a.len() != b.len() {
        return (u32::MAX, f64::INFINITY);
    }
    let mut max_d: u32 = 0;
    let mut sum_d: u64 = 0;
    for i in 0..a.len() {
        let d: u32 = (a[i] as i32 - b[i] as i32).unsigned_abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d as u64;
    }
    (max_d, sum_d as f64 / a.len() as f64)
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

fn synth_rgb(width: usize, height: usize, seed: u32) -> Vec<u8> {
    // Procedural RGB content that exercises both luma and chroma planes.
    // Combines a smooth gradient with a higher-frequency pattern; the seed
    // perturbs phase so different fixtures don't all collapse to identical
    // chroma after subsampling.
    let mut out: Vec<u8> = vec![0u8; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let r = (((x.wrapping_add(seed as usize) * 7) ^ (y * 11)) & 0xff) as u8;
            let g = (((x * 13).wrapping_add(y.wrapping_mul(5)) ^ (seed as usize * 3)) & 0xff) as u8;
            let b = (((x ^ y).wrapping_mul(17).wrapping_add(seed as usize)) & 0xff) as u8;
            out[(y * width + x) * 3] = r;
            out[(y * width + x) * 3 + 1] = g;
            out[(y * width + x) * 3 + 2] = b;
        }
    }
    out
}

fn write_ppm(width: usize, height: usize, pixels: &[u8], path: &Path) {
    let mut data: Vec<u8> = Vec::with_capacity(pixels.len() + 32);
    data.extend_from_slice(format!("P6\n{} {}\n255\n", width, height).as_bytes());
    data.extend_from_slice(pixels);
    std::fs::write(path, &data).expect("write ppm");
}

fn cjpeg_encode(cjpeg: &Path, ppm_path: &Path, samp_arg: &str) -> Vec<u8> {
    let out = Command::new(cjpeg)
        .arg("-sa")
        .arg(samp_arg)
        .arg(ppm_path)
        .output()
        .expect("cjpeg");
    assert!(out.status.success(), "cjpeg: {:?}", out);
    out.stdout
}

fn main() {
    let cjpeg = locate("cjpeg").expect("cjpeg required");
    let jpegtran = locate("jpegtran").expect("jpegtran required");
    let djpeg = locate("djpeg").expect("djpeg required");

    // Mix of iMCU-aligned and non-aligned sizes for all four sampling
    // factors (iMCU LCM = 32x32). Each fixture exercises a different
    // partial-MCU corner case.
    let sizes: [(usize, usize); 6] = [
        (32, 32),   // tightest iMCU-aligned for all 4 sampling factors
        (96, 64),   // iMCU-aligned (3x2 iMCU rows for S24/S441; 3x8 for S411)
        (227, 149), // matches existing testorig fixture; non-aligned both axes
        (33, 33),   // 1-pixel partial in both axes
        (49, 65),   // partial-only-width / partial-only-height vary per samp
        (96, 33),   // width-aligned for S411/S410; height non-aligned for all
    ];

    let subs: [(&str, Subsampling, &str); 4] = [
        ("S411", Subsampling::S411, "4x1"),
        ("S441", Subsampling::S441, "1x4"),
        ("S410", Subsampling::S410, "4x2"),
        ("S24", Subsampling::S24, "2x4"),
    ];

    let ops: [TransformOp; 8] = [
        TransformOp::None,
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Transpose,
        TransformOp::Transverse,
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
    ];

    let tmpdir = std::env::temp_dir().join("p3_4_diag");
    std::fs::create_dir_all(&tmpdir).expect("mkdir temp");

    let mut total_cases: usize = 0;
    let mut byte_mismatches: usize = 0;
    let mut pixel_mismatches: usize = 0;

    for &(w, h) in &sizes {
        let pixels: Vec<u8> = synth_rgb(w, h, (w * 31 + h * 17) as u32);
        let ppm_path: PathBuf = tmpdir.join(format!("synth_{}x{}.ppm", w, h));
        write_ppm(w, h, &pixels, &ppm_path);

        for (sub_label, sub, samp_arg) in subs {
            // Two source baselines per sampling factor: Rust-encoded and
            // cjpeg-encoded. Both are baseline (sequential Huffman); the
            // transform writer must reproduce jpegtran -progressive over
            // either.
            let rust_baseline: Vec<u8> = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
                .fancy_downsampling(false)
                .subsampling(sub)
                .progressive(false)
                .encode()
                .expect("Rust encode baseline");
            let cjpeg_baseline: Vec<u8> = cjpeg_encode(&cjpeg, &ppm_path, samp_arg);

            for (origin_label, baseline) in [("rust", &rust_baseline), ("cjpg", &cjpeg_baseline)] {
                for op in ops {
                    let rust_opts = TransformOptions {
                        op,
                        progressive: true,
                        copy_markers: MarkerCopyMode::All,
                        ..Default::default()
                    };

                    let rust_jpg: Vec<u8> = transform_jpeg_with_options(baseline, &rust_opts)
                        .expect("Rust transform_jpeg_with_options");
                    let c_jpg: Vec<u8> = run_jpegtran(&jpegtran, op, baseline);

                    let byte_match = rust_jpg == c_jpg;
                    let first_d: String = match first_byte_diff(&rust_jpg, &c_jpg) {
                        Some(o) => format!("0x{:x}", o),
                        None => "-".to_string(),
                    };

                    // Decoded-pixel comparison on mismatch.
                    let (px_max, px_mean) = if byte_match {
                        (0u32, 0.0f64)
                    } else {
                        let rust_pnm = run_djpeg(&djpeg, &rust_jpg);
                        let c_pnm = run_djpeg(&djpeg, &c_jpg);
                        let (_, _, rust_px) = parse_ppm(&rust_pnm);
                        let (_, _, c_px) = parse_ppm(&c_pnm);
                        pixel_stats(&rust_px, &c_px)
                    };

                    total_cases += 1;
                    if !byte_match {
                        byte_mismatches += 1;
                        if px_max > 0 {
                            pixel_mismatches += 1;
                        }
                        println!(
                            "MISMATCH {:>4}  {:>4}  size={}x{}  src={}  rust={}b  c={}b  first_d={}  px_max={}  px_mean={:.4}",
                            sub_label,
                            op_label(op),
                            w,
                            h,
                            origin_label,
                            rust_jpg.len(),
                            c_jpg.len(),
                            first_d,
                            px_max,
                            px_mean,
                        );
                    }
                }
            }
        }
    }

    println!(
        "\nTOTAL cases={}  byte_mismatches={}  pixel_mismatches={}",
        total_cases, byte_mismatches, pixel_mismatches
    );
}
