//! Sweeps CMYK encoder configurations against the C oracle and reports, per
//! configuration, how many cases are byte-identical.
//!
//! Issue #313. `cjpeg` cannot read CMYK, so the four-component path had no C
//! reference at all; `examples/cmyk_encode_c_oracle.c` drives libjpeg directly
//! to supply one. This is the harness that measures the gap and the evidence
//! behind any claim that CMYK matches C.
//!
//! Usage: `probe_cmyk_vs_c <path-to-cmyk_encode_c_oracle>`

use libjpeg_turbo_rs::encode::pipeline::{
    compress_optimized_with_params, compress_with_params, CompressParams,
};
use libjpeg_turbo_rs::{DctMethod, PixelFormat, Subsampling};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

fn cmyk_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = vec![0u8; width * height * 4];
    let mut rng_state: u32 = 0x9e37_79b9;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let in_rect: bool = x * 3 >= width && x * 3 < width * 2;
            let offset: usize = (y * width + x) * 4;
            buffer[offset] = ((x * 255 / width.max(1)) as i32 + noise).clamp(0, 255) as u8;
            buffer[offset + 1] = ((y * 255 / height.max(1)) as i32 - noise).clamp(0, 255) as u8;
            buffer[offset + 2] = (if in_rect { 220 } else { 40 } + noise).clamp(0, 255) as u8;
            buffer[offset + 3] = ((x + y) % 256) as u8;
        }
    }
    buffer
}

fn c_encode(oracle: &str, pixels: &[u8], args: &[String]) -> Option<Vec<u8>> {
    let mut child = Command::new(oracle)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let payload: Vec<u8> = pixels.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let output = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !output.status.success() || output.stdout.is_empty() {
        return None;
    }
    Some(output.stdout)
}

fn first_difference(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

fn main() {
    let oracle: String = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cmyk_encode_c_oracle".to_string());

    // Only the subsamplings CMYK can legally carry: comp 0 and comp 3 both take
    // the luma factors, so `2 * h * v + 2` must stay within the 10-block MCU cap.
    let subsamplings: &[(Subsampling, usize, usize)] = &[
        (Subsampling::S444, 1, 1),
        (Subsampling::S422, 2, 1),
        (Subsampling::S440, 1, 2),
        (Subsampling::S420, 2, 2),
    ];
    let geometries: &[(usize, usize)] = &[
        (64, 48),
        (17, 17),
        (32, 2),
        (7, 16),
        (33, 18),
        (1, 1),
        (48, 48),
    ];
    let qualities: &[u8] = &[25, 75, 95];

    #[derive(Clone, Copy)]
    enum Option_ {
        Plain,
        Optimize,
        Smooth(u8),
        Restart(u16),
        Dct(DctMethod, &'static str),
    }

    let options: &[(&str, Option_)] = &[
        ("plain", Option_::Plain),
        ("optimize", Option_::Optimize),
        ("smooth25", Option_::Smooth(25)),
        ("smooth100", Option_::Smooth(100)),
        ("restart3", Option_::Restart(3)),
        ("dct-fast", Option_::Dct(DctMethod::IsFast, "fast")),
        ("dct-float", Option_::Dct(DctMethod::Float, "float")),
    ];

    let mut tally: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();
    let mut shown: usize = 0;

    for &(label, option) in options {
        for &(subsampling, h_samp, v_samp) in subsamplings {
            for &(width, height) in geometries {
                for &quality in qualities {
                    let pixels: Vec<u8> = cmyk_pixels(width, height);
                    let base = CompressParams::new(
                        &pixels,
                        width,
                        height,
                        PixelFormat::Cmyk,
                        quality,
                        subsampling,
                    );
                    let (rust, extra): (Option<Vec<u8>>, Vec<String>) = match option {
                        Option_::Plain => (compress_with_params(&base).ok(), vec![]),
                        Option_::Optimize => (
                            compress_optimized_with_params(&base.optimize_huffman(true)).ok(),
                            vec!["--optimize".to_string()],
                        ),
                        Option_::Smooth(factor) => (
                            compress_optimized_with_params(&base.smoothing_factor(factor)).ok(),
                            // Smoothing does not imply optimized Huffman —
                            // they are independent in C and here.
                            vec!["--smooth".to_string(), factor.to_string()],
                        ),
                        Option_::Restart(interval) => (
                            compress_with_params(&base.restart_interval(interval)).ok(),
                            vec!["--restart".to_string(), interval.to_string()],
                        ),
                        Option_::Dct(method, name) => (
                            compress_with_params(&base.dct_method(method)).ok(),
                            vec!["--dct".to_string(), name.to_string()],
                        ),
                    };

                    let mut args: Vec<String> = vec![
                        width.to_string(),
                        height.to_string(),
                        quality.to_string(),
                        h_samp.to_string(),
                        v_samp.to_string(),
                    ];
                    args.extend(extra);
                    let c: Option<Vec<u8>> = c_encode(&oracle, &pixels, &args);

                    let key: String = format!("{label}|{h_samp}x{v_samp}");
                    let entry = tally.entry(key.clone()).or_insert((0, 0, 0));
                    match (rust, c) {
                        (Some(r), Some(c)) if r == c => entry.0 += 1,
                        (Some(r), Some(c)) => {
                            entry.1 += 1;
                            if shown < 24 {
                                shown += 1;
                                println!(
                                    "  DIFFER {key} {width}x{height} q{quality}: rust={} c={} \
                                     first diff at byte {}",
                                    r.len(),
                                    c.len(),
                                    first_difference(&r, &c)
                                );
                            }
                        }
                        _ => entry.2 += 1,
                    }
                }
            }
        }
    }

    println!(
        "\n{:<24} {:>7} {:>8} {:>5}",
        "option|sampling", "MATCH", "DIFFER", "ERR"
    );
    println!("{}", "-".repeat(48));
    let (mut total_match, mut total_differ, mut total_error) = (0usize, 0usize, 0usize);
    for (key, (matched, differed, errored)) in &tally {
        println!("{key:<24} {matched:>7} {differed:>8} {errored:>5}");
        total_match += matched;
        total_differ += differed;
        total_error += errored;
    }
    println!("{}", "-".repeat(48));
    println!(
        "{:<24} {total_match:>7} {total_differ:>8} {total_error:>5}",
        "TOTAL"
    );
}
