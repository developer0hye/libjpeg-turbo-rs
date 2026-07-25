//! Probe: which encode configurations are byte-identical to stock `cjpeg`?
//!
//! `fuzz_encode_diff_c` is being given a reference oracle (compare our bytes
//! against `cjpeg`'s), but that oracle is only useful where byte-equality is
//! actually the contract. This measures the current state per entropy mode ×
//! colourspace × subsampling so the fuzzer's gate is evidence-based rather than
//! assumed — a fuzzer that fires on legitimate differences is just noise.
//!
//! Emits `mode|colourspace|sample|WxH  MATCH/DIFFER/ERR` per case plus a
//! summary, so the gate can be set to exactly the matching set.

use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_progressive,
    PixelFormat, Subsampling,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::process::{Command, Stdio};

fn pixels(width: usize, height: usize, channels: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = vec![0u8; width * height * channels];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let offset: usize = (y * width + x) * channels;
            for channel in 0..channels {
                let base: i32 = match channel {
                    0 => (x * 255 / width.max(1)) as i32,
                    1 => (y * 255 / height.max(1)) as i32,
                    _ => ((x ^ y) & 0xff) as i32,
                };
                buffer[offset + channel] = (base + noise).clamp(0, 255) as u8;
            }
        }
    }
    buffer
}

fn cjpeg_encode(cjpeg: &str, pnm: &[u8], args: &[&str]) -> Option<Vec<u8>> {
    let mut child = Command::new(cjpeg)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = child.stdin.take()?;
    let payload: Vec<u8> = pnm.to_vec();
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&payload);
    });
    let out = child.wait_with_output().ok()?;
    let _ = writer.join();
    if !out.status.success() || out.stdout.is_empty() {
        return None;
    }
    Some(out.stdout)
}

fn main() {
    let cjpeg: String = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cjpeg".to_string());

    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "1x1"),
        (Subsampling::S422, "2x1"),
        (Subsampling::S420, "2x2"),
        (Subsampling::S440, "1x2"),
        (Subsampling::S441, "1x4"),
        (Subsampling::S411, "4x1"),
        (Subsampling::S410, "4x2"),
        (Subsampling::S24, "2x4"),
    ];
    // Deliberately mixes MCU-aligned and partial-MCU geometries, since that is
    // where the encoder has historically diverged (#314, #316).
    // Deliberately mixes MCU-aligned with the geometry classes that have
    // actually broken: partial-MCU widths (#314, #316) and even heights that
    // are not multiples of 16 (#324).
    // Covers every partial-MCU residue in both axes: partial-MCU widths
    // (#314, #316) and the even/odd height classes behind #324, plus real
    // photo sizes as end-to-end controls.
    let geometries: &[(usize, usize)] = &[
        (32, 1),
        (32, 2),
        (32, 3),
        (32, 4),
        (32, 5),
        (32, 6),
        (32, 7),
        (32, 8),
        (32, 9),
        (32, 10),
        (32, 11),
        (32, 12),
        (32, 13),
        (32, 14),
        (32, 15),
        (32, 16),
        (32, 17),
        (32, 18),
        (32, 19),
        (32, 20),
        (7, 16),
        (17, 17),
        (23, 33),
        (33, 18),
        (48, 48),
        (64, 48),
        (800, 600),
        (1920, 1080),
    ];
    let qualities: &[u8] = &[1, 25, 75, 95];

    // (mode label, cjpeg extra flags)
    let modes: &[(&str, &[&str])] = &[
        ("baseline", &["-baseline"]),
        ("progressive", &["-progressive", "-baseline"]),
        ("arithmetic", &["-arithmetic", "-baseline"]),
        ("arith-prog", &["-arithmetic", "-progressive", "-baseline"]),
    ];

    let mut tally: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();

    for &(mode, extra) in modes {
        for &grayscale in &[false, true] {
            let channels: usize = if grayscale { 1 } else { 3 };
            for &(subsampling, sample) in subsamplings {
                // Subsampling is meaningless for a single-component image;
                // testing it once avoids four identical rows.
                if grayscale && sample != "1x1" {
                    continue;
                }
                for &(width, height) in geometries {
                    for &quality in qualities {
                        let raw: Vec<u8> = pixels(width, height, channels);
                        let rust: Option<Vec<u8>> = match (mode, grayscale) {
                            ("baseline", _) => compress(
                                &raw,
                                width,
                                height,
                                if grayscale {
                                    PixelFormat::Grayscale
                                } else {
                                    PixelFormat::Rgb
                                },
                                quality,
                                subsampling,
                            )
                            .ok(),
                            ("progressive", _) => compress_progressive(
                                &raw,
                                width,
                                height,
                                if grayscale {
                                    PixelFormat::Grayscale
                                } else {
                                    PixelFormat::Rgb
                                },
                                quality,
                                subsampling,
                            )
                            .ok(),
                            ("arithmetic", _) => compress_arithmetic(
                                &raw,
                                width,
                                height,
                                if grayscale {
                                    PixelFormat::Grayscale
                                } else {
                                    PixelFormat::Rgb
                                },
                                quality,
                                subsampling,
                            )
                            .ok(),
                            _ => compress_arithmetic_progressive(
                                &raw,
                                width,
                                height,
                                if grayscale {
                                    PixelFormat::Grayscale
                                } else {
                                    PixelFormat::Rgb
                                },
                                quality,
                                subsampling,
                            )
                            .ok(),
                        };

                        let magic: &str = if grayscale { "P5" } else { "P6" };
                        let mut pnm: Vec<u8> =
                            format!("{magic}\n{width} {height}\n255\n").into_bytes();
                        pnm.extend_from_slice(&raw);

                        let quality_arg: String = quality.to_string();
                        let mut args: Vec<&str> = vec!["-quality", &quality_arg, "-dct", "int"];
                        args.extend_from_slice(extra);
                        if grayscale {
                            args.push("-grayscale");
                        } else {
                            args.push("-sample");
                            args.push(sample);
                        }
                        let c: Option<Vec<u8>> = cjpeg_encode(&cjpeg, &pnm, &args);

                        let colour: &str = if grayscale { "gray" } else { "rgb" };
                        let key: String = format!("{mode}|{colour}|{sample}");
                        let entry = tally.entry(key).or_insert((0, 0, 0));
                        if let (Some(r), Some(c)) = (&rust, &c) {
                            if r != c {
                                println!(
                                    "  DIFFER {mode}|{colour}|{sample} {width}x{height} q{quality}: rust={} c={}",
                                    r.len(),
                                    c.len()
                                );
                            }
                        }
                        match (rust, c) {
                            (Some(r), Some(c)) if r == c => entry.0 += 1,
                            (Some(_), Some(_)) => entry.1 += 1,
                            _ => entry.2 += 1,
                        }
                    }
                }
            }
        }
    }

    println!(
        "{:<28} {:>7} {:>8} {:>5}",
        "mode|colour|sample", "MATCH", "DIFFER", "ERR"
    );
    println!("{}", "-".repeat(52));
    let (mut tm, mut td, mut te) = (0usize, 0usize, 0usize);
    for (key, (matched, differed, errored)) in &tally {
        println!("{key:<28} {matched:>7} {differed:>8} {errored:>5}");
        tm += matched;
        td += differed;
        te += errored;
    }
    println!("{}", "-".repeat(52));
    println!("{:<28} {tm:>7} {td:>8} {te:>5}", "TOTAL");
}
