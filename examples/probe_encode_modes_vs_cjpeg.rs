//! Sweeps encoder configurations against stock `cjpeg` and reports, per
//! configuration, how many cases are byte-identical.
//!
//! This is the harness behind the encode-side conformance claims: it is what
//! established that byte-equality is the contract before `fuzz_encode_diff_c`
//! was given a reference oracle, and what pinned down #314, #316 and #324.
//!
//! The `-dct` axis exists for #319 / P4-44: every cross-check in the tree has
//! only ever passed `int`, so whether `fast` and `float` match C had never
//! been established on any backend. It is swept for the baseline mode only —
//! the public progressive and arithmetic helpers take no `dct_method`.
//!
//! Usage: `probe_encode_modes_vs_cjpeg [path-to-cjpeg]`

use libjpeg_turbo_rs::encode::pipeline::{compress_with_params, CompressParams};
use libjpeg_turbo_rs::{
    compress_arithmetic, compress_arithmetic_progressive, compress_progressive, DctMethod,
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
    // Covers every partial-MCU residue in both axes: the widths behind #314 /
    // #316 and the even/odd height classes behind #324, plus real photo sizes
    // as end-to-end controls.
    let geometries: &[(usize, usize)] = &[
        (32, 1),
        (32, 2),
        (32, 4),
        (32, 7),
        (32, 8),
        (32, 15),
        (32, 16),
        (32, 18),
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

    // `-baseline` is passed for every mode: in cjpeg it controls
    // `force_baseline` (clamping scaled quant values to 255), which is
    // orthogonal to the entropy mode and is what `quality_scale_quant_table`
    // does unconditionally. Omitting it diverges below quality 20 over a flag
    // mismatch rather than an encoder defect.
    let modes: &[(&str, &[&str])] = &[
        ("baseline", &["-baseline"] as &[&str]),
        ("progressive", &["-progressive", "-baseline"]),
        ("arithmetic", &["-arithmetic", "-baseline"]),
        ("arith-prog", &["-arithmetic", "-progressive", "-baseline"]),
    ];
    let dct_methods: &[(&str, DctMethod)] = &[
        ("int", DctMethod::IsLow),
        ("fast", DctMethod::IsFast),
        ("float", DctMethod::Float),
    ];

    let mut tally: BTreeMap<String, (usize, usize, usize)> = BTreeMap::new();

    for &(mode, extra) in modes {
        let dcts: &[(&str, DctMethod)] = if mode == "baseline" {
            dct_methods
        } else {
            &dct_methods[..1]
        };
        for &(dct_name, dct_method) in dcts {
            for &grayscale in &[false, true] {
                let channels: usize = if grayscale { 1 } else { 3 };
                let format: PixelFormat = if grayscale {
                    PixelFormat::Grayscale
                } else {
                    PixelFormat::Rgb
                };
                for &(subsampling, sample) in subsamplings {
                    // Subsampling is meaningless for a single-component image.
                    if grayscale && sample != "1x1" {
                        continue;
                    }
                    for &(width, height) in geometries {
                        for &quality in qualities {
                            let raw: Vec<u8> = pixels(width, height, channels);
                            let rust: Option<Vec<u8>> = match mode {
                                "baseline" => compress_with_params(
                                    &CompressParams::new(
                                        &raw,
                                        width,
                                        height,
                                        format,
                                        quality,
                                        subsampling,
                                    )
                                    .dct_method(dct_method),
                                )
                                .ok(),
                                "progressive" => compress_progressive(
                                    &raw,
                                    width,
                                    height,
                                    format,
                                    quality,
                                    subsampling,
                                )
                                .ok(),
                                "arithmetic" => compress_arithmetic(
                                    &raw,
                                    width,
                                    height,
                                    format,
                                    quality,
                                    subsampling,
                                )
                                .ok(),
                                _ => compress_arithmetic_progressive(
                                    &raw,
                                    width,
                                    height,
                                    format,
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
                            let mut args: Vec<&str> =
                                vec!["-quality", &quality_arg, "-dct", dct_name];
                            args.extend_from_slice(extra);
                            if grayscale {
                                args.push("-grayscale");
                            } else {
                                args.push("-sample");
                                args.push(sample);
                            }
                            let c: Option<Vec<u8>> = cjpeg_encode(&cjpeg, &pnm, &args);

                            let colour: &str = if grayscale { "gray" } else { "rgb" };
                            let key: String = format!("{mode}|{dct_name}|{colour}|{sample}");
                            if let (Some(r), Some(c)) = (&rust, &c) {
                                if r != c {
                                    println!(
                                        "  DIFFER {key} {width}x{height} q{quality}: rust={} c={}",
                                        r.len(),
                                        c.len()
                                    );
                                }
                            }
                            let entry = tally.entry(key).or_insert((0, 0, 0));
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
    }

    println!(
        "\n{:<32} {:>7} {:>8} {:>5}",
        "mode|dct|colour|sample", "MATCH", "DIFFER", "ERR"
    );
    println!("{}", "-".repeat(56));
    let (mut total_match, mut total_differ, mut total_error) = (0usize, 0usize, 0usize);
    for (key, (matched, differed, errored)) in &tally {
        println!("{key:<32} {matched:>7} {differed:>8} {errored:>5}");
        total_match += matched;
        total_differ += differed;
        total_error += errored;
    }
    println!("{}", "-".repeat(56));
    println!(
        "{:<32} {total_match:>7} {total_differ:>8} {total_error:>5}",
        "TOTAL"
    );
}
