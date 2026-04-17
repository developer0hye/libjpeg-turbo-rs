//! Byte-identical parity for progressive transforms with restart intervals.
//!
//! Regression guard for the progressive RST emission path: every progressive
//! scan (DC first, DC refine, AC first, AC refine) must emit RST markers at
//! the declared DRI boundaries, with DC predictor / EOBRUN / BE correction
//! state resets applied. The resulting stream has to match C jpegtran output
//! byte-for-byte, not just pixel-for-pixel.

mod helpers;

use std::path::{Path, PathBuf};

use libjpeg_turbo_rs::{transform_jpeg_with_options, TransformOp, TransformOptions};

fn encode_source(cjpeg: &Path, sample_arg: &str, label: &str) -> Vec<u8> {
    let ppm: PathBuf = helpers::c_testimages_dir().join("testorig.ppm");
    let out: helpers::TempFile = helpers::TempFile::new(&format!("src_prog_rst_{}.jpg", label));
    helpers::run_c_cjpeg(cjpeg, &["-sample", sample_arg], &ppm, out.path());
    std::fs::read(out.path()).unwrap_or_else(|e| panic!("read source {}: {:?}", label, e))
}

fn encode_gray_source(cjpeg: &Path) -> Vec<u8> {
    let ppm: PathBuf = helpers::c_testimages_dir().join("testorig.ppm");
    let out: helpers::TempFile = helpers::TempFile::new("src_prog_rst_gray.jpg");
    helpers::run_c_cjpeg(cjpeg, &["-grayscale"], &ppm, out.path());
    std::fs::read(out.path()).unwrap_or_else(|e| panic!("read gray source: {:?}", e))
}

/// Rust vs jpegtran byte-identical comparison for `-progressive -restart N`.
///
/// `restart_token` is passed directly to `jpegtran -restart <token>`:
/// `"1"`   → every 1 MCU row (row-mode)
/// `"8b"`  → every 8 MCU blocks (block-mode)
fn compare_progressive_restart(
    jpegtran: &Path,
    source: &[u8],
    op: TransformOp,
    xform_flag: &str,
    restart_token: &str,
    restart_in_rows: bool,
    restart_value: u16,
    label: &str,
) {
    let src_file: helpers::TempFile = helpers::TempFile::new(&format!("src_{}.jpg", label));
    src_file.write_bytes(source);

    // --- C jpegtran ---
    let c_file: helpers::TempFile = helpers::TempFile::new(&format!("c_{}.jpg", label));
    let mut jtran_args: Vec<String> = vec!["-progressive".into()];
    if !xform_flag.is_empty() {
        for tok in xform_flag.split_whitespace() {
            jtran_args.push(tok.to_string());
        }
    }
    jtran_args.push("-restart".into());
    jtran_args.push(restart_token.into());
    let jtran_refs: Vec<&str> = jtran_args.iter().map(|s| s.as_str()).collect();
    helpers::run_c_jpegtran(jpegtran, &jtran_refs, src_file.path(), c_file.path());

    // --- Rust ---
    let rust_opts: TransformOptions = TransformOptions {
        op,
        progressive: true,
        restart_interval: restart_value,
        restart_in_rows,
        ..TransformOptions::default()
    };
    let rust_bytes: Vec<u8> = transform_jpeg_with_options(source, &rust_opts)
        .unwrap_or_else(|e| panic!("{}: Rust transform failed: {:?}", label, e));

    let rust_file: helpers::TempFile = helpers::TempFile::new(&format!("rust_{}.jpg", label));
    rust_file.write_bytes(&rust_bytes);

    helpers::assert_files_identical(c_file.path(), rust_file.path(), label);
}

/// Matrix: YCbCr sources × {identity, rotate, flip} × {row-mode, byte-mode restart}.
#[test]
fn progressive_restart_byte_parity_ycbcr() {
    let jpegtran: PathBuf = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let cjpeg: PathBuf = match helpers::cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    // Subsamplings: covers interleaved DC scans + non-interleaved AC scans.
    let sources: &[(&str, &str)] = &[
        ("1x1", "444"),
        ("2x1", "422"),
        ("2x2", "420"),
        ("1x2", "440"),
    ];

    // (op, jpegtran flag): keep identity + one dimension-preserving + one
    // dimension-swapping transform.
    let ops: &[(TransformOp, &str)] = &[
        (TransformOp::None, ""),
        (TransformOp::HFlip, "-flip horizontal"),
        (TransformOp::Rot90, "-rotate 90"),
    ];

    // (jpegtran token, restart_in_rows, numeric value).
    // Row mode: "-restart N" where N is MCU rows — Rust recomputes actual DRI
    //           from output dimensions.
    // Block mode: "-restart Nb" where N is DRI in MCUs directly.
    let restart_cases: &[(&str, bool, u16)] = &[("1", true, 1), ("3", true, 3), ("7b", false, 7)];

    let mut tested: u32 = 0;
    for &(sample, subsamp) in sources {
        let src: Vec<u8> = encode_source(&cjpeg, sample, subsamp);
        for &(op, xform_flag) in ops {
            for &(token, in_rows, val) in restart_cases {
                let xform_label: &str = match op {
                    TransformOp::None => "none",
                    TransformOp::HFlip => "hflip",
                    TransformOp::Rot90 => "rot90",
                    _ => "other",
                };
                let label: String = format!("{}-{}-r{}", subsamp, xform_label, token);
                compare_progressive_restart(
                    &jpegtran, &src, op, xform_flag, token, in_rows, val, &label,
                );
                tested += 1;
            }
        }
    }
    assert!(tested > 0, "no progressive+restart combos exercised");
}

/// Gray source: exercises single-component progressive scans with RST.
#[test]
fn progressive_restart_byte_parity_gray() {
    let jpegtran: PathBuf = match helpers::jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    let cjpeg: PathBuf = match helpers::cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    let src: Vec<u8> = encode_gray_source(&cjpeg);
    let restart_cases: &[(&str, bool, u16)] = &[("1", true, 1), ("5b", false, 5)];

    for &(token, in_rows, val) in restart_cases {
        compare_progressive_restart(
            &jpegtran,
            &src,
            TransformOp::None,
            "",
            token,
            in_rows,
            val,
            &format!("gray-r{}", token),
        );
    }
}
