//! P3-4 regression: progressive transform writer must be byte-identical to
//! `jpegtran -progressive -copy all <op>` for the four 4-pixel chroma
//! sampling factors (S411 / S441 / S410 / S24) and every spatial transform op.
//!
//! Pins the closure of the gate at `src/api/coefficient.rs::write_progressive`'s
//! `progressive_safe` predicate. Before P3-4 closure that gate fell back to
//! baseline whenever `max_h > 2 || max_v > 2`; this regression catches any
//! reintroduction.
//!
//! Coverage:
//! - 4 chroma sampling factors with `max_h > 2` or `max_v > 2`.
//! - 8 transform ops (None, HFlip, VFlip, Transpose, Transverse, Rot90, Rot180, Rot270).
//! - 2 source baselines per (size, sampling): Rust-encoded and cjpeg-encoded.
//!   Both feed the same downstream transform, but the source DCT coefficients
//!   differ in the trailing zero pattern, exercising different Huffman gather
//!   distributions.
//! - 4 image sizes covering iMCU-aligned and non-aligned (partial-MCU) layouts:
//!   * 32×32  — minimal iMCU-aligned for all four sampling factors.
//!   * 96×64  — multi-iMCU, aligned.
//!   * 33×33  — 1-pixel partial in both axes.
//!   * 49×65  — partial in both axes with non-coprime offsets.
//!
//! Each case asserts byte-equality.
//!
//! Skip rules: only when stock C tools are absent on a developer's local box.
//! In CI (`CI=true` / `GITHUB_ACTIONS=true`), missing tools are a hard panic
//! that names the LAST_MILE.md gate so the silent skip never disappears into
//! green CI.

mod helpers;

use std::path::PathBuf;
use std::process::Command;

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, Encoder, MarkerCopyMode, PixelFormat, Subsampling, TransformOp,
    TransformOptions,
};

const CHROMA_SAMPLINGS: &[(&str, Subsampling, &str)] = &[
    ("S411", Subsampling::S411, "4x1"),
    ("S441", Subsampling::S441, "1x4"),
    ("S410", Subsampling::S410, "4x2"),
    ("S24", Subsampling::S24, "2x4"),
];

const ALL_OPS: &[TransformOp] = &[
    TransformOp::None,
    TransformOp::HFlip,
    TransformOp::VFlip,
    TransformOp::Transpose,
    TransformOp::Transverse,
    TransformOp::Rot90,
    TransformOp::Rot180,
    TransformOp::Rot270,
];

const SIZES: &[(usize, usize)] = &[(32, 32), (96, 64), (33, 33), (49, 65)];

fn op_label(op: TransformOp) -> &'static str {
    match op {
        TransformOp::None => "none",
        TransformOp::HFlip => "hflip",
        TransformOp::VFlip => "vflip",
        TransformOp::Transpose => "transpose",
        TransformOp::Transverse => "transverse",
        TransformOp::Rot90 => "rot90",
        TransformOp::Rot180 => "rot180",
        TransformOp::Rot270 => "rot270",
    }
}

fn jpegtran_op_args(op: TransformOp) -> Vec<&'static str> {
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

/// Procedural RGB pattern that mixes a smooth gradient with a higher-frequency
/// component so chroma planes carry non-trivial signal after subsampling.
fn synth_rgb(width: usize, height: usize, seed: u32) -> Vec<u8> {
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

fn require_tools_or_skip() -> Option<(PathBuf, PathBuf)> {
    // Lifting the LAST_MILE.md P3-4 gate requires this regression to actually
    // run, so CI must not skip: `optional_c_tool` panics there. It also
    // searches PATH, which the hand-rolled lookup's skip message denied
    // (P4-116).
    let cjpeg: PathBuf = helpers::optional_c_tool("cjpeg")?;
    let jpegtran: PathBuf = helpers::optional_c_tool("jpegtran")?;
    Some((cjpeg, jpegtran))
}

#[test]
fn progressive_transform_4pixel_chroma_byte_exact_vs_jpegtran() {
    let (cjpeg, jpegtran) = match require_tools_or_skip() {
        Some(t) => t,
        None => return,
    };

    let tmpdir = std::env::temp_dir().join("regression_p3_4_progressive_4pixel_chroma");
    std::fs::create_dir_all(&tmpdir).expect("mkdir tempdir");

    let mut tested: u32 = 0;

    for &(width, height) in SIZES {
        let seed: u32 = (width * 31 + height * 17) as u32;
        let pixels: Vec<u8> = synth_rgb(width, height, seed);
        let ppm_path: PathBuf = tmpdir.join(format!("synth_{}x{}.ppm", width, height));
        helpers::write_ppm_file(&ppm_path, width, height, &pixels);

        for &(samp_label, subsampling, samp_arg) in CHROMA_SAMPLINGS {
            // Source baseline #1 — Rust encoder.
            let rust_baseline: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .fancy_downsampling(false)
                .subsampling(subsampling)
                .progressive(false)
                .encode()
                .expect("Rust baseline encode");

            // Source baseline #2 — stock cjpeg, same sampling factor and
            // baseline Huffman. Validates that the transform writer doesn't
            // depend on a Rust-encoded source's specific Huffman / DC layout.
            let cjpeg_jpg =
                tmpdir.join(format!("src_cjpeg_{}_{}x{}.jpg", samp_label, width, height));
            let cjpeg_status = Command::new(&cjpeg)
                .arg("-sample")
                .arg(samp_arg)
                .arg("-outfile")
                .arg(&cjpeg_jpg)
                .arg(&ppm_path)
                .output()
                .expect("run cjpeg");
            assert!(
                cjpeg_status.status.success(),
                "cjpeg failed for {}/{}x{}: {}",
                samp_label,
                width,
                height,
                String::from_utf8_lossy(&cjpeg_status.stderr)
            );
            let cjpeg_baseline: Vec<u8> =
                std::fs::read(&cjpeg_jpg).expect("read cjpeg baseline output");

            for (origin_label, baseline_bytes) in
                [("rust_src", &rust_baseline), ("cjpg_src", &cjpeg_baseline)]
            {
                let baseline_jpg = tmpdir.join(format!(
                    "in_{}_{}_{}x{}.jpg",
                    origin_label, samp_label, width, height
                ));
                std::fs::write(&baseline_jpg, baseline_bytes).expect("write baseline jpg");

                for &op in ALL_OPS {
                    let case_label = format!(
                        "{} src={} {} {}x{}",
                        samp_label,
                        origin_label,
                        op_label(op),
                        width,
                        height
                    );

                    // Rust progressive transform.
                    let rust_opts = TransformOptions {
                        op,
                        progressive: true,
                        copy_markers: MarkerCopyMode::All,
                        ..Default::default()
                    };
                    let rust_out: Vec<u8> = transform_jpeg_with_options(baseline_bytes, &rust_opts)
                        .unwrap_or_else(|e| {
                            panic!("Rust transform failed [{}]: {:?}", case_label, e)
                        });

                    // Stock jpegtran progressive transform.
                    let mut tran_args: Vec<&str> = vec!["-progressive", "-copy", "all"];
                    tran_args.extend(jpegtran_op_args(op));
                    let c_out: Vec<u8> = helpers::transform_with_c_jpegtran(
                        &jpegtran,
                        baseline_bytes,
                        &tran_args,
                        &case_label.replace(' ', "_"),
                    );

                    if rust_out != c_out {
                        let n: usize = rust_out.len().min(c_out.len());
                        let first_diff: Option<usize> = (0..n).find(|&i| rust_out[i] != c_out[i]);
                        let len_match = rust_out.len() == c_out.len();
                        panic!(
                            "P3-4 regression failed [{}]: rust_len={} c_len={} len_match={} first_diff={:?}",
                            case_label,
                            rust_out.len(),
                            c_out.len(),
                            len_match,
                            first_diff,
                        );
                    }
                    tested += 1;
                }
            }
        }
    }

    // Sanity floor: 4 sizes × 4 samplings × 2 origins × 8 ops = 256.
    assert_eq!(
        tested, 256,
        "expected exactly 256 cross-validated cases, got {}",
        tested
    );
}
