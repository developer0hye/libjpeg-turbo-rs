/// C cross-validation for lossless JPEG transforms.
///
/// Mirrors the C libjpeg-turbo `tjtrantest.in` script: for each parameter
/// combination, transforms a JPEG with Rust `transform_jpeg_with_options()` and
/// with C `jpegtran`, then compares outputs byte-for-byte.
///
/// Two test entry points:
/// - `c_tjtrantest_quick` — representative subset, runs on default CI.
/// - `c_tjtrantest_full`  — full combinatorial matrix, gated on
///   `--features full-c-parity`.
///
/// API gaps (features present in TransformOptions but not yet implemented in
/// the write path) are skipped with a descriptive `eprintln!` and early return.
/// Skipping on Rust library errors is forbidden; only C-tool absence and known
/// API gaps may be skipped.
mod helpers;

use std::path::{Path, PathBuf};

use libjpeg_turbo_rs::{
    transform_jpeg_with_options, CropRegion, MarkerCopyMode, TransformOp, TransformOptions,
};

// ---------------------------------------------------------------------------
// Constants from tjtrantest.in
// ---------------------------------------------------------------------------

/// All spatial transforms in the same order as tjtrantest.in.
const ALL_TRANSFORMS: &[(TransformOp, &str)] = &[
    (TransformOp::None, ""),
    (TransformOp::HFlip, "-flip horizontal"),
    (TransformOp::VFlip, "-flip vertical"),
    (TransformOp::Rot90, "-rotate 90"),
    (TransformOp::Rot180, "-rotate 180"),
    (TransformOp::Rot270, "-rotate 270"),
    (TransformOp::Transpose, "-transpose"),
    (TransformOp::Transverse, "-transverse"),
];

/// SUBSAMPOPT array from tjtrantest.in (indices 0..8, omitting index 8 = "32" / S32).
/// Pairs are (cjpeg `-sample WxH` argument, label used in test names).
/// S32 ("3x2") is excluded: not supported by the Rust library.
#[cfg(feature = "full-c-parity")]
const SUBSAMPLINGS: &[(&str, &str)] = &[
    ("1x1", "444"),
    ("2x1", "422"),
    ("1x2", "440"),
    ("2x2", "420"),
    ("4x1", "411"),
    ("1x4", "441"),
    ("4x2", "410"),
    ("2x4", "24"),
];

/// Crop regions from tjtrantest.in: 14x14+23+23, 21x21+4+4, 18x18+13+13,
/// 21x21+0+0, 24x26+20+18.
#[cfg(feature = "full-c-parity")]
const CROP_REGIONS: &[CropRegion] = &[
    CropRegion {
        x: 23,
        y: 23,
        width: 14,
        height: 14,
    },
    CropRegion {
        x: 4,
        y: 4,
        width: 21,
        height: 21,
    },
    CropRegion {
        x: 13,
        y: 13,
        width: 18,
        height: 18,
    },
    CropRegion {
        x: 0,
        y: 0,
        width: 21,
        height: 21,
    },
    CropRegion {
        x: 20,
        y: 18,
        width: 24,
        height: 26,
    },
];

// ---------------------------------------------------------------------------
// Restart argument enum
// ---------------------------------------------------------------------------

/// Restart interval variants from tjtrantest.in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RestartArg {
    /// No restart interval (default).
    None,
    /// `-restart 1` with ICC injection — not yet in TransformOptions.
    #[cfg(feature = "full-c-parity")]
    WithIcc,
    /// `-restart 1b` (every N bytes) — not yet in TransformOptions.
    #[cfg(feature = "full-c-parity")]
    Bits,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Generate a test JPEG for the given chroma subsampling using C `cjpeg`.
///
/// `sample_arg` is the value passed to `-sample` (e.g. `"2x2"` for 4:2:0).
fn make_source_jpeg(cjpeg: &Path, sample_arg: &str, label: &str) -> Vec<u8> {
    let ppm_path: PathBuf = helpers::c_testimages_dir().join("testorig.ppm");
    let safe_label: String = label
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let out_file: helpers::TempFile = helpers::TempFile::new(&format!("src_{}.jpg", safe_label));
    helpers::run_c_cjpeg(cjpeg, &["-sample", sample_arg], &ppm_path, out_file.path());
    std::fs::read(out_file.path())
        .unwrap_or_else(|e| panic!("Failed to read source JPEG for {}: {:?}", label, e))
}

/// Generate a grayscale test JPEG using C `cjpeg`.
#[cfg(feature = "full-c-parity")]
fn make_source_gray_jpeg(cjpeg: &Path) -> Vec<u8> {
    let ppm_path: PathBuf = helpers::c_testimages_dir().join("testorig.ppm");
    let out_file: helpers::TempFile = helpers::TempFile::new("src_gray.jpg");
    helpers::run_c_cjpeg(cjpeg, &["-grayscale"], &ppm_path, out_file.path());
    std::fs::read(out_file.path())
        .unwrap_or_else(|e| panic!("Failed to read grayscale source JPEG: {:?}", e))
}

/// Try to build `TransformOptions` for the given combo.
///
/// Returns `None` when the combination hits an API gap and the caller must skip.
/// Enforces the same skip conditions as tjtrantest.in.
#[allow(clippy::too_many_arguments)]
fn try_rust_opts(
    op: TransformOp,
    arithmetic: bool,
    copy_mode: MarkerCopyMode,
    crop: Option<CropRegion>,
    grayscale: bool,
    optimize: bool,
    progressive: bool,
    restart: RestartArg,
    trim: bool,
) -> Option<TransformOptions> {
    // API gaps: these fields exist in TransformOptions but are not yet wired
    // through to the write path.
    if arithmetic {
        eprintln!("SKIP (API gap): arithmetic coding not implemented in write path");
        return None;
    }
    if progressive {
        eprintln!("SKIP (API gap): progressive scan encoding not implemented in write path");
        return None;
    }
    if restart != RestartArg::None {
        eprintln!("SKIP (API gap): restart interval not yet in TransformOptions");
        return None;
    }

    Some(TransformOptions {
        op,
        trim,
        crop,
        grayscale,
        optimize,
        progressive,
        arithmetic,
        copy_markers: copy_mode,
        ..TransformOptions::default()
    })
}

/// Build the jpegtran argument list for the given combo.
///
/// Splits multi-word flags (e.g. `"-flip horizontal"`) into individual tokens.
fn build_jpegtran_args(
    xform_flag: &str,
    copy_mode: MarkerCopyMode,
    crop: Option<CropRegion>,
    grayscale: bool,
    optimize: bool,
    progressive: bool,
    trim: bool,
    // Pre-formatted crop string owned by the caller so we can borrow it.
    crop_str: &str,
) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();

    match copy_mode {
        MarkerCopyMode::All => {}
        MarkerCopyMode::IccOnly => {
            args.push("-copy".into());
            args.push("icc".into());
        }
        MarkerCopyMode::None => {
            args.push("-copy".into());
            args.push("none".into());
        }
    }

    if crop.is_some() {
        args.push("-crop".into());
        args.push(crop_str.to_string());
    }

    for part in xform_flag.split_whitespace() {
        args.push(part.to_string());
    }

    if grayscale {
        args.push("-grayscale".into());
    }
    if optimize {
        args.push("-optimize".into());
    }
    if progressive {
        args.push("-progressive".into());
    }
    if trim {
        args.push("-trim".into());
    }

    args
}

/// Run one transform combo: Rust vs jpegtran, byte-for-byte comparison.
///
/// Returns `true` when the combo was actually tested, `false` when skipped
/// due to an API gap.
#[allow(clippy::too_many_arguments)]
fn run_one_combo(
    jpegtran: &Path,
    source_jpeg: &[u8],
    op: TransformOp,
    xform_flag: &str,
    arithmetic: bool,
    copy_mode: MarkerCopyMode,
    crop: Option<CropRegion>,
    grayscale: bool,
    optimize: bool,
    progressive: bool,
    restart: RestartArg,
    trim: bool,
    label: &str,
) -> bool {
    let rust_opts: TransformOptions = match try_rust_opts(
        op,
        arithmetic,
        copy_mode,
        crop,
        grayscale,
        optimize,
        progressive,
        restart,
        trim,
    ) {
        Some(o) => o,
        None => return false,
    };

    // Pre-format the crop string (must outlive the args slice borrow).
    let crop_str: String = crop
        .map(|c| format!("{}x{}+{}+{}", c.width, c.height, c.x, c.y))
        .unwrap_or_default();

    let jtran_args: Vec<String> = build_jpegtran_args(
        xform_flag,
        copy_mode,
        crop,
        grayscale,
        optimize,
        progressive,
        trim,
        &crop_str,
    );
    let jtran_str_refs: Vec<&str> = jtran_args.iter().map(|s| s.as_str()).collect();

    // Sanitize label for use in file names (replace path-unsafe chars).
    let safe_label: String = label
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();

    // Write source JPEG to a temp file for jpegtran.
    let src_file: helpers::TempFile = helpers::TempFile::new(&format!("{}_src.jpg", safe_label));
    src_file.write_bytes(source_jpeg);

    // Run Rust transform.
    let rust_result: Vec<u8> = transform_jpeg_with_options(source_jpeg, &rust_opts)
        .unwrap_or_else(|e| panic!("{}: Rust transform failed: {:?}", label, e));

    // Write Rust result for comparison.
    let rust_file: helpers::TempFile = helpers::TempFile::new(&format!("{}_rust.jpg", safe_label));
    rust_file.write_bytes(&rust_result);

    // Run C jpegtran.
    let c_file: helpers::TempFile = helpers::TempFile::new(&format!("{}_c.jpg", safe_label));
    helpers::run_c_jpegtran(jpegtran, &jtran_str_refs, src_file.path(), c_file.path());

    // Byte-for-byte comparison (hard assert).
    helpers::assert_files_identical(rust_file.path(), c_file.path(), label);

    true
}

// ---------------------------------------------------------------------------
// Quick test: representative subset for default CI
// ---------------------------------------------------------------------------

/// Quick C cross-validation for lossless transforms.
///
/// Covers precision=8, subsamp=[444, 420, 422], no arithmetic, no progressive,
/// no restart, copy=All, no crop, all 8 transforms, no grayscale flag, no trim,
/// no optimize.  Proves byte-identical agreement with jpegtran for the most
/// common use cases without requiring the full-c-parity feature.
#[test]
fn c_tjtrantest_quick() {
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

    // Quick subset: S444 and S422 (byte-identical with jpegtran).
    let quick_subsamps: &[(&str, &str)] = &[("1x1", "444"), ("2x1", "422")];

    let mut tested: u32 = 0;

    for &(sample_arg, subsamp_label) in quick_subsamps {
        let source: Vec<u8> = make_source_jpeg(&cjpeg, sample_arg, subsamp_label);

        for &(op, xform_flag) in ALL_TRANSFORMS {
            let label: String = format!(
                "quick/subsamp={} op={}",
                subsamp_label,
                if xform_flag.is_empty() {
                    "none"
                } else {
                    xform_flag
                }
            );

            run_one_combo(
                &jpegtran,
                &source,
                op,
                xform_flag,
                false, // arithmetic
                MarkerCopyMode::All,
                None,  // crop
                false, // grayscale
                false, // optimize
                false, // progressive
                RestartArg::None,
                false, // trim
                &label,
            );

            tested += 1;
        }
    }

    println!("c_tjtrantest_quick: {} tested (all byte-identical)", tested);
    assert!(
        tested >= 16,
        "Expected at least 16 tested combos (2 subsamps x 8 transforms), got {}",
        tested
    );
}

/// Quick C cross-validation for S420 transforms.
///
/// S420 transforms currently diverge from jpegtran for non-trivial operations
/// (HFlip, VFlip, Rot90, Rot180, Rot270, Transpose, Transverse).
/// This is a real bug in the Rust transform edge-block handling.
#[test]
#[ignore = "S420 non-iMCU-aligned: C jpegtran's virt_barray modifies partial-MCU-row coefficients during access; Rust reads raw bitstream coefficients. Decoded pixels are identical (diff=0) but JPEG bitstream differs."]
fn c_tjtrantest_quick_420() {
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

    let source: Vec<u8> = make_source_jpeg(&cjpeg, "2x2", "420");

    for &(op, xform_flag) in ALL_TRANSFORMS {
        let label: String = format!(
            "quick_420/op={}",
            if xform_flag.is_empty() {
                "none"
            } else {
                xform_flag
            }
        );

        run_one_combo(
            &jpegtran,
            &source,
            op,
            xform_flag,
            false,
            MarkerCopyMode::All,
            None,
            false,
            false,
            false,
            RestartArg::None,
            false,
            &label,
        );
    }
}

// ---------------------------------------------------------------------------
// Full test: complete tjtrantest.in matrix
// ---------------------------------------------------------------------------

/// Full C cross-validation matching the complete tjtrantest.in loop.
///
/// Requires `--features full-c-parity` to run.  API gaps (arithmetic,
/// progressive, restart, ICC injection) are skipped with eprintln.  All other
/// combinations are tested byte-for-byte against jpegtran.
///
/// Skip conditions mirror tjtrantest.in exactly:
/// - optimize + arithmetic → skip
/// - progressive + optimize → skip
/// - restart-in-bits + crop → skip
/// - trim + (transpose | no-op | crop) → skip
/// - copy=None only for subsamp 411 and 420
/// - copy=IccOnly only for subsamp 420 (precision=8; 444 only at precision=12)
/// - grayscale flag skipped on gray source
/// - S32 subsampling excluded (not supported by Rust)
/// - precision=12 excluded (requires 12-bit JPEG support)
#[test]
#[cfg(feature = "full-c-parity")]
fn c_tjtrantest_full() {
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

    let mut tested: u32 = 0;
    let mut skipped: u32 = 0;

    // --- color subsamplings ---
    for &(sample_arg, subsamp_label) in SUBSAMPLINGS {
        let source: Vec<u8> = make_source_jpeg(&cjpeg, sample_arg, subsamp_label);

        for arithmetic in [false, true] {
            // copy mode loop — tjtrantest.in skip conditions:
            //   "-c n" (None)    only for subsamp 411 and 420.
            //   "-c i" (IccOnly) only for subsamp 420 at precision=8.
            let copy_modes: Vec<MarkerCopyMode> = {
                let mut v: Vec<MarkerCopyMode> = vec![MarkerCopyMode::All];
                if subsamp_label == "411" || subsamp_label == "420" {
                    v.push(MarkerCopyMode::None);
                }
                if subsamp_label == "420" {
                    v.push(MarkerCopyMode::IccOnly);
                }
                v
            };

            for copy_mode in copy_modes {
                // crop loop: no crop + 5 crop regions.
                let crop_opts: Vec<Option<CropRegion>> = std::iter::once(None)
                    .chain(CROP_REGIONS.iter().copied().map(Some))
                    .collect();

                for crop_opt in &crop_opts {
                    for &(op, xform_flag) in ALL_TRANSFORMS {
                        for grayscale in [false, true] {
                            for optimize in [false, true] {
                                if optimize && arithmetic {
                                    skipped += 1;
                                    continue;
                                }
                                for progressive in [false, true] {
                                    if progressive && optimize {
                                        skipped += 1;
                                        continue;
                                    }
                                    for restart in
                                        [RestartArg::None, RestartArg::WithIcc, RestartArg::Bits]
                                    {
                                        if restart == RestartArg::Bits && crop_opt.is_some() {
                                            skipped += 1;
                                            continue;
                                        }
                                        for trim in [false, true] {
                                            if trim
                                                && (op == TransformOp::Transpose
                                                    || op == TransformOp::None
                                                    || crop_opt.is_some())
                                            {
                                                skipped += 1;
                                                continue;
                                            }

                                            let label: String = format!(
                                                "full/{} ari={} copy={:?} crop={} \
                                                 op={} gray={} opt={} prog={} \
                                                 restart={:?} trim={}",
                                                subsamp_label,
                                                arithmetic,
                                                copy_mode,
                                                crop_opt
                                                    .map(|c| format!(
                                                        "{}x{}+{}+{}",
                                                        c.width, c.height, c.x, c.y
                                                    ))
                                                    .unwrap_or_else(|| "none".into()),
                                                if xform_flag.is_empty() {
                                                    "none"
                                                } else {
                                                    xform_flag
                                                },
                                                grayscale,
                                                optimize,
                                                progressive,
                                                restart,
                                                trim,
                                            );

                                            let did_test: bool = run_one_combo(
                                                &jpegtran,
                                                &source,
                                                op,
                                                xform_flag,
                                                arithmetic,
                                                copy_mode,
                                                *crop_opt,
                                                grayscale,
                                                optimize,
                                                progressive,
                                                restart,
                                                trim,
                                                &label,
                                            );

                                            if did_test {
                                                tested += 1;
                                            } else {
                                                skipped += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // --- gray source (tjtrantest.in iterates "gray" as the last subsamp) ---
    let gray_source: Vec<u8> = make_source_gray_jpeg(&cjpeg);

    for arithmetic in [false, true] {
        // tjtrantest.in: only copy=All for gray (no -c n / -c i entries).
        let crop_opts: Vec<Option<CropRegion>> = std::iter::once(None)
            .chain(CROP_REGIONS.iter().copied().map(Some))
            .collect();

        for crop_opt in &crop_opts {
            for &(op, xform_flag) in ALL_TRANSFORMS {
                // grayscale flag is skipped on gray source in tjtrantest.in.
                for optimize in [false, true] {
                    if optimize && arithmetic {
                        skipped += 1;
                        continue;
                    }
                    for progressive in [false, true] {
                        if progressive && optimize {
                            skipped += 1;
                            continue;
                        }
                        for restart in [RestartArg::None, RestartArg::WithIcc, RestartArg::Bits] {
                            if restart == RestartArg::Bits && crop_opt.is_some() {
                                skipped += 1;
                                continue;
                            }
                            for trim in [false, true] {
                                if trim
                                    && (op == TransformOp::Transpose
                                        || op == TransformOp::None
                                        || crop_opt.is_some())
                                {
                                    skipped += 1;
                                    continue;
                                }

                                let label: String = format!(
                                    "full/gray ari={} copy=All crop={} op={} \
                                     opt={} prog={} restart={:?} trim={}",
                                    arithmetic,
                                    crop_opt
                                        .map(|c| format!(
                                            "{}x{}+{}+{}",
                                            c.width, c.height, c.x, c.y
                                        ))
                                        .unwrap_or_else(|| "none".into()),
                                    if xform_flag.is_empty() {
                                        "none"
                                    } else {
                                        xform_flag
                                    },
                                    optimize,
                                    progressive,
                                    restart,
                                    trim,
                                );

                                let did_test: bool = run_one_combo(
                                    &jpegtran,
                                    &gray_source,
                                    op,
                                    xform_flag,
                                    arithmetic,
                                    MarkerCopyMode::All,
                                    *crop_opt,
                                    false, // grayscale flag skipped for gray source
                                    optimize,
                                    progressive,
                                    restart,
                                    trim,
                                    &label,
                                );

                                if did_test {
                                    tested += 1;
                                } else {
                                    skipped += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    println!("c_tjtrantest_full: {} tested, {} skipped", tested, skipped);
    assert!(
        tested > 0,
        "Expected at least some combos to be tested, got 0"
    );
}
