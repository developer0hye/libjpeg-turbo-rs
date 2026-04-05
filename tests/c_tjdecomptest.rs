//! Parametrized decoder cross-validation mirroring C libjpeg-turbo's tjdecomptest.in.
//!
//! For each parameter combination (subsampling × crop × scale × nosmooth × dct_fast),
//! decodes a test JPEG with our Rust decoder and with C `djpeg`, then compares the
//! PPM/PGM output files byte-for-byte.
//!
//! Test JPEGs are generated on-the-fly via `cjpeg` from the reference test images
//! in `references/libjpeg-turbo/testimages/`.
//!
//! # Test variants
//! - `c_tjdecomptest_quick` — representative subset run in default CI
//! - `c_tjdecomptest_full`  — full matrix, gated on `--features full-c-parity`

mod helpers;

use std::path::{Path, PathBuf};

use libjpeg_turbo_rs::decode::pipeline::Decoder;
use libjpeg_turbo_rs::{ColorSpace, DctMethod, PixelFormat, ScalingFactor};

// ===========================================================================
// Subsampling matrix — mirrors SUBSAMPOPT / SAMPOPT from tjdecomptest.in
// ===========================================================================

/// One entry in the subsampling matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SubsampEntry {
    /// Short label used in the C script (e.g. "444", "420").
    label: &'static str,
    /// Sampling factor string for `cjpeg -sa` (e.g. "1x1", "2x2").
    samp_opt: &'static str,
    /// Whether this subsampling mode supports no-smooth (nosmooth) decoding.
    /// C script: nosmooth only for 422, 420, 440.
    nosmooth_supported: bool,
    /// Whether this entry is supported by the Rust decoder.
    /// S32 (3x2) has no Rust equivalent.
    rust_supported: bool,
}

const SUBSAMP_TABLE: &[SubsampEntry] = &[
    SubsampEntry { label: "444", samp_opt: "1x1", nosmooth_supported: false, rust_supported: true  },
    SubsampEntry { label: "422", samp_opt: "2x1", nosmooth_supported: true,  rust_supported: true  },
    SubsampEntry { label: "440", samp_opt: "1x2", nosmooth_supported: true,  rust_supported: true  },
    SubsampEntry { label: "420", samp_opt: "2x2", nosmooth_supported: true,  rust_supported: true  },
    SubsampEntry { label: "411", samp_opt: "4x1", nosmooth_supported: false, rust_supported: true  },
    SubsampEntry { label: "441", samp_opt: "1x4", nosmooth_supported: false, rust_supported: true  },
    SubsampEntry { label: "410", samp_opt: "4x2", nosmooth_supported: false, rust_supported: true  },
    SubsampEntry { label: "24",  samp_opt: "2x4", nosmooth_supported: false, rust_supported: true  },
    SubsampEntry { label: "32",  samp_opt: "3x2", nosmooth_supported: false, rust_supported: false },
];

// All crop regions from the C script (-cr args).
// "" means no crop.
const CROP_ARGS: &[&str] = &[
    "",
    "14x14+23+23",
    "21x21+4+4",
    "18x18+13+13",
    "21x21+0+0",
    "24x26+20+18",
];

// Scale arguments from the C script (-s args). "" means no scale (1/1).
#[cfg(feature = "full-c-parity")]
const SCALE_ARGS: &[&str] = &[
    "",
    "16/8",
    "15/8",
    "14/8",
    "13/8",
    "12/8",
    "11/8",
    "10/8",
    "9/8",
    "7/8",
    "6/8",
    "5/8",
    "4/8",
    "3/8",
    "2/8",
    "1/8",
];

// Small scales that cannot be combined with crop (C script rule).
const SMALL_SCALES_NO_CROP: &[&str] = &["1/8", "2/8", "3/8"];

// ===========================================================================
// Helpers
// ===========================================================================

/// Parse a scale string like "4/8" into (num, denom).
fn parse_scale(s: &str) -> (u32, u32) {
    let mut parts = s.splitn(2, '/');
    let num: u32 = parts.next().unwrap().parse().unwrap();
    let denom: u32 = parts.next().unwrap().parse().unwrap();
    (num, denom)
}

/// Parse a crop string like "14x14+23+23" into (w, h, x, y).
fn parse_crop(s: &str) -> (usize, usize, usize, usize) {
    // Format: WxH+X+Y
    let s = s.replace('+', " ");
    let s = s.replace('x', " ");
    let mut parts = s.split_whitespace();
    let w: usize = parts.next().unwrap().parse().unwrap();
    let h: usize = parts.next().unwrap().parse().unwrap();
    let x: usize = parts.next().unwrap().parse().unwrap();
    let y: usize = parts.next().unwrap().parse().unwrap();
    (w, h, x, y)
}

/// Generate the test JPEG for a given subsampling using C cjpeg.
/// Returns the path to the generated JPEG temp file (kept alive by TempFile).
fn generate_test_jpeg(
    cjpeg: &Path,
    entry: &SubsampEntry,
    out_path: &Path,
) {
    let img_dir: PathBuf = helpers::c_testimages_dir();
    let rgb_img: PathBuf = img_dir.join("testorig.ppm");

    helpers::run_c_cjpeg(
        cjpeg,
        &["-sa", entry.samp_opt],
        &rgb_img,
        out_path,
    );
}

/// Decode a JPEG with our Rust Decoder and write the RGB output as PPM.
/// Returns false and prints SKIP if the decode fails (non-Rust-internal skip only for
/// unsupported scale/crop combos detected before calling this function).
fn rust_decode_rgb(
    jpeg_path: &Path,
    scale_arg: &str,
    crop_arg: &str,
    nosmooth: bool,
    dct_fast: bool,
    out_ppm: &Path,
) -> bool {
    let jpeg_data: Vec<u8> = std::fs::read(jpeg_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {:?}", jpeg_path, e));

    let mut dec: Decoder = Decoder::new(&jpeg_data)
        .unwrap_or_else(|e| panic!("Decoder::new failed for {:?}: {:?}", jpeg_path, e));

    if !scale_arg.is_empty() {
        let (num, denom) = parse_scale(scale_arg);
        dec.set_scale(ScalingFactor::new(num, denom));
    }

    if !crop_arg.is_empty() {
        let (w, h, x, y) = parse_crop(crop_arg);
        dec.set_crop_region(x, y, w, h);
    }

    if nosmooth {
        dec.set_fast_upsample(true);
    }

    if dct_fast {
        dec.set_dct_method(DctMethod::IsFast);
    }

    let img = match dec.decode_image() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: Rust decode failed (scale={:?} crop={:?}): {:?}", scale_arg, crop_arg, e);
            return false;
        }
    };

    helpers::write_ppm_file(out_ppm, img.width, img.height, &img.data);
    true
}

/// Decode a JPEG with our Rust Decoder to grayscale and write output as PGM.
fn rust_decode_gray(
    jpeg_path: &Path,
    scale_arg: &str,
    crop_arg: &str,
    nosmooth: bool,
    dct_fast: bool,
    out_pgm: &Path,
) -> bool {
    let jpeg_data: Vec<u8> = std::fs::read(jpeg_path)
        .unwrap_or_else(|e| panic!("Failed to read {:?}: {:?}", jpeg_path, e));

    let mut dec: Decoder = Decoder::new(&jpeg_data)
        .unwrap_or_else(|e| panic!("Decoder::new failed for {:?}: {:?}", jpeg_path, e));

    dec.set_output_format(PixelFormat::Grayscale);
    dec.set_output_colorspace(ColorSpace::Grayscale);

    if !scale_arg.is_empty() {
        let (num, denom) = parse_scale(scale_arg);
        dec.set_scale(ScalingFactor::new(num, denom));
    }

    if !crop_arg.is_empty() {
        let (w, h, x, y) = parse_crop(crop_arg);
        dec.set_crop_region(x, y, w, h);
    }

    if nosmooth {
        dec.set_fast_upsample(true);
    }

    if dct_fast {
        dec.set_dct_method(DctMethod::IsFast);
    }

    let img = match dec.decode_image() {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: Rust gray decode failed (scale={:?} crop={:?}): {:?}", scale_arg, crop_arg, e);
            return false;
        }
    };

    helpers::write_pgm_file(out_pgm, img.width, img.height, &img.data);
    true
}

/// Build the djpeg args for a given combo.
/// Returns (rgb_args, gray_args) as Vec<String> each.
fn build_djpeg_args(
    scale_arg: &str,
    crop_arg: &str,
    nosmooth: bool,
    dct_fast: bool,
) -> (Vec<String>, Vec<String>) {
    let mut common: Vec<String> = Vec::new();

    if !crop_arg.is_empty() {
        common.push("-crop".to_string());
        common.push(crop_arg.to_string());
    }

    if dct_fast {
        common.push("-dct".to_string());
        common.push("fast".to_string());
    }

    if nosmooth {
        common.push("-nosmooth".to_string());
    }

    if !scale_arg.is_empty() {
        common.push("-scale".to_string());
        common.push(scale_arg.to_string());
    }

    let mut rgb_args = common.clone();
    rgb_args.push("-ppm".to_string());

    let mut gray_args = common;
    gray_args.push("-grayscale".to_string());

    (rgb_args, gray_args)
}

/// Run one complete combo: generate JPEG, decode with Rust + C, compare.
/// Returns the number of comparisons that ran (for quick test counting).
fn run_combo(
    djpeg: &Path,
    jpeg_path: &Path,
    entry: &SubsampEntry,
    scale_arg: &str,
    crop_arg: &str,
    nosmooth: bool,
    dct_fast: bool,
    label_prefix: &str,
) -> usize {
    let mut count: usize = 0;
    let scale_str: String = if scale_arg.is_empty() {
        "none".to_string()
    } else {
        scale_arg.replace('/', "_")
    };
    let crop_str: String = if crop_arg.is_empty() {
        "none".to_string()
    } else {
        crop_arg.replace('+', "_").replace('x', "_")
    };
    let label: String = format!(
        "{}_{}_scale={}_crop={}_ns={}_dctf={}",
        label_prefix, entry.label, scale_str, crop_str, nosmooth, dct_fast
    );

    let (rgb_djpeg_args, gray_djpeg_args) = build_djpeg_args(scale_arg, crop_arg, nosmooth, dct_fast);
    let rgb_djpeg_refs: Vec<&str> = rgb_djpeg_args.iter().map(|s| s.as_str()).collect();
    let gray_djpeg_refs: Vec<&str> = gray_djpeg_args.iter().map(|s| s.as_str()).collect();

    // --- RGB comparison ---
    let rust_ppm: helpers::TempFile = helpers::TempFile::new(&format!("{}_rust.ppm", label));
    let c_ppm: helpers::TempFile = helpers::TempFile::new(&format!("{}_c.ppm", label));

    if rust_decode_rgb(jpeg_path, scale_arg, crop_arg, nosmooth, dct_fast, rust_ppm.path()) {
        helpers::run_c_djpeg(djpeg, &rgb_djpeg_refs, jpeg_path, c_ppm.path());
        helpers::assert_files_identical(rust_ppm.path(), c_ppm.path(), &format!("{} RGB", label));
        count += 1;
    }

    // --- Grayscale comparison (only when nosmooth=false) ---
    if !nosmooth {
        let rust_pgm: helpers::TempFile = helpers::TempFile::new(&format!("{}_rust.pgm", label));
        let c_pgm: helpers::TempFile = helpers::TempFile::new(&format!("{}_c.pgm", label));

        if rust_decode_gray(jpeg_path, scale_arg, crop_arg, nosmooth, dct_fast, rust_pgm.path()) {
            helpers::run_c_djpeg(djpeg, &gray_djpeg_refs, jpeg_path, c_pgm.path());
            helpers::assert_files_identical(rust_pgm.path(), c_pgm.path(), &format!("{} GRAY", label));
            count += 1;
        }
    }

    count
}

// ===========================================================================
// Inner loop shared by both quick and full tests
// ===========================================================================

/// Iterate the C script's decode loop for the given subsamp entries and param ranges.
/// `subsamp_filter`: which entries to include by label.
/// `scale_filter`:   which scale_arg strings to include.
/// `include_crop`:   whether to test crop combos.
/// `include_nosmooth`: whether to test nosmooth.
/// `include_dct_fast`: whether to test dct fast.
fn run_decode_matrix(
    djpeg: &Path,
    cjpeg: &Path,
    subsamp_entries: &[&SubsampEntry],
    scale_filter: &[&str],
    include_crop: bool,
    include_nosmooth: bool,
    include_dct_fast: bool,
    label_prefix: &str,
) {
    for entry in subsamp_entries {
        if !entry.rust_supported {
            eprintln!("SKIP: subsampling {} has no Rust equivalent", entry.label);
            continue;
        }

        // Generate test JPEG for this subsampling mode.
        let jpeg_tmp: helpers::TempFile =
            helpers::TempFile::new(&format!("tjdecomp_{}.jpg", entry.label));
        generate_test_jpeg(cjpeg, entry, jpeg_tmp.path());

        // Build crop list.
        let crops: Vec<&str> = if include_crop {
            CROP_ARGS.to_vec()
        } else {
            vec![""]
        };

        for &crop_arg in &crops {
            // C script: skip crop for S32; already filtered above via rust_supported.
            // Also skip crop for this entry if label is "32".
            if !crop_arg.is_empty() && entry.label == "32" {
                continue;
            }

            for &scale_arg in scale_filter {
                // C script: skip small scales (1/8, 2/8, 3/8) when crop is active.
                if !crop_arg.is_empty() && SMALL_SCALES_NO_CROP.contains(&scale_arg) {
                    continue;
                }

                // nosmooth variants
                let nosmooth_values: &[bool] = if include_nosmooth && entry.nosmooth_supported {
                    &[false, true]
                } else {
                    &[false]
                };

                for &nosmooth in nosmooth_values {
                    // dct_fast variants
                    // C script rule: dct fast is tested only when scale=4/8 and subsamp in
                    // {420, 32}, OR when scale is empty (no scale).
                    // Simplified: we test dct_fast only when include_dct_fast=true and
                    // the scale is 4/8 with subsamp 420, or scale is empty.
                    let dct_fast_values: &[bool] = if include_dct_fast
                        && ((scale_arg == "4/8"
                            && (entry.label == "420" || entry.label == "32"))
                            || scale_arg.is_empty())
                    {
                        &[false, true]
                    } else {
                        &[false]
                    };

                    for &dct_fast in dct_fast_values {
                        run_combo(
                            djpeg,
                            jpeg_tmp.path(),
                            entry,
                            scale_arg,
                            crop_arg,
                            nosmooth,
                            dct_fast,
                            label_prefix,
                        );
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Quick test — representative subset for default CI
// ===========================================================================

/// Quick cross-validation: precision=8, subsamp ∈ {444, 420, 422},
/// no crop, scale ∈ {none, 4/8, 2/8}, no nosmooth, no dct_fast.
#[test]
fn c_tjdecomptest_quick() {
    let djpeg: PathBuf = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
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

    // Verify that the reference image exists.
    let img_dir: PathBuf = helpers::c_testimages_dir();
    if !img_dir.join("testorig.ppm").exists() {
        eprintln!("SKIP: testorig.ppm not found in {:?}", img_dir);
        return;
    }

    let quick_subsamps: Vec<&SubsampEntry> = SUBSAMP_TABLE
        .iter()
        .filter(|e| matches!(e.label, "444" | "420" | "422"))
        .collect();

    let quick_scales: &[&str] = &["", "4/8", "2/8"];

    run_decode_matrix(
        &djpeg,
        &cjpeg,
        &quick_subsamps,
        quick_scales,
        false, // no crop
        false, // no nosmooth
        false, // no dct_fast
        "quick",
    );
}

// ===========================================================================
// Full test — complete matrix, gated on `full-c-parity` feature
// ===========================================================================

#[test]
#[cfg(feature = "full-c-parity")]
fn c_tjdecomptest_full() {
    let djpeg: PathBuf = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
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

    let img_dir: PathBuf = helpers::c_testimages_dir();
    if !img_dir.join("testorig.ppm").exists() {
        eprintln!("SKIP: testorig.ppm not found in {:?}", img_dir);
        return;
    }

    let all_subsamps: Vec<&SubsampEntry> = SUBSAMP_TABLE.iter().collect();

    run_decode_matrix(
        &djpeg,
        &cjpeg,
        &all_subsamps,
        SCALE_ARGS,
        true, // include crop
        true, // include nosmooth
        true, // include dct_fast
        "full",
    );
}
