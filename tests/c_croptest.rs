//! Parametrized crop boundary test mirroring C libjpeg-turbo's croptest.in.
//!
//! The C script generates 5 test JPEGs (GRAY, 420, 422, 440, 444) from a 128x95 BMP,
//! then for each (Y, H) pair computes X and W from the formula in croptest.in and
//! calls `djpeg -crop WxH+X+Y` on each JPEG. The reference output is produced by
//! fully decoding then cropping with ImageMagick. Here we compare Rust decompress_cropped
//! against C djpeg -crop directly.
//!
//! Quick test: baseline only, no nosmooth, no colors, representative subset of Y/H/samp.
//! Full test (feature = "full-c-parity"): 10,880-case grid, of which 5,440 are
//! compared and 5,440 excluded by count (colour quantization is unimplemented).
//!
//! `croptest.in`'s formula only ever produces an x that is already iMCU-aligned,
//! so `c_croptest_unaligned_x` adds specs outside the mirrored grid to cover
//! `jpeg_crop_scanline`'s snap-and-widen path.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::{compress, PixelFormat, Subsampling};

// ===========================================================================
// Constants matching C croptest.in
// ===========================================================================

const WIDTH: usize = 128;
const HEIGHT: usize = 95;

// Subsampling configurations used in the C script.
#[derive(Clone, Copy, Debug)]
struct SampConfig {
    name: &'static str,
    subsampling: Subsampling,
    /// cjpeg flag(s) for this subsampling (empty means -grayscale).
    cjpeg_args: &'static [&'static str],
    is_gray: bool,
}

const SAMP_CONFIGS: &[SampConfig] = &[
    SampConfig {
        name: "GRAY",
        subsampling: Subsampling::S444, // grayscale: use PixelFormat::Grayscale in compress
        cjpeg_args: &["-grayscale"],
        is_gray: true,
    },
    SampConfig {
        name: "420",
        subsampling: Subsampling::S420,
        cjpeg_args: &["-sample", "2x2"],
        is_gray: false,
    },
    SampConfig {
        name: "422",
        subsampling: Subsampling::S422,
        cjpeg_args: &["-sample", "2x1"],
        is_gray: false,
    },
    SampConfig {
        name: "440",
        subsampling: Subsampling::S440,
        cjpeg_args: &["-sample", "1x2"],
        is_gray: false,
    },
    SampConfig {
        name: "444",
        subsampling: Subsampling::S444,
        cjpeg_args: &["-sample", "1x1"],
        is_gray: false,
    },
];

// ===========================================================================
// Test image generation
// ===========================================================================

/// Generate a 128x95 RGB pixel gradient matching the BMP source dimensions.
fn generate_test_pixels() -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(WIDTH * HEIGHT * 3);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let r: u8 = ((x * 255) / WIDTH.max(1)) as u8;
            let g: u8 = ((y * 255) / HEIGHT.max(1)) as u8;
            let b: u8 = (((x + y) * 127) / (WIDTH + HEIGHT).max(1)) as u8;
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
    }
    pixels
}

/// Generate a test JPEG using C cjpeg (matching the C croptest.in approach).
/// Falls back to Rust compress if cjpeg is unavailable.
fn make_test_jpeg_with_c(cjpeg: &Path, pixels: &[u8], samp: &SampConfig, prog: bool) -> Vec<u8> {
    let ppm: Vec<u8> = if samp.is_gray {
        // Convert RGB to grayscale for cjpeg input
        let gray_pixels: Vec<u8> = pixels
            .chunks_exact(3)
            .map(|px| {
                let r: u32 = px[0] as u32;
                let g: u32 = px[1] as u32;
                let b: u32 = px[2] as u32;
                // Standard luma formula
                ((r * 299 + g * 587 + b * 114 + 500) / 1000) as u8
            })
            .collect();
        helpers::build_pgm(&gray_pixels, WIDTH, HEIGHT)
    } else {
        helpers::build_ppm(pixels, WIDTH, HEIGHT)
    };

    let ppm_tmp: helpers::TempFile = helpers::TempFile::new("croptest_src.ppm");
    ppm_tmp.write_bytes(&ppm);

    let jpeg_tmp: helpers::TempFile =
        helpers::TempFile::new(&format!("croptest_{}.jpg", samp.name));

    let mut args: Vec<&str> = Vec::new();
    if prog {
        args.push("-progressive");
    }
    args.extend_from_slice(samp.cjpeg_args);

    helpers::run_c_cjpeg(cjpeg, &args, ppm_tmp.path(), jpeg_tmp.path());

    std::fs::read(jpeg_tmp.path()).unwrap_or_else(|e| panic!("failed to read cjpeg output: {e}"))
}

/// Generate test JPEG using Rust encoder (fallback when cjpeg is unavailable).
fn make_test_jpeg_rust(pixels: &[u8], samp: &SampConfig) -> Vec<u8> {
    if samp.is_gray {
        let gray_pixels: Vec<u8> = pixels
            .chunks_exact(3)
            .map(|px| {
                let r: u32 = px[0] as u32;
                let g: u32 = px[1] as u32;
                let b: u32 = px[2] as u32;
                ((r * 299 + g * 587 + b * 114 + 500) / 1000) as u8
            })
            .collect();
        compress(
            &gray_pixels,
            WIDTH,
            HEIGHT,
            PixelFormat::Grayscale,
            95,
            Subsampling::S444,
        )
        .expect("compress grayscale must succeed")
    } else {
        compress(
            pixels,
            WIDTH,
            HEIGHT,
            PixelFormat::Rgb,
            95,
            samp.subsampling,
        )
        .expect("compress must succeed")
    }
}

// ===========================================================================
// Crop spec computation (mirrors C croptest.in exactly)
// ===========================================================================

/// Compute crop spec (w, h, x, y) from loop variables following croptest.in.
/// y_iter: loop variable 0..=16, h_iter: loop variable 1..=16.
fn compute_crop_spec(y_iter: usize, h_iter: usize) -> (usize, usize, usize, usize) {
    let x: usize = (y_iter * 16) % 128;
    let w: usize = WIDTH - x - 7;
    let (crop_x, crop_y, crop_w, crop_h) = if y_iter <= 15 {
        (x, y_iter, w, h_iter)
    } else {
        // y=16 special case: Y2 = HEIGHT - H
        let y2: usize = HEIGHT.saturating_sub(h_iter);
        (x, y2, w, h_iter)
    };
    (crop_w, crop_h, crop_x, crop_y)
}

// ===========================================================================
// Core comparison logic
// ===========================================================================

/// Run a single crop scenario: compare Rust decompress_cropped vs C djpeg -crop.
///
/// Returns `None` when the comparison ran, or `Some(reason)` when the case is
/// deliberately outside the matrix. Since P4-116 the only such case is a crop
/// spec with zero width or height; every other disagreement asserts.
fn run_crop_scenario(
    djpeg: &Path,
    jpeg_data: &[u8],
    crop_w: usize,
    crop_h: usize,
    crop_x: usize,
    crop_y: usize,
    nosmooth: bool,
    samp_name: &str,
    scenario_label: &str,
) -> Option<&'static str> {
    // Guard: crop must have valid size
    if crop_w == 0 || crop_h == 0 {
        return Some("crop spec has zero width or height");
    }

    // Rust side — use Decoder directly to support nosmooth flag.
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(jpeg_data)
        .unwrap_or_else(|e| panic!("[{scenario_label}] Decoder::new failed: {e}"));
    if nosmooth {
        decoder.set_fast_upsample(true);
    }
    // Hand Rust the *requested* crop, exactly as we hand it to djpeg, and let it
    // do its own iMCU alignment. The harness used to pre-align the request with a
    // local copy of `jpeg_crop_scanline`'s formula, which meant the decoder's own
    // alignment was never the thing under test — the test compared C against a
    // second implementation of C living in this file. Passing the raw spec makes
    // the geometry assertions below a real differential check.
    let header = decoder.header();
    let img_h: usize = header.height as usize;
    let clamped_h: usize = crop_h.min(img_h.saturating_sub(crop_y.min(img_h)));
    // Track C's own choice instead of pinning this off. `use_merged_upsample`
    // returns TRUE exactly when fancy upsampling is off and the layout is
    // 2hNv YCbCr→RGB (jdmaster.c:43-71), which `-nosmooth` selects for 420/422 —
    // so hardcoding `false` compared C's merged path against our non-merged one
    // and left our merged crop path untested for half the grid.
    decoder.set_merged_upsample(nosmooth);
    decoder.set_crop_region(crop_x, crop_y, crop_w, clamped_h);
    let rust_img = decoder
        .decode_image()
        .unwrap_or_else(|e| panic!("[{scenario_label}] Rust decode_image failed: {e}"));

    // P4-116: this used to be an exclusion, but it is decided *after* the Rust
    // decode and *before* djpeg runs, so a crop regression that clamped Rust to
    // empty where C yields pixels would be recorded as "deliberately out of
    // scope". Exclusions may only rest on facts derivable from the matrix
    // definition — `crop_w == 0 || crop_h == 0` above qualifies, this does not.
    assert!(
        rust_img.width > 0 && rust_img.height > 0,
        "[{scenario_label}] Rust clamped crop {crop_w}x{crop_h}+{crop_x}+{crop_y} to an \
         empty region; C was never consulted, so this cannot be assumed benign"
    );

    // C djpeg side
    let jpeg_tmp: helpers::TempFile =
        helpers::TempFile::new(&format!("{scenario_label}_{samp_name}.jpg"));
    let ppm_tmp: helpers::TempFile =
        helpers::TempFile::new(&format!("{scenario_label}_{samp_name}.ppm"));
    std::fs::write(jpeg_tmp.path(), jpeg_data)
        .unwrap_or_else(|e| panic!("[{scenario_label}] failed to write jpeg tmp: {e}"));

    let crop_arg: String = format!("{crop_w}x{crop_h}+{crop_x}+{crop_y}");

    let mut djpeg_args: Vec<&str> = Vec::new();
    if nosmooth {
        djpeg_args.push("-nosmooth");
    }
    djpeg_args.push("-rgb");
    djpeg_args.push("-crop");

    // Build crop_arg as owned String before referencing
    let output: std::process::Output = Command::new(djpeg)
        .args(&djpeg_args)
        .arg(&crop_arg)
        .arg("-outfile")
        .arg(ppm_tmp.path())
        .arg(jpeg_tmp.path())
        .output()
        .unwrap_or_else(|e| panic!("[{scenario_label}] failed to spawn djpeg: {e}"));

    // P4-116: an oracle that refuses a crop spec this matrix claims to support
    // is a defect in the spec we generated, and returning `false` here made the
    // whole case vanish from the count. Callers now record every case, so an
    // out-of-image crop must be excluded from the plan up front, not here.
    assert!(
        output.status.success(),
        "[{scenario_label}] djpeg -crop {crop_arg} failed: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );

    let ppm_data: Vec<u8> = std::fs::read(ppm_tmp.path())
        .unwrap_or_else(|e| panic!("[{scenario_label}] failed to read djpeg PPM: {e}"));

    // djpeg -crop on grayscale JPEG may output PGM (P5)
    let (c_w, c_h, c_pixels, channels) = if ppm_data.starts_with(b"P5") {
        let (w, h, pix) = helpers::parse_pgm(&ppm_data)
            .unwrap_or_else(|| panic!("[{scenario_label}] failed to parse PGM from djpeg"));
        (w, h, pix, 1usize)
    } else {
        let (w, h, pix) = helpers::parse_ppm(&ppm_data)
            .unwrap_or_else(|| panic!("[{scenario_label}] failed to parse PPM from djpeg"));
        (w, h, pix, 3usize)
    };

    // Both sides got the same request and both align it the same way, so both
    // buffers start at the same image column. `jpeg_crop_scanline` snaps the
    // requested x down to an iMCU boundary and widens the region so the right
    // edge stays put (jdapistd.c:245-255), and djpeg emits exactly that aligned
    // region — its column 0 is the snapped origin, not image column 0. Applying
    // `crop_x` as a column offset here would therefore count the alignment twice
    // and run off the end of the C buffer, which is what P4-168 fixed.
    let effective_w: usize = rust_img.width;
    let effective_h: usize = rust_img.height;

    assert_eq!(
        c_h, effective_h,
        "[{scenario_label}] cropped height mismatch — Rust and C disagreeing on \
         geometry is the defect, not a reason to skip"
    );
    assert_eq!(
        c_w, effective_w,
        "[{scenario_label}] cropped width mismatch — Rust and C disagreeing on \
         geometry is the defect, not a reason to skip"
    );

    let rust_pixels: &[u8] = &rust_img.data;
    let bpp: usize = rust_img.pixel_format.bytes_per_pixel();

    let c_extracted: Vec<u8> = if channels == bpp {
        c_pixels
    } else if channels == 3 && bpp == 1 {
        // djpeg -rgb emits a 3-channel PPM even for a grayscale JPEG, while Rust
        // keeps the single channel. R == G == B there, so compare against R.
        c_pixels.iter().step_by(3).copied().collect()
    } else {
        panic!(
            "[{scenario_label}] channel mismatch c_ch={channels} bpp={bpp} \
             (geometry {c_w}x{c_h})"
        );
    };

    assert_eq!(
        rust_pixels.len(),
        c_extracted.len(),
        "[{scenario_label}] cropped data length mismatch"
    );

    let max_diff: u8 = helpers::pixel_max_diff(rust_pixels, &c_extracted);
    assert_eq!(
        max_diff,
        0,
        "[{scenario_label}] samp={samp_name} crop={crop_w}x{crop_h}+{crop_x}+{crop_y} nosmooth={nosmooth}: Rust vs C djpeg max_diff={max_diff} (must be 0)"
    );

    None
}

// ===========================================================================
// Test: quick subset
// ===========================================================================

/// Quick crop boundary test — representative subset of the full C croptest.in grid.
///
/// Covers: baseline only, no nosmooth, no colors, Y=[0,1,8,16], H=[8,16],
/// samp=[444]. Runs 8 scenarios.
#[test]
fn c_croptest_quick() {
    let cjpeg: Option<PathBuf> = helpers::cjpeg_path();
    let djpeg: PathBuf = require_c_tool!("djpeg");

    // Verify djpeg supports -crop
    {
        let probe_jpeg: Vec<u8> = compress(
            &generate_test_pixels(),
            WIDTH,
            HEIGHT,
            PixelFormat::Rgb,
            75,
            Subsampling::S444,
        )
        .expect("probe compress must succeed");
        let probe_tmp: helpers::TempFile = helpers::TempFile::new("probe.jpg");
        let probe_out: helpers::TempFile = helpers::TempFile::new("probe.ppm");
        probe_tmp.write_bytes(&probe_jpeg);
        let probe_result: std::process::Output = Command::new(&djpeg)
            .args(["-crop", "8x8+0+0", "-outfile"])
            .arg(probe_out.path())
            .arg(probe_tmp.path())
            .output()
            .expect("failed to spawn djpeg probe");
        if !probe_result.status.success() {
            // P4-116: CI provisions libjpeg-turbo 3.x, which has -crop.
            assert!(
                !helpers::is_ci(),
                "CI must provide a djpeg supporting -crop"
            );
            eprintln!("SKIP: djpeg does not support -crop flag");
            return;
        }
    }

    let pixels: Vec<u8> = generate_test_pixels();

    // Quick subset: Y in [0, 1, 8, 16], H in [8, 16], samp = S444 only.
    // S420 and GRAY get their own tiers below so a failure names the layout.
    // `y_iter = 1` is the only value here with a non-zero crop x, which here
    // means a non-zero `first_iMCU_col` through the crop-aware decode path
    // rather than the column-anchoring case `c_croptest_quick_gray` covers.
    let y_values: &[usize] = &[0, 1, 8, 16];
    let h_values: &[usize] = &[8, 16];
    let samp_indices: &[usize] = &[4]; // 444 only

    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new(
        "c_croptest_quick",
        y_values.len() * h_values.len() * samp_indices.len(),
    );
    for &y_iter in y_values {
        for &h_iter in h_values {
            let (crop_w, crop_h, crop_x, crop_y) = compute_crop_spec(y_iter, h_iter);

            for &si in samp_indices {
                let samp: &SampConfig = &SAMP_CONFIGS[si];

                // Generate test JPEG
                let jpeg_data: Vec<u8> = match &cjpeg {
                    Some(cj) => make_test_jpeg_with_c(cj, &pixels, samp, false),
                    None => make_test_jpeg_rust(&pixels, samp),
                };

                let label: String = format!("quick_y{y_iter}_h{h_iter}_{}", samp.name);

                match run_crop_scenario(
                    &djpeg, &jpeg_data, crop_w, crop_h, crop_x, crop_y,
                    false, // nosmooth=false for quick test
                    samp.name, &label,
                ) {
                    None => tally.compared(),
                    Some(reason) => tally.excluded(reason),
                }
            }
        }
    }

    tally.finish();
}

/// Quick crop test for S420 — the subsampled layer of the crop path.
///
/// Kept as its own tier so a failure names the layout rather than pointing at
/// the shared 444 grid. It once diverged from C by max_diff=75 at the edge
/// blocks; crop-aware upsampling closed that (issue #164) and the shared
/// `max_diff == 0` assertion has held it since.
#[test]
fn c_croptest_quick_420() {
    let cjpeg: Option<PathBuf> = helpers::cjpeg_path();
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let pixels: Vec<u8> = generate_test_pixels();
    let y_values: &[usize] = &[0, 1, 8, 16];
    let h_values: &[usize] = &[8, 16];
    let samp_indices: &[usize] = &[1]; // 420 only

    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new(
        "c_croptest_quick_420",
        y_values.len() * h_values.len() * samp_indices.len(),
    );
    for &y_iter in y_values {
        for &h_iter in h_values {
            let (crop_w, crop_h, crop_x, crop_y) = compute_crop_spec(y_iter, h_iter);
            for &si in samp_indices {
                let samp: &SampConfig = &SAMP_CONFIGS[si];
                let jpeg_data: Vec<u8> = match &cjpeg {
                    Some(cj) => make_test_jpeg_with_c(cj, &pixels, samp, false),
                    None => make_test_jpeg_rust(&pixels, samp),
                };
                let label: String = format!("quick420_y{y_iter}_h{h_iter}_{}", samp.name);
                match run_crop_scenario(
                    &djpeg, &jpeg_data, crop_w, crop_h, crop_x, crop_y, false, samp.name, &label,
                ) {
                    None => tally.compared(),
                    Some(reason) => tally.excluded(reason),
                }
            }
        }
    }
    tally.finish();
}

/// Quick crop test for a grayscale JPEG.
///
/// `djpeg -rgb` emits a 3-channel PPM here while Rust keeps one channel, so the
/// comparison reads C's R channel (R == G == B for grayscale).
///
/// `y_iter = 1` is load-bearing: `compute_crop_spec` derives x from
/// `(y_iter * 16) % 128`, so every other value here yields `crop_x == 0` and a
/// harness that mis-anchors the C columns still compares equal. Only a non-zero
/// crop x exercises the offset, which is how the full grid caught the buffer
/// overrun that the quick tier missed.
#[test]
fn c_croptest_quick_gray() {
    let cjpeg: Option<PathBuf> = helpers::cjpeg_path();
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let pixels: Vec<u8> = generate_test_pixels();
    let y_values: &[usize] = &[0, 1, 8, 16];
    let h_values: &[usize] = &[8, 16];

    let mut tally: helpers::ComparisonTally =
        helpers::ComparisonTally::new("c_croptest_quick_gray", y_values.len() * h_values.len());
    for &y_iter in y_values {
        for &h_iter in h_values {
            let (crop_w, crop_h, crop_x, crop_y) = compute_crop_spec(y_iter, h_iter);
            let samp: &SampConfig = &SAMP_CONFIGS[0]; // GRAY
            let jpeg_data: Vec<u8> = match &cjpeg {
                Some(cj) => make_test_jpeg_with_c(cj, &pixels, samp, false),
                None => make_test_jpeg_rust(&pixels, samp),
            };
            let label: String = format!("quick_gray_y{y_iter}_h{h_iter}");
            match run_crop_scenario(
                &djpeg, &jpeg_data, crop_w, crop_h, crop_x, crop_y, false, samp.name, &label,
            ) {
                None => tally.compared(),
                Some(reason) => tally.excluded(reason),
            }
        }
    }
    tally.finish();
}

/// Crop specs whose x is *not* an iMCU multiple — the alignment path itself.
///
/// `croptest.in`'s formula derives x from `(y_iter * 16) % 128`, so every x in
/// the mirrored grid is already a multiple of 16 while `align` is 8 or 16.
/// Nothing in that grid ever makes `jpeg_crop_scanline` snap the origin or widen
/// the region (jdapistd.c:245-255) — both sides just pass the request through.
/// These specs do: x=20 with align 8 snaps to 16 and widens by 4, x=12 with
/// align 16 snaps to 0 and widens by 12. Since both sides now receive the raw
/// request, that puts two things under test that the mirrored grid cannot reach:
/// our decoder's own snap-and-widen against C's (the width assertion fails if
/// they disagree), and the offset-0 read against a genuinely shifted origin
/// rather than a trivially equal one.
#[test]
fn c_croptest_unaligned_x() {
    let cjpeg: Option<PathBuf> = helpers::cjpeg_path();
    let djpeg: PathBuf = require_c_tool!("djpeg");

    let pixels: Vec<u8> = generate_test_pixels();
    // (crop_w, crop_h, crop_x, crop_y) — each x is odd against align 8 and/or 16,
    // and every spec stays inside 128x95 so djpeg never refuses it.
    let specs: &[(usize, usize, usize, usize)] = &[
        (101, 4, 20, 1),
        (100, 8, 12, 3),
        (60, 5, 4, 0),
        (33, 6, 44, 7),
    ];
    let nosmooth_flags: &[bool] = &[false, true];

    let mut tally: helpers::ComparisonTally = helpers::ComparisonTally::new(
        "c_croptest_unaligned_x",
        specs.len() * SAMP_CONFIGS.len() * nosmooth_flags.len(),
    );
    for &(crop_w, crop_h, crop_x, crop_y) in specs {
        for samp in SAMP_CONFIGS {
            let jpeg_data: Vec<u8> = match &cjpeg {
                Some(cj) => make_test_jpeg_with_c(cj, &pixels, samp, false),
                None => make_test_jpeg_rust(&pixels, samp),
            };
            for &nosmooth in nosmooth_flags {
                let label: String = format!(
                    "unaligned_{crop_w}x{crop_h}+{crop_x}+{crop_y}_ns{}_{}",
                    nosmooth as u8, samp.name
                );
                match run_crop_scenario(
                    &djpeg, &jpeg_data, crop_w, crop_h, crop_x, crop_y, nosmooth, samp.name, &label,
                ) {
                    None => tally.compared(),
                    Some(reason) => tally.excluded(reason),
                }
            }
        }
    }
    tally.finish();
}

// ===========================================================================
// Test: full grid (feature = "full-c-parity")
// ===========================================================================

/// Full crop boundary test — exhaustive C croptest.in grid.
///
/// Covers: baseline + progressive, nosmooth on/off (nosmooth → fast upsample),
/// colors skipped (not implemented), Y=0..=16, H=1..=16, all 5 subsamplings.
/// Total: 2 (prog) × 2 (nosmooth) × 17 (Y) × 16 (H) × 5 (samp) = 5,440 per
/// colors loop iteration × 1 (colors skipped) = 5,440 scenarios. With colors
/// loop the C script also passes "-colors 256 -dither none -onepass" which we
/// skip, so 5,440 total (not 10,880).
#[test]
#[cfg(feature = "full-c-parity")]
fn c_croptest_full() {
    let cjpeg: Option<PathBuf> = helpers::cjpeg_path();
    let djpeg: PathBuf = require_c_tool!("djpeg");

    // Verify djpeg supports -crop
    {
        let probe_jpeg: Vec<u8> = compress(
            &generate_test_pixels(),
            WIDTH,
            HEIGHT,
            PixelFormat::Rgb,
            75,
            Subsampling::S444,
        )
        .expect("probe compress");
        let probe_tmp: helpers::TempFile = helpers::TempFile::new("full_probe.jpg");
        let probe_out: helpers::TempFile = helpers::TempFile::new("full_probe.ppm");
        probe_tmp.write_bytes(&probe_jpeg);
        let probe_result: std::process::Output = Command::new(&djpeg)
            .args(["-crop", "8x8+0+0", "-outfile"])
            .arg(probe_out.path())
            .arg(probe_tmp.path())
            .output()
            .expect("failed to spawn djpeg probe");
        if !probe_result.status.success() {
            // P4-116: CI provisions libjpeg-turbo 3.x, which has -crop.
            assert!(
                !helpers::is_ci(),
                "CI must provide a djpeg supporting -crop"
            );
            eprintln!("SKIP: djpeg does not support -crop flag");
            return;
        }
    }

    let pixels: Vec<u8> = generate_test_pixels();
    let prog_flags: &[bool] = &[false, true];
    let nosmooth_flags: &[bool] = &[false, true];

    // Exact accounting (P4-116). The plan is the full product of the C
    // script's axes; the colour-quantization half is excluded *by count*, so
    // the summary states exactly how many cases that missing feature costs
    // rather than letting them disappear.
    let cases_per_colors_value: usize = 17 * 16 * SAMP_CONFIGS.len();
    let planned: usize = prog_flags.len() * nosmooth_flags.len() * 2 * cases_per_colors_value;
    let mut tally: helpers::ComparisonTally =
        helpers::ComparisonTally::new("c_croptest_full", planned);

    for &prog in prog_flags {
        for &nosmooth in nosmooth_flags {
            // colors loop: "" and "-colors 256 -dither none -onepass"
            // We only run the "" iteration; the colors one is unimplemented.
            for colors_active in [false, true] {
                if colors_active {
                    tally.excluded_n(
                        cases_per_colors_value,
                        "color quantization (-colors 256 -dither none -onepass) \
                         is not implemented in Rust",
                    );
                    continue;
                }

                for y_iter in 0..=16usize {
                    for h_iter in 1..=16usize {
                        let (crop_w, crop_h, crop_x, crop_y) = compute_crop_spec(y_iter, h_iter);

                        for samp in SAMP_CONFIGS {
                            // Generate test JPEG
                            let jpeg_data: Vec<u8> = match &cjpeg {
                                Some(cj) => make_test_jpeg_with_c(cj, &pixels, samp, prog),
                                None => make_test_jpeg_rust(&pixels, samp),
                            };

                            let label: String = format!(
                                "full_prog{}_ns{}_y{y_iter}_h{h_iter}_{}",
                                prog as u8, nosmooth as u8, samp.name
                            );

                            match run_crop_scenario(
                                &djpeg, &jpeg_data, crop_w, crop_h, crop_x, crop_y, nosmooth,
                                samp.name, &label,
                            ) {
                                None => tally.compared(),
                                Some(reason) => tally.excluded(reason),
                            }
                        }
                    }
                }
            }
        }
    }

    tally.finish();
}
