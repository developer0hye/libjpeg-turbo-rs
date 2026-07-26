//! Issue #330: `DctMethod::IsFast` produced output both lower quality *and*
//! larger than C's `-dct fast`. Also the measurement behind #319 / P4-44.
//!
//! The fused SIMD extract+FDCT+quantize kernels hardcode the **islow**
//! transform. `ifast` and `float` come with divisor tables scaled for their own
//! transforms, so feeding islow coefficients to those divisors mis-scaled every
//! output by the AA&N factor. Several call sites guarded against this
//! (`encode_single_block`, `encode_downsampled_chroma_block`) and several did
//! not — `encode_mcu_{444,422,420}_x86_64`, `encode_mcu_420_half_chroma`,
//! `fdct_quantize_block`, `fdct_quantize_chroma_h2v1`, and the AVX2 4:2:0 row
//! path — so the defect appeared only where a SIMD shortcut existed.
//!
//! That is why the symptom looked selective: grayscale and the 4-factor
//! subsamplings (which take generic paths) matched C, while 4:4:4, 4:2:2 and
//! 4:2:0 did not. The guard is now a single `may_use_islow_simd_kernel` helper
//! applied at every site.
//!
//! # What each DCT method guarantees
//!
//! - `IsLow` — byte-identical to `cjpeg -dct int`.
//! - `IsFast` — byte-identical to `cjpeg -dct fast`.
//! - `Float` — *not* byte-identical. It matches C's quality and size but
//!   differs in the low bits from floating-point operation ordering; see
//!   `float_is_pixel_equivalent_to_cjpeg` for the measured bound.

mod helpers;

use libjpeg_turbo_rs::encode::pipeline::{compress_with_params, CompressParams};
use libjpeg_turbo_rs::{decompress_to, DctMethod, PixelFormat, Subsampling};

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

const SUBSAMPLINGS: &[(Subsampling, &str)] = &[
    (Subsampling::S444, "1x1"),
    (Subsampling::S422, "2x1"),
    (Subsampling::S420, "2x2"),
    (Subsampling::S440, "1x2"),
    (Subsampling::S441, "1x4"),
    (Subsampling::S411, "4x1"),
    (Subsampling::S410, "4x2"),
    (Subsampling::S24, "2x4"),
];

fn encode(
    raw: &[u8],
    width: usize,
    height: usize,
    format: PixelFormat,
    quality: u8,
    subsampling: Subsampling,
    dct_method: DctMethod,
) -> Vec<u8> {
    compress_with_params(
        &CompressParams::new(raw, width, height, format, quality, subsampling)
            .dct_method(dct_method),
    )
    .unwrap_or_else(|error| panic!("{dct_method:?} encode failed: {error:?}"))
}

/// `int` and `fast` must both be byte-identical to C. The subsampling axis
/// matters because the bug only appeared where a SIMD shortcut existed.
#[test]
fn issue_330_int_and_fast_match_cjpeg_byte_for_byte() {
    let cjpeg = require_c_tool!("cjpeg");

    let geometries: &[(usize, usize)] = &[(64, 48), (17, 17), (32, 2), (48, 48), (16, 16)];
    // q100 and 16x16 are here on purpose: those are the exact cases #319
    // reported diverging between x86_64 and aarch64 for `ifast` and `float`.
    let qualities: &[u8] = &[25, 75, 95, 100];
    let methods: &[(DctMethod, &str)] = &[(DctMethod::IsLow, "int"), (DctMethod::IsFast, "fast")];

    let mut failures: Vec<String> = Vec::new();

    for &(dct_method, dct_name) in methods {
        for &grayscale in &[false, true] {
            let channels: usize = if grayscale { 1 } else { 3 };
            let format: PixelFormat = if grayscale {
                PixelFormat::Grayscale
            } else {
                PixelFormat::Rgb
            };
            for &(subsampling, sample) in SUBSAMPLINGS {
                if grayscale && sample != "1x1" {
                    continue;
                }
                for &(width, height) in geometries {
                    for &quality in qualities {
                        let raw: Vec<u8> = pixels(width, height, channels);
                        let rust: Vec<u8> = encode(
                            &raw,
                            width,
                            height,
                            format,
                            quality,
                            subsampling,
                            dct_method,
                        );

                        let magic: &str = if grayscale { "P5" } else { "P6" };
                        let mut pnm: Vec<u8> =
                            format!("{magic}\n{width} {height}\n255\n").into_bytes();
                        pnm.extend_from_slice(&raw);
                        let quality_arg: String = quality.to_string();
                        let mut args: Vec<&str> =
                            vec!["-quality", &quality_arg, "-dct", dct_name, "-baseline"];
                        if grayscale {
                            args.push("-grayscale");
                        } else {
                            args.push("-sample");
                            args.push(sample);
                        }
                        let c: Vec<u8> = helpers::encode_with_c_cjpeg(
                            &cjpeg,
                            &pnm,
                            &args,
                            &format!("i330_{dct_name}_{width}x{height}_{sample}"),
                        );

                        if rust != c {
                            failures.push(format!(
                                "  -dct {dct_name} {width}x{height} {sample} q{quality} \
                                 gray={grayscale}: rust={} c={}",
                                rust.len(),
                                c.len()
                            ));
                        }
                    }
                }
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} of the swept cases diverged from cjpeg (issue #330):\n{}",
        failures.len(),
        failures
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// The defining symptom of #330: `fast` was *simultaneously* worse and bigger.
///
/// Upstream's AA&N transform costs almost nothing in quality, so `fast` should
/// sit within a hair of `int` on both axes. Before the fix ours had 2.5x the
/// error and a 22% larger file — which no tradeoff explains, and which a
/// byte-comparison test alone would report without making the severity clear.
#[test]
fn issue_330_fast_is_not_worse_and_bigger_than_int() {
    let (width, height) = (64usize, 48usize);
    let raw: Vec<u8> = pixels(width, height, 3);

    let mean_error = |encoded: &[u8]| -> f64 {
        let decoded = decompress_to(encoded, PixelFormat::Rgb).expect("decode");
        let total: u64 = decoded
            .data
            .iter()
            .zip(raw.iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs() as u64)
            .sum();
        total as f64 / raw.len() as f64
    };

    let int_jpeg: Vec<u8> = encode(
        &raw,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
        DctMethod::IsLow,
    );
    for dct_method in [DctMethod::IsFast, DctMethod::Float] {
        let jpeg: Vec<u8> = encode(
            &raw,
            width,
            height,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
            dct_method,
        );
        let (error, int_error) = (mean_error(&jpeg), mean_error(&int_jpeg));
        // Generous bounds: the point is to catch a scaling defect (2.5x error,
        // +22% size), not to pin exact rounding.
        assert!(
            error < int_error * 1.25,
            "{dct_method:?}: mean error {error:.3} is far worse than int's \
             {int_error:.3} — the divisors are mis-scaled"
        );
        assert!(
            jpeg.len() < int_jpeg.len() * 5 / 4,
            "{dct_method:?}: {} bytes vs int's {} — the divisors are mis-scaled",
            jpeg.len(),
            int_jpeg.len()
        );
    }
}

/// #319 asked whether the aarch64 backend matches `cjpeg` the way x86_64 does.
/// Every test in this file is unguarded, so the `Test (linux-aarch64 NEON)` CI
/// leg answers that on each run — but only for the input orders the sweep above
/// uses, and #319's divergences were all **BGR** at 16x16 q100.
///
/// A channel order that reaches a different SIMD colour-conversion kernel is
/// exactly where a backend difference would hide. cjpeg has no BGR input, so
/// the reference is its RGB encode of the same pixels in the order it reads.
#[test]
fn issue_319_bgr_input_matches_cjpeg_on_every_backend() {
    let cjpeg = require_c_tool!("cjpeg");

    let mut failures: Vec<String> = Vec::new();
    let mut compared: usize = 0;
    for &(width, height) in &[(16usize, 16usize), (17, 17), (64, 48)] {
        let rgb: Vec<u8> = pixels(width, height, 3);
        let bgr: Vec<u8> = rgb
            .chunks_exact(3)
            .flat_map(|pixel| [pixel[2], pixel[1], pixel[0]])
            .collect();

        for &(subsampling, sample) in SUBSAMPLINGS {
            for &quality in &[75u8, 100] {
                for &(dct_method, dct_name) in
                    &[(DctMethod::IsLow, "int"), (DctMethod::IsFast, "fast")]
                {
                    let ours: Vec<u8> = encode(
                        &bgr,
                        width,
                        height,
                        PixelFormat::Bgr,
                        quality,
                        subsampling,
                        dct_method,
                    );

                    let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
                    ppm.extend_from_slice(&rgb);
                    let quality_arg: String = quality.to_string();
                    let theirs: Vec<u8> = helpers::encode_with_c_cjpeg(
                        &cjpeg,
                        &ppm,
                        &[
                            "-quality",
                            &quality_arg,
                            "-dct",
                            dct_name,
                            "-baseline",
                            "-sample",
                            sample,
                        ],
                        &format!("i319_bgr_{dct_name}_{width}x{height}_{sample}"),
                    );

                    compared += 1;
                    if ours != theirs {
                        failures.push(format!(
                            "  BGR -dct {dct_name} {width}x{height} {sample} q{quality}: \
                             ours={} c={}",
                            ours.len(),
                            theirs.len()
                        ));
                    }
                }
            }
        }
    }

    assert_eq!(
        compared,
        3 * SUBSAMPLINGS.len() * 2 * 2,
        "the sweep must compare every case; a short run reads as a pass"
    );
    assert!(
        failures.is_empty(),
        "{} BGR cases diverged from cjpeg — on aarch64 that is #319's open \
         question, on x86_64 it is a regression:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `float` is deliberately *not* claimed byte-exact. This pins the guarantee
/// that is actually made, so the claim in the module docs stays honest.
#[test]
fn float_is_pixel_equivalent_to_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");
    let (width, height) = (64usize, 48usize);
    let raw: Vec<u8> = pixels(width, height, 3);

    let rust: Vec<u8> = encode(
        &raw,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
        DctMethod::Float,
    );
    let mut pnm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
    pnm.extend_from_slice(&raw);
    let c: Vec<u8> = helpers::encode_with_c_cjpeg(
        &cjpeg,
        &pnm,
        &[
            "-quality",
            "75",
            "-dct",
            "float",
            "-baseline",
            "-sample",
            "2x2",
        ],
        "i330_float",
    );

    let rust_decoded = decompress_to(&rust, PixelFormat::Rgb).expect("decode rust");
    let c_decoded = decompress_to(&c, PixelFormat::Rgb).expect("decode c");
    let max_difference: i32 = rust_decoded
        .data
        .iter()
        .zip(c_decoded.data.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .max()
        .unwrap_or(0);

    // Measured 7 on this fixture; the bound is that plus margin. If this ever
    // reaches islow-like divergence it is a real defect, not float ordering.
    assert!(
        max_difference <= 12,
        "float diverged from cjpeg by {max_difference} per sample — that is \
         beyond floating-point ordering and indicates a real defect"
    );
}
