//! Differential fuzzing of YCbCr output (`Decoder::set_output_colorspace(
//! ColorSpace::YCbCr)`) against libjpeg's `out_color_space = JCS_YCbCr`,
//! through `examples/ycbcr_decode_c_oracle.c`.
//!
//! No other target reaches this path, which is how P4-244 (#683) — chroma
//! replicated instead of upsampled, and panics on two legal layouts —
//! survived. The fuzzed input is three option bytes followed by the JPEG:
//!
//! - byte 0: bits 0–3 pick a `ScalingFactor::SUPPORTED` entry; bit 4 set
//!   asks for box upsampling (`set_fast_upsample`, libjpeg's
//!   `do_fancy_upsampling = FALSE`); bit 5 set enables the crop below.
//! - bytes 1 and 2: the crop's x/width and y/height, as sixteenths of the
//!   scaled output (see [`crop_region`]).
//!
//! Asserts, when libjpeg decodes the input without a warning:
//! 1. Rust accepts it too (drop-in floor);
//! 2. the output dimensions agree;
//! 3. every Y, Cb and Cr sample is identical — YCbCr output involves no
//!    colour conversion, so there is no rounding to tolerate.
//!
//! Skips, with the reason at each `return`: arithmetic-coded streams (the
//! same open divergence `fuzz_decode_diff_c` skips), 12-bit and lossless
//! frames (P4-246: those decodes ignore the output colour space), and inputs
//! either decoder recovered from (their recovery strategies differ). A sample
//! mismatch is excused, unless `LTRS_FUZZ_INCLUDE_KNOWN=1`, when it has a
//! filed explanation: a scaled decode of coefficients no 8-bit image produces
//! (the scaled IDCTs saturate by design), or 0xFF fill before a stuffed byte
//! (P4-248).

#![no_main]

#[path = "common/c_oracle.rs"]
mod c_oracle;

use std::path::PathBuf;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use libjpeg_turbo_rs::{ColorSpace, Decoder, JpegError, ScalingFactor};

const MAX_FUZZ_PIXELS: u64 = 1_048_576;

fn oracle() -> &'static PathBuf {
    static ORACLE: OnceLock<PathBuf> = OnceLock::new();
    ORACLE.get_or_init(|| c_oracle::oracle_path("YCBCR_C_ORACLE"))
}

/// The crop as `(x, y, width, height)` in output pixels. Each byte's low
/// nibble places the start at `n/16` of the extent, the high nibble sets the
/// length as `n/15` of what remains (at least one pixel), so every region
/// fits the output as `djpeg -crop` requires.
fn crop_region(out_w: usize, out_h: usize, x_w: u8, y_h: u8) -> (usize, usize, usize, usize) {
    let axis = |extent: usize, spec: u8| -> (usize, usize) {
        let start: usize = usize::from(spec & 0x0F) * extent / 16;
        let length: usize = 1 + usize::from(spec >> 4) * (extent - start - 1) / 15;
        (start, length)
    };
    let (x, w) = axis(out_w, x_w);
    let (y, h) = axis(out_h, y_h);
    (x, y, w, h)
}

enum RustOutcome {
    /// `(width, height, samples, recovered)`.
    Pixels(usize, usize, Vec<u8>, bool),
    /// A deliberate, documented divergence (the #355 scan cap, P4-55).
    OutOfScope,
    Rejected,
}

fn rust_ycbcr(
    jpeg: &[u8],
    scale: ScalingFactor,
    fancy: bool,
    crop: Option<(usize, usize, usize, usize)>,
) -> RustOutcome {
    let mut decoder = match Decoder::new(jpeg) {
        Ok(decoder) => decoder,
        Err(JpegError::LimitExceeded { .. }) => return RustOutcome::OutOfScope,
        Err(_) => return RustOutcome::Rejected,
    };
    decoder.set_scale(scale);
    decoder.set_fast_upsample(!fancy);
    // Lenient and smoothed, as libjpeg decodes by default; strictness is
    // `fuzz_decompress`'s concern, not this differential's.
    decoder.set_lenient(true);
    decoder.set_block_smoothing(true);
    if let Some((x, y, w, h)) = crop {
        decoder.set_crop_region(x, y, w, h);
    }
    decoder.set_output_colorspace(ColorSpace::YCbCr);
    match decoder.decode_image() {
        Ok(image) => RustOutcome::Pixels(
            image.width,
            image.height,
            image.data,
            !image.warnings.is_empty(),
        ),
        Err(_) => RustOutcome::Rejected,
    }
}

fuzz_target!(|data: &[u8]| {
    c_oracle::ensure_sigpipe_ignored();
    let oracle: &PathBuf = oracle();
    let [options, crop_x_w, crop_y_h, ref jpeg @ ..] = *data else {
        return;
    };
    let scale: ScalingFactor = ScalingFactor::SUPPORTED[usize::from(options & 0x0F)];
    let fancy: bool = options & 0x10 == 0;

    let Ok(mut probe) = Decoder::new(jpeg) else {
        return;
    };
    let header = probe.header();
    let pixels: u64 = u64::from(header.width).saturating_mul(u64::from(header.height));
    if header.width == 0 || header.height == 0 || pixels > MAX_FUZZ_PIXELS {
        return;
    }
    // Known, filed divergence shared with fuzz_decode_diff_c: the arithmetic
    // decoder disagrees with libjpeg on some fuzz inputs mid-scan.
    if probe.is_arithmetic() && !c_oracle::include_known_gaps() {
        return;
    }
    // P4-246: 12-bit and lossless decodes ignore set_output_colorspace.
    if (header.precision != 8 || header.is_lossless) && !c_oracle::include_known_gaps() {
        return;
    }
    probe.set_scale(scale);
    let (out_w, out_h): (usize, usize) = (probe.output_width(), probe.output_height());
    if out_w == 0 || out_h == 0 {
        return;
    }
    let crop = (options & 0x20 != 0).then(|| crop_region(out_w, out_h, crop_x_w, crop_y_h));

    let mut args: Vec<String> = vec![
        "/dev/stdin".to_string(),
        scale.num().to_string(),
        scale.denom().to_string(),
        if fancy { "1" } else { "0" }.to_string(),
    ];
    if let Some((x, y, w, h)) = crop {
        args.extend([x, y, w, h].map(|v| v.to_string()));
    }
    let c_result = c_oracle::run_oracle(oracle, &args, jpeg);
    let rust_result = rust_ycbcr(jpeg, scale, fancy, crop);

    let label = format!(
        "scale {}/{} fancy={fancy} crop={crop:?}",
        scale.num(),
        scale.denom()
    );
    match (c_result, rust_result) {
        // libjpeg refused, or recovered from corruption: no reference.
        (None, _) => {}
        (Some(c), _) if c.warned => {}
        (_, RustOutcome::OutOfScope) => {}
        (Some(_), RustOutcome::Rejected) => {
            panic!("drop-in regression: libjpeg decoded YCbCr output cleanly ({label}) but Rust rejected it");
        }
        (Some(c), RustOutcome::Pixels(rw, rh, r_px, r_recovered)) => {
            let Some((header, c_px)) = c_oracle::split_header(&c.stdout) else {
                panic!("malformed YCbCr oracle output");
            };
            let (cw, ch): (usize, usize) = match header[..] {
                ["YCbCr", w, h] => (w.parse().unwrap(), h.parse().unwrap()),
                _ => panic!("malformed YCbCr oracle header {header:?}"),
            };
            assert_eq!(
                (rw, rh),
                (cw, ch),
                "YCbCr output dimensions diverge ({label})"
            );
            if r_recovered {
                return;
            }
            assert_eq!(r_px.len(), c_px.len(), "YCbCr output length ({label})");
            if r_px != c_px {
                if !c_oracle::include_known_gaps() {
                    // By design: out-of-range coefficients through a scaled
                    // IDCT (see max_dequantized_coefficient).
                    let scaled: bool = (scale.num(), scale.denom()) != (1, 1);
                    if scaled
                        && c_oracle::max_dequantized_coefficient(jpeg)
                            .is_some_and(|m| m > c_oracle::MAX_LEGAL_DEQUANTIZED)
                    {
                        return;
                    }
                    // P4-248: 0xFF fill before a stuffed byte.
                    if c_oracle::has_fill_before_stuffed_zero(jpeg) {
                        return;
                    }
                }
                let differing: usize = r_px.iter().zip(c_px).filter(|(a, b)| a != b).count();
                panic!(
                    "YCbCr output diverges from libjpeg ({label}, {cw}x{ch}): {differing} samples differ; first: {}",
                    c_oracle::first_differences(&r_px, c_px, 3, 6)
                );
            }
        }
    }
});
