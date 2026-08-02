//! Issue #327: `smoothing_factor` was a silent no-op for grayscale input.
//!
//! Before #327, the code now in `src/encode/pipeline_impl/optimized.rs` gated
//! full-size smoothing on `!is_grayscale`.
//! C selects `fullsize_smooth_downsample` for **every** component sampled at
//! the maximum factors (`jcsample.c:506-513`), which for a single-component
//! image is the grayscale plane itself — so `cjpeg -grayscale -smooth 50`
//! differs from `-smooth 0` while ours did not.
//!
//! Only the *luma* gate was wrong. The adjacent `use_smooth_chroma` gate keeps
//! its `!is_grayscale` term correctly: a grayscale image has no chroma planes.
//!
//! The bug was invisible until #322 was fixed. Before that, grayscale +
//! smoothing routed through the optimized-Huffman path unconditionally, and
//! those optimal tables changed the bytes — so smoothing *appeared* to do
//! something while contributing nothing.

mod helpers;

use libjpeg_turbo_rs::{Encoder, PixelFormat};

/// Noisy content: smoothing is a low-pass filter, so a smooth ramp would barely
/// register the difference this test exists to detect.
fn grayscale_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = vec![0u8; width * height];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x3f) - 32;
            buffer[y * width + x] = ((x * 255 / width) as i32 + noise).clamp(0, 255) as u8;
        }
    }
    buffer
}

fn encode_grayscale(pixels: &[u8], width: usize, height: usize, smoothing: u8) -> Vec<u8> {
    Encoder::new(pixels, width, height, PixelFormat::Grayscale)
        .quality(75)
        .smoothing_factor(smoothing)
        .encode()
        .unwrap_or_else(|error| panic!("grayscale smooth={smoothing} encode failed: {error:?}"))
}

#[test]
fn issue_327_grayscale_smoothing_has_an_effect() {
    let (width, height) = (48usize, 32usize);
    let pixels: Vec<u8> = grayscale_pixels(width, height);

    let unsmoothed: Vec<u8> = encode_grayscale(&pixels, width, height, 0);
    // Starts at 2, not 1: C produces byte-identical output for `-smooth 0` and
    // `-smooth 1` on this image (both 811 bytes), because at factor 1 the
    // filter weights — `memberscale = 16384 - factor * 80`, `neighscale =
    // factor * 16` (`jcsample.c:338-339`) — round away on this content.
    // Requiring an effect there would assert something C does not do.
    for smoothing in [2u8, 25, 50, 100] {
        let smoothed: Vec<u8> = encode_grayscale(&pixels, width, height, smoothing);
        assert_ne!(
            smoothed, unsmoothed,
            "smoothing_factor({smoothing}) produced output identical to \
             smoothing_factor(0) on grayscale — the option is being ignored"
        );
    }
}

#[test]
fn issue_327_grayscale_smoothing_matches_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");

    // Mixes MCU-aligned and partial geometries: smoothing reads neighbouring
    // rows, so the image edges are where an off-by-one would surface.
    let geometries: &[(usize, usize)] = &[(48, 32), (17, 17), (8, 8), (1, 1), (64, 48), (33, 25)];
    let smoothings: &[u8] = &[0, 1, 25, 50, 100];

    let mut failures: Vec<String> = Vec::new();

    for &(width, height) in geometries {
        let pixels: Vec<u8> = grayscale_pixels(width, height);
        let mut pgm: Vec<u8> = format!("P5\n{width} {height}\n255\n").into_bytes();
        pgm.extend_from_slice(&pixels);

        for &smoothing in smoothings {
            let rust_jpeg: Vec<u8> = encode_grayscale(&pixels, width, height, smoothing);
            let smoothing_arg: String = smoothing.to_string();
            let mut args: Vec<&str> =
                vec!["-quality", "75", "-dct", "int", "-baseline", "-grayscale"];
            // `-smooth 0` is the default; passing it explicitly is equivalent,
            // but omitting it keeps the invocation closest to a plain encode.
            if smoothing > 0 {
                args.push("-smooth");
                args.push(&smoothing_arg);
            }
            let c_jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
                &cjpeg,
                &pgm,
                &args,
                &format!("issue327_{width}x{height}_sm{smoothing}"),
            );

            if rust_jpeg != c_jpeg {
                failures.push(format!(
                    "  {width}x{height} smooth={smoothing}: rust={} c={}",
                    rust_jpeg.len(),
                    c_jpeg.len()
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "grayscale smoothing diverged from cjpeg in {} of {} cases (issue #327):\n{}",
        failures.len(),
        geometries.len() * smoothings.len(),
        failures.join("\n")
    );
}
