//! Issue #343: `colorspace(Rgb)` silently dropped every builder option.
//!
//! `Encoder::encode` took an early return into `compress_rgb_direct`, whose
//! signature carried pixels, dimensions, quality and an ICC profile and
//! nothing else. Restart intervals, custom quantization and Huffman tables,
//! optimized Huffman, smoothing and the DCT method were all discarded, and the
//! caller got `Ok(bytes)` either way. So did the comment / EXIF / saved-marker
//! injection that every other colorspace runs.
//!
//! This is #313 in a second colorspace, from the same cause: an entry point
//! whose signature cannot express the option set, sitting behind an early
//! return. Both now share one `compress_direct_planar`.
//!
//! Unlike CMYK this path *is* cross-validatable — `cjpeg -rgb` reads ordinary
//! PPM — so every case here is checked byte-for-byte against C rather than
//! merely against itself.

mod helpers;

use libjpeg_turbo_rs::{ColorSpace, DctMethod, Encoder, HuffmanTableDef, PixelFormat};

const GEOMETRIES: &[(usize, usize)] = &[(48, 32), (17, 17), (7, 16), (1, 1), (64, 48)];
const QUALITIES: &[u8] = &[25, 75, 95];

fn rgb_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x3f) - 32;
            let offset: usize = (y * width + x) * 3;
            pixels[offset] = ((x * 255 / width.max(1)) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 1] = ((y * 255 / height.max(1)) as i32 - noise).clamp(0, 255) as u8;
            pixels[offset + 2] = (((x ^ y) & 0xff) as i32 + noise).clamp(0, 255) as u8;
        }
    }
    pixels
}

fn coarse_quant() -> [u16; 64] {
    let mut table: [u16; 64] = [0; 64];
    for (index, entry) in table.iter_mut().enumerate() {
        *entry = 40 + index as u16;
    }
    table
}

/// Each option applied alone, byte-compared against the `cjpeg -rgb`
/// invocation that means the same thing.
#[test]
fn issue_343_rgb_direct_options_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");

    let mut failures: Vec<String> = Vec::new();
    let mut compared: usize = 0;

    for &(width, height) in GEOMETRIES {
        let pixels: Vec<u8> = rgb_pixels(width, height);
        let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
        ppm.extend_from_slice(&pixels);

        for &quality in QUALITIES {
            let quality_arg: String = quality.to_string();
            let base = || {
                Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                    .quality(quality)
                    .colorspace(ColorSpace::Rgb)
            };

            // (label, our bytes, cjpeg's extra arguments)
            let cases: Vec<(&str, Vec<u8>, Vec<&str>)> = vec![
                ("plain", base().encode().expect("rgb-direct encode"), vec![]),
                (
                    "dct-fast",
                    base()
                        .dct_method(DctMethod::IsFast)
                        .encode()
                        .expect("rgb-direct ifast"),
                    vec!["-dct", "fast"],
                ),
                (
                    "restart3",
                    base()
                        .restart_blocks(3)
                        .encode()
                        .expect("rgb-direct restart"),
                    vec!["-restart", "3B"],
                ),
                (
                    "optimize",
                    base()
                        .optimize_huffman(true)
                        .encode()
                        .expect("rgb-direct optimize"),
                    vec!["-optimize"],
                ),
                (
                    "smooth50",
                    base()
                        .smoothing_factor(50)
                        .encode()
                        .expect("rgb-direct smoothing"),
                    vec!["-smooth", "50"],
                ),
            ];

            for (label, ours, extra) in cases {
                let mut args: Vec<&str> =
                    vec!["-quality", &quality_arg, "-dct", "int", "-baseline", "-rgb"];
                // `-dct` is already present; a case that overrides it appends
                // its own pair, and cjpeg takes the last one.
                args.extend(extra);
                let theirs: Vec<u8> = helpers::encode_with_c_cjpeg(
                    &cjpeg,
                    &ppm,
                    &args,
                    &format!("i343_{label}_{width}x{height}_q{quality}"),
                );
                compared += 1;
                if ours != theirs {
                    let first_difference: usize = ours
                        .iter()
                        .zip(theirs.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    failures.push(format!(
                        "  {label} {width}x{height} q{quality}: ours={} c={} \
                         first diff at byte {first_difference}",
                        ours.len(),
                        theirs.len()
                    ));
                }
            }
        }
    }

    assert_eq!(
        compared,
        GEOMETRIES.len() * QUALITIES.len() * 5,
        "the sweep must compare every case; a short run reads as a pass"
    );
    assert!(
        failures.is_empty(),
        "{} of {compared} RGB-direct cases diverged from cjpeg (issue #343):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Custom tables have no `cjpeg` flag that expresses them, so they get the
/// weaker property the option matrix uses: setting them must change the bytes.
/// An option that changes nothing is being dropped.
#[test]
fn issue_343_rgb_direct_honours_custom_tables() {
    let (width, height) = (48usize, 32usize);
    let pixels: Vec<u8> = rgb_pixels(width, height);
    let base = || {
        Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(75)
            .colorspace(ColorSpace::Rgb)
    };
    let plain: Vec<u8> = base().encode().expect("rgb-direct encode");

    let with_quant: Vec<u8> = base()
        .quant_table(0, coarse_quant())
        .encode()
        .expect("rgb-direct custom quant");
    assert_ne!(
        with_quant, plain,
        "a custom quantization table changed nothing — it is being dropped (#343)"
    );
    assert!(
        with_quant.len() < plain.len(),
        "a much coarser table must shrink the file: custom={} plain={}",
        with_quant.len(),
        plain.len()
    );

    let mut dc_bits: [u8; 17] = [0; 17];
    dc_bits[4] = 16;
    let mut ac_bits: [u8; 17] = [0; 17];
    ac_bits[5] = 16;
    ac_bits[6] = 16;
    let with_huffman: Vec<u8> = base()
        .huffman_dc_table(
            0,
            HuffmanTableDef {
                bits: dc_bits,
                values: (0u8..16).collect(),
            },
        )
        .huffman_ac_table(
            0,
            HuffmanTableDef {
                bits: ac_bits,
                values: (0u8..32).collect(),
            },
        )
        .encode()
        .expect("rgb-direct custom huffman");
    assert_ne!(
        with_huffman, plain,
        "custom Huffman tables changed nothing — they are being dropped (#343)"
    );
}

/// The early return also skipped the metadata chain every other colorspace
/// runs, so a comment on an RGB-direct encode simply vanished.
#[test]
fn issue_343_rgb_direct_keeps_the_comment_and_exif() {
    let (width, height) = (32usize, 16usize);
    let pixels: Vec<u8> = rgb_pixels(width, height);

    let jpeg: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .comment("rgb-direct")
        .encode()
        .expect("rgb-direct encode with comment");

    let has_comment: bool = jpeg.windows(12).any(|window| {
        window[0] == 0xFF
            && window[1] == 0xFE
            && window[4..14.min(window.len())].starts_with(b"rgb")
    });
    assert!(
        has_comment,
        "the COM marker is missing — RGB-direct skipped the metadata chain (#343)"
    );
}

/// `colorspace(Rgb)` has always taken precedence over the mode switches, and
/// this test pins that so the precedence is a decision rather than an accident
/// of where the branch sits. JCS_RGB progressive and arithmetic are #345.
#[test]
fn issue_343_rgb_direct_still_wins_over_unimplemented_modes() {
    let (width, height) = (32usize, 16usize);
    let pixels: Vec<u8> = rgb_pixels(width, height);

    let baseline: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .encode()
        .expect("rgb-direct encode");

    for (label, encoder) in [
        (
            "progressive",
            Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .colorspace(ColorSpace::Rgb)
                .progressive(true),
        ),
        (
            "arithmetic",
            Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .colorspace(ColorSpace::Rgb)
                .arithmetic(true),
        ),
    ] {
        let encoded: Vec<u8> = encoder
            .encode()
            .unwrap_or_else(|error| panic!("colorspace(Rgb) + {label} failed: {error:?}"));
        assert_eq!(
            encoded, baseline,
            "colorspace(Rgb) + {label} must still produce the baseline \
             RGB-direct stream — that precedence predates #343 and changing it \
             is #345's call, not this fix's"
        );
    }
}
