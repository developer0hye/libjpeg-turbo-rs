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

/// Issue #345: `colorspace(Rgb)` used to take precedence over the mode
/// switches and discard them, so `progressive(true)` returned a baseline
/// Huffman stream with nothing to say it had not been applied.
///
/// C has none of that: `jcmaster.c` builds the scan script from the component
/// count and `jcarith.c` codes coefficients — neither looks at the colorspace.
/// So all four combinations are compared against the `cjpeg -rgb` invocation
/// that means the same thing.
#[test]
fn issue_345_rgb_direct_composes_with_every_mode() {
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

            let cases: Vec<(&str, Vec<u8>, Vec<&str>)> = vec![
                (
                    "progressive",
                    base()
                        .progressive(true)
                        .encode()
                        .expect("rgb-direct progressive"),
                    vec!["-progressive"],
                ),
                (
                    "arithmetic",
                    base()
                        .arithmetic(true)
                        .encode()
                        .expect("rgb-direct arithmetic"),
                    vec!["-arithmetic"],
                ),
                (
                    "arith-progressive",
                    base()
                        .arithmetic(true)
                        .progressive(true)
                        .encode()
                        .expect("rgb-direct arithmetic progressive"),
                    vec!["-arithmetic", "-progressive"],
                ),
                (
                    "lossless",
                    base().lossless(true).encode().expect("rgb-direct lossless"),
                    vec!["-lossless", "1,0"],
                ),
            ];

            for (label, ours, extra) in cases {
                let mut args: Vec<&str> = vec!["-quality", &quality_arg, "-baseline", "-rgb"];
                args.extend(extra);
                let theirs: Vec<u8> = helpers::encode_with_c_cjpeg(
                    &cjpeg,
                    &ppm,
                    &args,
                    &format!("i345_{label}_{width}x{height}_q{quality}"),
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
        GEOMETRIES.len() * QUALITIES.len() * 4,
        "the sweep must compare every case; a short run reads as a pass"
    );
    assert!(
        failures.is_empty(),
        "{} of {compared} JCS_RGB mode cases diverged from cjpeg (issue #345):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// The mode markers themselves, without a C tool, so the contract holds even
/// where cjpeg is unavailable. A baseline SOF0 here would mean the mode was
/// dropped — the exact failure #345 was about.
#[test]
fn issue_345_each_mode_writes_its_own_frame_marker() {
    let (width, height) = (32usize, 16usize);
    let pixels: Vec<u8> = rgb_pixels(width, height);

    for (label, marker, encoded) in [
        (
            "progressive",
            0xC2u8,
            Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .colorspace(ColorSpace::Rgb)
                .progressive(true)
                .encode()
                .expect("progressive"),
        ),
        (
            "arithmetic",
            0xC9,
            Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .colorspace(ColorSpace::Rgb)
                .arithmetic(true)
                .encode()
                .expect("arithmetic"),
        ),
        (
            "arith-progressive",
            0xCA,
            Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .colorspace(ColorSpace::Rgb)
                .arithmetic(true)
                .progressive(true)
                .encode()
                .expect("arithmetic progressive"),
        ),
        (
            "lossless",
            0xC3,
            Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .colorspace(ColorSpace::Rgb)
                .lossless(true)
                .encode()
                .expect("lossless"),
        ),
    ] {
        let found: bool = encoded
            .windows(2)
            .any(|window| window[0] == 0xFF && window[1] == marker);
        assert!(
            found,
            "colorspace(Rgb) + {label} did not write its SOF marker (FF{marker:02X}) — \
             the mode was dropped (#345)"
        );
        // Still JCS_RGB: Adobe APP14, never a JFIF APP0.
        assert_eq!(
            &encoded[..4],
            &[0xFF, 0xD8, 0xFF, 0xEE],
            "{label}: JCS_RGB must open SOI then Adobe APP14"
        );
    }
}

/// Two things the sweep above cannot reach, both found by review of the fix
/// rather than by the fix's own tests.
///
/// - **Row-based restarts.** Without an explicit sampling request, RGB-direct
///   defaults every component to 1x1, so its MCU is 8 pixels wide. Letting the
///   encoder's YCbCr-oriented 4:2:0 default leak into this path instead counts
///   a row against a 16-pixel MCU and lands the markers on the wrong rows —
///   visible only where `ceil(width/8) != ceil(width/16)`. Explicit RGB
///   component sampling is covered separately and must determine its own MCU
///   width.
/// - **16-bit quantization tables.** Below quality ~20 without `force_baseline`,
///   the scaled table exceeds 255 and needs 16-bit DQT entries, which SOF0
///   forbids. `cjpeg` switches to SOF1 and warns; writing SOF0 there produces a
///   non-conforming stream.
#[test]
fn issue_343_rgb_direct_row_restarts_and_16bit_tables_match_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");

    let mut failures: Vec<String> = Vec::new();

    // Widths where the 8-wide and 16-wide MCU counts differ, so a row interval
    // computed from the wrong MCU width is observable.
    for &(width, height) in &[(17usize, 16usize), (33, 24), (48, 32)] {
        let pixels: Vec<u8> = rgb_pixels(width, height);
        let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
        ppm.extend_from_slice(&pixels);

        for &rows in &[1u16, 2] {
            let ours: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(75)
                .colorspace(ColorSpace::Rgb)
                .restart_rows(rows)
                .encode()
                .expect("rgb-direct restart_rows");
            let rows_arg: String = rows.to_string();
            let theirs: Vec<u8> = helpers::encode_with_c_cjpeg(
                &cjpeg,
                &ppm,
                &[
                    "-quality",
                    "75",
                    "-dct",
                    "int",
                    "-baseline",
                    "-rgb",
                    "-restart",
                    &rows_arg,
                ],
                &format!("i343_rows{rows}_{width}x{height}"),
            );
            if ours != theirs {
                failures.push(format!(
                    "  restart_rows({rows}) {width}x{height}: ours={} c={}",
                    ours.len(),
                    theirs.len()
                ));
            }
        }

        // Quality 1 scales the Annex K table well past 255. `force_baseline`
        // is off by default here and in cjpeg, so both must emit SOF1.
        let ours: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
            .quality(1)
            .colorspace(ColorSpace::Rgb)
            .encode()
            .expect("rgb-direct q1");
        let theirs: Vec<u8> = helpers::encode_with_c_cjpeg(
            &cjpeg,
            &ppm,
            &["-quality", "1", "-dct", "int", "-rgb"],
            &format!("i343_q1_{width}x{height}"),
        );
        let sof1: bool = ours.windows(2).any(|window| window == [0xFF, 0xC1]);
        if !sof1 {
            failures.push(format!(
                "  q1 {width}x{height}: no SOF1 — 16-bit quantization tables are \
                 not legal in a baseline (SOF0) frame"
            ));
        }
        if ours != theirs {
            failures.push(format!(
                "  q1 {width}x{height}: ours={} c={}",
                ours.len(),
                theirs.len()
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "{} RGB-direct cases diverged from cjpeg (issue #343):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
