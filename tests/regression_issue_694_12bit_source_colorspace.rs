//! Issue #694: 12-bit decode never classified the source colour space.
//!
//! `decompress_12bit` converted every three-component frame from YCbCr to RGB
//! and copied every four-component frame through unchanged. libjpeg
//! classifies the source first (`jdapimin.c` `default_decompress_parms`), as
//! the 8-bit decoder does through `ColorSpace::classify_source`: an RGB-coded
//! frame (`cjpeg -rgb`: Adobe transform 0, component IDs `R`, `G`, `B`) is
//! output without conversion, and a YCCK frame (Adobe transform 2) is
//! converted to CMYK (`jdcolor.c` `ycck_cmyk_convert`). Before the fix the
//! RGB frame was colour-converted a second time and the YCCK frame came back
//! as YCCK, both with `Ok`.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

use helpers::c_oracle::{
    cmyk12_c_oracle, decode_with_cmyk12_c_oracle, encode_with_cmyk12_c_oracle,
};
use libjpeg_turbo_rs::precision::{decompress_12bit, Image12};
use libjpeg_turbo_rs::raw_data_12::compress_raw_12;
use libjpeg_turbo_rs::{compress_raw, decompress, Subsampling};

/// Deterministic 12-bit noise, `count` samples.
fn noise_12bit(count: usize, seed: u32) -> Vec<i16> {
    let mut state: u32 = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 16) % 4096) as i16
        })
        .collect()
}

fn ppm_12bit(width: usize, height: usize, samples: &[i16]) -> Vec<u8> {
    let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n4095\n").into_bytes();
    for &sample in samples {
        ppm.extend_from_slice(&(sample as u16).to_be_bytes());
    }
    ppm
}

fn cjpeg_supports_precision(cjpeg: &Path) -> bool {
    Command::new(cjpeg)
        .arg("-help")
        .output()
        .is_ok_and(|output| {
            String::from_utf8_lossy(&output.stderr).contains("precision")
                || String::from_utf8_lossy(&output.stdout).contains("precision")
        })
}

fn count_differing(actual: &[i16], expected: &[i16]) -> usize {
    assert_eq!(actual.len(), expected.len(), "sample count");
    actual
        .iter()
        .zip(expected)
        .filter(|(actual, expected)| actual != expected)
        .count()
}

/// Mark a baseline/extended stream from this crate's raw encoder as RGB the
/// way `cjpeg -rgb` does: drop JFIF APP0, insert Adobe APP14 with transform 0,
/// and set the component IDs to `R`, `G`, `B` in SOF and SOS.
fn make_rgb_coded(jpeg: &[u8]) -> Vec<u8> {
    let mut out: Vec<u8> = vec![
        0xFF, 0xD8, 0xFF, 0xEE, 0x00, 0x0E, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00,
        0x00, 0x00, 0x00,
    ];
    let mut pos: usize = 2;
    loop {
        let marker: u8 = jpeg[pos + 1];
        let length: usize = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let mut segment: Vec<u8> = jpeg[pos..pos + 2 + length].to_vec();
        match marker {
            0xE0 => segment.clear(),
            0xC0 | 0xC1 => {
                for (component, id) in b"RGB".iter().enumerate() {
                    segment[10 + 3 * component] = *id;
                }
            }
            0xDA => {
                for (component, id) in b"RGB".iter().enumerate() {
                    segment[5 + 2 * component] = *id;
                }
                out.extend_from_slice(&segment);
                out.extend_from_slice(&jpeg[pos + 2 + length..]);
                return out;
            }
            _ => {}
        }
        out.extend_from_slice(&segment);
        pos += 2 + length;
    }
}

/// Issue #694: an RGB-coded 12-bit frame from `cjpeg -rgb -precision 12`
/// decodes identically to djpeg, which outputs it without colour conversion.
#[test]
fn rgb_coded_12bit_frame_from_cjpeg_matches_djpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    if !cjpeg_supports_precision(&cjpeg) {
        helpers::skip_missing_c_capability("cjpeg", "-precision 12");
        return;
    }
    let (width, height): (usize, usize) = (37, 29);
    let jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
        &cjpeg,
        &ppm_12bit(width, height, &noise_12bit(width * height * 3, 694)),
        &["-precision", "12", "-quality", "95", "-rgb"],
        "issue_694_cjpeg_rgb",
    );
    let (c_width, c_height, components, maxval, expected): (usize, usize, usize, usize, Vec<i16>) =
        helpers::decode_with_c_djpeg_i16(&djpeg, &jpeg, "issue_694_cjpeg_rgb");
    assert_eq!(
        (c_width, c_height, components, maxval),
        (width, height, 3, 4095)
    );
    let image: Image12 = decompress_12bit(&jpeg).expect("decompress_12bit");
    assert_eq!(
        (image.width, image.height, image.num_components),
        (width, height, 3)
    );
    let differing: usize = count_differing(&image.data, &expected);
    assert_eq!(
        differing,
        0,
        "{differing} of {} samples differ from djpeg",
        expected.len()
    );
}

/// Issue #694 (reporter's reproducer): the same flat planes, marked RGB, come
/// back unconverted from both the 8-bit and the 12-bit decoder, as djpeg
/// returns them.
#[test]
fn rgb_coded_12bit_flat_planes_are_not_colour_converted() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let (width, height): (usize, usize) = (16, 16);

    let stored8: [u8; 3] = [186, 62, 31];
    let planes8: Vec<Vec<u8>> = stored8.iter().map(|&v| vec![v; width * height]).collect();
    let refs8: Vec<&[u8]> = planes8.iter().map(|plane| plane.as_slice()).collect();
    let jpeg8: Vec<u8> = compress_raw(
        &refs8,
        &[width; 3],
        &[height; 3],
        width,
        height,
        100,
        Subsampling::S444,
    )
    .expect("compress_raw");
    let image8 = decompress(&make_rgb_coded(&jpeg8)).expect("decompress 8-bit");
    assert_eq!(&image8.data[..3], &stored8);

    let stored12: [i16; 3] = [3000, 1000, 500];
    let planes12: Vec<Vec<i16>> = stored12.iter().map(|&v| vec![v; width * height]).collect();
    let refs12: Vec<&[i16]> = planes12.iter().map(|plane| plane.as_slice()).collect();
    let jpeg12: Vec<u8> = make_rgb_coded(
        &compress_raw_12(
            &refs12,
            &[width; 3],
            &[height; 3],
            width,
            height,
            100,
            Subsampling::S444,
        )
        .expect("compress_raw_12"),
    );
    let (_, _, _, _, expected): (usize, usize, usize, usize, Vec<i16>) =
        helpers::decode_with_c_djpeg_i16(&djpeg, &jpeg12, "issue_694_flat_rgb");
    let image12: Image12 = decompress_12bit(&jpeg12).expect("decompress_12bit");
    assert_eq!(&image12.data[..3], &stored12);
    assert_eq!(count_differing(&image12.data, &expected), 0);
}

fn cmyk12_oracle() -> Option<PathBuf> {
    let oracle: Option<PathBuf> = cmyk12_c_oracle();
    if oracle.is_none() {
        assert!(
            !helpers::is_ci(),
            "CI provisions a libjpeg development install, so the 12-bit CMYK \
             oracle must be buildable there"
        );
        eprintln!(
            "SKIP: no libjpeg development install (headers + library) found; \
             cannot build the 12-bit CMYK oracle"
        );
    }
    oracle
}

/// Issue #694: a 12-bit YCCK frame (libjpeg's default 2x2/1x1/1x1/2x2 layout)
/// is converted to CMYK, and a 12-bit CMYK frame is passed through, exactly as
/// libjpeg outputs them at its default `out_color_space` (`JCS_CMYK` for
/// both). The odd and even sizes also cover issue #693's edge upsampling on
/// the four-component path.
#[test]
fn four_component_12bit_frames_match_libjpeg_cmyk_output() {
    let Some(oracle) = cmyk12_oracle() else {
        return;
    };
    for ycck in [true, false] {
        for (width, height) in [(37, 29), (62, 46), (64, 48)] {
            let label: String = format!("{} {width}x{height}", if ycck { "YCCK" } else { "CMYK" });
            let jpeg: Vec<u8> = encode_with_cmyk12_c_oracle(
                &oracle,
                &noise_12bit(width * height * 4, 6940),
                width,
                height,
                95,
                ycck,
            );
            let (c_width, c_height, components, expected): (usize, usize, usize, Vec<i16>) =
                decode_with_cmyk12_c_oracle(&oracle, &jpeg);
            assert_eq!(
                (c_width, c_height, components),
                (width, height, 4),
                "{label}"
            );
            let image: Image12 =
                decompress_12bit(&jpeg).unwrap_or_else(|error| panic!("{label}: {error}"));
            assert_eq!(
                (image.width, image.height, image.num_components),
                (width, height, 4),
                "{label}"
            );
            let differing: usize = count_differing(&image.data, &expected);
            assert_eq!(
                differing,
                0,
                "{label}: {differing} of {} samples differ from libjpeg",
                expected.len()
            );
        }
    }
}
