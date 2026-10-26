//! Issue #693: 12-bit decode upsampled the last chroma column (row) from MCU
//! padding when the image width (height) was even but not a whole number of
//! MCUs.
//!
//! `decompress_12bit` bounded the fancy upsampler's input at
//! `width / h_factor + 1` columns, which is libjpeg's `downsampled_width`
//! (`ceil(width / h_factor)`) only for an odd width. For an even width it
//! counted one padding column as real, so the last real chroma sample was
//! blended with it instead of being replicated (`jdsample.c`
//! `h2v1_fancy_upsample` / `h2v2_fancy_upsample`), and output column
//! `width - 1` (row `height - 1`) came back wrong with `Ok`.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs::precision::{decompress_12bit, Image12};
use libjpeg_turbo_rs::raw_data_12::compress_raw_12;
use libjpeg_turbo_rs::Subsampling;

/// A 12-bit PPM of deterministic noise: every chroma sample differs from its
/// neighbours, so blending in a padding sample changes the edge pixels.
fn ppm_12bit_noise(width: usize, height: usize) -> Vec<u8> {
    let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n4095\n").into_bytes();
    let mut state: u32 = 0x1234_5678;
    for _ in 0..width * height * 3 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let sample: u16 = ((state >> 16) % 4096) as u16;
        ppm.extend_from_slice(&sample.to_be_bytes());
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

/// Samples of `image` that differ from djpeg's decode of `jpeg`.
fn samples_differing_from_djpeg(djpeg: &Path, jpeg: &[u8], image: &Image12, label: &str) -> usize {
    let (width, height, components, maxval, expected): (usize, usize, usize, usize, Vec<i16>) =
        helpers::decode_with_c_djpeg_i16(djpeg, jpeg, label);
    assert_eq!(
        (width, height, components, maxval),
        (image.width, image.height, image.num_components, 4095),
        "{label}: djpeg must decode the same 12-bit frame"
    );
    image
        .data
        .iter()
        .zip(&expected)
        .filter(|(actual, expected)| actual != expected)
        .count()
}

/// Issue #693: cjpeg-encoded noise at every subsampling the 12-bit upsampler
/// handles decodes identically to djpeg — at even, odd and whole-MCU sizes,
/// so both the padded-edge case and its controls are pinned.
#[test]
fn twelve_bit_subsampled_edges_match_djpeg_at_every_size() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let djpeg: PathBuf = require_c_tool!("djpeg");
    if !cjpeg_supports_precision(&cjpeg) {
        helpers::skip_missing_c_capability("cjpeg", "-precision 12");
        return;
    }
    let samplings: [(&str, &str); 5] = [
        ("420", "2x2,1x1,1x1"),
        ("422", "2x1,1x1,1x1"),
        ("440", "1x2,1x1,1x1"),
        ("411", "4x1,1x1,1x1"),
        ("410", "4x2,1x1,1x1"),
    ];
    // Even, odd and whole-MCU sizes, the widths where the h2 filters switch
    // to replication (two chroma samples or fewer), and frames one or two
    // samples tall or wide, where the vertical filters run out of rows.
    let sizes: [(usize, usize); 15] = [
        (62, 46),
        (63, 47),
        (64, 48),
        (62, 47),
        (63, 46),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
        (6, 6),
        (62, 1),
        (62, 2),
        (5, 1),
        (1, 62),
        (2, 62),
    ];
    let mut failures: Vec<String> = Vec::new();
    for (name, sample) in samplings {
        for (width, height) in sizes {
            let label: String = format!("issue_693_{name}_{width}x{height}");
            let jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
                &cjpeg,
                &ppm_12bit_noise(width, height),
                &["-precision", "12", "-quality", "95", "-sample", sample],
                &label,
            );
            let image: Image12 = decompress_12bit(&jpeg)
                .unwrap_or_else(|error| panic!("{label}: decompress_12bit: {error}"));
            let differing: usize = samples_differing_from_djpeg(&djpeg, &jpeg, &image, &label);
            if differing != 0 {
                failures.push(format!(
                    "{label}: {differing} of {} samples differ",
                    image.data.len()
                ));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "differ from djpeg:\n{}",
        failures.join("\n")
    );
}

/// Issue #693 (reporter's reproducer): a flat mid-grey 62x46 4:2:0 image whose
/// Cb plane is 4095 only in the padding column and row. No real chroma sample
/// differs from 2048, so djpeg's output is flat; before the fix the padding
/// reached output column 61 and row 45.
#[test]
fn twelve_bit_padding_chroma_does_not_reach_the_output() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let (width, height): (usize, usize) = (62, 46);
    let (chroma_width, chroma_height): (usize, usize) = (32, 24);
    let luma: Vec<i16> = vec![2048; width * height];
    let cr: Vec<i16> = vec![2048; chroma_width * chroma_height];
    let mut cb: Vec<i16> = cr.clone();
    for y in 0..chroma_height {
        for x in 0..chroma_width {
            if x == chroma_width - 1 || y == chroma_height - 1 {
                cb[y * chroma_width + x] = 4095;
            }
        }
    }
    let jpeg: Vec<u8> = compress_raw_12(
        &[&luma, &cb, &cr],
        &[width, chroma_width, chroma_width],
        &[height, chroma_height, chroma_height],
        width,
        height,
        100,
        Subsampling::S420,
    )
    .expect("compress_raw_12");
    let image: Image12 = decompress_12bit(&jpeg).expect("decompress_12bit");
    let differing: usize =
        samples_differing_from_djpeg(&djpeg, &jpeg, &image, "issue_693_padding_chroma");
    assert_eq!(
        differing,
        0,
        "{differing} of {} samples differ from djpeg",
        image.data.len()
    );
}
