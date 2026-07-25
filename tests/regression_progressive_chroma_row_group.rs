//! Issue #324: progressive chroma diverged from `cjpeg` at every even height
//! that is not a multiple of the MCU height.
//!
//! `downsample_chroma_block` clamped the *source* row
//! (`(block_y + row * v_factor + dy).min(plane_height - 1)`), which models C
//! only when the final row group is incomplete. C works in two phases
//! (`jcprepct.c` then `jccoefct.c`): pad the source up to a complete row
//! **group**, downsample, then replicate the resulting **downsampled** row.
//!
//! With `v_factor == 2` and an even height the last group is complete, so C
//! replicates `avg(last_two_rows)` while a source clamp yields `last_row`
//! alone. Odd heights agreed by accident — their last group is incomplete, so
//! both models replicate the same single row.
//!
//! Baseline was unaffected because it feeds already-padded planes (built by
//! `pad_chroma_plane`, which implements the two phases), so the clamp never
//! engaged. Only the progressive path passed an unpadded plane and relied on
//! it — which is why 1920x1080 progressive 4:2:0, the most common web
//! configuration, silently differed from `cjpeg`.

mod helpers;

use libjpeg_turbo_rs::{compress_progressive, PixelFormat, Subsampling};

/// Non-uniform content: a flat image cannot distinguish "replicate the last
/// row" from "replicate the average of the last two".
fn pixels(width: usize, height: usize) -> Vec<u8> {
    let mut buffer: Vec<u8> = vec![0u8; width * height * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..height {
        for x in 0..width {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let offset: usize = (y * width + x) * 3;
            buffer[offset] = ((x * 255 / width) as i32 + noise).clamp(0, 255) as u8;
            buffer[offset + 1] = ((y * 255 / height) as i32 - noise).clamp(0, 255) as u8;
            buffer[offset + 2] = (((x ^ y) & 0xff) as i32 + noise).clamp(0, 255) as u8;
        }
    }
    buffer
}

/// Only subsamplings with vertical chroma decimation can trigger this; 4:4:4
/// and 4:2:2 are included as controls that must stay clean.
const SUBSAMPLINGS: &[(Subsampling, &str)] = &[
    (Subsampling::S420, "2x2"),
    (Subsampling::S440, "1x2"),
    (Subsampling::S441, "1x4"),
    (Subsampling::S24, "2x4"),
    (Subsampling::S410, "4x2"),
    (Subsampling::S444, "1x1"),
    (Subsampling::S422, "2x1"),
];

#[test]
fn issue_324_progressive_chroma_matches_cjpeg_at_every_height() {
    let cjpeg = require_c_tool!("cjpeg");

    // Heights 1..=20 cover both parities either side of the 16-pixel MCU
    // boundary, which is exactly the axis that broke. 1080 and 600 are the
    // real-world sizes from the report; 1088 and 375 are passing controls.
    let mut geometries: Vec<(usize, usize)> = (1..=20).map(|height| (32, height)).collect();
    geometries.extend_from_slice(&[(1920, 1080), (800, 600), (1920, 1088), (500, 375)]);

    let mut failures: Vec<String> = Vec::new();

    for &(subsampling, sample) in SUBSAMPLINGS {
        for &(width, height) in &geometries {
            // Keep the large control sizes to one subsampling: they cost
            // seconds each and 4:2:0 is the configuration that was broken.
            if width > 100 && subsampling != Subsampling::S420 {
                continue;
            }

            let raw: Vec<u8> = pixels(width, height);
            let rust_jpeg: Vec<u8> =
                compress_progressive(&raw, width, height, PixelFormat::Rgb, 90, subsampling)
                    .unwrap_or_else(|error| {
                        panic!("{width}x{height} {sample} progressive encode failed: {error:?}")
                    });

            let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
            ppm.extend_from_slice(&raw);
            let c_jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
                &cjpeg,
                &ppm,
                &[
                    "-quality",
                    "90",
                    "-dct",
                    "int",
                    "-baseline",
                    "-progressive",
                    "-sample",
                    sample,
                ],
                &format!("issue324_{width}x{height}_{sample}"),
            );

            if rust_jpeg != c_jpeg {
                failures.push(format!(
                    "  {width}x{height} {sample} (h even: {}, h % 16 == {}): rust={} c={}",
                    height % 2 == 0,
                    height % 16,
                    rust_jpeg.len(),
                    c_jpeg.len()
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "progressive output diverged from cjpeg at {} geometries (issue #324):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
