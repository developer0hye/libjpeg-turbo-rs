//! Issue #362: 4:2:0 encode diverged from C libjpeg-turbo at every width whose
//! residue `width % 16` falls in `1..=8` — a ~0.7% larger scan for a
//! bit-identical decode. Reported against 0.6.2 / 0.6.3.
//!
//! Same defect as issue #314 (P4-41): the AVX2 4:2:0 row fast path guarded the
//! last partial MCU *row* but not the last partial MCU *column*, so it
//! forward-transformed replicated edge pixels where C emits a dummy block.
//! `width % 16` in `1..=8` is exactly the residue set that makes
//! `ceil(width/8)` odd, which is why the reported window has that shape. Fixed
//! in v0.7.0 (`5d88fcf`), before this report was filed.
//!
//! The report attributed the divergence to chroma — `expand_right_edge`
//! running before rather than after `h2v2` downsampling. That diagnosis is
//! wrong, and the difference matters for anyone reading this code later, so
//! both halves are pinned here:
//!
//!   * the trailing MCU's second luma block column lies entirely outside the
//!     image and must be a **dummy** block — AC all zero, the previous block's
//!     DC copied in so the coded DC difference is zero (`jccoefct.c:184-192`
//!     in `compress_data`, the single-pass path these tests drive; the
//!     full-buffer variant is `:292-312`) — never the FDCT of replicated edge
//!     pixels;
//!   * 4:2:0 chroma has exactly `ceil(width/16)` block columns, i.e. never a
//!     padding column at all, so no chroma-side padding strategy can produce
//!     the reported window.
//!
//! Two layers of coverage:
//!   1. `issue_362_420_trailing_luma_column_is_a_dummy_block` — reads back the
//!      encoded coefficients, so it needs no C toolchain and guards every
//!      platform and every backend.
//!   2. `issue_362_width_sweep_matches_cjpeg` — the issue's own sweep
//!      (4:4:4 / 4:2:2 / 4:2:0 across `width % 16 = 0..=15`), byte-exact
//!      against stock `cjpeg`, skipped with a reason when C tools are absent.

mod helpers;

use libjpeg_turbo_rs::encode::pipeline;
use libjpeg_turbo_rs::{read_coefficients, DctMethod, JpegCoefficients, PixelFormat, Subsampling};

/// The issue's sweep height. A multiple of 16, so the trailing *row* of MCUs is
/// full and the only padding under test is horizontal.
const SWEEP_HEIGHT: usize = 320;

const QUALITY: u8 = 90;

/// Uniform noise: the content the issue measured its +0.7% on, and the content
/// that makes a wrongly-transformed padding block maximally visible (replicated
/// edge pixels of a smooth image quantize to nearly nothing).
fn noise_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; width * height * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for byte in pixels.iter_mut() {
        rng_state = rng_state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        *byte = (rng_state >> 24) as u8;
    }
    pixels
}

fn encode(width: usize, height: usize, subsampling: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = noise_pixels(width, height);
    pipeline::compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        QUALITY,
        subsampling,
        DctMethod::IsLow,
    )
    .unwrap_or_else(|error| panic!("{width}x{height} encode failed: {error:?}"))
}

/// Assert the C dummy-block contract on the padding column of a 4:2:0 encode.
///
/// Pushes findings onto `failures` rather than asserting inline so one run
/// reports the whole sweep instead of the first bad width.
fn check_trailing_luma_column(width: usize, height: usize, failures: &mut Vec<String>) {
    let jpeg: Vec<u8> = encode(width, height, Subsampling::S420);
    let coefficients: JpegCoefficients = read_coefficients(&jpeg)
        .unwrap_or_else(|error| panic!("{width}x{height} coefficient read failed: {error:?}"));

    let luma = &coefficients.components[0];
    let real_block_columns: usize = width.div_ceil(8);
    let mcu_columns: usize = width.div_ceil(16);

    // The luma block grid is padded out to whole MCUs; chroma is not, because
    // its own block count already lands on the MCU grid.
    assert_eq!(
        luma.blocks_x,
        mcu_columns * 2,
        "{width}x{height}: luma block grid is not MCU-aligned"
    );
    for chroma in &coefficients.components[1..] {
        assert_eq!(
            chroma.blocks_x, mcu_columns,
            "{width}x{height}: 4:2:0 chroma should have exactly ceil(width/16) \
             block columns and therefore no padding column"
        );
    }

    let has_padding_column: bool = real_block_columns < luma.blocks_x;
    assert_eq!(
        has_padding_column,
        (1..=8).contains(&(width % 16)),
        "{width}x{height}: width % 16 = {} should{} produce a luma padding column",
        width % 16,
        if has_padding_column { "" } else { " not" }
    );
    if !has_padding_column {
        return;
    }

    for block_row in 0..luma.blocks_y {
        let padding = &luma.blocks[block_row * luma.blocks_x + real_block_columns];
        let previous = &luma.blocks[block_row * luma.blocks_x + real_block_columns - 1];

        let nonzero_ac: usize = padding[1..].iter().filter(|&&c| c != 0).count();
        if nonzero_ac != 0 {
            failures.push(format!(
                "  {width}x{height} row {block_row}: padding block has {nonzero_ac} \
                 non-zero AC coefficients (C emits an all-zero dummy)"
            ));
        }
        if padding[0] != previous[0] {
            failures.push(format!(
                "  {width}x{height} row {block_row}: padding block DC {} != previous \
                 block DC {} (C copies the previous DC so the difference codes as 0)",
                padding[0], previous[0]
            ));
        }
    }
}

#[test]
fn issue_362_420_trailing_luma_column_is_a_dummy_block() {
    let mut failures: Vec<String> = Vec::new();

    // Every residue of `width % 16`, so both the reported divergence window
    // (`1..=8`) and the widths that were always correct are covered.
    for width in 320..=335 {
        check_trailing_luma_column(width, SWEEP_HEIGHT, &mut failures);
    }

    // A height that also needs a trailing dummy *row*, so the column contract
    // is pinned in the geometry where both kinds of dummy meet.
    check_trailing_luma_column(326, 321, &mut failures);

    assert!(
        failures.is_empty(),
        "4:2:0 trailing-MCU padding blocks are not C dummy blocks (issue #362):\n{}",
        failures.join("\n")
    );
}

#[test]
fn issue_362_width_sweep_matches_cjpeg() {
    let cjpeg = require_c_tool!("cjpeg");

    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "1x1"),
        (Subsampling::S422, "2x1"),
        (Subsampling::S420, "2x2"),
    ];

    let mut failures: Vec<String> = Vec::new();
    let mut compared: usize = 0;

    for &(subsampling, sample) in subsamplings {
        for width in 320..=335 {
            let pixels: Vec<u8> = noise_pixels(width, SWEEP_HEIGHT);
            let rust_jpeg: Vec<u8> = encode(width, SWEEP_HEIGHT, subsampling);

            let mut ppm: Vec<u8> = format!("P6\n{width} {SWEEP_HEIGHT}\n255\n").into_bytes();
            ppm.extend_from_slice(&pixels);
            let c_jpeg: Vec<u8> = helpers::encode_with_c_cjpeg(
                &cjpeg,
                &ppm,
                &[
                    "-quality",
                    &QUALITY.to_string(),
                    "-sample",
                    sample,
                    "-dct",
                    "int",
                ],
                &format!("issue362_{sample}_{width}"),
            );

            compared += 1;
            if rust_jpeg != c_jpeg {
                let first_difference: usize = rust_jpeg
                    .iter()
                    .zip(c_jpeg.iter())
                    .position(|(a, b)| a != b)
                    .unwrap_or(rust_jpeg.len().min(c_jpeg.len()));
                failures.push(format!(
                    "  {sample} {width}x{SWEEP_HEIGHT} (width%16={:2}): rust={} bytes, \
                     c={} bytes, first difference at offset {first_difference}",
                    width % 16,
                    rust_jpeg.len(),
                    c_jpeg.len(),
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "encode diverged from cjpeg at {} of {compared} widths (issue #362):\n{}",
        failures.len(),
        failures.join("\n")
    );
}
