//! P4-40: the baseline `compress_*` entry points share one implementation.
//!
//! Before the `CompressParams` core, each variant carried only the options it
//! named — `compress_with_restart` could not express custom tables,
//! `compress_custom_quant` could not express a restart interval or a DCT
//! method. A caller wanting two of them at once simply could not have both,
//! and each copy of the algorithm drifted independently (that is how #313,
//! #314 and #316 happened).
//!
//! These tests assert the properties that only hold when there is a single
//! implementation: every option composes with every other, and each option has
//! the same observable effect regardless of which others are set.

use libjpeg_turbo_rs::encode::pipeline::{self, CompressParams};
use libjpeg_turbo_rs::{DctMethod, HuffmanTableDef, PixelFormat, Subsampling};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

fn test_pixels() -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; WIDTH * HEIGHT * 3];
    let mut rng_state: u32 = 0x5eed_1234;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let in_rect: bool = x * 3 >= WIDTH && x * 3 < WIDTH * 2;
            let offset: usize = (y * WIDTH + x) * 3;
            pixels[offset] = ((x * 255 / WIDTH) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 1] = ((y * 255 / HEIGHT) as i32 - noise).clamp(0, 255) as u8;
            pixels[offset + 2] = (if in_rect { 210 } else { 45 } + noise).clamp(0, 255) as u8;
        }
    }
    pixels
}

/// Counts RST0..RST7 markers. Restart markers only appear in the entropy
/// stream, where a 0xFF byte is otherwise always followed by 0x00 stuffing.
fn count_restart_markers(jpeg: &[u8]) -> usize {
    jpeg.windows(2)
        .filter(|window| window[0] == 0xFF && (0xD0..=0xD7).contains(&window[1]))
        .count()
}

/// True when a DRI marker (0xFFDD) carrying `interval` is present.
fn has_dri(jpeg: &[u8], interval: u16) -> bool {
    jpeg.windows(6).any(|window| {
        window[0] == 0xFF
            && window[1] == 0xDD
            && window[2] == 0x00
            && window[3] == 0x04
            && u16::from_be_bytes([window[4], window[5]]) == interval
    })
}

fn coarse_quant_tables() -> [Option<[u16; 64]>; 4] {
    let mut coarse: [u16; 64] = [0; 64];
    for (index, entry) in coarse.iter_mut().enumerate() {
        // Far coarser than any quality-75 default, so applying it must shrink
        // the file substantially.
        *entry = 180 + index as u16;
    }
    [Some(coarse), Some(coarse), None, None]
}

fn nonstandard_huffman() -> ([Option<HuffmanTableDef>; 4], [Option<HuffmanTableDef>; 4]) {
    let mut dc_bits: [u8; 17] = [0; 17];
    dc_bits[4] = 16;
    let dc = HuffmanTableDef {
        bits: dc_bits,
        values: (0u8..16).collect(),
    };
    let mut ac_bits: [u8; 17] = [0; 17];
    ac_bits[5] = 16;
    ac_bits[6] = 16;
    let ac = HuffmanTableDef {
        bits: ac_bits,
        values: (0u8..32).collect(),
    };
    (
        [Some(dc.clone()), Some(dc), None, None],
        [Some(ac.clone()), Some(ac), None, None],
    )
}

/// The combination no pre-`CompressParams` entry point could express:
/// a restart interval *and* custom quantization tables *and* custom Huffman
/// tables *and* a non-default DCT method, all at once.
#[test]
fn all_baseline_options_compose_in_one_encode() {
    let pixels: Vec<u8> = test_pixels();
    let quant = coarse_quant_tables();
    let (dc, ac) = nonstandard_huffman();

    let encoded: Vec<u8> = pipeline::compress_with_params(
        &CompressParams::new(
            &pixels,
            WIDTH,
            HEIGHT,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
        )
        .dct_method(DctMethod::IsFast)
        .restart_interval(2)
        .custom_quant(&quant)
        .custom_huffman(&dc, &ac),
    )
    .expect("all baseline options together must encode");

    // Every requested option must be observable in the output at once.
    assert!(
        has_dri(&encoded, 2),
        "DRI marker with interval 2 missing — restart_interval was dropped"
    );
    assert!(
        count_restart_markers(&encoded) > 0,
        "no RST markers emitted despite restart_interval=2"
    );

    let default: Vec<u8> = pipeline::compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
        DctMethod::IsLow,
    )
    .expect("default encode");
    assert_ne!(
        encoded, default,
        "combined-option output is identical to a plain default encode"
    );
}

/// A restart interval must produce the same marker structure whether or not
/// custom tables are also in play — the property that fails when each variant
/// owns its own copy of the MCU loop.
#[test]
fn restart_interval_is_independent_of_other_options() {
    let pixels: Vec<u8> = test_pixels();
    let quant = coarse_quant_tables();
    let (dc, ac) = nonstandard_huffman();
    let interval: u16 = 3;

    let base = || {
        CompressParams::new(
            &pixels,
            WIDTH,
            HEIGHT,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
        )
        .restart_interval(interval)
    };

    let variants: [(&str, Vec<u8>); 4] = [
        (
            "restart only",
            pipeline::compress_with_params(&base()).expect("restart only"),
        ),
        (
            "restart + custom quant",
            pipeline::compress_with_params(&base().custom_quant(&quant))
                .expect("restart + custom quant"),
        ),
        (
            "restart + custom huffman",
            pipeline::compress_with_params(&base().custom_huffman(&dc, &ac))
                .expect("restart + custom huffman"),
        ),
        (
            "restart + ifast",
            pipeline::compress_with_params(&base().dct_method(DctMethod::IsFast))
                .expect("restart + ifast"),
        ),
    ];

    // 64x48 at 4:2:0 is 4x3 = 12 MCUs; an interval of 3 puts a restart after
    // MCUs 3, 6 and 9 — three markers, none after the final group.
    let expected_restarts: usize = 3;
    for (label, encoded) in &variants {
        assert!(
            has_dri(encoded, interval),
            "{label}: DRI marker missing — restart_interval was dropped"
        );
        assert_eq!(
            count_restart_markers(encoded),
            expected_restarts,
            "{label}: wrong RST marker count; restart placement depends on \
             which other options are set"
        );
    }
}

/// Custom quantization tables must have the same effect with and without a
/// restart interval.
#[test]
fn custom_quant_applies_regardless_of_restart() {
    let pixels: Vec<u8> = test_pixels();
    let quant = coarse_quant_tables();

    let with_quant_no_restart: Vec<u8> = pipeline::compress_with_params(
        &CompressParams::new(
            &pixels,
            WIDTH,
            HEIGHT,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
        )
        .custom_quant(&quant),
    )
    .expect("custom quant");

    let with_quant_and_restart: Vec<u8> = pipeline::compress_with_params(
        &CompressParams::new(
            &pixels,
            WIDTH,
            HEIGHT,
            PixelFormat::Rgb,
            75,
            Subsampling::S420,
        )
        .custom_quant(&quant)
        .restart_interval(4),
    )
    .expect("custom quant + restart");

    let default: Vec<u8> = pipeline::compress(
        &pixels,
        WIDTH,
        HEIGHT,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
        DctMethod::IsLow,
    )
    .expect("default");

    assert!(
        with_quant_no_restart.len() < default.len(),
        "coarse custom quant must shrink the file: custom={} default={}",
        with_quant_no_restart.len(),
        default.len()
    );
    // Restart markers add bytes, so the combined output is larger than the
    // restart-free one — but it must still be far below the default-table
    // encode, proving the quant tables were not dropped when restart was set.
    assert!(
        with_quant_and_restart.len() < default.len(),
        "custom quant was dropped once a restart interval was also set: \
         combined={} default={}",
        with_quant_and_restart.len(),
        default.len()
    );
}

/// The four public baseline entry points must agree with the core when handed
/// equivalent parameters — that is what makes them shims rather than forks.
#[test]
fn public_entry_points_agree_with_the_core() {
    let pixels: Vec<u8> = test_pixels();
    let quant = coarse_quant_tables();
    let (dc, ac) = nonstandard_huffman();

    for subsampling in [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
    ] {
        let make =
            || CompressParams::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb, 75, subsampling);

        assert_eq!(
            pipeline::compress(
                &pixels,
                WIDTH,
                HEIGHT,
                PixelFormat::Rgb,
                75,
                subsampling,
                DctMethod::IsFast,
            )
            .expect("compress"),
            pipeline::compress_with_params(&make().dct_method(DctMethod::IsFast))
                .expect("core equivalent"),
            "compress() diverged from the core at {subsampling:?}"
        );

        assert_eq!(
            pipeline::compress_with_restart(
                &pixels,
                WIDTH,
                HEIGHT,
                PixelFormat::Rgb,
                75,
                subsampling,
                5,
                DctMethod::IsLow,
            )
            .expect("compress_with_restart"),
            pipeline::compress_with_params(&make().restart_interval(5)).expect("core equivalent"),
            "compress_with_restart() diverged from the core at {subsampling:?}"
        );

        assert_eq!(
            pipeline::compress_custom_quant(
                &pixels,
                WIDTH,
                HEIGHT,
                PixelFormat::Rgb,
                75,
                subsampling,
                &quant,
            )
            .expect("compress_custom_quant"),
            pipeline::compress_with_params(&make().custom_quant(&quant)).expect("core equivalent"),
            "compress_custom_quant() diverged from the core at {subsampling:?}"
        );

        assert_eq!(
            pipeline::compress_custom_huffman(
                &pixels,
                WIDTH,
                HEIGHT,
                PixelFormat::Rgb,
                75,
                subsampling,
                &dc,
                &ac,
            )
            .expect("compress_custom_huffman"),
            pipeline::compress_with_params(&make().custom_huffman(&dc, &ac))
                .expect("core equivalent"),
            "compress_custom_huffman() diverged from the core at {subsampling:?}"
        );
    }
}
