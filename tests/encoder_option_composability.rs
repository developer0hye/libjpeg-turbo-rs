//! Issue #322: `Encoder` silently drops builder options when two of them fall
//! in different arms of its dispatch chain (`src/api/encoder.rs:918-980`).
//!
//! Each arm calls a shim that forwards only the options it names, and the first
//! matching arm wins, so everything it cannot express is discarded without an
//! error. Unlike #313 this is not CMYK-specific — it hits ordinary RGB input.
//!
//! The `#[ignore]`d tests below assert the correct behaviour and fail today;
//! they are the reproduction. `readme_compress_params_example_compiles` is not
//! ignored: it pins the README snippet, which uses the `CompressParams` core
//! precisely because that core *does* compose.

use libjpeg_turbo_rs::encode::pipeline::{compress_with_params, CompressParams};
use libjpeg_turbo_rs::{DctMethod, Encoder, HuffmanTableDef, PixelFormat, Subsampling};

const WIDTH: usize = 64;
const HEIGHT: usize = 48;

fn test_pixels() -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; WIDTH * HEIGHT * 3];
    let mut rng_state: u32 = 0x1234_5678;
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            rng_state = rng_state
                .wrapping_mul(1_664_525)
                .wrapping_add(1_013_904_223);
            let noise: i32 = ((rng_state >> 24) as i32 & 0x1f) - 16;
            let offset: usize = (y * WIDTH + x) * 3;
            pixels[offset] = ((x * 255 / WIDTH) as i32 + noise).clamp(0, 255) as u8;
            pixels[offset + 1] = ((y * 255 / HEIGHT) as i32 - noise).clamp(0, 255) as u8;
            pixels[offset + 2] = (((x + y) * 255 / (WIDTH + HEIGHT)) as i32).clamp(0, 255) as u8;
        }
    }
    pixels
}

/// RST0..RST7 markers in the entropy stream.
fn restart_markers(jpeg: &[u8]) -> usize {
    jpeg.windows(2)
        .filter(|window| window[0] == 0xFF && (0xD0..=0xD7).contains(&window[1]))
        .count()
}

/// Much coarser than any quality-75 default, so applying it must shrink output.
fn coarse_quant() -> [u16; 64] {
    let mut table: [u16; 64] = [0; 64];
    for (index, entry) in table.iter_mut().enumerate() {
        *entry = 180 + index as u16;
    }
    table
}

/// Valid but deliberately non-standard: 16 DC codes of length 4, 32 AC codes.
fn nonstandard_huffman() -> (HuffmanTableDef, HuffmanTableDef) {
    let mut dc_bits: [u8; 17] = [0; 17];
    dc_bits[4] = 16;
    let mut ac_bits: [u8; 17] = [0; 17];
    ac_bits[5] = 16;
    ac_bits[6] = 16;
    (
        HuffmanTableDef {
            bits: dc_bits,
            values: (0u8..16).collect(),
        },
        HuffmanTableDef {
            bits: ac_bits,
            values: (0u8..32).collect(),
        },
    )
}

/// 64x48 at the default 4:2:0 is 4x3 = 12 MCUs; `restart_blocks(3)` puts a
/// marker after MCUs 3, 6 and 9.
const EXPECTED_RESTARTS: usize = 3;

#[test]
#[ignore = "Issue #322: Encoder drops restart_blocks when custom quant tables are set"]
fn encoder_keeps_restart_with_custom_quant() {
    let pixels: Vec<u8> = test_pixels();
    let encoded: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .quality(75)
        .quant_table(0, coarse_quant())
        .quant_table(1, coarse_quant())
        .restart_blocks(3)
        .encode()
        .expect("custom quant + restart must encode");

    assert_eq!(
        restart_markers(&encoded),
        EXPECTED_RESTARTS,
        "restart interval was dropped once custom quant tables were set (issue #322)"
    );
}

#[test]
#[ignore = "Issue #322: Encoder drops restart_blocks when custom Huffman tables are set"]
fn encoder_keeps_restart_with_custom_huffman() {
    let pixels: Vec<u8> = test_pixels();
    let (dc, ac) = nonstandard_huffman();
    let encoded: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .quality(75)
        .huffman_dc_table(0, dc)
        .huffman_ac_table(0, ac)
        .restart_blocks(3)
        .encode()
        .expect("custom huffman + restart must encode");

    assert_eq!(
        restart_markers(&encoded),
        EXPECTED_RESTARTS,
        "restart interval was dropped once custom Huffman tables were set (issue #322)"
    );
}

#[test]
#[ignore = "Issue #322: Encoder drops custom quant tables when custom Huffman tables are set"]
fn encoder_keeps_custom_quant_with_custom_huffman() {
    let pixels: Vec<u8> = test_pixels();
    let (dc, ac) = nonstandard_huffman();

    let huffman_only: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .quality(75)
        .huffman_dc_table(0, dc.clone())
        .huffman_ac_table(0, ac.clone())
        .encode()
        .expect("custom huffman alone");

    let both: Vec<u8> = Encoder::new(&pixels, WIDTH, HEIGHT, PixelFormat::Rgb)
        .quality(75)
        .huffman_dc_table(0, dc)
        .huffman_ac_table(0, ac)
        .quant_table(0, coarse_quant())
        .quant_table(1, coarse_quant())
        .encode()
        .expect("custom huffman + custom quant");

    assert_ne!(
        both, huffman_only,
        "output is byte-identical to custom-Huffman-alone — the quant tables \
         were dropped (issue #322)"
    );
    assert!(
        both.len() < huffman_only.len(),
        "a much coarser quant table must shrink output: both={} huffman_only={}",
        both.len(),
        huffman_only.len()
    );
}

/// Pins the README's "Composing baseline options" snippet — the claim being
/// that the core honours *every* option at once, which is why the README
/// documents it rather than `Encoder` (#322).
///
/// Each option is pinned **individually**, by applying it alone to a default
/// encode and requiring the output to change. Two weaker formulations were
/// tried and rejected:
///
/// - A joint check ("smaller than a plain encode") passes even when one of
///   `dct_method` / `custom_quant` / `custom_huffman` is dropped, so a
///   single-option regression would go unnoticed.
/// - Dropping one option from the *full* set is platform-dependent: on top of
///   the coarse quant table almost every coefficient quantizes to zero, so
///   `dct_method` stops being observable. That version passed on x86_64 (685
///   vs 854 bytes) and failed on aarch64, where the two agree exactly.
///
/// Applied alone against the default tables the margins are large on every
/// backend (measured on x86_64: 1153, 1169, 190 and 1343 differing bytes),
/// which is what makes this formulation robust.
#[test]
fn readme_compress_params_example_compiles() {
    let rgb_pixels: Vec<u8> = test_pixels();
    let (width, height) = (WIDTH, HEIGHT);
    let quant_tables: [Option<[u16; 64]>; 4] =
        [Some(coarse_quant()), Some(coarse_quant()), None, None];
    let (dc, ac) = nonstandard_huffman();
    let dc_tables: [Option<HuffmanTableDef>; 4] = [Some(dc.clone()), Some(dc), None, None];
    let ac_tables: [Option<HuffmanTableDef>; 4] = [Some(ac.clone()), Some(ac), None, None];

    // --- README snippet, verbatim apart from the `?` -> expect ---
    let jpeg = compress_with_params(
        &CompressParams::new(
            &rgb_pixels,
            width,
            height,
            PixelFormat::Rgb,
            85,
            Subsampling::S420,
        )
        .dct_method(DctMethod::IsFast)
        .restart_interval(8)
        .custom_quant(&quant_tables)
        .custom_huffman(&dc_tables, &ac_tables),
    )
    .expect("README example must encode");
    // --- end snippet ---

    assert!(
        jpeg.starts_with(&[0xFF, 0xD8]) && jpeg.ends_with(&[0xFF, 0xD9]),
        "README example did not produce a complete JPEG datastream"
    );
    // 4:2:0 at 64x48 is 12 MCUs; interval 8 yields one restart marker.
    assert_eq!(
        restart_markers(&jpeg),
        1,
        "README example lost its restart interval — the composing claim is false"
    );

    // Apply each option alone to a default encode; every one must change the
    // output. If any matches the plain encode, that option is being ignored.
    let base = || {
        CompressParams::new(
            &rgb_pixels,
            width,
            height,
            PixelFormat::Rgb,
            85,
            Subsampling::S420,
        )
    };
    let plain: Vec<u8> = compress_with_params(&base()).expect("plain encode");
    let each_alone: [(&str, Vec<u8>); 4] = [
        (
            "dct_method",
            compress_with_params(&base().dct_method(DctMethod::IsFast)).expect("dct_method alone"),
        ),
        (
            "restart_interval",
            compress_with_params(&base().restart_interval(8)).expect("restart alone"),
        ),
        (
            "custom_quant",
            compress_with_params(&base().custom_quant(&quant_tables)).expect("quant alone"),
        ),
        (
            "custom_huffman",
            compress_with_params(&base().custom_huffman(&dc_tables, &ac_tables))
                .expect("huffman alone"),
        ),
    ];

    for (option, encoded) in &each_alone {
        assert_ne!(
            encoded, &plain,
            "`{option}` alone produced output identical to a default encode — \
             the option is being ignored"
        );
    }
}
