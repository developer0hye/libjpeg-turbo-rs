//! Issue #325, blind spot 1: `grayscale_from_color` had no correctness coverage
//! for non-RGB input.
//!
//! `Encoder::extract_luminance` implements the BT.601 luma weights
//! (`19595*R + 38470*G + 7471*B + 32768 >> 16`) and is the path taken for
//! every non-RGB pixel format. Plain `Rgb` deliberately routes through the
//! SIMD `rgb_to_ycbcr_row` instead, so `extract_luminance` is what handles
//! Rgba, Bgr, Bgra, Rgbx, Xrgb, Argb, Bgrx, Xbgr and Abgr.
//!
//! Mutation testing showed 8 mutants surviving there — `*` swapped for `+`,
//! `+` for `-`, `*` for `/` — with the whole suite green. The tests that
//! exercised it asserted only `img.pixel_format == PixelFormat::Grayscale`,
//! i.e. metadata rather than content, on uniform `vec![128u8; ..]` input that
//! could not have discriminated anyway.
//!
//! These tests assert the actual luma values, and pin the channel *order* per
//! format so that an R/B swap cannot pass either.

use libjpeg_turbo_rs::{decompress, Encoder, PixelFormat};

/// The reference the implementation must agree with, written independently of
/// it: ITU-R BT.601 with the same fixed-point rounding libjpeg uses.
fn bt601_luma(red: u8, green: u8, blue: u8) -> u8 {
    ((19595 * red as u32 + 38470 * green as u32 + 7471 * blue as u32 + 32768) >> 16) as u8
}

/// Interleaves one RGB triple into `format`'s channel order, filling any pad
/// byte with a value that must be ignored.
fn interleave(format: PixelFormat, red: u8, green: u8, blue: u8) -> Vec<u8> {
    const PAD: u8 = 0xA5;
    match format {
        PixelFormat::Rgb => vec![red, green, blue],
        PixelFormat::Bgr => vec![blue, green, red],
        PixelFormat::Rgba => vec![red, green, blue, PAD],
        PixelFormat::Bgra => vec![blue, green, red, PAD],
        PixelFormat::Rgbx => vec![red, green, blue, PAD],
        PixelFormat::Bgrx => vec![blue, green, red, PAD],
        PixelFormat::Xrgb => vec![PAD, red, green, blue],
        PixelFormat::Xbgr => vec![PAD, blue, green, red],
        PixelFormat::Argb => vec![PAD, red, green, blue],
        PixelFormat::Abgr => vec![PAD, blue, green, red],
        other => panic!("unsupported format in this test: {other:?}"),
    }
}

/// Every format that reaches `extract_luminance`. `Rgb` is included as a
/// control: it takes the SIMD path, and must still agree with BT.601.
const FORMATS: &[PixelFormat] = &[
    PixelFormat::Rgb,
    PixelFormat::Bgr,
    PixelFormat::Rgba,
    PixelFormat::Bgra,
    PixelFormat::Rgbx,
    PixelFormat::Bgrx,
    PixelFormat::Xrgb,
    PixelFormat::Xbgr,
    PixelFormat::Argb,
    PixelFormat::Abgr,
];

/// Colours chosen so the three weights are individually identifiable: pure
/// primaries isolate each coefficient, and the asymmetric mixes catch a
/// swapped R/B (which a grey or symmetric colour never would).
const COLOURS: &[(u8, u8, u8)] = &[
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 128, 0),
    (0, 128, 255),
    (17, 200, 93),
    (240, 12, 33),
    (255, 255, 255),
    (0, 0, 0),
];

/// Encodes a flat colour at quality 100 and reads back the decoded luma.
///
/// A flat image is exactly right here: with no spatial variation the DCT
/// carries only DC, so quantization cannot shift the value and the decoded
/// sample is the luma the encoder computed. (Flat input would be useless for
/// testing *smoothing*, which is why the sibling suites do not use it.)
fn encoded_luma(format: PixelFormat, red: u8, green: u8, blue: u8) -> u8 {
    const SIZE: usize = 16;
    let one: Vec<u8> = interleave(format, red, green, blue);
    let pixels: Vec<u8> = one.repeat(SIZE * SIZE);

    let jpeg: Vec<u8> = Encoder::new(&pixels, SIZE, SIZE, format)
        .quality(100)
        .grayscale_from_color(true)
        .encode()
        .unwrap_or_else(|error| panic!("{format:?} grayscale encode failed: {error:?}"));

    let image =
        decompress(&jpeg).unwrap_or_else(|error| panic!("{format:?} decode failed: {error:?}"));
    assert_eq!(
        image.pixel_format,
        PixelFormat::Grayscale,
        "{format:?}: grayscale_from_color did not produce a grayscale JPEG"
    );
    // Sample the centre, away from any edge handling.
    image.data[(SIZE / 2) * SIZE + SIZE / 2]
}

#[test]
fn issue_325_grayscale_from_color_matches_bt601_for_every_format() {
    let mut failures: Vec<String> = Vec::new();

    for &format in FORMATS {
        for &(red, green, blue) in COLOURS {
            let expected: u8 = bt601_luma(red, green, blue);
            let actual: u8 = encoded_luma(format, red, green, blue);
            // Quality 100 on a flat image is lossless for DC in practice; allow
            // 1 for the round-trip through DCT/IDCT rounding.
            let difference: i32 = (actual as i32 - expected as i32).abs();
            if difference > 1 {
                failures.push(format!(
                    "  {format:?} rgb({red},{green},{blue}): expected luma {expected}, got {actual}"
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "grayscale_from_color diverged from BT.601 in {} cases (issue #325):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A swapped red/blue channel is the most likely way to get this wrong, and it
/// is invisible on greys. Pinning that RGB and BGR of the *same* colour agree
/// catches it independently of the absolute values above.
#[test]
fn issue_325_channel_order_is_honoured_per_format() {
    let (red, green, blue) = (240u8, 12u8, 33u8);
    let reference: u8 = encoded_luma(PixelFormat::Rgb, red, green, blue);

    // A colour where swapping R and B must change the luma, so this test can
    // actually fail if the order is wrong.
    assert_ne!(
        bt601_luma(red, green, blue),
        bt601_luma(blue, green, red),
        "test colour is R/B symmetric and cannot detect a channel swap"
    );

    for &format in FORMATS {
        let actual: u8 = encoded_luma(format, red, green, blue);
        assert!(
            (actual as i32 - reference as i32).abs() <= 1,
            "{format:?}: luma {actual} differs from the RGB reference {reference} \
             for the same colour — channel order is wrong"
        );
    }
}

/// The 4-byte formats carry a pad or alpha byte that must not reach the luma
/// computation. Varying it alone must not move the output.
#[test]
fn issue_325_padding_byte_does_not_affect_luma() {
    let four_byte: &[PixelFormat] = &[
        PixelFormat::Rgba,
        PixelFormat::Bgra,
        PixelFormat::Rgbx,
        PixelFormat::Bgrx,
        PixelFormat::Xrgb,
        PixelFormat::Xbgr,
        PixelFormat::Argb,
        PixelFormat::Abgr,
    ];
    const SIZE: usize = 16;
    let (red, green, blue) = (17u8, 200u8, 93u8);

    for &format in four_byte {
        let mut lumas: Vec<u8> = Vec::new();
        for pad in [0u8, 0x7F, 0xFF] {
            let mut one: Vec<u8> = interleave(format, red, green, blue);
            // Whichever slot is not R, G or B is the pad; find it by elimination.
            let pad_index: usize = (0..4)
                .find(|&i| one[i] != red && one[i] != green && one[i] != blue)
                .expect("4-byte format has a pad slot");
            one[pad_index] = pad;
            let pixels: Vec<u8> = one.repeat(SIZE * SIZE);
            let jpeg: Vec<u8> = Encoder::new(&pixels, SIZE, SIZE, format)
                .quality(100)
                .grayscale_from_color(true)
                .encode()
                .unwrap_or_else(|error| panic!("{format:?} encode failed: {error:?}"));
            let image = decompress(&jpeg).expect("decode");
            lumas.push(image.data[(SIZE / 2) * SIZE + SIZE / 2]);
        }
        assert!(
            lumas.windows(2).all(|w| w[0] == w[1]),
            "{format:?}: the pad byte changed the luma ({lumas:?}) — it is being \
             read as colour data"
        );
    }
}
