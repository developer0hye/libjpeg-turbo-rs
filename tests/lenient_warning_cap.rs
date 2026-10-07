//! Issue #641 (P4-215): a lenient decode records at most
//! `MAX_DECODE_WARNINGS` per-MCU Huffman warnings and counts the rest.
//!
//! Before the fix the interleaved lenient loop pushed one
//! `DecodeWarning::HuffmanError` (with a heap `String`) per corrupt MCU, so a
//! stream corrupt everywhere produced a list proportional to the MCU count —
//! about 16.7 M entries for a 65500x65500 frame. C's `emit_message`
//! (`jerror.c`) prints the first warning and only counts the others in
//! `num_warnings`; the Rust list now keeps the first `MAX_DECODE_WARNINGS` and
//! ends with one `WarningsSuppressed { count }` entry for the rest.
//!
//! There is no C oracle for the list itself — it is a Rust API — so these
//! tests pin its bound and its arithmetic, not pixels.

use libjpeg_turbo_rs::{
    decompress_lenient, DecodeWarning, Encoder, PixelFormat, Subsampling, MAX_DECODE_WARNINGS,
};

/// 256x256 4:4:4 baseline: 32x32 = 1024 interleaved MCUs, far above the cap.
/// Three components, because a single-component frame takes the
/// non-interleaved path, which warns once per scan rather than per MCU.
const SIDE: usize = 256;

fn mcus(side: usize) -> usize {
    (side / 8) * (side / 8)
}

/// The headers of a baseline 4:4:4 encode of a `side` x `side` frame, up to
/// and including its SOS, optionally with a one-MCU restart interval.
fn headers(side: usize, restart_every_mcu: bool) -> Vec<u8> {
    let pixels: Vec<u8> = (0..side * side * 3)
        .map(|i: usize| (i % 251) as u8)
        .collect();
    let mut encoder = Encoder::new(&pixels, side, side, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444);
    if restart_every_mcu {
        encoder = encoder.restart_blocks(1);
    }
    let jpeg: Vec<u8> = encoder.encode().expect("encode the source frame");
    let sos: usize = jpeg
        .windows(2)
        .position(|marker: &[u8]| marker == [0xFF, 0xDA])
        .expect("baseline stream has an SOS");
    let sos_length: usize = usize::from(u16::from_be_bytes([jpeg[sos + 2], jpeg[sos + 3]]));
    jpeg[..sos + 2 + sos_length].to_vec()
}

/// A stream whose entropy-coded segment is all `0xFF 0x00` pairs: stuffed
/// 0xFF bytes, so every bit is one. The default luminance DC table has no
/// all-ones code, so every MCU's first block fails and the lenient loop zeroes
/// it and moves on. The stream ends with EOI, so the reader never runs dry and
/// the decode reports one Huffman error per MCU and no truncation.
fn corrupt_everywhere(side: usize) -> Vec<u8> {
    let mut corrupt: Vec<u8> = headers(side, false);
    for _ in 0..mcus(side) * 8 {
        corrupt.extend_from_slice(&[0xFF, 0x00]);
    }
    corrupt.extend_from_slice(&[0xFF, 0xD9]);
    corrupt
}

/// The HuffmanError entries of `warnings`, and the `WarningsSuppressed` count
/// (0 when there is no such entry, which must then be absent, not zero).
fn tally(warnings: &[DecodeWarning]) -> (usize, usize) {
    let recorded: usize = warnings
        .iter()
        .filter(|warning| matches!(warning, DecodeWarning::HuffmanError { .. }))
        .count();
    let suppressed: Vec<usize> = warnings
        .iter()
        .filter_map(|warning| match warning {
            DecodeWarning::WarningsSuppressed { count } => Some(*count),
            _ => None,
        })
        .collect();
    assert!(suppressed.len() <= 1, "one suppression entry at most");
    assert!(suppressed.first() != Some(&0), "never a zero count");
    (recorded, suppressed.first().copied().unwrap_or(0))
}

/// Issue #641: a fully corrupt lenient decode keeps the first
/// `MAX_DECODE_WARNINGS` Huffman errors, then one `WarningsSuppressed` entry
/// whose count accounts for every other corrupt MCU. Before the fix this list
/// had 1024 entries, one per MCU.
#[test]
fn issue_641_fully_corrupt_decode_caps_huffman_warnings() {
    let image = decompress_lenient(&corrupt_everywhere(SIDE)).expect("lenient decode recovers");
    assert_eq!((image.width, image.height), (SIDE, SIDE));

    let warnings: &[DecodeWarning] = &image.warnings;
    assert_eq!(
        warnings.len(),
        MAX_DECODE_WARNINGS + 1,
        "cap plus one suppression entry"
    );
    for (index, warning) in warnings[..MAX_DECODE_WARNINGS].iter().enumerate() {
        match warning {
            // The first N MCUs in raster order: like C, the earliest are kept.
            DecodeWarning::HuffmanError { mcu_x, mcu_y, .. } => {
                assert_eq!(
                    (*mcu_x, *mcu_y),
                    (index % (SIDE / 8), index / (SIDE / 8)),
                    "entry {index}"
                );
            }
            other => panic!("entry {index}: expected HuffmanError, got {other:?}"),
        }
    }
    match &warnings[MAX_DECODE_WARNINGS] {
        DecodeWarning::WarningsSuppressed { count } => {
            assert_eq!(*count, mcus(SIDE) - MAX_DECODE_WARNINGS);
        }
        other => panic!("last entry: expected WarningsSuppressed, got {other:?}"),
    }
}

/// Issue #641: the list's length does not grow with the frame — four times the
/// MCUs gives the same length, and only the suppressed count grows.
#[test]
fn issue_641_warning_list_length_is_independent_of_mcu_count() {
    let side: usize = SIDE * 2;
    let image = decompress_lenient(&corrupt_everywhere(side)).expect("lenient decode recovers");
    assert_eq!(image.warnings.len(), MAX_DECODE_WARNINGS + 1);
    assert_eq!(
        tally(&image.warnings),
        (MAX_DECODE_WARNINGS, mcus(side) - MAX_DECODE_WARNINGS)
    );
}

/// Issue #641: below the cap every corrupt MCU is recorded and no suppression
/// entry is added.
#[test]
fn issue_641_few_corrupt_mcus_are_all_recorded() {
    let side: usize = 32;
    assert!(mcus(side) < MAX_DECODE_WARNINGS);
    let image = decompress_lenient(&corrupt_everywhere(side)).expect("lenient decode recovers");
    assert_eq!(image.warnings.len(), mcus(side));
    assert_eq!(tally(&image.warnings), (mcus(side), 0));
}

/// Issue #641: the one-shot `TruncatedData` warning is not subject to the cap.
/// A one-MCU restart interval lets each corrupt MCU resynchronise at its RST
/// marker; the data then ends without EOI after 200 MCUs, so the decode
/// reports far more Huffman errors than the cap *and* the truncation, which a
/// caller checking for `TruncatedData` must still see.
#[test]
fn issue_641_truncation_is_reported_past_the_cap() {
    let corrupt_mcus: usize = 200;
    let mut corrupt: Vec<u8> = headers(SIDE, true);
    for restart in 0..corrupt_mcus {
        corrupt.extend_from_slice(&[0xFF, 0x00, 0xFF, 0x00]);
        if restart + 1 < corrupt_mcus {
            corrupt.extend_from_slice(&[0xFF, 0xD0 + (restart % 8) as u8]);
        }
    }

    let image = decompress_lenient(&corrupt).expect("lenient decode recovers");
    let decoded_mcus: usize = image
        .warnings
        .iter()
        .find_map(|warning| match warning {
            DecodeWarning::TruncatedData {
                decoded_mcus,
                total_mcus,
            } => {
                assert_eq!(*total_mcus, mcus(SIDE));
                Some(*decoded_mcus)
            }
            _ => None,
        })
        .unwrap_or_else(|| panic!("no TruncatedData in {:?}", image.warnings));
    assert!(decoded_mcus > MAX_DECODE_WARNINGS, "{decoded_mcus}");
    let (recorded, suppressed): (usize, usize) = tally(&image.warnings);
    assert_eq!(recorded, MAX_DECODE_WARNINGS);
    // Every MCU the decode reached failed; the cap only moves them from the
    // list into the count.
    assert_eq!(recorded + suppressed, decoded_mcus);
    assert_eq!(image.warnings.len(), MAX_DECODE_WARNINGS + 2);
}
