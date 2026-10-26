//! `Decoder::decode_raw_with_warnings` returns `decode_raw`'s planes together
//! with the warnings `decode_image` reports for the same stream, so a caller
//! decoding planes leniently can tell recovered output from clean output.

mod helpers;

use libjpeg_turbo_rs::{compress, DecodeWarning, Decoder, PixelFormat, Subsampling};

const PHOTO: &[u8] = include_bytes!("fixtures/photo_320x240_420.jpg");

fn planes_and_warnings(jpeg: &[u8], lenient: bool) -> (Vec<Vec<u8>>, Vec<DecodeWarning>) {
    let (raw, warnings) = Decoder::new(jpeg)
        .and_then(|d| d.with_lenient(lenient).decode_raw_with_warnings())
        .expect("decode_raw_with_warnings");
    (raw.planes, warnings)
}

fn image_warnings(jpeg: &[u8], lenient: bool) -> Vec<DecodeWarning> {
    Decoder::new(jpeg)
        .and_then(|d| d.with_lenient(lenient).decode_image())
        .expect("decode_image")
        .warnings
}

#[test]
fn a_clean_stream_has_no_warnings_and_decode_raws_planes() {
    for lenient in [false, true] {
        let (planes, warnings) = planes_and_warnings(PHOTO, lenient);
        assert!(warnings.is_empty(), "lenient {lenient}: {warnings:?}");
        let raw = Decoder::new(PHOTO)
            .unwrap()
            .with_lenient(lenient)
            .decode_raw()
            .unwrap();
        assert_eq!(planes, raw.planes, "lenient {lenient}");
    }
}

#[test]
fn stray_bytes_between_markers_are_reported() {
    let (width, height): (usize, usize) = (48, 32);
    let pixels: Vec<u8> = helpers::generate_gradient(width, height);
    let clean: Vec<u8> = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap();
    // Two stray bytes before the segment after the first one.
    let next: usize = 4 + u16::from_be_bytes([clean[4], clean[5]]) as usize;
    let mut damaged: Vec<u8> = clean[..next].to_vec();
    damaged.extend_from_slice(&[0x12, 0x34]);
    damaged.extend_from_slice(&clean[next..]);
    let (planes, warnings) = planes_and_warnings(&damaged, false);
    assert_eq!(warnings, vec![DecodeWarning::ExtraneousData { bytes: 2 }]);
    assert_eq!(warnings, image_warnings(&damaged, false));
    assert_eq!(
        planes,
        planes_and_warnings(&clean, false).0,
        "the same planes"
    );
}

#[test]
fn concealed_blocks_are_reported_in_lenient_mode() {
    // 50 stuffed 0xFF bytes in the middle of the entropy-coded data: every bit
    // a one, which no default table codes, so the blocks there fail to decode.
    let mut corrupt: Vec<u8> = PHOTO.to_vec();
    let mid: usize = corrupt.len() / 2;
    for pair in corrupt[mid..mid + 100].chunks_mut(2) {
        pair.copy_from_slice(&[0xFF, 0x00]);
    }
    let (_, warnings) = planes_and_warnings(&corrupt, true);
    assert!(
        warnings
            .iter()
            .any(|w| matches!(w, DecodeWarning::HuffmanError { .. })),
        "{warnings:?}"
    );
    assert_eq!(warnings, image_warnings(&corrupt, true));
    // Strict decoding still refuses the stream, as decode_raw does.
    assert!(Decoder::new(&corrupt)
        .and_then(|d| d.decode_raw_with_warnings())
        .is_err());
    assert!(Decoder::new(&corrupt).and_then(|d| d.decode_raw()).is_err());
}
