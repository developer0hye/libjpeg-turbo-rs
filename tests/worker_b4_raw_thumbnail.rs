#![cfg(not(target_arch = "wasm32"))]
//! Integration test for `extract_embedded_jpeg` (worker-b4 / B4-6).
//!
//! Emulates the layout that Sony ARW, Canon CR2, Nikon NEF, and similar
//! camera-RAW formats use to carry their JPEG preview / thumbnail:
//!
//!   TIFF 6.0 header
//!   IFD0 with JPEGInterchangeFormat + JPEGInterchangeFormatLength tags
//!   payload = raw JPEG bytes at the declared offset
//!
//! The test synthesises such a container around a real JPEG produced by
//! our own encoder, feeds it through `extract_embedded_jpeg`, then
//! decodes the extracted bytes and asserts that the pixels survive the
//! round-trip unmodified.

use libjpeg_turbo_rs::{compress, decompress_to, extract_embedded_jpeg, PixelFormat, Subsampling};

/// Build a minimal TIFF wrapping the supplied JPEG under IFD0 via the
/// JPEGInterchangeFormat / JPEGInterchangeFormatLength tag pair.
fn wrap_jpeg_as_tiff(jpeg: &[u8], is_le: bool) -> Vec<u8> {
    fn push_u16(buf: &mut Vec<u8>, v: u16, is_le: bool) {
        if is_le {
            buf.extend_from_slice(&v.to_le_bytes());
        } else {
            buf.extend_from_slice(&v.to_be_bytes());
        }
    }
    fn push_u32(buf: &mut Vec<u8>, v: u32, is_le: bool) {
        if is_le {
            buf.extend_from_slice(&v.to_le_bytes());
        } else {
            buf.extend_from_slice(&v.to_be_bytes());
        }
    }

    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(if is_le { b"II" } else { b"MM" });
    push_u16(&mut out, 42, is_le);
    push_u32(&mut out, 8, is_le); // IFD0 at offset 8

    // IFD0: 3 entries — pad with a "make" tag to more faithfully mimic
    // real camera TIFFs that interleave metadata with the JPEG pointers.
    push_u16(&mut out, 3, is_le);

    // Entry 1: Make (0x010F), ASCII, count=8, value offset points past IFD+JPEG
    // For simplicity we embed "TestCam\0" inline — ASCII with count <= 4 fits
    // in the value slot; count=8 requires an offset.  We point to a static
    // location after the JPEG.  To keep the test self-contained we'll skip
    // this entry by using a simple SHORT instead.
    push_u16(&mut out, 0x010F, is_le); // Make
    push_u16(&mut out, 3, is_le); // SHORT
    push_u32(&mut out, 1, is_le);
    push_u16(&mut out, 0, is_le); // dummy value
    push_u16(&mut out, 0, is_le); // pad

    // Entry 2: JPEGInterchangeFormat (0x0201), LONG, count=1, value=offset
    // JPEG payload begins after: header(8) + entry_count(2) + 3*12 + next_ifd(4) = 50.
    let jpeg_offset: u32 = 50;
    push_u16(&mut out, 0x0201, is_le);
    push_u16(&mut out, 4, is_le); // LONG
    push_u32(&mut out, 1, is_le);
    push_u32(&mut out, jpeg_offset, is_le);

    // Entry 3: JPEGInterchangeFormatLength (0x0202), LONG, count=1, value=len
    push_u16(&mut out, 0x0202, is_le);
    push_u16(&mut out, 4, is_le);
    push_u32(&mut out, 1, is_le);
    push_u32(&mut out, jpeg.len() as u32, is_le);

    // Next IFD offset = 0 (end of chain)
    push_u32(&mut out, 0, is_le);

    assert_eq!(
        out.len() as u32,
        jpeg_offset,
        "TIFF header layout mismatch: expected JPEG at {} but header ends at {}",
        jpeg_offset,
        out.len()
    );

    out.extend_from_slice(jpeg);
    out
}

fn make_gradient_rgb(w: usize, h: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            out.push(((x * 255) / w.max(1)) as u8);
            out.push(((y * 255) / h.max(1)) as u8);
            out.push((((x + y) * 127) / (w + h).max(1)) as u8);
        }
    }
    out
}

#[test]
fn raw_thumbnail_round_trip_little_endian() {
    // Step 1: produce a real JPEG via our encoder.
    let (w, h) = (32usize, 24usize);
    let rgb: Vec<u8> = make_gradient_rgb(w, h);
    let jpeg: Vec<u8> =
        compress(&rgb, w, h, PixelFormat::Rgb, 90, Subsampling::S444).expect("encode gradient");

    // Step 2: wrap in a minimal TIFF-6.0 container (LE, like Sony ARW).
    let container: Vec<u8> = wrap_jpeg_as_tiff(&jpeg, true);

    // Step 3: extract the embedded JPEG and verify bytes match exactly.
    let extracted: Vec<u8> = extract_embedded_jpeg(&container).expect("extract from LE container");
    assert_eq!(extracted, jpeg, "extracted JPEG must match original bytes");

    // Step 4: decode the extracted JPEG and compare dimensions.
    let img = decompress_to(&extracted, PixelFormat::Rgb).expect("decode extracted JPEG");
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
}

#[test]
fn raw_thumbnail_round_trip_big_endian() {
    // Some older / scientific raws use MM byte order.  Exercise that path.
    let (w, h) = (16usize, 16usize);
    let rgb: Vec<u8> = make_gradient_rgb(w, h);
    let jpeg: Vec<u8> =
        compress(&rgb, w, h, PixelFormat::Rgb, 85, Subsampling::S420).expect("encode gradient");
    let container: Vec<u8> = wrap_jpeg_as_tiff(&jpeg, false);

    let extracted: Vec<u8> = extract_embedded_jpeg(&container).expect("extract from BE container");
    assert_eq!(extracted, jpeg);

    let img = decompress_to(&extracted, PixelFormat::Rgb).expect("decode extracted JPEG");
    assert_eq!(img.width, w);
    assert_eq!(img.height, h);
}

#[test]
fn raw_thumbnail_from_checked_in_arw_fixture() {
    // The fixture under tests/fixtures/raw_thumbnail/ is a synthetic
    // ARW-style container: II-TIFF header, IFD0 with Make +
    // JPEGInterchangeFormat + JPEGInterchangeFormatLength tags, followed
    // by a real JPEG payload (24x16 RGB gradient at quality 85, 4:2:0)
    // produced by libjpeg-turbo's cjpeg.  This exercises the full
    // on-disk path — important because the inlined round-trip tests
    // cannot catch regressions in byte-order or offset parsing that
    // only surface with a real camera-shaped container.
    let path: std::path::PathBuf =
        std::path::PathBuf::from("tests/fixtures/raw_thumbnail/synthetic_arw_24x16.tiff");
    let blob: Vec<u8> = std::fs::read(&path).unwrap_or_else(|e| panic!("read {:?}: {:?}", path, e));

    let jpeg: Vec<u8> = extract_embedded_jpeg(&blob).expect("extract from ARW fixture");
    assert!(
        jpeg.starts_with(&[0xFF, 0xD8]),
        "extracted bytes must begin with SOI"
    );
    assert!(
        jpeg.ends_with(&[0xFF, 0xD9]),
        "extracted bytes must end with EOI"
    );

    let img =
        decompress_to(&jpeg, PixelFormat::Rgb).expect("decode extracted JPEG from ARW fixture");
    assert_eq!(img.width, 24, "fixture width mismatch");
    assert_eq!(img.height, 16, "fixture height mismatch");
}

#[test]
fn raw_thumbnail_rejects_malformed_container() {
    // Truncated header → CorruptData.
    let err = extract_embedded_jpeg(&[0x49, 0x49]).unwrap_err();
    let msg: String = format!("{}", err);
    assert!(
        msg.to_lowercase().contains("corrupt") || msg.to_lowercase().contains("tiff"),
        "unexpected error: {}",
        msg
    );

    // Well-formed TIFF with no JPEG-interchange tag → Unsupported.
    let mut tiff: Vec<u8> = Vec::new();
    tiff.extend_from_slice(b"II");
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&0u16.to_le_bytes()); // 0 entries
    tiff.extend_from_slice(&0u32.to_le_bytes()); // next IFD = 0
    let err = extract_embedded_jpeg(&tiff).unwrap_err();
    let msg: String = format!("{}", err);
    assert!(
        msg.to_lowercase().contains("no embedded jpeg")
            || msg.to_lowercase().contains("unsupported"),
        "expected 'no embedded JPEG' error, got: {}",
        msg
    );
}
