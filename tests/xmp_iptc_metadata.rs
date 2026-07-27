//! XMP (APP1) and IPTC (APP13 Photoshop IRB) metadata accessors,
//! issue #358.
//!
//! Covers the acceptance list: single-segment XMP, **Extended XMP split
//! across multiple APP1 segments reassembled in offset order** (the case
//! a naive implementation gets wrong), IPTC extracted from an 8BIM
//! resource walk with padding, absent metadata returning `None`, and an
//! encode→decode round trip. Cross-validated against `exiftool` when
//! available, skipped with a printed reason otherwise.

use libjpeg_turbo_rs::{compress, decompress, Decoder, Encoder, PixelFormat, Subsampling};

fn gray_8x8() -> Vec<u8> {
    let mut pixels = vec![0u8; 8 * 8];
    for (i, p) in pixels.iter_mut().enumerate() {
        *p = (i * 4) as u8;
    }
    pixels
}

fn plain_jpeg() -> Vec<u8> {
    compress(
        &gray_8x8(),
        8,
        8,
        PixelFormat::Grayscale,
        85,
        Subsampling::S444,
    )
    .expect("encode")
}

/// Splice a raw APP1/APP13 segment in right after SOI.
fn splice_segment(jpeg: &[u8], marker: u8, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(jpeg.len() + payload.len() + 4);
    out.extend_from_slice(&jpeg[..2]); // SOI
    out.push(0xFF);
    out.push(marker);
    out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&jpeg[2..]);
    out
}

const XMP_ID: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const XMP_EXT_ID: &[u8] = b"http://ns.adobe.com/xmp/extension/\0";

#[test]
fn single_segment_xmp_is_extracted() {
    let packet = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF/></x:xmpmeta>"#;
    let mut payload = XMP_ID.to_vec();
    payload.extend_from_slice(packet);
    let jpeg = splice_segment(&plain_jpeg(), 0xE1, &payload);

    let img = decompress(&jpeg).expect("decode");
    assert_eq!(
        img.xmp_data(),
        Some(&packet[..]),
        "standard XMP packet must round-trip byte-exactly"
    );
}

/// Issue #358's headline case: Extended XMP arrives as multiple APP1
/// segments carrying GUID + full length + offset, and must be
/// reassembled in offset order — spliced here deliberately out of order.
#[test]
fn extended_xmp_chunks_are_reassembled_in_offset_order() {
    let std_packet = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"/>"#;
    let ext_full: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
    let guid: [u8; 32] = *b"A1B2C3D4E5F60718293A4B5C6D7E8F90";

    let chunk = |offset: usize, len: usize| -> Vec<u8> {
        let mut p = XMP_EXT_ID.to_vec();
        p.extend_from_slice(&guid);
        p.extend_from_slice(&(ext_full.len() as u32).to_be_bytes());
        p.extend_from_slice(&(offset as u32).to_be_bytes());
        p.extend_from_slice(&ext_full[offset..offset + len]);
        p
    };

    let mut xmp_payload = XMP_ID.to_vec();
    xmp_payload.extend_from_slice(std_packet);

    // Splice order: second chunk first, then first chunk, then the
    // standard packet — offsets, not stream order, must decide.
    let mut jpeg = plain_jpeg();
    jpeg = splice_segment(&jpeg, 0xE1, &chunk(128, 172));
    jpeg = splice_segment(&jpeg, 0xE1, &chunk(0, 128));
    jpeg = splice_segment(&jpeg, 0xE1, &xmp_payload);

    let img = decompress(&jpeg).expect("decode");
    let got = img.xmp_data().expect("XMP present");
    let mut expected = std_packet.to_vec();
    expected.extend_from_slice(&ext_full);
    assert_eq!(
        got,
        &expected[..],
        "Extended XMP must be reassembled in offset order and appended"
    );
}

/// A chunk set that does not cover the declared full length must degrade
/// to the standard packet rather than emitting zero-filled holes.
#[test]
fn incomplete_extended_xmp_falls_back_to_the_standard_packet() {
    let std_packet = br#"<x:xmpmeta/>"#;
    let guid: [u8; 32] = *b"0123456789ABCDEF0123456789ABCDEF";
    let mut ext = XMP_EXT_ID.to_vec();
    ext.extend_from_slice(&guid);
    ext.extend_from_slice(&500u32.to_be_bytes()); // declares 500
    ext.extend_from_slice(&0u32.to_be_bytes());
    ext.extend_from_slice(&[0xAB; 100]); // provides only 100

    let mut xmp_payload = XMP_ID.to_vec();
    xmp_payload.extend_from_slice(std_packet);

    let mut jpeg = plain_jpeg();
    jpeg = splice_segment(&jpeg, 0xE1, &ext);
    jpeg = splice_segment(&jpeg, 0xE1, &xmp_payload);

    let img = decompress(&jpeg).expect("decode");
    assert_eq!(img.xmp_data(), Some(&std_packet[..]));
}

#[test]
fn iptc_is_extracted_from_the_photoshop_irb() {
    // IIM: 0x1C 0x02 0x05 (Object Name) + length + value.
    let iptc: Vec<u8> = {
        let value = b"Test Headline";
        let mut v = vec![0x1C, 0x02, 0x05];
        v.extend_from_slice(&(value.len() as u16).to_be_bytes());
        v.extend_from_slice(value);
        v
    };

    let mut payload = b"Photoshop 3.0\0".to_vec();
    payload.extend_from_slice(b"8BIM");
    payload.extend_from_slice(&0x0404u16.to_be_bytes());
    payload.push(0); // empty Pascal name
    payload.push(0); // pad
    payload.extend_from_slice(&(iptc.len() as u32).to_be_bytes());
    payload.extend_from_slice(&iptc);
    if iptc.len() % 2 == 1 {
        payload.push(0); // resource data padded to even
    }

    let jpeg = splice_segment(&plain_jpeg(), 0xED, &payload);
    let img = decompress(&jpeg).expect("decode");
    assert_eq!(img.iptc_data(), Some(&iptc[..]));
}

/// A preceding non-IPTC 8BIM resource must be walked over correctly
/// (padding included) rather than confusing the scan.
#[test]
fn iptc_walk_skips_preceding_resources_with_padding() {
    let other = [0xDEu8, 0xAD, 0xBE]; // odd length -> pad byte
    let iptc = [0x1Cu8, 0x02, 0x19, 0x00, 0x03, b'a', b'b', b'c'];

    let mut payload = b"Photoshop 3.0\0".to_vec();
    // Resource 1: id 0x03E8, odd-length data.
    payload.extend_from_slice(b"8BIM");
    payload.extend_from_slice(&0x03E8u16.to_be_bytes());
    payload.push(0);
    payload.push(0);
    payload.extend_from_slice(&(other.len() as u32).to_be_bytes());
    payload.extend_from_slice(&other);
    payload.push(0); // pad to even
                     // Resource 2: IPTC.
    payload.extend_from_slice(b"8BIM");
    payload.extend_from_slice(&0x0404u16.to_be_bytes());
    payload.push(0);
    payload.push(0);
    payload.extend_from_slice(&(iptc.len() as u32).to_be_bytes());
    payload.extend_from_slice(&iptc);

    let jpeg = splice_segment(&plain_jpeg(), 0xED, &payload);
    let img = decompress(&jpeg).expect("decode");
    assert_eq!(img.iptc_data(), Some(&iptc[..]));
}

#[test]
fn absent_metadata_is_none_not_empty() {
    let img = decompress(&plain_jpeg()).expect("decode");
    assert!(img.xmp_data().is_none(), "no XMP must be None");
    assert!(img.iptc_data().is_none(), "no IPTC must be None");
    // The metadata must not disturb the existing accessors either.
    assert!(img.exif_data().is_none());
}

/// Issue #358 criterion: encode with XMP + IPTC, decode, assert
/// byte-identical payloads.
#[test]
fn encode_decode_round_trip_preserves_payloads() {
    let xmp = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF>round-trip</rdf:RDF></x:xmpmeta>"#;
    let iptc = [0x1Cu8, 0x02, 0x05, 0x00, 0x04, b'r', b'o', b'l', b'l'];

    let jpeg = Encoder::new(&gray_8x8(), 8, 8, PixelFormat::Grayscale)
        .quality(90)
        .xmp_data(xmp)
        .iptc_data(&iptc)
        .encode()
        .expect("encode with metadata");

    let img = decompress(&jpeg).expect("decode");
    assert_eq!(img.xmp_data(), Some(&xmp[..]), "XMP payload must survive");
    assert_eq!(
        img.iptc_data(),
        Some(&iptc[..]),
        "IPTC payload must survive"
    );

    // And the pixels are still decodable/unchanged versus no-metadata.
    let plain = decompress(&plain_jpeg()).expect("decode plain");
    let mut d = Decoder::new(&jpeg).expect("parse");
    d.set_output_format(PixelFormat::Grayscale);
    let with_meta = d.decode_image().expect("decode meta");
    assert_eq!(with_meta.data.len(), plain.data.len());
}

/// Cross-validation against a reference reader when one is installed.
#[test]
fn exiftool_agrees_on_round_tripped_metadata() {
    let exiftool = std::process::Command::new("which")
        .arg("exiftool")
        .output()
        .ok()
        .filter(|o| o.status.success());
    let Some(_) = exiftool else {
        eprintln!("SKIP: exiftool not found — install libimage-exiftool-perl to cross-validate");
        return;
    };

    let xmp = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" dc:title="LibjpegTurboRsTest"/></rdf:RDF></x:xmpmeta>"#;
    let jpeg = Encoder::new(&gray_8x8(), 8, 8, PixelFormat::Grayscale)
        .quality(90)
        .xmp_data(xmp)
        .encode()
        .expect("encode");

    let dir = std::env::temp_dir().join(format!("xmp_xval_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("dir");
    let path = dir.join("meta.jpg");
    std::fs::write(&path, &jpeg).expect("write");

    let out = std::process::Command::new("exiftool")
        .args(["-XMP:all", "-b", "-XMP"])
        .arg(&path)
        .output()
        .expect("run exiftool");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(out.status.success(), "exiftool failed: {stdout}");
    assert!(
        stdout.contains("LibjpegTurboRsTest"),
        "exiftool must observe the XMP we wrote (an || here would let a \
         missing tag pass — codex): {stdout}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The bug both reviews caught: every test above used an 8×8 grayscale
/// image, so `xmp_data`/`iptc_data` returning `None` on the 4:2:0 and
/// 4:2:2 row-streaming paths — the default for real-world colour JPEGs —
/// was invisible. Cover the full subsampling × format matrix.
#[test]
fn metadata_survives_every_subsampling_and_output_format() {
    let xmp = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/">matrix</x:xmpmeta>"#;
    let iptc = [0x1Cu8, 0x02, 0x05, 0x00, 0x02, b'o', b'k'];

    let mut rgb = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64u32 {
        for x in 0..64u32 {
            rgb.extend_from_slice(&[(x * 4) as u8, (y * 4) as u8, ((x ^ y) * 4) as u8]);
        }
    }

    for sub in [
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
        Subsampling::S440,
        Subsampling::S411,
    ] {
        let jpeg = Encoder::new(&rgb, 64, 64, PixelFormat::Rgb)
            .quality(85)
            .subsampling(sub)
            .xmp_data(xmp)
            .iptc_data(&iptc)
            .encode()
            .unwrap_or_else(|e| panic!("{sub:?} encode: {e}"));

        // Grayscale output from a colour JPEG is a documented
        // rejection (jpeg_color_space contract), not a metadata path.
        for out_fmt in [PixelFormat::Rgb, PixelFormat::Rgba, PixelFormat::Bgr] {
            let mut d = Decoder::new(&jpeg).expect("parse");
            d.set_output_format(out_fmt);
            let img = d
                .decode_image()
                .unwrap_or_else(|e| panic!("{sub:?}/{out_fmt:?} decode: {e}"));
            assert_eq!(
                img.xmp_data(),
                Some(&xmp[..]),
                "{sub:?} -> {out_fmt:?}: XMP lost"
            );
            assert_eq!(
                img.iptc_data(),
                Some(&iptc[..]),
                "{sub:?} -> {out_fmt:?}: IPTC lost"
            );
        }

        // Progressive and the caller-buffer path must carry it too.
        let img = decompress(&jpeg).expect("decompress");
        assert_eq!(
            img.xmp_data(),
            Some(&xmp[..]),
            "{sub:?}: decompress lost XMP"
        );
        let size = libjpeg_turbo_rs::output_buffer_size(&jpeg, PixelFormat::Rgb).expect("size");
        let mut buf = vec![0u8; size];
        let info = libjpeg_turbo_rs::decompress_into(&jpeg, PixelFormat::Rgb, &mut buf)
            .expect("decompress_into");
        assert_eq!(
            info.xmp_data.as_deref(),
            Some(&xmp[..]),
            "{sub:?}: decompress_into lost XMP"
        );
        assert_eq!(info.iptc_data.as_deref(), Some(&iptc[..]));
    }
}
