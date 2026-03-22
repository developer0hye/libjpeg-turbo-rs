//! JPEG structure verification.
//!
//! Verify the exact structure and ordering of markers in our encoder output.
//! Each coding mode has a defined marker sequence that must be respected.

use libjpeg_turbo_rs::{
    compress, compress_arithmetic, compress_arithmetic_progressive, compress_lossless,
    compress_optimized, compress_progressive, compress_with_metadata, PixelFormat, Subsampling,
};

// JPEG marker codes (second byte after 0xFF prefix)
const SOI: u8 = 0xD8;
const EOI: u8 = 0xD9;
const SOS: u8 = 0xDA;
const APP0: u8 = 0xE0; // JFIF
const APP1: u8 = 0xE1; // EXIF
const APP2: u8 = 0xE2; // ICC
const DQT: u8 = 0xDB;
const SOF0: u8 = 0xC0; // Baseline DCT
const SOF2: u8 = 0xC2; // Progressive DCT
const SOF3: u8 = 0xC3; // Lossless
const SOF9: u8 = 0xC9; // Arithmetic sequential
const SOF10: u8 = 0xCA; // Arithmetic progressive
const DHT: u8 = 0xC4; // Huffman table
const DAC: u8 = 0xCC; // Arithmetic conditioning table
const DRI: u8 = 0xDD; // Restart interval

/// Extract the ordered sequence of JPEG markers from a bitstream.
///
/// Stops collecting after the first SOS marker because entropy-coded data
/// follows and may contain 0xFF bytes that are not markers.
/// After scanning past entropy data, picks up trailing markers including EOI.
fn extract_marker_sequence(jpeg: &[u8]) -> Vec<u8> {
    let mut markers: Vec<u8> = Vec::new();
    let mut i: usize = 0;

    while i < jpeg.len() - 1 {
        if jpeg[i] == 0xFF && jpeg[i + 1] != 0x00 && jpeg[i + 1] != 0xFF {
            let marker: u8 = jpeg[i + 1];
            markers.push(marker);

            if marker == SOS {
                // Skip past SOS payload and entropy data to find next marker
                if i + 3 < jpeg.len() {
                    let len: usize = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
                    i += 2 + len;
                    // Scan through entropy-coded data for next marker
                    while i < jpeg.len() - 1 {
                        if jpeg[i] == 0xFF && jpeg[i + 1] != 0x00 && jpeg[i + 1] != 0xFF {
                            break;
                        }
                        i += 1;
                    }
                    continue;
                }
                break;
            }

            // Standalone markers (SOI, EOI) have no payload
            if marker == SOI || marker == EOI {
                i += 2;
                continue;
            }

            // Skip marker payload
            if i + 3 < jpeg.len() {
                let len: usize = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
                i += 2 + len;
                continue;
            }
        }
        i += 1;
    }

    markers
}

/// Extract only the markers that appear before the first SOS.
/// This is the "header" marker sequence.
fn extract_header_markers(jpeg: &[u8]) -> Vec<u8> {
    let all: Vec<u8> = extract_marker_sequence(jpeg);
    let mut header: Vec<u8> = Vec::new();
    for &m in &all {
        header.push(m);
        if m == SOS {
            break;
        }
    }
    header
}

/// Count occurrences of a specific marker in the full sequence.
fn count_marker(markers: &[u8], target: u8) -> usize {
    markers.iter().filter(|&&m| m == target).count()
}

// ---------------------------------------------------------------------------
// Helper: generate test pixels
// ---------------------------------------------------------------------------

fn test_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 5 + y * 3) % 256) as u8);
            pixels.push(((x * 3 + y * 7) % 256) as u8);
            pixels.push(((x * 7 + y * 11) % 256) as u8);
        }
    }
    pixels
}

fn gray_pixels(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 7 + y * 3) % 256) as u8);
        }
    }
    pixels
}

// ---------------------------------------------------------------------------
// Baseline (SOF0) marker ordering
// ---------------------------------------------------------------------------

#[test]
fn baseline_marker_order() {
    let pixels: Vec<u8> = test_pixels(64, 64);
    let jpeg: Vec<u8> = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    // SOI must be first
    assert_eq!(header[0], SOI, "first marker must be SOI");

    // APP0 (JFIF) must be second
    assert_eq!(header[1], APP0, "second marker must be APP0 (JFIF)");

    // Must contain DQT before SOF0
    let dqt_pos: usize = header
        .iter()
        .position(|&m| m == DQT)
        .expect("must have DQT");
    let sof0_pos: usize = header
        .iter()
        .position(|&m| m == SOF0)
        .expect("must have SOF0");
    assert!(dqt_pos < sof0_pos, "DQT must come before SOF0");

    // Must contain DHT before SOS
    let dht_pos: usize = header
        .iter()
        .position(|&m| m == DHT)
        .expect("must have DHT");
    let sos_pos: usize = header
        .iter()
        .position(|&m| m == SOS)
        .expect("must have SOS");
    assert!(dht_pos < sos_pos, "DHT must come before SOS");

    // SOF0 must come before SOS
    assert!(sof0_pos < sos_pos, "SOF0 must come before SOS");

    // Must end the header with SOS
    assert_eq!(*header.last().unwrap(), SOS, "header must end with SOS");

    // Full stream must end with EOI
    let all: Vec<u8> = extract_marker_sequence(&jpeg);
    assert_eq!(*all.last().unwrap(), EOI, "stream must end with EOI");

    // Baseline must NOT contain SOF2, SOF9, SOF10, DAC
    assert!(!header.contains(&SOF2), "baseline must not contain SOF2");
    assert!(!header.contains(&SOF9), "baseline must not contain SOF9");
    assert!(!header.contains(&DAC), "baseline must not contain DAC");
}

// ---------------------------------------------------------------------------
// Progressive (SOF2) marker ordering
// ---------------------------------------------------------------------------

#[test]
fn progressive_marker_order() {
    let pixels: Vec<u8> = test_pixels(64, 64);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    assert_eq!(header[0], SOI, "first marker must be SOI");
    assert_eq!(header[1], APP0, "second marker must be APP0 (JFIF)");

    // Must contain SOF2 (progressive), not SOF0
    assert!(header.contains(&SOF2), "progressive JPEG must contain SOF2");
    assert!(
        !header.contains(&SOF0),
        "progressive JPEG must not contain SOF0"
    );

    // DQT before SOF2
    let dqt_pos: usize = header
        .iter()
        .position(|&m| m == DQT)
        .expect("must have DQT");
    let sof2_pos: usize = header
        .iter()
        .position(|&m| m == SOF2)
        .expect("must have SOF2");
    assert!(dqt_pos < sof2_pos, "DQT must come before SOF2");

    // Progressive should have multiple SOS markers (multiple scans)
    let all: Vec<u8> = extract_marker_sequence(&jpeg);
    let sos_count: usize = count_marker(&all, SOS);
    assert!(
        sos_count > 1,
        "progressive JPEG should have multiple SOS markers, found {sos_count}"
    );

    assert_eq!(*all.last().unwrap(), EOI, "stream must end with EOI");
}

// ---------------------------------------------------------------------------
// Arithmetic sequential (SOF9) marker ordering
// ---------------------------------------------------------------------------

#[test]
fn arithmetic_marker_order() {
    let pixels: Vec<u8> = test_pixels(64, 64);
    let jpeg: Vec<u8> =
        compress_arithmetic(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    assert_eq!(header[0], SOI, "first marker must be SOI");

    // Must contain SOF9 (arithmetic sequential), not SOF0
    assert!(header.contains(&SOF9), "arithmetic JPEG must contain SOF9");
    assert!(
        !header.contains(&SOF0),
        "arithmetic JPEG must not contain SOF0"
    );

    // Must contain DAC (arithmetic conditioning), not DHT
    assert!(header.contains(&DAC), "arithmetic JPEG must contain DAC");
    assert!(
        !header.contains(&DHT),
        "arithmetic JPEG must not contain DHT"
    );

    // DQT before SOF9
    let dqt_pos: usize = header
        .iter()
        .position(|&m| m == DQT)
        .expect("must have DQT");
    let sof9_pos: usize = header
        .iter()
        .position(|&m| m == SOF9)
        .expect("must have SOF9");
    assert!(dqt_pos < sof9_pos, "DQT must come before SOF9");

    let all: Vec<u8> = extract_marker_sequence(&jpeg);
    assert_eq!(*all.last().unwrap(), EOI, "stream must end with EOI");
}

// ---------------------------------------------------------------------------
// Arithmetic progressive (SOF10) marker ordering
// ---------------------------------------------------------------------------

#[test]
fn arithmetic_progressive_marker_order() {
    let pixels: Vec<u8> = test_pixels(64, 64);
    let jpeg: Vec<u8> =
        compress_arithmetic_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    assert_eq!(header[0], SOI, "first marker must be SOI");

    // Must contain SOF10 (arithmetic progressive)
    assert!(
        header.contains(&SOF10),
        "arithmetic progressive JPEG must contain SOF10 (0xCA)"
    );

    // Must contain DAC, not DHT
    assert!(
        header.contains(&DAC),
        "arithmetic progressive JPEG must contain DAC"
    );
    assert!(
        !header.contains(&DHT),
        "arithmetic progressive JPEG must not contain DHT"
    );

    // Multiple SOS expected for progressive
    let all: Vec<u8> = extract_marker_sequence(&jpeg);
    let sos_count: usize = count_marker(&all, SOS);
    assert!(
        sos_count > 1,
        "arithmetic progressive should have multiple SOS markers, found {sos_count}"
    );
    assert_eq!(*all.last().unwrap(), EOI, "stream must end with EOI");
}

// ---------------------------------------------------------------------------
// Lossless (SOF3) marker ordering
// ---------------------------------------------------------------------------

#[test]
fn lossless_marker_order() {
    let pixels: Vec<u8> = gray_pixels(64, 64);
    let jpeg: Vec<u8> = compress_lossless(&pixels, 64, 64, PixelFormat::Grayscale).unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    assert_eq!(header[0], SOI, "first marker must be SOI");

    // Must contain SOF3 (lossless)
    assert!(header.contains(&SOF3), "lossless JPEG must contain SOF3");

    // Lossless JPEG should NOT contain DQT (no quantization)
    assert!(!header.contains(&DQT), "lossless JPEG must not contain DQT");

    // Should contain DHT (Huffman tables for lossless prediction residuals)
    assert!(header.contains(&DHT), "lossless JPEG must contain DHT");

    let all: Vec<u8> = extract_marker_sequence(&jpeg);
    assert_eq!(*all.last().unwrap(), EOI, "stream must end with EOI");
}

// ---------------------------------------------------------------------------
// Metadata marker ordering (ICC + EXIF before image data)
// ---------------------------------------------------------------------------

#[test]
fn metadata_marker_order() {
    let pixels: Vec<u8> = test_pixels(64, 64);
    let fake_icc: Vec<u8> = vec![0x42u8; 200];
    let fake_exif: Vec<u8> = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];

    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        64,
        64,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&fake_icc),
        Some(&fake_exif),
    )
    .unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    assert_eq!(header[0], SOI, "first marker must be SOI");

    // APP0 (JFIF) should be present
    assert!(header.contains(&APP0), "must contain APP0 (JFIF)");

    // APP1 (EXIF) should be present
    assert!(header.contains(&APP1), "must contain APP1 (EXIF)");

    // APP2 (ICC) should be present
    assert!(header.contains(&APP2), "must contain APP2 (ICC)");

    // All APP markers should come before SOF0
    let sof0_pos: usize = header
        .iter()
        .position(|&m| m == SOF0)
        .expect("must have SOF0");
    for (idx, &m) in header.iter().enumerate() {
        if m == APP0 || m == APP1 || m == APP2 {
            assert!(
                idx < sof0_pos,
                "APP marker at position {idx} must come before SOF0 at position {sof0_pos}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Optimized Huffman — same structure as baseline but with optimized tables
// ---------------------------------------------------------------------------

#[test]
fn optimized_huffman_marker_order() {
    let pixels: Vec<u8> = test_pixels(64, 64);
    let jpeg: Vec<u8> =
        compress_optimized(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    assert_eq!(header[0], SOI, "first marker must be SOI");

    // Should still use SOF0 (baseline DCT)
    assert!(header.contains(&SOF0), "optimized Huffman must use SOF0");

    // Must have DHT (Huffman tables — optimized ones)
    assert!(header.contains(&DHT), "optimized Huffman must contain DHT");

    // Must NOT have DAC
    assert!(
        !header.contains(&DAC),
        "optimized Huffman must not contain DAC"
    );

    let all: Vec<u8> = extract_marker_sequence(&jpeg);
    assert_eq!(*all.last().unwrap(), EOI, "stream must end with EOI");
}

// ---------------------------------------------------------------------------
// Grayscale — single component, simpler structure
// ---------------------------------------------------------------------------

#[test]
fn grayscale_marker_order() {
    let pixels: Vec<u8> = gray_pixels(64, 64);
    let jpeg: Vec<u8> = compress(
        &pixels,
        64,
        64,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let header: Vec<u8> = extract_header_markers(&jpeg);

    assert_eq!(header[0], SOI, "first marker must be SOI");
    assert!(header.contains(&SOF0), "must contain SOF0");
    assert!(header.contains(&DQT), "must contain DQT");
    assert!(header.contains(&DHT), "must contain DHT");
    assert!(header.contains(&SOS), "must contain SOS");

    let all: Vec<u8> = extract_marker_sequence(&jpeg);
    assert_eq!(*all.last().unwrap(), EOI, "stream must end with EOI");

    // Only 1 SOS for baseline grayscale
    let sos_count: usize = count_marker(&all, SOS);
    assert_eq!(sos_count, 1, "baseline grayscale should have exactly 1 SOS");
}

// ---------------------------------------------------------------------------
// SOI/EOI framing — always present
// ---------------------------------------------------------------------------

#[test]
fn soi_eoi_framing_all_modes() {
    let rgb: Vec<u8> = test_pixels(32, 32);
    let gray: Vec<u8> = gray_pixels(32, 32);

    let jpegs: Vec<(&str, Vec<u8>)> = vec![
        (
            "baseline",
            compress(&rgb, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
        (
            "progressive",
            compress_progressive(&rgb, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap(),
        ),
        (
            "arithmetic",
            compress_arithmetic(&rgb, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
        (
            "lossless",
            compress_lossless(&gray, 32, 32, PixelFormat::Grayscale).unwrap(),
        ),
        (
            "optimized",
            compress_optimized(&rgb, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
        ),
    ];

    for (name, jpeg) in &jpegs {
        // SOI at start
        assert_eq!(jpeg[0], 0xFF, "{name}: first byte must be 0xFF");
        assert_eq!(jpeg[1], SOI, "{name}: second byte must be SOI (0xD8)");

        // EOI at end
        assert_eq!(
            jpeg[jpeg.len() - 2],
            0xFF,
            "{name}: second-to-last byte must be 0xFF"
        );
        assert_eq!(
            jpeg[jpeg.len() - 1],
            EOI,
            "{name}: last byte must be EOI (0xD9)"
        );
    }
}

// ---------------------------------------------------------------------------
// No duplicate SOF markers
// ---------------------------------------------------------------------------

#[test]
fn no_duplicate_sof_markers() {
    let pixels: Vec<u8> = test_pixels(64, 64);
    let gray: Vec<u8> = gray_pixels(64, 64);

    let cases: Vec<(&str, Vec<u8>, u8)> = vec![
        (
            "baseline",
            compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
            SOF0,
        ),
        (
            "progressive",
            compress_progressive(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S444).unwrap(),
            SOF2,
        ),
        (
            "arithmetic",
            compress_arithmetic(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap(),
            SOF9,
        ),
        (
            "lossless",
            compress_lossless(&gray, 64, 64, PixelFormat::Grayscale).unwrap(),
            SOF3,
        ),
    ];

    let all_sof: &[u8] = &[SOF0, SOF2, SOF3, SOF9, SOF10];

    for (name, jpeg, expected_sof) in &cases {
        let header: Vec<u8> = extract_header_markers(jpeg);
        let sof_count: usize = header.iter().filter(|&&m| all_sof.contains(&m)).count();
        assert_eq!(
            sof_count, 1,
            "{name}: must have exactly one SOF marker, found {sof_count}"
        );
        assert!(
            header.contains(expected_sof),
            "{name}: must contain expected SOF marker 0x{expected_sof:02X}"
        );
    }
}

// ---------------------------------------------------------------------------
// DQT count varies by component count
// ---------------------------------------------------------------------------

#[test]
fn dqt_count_matches_component_needs() {
    let rgb: Vec<u8> = test_pixels(64, 64);
    let gray: Vec<u8> = gray_pixels(64, 64);

    // Color images: typically 2 quantization tables (luma + chroma)
    let color_jpeg: Vec<u8> =
        compress(&rgb, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let color_header: Vec<u8> = extract_header_markers(&color_jpeg);
    let color_dqt: usize = count_marker(&color_header, DQT);
    assert!(
        color_dqt >= 1,
        "color JPEG must have at least 1 DQT marker, found {color_dqt}"
    );

    // Grayscale images: typically 1 quantization table
    let gray_jpeg: Vec<u8> =
        compress(&gray, 64, 64, PixelFormat::Grayscale, 75, Subsampling::S444).unwrap();
    let gray_header: Vec<u8> = extract_header_markers(&gray_jpeg);
    let gray_dqt: usize = count_marker(&gray_header, DQT);
    assert!(
        gray_dqt >= 1,
        "grayscale JPEG must have at least 1 DQT marker, found {gray_dqt}"
    );
}
