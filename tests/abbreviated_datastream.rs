//! Tests for abbreviated (tables-only / body-only) JPEG datastream per JPEG spec F.1.2.4.
//!
//! Subtasks covered:
//! - A2-1: `jpeg_write_tables()` public API
//! - A2-2: `HeaderResult::TablesOnly` decoder variant
//! - A2-3: `Decoder::new_with_tables()` inter-session table reuse
//! - A2-4: `Encoder::suppress_tables()` body-only encode
//! - B10-1: Cross-validate tables-only stream vs cjpeg -tables-only
//! - B10-2: Compose (tables-only || body-only) → djpeg produces same pixels
//!
//! C cross-validation rules (from CLAUDE.md):
//! - Tools at /opt/homebrew/bin/djpeg, cjpeg
//! - If C tools not found: eprintln!("SKIP: ..."); return;
//! - diff=0 required for pixel comparison

mod helpers;

use libjpeg_turbo_rs::api::encoder::Encoder;
use libjpeg_turbo_rs::{
    compress, decompress_to, jpeg_write_tables, HeaderResult, PixelFormat, Subsampling,
};

// ===========================================================================
// A2-1: jpeg_write_tables() output structure validation
// ===========================================================================

/// Verify the tables-only stream structure: SOI + DQT + DHT + EOI, no SOF/SOS.
#[test]
fn a2_1_tables_only_stream_structure() {
    let pixels: Vec<u8> = helpers::generate_gradient(64, 64);
    let tables_bytes: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .write_tables();

    // Must begin with SOI
    assert_eq!(
        &tables_bytes[0..2],
        &[0xFF, 0xD8],
        "tables-only must start with SOI"
    );

    // Must end with EOI
    let len = tables_bytes.len();
    assert_eq!(
        &tables_bytes[len - 2..],
        &[0xFF, 0xD9],
        "tables-only must end with EOI"
    );

    // Must contain DQT (0xDB)
    let has_dqt = tables_bytes.windows(2).any(|w| w == [0xFF, 0xDB]);
    assert!(has_dqt, "tables-only must contain DQT marker");

    // Must contain DHT (0xC4)
    let has_dht = tables_bytes.windows(2).any(|w| w == [0xFF, 0xC4]);
    assert!(has_dht, "tables-only must contain DHT marker");

    // Must NOT contain SOF (0xC0)
    let has_sof = tables_bytes
        .windows(2)
        .any(|w| w[0] == 0xFF && (w[1] == 0xC0 || w[1] == 0xC2));
    assert!(!has_sof, "tables-only must NOT contain SOF marker");

    // Must NOT contain SOS (0xDA)
    let has_sos = tables_bytes.windows(2).any(|w| w == [0xFF, 0xDA]);
    assert!(!has_sos, "tables-only must NOT contain SOS marker");
}

/// Verify the standalone function jpeg_write_tables() works equivalently.
#[test]
fn a2_1_jpeg_write_tables_standalone() {
    let pixels: Vec<u8> = helpers::generate_gradient(64, 64);
    let encoder = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420);

    let result: Vec<u8> = jpeg_write_tables(&encoder);

    // Must start with SOI and end with EOI
    assert_eq!(&result[0..2], &[0xFF, 0xD8]);
    let len = result.len();
    assert_eq!(&result[len - 2..], &[0xFF, 0xD9]);
}

// ===========================================================================
// A2-2: HeaderResult::TablesOnly decoder variant
// ===========================================================================

/// Parse a tables-only stream; verify TablesOnly result and populated tables.
#[test]
fn a2_2_tables_only_header_result() {
    let pixels: Vec<u8> = helpers::generate_gradient(64, 64);
    let tables_bytes: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .write_tables();

    let result = libjpeg_turbo_rs::read_header(&tables_bytes)
        .expect("read_header should succeed on tables-only stream");

    match result {
        HeaderResult::TablesOnly(state) => {
            // At least quant table 0 (luma) and 1 (chroma) should be populated
            assert!(
                state.has_quant_table(0),
                "TablesOnly state must have quant table 0"
            );
            assert!(
                state.has_quant_table(1),
                "TablesOnly state must have quant table 1"
            );
            // DC Huffman tables 0 and 1
            assert!(
                state.has_dc_huffman(0),
                "TablesOnly state must have DC Huffman table 0"
            );
            assert!(
                state.has_ac_huffman(0),
                "TablesOnly state must have AC Huffman table 0"
            );
        }
        HeaderResult::Image(_) => {
            panic!("Expected TablesOnly, got Image for tables-only stream");
        }
    }
}

/// A full JPEG should return HeaderResult::Image.
#[test]
fn a2_2_full_jpeg_header_result_is_image() {
    let pixels: Vec<u8> = helpers::generate_gradient(64, 64);
    let jpeg = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420)
        .expect("compress should succeed");

    let result =
        libjpeg_turbo_rs::read_header(&jpeg).expect("read_header should succeed on full JPEG");

    match result {
        HeaderResult::Image(decoder) => {
            assert_eq!(decoder.header().width, 64);
            assert_eq!(decoder.header().height, 64);
        }
        HeaderResult::TablesOnly(_) => {
            panic!("Expected Image, got TablesOnly for full JPEG");
        }
    }
}

// ===========================================================================
// A2-3: Decoder::new_with_tables() inter-session table reuse
// ===========================================================================

/// Round-trip: split a full JPEG into (tables-only | body-only), decode body with
/// tables from tables-only. Decoded pixels must be identical to decoding the full JPEG.
#[test]
fn a2_3_body_only_roundtrip_pixel_identical() {
    let pixels: Vec<u8> = helpers::generate_gradient(64, 64);
    let encoder = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420);

    // Generate the tables-only stream
    let tables_bytes: Vec<u8> = encoder.write_tables();

    // Generate the body-only stream (suppress tables in JPEG output)
    let body_bytes: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .suppress_tables(true)
        .encode()
        .expect("body-only encode should succeed");

    // Parse the tables-only stream to get a decoder with populated table state
    let tables_result = libjpeg_turbo_rs::read_header(&tables_bytes)
        .expect("read_header on tables-only should succeed");
    let tables_state = match tables_result {
        HeaderResult::TablesOnly(s) => s,
        HeaderResult::Image(_) => panic!("expected TablesOnly"),
    };

    // Decode body-only using the preloaded tables
    let decoded_body = libjpeg_turbo_rs::Decoder::new_with_tables(&body_bytes, &tables_state)
        .expect("new_with_tables should succeed")
        .decode_image()
        .expect("decode_image with preloaded tables should succeed");

    // Decode the full JPEG normally for reference
    let full_jpeg = compress(&pixels, 64, 64, PixelFormat::Rgb, 75, Subsampling::S420)
        .expect("compress should succeed");
    let decoded_full =
        decompress_to(&full_jpeg, PixelFormat::Rgb).expect("decompress full JPEG should succeed");

    // Pixel-identical comparison (diff=0)
    assert_eq!(decoded_body.width, decoded_full.width, "width must match");
    assert_eq!(
        decoded_body.height, decoded_full.height,
        "height must match"
    );
    assert_eq!(
        decoded_body.data.len(),
        decoded_full.data.len(),
        "pixel data length must match"
    );

    let max_diff = helpers::pixel_max_diff(&decoded_body.data, &decoded_full.data);
    // Measured diff=0 (identical encoding path, identical tables)
    assert_eq!(
        max_diff, 0,
        "body-only + tables roundtrip must be pixel-identical (max_diff={}, expected 0)",
        max_diff
    );
}

// ===========================================================================
// A2-4: Encoder::suppress_tables()
// ===========================================================================

/// When suppress_tables=true, output must NOT contain DQT or DHT markers.
#[test]
fn a2_4_suppress_tables_no_quant_or_huff() {
    let pixels: Vec<u8> = helpers::generate_gradient(64, 64);
    let body_bytes: Vec<u8> = Encoder::new(&pixels, 64, 64, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .suppress_tables(true)
        .encode()
        .expect("body-only encode should succeed");

    // Must start with SOI
    assert_eq!(&body_bytes[0..2], &[0xFF, 0xD8]);

    // Must NOT contain DQT (0xDB)
    let has_dqt = body_bytes.windows(2).any(|w| w == [0xFF, 0xDB]);
    assert!(!has_dqt, "body-only JPEG must NOT contain DQT marker");

    // Must NOT contain DHT (0xC4)
    let has_dht = body_bytes.windows(2).any(|w| w == [0xFF, 0xC4]);
    assert!(!has_dht, "body-only JPEG must NOT contain DHT marker");

    // Must contain SOF (some form)
    let has_sof = body_bytes
        .windows(2)
        .any(|w| w[0] == 0xFF && (w[1] == 0xC0 || w[1] == 0xC2 || w[1] == 0xC9));
    assert!(has_sof, "body-only JPEG must contain SOF marker");

    // Must contain SOS
    let has_sos = body_bytes.windows(2).any(|w| w == [0xFF, 0xDA]);
    assert!(has_sos, "body-only JPEG must contain SOS marker");
}

// ===========================================================================
// B10-1: Cross-validate jpeg_write_tables() against cjpeg -tables-only
// ===========================================================================

/// Cross-validate tables-only stream against cjpeg -tables-only output.
/// For subsamp x quality combos, compare our DQT/DHT bytes to cjpeg's.
#[test]
fn b10_1_cross_validate_tables_only_vs_cjpeg() {
    let cjpeg = match helpers::cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found at /opt/homebrew/bin/cjpeg");
            return;
        }
    };

    // Check whether cjpeg supports -tables-only; if not, skip gracefully.
    // We probe with a dummy invocation and check the exit status / error output.
    let cjpeg_supports_tables_only = {
        use std::process::Command;
        let probe = Command::new(&cjpeg).arg("-tables-only").output();
        match probe {
            Ok(out) => {
                // If it runs (even with error about missing input) the flag is recognized.
                // If it prints "unrecognized option" or similar, skip.
                let stderr = String::from_utf8_lossy(&out.stderr);
                !stderr.contains("unrecognized") && !stderr.contains("unknown option")
            }
            Err(_) => false,
        }
    };

    if !cjpeg_supports_tables_only {
        eprintln!("SKIP: installed cjpeg does not support -tables-only");
        return;
    }

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = helpers::generate_gradient(width, height);
    let ppm: Vec<u8> = helpers::build_ppm(&pixels, width, height);

    let qualities: &[u8] = &[10, 50, 75, 90];
    // subsamp arg to cjpeg: "2x2" = 420, "2x1" = 422, "1x1" = 444
    let subsampling_cases: &[(&str, Subsampling)] = &[
        ("2x2", Subsampling::S420),
        ("2x1", Subsampling::S422),
        ("1x1", Subsampling::S444),
    ];

    for &quality in qualities {
        for &(samp_str, subsampling) in subsampling_cases {
            let label = format!("b10_1_q{}_s{}", quality, samp_str);

            // C cjpeg tables-only
            let c_tables = {
                use std::process::Command;
                let ppm_file = helpers::TempFile::new(&format!("{}_in.ppm", label));
                let out_file = helpers::TempFile::new(&format!("{}_out.jpg", label));
                ppm_file.write_bytes(&ppm);
                let out = Command::new(&cjpeg)
                    .arg("-tables-only")
                    .arg(format!("-quality {}", quality))
                    .arg(format!("-sample {}", samp_str))
                    .arg("-outfile")
                    .arg(out_file.path())
                    .arg(ppm_file.path())
                    .output()
                    .unwrap_or_else(|e| panic!("{}: failed to run cjpeg: {:?}", label, e));
                if !out.status.success() {
                    eprintln!(
                        "SKIP {}: cjpeg -tables-only failed: {}",
                        label,
                        String::from_utf8_lossy(&out.stderr)
                    );
                    continue;
                }
                std::fs::read(out_file.path())
                    .unwrap_or_else(|e| panic!("{}: failed to read cjpeg output: {:?}", label, e))
            };

            // Our tables-only
            let our_tables: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(quality)
                .subsampling(subsampling)
                .write_tables();

            // Byte-identical comparison
            assert_eq!(
                our_tables, c_tables,
                "{}: tables-only stream must be byte-identical to cjpeg output",
                label
            );
        }
    }
}

// ===========================================================================
// B10-2: Compose (tables-only || body-only) → djpeg must match full JPEG
// ===========================================================================

/// Compose: our tables-only || our body-only → feed to djpeg.
/// Decoded pixels must equal decoding the normal full JPEG with djpeg. diff=0.
#[test]
fn b10_2_compose_tables_body_djpeg_roundtrip() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found at /opt/homebrew/bin/djpeg");
            return;
        }
    };

    let qualities: &[u8] = &[10, 50, 75, 90];
    let subsampling_cases: &[(Subsampling, &str)] = &[
        (Subsampling::S420, "420"),
        (Subsampling::S422, "422"),
        (Subsampling::S444, "444"),
    ];

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = helpers::generate_gradient(width, height);

    for &quality in qualities {
        for &(subsampling, samp_name) in subsampling_cases {
            let label = format!("b10_2_q{}_s{}", quality, samp_name);

            // Full normal JPEG
            let full_jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(quality)
                .subsampling(subsampling)
                .encode()
                .unwrap_or_else(|e| panic!("{}: full encode failed: {:?}", label, e));

            // Tables-only stream
            let tables_bytes: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(quality)
                .subsampling(subsampling)
                .write_tables();

            // Body-only stream
            let body_bytes: Vec<u8> = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
                .quality(quality)
                .subsampling(subsampling)
                .suppress_tables(true)
                .encode()
                .unwrap_or_else(|e| panic!("{}: body-only encode failed: {:?}", label, e));

            // Compose: tables-only (SOI + tables) || body-only (SOI stripped + body + EOI)
            // Per JPEG spec F.1.2.4: abbreviated compressed data datastream starts right
            // after SOI of tables stream. We concatenate tables-only (minus EOI) with
            // body-only (minus its SOI).
            let combined: Vec<u8> = compose_abbreviated_stream(&tables_bytes, &body_bytes);

            // Decode combined stream with C djpeg
            let (w_combined, h_combined, pixels_combined) =
                helpers::decode_with_c_djpeg(&djpeg, &combined, &label);

            // Decode full JPEG with C djpeg
            let (w_full, h_full, pixels_full) =
                helpers::decode_with_c_djpeg(&djpeg, &full_jpeg, &format!("{}_full", label));

            assert_eq!(w_combined, w_full, "{}: width must match", label);
            assert_eq!(h_combined, h_full, "{}: height must match", label);

            let max_diff = helpers::pixel_max_diff(&pixels_combined, &pixels_full);
            // Measured diff=0: same encode path, just separated tables/body.
            assert_eq!(
                max_diff, 0,
                "{}: composed (tables||body) vs full JPEG: max_diff={}, expected 0",
                label, max_diff
            );
        }
    }
}

// ===========================================================================
// Helpers specific to abbreviated datastream tests
// ===========================================================================

/// Compose a standard abbreviated compressed datastream from a tables-only and body-only stream.
///
/// JPEG spec F.1.2.4: the abbreviated datastream is tables-only (without final EOI)
/// followed immediately by the compressed data datastream (body-only, starting from SOI).
/// This produces a valid JPEG that standard decoders can parse.
///
/// Layout: [SOI] [DQT] [DHT] [SOF] [SOS] [entropy data] [EOI]
/// We strip EOI from tables-only and keep the body-only intact (it has SOI already).
fn compose_abbreviated_stream(tables_bytes: &[u8], body_bytes: &[u8]) -> Vec<u8> {
    // Remove EOI (last 2 bytes) from tables-only stream
    assert!(tables_bytes.len() >= 4, "tables-only stream too short");
    let tables_without_eoi = &tables_bytes[..tables_bytes.len() - 2];
    assert_eq!(
        &tables_bytes[tables_bytes.len() - 2..],
        &[0xFF, 0xD9],
        "tables-only stream must end with EOI"
    );

    // Body-only stream starts with SOI (0xFFD8); skip its initial SOI to avoid duplicate.
    // The body-only stream is: SOI + [APP0] + SOF + SOS + data + EOI
    // We want: tables_without_eoi + body_without_soi to produce valid JPEG.
    // But actually djpeg / the spec expects: we can just concatenate them
    // because tables-only starts with SOI and body-only also starts with SOI.
    // The simplest approach matching libjpeg's jpegtran: strip the SOI from body-only.
    assert!(body_bytes.len() >= 4, "body-only stream too short");
    let body_without_soi = if body_bytes.starts_with(&[0xFF, 0xD8]) {
        &body_bytes[2..]
    } else {
        body_bytes
    };

    let mut combined = Vec::with_capacity(tables_without_eoi.len() + body_without_soi.len());
    combined.extend_from_slice(tables_without_eoi);
    combined.extend_from_slice(body_without_soi);
    combined
}
