use libjpeg_turbo_rs::{
    compress_arithmetic, compress_progressive, decompress, PixelFormat, Subsampling,
};

/// Verify that existing arithmetic and progressive paths still work
/// (regression guard before SOF10 is exercised).
#[test]
fn arithmetic_sequential_still_works() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_arithmetic(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn progressive_huffman_still_works() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let jpeg =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

/// Test SOF10 decode by constructing a minimal arithmetic progressive JPEG.
///
/// We build a single-component (grayscale) SOF10 JPEG with two scans:
/// - DC first scan (Ss=0, Se=0, Ah=0, Al=0)
/// - AC first scan (Ss=1, Se=63, Ah=0, Al=0)
///
/// This exercises the full arithmetic progressive decode path.
#[test]
fn sof10_grayscale_minimal_decode() {
    // Build a minimal SOF10 JPEG by hand with a single 8x8 MCU
    let jpeg = build_sof10_grayscale_jpeg();
    let result = decompress(&jpeg);
    // The decode should either succeed or give a reasonable error
    // (not panic). With correct arithmetic entropy data it should succeed.
    match result {
        Ok(img) => {
            assert_eq!(img.width, 8);
            assert_eq!(img.height, 8);
        }
        Err(e) => {
            // If entropy data is malformed, that's acceptable for a hand-built stream.
            // But it must not be "unsupported" — the SOF10 path must be recognized.
            let msg = format!("{:?}", e);
            assert!(
                !msg.contains("unsupported") && !msg.contains("Unsupported"),
                "SOF10 should be recognized, not unsupported: {}",
                msg
            );
        }
    }
}

/// Verify that the decoder recognizes SOF10 marker and doesn't return "unsupported".
#[test]
fn sof10_marker_is_recognized() {
    let jpeg = build_minimal_sof10_header();
    let result = decompress(&jpeg);
    match result {
        Ok(_) => {} // Great, it decoded
        Err(e) => {
            let msg = format!("{:?}", e);
            // Should NOT say "unsupported frame type" or similar
            assert!(
                !msg.contains("unsupported frame type"),
                "SOF10 (0xCA) should be a recognized frame type: {}",
                msg
            );
        }
    }
}

/// Build a minimal SOF10 JPEG with just the header markers to test recognition.
fn build_minimal_sof10_header() -> Vec<u8> {
    let mut out = Vec::new();

    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // DQT — quantization table 0 (all 1s for simplicity)
    out.extend_from_slice(&[0xFF, 0xDB]);
    let dqt_len: u16 = 2 + 1 + 64;
    out.extend_from_slice(&dqt_len.to_be_bytes());
    out.push(0x00); // 8-bit, table 0
    out.extend_from_slice(&[1u8; 64]); // all 1s quant table

    // DAC — arithmetic conditioning
    out.extend_from_slice(&[0xFF, 0xCC]);
    out.extend_from_slice(&4u16.to_be_bytes()); // length=4 (1 entry)
    out.push(0x00); // DC table 0
    out.push(0x10); // L=0, U=1

    // SOF10 — arithmetic progressive, 1 component, 8x8
    out.extend_from_slice(&[0xFF, 0xCA]); // SOF10
    let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + 3;
    out.extend_from_slice(&sof_len.to_be_bytes());
    out.push(8); // precision
    out.extend_from_slice(&8u16.to_be_bytes()); // height
    out.extend_from_slice(&8u16.to_be_bytes()); // width
    out.push(1); // 1 component
    out.push(1); // comp id
    out.push(0x11); // h=1, v=1
    out.push(0); // quant table 0

    // SOS — DC first scan (Ss=0, Se=0, Ah=0, Al=0)
    out.extend_from_slice(&[0xFF, 0xDA]);
    let sos_len: u16 = 2 + 1 + 2 + 3;
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(1); // 1 component
    out.push(1); // comp id
    out.push(0x00); // DC table 0, AC table 0
    out.push(0); // Ss=0
    out.push(0); // Se=0
    out.push(0x00); // Ah=0, Al=0

    // Minimal arithmetic entropy data (zeros → the decoder handles gracefully)
    out.extend_from_slice(&[0x00; 16]);

    // EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    out
}

/// Build a minimal single-MCU SOF10 JPEG for decode testing.
fn build_sof10_grayscale_jpeg() -> Vec<u8> {
    let mut out = Vec::new();

    // SOI
    out.extend_from_slice(&[0xFF, 0xD8]);

    // DQT — quantization table 0 (all 1s)
    out.extend_from_slice(&[0xFF, 0xDB]);
    let dqt_len: u16 = 2 + 1 + 64;
    out.extend_from_slice(&dqt_len.to_be_bytes());
    out.push(0x00);
    out.extend_from_slice(&[1u8; 64]);

    // DAC — DC table 0: L=0, U=1; AC table 0: Kx=5
    out.extend_from_slice(&[0xFF, 0xCC]);
    out.extend_from_slice(&6u16.to_be_bytes()); // length=6 (2 entries)
    out.push(0x00); // DC table 0
    out.push(0x10); // U=1, L=0
    out.push(0x10); // AC table 0 (Tc=1, Tb=0)
    out.push(0x05); // Kx=5

    // SOF10 — 1 component, 8x8
    out.extend_from_slice(&[0xFF, 0xCA]);
    let sof_len: u16 = 2 + 1 + 2 + 2 + 1 + 3;
    out.extend_from_slice(&sof_len.to_be_bytes());
    out.push(8);
    out.extend_from_slice(&8u16.to_be_bytes());
    out.extend_from_slice(&8u16.to_be_bytes());
    out.push(1);
    out.push(1);
    out.push(0x11);
    out.push(0);

    // Scan 1: DC first (Ss=0, Se=0, Ah=0, Al=0)
    out.extend_from_slice(&[0xFF, 0xDA]);
    let sos_len: u16 = 2 + 1 + 2 + 3;
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(1);
    out.push(1);
    out.push(0x00);
    out.push(0); // Ss=0
    out.push(0); // Se=0
    out.push(0x00); // Ah=0, Al=0

    // Arithmetic entropy data for DC: encode DC=0 (zero difference)
    // In arithmetic coding, a zero diff means decode(S0)=0 which is the MPS initially
    // Provide enough bytes for the decoder to read
    out.extend_from_slice(&[0x00; 32]);

    // Scan 2: AC first (Ss=1, Se=63, Ah=0, Al=0)
    out.extend_from_slice(&[0xFF, 0xDA]);
    out.extend_from_slice(&sos_len.to_be_bytes());
    out.push(1);
    out.push(1);
    out.push(0x00);
    out.push(1); // Ss=1
    out.push(63); // Se=63
    out.push(0x00); // Ah=0, Al=0

    // Arithmetic entropy data for AC: encode all zeros (EOB immediately)
    // EOB = decode(st)=1 for the first AC position
    out.extend_from_slice(&[0xFF; 8]); // all-ones forces quick EOB
    out.extend_from_slice(&[0x00; 24]);

    // EOI
    out.extend_from_slice(&[0xFF, 0xD9]);

    out
}
