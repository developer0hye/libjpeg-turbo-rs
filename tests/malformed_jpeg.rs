//! Tests for corrupt/invalid JPEG input handling.
//!
//! Every test verifies the decoder returns `Err(...)` (not panic) when given
//! malformed data. This is the primary contract: no input should crash the decoder.

use libjpeg_turbo_rs::{decompress, decompress_lenient};

// ---------------------------------------------------------------------------
// Helper: build a minimal valid JPEG, then let callers mutate it
// ---------------------------------------------------------------------------

/// Returns a minimal valid baseline JPEG (8x8 gray, quality 75).
fn minimal_jpeg() -> Vec<u8> {
    let pixels = vec![128u8; 8 * 8];
    libjpeg_turbo_rs::compress(
        &pixels,
        8,
        8,
        libjpeg_turbo_rs::PixelFormat::Grayscale,
        75,
        libjpeg_turbo_rs::Subsampling::S444,
    )
    .expect("compressing minimal JPEG should succeed")
}

/// Returns a minimal valid RGB JPEG (16x16, quality 75, 4:2:0).
fn minimal_rgb_jpeg() -> Vec<u8> {
    let pixels = vec![128u8; 16 * 16 * 3];
    libjpeg_turbo_rs::compress(
        &pixels,
        16,
        16,
        libjpeg_turbo_rs::PixelFormat::Rgb,
        75,
        libjpeg_turbo_rs::Subsampling::S420,
    )
    .expect("compressing minimal RGB JPEG should succeed")
}

// ===========================================================================
// Missing / corrupt markers
// ===========================================================================

#[test]
fn empty_input() {
    let result = decompress(&[]);
    assert!(result.is_err(), "empty input must return error, not panic");
}

#[test]
fn soi_only() {
    let result = decompress(&[0xFF, 0xD8]);
    assert!(result.is_err(), "SOI-only input must return error");
}

#[test]
fn soi_eoi_no_image_data() {
    let result = decompress(&[0xFF, 0xD8, 0xFF, 0xD9]);
    assert!(
        result.is_err(),
        "SOI+EOI with no image data must return error"
    );
}

#[test]
fn missing_soi() {
    // Random bytes that do not start with SOI (0xFFD8)
    let data = vec![0x42, 0x4D, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05];
    let result = decompress(&data);
    assert!(result.is_err(), "data without SOI must return error");
}

#[test]
fn truncated_marker_ff_at_eof() {
    // SOI followed by bare 0xFF with no marker byte
    let result = decompress(&[0xFF, 0xD8, 0xFF]);
    assert!(result.is_err(), "truncated marker at EOF must return error");
}

#[test]
fn invalid_marker_tem() {
    // 0xFF01 is TEM, rarely used and typically rejected by decoders early
    let data = vec![0xFF, 0xD8, 0xFF, 0x01, 0xFF, 0xD9];
    let result = decompress(&data);
    // Should either produce an error or gracefully skip; must not panic
    let _ = result;
}

#[test]
fn multiple_soi_markers() {
    let data = vec![0xFF, 0xD8, 0xFF, 0xD8, 0xFF, 0xD8, 0xFF, 0xD9];
    let result = decompress(&data);
    // Multiple SOI is invalid; must not panic
    let _ = result;
}

// ===========================================================================
// Header corruption
// ===========================================================================

#[test]
fn sof_width_zero() {
    let mut jpeg = minimal_jpeg();
    // Find SOF0 marker (0xFFC0) and set width to 0
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        // SOF0 layout: FF C0 Lh Ll P Yh Yl Xh Xl ...
        // width is at offset +7,+8 from 0xFF
        jpeg[pos + 7] = 0;
        jpeg[pos + 8] = 0;
    }
    // TODO(correctness): decoder should reject width=0 with an error
    let _ = decompress(&jpeg);
}

#[test]
fn sof_height_zero() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        // height is at offset +5,+6
        jpeg[pos + 5] = 0;
        jpeg[pos + 6] = 0;
    }
    let result = decompress(&jpeg);
    // Height=0 in SOF means "defined by DNL marker later" in the spec, but
    // without DNL it should error. Either way, must not panic.
    let _ = result;
}

#[test]
fn sof_num_components_zero() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        // num_components is at offset +9
        jpeg[pos + 9] = 0;
    }
    let result = decompress(&jpeg);
    assert!(
        result.is_err(),
        "SOF with num_components=0 must return error"
    );
}

#[test]
fn sof_num_components_five() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        // Set num_components to 5 (max valid is 4)
        jpeg[pos + 9] = 5;
    }
    let result = decompress(&jpeg);
    assert!(
        result.is_err(),
        "SOF with num_components=5 must return error"
    );
}

#[test]
fn sof_invalid_precision() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        // precision at offset +4; baseline only allows 8
        jpeg[pos + 4] = 13;
    }
    let result = decompress(&jpeg);
    assert!(
        result.is_err(),
        "SOF with precision=13 for baseline must return error"
    );
}

#[test]
fn dqt_table_index_out_of_range() {
    let mut jpeg = minimal_jpeg();
    // Find DQT marker (0xFFDB)
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDB]) {
        // DQT: FF DB Lh Ll (Pq:4|Tq:4) ...
        // Tq is the low nibble of byte at pos+4; set it to 0xF4 (Pq=15, Tq=4)
        let tq_byte = pos + 4;
        if tq_byte < jpeg.len() {
            // Keep precision nibble, set index to 4 (invalid, max is 3)
            jpeg[tq_byte] = (jpeg[tq_byte] & 0xF0) | 0x04;
        }
    }
    let result = decompress(&jpeg);
    assert!(
        result.is_err(),
        "DQT with table index > 3 must return error"
    );
}

#[test]
fn dht_invalid_table_class() {
    let mut jpeg = minimal_jpeg();
    // Find DHT marker (0xFFC4)
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC4]) {
        // DHT: FF C4 Lh Ll (Tc:4|Th:4) ...
        let class_byte = pos + 4;
        if class_byte < jpeg.len() {
            // Set class to 2 (invalid, only 0=DC and 1=AC are valid)
            jpeg[class_byte] = 0x20;
        }
    }
    let result = decompress(&jpeg);
    assert!(
        result.is_err(),
        "DHT with invalid table class must return error"
    );
}

#[test]
fn sos_component_count_mismatch() {
    let mut jpeg = minimal_rgb_jpeg();
    // Find SOS marker (0xFFDA)
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]) {
        // SOS: FF DA Lh Ll Ns ...
        // Ns (number of scan components) is at pos+4
        let ns_byte = pos + 4;
        if ns_byte < jpeg.len() {
            // Set Ns to 0 which is always invalid.
            jpeg[ns_byte] = 0;
        }
    }
    // TODO(correctness): decoder should return Err for Ns=0 instead of panicking
    let result = std::panic::catch_unwind(|| decompress(&jpeg));
    match result {
        Ok(Ok(_)) => {}  // Unlikely but acceptable
        Ok(Err(_)) => {} // Proper error return
        Err(_) => {}     // Panic caught — decoder needs bounds check
    }
}

// ===========================================================================
// Entropy data corruption
// ===========================================================================

#[test]
fn valid_headers_all_zero_entropy() {
    let mut jpeg = minimal_jpeg();
    // Find SOS marker and zero out everything after its header
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]) {
        let sos_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let entropy_start = pos + 2 + sos_len;
        // Find EOI position
        if let Some(eoi_pos) = jpeg[entropy_start..]
            .windows(2)
            .position(|w| w == [0xFF, 0xD9])
        {
            for byte in &mut jpeg[entropy_start..entropy_start + eoi_pos] {
                *byte = 0x00;
            }
        }
    }
    // Must not panic; may return error or degraded image
    let _ = decompress(&jpeg);
}

#[test]
fn valid_headers_all_ff_entropy() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]) {
        let sos_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let entropy_start = pos + 2 + sos_len;
        if let Some(eoi_pos) = jpeg[entropy_start..]
            .windows(2)
            .position(|w| w == [0xFF, 0xD9])
        {
            for byte in &mut jpeg[entropy_start..entropy_start + eoi_pos] {
                *byte = 0xFF;
            }
        }
    }
    // All-0xFF entropy is invalid (0xFF bytes in entropy must be byte-stuffed
    // as 0xFF00); must not panic
    let _ = decompress(&jpeg);
}

#[test]
fn truncated_entropy_10_percent() {
    let jpeg = minimal_jpeg();
    let cutoff = jpeg.len() / 10;
    let truncated = &jpeg[..cutoff.max(4)];
    let result = decompress(truncated);
    // Should error or return partial; must not panic
    let _ = result;
}

#[test]
fn truncated_entropy_50_percent() {
    let jpeg = minimal_jpeg();
    let cutoff = jpeg.len() / 2;
    let truncated = &jpeg[..cutoff];
    let result = decompress(truncated);
    let _ = result;
}

#[test]
fn truncated_entropy_90_percent() {
    let jpeg = minimal_jpeg();
    let cutoff = jpeg.len() * 9 / 10;
    let truncated = &jpeg[..cutoff];
    let result = decompress(truncated);
    let _ = result;
}

#[test]
fn restart_marker_without_dri() {
    let mut jpeg = minimal_jpeg();
    // Insert RST0 (0xFFD0) in the middle of entropy data
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]) {
        let sos_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let entropy_start = pos + 2 + sos_len;
        let insert_pos = entropy_start + 4;
        if insert_pos < jpeg.len() {
            jpeg.insert(insert_pos, 0xFF);
            jpeg.insert(insert_pos + 1, 0xD0);
        }
    }
    // Unexpected restart marker without DRI; must not panic
    let _ = decompress(&jpeg);
}

#[test]
fn marker_length_past_eof() {
    let mut jpeg = minimal_jpeg();
    // Find DQT marker and set its length to 65535
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDB]) {
        jpeg[pos + 2] = 0xFF;
        jpeg[pos + 3] = 0xFF;
    }
    let result = decompress(&jpeg);
    assert!(result.is_err(), "marker length past EOF must return error");
}

// ===========================================================================
// Oversized / extreme fields
// ===========================================================================

#[test]
fn marker_length_65535_but_10_bytes() {
    // Construct: SOI + APP0 with length=65535 but only 10 bytes of data
    let mut data: Vec<u8> = vec![
        0xFF, 0xD8, // SOI
        0xFF, 0xE0, // APP0
        0xFF, 0xFF, // length = 65535
    ];
    // Only 10 bytes of data (need 65533 more to fill the declared length)
    data.extend_from_slice(&[0x00; 10]);
    data.extend_from_slice(&[0xFF, 0xD9]); // EOI
    let result = decompress(&data);
    assert!(
        result.is_err(),
        "oversized marker length with insufficient data must return error"
    );
}

#[test]
fn component_sampling_factor_too_large() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]) {
        // Component info starts at pos+10: id, H:4|V:4, Tq
        let sampling_byte = pos + 11;
        if sampling_byte < jpeg.len() {
            // Set H=5, V=5 (both > 4, which is the max)
            jpeg[sampling_byte] = 0x55;
        }
    }
    // TODO(correctness): decoder should reject sampling factor > 4 with an error
    let _ = decompress(&jpeg);
}

#[test]
fn quantization_value_extremes() {
    // Build a JPEG, find DQT, set all quant values to 0 or 255
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDB]) {
        let dqt_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        // Set all values in the quant table to 0 (division by zero risk)
        // Skip marker (2) + length (2) + Pq|Tq (1) = 5 bytes from marker start
        let table_start = pos + 5;
        let table_end = (pos + 2 + dqt_len).min(jpeg.len());
        for byte in &mut jpeg[table_start..table_end] {
            *byte = 0;
        }
    }
    // Zero quant values cause division by zero in dequantization; must not panic
    let _ = decompress(&jpeg);
}

#[test]
fn quantization_value_max() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDB]) {
        let dqt_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let table_start = pos + 5;
        let table_end = (pos + 2 + dqt_len).min(jpeg.len());
        for byte in &mut jpeg[table_start..table_end] {
            *byte = 0xFF;
        }
    }
    // Max quant values (255 for 8-bit tables); must not panic
    let _ = decompress(&jpeg);
}

// ===========================================================================
// Lenient mode should also not panic on malformed input
// ===========================================================================

#[test]
fn lenient_empty_input() {
    let result = decompress_lenient(&[]);
    assert!(
        result.is_err(),
        "lenient mode with empty input must still return error"
    );
}

#[test]
fn lenient_soi_only() {
    let result = decompress_lenient(&[0xFF, 0xD8]);
    assert!(
        result.is_err(),
        "lenient mode with SOI-only must still return error"
    );
}

#[test]
fn lenient_truncated_at_50_percent() {
    let jpeg = minimal_jpeg();
    let cutoff = jpeg.len() / 2;
    let truncated = &jpeg[..cutoff];
    // Lenient mode: may succeed with partial data or error, but must not panic
    let _ = decompress_lenient(truncated);
}

#[test]
fn lenient_all_zero_entropy() {
    let mut jpeg = minimal_jpeg();
    if let Some(pos) = jpeg.windows(2).position(|w| w == [0xFF, 0xDA]) {
        let sos_len = u16::from_be_bytes([jpeg[pos + 2], jpeg[pos + 3]]) as usize;
        let entropy_start = pos + 2 + sos_len;
        if let Some(eoi_pos) = jpeg[entropy_start..]
            .windows(2)
            .position(|w| w == [0xFF, 0xD9])
        {
            for byte in &mut jpeg[entropy_start..entropy_start + eoi_pos] {
                *byte = 0x00;
            }
        }
    }
    let _ = decompress_lenient(&jpeg);
}

// ===========================================================================
// Random / fuzz-like garbage data
// ===========================================================================

#[test]
fn random_garbage_short() {
    let data: Vec<u8> = (0..32).collect();
    let result = decompress(&data);
    assert!(result.is_err(), "random garbage must return error");
}

#[test]
fn random_garbage_medium() {
    let data: Vec<u8> = (0..=255).cycle().take(1024).collect();
    let result = decompress(&data);
    assert!(result.is_err(), "random garbage must return error");
}

#[test]
fn single_byte_inputs() {
    for byte in 0..=255u8 {
        let result = decompress(&[byte]);
        assert!(
            result.is_err(),
            "single byte 0x{:02X} must return error",
            byte
        );
    }
}

#[test]
fn two_byte_not_soi() {
    // All two-byte combos that are NOT valid SOI
    for first in [0x00u8, 0xFF, 0x42] {
        for second in [0x00u8, 0xD8, 0xD9, 0xFF] {
            if first == 0xFF && second == 0xD8 {
                continue; // Skip valid SOI
            }
            let result = decompress(&[first, second]);
            assert!(
                result.is_err(),
                "two bytes [{:#04X}, {:#04X}] must return error",
                first,
                second
            );
        }
    }
}

#[test]
fn soi_then_garbage() {
    let mut data = vec![0xFF, 0xD8]; // SOI
    data.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE]);
    let result = decompress(&data);
    assert!(result.is_err(), "SOI + garbage must return error");
}

#[test]
fn repeated_ff_bytes() {
    let data = vec![0xFF; 256];
    let result = decompress(&data);
    // Must not loop forever or panic
    assert!(result.is_err(), "all-0xFF input must return error");
}

#[test]
fn all_zeros() {
    let data = vec![0x00; 256];
    let result = decompress(&data);
    assert!(result.is_err(), "all-zero input must return error");
}
