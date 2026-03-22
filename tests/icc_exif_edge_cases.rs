use libjpeg_turbo_rs::{compress_with_metadata, decompress, Encoder, PixelFormat, Subsampling};

// ============================================================
// Helper: build minimal TIFF/EXIF data with a given orientation value
// ============================================================

fn build_tiff_with_orientation(orientation: u16) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();

    // Little-endian byte order
    data.extend_from_slice(b"II");
    // TIFF magic (42)
    data.extend_from_slice(&42u16.to_le_bytes());
    // IFD0 offset = 8 (immediately after header)
    data.extend_from_slice(&8u32.to_le_bytes());
    // IFD0: 1 entry
    data.extend_from_slice(&1u16.to_le_bytes());
    // Entry: tag=0x0112 (Orientation), type=3 (SHORT), count=1, value=orientation
    data.extend_from_slice(&0x0112u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes()); // SHORT
    data.extend_from_slice(&1u32.to_le_bytes()); // count
    data.extend_from_slice(&orientation.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes()); // padding
                                                 // Next IFD offset = 0 (end)
    data.extend_from_slice(&0u32.to_le_bytes());

    data
}

/// Build minimal valid TIFF header (8 bytes): II + magic 42 + IFD offset 8, then 0 entries.
fn build_minimal_tiff() -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    // IFD0: 0 entries
    data.extend_from_slice(&0u16.to_le_bytes());
    // Next IFD offset = 0
    data.extend_from_slice(&0u32.to_le_bytes());
    data
}

/// Small helper to create a test image's pixel data.
fn test_pixels(width: usize, height: usize) -> Vec<u8> {
    vec![128u8; width * height * 3]
}

// ============================================================
// ICC profile edge cases
// ============================================================

#[test]
fn icc_large_profile_over_65kb_multi_chunk_roundtrip() {
    // 100KB ICC profile requires multiple APP2 chunks (each chunk max 65519 bytes)
    let large_icc: Vec<u8> = (0..100_000).map(|i| (i % 256) as u8).collect();
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(16, 16),
        16,
        16,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&large_icc),
        None,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.icc_profile(),
        Some(large_icc.as_slice()),
        "100KB ICC profile should survive multi-chunk roundtrip"
    );
}

#[test]
fn icc_very_small_profile_1_byte_roundtrip() {
    let tiny_icc: Vec<u8> = vec![0xAA];
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&tiny_icc),
        None,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.icc_profile(),
        Some(tiny_icc.as_slice()),
        "1-byte ICC profile should roundtrip"
    );
}

#[test]
fn icc_empty_profile_0_bytes_handled_gracefully() {
    // An empty ICC profile should either be ignored or produce no icc_profile on decode
    let empty_icc: Vec<u8> = vec![];
    let result = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&empty_icc),
        None,
    );
    // Either succeeds with no ICC on decode, or returns an error -- either is acceptable
    match result {
        Ok(jpeg) => {
            let img = decompress(&jpeg).unwrap();
            // Empty ICC might be stored as None or as empty slice
            if let Some(profile) = img.icc_profile() {
                assert!(
                    profile.is_empty(),
                    "empty ICC input should yield empty or None on decode"
                );
            }
        }
        Err(_) => {
            // Rejecting empty ICC is also acceptable behavior
        }
    }
}

#[test]
fn icc_profile_exactly_at_chunk_boundary() {
    // Each APP2 ICC chunk payload = 65535 - 2 (length field) - 14 (ICC_PROFILE header) = 65519 bytes
    // A profile of exactly 65519 bytes fits in one chunk with no remainder
    let boundary_icc: Vec<u8> = vec![0xBB; 65519];
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&boundary_icc),
        None,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.icc_profile(),
        Some(boundary_icc.as_slice()),
        "ICC profile at exact chunk boundary should roundtrip"
    );
}

#[test]
fn icc_two_sequential_encodes_independent() {
    // Two separate compress_with_metadata calls with different ICC profiles
    // should produce independent results
    let icc_a: Vec<u8> = vec![0x11; 500];
    let icc_b: Vec<u8> = vec![0x22; 300];

    let jpeg_a: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&icc_a),
        None,
    )
    .unwrap();
    let jpeg_b: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&icc_b),
        None,
    )
    .unwrap();

    let img_a = decompress(&jpeg_a).unwrap();
    let img_b = decompress(&jpeg_b).unwrap();
    assert_eq!(img_a.icc_profile(), Some(icc_a.as_slice()));
    assert_eq!(img_b.icc_profile(), Some(icc_b.as_slice()));
    assert_ne!(
        img_a.icc_profile(),
        img_b.icc_profile(),
        "two different ICC profiles should produce different decode results"
    );
}

#[test]
fn icc_malformed_chunk_wrong_sequence_number_no_panic() {
    // Construct a JPEG with a hand-crafted malformed ICC APP2 marker
    // that has an incorrect sequence number. Decoding should not panic.
    let base_jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .unwrap();

    // Inject a malformed APP2 ICC_PROFILE marker after SOI + APP0
    let insert_pos: usize = if base_jpeg.len() >= 4 && base_jpeg[2] == 0xFF && base_jpeg[3] == 0xE0
    {
        let app0_len: usize = u16::from_be_bytes([base_jpeg[4], base_jpeg[5]]) as usize;
        2 + 2 + app0_len
    } else {
        2
    };

    let mut malformed: Vec<u8> = Vec::new();
    malformed.extend_from_slice(&base_jpeg[..insert_pos]);

    // APP2 marker
    malformed.push(0xFF);
    malformed.push(0xE2);
    // ICC_PROFILE header: "ICC_PROFILE\0" + seq_num + num_chunks
    let icc_header: &[u8] = b"ICC_PROFILE\0";
    let payload: &[u8] = &[0xAA; 10];
    let seg_len: u16 = (2 + icc_header.len() + 2 + payload.len()) as u16;
    malformed.extend_from_slice(&seg_len.to_be_bytes());
    malformed.extend_from_slice(icc_header);
    malformed.push(5); // sequence number 5 (wrong -- should be 1)
    malformed.push(1); // total chunks = 1
    malformed.extend_from_slice(payload);

    malformed.extend_from_slice(&base_jpeg[insert_pos..]);

    // Should not panic. May return error or succeed with no/partial ICC.
    let result = decompress(&malformed);
    match result {
        Ok(img) => {
            // If it succeeds, ICC might be None or partial -- that is acceptable
            let _ = img.icc_profile();
        }
        Err(_) => {
            // Error is also acceptable for malformed data
        }
    }
}

#[test]
fn icc_missing_chunk_in_sequence_no_panic() {
    // Construct JPEG with ICC chunks 1 and 3 (of 3) but not chunk 2
    let base_jpeg: Vec<u8> = libjpeg_turbo_rs::compress(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let insert_pos: usize = if base_jpeg.len() >= 4 && base_jpeg[2] == 0xFF && base_jpeg[3] == 0xE0
    {
        let app0_len: usize = u16::from_be_bytes([base_jpeg[4], base_jpeg[5]]) as usize;
        2 + 2 + app0_len
    } else {
        2
    };

    let mut broken: Vec<u8> = Vec::new();
    broken.extend_from_slice(&base_jpeg[..insert_pos]);

    // Insert chunk 1 of 3
    inject_icc_chunk(&mut broken, 1, 3, &[0x11; 10]);
    // Skip chunk 2 of 3
    // Insert chunk 3 of 3
    inject_icc_chunk(&mut broken, 3, 3, &[0x33; 10]);

    broken.extend_from_slice(&base_jpeg[insert_pos..]);

    // Should not panic. May return error or partial ICC.
    let result = decompress(&broken);
    match result {
        Ok(img) => {
            // ICC reassembly may fail gracefully (None)
            let _ = img.icc_profile();
        }
        Err(_) => {
            // Error is also acceptable
        }
    }
}

/// Helper: inject a single ICC APP2 chunk into a buffer.
fn inject_icc_chunk(buf: &mut Vec<u8>, seq: u8, total: u8, payload: &[u8]) {
    let icc_header: &[u8] = b"ICC_PROFILE\0";
    let seg_len: u16 = (2 + icc_header.len() + 2 + payload.len()) as u16;
    buf.push(0xFF);
    buf.push(0xE2);
    buf.extend_from_slice(&seg_len.to_be_bytes());
    buf.extend_from_slice(icc_header);
    buf.push(seq);
    buf.push(total);
    buf.extend_from_slice(payload);
}

// ============================================================
// EXIF edge cases
// ============================================================

#[test]
fn exif_large_data_over_64kb() {
    // EXIF data larger than 64KB -- a single APP1 marker segment cannot exceed ~65533 bytes.
    // The encoder may produce a malformed segment or the decoder may reject it.
    // The key requirement: no panic.
    let large_exif: Vec<u8> = {
        let mut data: Vec<u8> = build_minimal_tiff();
        data.resize(70_000, 0x00);
        data
    };
    let encode_result = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        Some(&large_exif),
    );
    match encode_result {
        Ok(jpeg) => {
            // Encoding succeeded; decoding may fail because the APP1 segment overflows.
            // Either outcome is acceptable as long as there is no panic.
            match decompress(&jpeg) {
                Ok(img) => {
                    if let Some(exif) = img.exif_data() {
                        assert!(
                            !exif.is_empty(),
                            "large EXIF should produce non-empty data on decode"
                        );
                    }
                }
                Err(_) => {
                    // Decode error is acceptable for oversized EXIF
                }
            }
        }
        Err(_) => {
            // Rejecting oversized EXIF at encode time is also acceptable
        }
    }
}

#[test]
fn exif_near_limit_fits_in_app1() {
    // EXIF data just under the APP1 limit (~65527 bytes after "Exif\0\0" header)
    // APP1 segment: 2 (marker) + 2 (length) + 6 ("Exif\0\0") + data
    // Max segment = 65535, length field = 2 + 6 + data_len, so max data_len = 65535 - 2 - 6 = 65527
    let near_limit_exif: Vec<u8> = {
        let mut data: Vec<u8> = build_minimal_tiff();
        data.resize(60_000, 0x00); // safely under the limit
        data
    };
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        Some(&near_limit_exif),
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.exif_data(),
        Some(near_limit_exif.as_slice()),
        "EXIF near the APP1 limit should roundtrip"
    );
}

#[test]
fn exif_minimal_tiff_header_roundtrip() {
    // Minimal EXIF: just a valid TIFF header with 0 IFD entries
    let minimal_exif: Vec<u8> = build_minimal_tiff();
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        Some(&minimal_exif),
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.exif_data(),
        Some(minimal_exif.as_slice()),
        "minimal TIFF header should roundtrip"
    );
}

#[test]
fn exif_empty_0_bytes_handled_gracefully() {
    let empty_exif: Vec<u8> = vec![];
    let result = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        Some(&empty_exif),
    );
    match result {
        Ok(jpeg) => {
            let img = decompress(&jpeg).unwrap();
            if let Some(exif) = img.exif_data() {
                assert!(
                    exif.is_empty(),
                    "empty EXIF input should yield empty or None on decode"
                );
            }
        }
        Err(_) => {
            // Rejecting empty EXIF is also acceptable
        }
    }
}

#[test]
fn exif_all_8_orientation_values_parse_correctly() {
    // Embed EXIF with each valid orientation (1-8) and verify it parses back
    for orientation in 1u16..=8 {
        let exif_data: Vec<u8> = build_tiff_with_orientation(orientation);
        let jpeg: Vec<u8> = compress_with_metadata(
            &test_pixels(8, 8),
            8,
            8,
            PixelFormat::Rgb,
            75,
            Subsampling::S444,
            None,
            Some(&exif_data),
        )
        .unwrap();
        let img = decompress(&jpeg).unwrap();
        assert_eq!(
            img.exif_orientation(),
            Some(orientation as u8),
            "orientation {} should parse correctly after roundtrip",
            orientation
        );
    }
}

#[test]
fn exif_orientation_0_invalid_handled_gracefully() {
    let exif_data: Vec<u8> = build_tiff_with_orientation(0);
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        Some(&exif_data),
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    // Orientation 0 is invalid; parse_orientation should return None
    assert_eq!(
        img.exif_orientation(),
        None,
        "orientation=0 is invalid, should return None"
    );
}

#[test]
fn exif_orientation_9_invalid_handled_gracefully() {
    let exif_data: Vec<u8> = build_tiff_with_orientation(9);
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        Some(&exif_data),
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.exif_orientation(),
        None,
        "orientation=9 is invalid, should return None"
    );
}

#[test]
fn jpeg_with_both_icc_and_exif_preserved_on_roundtrip() {
    let icc: Vec<u8> = vec![0x42; 1024];
    let exif: Vec<u8> = build_tiff_with_orientation(6);
    let jpeg: Vec<u8> = compress_with_metadata(
        &test_pixels(16, 16),
        16,
        16,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&icc),
        Some(&exif),
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(icc.as_slice()));
    assert_eq!(img.exif_data(), Some(exif.as_slice()));
    assert_eq!(img.exif_orientation(), Some(6));
}

// ============================================================
// COM marker edge cases
// ============================================================

#[test]
fn comment_very_long_over_65kb() {
    // A comment longer than a single COM marker segment can hold (>65533 bytes).
    // COM segments have the same 64KB limit as other JPEG markers.
    // The encoder may produce a malformed segment or the decoder may reject it.
    // The key requirement: no panic.
    let long_comment: String = "A".repeat(70_000);
    let encode_result = Encoder::new(&test_pixels(8, 8), 8, 8, PixelFormat::Rgb)
        .quality(75)
        .comment(&long_comment)
        .encode();
    match encode_result {
        Ok(jpeg) => {
            // Encoding succeeded; decoding may fail for oversized COM segment
            match decompress(&jpeg) {
                Ok(img) => {
                    if let Some(comment) = &img.comment {
                        assert!(
                            !comment.is_empty(),
                            "long comment should produce non-empty result"
                        );
                    }
                }
                Err(_) => {
                    // Decode error is acceptable for oversized comment
                }
            }
        }
        Err(_) => {
            // Rejecting oversized comment at encode time is also acceptable
        }
    }
}

#[test]
fn comment_near_limit_fits_in_segment() {
    // Comment just under the COM segment limit
    // COM: 2 (marker) + 2 (length) + data; max data = 65535 - 2 = 65533
    let near_limit_comment: String = "B".repeat(60_000); // safely under limit
    let jpeg: Vec<u8> = Encoder::new(&test_pixels(8, 8), 8, 8, PixelFormat::Rgb)
        .quality(75)
        .comment(&near_limit_comment)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.comment.as_deref(),
        Some(near_limit_comment.as_str()),
        "comment near segment limit should roundtrip"
    );
}

#[test]
fn comment_empty_string_roundtrip() {
    let jpeg: Vec<u8> = Encoder::new(&test_pixels(8, 8), 8, 8, PixelFormat::Rgb)
        .quality(75)
        .comment("")
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    // Empty comment might be stored as Some("") or None
    match &img.comment {
        Some(c) => assert!(
            c.is_empty(),
            "empty comment input should yield empty string"
        ),
        None => {} // Also acceptable
    }
}

#[test]
fn comment_with_utf8_characters_roundtrip() {
    let utf8_comment: &str = "Hello, world! Orientation: 90\u{00b0}. \u{1F4F7}";
    let jpeg: Vec<u8> = Encoder::new(&test_pixels(8, 8), 8, 8, PixelFormat::Rgb)
        .quality(75)
        .comment(utf8_comment)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.comment.as_deref(),
        Some(utf8_comment),
        "UTF-8 comment should roundtrip"
    );
}

#[test]
fn comment_with_null_bytes_roundtrip() {
    // COM marker data is raw bytes; null bytes should be preserved if stored as binary
    let comment_with_nulls: &str = "before\0after";
    let jpeg: Vec<u8> = Encoder::new(&test_pixels(8, 8), 8, 8, PixelFormat::Rgb)
        .quality(75)
        .comment(comment_with_nulls)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    // The comment may be truncated at the null byte if treated as C-string,
    // or it may preserve the null. Either behavior is acceptable.
    assert!(
        img.comment.is_some(),
        "comment with null bytes should produce some result"
    );
    let decoded: &str = img.comment.as_deref().unwrap();
    assert!(
        decoded.starts_with("before"),
        "comment should at least start with the prefix before null"
    );
}

#[test]
fn multiple_comments_via_saved_markers() {
    // Use saved_marker API to inject multiple COM markers
    let jpeg: Vec<u8> = Encoder::new(&test_pixels(8, 8), 8, 8, PixelFormat::Rgb)
        .quality(75)
        .saved_marker(libjpeg_turbo_rs::SavedMarker {
            code: 0xFE,
            data: b"comment-one".to_vec(),
        })
        .saved_marker(libjpeg_turbo_rs::SavedMarker {
            code: 0xFE,
            data: b"comment-two".to_vec(),
        })
        .encode()
        .unwrap();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let img = decoder.decode_image().unwrap();

    let com_markers: Vec<&libjpeg_turbo_rs::SavedMarker> = img
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xFE)
        .collect();
    assert!(
        com_markers.len() >= 2,
        "expected at least 2 COM markers, got {}",
        com_markers.len()
    );
    // Verify order and content
    let data_list: Vec<&[u8]> = com_markers.iter().map(|m| m.data.as_slice()).collect();
    assert!(
        data_list.contains(&b"comment-one".as_slice()),
        "first comment should be preserved"
    );
    assert!(
        data_list.contains(&b"comment-two".as_slice()),
        "second comment should be preserved"
    );
}
