use libjpeg_turbo_rs::{
    compress_progressive, decompress, Encoder, PixelFormat, ProgressiveDecoder, ScanScript,
    Subsampling,
};

/// Helper: create synthetic pixel data with some spatial variation.
fn make_pixels(width: usize, height: usize, bpp: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * bpp);
    for y in 0..height {
        for x in 0..width {
            for c in 0..bpp {
                pixels.push(((x * 7 + y * 3 + c * 50) % 256) as u8);
            }
        }
    }
    pixels
}

/// Helper: create grayscale pixel data.
fn make_gray_pixels(width: usize, height: usize) -> Vec<u8> {
    (0..width * height).map(|i| (i % 256) as u8).collect()
}

/// Sum of absolute pixel differences between two buffers.
fn pixel_diff(a: &[u8], b: &[u8]) -> u64 {
    let len: usize = a.len().min(b.len());
    let mut total: u64 = 0;
    for i in 0..len {
        total += (a[i] as i64 - b[i] as i64).unsigned_abs();
    }
    total
}

// ============================================================
// Scan order variations
// ============================================================

#[test]
fn progressive_dc_only_no_ac_refinement_decodes() {
    // A progressive scan script with only DC scans (no AC data at all).
    // The result should be a very blocky but valid image.
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let script: Vec<ScanScript> = vec![
        // Single interleaved DC scan, no successive approximation
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        },
    ];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();

    // Should contain SOF2 marker
    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "DC-only progressive should still have SOF2");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert!(!img.data.is_empty());
}

#[test]
fn progressive_single_all_component_scan_decodes() {
    // Degenerate progressive: a single scan that covers DC+AC for all components.
    // This is technically a valid progressive JPEG with just one scan.
    let pixels: Vec<u8> = make_pixels(16, 16, 3);

    // For progressive, each scan can only contain either DC (ss=0,se=0) or AC (ss>0).
    // Also, interleaved scans can only do DC. So we need separate scans:
    // DC interleaved + AC per-component = "degenerate" progressive but still multi-scan.
    let script: Vec<ScanScript> = vec![
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
    ];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
}

#[test]
fn progressive_custom_unusual_spectral_ordering_decodes() {
    // Custom script: send low-frequency AC first, then high-frequency AC
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let script: Vec<ScanScript> = vec![
        // DC for all components
        ScanScript {
            components: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 0,
        },
        // Y: low AC (1-5)
        ScanScript {
            components: vec![0],
            ss: 1,
            se: 5,
            ah: 0,
            al: 0,
        },
        // Y: high AC (6-63)
        ScanScript {
            components: vec![0],
            ss: 6,
            se: 63,
            ah: 0,
            al: 0,
        },
        // Cb: full AC
        ScanScript {
            components: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
        // Cr: full AC
        ScanScript {
            components: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 0,
        },
    ];
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .scan_script(script)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert!(!img.data.is_empty());
}

// ============================================================
// Incomplete progressive decoding
// ============================================================

#[test]
fn progressive_decoder_first_scan_only_produces_valid_image() {
    let jpeg_data: Vec<u8> = compress_progressive(
        &make_pixels(32, 32, 3),
        32,
        32,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .unwrap();

    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    assert!(decoder.num_scans() > 1, "expected multiple scans");

    // Consume only the first scan
    let consumed: bool = decoder.consume_input().unwrap();
    assert!(consumed, "should consume first scan");
    assert_eq!(decoder.scans_consumed(), 1);
    assert!(
        !decoder.input_complete(),
        "should not be complete after 1 scan"
    );

    // Output after first scan should be valid (low quality) image
    let early_image = decoder.output().unwrap();
    assert_eq!(early_image.width, 32);
    assert_eq!(early_image.height, 32);
    assert!(
        !early_image.data.is_empty(),
        "first-scan output should have pixel data"
    );
}

#[test]
fn progressive_decoder_half_scans_intermediate_quality() {
    let jpeg_data: Vec<u8> = compress_progressive(
        &make_pixels(32, 32, 3),
        32,
        32,
        PixelFormat::Rgb,
        75,
        Subsampling::S420,
    )
    .unwrap();

    let mut decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg_data).unwrap();
    let total_scans: usize = decoder.num_scans();
    let half_scans: usize = total_scans / 2;

    // Consume first scan for early reference
    decoder.consume_input().unwrap();
    let early_image = decoder.output().unwrap();

    // Consume up to half scans
    for _ in 1..half_scans {
        decoder.consume_input().unwrap();
    }
    let half_image = decoder.output().unwrap();

    // Consume remaining scans
    while decoder.consume_input().unwrap() {}
    let final_image = decoder.output().unwrap();

    // Compare against full decompress reference
    let reference = decompress(&jpeg_data).unwrap();

    let early_diff: u64 = pixel_diff(&early_image.data, &reference.data);
    let half_diff: u64 = pixel_diff(&half_image.data, &reference.data);
    let final_diff: u64 = pixel_diff(&final_image.data, &reference.data);

    // Quality should improve monotonically: early >= half >= final
    assert!(
        early_diff >= half_diff || half_diff == 0,
        "early diff ({}) should be >= half diff ({})",
        early_diff,
        half_diff
    );
    assert!(
        half_diff >= final_diff || final_diff == 0,
        "half diff ({}) should be >= final diff ({})",
        half_diff,
        final_diff
    );
}

// ============================================================
// Progressive with various subsampling
// ============================================================

#[test]
fn progressive_420_roundtrip() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S420).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.pixel_format, PixelFormat::Rgb);

    // Also verify via ProgressiveDecoder
    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

#[test]
fn progressive_444_roundtrip() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S444).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

#[test]
fn progressive_grayscale_roundtrip() {
    let pixels: Vec<u8> = make_gray_pixels(32, 32);
    let jpeg: Vec<u8> = compress_progressive(
        &pixels,
        32,
        32,
        PixelFormat::Grayscale,
        75,
        Subsampling::S444,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.pixel_format, PixelFormat::Grayscale);

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

#[test]
fn progressive_422_roundtrip() {
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let jpeg: Vec<u8> =
        compress_progressive(&pixels, 32, 32, PixelFormat::Rgb, 75, Subsampling::S422).unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let progressive_img = decoder.finish().unwrap();
    assert_eq!(progressive_img.data, img.data);
}

// ============================================================
// Progressive + metadata
// ============================================================

#[test]
fn progressive_with_icc_profile() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let icc: Vec<u8> = vec![0x42; 500];

    // Use Encoder to create progressive JPEG with ICC
    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .icc_profile(&icc)
        .encode()
        .unwrap();

    // Verify it is progressive (SOF2)
    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "should be progressive JPEG");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, 16);
    assert_eq!(img.height, 16);
    assert_eq!(
        img.icc_profile(),
        Some(icc.as_slice()),
        "ICC profile should survive progressive encoding"
    );
}

#[test]
fn progressive_with_exif() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    // Valid EXIF with orientation = 3 (rotated 180)
    let exif: Vec<u8> = build_tiff_with_orientation(3);

    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .exif_data(&exif)
        .encode()
        .unwrap();

    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "should be progressive JPEG");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.exif_data(),
        Some(exif.as_slice()),
        "EXIF data should survive progressive encoding"
    );
    assert_eq!(img.exif_orientation(), Some(3));
}

#[test]
fn progressive_with_comment() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let comment: &str = "Progressive test comment";

    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .comment(comment)
        .encode()
        .unwrap();

    let has_sof2: bool = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xC2);
    assert!(has_sof2, "should be progressive JPEG");

    let img = decompress(&jpeg).unwrap();
    assert_eq!(
        img.comment.as_deref(),
        Some(comment),
        "comment should survive progressive encoding"
    );
}

#[test]
fn progressive_with_all_metadata_combined() {
    let pixels: Vec<u8> = make_pixels(16, 16, 3);
    let icc: Vec<u8> = vec![0xCC; 2000];
    let exif: Vec<u8> = build_tiff_with_orientation(8);
    let comment: &str = "All metadata progressive test";

    let jpeg: Vec<u8> = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S420)
        .progressive(true)
        .icc_profile(&icc)
        .exif_data(&exif)
        .comment(comment)
        .encode()
        .unwrap();

    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(icc.as_slice()));
    assert_eq!(img.exif_data(), Some(exif.as_slice()));
    assert_eq!(img.exif_orientation(), Some(8));
    assert_eq!(img.comment.as_deref(), Some(comment));
}

#[test]
fn progressive_decoder_preserves_metadata() {
    // Verify that ProgressiveDecoder.output() also includes ICC and EXIF
    let pixels: Vec<u8> = make_pixels(32, 32, 3);
    let icc: Vec<u8> = vec![0xDD; 300];
    let exif: Vec<u8> = build_tiff_with_orientation(6);

    let jpeg: Vec<u8> = Encoder::new(&pixels, 32, 32, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .progressive(true)
        .icc_profile(&icc)
        .exif_data(&exif)
        .comment("progressive decoder metadata test")
        .encode()
        .unwrap();

    let decoder: ProgressiveDecoder = ProgressiveDecoder::new(&jpeg).unwrap();
    let img = decoder.finish().unwrap();
    assert_eq!(
        img.icc_profile,
        Some(icc.clone()),
        "ProgressiveDecoder should preserve ICC"
    );
    assert_eq!(
        img.exif_data,
        Some(exif.clone()),
        "ProgressiveDecoder should preserve EXIF"
    );
    assert_eq!(
        img.comment.as_deref(),
        Some("progressive decoder metadata test"),
        "ProgressiveDecoder should preserve comment"
    );
}

// ============================================================
// Helper: build minimal TIFF with orientation (duplicated for this test file)
// ============================================================

fn build_tiff_with_orientation(orientation: u16) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&0x0112u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&orientation.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data
}
