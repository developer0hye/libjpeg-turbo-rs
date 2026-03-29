use libjpeg_turbo_rs::*;

/// Helper: create an RGB test JPEG with 4:2:2 subsampling at given dimensions.
fn make_test_jpeg_422(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let yv: u8 = ((y * 7 + x * 13) % 256) as u8;
            let cb: u8 = ((x * 5 + 80) % 256) as u8;
            let cr: u8 = ((y * 3 + 120) % 256) as u8;
            pixels.push(yv);
            pixels.push(cb);
            pixels.push(cr);
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S422,
    )
    .unwrap()
}

/// Helper: create an RGB test JPEG with 4:2:0 subsampling at given dimensions.
fn make_test_jpeg_420(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            let yv: u8 = ((y * 11 + x * 7) % 256) as u8;
            let cb: u8 = ((x * 3 + 100) % 256) as u8;
            let cr: u8 = ((y * 5 + 60) % 256) as u8;
            pixels.push(yv);
            pixels.push(cb);
            pixels.push(cr);
        }
    }
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .unwrap()
}

#[test]
fn merged_422_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_422(32, 16);
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(true);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 16);
    assert_eq!(img.data.len(), 32 * 16 * 3);
    // Verify pixels are reasonable (not all zero or all same)
    let distinct: usize = img
        .data
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > 10,
        "expected diverse pixel values, got {}",
        distinct
    );
}

#[test]
fn merged_420_produces_valid_output() {
    let jpeg: Vec<u8> = make_test_jpeg_420(32, 32);
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(true);
    let img: Image = dec.finish().unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
    assert_eq!(img.data.len(), 32 * 32 * 3);
    let distinct: usize = img
        .data
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    assert!(
        distinct > 10,
        "expected diverse pixel values, got {}",
        distinct
    );
}

#[test]
fn merged_matches_fast_upsample_exactly() {
    // Merged upsampling uses box-filter (nearest-neighbor) chroma replication,
    // same as fast_upsample. The two paths should produce pixel-identical output.
    let jpeg_422: Vec<u8> = make_test_jpeg_422(64, 48);
    let jpeg_420: Vec<u8> = make_test_jpeg_420(64, 48);

    for jpeg in [&jpeg_422, &jpeg_420] {
        // Decode with fast_upsample (separate box upsample + color convert)
        let mut dec_fast = ScanlineDecoder::new(jpeg).unwrap();
        dec_fast.set_fast_upsample(true);
        let fast: Image = dec_fast.finish().unwrap();

        // Decode with merged upsample (combined box upsample + color convert)
        let mut dec_merged = ScanlineDecoder::new(jpeg).unwrap();
        dec_merged.set_merged_upsample(true);
        let merged: Image = dec_merged.finish().unwrap();

        assert_eq!(fast.width, merged.width);
        assert_eq!(fast.height, merged.height);
        assert_eq!(fast.data.len(), merged.data.len());

        // Should be pixel-identical since both use box-filter chroma replication
        let max_diff: u8 = fast
            .data
            .iter()
            .zip(merged.data.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).unsigned_abs() as u8)
            .max()
            .unwrap_or(0);

        assert!(
            max_diff == 0,
            "merged and fast_upsample must produce identical output, max diff = {}",
            max_diff
        );
    }
}

#[test]
fn merged_differs_from_fancy_upsample() {
    // Merged uses box filter while default uses fancy triangle filter.
    // They should produce different (but both valid) results for subsampled images.
    let jpeg: Vec<u8> = make_test_jpeg_420(64, 48);

    let standard: Image = decompress(&jpeg).unwrap();

    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(true);
    let merged: Image = dec.finish().unwrap();

    assert_eq!(standard.width, merged.width);
    assert_eq!(standard.height, merged.height);

    // Should differ because interpolation method is different
    let differences: usize = standard
        .data
        .iter()
        .zip(merged.data.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differences > 0,
        "merged and fancy should produce different results for 4:2:0"
    );
}

#[test]
fn merged_422_various_widths() {
    // Test odd widths, small widths, non-MCU-aligned
    for width in [1, 3, 7, 15, 17, 31, 33, 63, 65] {
        let height: usize = 16;
        let jpeg: Vec<u8> = make_test_jpeg_422(width, height);
        let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
        dec.set_merged_upsample(true);
        let img: Image = dec.finish().unwrap();
        assert_eq!(img.width, width, "width mismatch for input width={}", width);
        assert_eq!(
            img.height, height,
            "height mismatch for input width={}",
            width
        );
        assert_eq!(
            img.data.len(),
            width * height * 3,
            "data size mismatch for width={}",
            width
        );
    }
}

#[test]
fn merged_420_various_sizes() {
    // Various width x height combinations
    for (width, height) in [
        (1, 1),
        (2, 2),
        (3, 3),
        (7, 9),
        (15, 17),
        (16, 16),
        (31, 33),
        (33, 31),
        (64, 48),
        (65, 49),
    ] {
        let jpeg: Vec<u8> = make_test_jpeg_420(width, height);
        let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
        dec.set_merged_upsample(true);
        let img: Image = dec.finish().unwrap();
        assert_eq!(img.width, width, "width mismatch for {}x{}", width, height);
        assert_eq!(
            img.height, height,
            "height mismatch for {}x{}",
            width, height
        );
        assert_eq!(
            img.data.len(),
            width * height * 3,
            "data size mismatch for {}x{}",
            width,
            height
        );
    }
}

#[test]
fn merged_default_off() {
    let jpeg: Vec<u8> = make_test_jpeg_420(16, 16);
    // Default decode (no merged)
    let standard: Image = decompress(&jpeg).unwrap();

    // Explicitly disable merged (should match standard)
    let mut dec = ScanlineDecoder::new(&jpeg).unwrap();
    dec.set_merged_upsample(false);
    let explicit_off: Image = dec.finish().unwrap();

    assert_eq!(
        standard.data, explicit_off.data,
        "merged should be off by default"
    );
}
