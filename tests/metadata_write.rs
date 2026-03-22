use libjpeg_turbo_rs::{compress_with_metadata, decompress, PixelFormat, Subsampling};

#[test]
fn icc_profile_roundtrip() {
    let pixels = vec![128u8; 32 * 32 * 3];
    let fake_icc = vec![0x42u8; 200]; // dummy ICC profile
    let jpeg = compress_with_metadata(
        &pixels,
        32,
        32,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&fake_icc),
        None,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(fake_icc.as_slice()));
}

#[test]
fn exif_data_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    // Minimal valid TIFF/EXIF: little-endian, magic 42, IFD at offset 8, 0 entries
    let fake_exif = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
    let jpeg = compress_with_metadata(
        &pixels,
        16,
        16,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        Some(&fake_exif),
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.exif_data(), Some(fake_exif.as_slice()));
}

#[test]
fn large_icc_profile_splits_into_chunks() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let large_icc = vec![0xAB; 100_000]; // > 65519 bytes, needs 2 chunks
    let jpeg = compress_with_metadata(
        &pixels,
        8,
        8,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&large_icc),
        None,
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(large_icc.as_slice()));
}

#[test]
fn both_icc_and_exif_roundtrip() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let fake_icc = vec![0x42u8; 500];
    let fake_exif = vec![0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00];
    let jpeg = compress_with_metadata(
        &pixels,
        16,
        16,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&fake_icc),
        Some(&fake_exif),
    )
    .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(fake_icc.as_slice()));
    assert_eq!(img.exif_data(), Some(fake_exif.as_slice()));
}

#[test]
fn no_metadata_same_as_compress() {
    let pixels = vec![128u8; 16 * 16 * 3];
    let jpeg = compress_with_metadata(
        &pixels,
        16,
        16,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        None,
        None,
    )
    .unwrap();
    let plain =
        libjpeg_turbo_rs::compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444)
            .unwrap();
    assert_eq!(jpeg, plain);
}
