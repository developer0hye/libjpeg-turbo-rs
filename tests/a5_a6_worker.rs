//! A5-A6 integration tests for TjHandle session reset and ICC/buffer/marker
//! symmetry. These tests belong to this worker (agent-a94c0e13) and live
//! outside the main tj3_handle.rs suite to avoid churn with other workers.

use libjpeg_turbo_rs::tj3::TjHandle;
use libjpeg_turbo_rs::{compress_with_metadata, PixelFormat, Subsampling};

/// A5-1: ICC profile symmetry between handle and decoded image.
///
/// After a successful `decompress()`, both `TjHandle::icc_profile()` and
/// `Image::icc_profile()` must report the same bytes as the embedded JPEG ICC.
/// This mirrors C libjpeg-turbo's `tj3GetICCProfile()` contract where the
/// handle retains a copy that the caller can query independently of the image.
#[test]
fn a5_1_icc_symmetry_handle_equals_image() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];
    let icc = vec![0xCDu8; 96];
    let jpeg = compress_with_metadata(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&icc),
        None,
    )
    .expect("compress_with_metadata must succeed");

    let mut handle = TjHandle::new();
    let img = handle.decompress(&jpeg).expect("decompress must succeed");

    assert_eq!(
        handle.icc_profile(),
        Some(icc.as_slice()),
        "handle ICC must equal the embedded profile"
    );
    assert_eq!(
        img.icc_profile(),
        Some(icc.as_slice()),
        "image ICC must equal the embedded profile"
    );
    assert_eq!(
        handle.icc_profile(),
        img.icc_profile(),
        "handle and image ICC must be symmetric"
    );
}

// === A5-2: TJPARAM_NOREALLOC behavior on compress_into ===
//
// When TJPARAM_NOREALLOC is set (1), `TjHandle::compress_into()` must emit
// JpegError::BufferTooSmall if the encoded JPEG exceeds the caller's buffer
// capacity. When the buffer is large enough, it returns Ok(bytes_written)
// without touching the underlying storage beyond the written slice (no
// reallocation). This mirrors C tj3Compress8 with TJPARAM_NOREALLOC.

#[test]
fn a5_2_norealloc_buffer_too_small_returns_error() {
    use libjpeg_turbo_rs::tj3::TjParam;
    use libjpeg_turbo_rs::JpegError;

    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();

    let mut handle = TjHandle::new();
    handle.set(TjParam::Quality, 90).unwrap();
    handle.set(TjParam::NoRealloc, 1).unwrap();

    let mut buf = vec![0u8; 64];
    let err = handle
        .compress_into(&pixels, width, height, PixelFormat::Rgb, &mut buf)
        .expect_err("NoRealloc must reject undersized buffer");

    match err {
        JpegError::BufferTooSmall { need, got } => {
            assert_eq!(got, 64);
            assert!(need > got, "need={need} should exceed got={got}");
        }
        other => panic!("expected BufferTooSmall, got {other:?}"),
    }
}

#[test]
fn a5_2_norealloc_adequate_buffer_writes_without_realloc() {
    use libjpeg_turbo_rs::tj3::TjParam;

    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    let mut handle = TjHandle::new();
    handle.set(TjParam::Quality, 75).unwrap();
    handle.set(TjParam::NoRealloc, 1).unwrap();

    let capacity: usize = 16 * 1024;
    let mut buf = vec![0u8; capacity];
    let buf_ptr_before: *const u8 = buf.as_ptr();
    let buf_cap_before: usize = buf.capacity();

    let written: usize = handle
        .compress_into(&pixels, width, height, PixelFormat::Rgb, &mut buf)
        .expect("adequate buffer must succeed");

    assert!(
        written >= 4,
        "written length should be meaningful, got {written}"
    );
    assert_eq!(buf[0], 0xFF);
    assert_eq!(buf[1], 0xD8);
    assert_eq!(buf[written - 2], 0xFF);
    assert_eq!(buf[written - 1], 0xD9);

    assert_eq!(
        buf.as_ptr(),
        buf_ptr_before,
        "buffer must not be reallocated"
    );
    assert_eq!(buf.capacity(), buf_cap_before, "capacity must be preserved");

    let img = libjpeg_turbo_rs::decompress(&buf[..written]).expect("decompress written JPEG");
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
}

#[test]
fn a5_2_norealloc_zero_still_errors_on_undersized_slice() {
    use libjpeg_turbo_rs::JpegError;

    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    let handle = TjHandle::new();
    let mut buf = vec![0u8; 32];
    let err = handle
        .compress_into(&pixels, width, height, PixelFormat::Rgb, &mut buf)
        .expect_err("undersized slice must error even with NoRealloc=0");
    assert!(matches!(err, JpegError::BufferTooSmall { .. }));
}

// === A5-3: TJPARAM_SAVEMARKERS wires Decoder::save_markers() ===

fn make_jpeg_with_app1_and_app2() -> Vec<u8> {
    use libjpeg_turbo_rs::Encoder;
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];
    let icc = vec![0xAAu8; 48];
    let exif = b"Exif\0\0MM\0*\0\0\0\x08\0\0\0\0".to_vec();
    Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .icc_profile(&icc)
        .exif_data(&exif)
        .encode()
        .expect("encode APP1+APP2 JPEG")
}

#[test]
fn a5_3_save_markers_level2_populates_saved_markers() {
    use libjpeg_turbo_rs::tj3::TjParam;
    let jpeg = make_jpeg_with_app1_and_app2();
    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 2).unwrap();
    let img = handle.decompress(&jpeg).expect("decompress must succeed");
    assert!(
        !img.markers().is_empty(),
        "SaveMarkers=2 must populate Image.saved_markers"
    );
    let has_app: bool = img
        .markers()
        .iter()
        .any(|m| m.code == 0xE1 || m.code == 0xE2);
    assert!(
        has_app,
        "expected at least one APP1 or APP2, got codes: {:?}",
        img.markers().iter().map(|m| m.code).collect::<Vec<_>>()
    );
}

#[test]
fn a5_3_save_markers_level0_leaves_saved_markers_empty() {
    use libjpeg_turbo_rs::tj3::TjParam;
    let jpeg = make_jpeg_with_app1_and_app2();
    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 0).unwrap();
    let img = handle.decompress(&jpeg).expect("decompress must succeed");
    assert!(
        img.markers().is_empty(),
        "SaveMarkers=0 must leave saved_markers empty, got {} markers",
        img.markers().len()
    );
}

// === A6-1: Encoder::reset_colorspace() matches jpeg_default_colorspace() ===
//
// `reset_colorspace()` clears any previously-set JPEG colorspace override
// and restores inference from `PixelFormat`, mirroring C libjpeg-turbo's
// `jpeg_default_colorspace()`. Concretely: after calling `.colorspace(Rgb)`
// followed by `.reset_colorspace()`, a CMYK input must encode as a
// 4-component JPEG (with the Adobe APP14 marker signaling CMYK), exactly
// as if the override had never been applied.

fn parse_sof_component_count(jpeg: &[u8]) -> Option<u8> {
    let mut i: usize = 2;
    while i + 3 < jpeg.len() && jpeg[i] == 0xFF {
        let code = jpeg[i + 1];
        if code == 0xD9 {
            return None;
        }
        // Standalone markers without payload length.
        if (0xD0..=0xD9).contains(&code) || code == 0x01 {
            i += 2;
            continue;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if matches!(code, 0xC0 | 0xC1 | 0xC2 | 0xC3 | 0xC9 | 0xCA | 0xCB) {
            // SOFn payload: [P(1)][Y(2)][X(2)][Nf(1)][components...]
            return Some(jpeg[i + 4 + 5]);
        }
        i += 2 + seg_len;
    }
    None
}

fn find_adobe_transform(jpeg: &[u8]) -> Option<u8> {
    let mut i: usize = 2;
    while i + 3 < jpeg.len() && jpeg[i] == 0xFF {
        let code = jpeg[i + 1];
        if code == 0xDA || code == 0xD9 {
            return None;
        }
        let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
        if code == 0xEE
            && seg_len >= 12
            && i + 4 + 5 <= jpeg.len()
            && &jpeg[i + 4..i + 9] == b"Adobe"
        {
            // Adobe APP14 layout after the len field (12 payload bytes):
            //   "Adobe"(5) ver(2) flags0(2) flags1(2) transform(1)
            return Some(jpeg[i + 2 + seg_len - 1]);
        }
        i += 2 + seg_len;
    }
    None
}

#[test]
fn a6_1_reset_colorspace_auto_selects_cmyk_for_cmyk_input() {
    use libjpeg_turbo_rs::{ColorSpace, Encoder};
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height * 4).map(|i| (i % 251) as u8).collect();

    // Force an incorrect override, then reset and encode.
    let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Cmyk)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .reset_colorspace()
        .encode()
        .expect("reset_colorspace then CMYK encode must succeed");

    // SOI check.
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);

    assert_eq!(
        parse_sof_component_count(&jpeg),
        Some(4),
        "CMYK input must encode as 4-component JPEG after reset_colorspace"
    );

    assert_eq!(
        find_adobe_transform(&jpeg),
        Some(0),
        "Adobe APP14 transform must be 0 (CMYK) after reset_colorspace"
    );
}

#[test]
fn a6_1_reset_colorspace_clears_rgb_override_for_rgb_input() {
    use libjpeg_turbo_rs::{ColorSpace, Encoder};
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    // With JCS_RGB override, encoder emits RGB-direct (no color conversion).
    let jpeg_rgb = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .encode()
        .unwrap();

    // After reset, auto-detection picks YCbCr — different output bytes.
    let jpeg_auto = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .colorspace(ColorSpace::Rgb)
        .reset_colorspace()
        .encode()
        .unwrap();

    assert_ne!(
        jpeg_rgb, jpeg_auto,
        "reset_colorspace must undo the RGB override"
    );
}
