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
