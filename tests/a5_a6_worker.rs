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
