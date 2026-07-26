//! Issue #325: the encoder's input validation had no tests.
//!
//! Mutation sampling of `src/encode/pipeline.rs` found the whole validation
//! prologue of `compress_with_params` survivable: `width == 0 || height == 0`
//! could become `&&`, `width > 65535` could become `>=` or `==`, and the
//! buffer-size check `width * height * bpp` could become `/` or `+` — all with
//! the encode suite green.
//!
//! Nothing exercised them because every other test passes well-formed input.
//! These are the arguments a caller most plausibly gets wrong, so an
//! unvalidated boundary here surfaces as a panic or a silent out-of-bounds
//! read rather than a clean error.

use libjpeg_turbo_rs::encode::pipeline::{compress_with_params, CompressParams};
use libjpeg_turbo_rs::{JpegError, PixelFormat, Subsampling};

fn encode(
    pixels: &[u8],
    width: usize,
    height: usize,
    format: PixelFormat,
) -> Result<Vec<u8>, JpegError> {
    compress_with_params(&CompressParams::new(
        pixels,
        width,
        height,
        format,
        75,
        Subsampling::S420,
    ))
}

/// Each dimension must be rejected on its own — an `&&` here would accept a
/// zero width as long as the height were non-zero, and then index an empty
/// plane.
#[test]
fn zero_dimensions_are_rejected_independently() {
    let pixels: Vec<u8> = vec![0u8; 64 * 64 * 3];
    for (width, height, label) in [
        (0usize, 16usize, "zero width"),
        (16, 0, "zero height"),
        (0, 0, "both zero"),
    ] {
        let result = encode(&pixels, width, height, PixelFormat::Rgb);
        assert!(
            matches!(result, Err(JpegError::CorruptData(_))),
            "{label} ({width}x{height}) was accepted; expected CorruptData"
        );
    }
    // Control: the smallest legal image must still encode.
    assert!(
        encode(&pixels, 1, 1, PixelFormat::Rgb).is_ok(),
        "1x1 must be accepted"
    );
}

/// JPEG's SOF carries 16-bit dimensions, so 65535 is the last legal value.
/// Both sides of that boundary are pinned so `>` cannot silently become `>=`.
#[test]
fn dimension_limit_boundary_is_exact() {
    // 65535 x 1 is legal and small enough to actually encode.
    let wide: Vec<u8> = vec![128u8; 65535 * 3];
    assert!(
        encode(&wide, 65535, 1, PixelFormat::Rgb).is_ok(),
        "65535 is the largest legal dimension and must be accepted"
    );

    let tall: Vec<u8> = vec![128u8; 65535 * 3];
    assert!(
        encode(&tall, 1, 65535, PixelFormat::Rgb).is_ok(),
        "65535 height must be accepted"
    );

    // One past the limit must be rejected, in each axis independently — an
    // `||` turned `&&` would only reject when *both* exceed it.
    for (width, height, label) in [
        (65536usize, 1usize, "width 65536"),
        (1, 65536, "height 65536"),
        (65536, 65536, "both 65536"),
    ] {
        let result = encode(&[], width, height, PixelFormat::Rgb);
        assert!(
            matches!(result, Err(JpegError::CorruptData(_))),
            "{label} was accepted; expected CorruptData"
        );
    }
}

/// The required buffer is `width * height * bytes_per_pixel`. A `/` or `+` in
/// that product under-computes it, so an undersized buffer would be accepted
/// and read out of bounds.
#[test]
fn undersized_buffer_is_rejected_for_every_format() {
    let formats: &[PixelFormat] = &[
        PixelFormat::Grayscale,
        PixelFormat::Rgb,
        PixelFormat::Rgba,
        PixelFormat::Bgr,
        PixelFormat::Bgra,
        PixelFormat::Cmyk,
    ];
    let (width, height) = (16usize, 16usize);

    for &format in formats {
        let needed: usize = width * height * format.bytes_per_pixel();

        // Exactly enough must work.
        let exact: Vec<u8> = vec![128u8; needed];
        assert!(
            encode(&exact, width, height, format).is_ok(),
            "{format:?}: an exactly-sized buffer ({needed} bytes) was rejected"
        );

        // One byte short must not.
        let short: Vec<u8> = vec![128u8; needed - 1];
        let result = encode(&short, width, height, format);
        assert!(
            matches!(result, Err(JpegError::BufferTooSmall { .. })),
            "{format:?}: a buffer one byte short of {needed} was accepted — \
             the size computation is wrong and this would read out of bounds"
        );

        // A buffer sized as if bytes_per_pixel were 1 catches a `*` turned `/`
        // or `+` for every multi-byte format.
        if format.bytes_per_pixel() > 1 {
            let single_channel: Vec<u8> = vec![128u8; width * height];
            let result = encode(&single_channel, width, height, format);
            assert!(
                matches!(result, Err(JpegError::BufferTooSmall { .. })),
                "{format:?}: a 1-byte-per-pixel buffer was accepted for a \
                 {}-byte format",
                format.bytes_per_pixel()
            );
        }
    }
}

/// The reported requirement must be the real one, not just "some error".
#[test]
fn buffer_too_small_reports_the_actual_requirement() {
    let (width, height) = (16usize, 16usize);
    let needed: usize = width * height * 3;
    let short: Vec<u8> = vec![128u8; needed - 10];

    match encode(&short, width, height, PixelFormat::Rgb) {
        Err(JpegError::BufferTooSmall { need, got }) => {
            assert_eq!(need, needed, "reported requirement is wrong");
            assert_eq!(got, needed - 10, "reported actual size is wrong");
        }
        other => panic!("expected BufferTooSmall, got {other:?}"),
    }
}
