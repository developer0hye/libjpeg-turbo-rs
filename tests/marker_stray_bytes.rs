//! Stray bytes before a marker.
//!
//! C libjpeg's `next_marker` (jdmarker.c) skips every byte that is not 0xFF before a marker, with
//! the warning JWRN_EXTRANEOUS_DATA, so `djpeg` decodes a JPEG whose writer left bytes between two
//! segments. The decoder must decode it the same way, pixel for pixel.

use libjpeg_turbo_rs::{compress, decompress, Image, PixelFormat, Subsampling};

mod helpers;

/// A 48x32 RGB gradient, encoded at quality 90 with 4:2:0 subsampling.
fn gradient_jpeg() -> Vec<u8> {
    let (width, height): (usize, usize) = (48, 32);
    let pixels: Vec<u8> = helpers::generate_gradient(width, height);
    compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        90,
        Subsampling::S420,
    )
    .expect("compressing the gradient must succeed")
}

/// `jpeg` with `stray` inserted before the marker of the segment that follows the first one after
/// SOI.
fn with_stray_bytes_after_first_segment(jpeg: &[u8], stray: &[u8]) -> Vec<u8> {
    assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "the stream starts with SOI");
    assert_eq!(jpeg[2], 0xFF, "a marker follows SOI");
    let length: usize = u16::from_be_bytes([jpeg[4], jpeg[5]]) as usize;
    let next: usize = 4 + length;
    assert_eq!(jpeg[next], 0xFF, "a marker follows the first segment");
    let mut damaged: Vec<u8> = Vec::with_capacity(jpeg.len() + stray.len());
    damaged.extend_from_slice(&jpeg[..next]);
    damaged.extend_from_slice(stray);
    damaged.extend_from_slice(&jpeg[next..]);
    damaged
}

#[test]
fn stray_bytes_between_two_segments_decode_as_without_them() {
    let jpeg: Vec<u8> = gradient_jpeg();
    let clean: Image = decompress(&jpeg).expect("the clean stream decodes");
    for stray in [&[0x00u8][..], &[0x12, 0x34, 0x56], &[0x00, 0xFF, 0x00]] {
        let damaged: Vec<u8> = with_stray_bytes_after_first_segment(&jpeg, stray);
        let decoded: Image = decompress(&damaged)
            .unwrap_or_else(|error| panic!("stray bytes {stray:02X?} must be skipped: {error}"));
        assert_eq!((decoded.width, decoded.height), (clean.width, clean.height));
        helpers::assert_pixels_identical(
            &decoded.data,
            &clean.data,
            clean.width,
            clean.height,
            3,
            &format!("stray bytes {stray:02X?} against the clean stream"),
        );
    }
}

#[test]
fn stray_bytes_between_two_segments_decode_as_c_djpeg_does() {
    let Some(djpeg) = helpers::djpeg_path() else {
        assert!(
            !helpers::is_ci(),
            "CI provisions libjpeg-turbo, so djpeg must be discoverable"
        );
        eprintln!("SKIP: djpeg not found");
        return;
    };
    let damaged: Vec<u8> = with_stray_bytes_after_first_segment(&gradient_jpeg(), &[0x12, 0x34]);
    let decoded: Image = decompress(&damaged).expect("stray bytes must be skipped");
    // djpeg decodes the file and warns, so it exits with EXIT_WARNING (2), which
    // `helpers::decode_with_c_djpeg` takes for a failure.
    let jpeg_file: helpers::TempFile = helpers::TempFile::new("marker_stray_bytes.jpg");
    let ppm_file: helpers::TempFile = helpers::TempFile::new("marker_stray_bytes.ppm");
    jpeg_file.write_bytes(&damaged);
    let output: std::process::Output = std::process::Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(ppm_file.path())
        .arg(jpeg_file.path())
        .output()
        .unwrap_or_else(|error| panic!("failed to run djpeg: {error:?}"));
    let stderr: std::borrow::Cow<'_, str> = String::from_utf8_lossy(&output.stderr);
    assert_eq!(
        output.status.code(),
        Some(2),
        "djpeg warns and decodes: {stderr}"
    );
    assert!(
        stderr.contains("2 extraneous bytes before marker"),
        "djpeg skips the stray bytes: {stderr}"
    );
    let (width, height, c_pixels) = helpers::parse_ppm_file(ppm_file.path());
    assert_eq!((decoded.width, decoded.height), (width, height));
    helpers::assert_pixels_identical(&decoded.data, &c_pixels, width, height, 3, "Rust vs djpeg");
}
