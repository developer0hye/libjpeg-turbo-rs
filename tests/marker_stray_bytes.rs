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

/// PR #678: recovery must expose the warning that C reports, including byte stuffing.
#[test]
fn recovered_marker_bytes_report_warnings_like_c() {
    let djpeg = require_c_tool!("djpeg");
    let jpeg = gradient_jpeg();
    for (stray, count) in [
        (&[0x12, 0x34][..], 2),
        (&[0x00][..], 1),
        (&[0x00, 0xFF, 0x00][..], 3),
        (&[0xFF, 0xFF, 0x00][..], 2),
        (&[0xFF, 0xFF][..], 0),
    ] {
        let damaged = with_stray_bytes_after_first_segment(&jpeg, stray);
        let input = helpers::TempFile::new("marker_warning.jpg");
        let output = helpers::TempFile::new("marker_warning.ppm");
        input.write_bytes(&damaged);
        let oracle = std::process::Command::new(&djpeg)
            .arg("-outfile")
            .arg(output.path())
            .arg(input.path())
            .output()
            .unwrap();
        assert_eq!(oracle.status.code(), Some(if count == 0 { 0 } else { 2 }));
        if count != 0 {
            assert!(String::from_utf8_lossy(&oracle.stderr)
                .contains(&format!("{count} extraneous bytes")));
        }
        let image = decompress(&damaged).unwrap();
        let (w, h, pixels) = helpers::parse_ppm_file(output.path());
        helpers::assert_pixels_identical(
            &image.data,
            &pixels,
            w,
            h,
            3,
            "marker warning recovery vs C",
        );
        assert_eq!(image.warnings.len(), usize::from(count != 0));
        if count != 0 {
            assert!(image.warnings[0]
                .to_string()
                .contains(&format!("{count} extraneous bytes")));
        }
        let mut decoder = libjpeg_turbo_rs::Decoder::new(&damaged).unwrap();
        let mut buffer = vec![0; image.data.len()];
        let info = decoder.decode_image_into(&mut buffer).unwrap();
        assert_eq!(info.warnings, image.warnings);
        assert_eq!(buffer, pixels);
        assert_eq!(decoder.decode_image().unwrap().warnings, image.warnings);
        if count == 0 {
            decoder.set_stop_on_warning(true);
            assert_eq!(decoder.decode_image().unwrap().data, pixels);
            let mut handle = libjpeg_turbo_rs::tj3::TjHandle::new();
            handle
                .set(libjpeg_turbo_rs::tj3::TjParam::StopOnWarning, 1)
                .unwrap();
            handle.decompress_header(&damaged).unwrap();
            assert_eq!(handle.decompress(&damaged).unwrap().data, pixels);
        }
    }
}

/// PR #678: STOPONWARNING must reject recovered header bytes, as djpeg -strict does.
#[test]
fn stop_on_warning_rejects_recovered_marker_bytes() {
    use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
    let djpeg = require_c_tool!("djpeg");
    let damaged = with_stray_bytes_after_first_segment(&gradient_jpeg(), &[0x12, 0x34]);
    let input = helpers::TempFile::new("marker_strict.jpg");
    input.write_bytes(&damaged);
    let oracle = std::process::Command::new(djpeg)
        .arg("-strict")
        .arg(input.path())
        .output()
        .unwrap();
    assert_eq!(oracle.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&oracle.stderr).contains("2 extraneous bytes"));
    let mut decoder = libjpeg_turbo_rs::Decoder::new(&damaged).unwrap();
    decoder.set_stop_on_warning(true);
    assert!(decoder.decode_image().is_err());
    let mut buffer = vec![0; 48 * 32 * 3];
    assert!(decoder.decode_image_into(&mut buffer).is_err());
    assert!(decoder.decode_raw().is_err());
    let mut handle = TjHandle::new();
    handle.set(TjParam::StopOnWarning, 1).unwrap();
    assert!(handle.decompress_header(&damaged).is_err());
    assert!(handle.decompress(&damaged).is_err());
}

/// PR #678: a later scan's marker warning must survive the full metadata walk.
#[test]
fn progressive_late_marker_warning_does_not_poison_header_only_reads() {
    use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
    let cjpeg = require_c_tool!("cjpeg");
    let djpeg = require_c_tool!("djpeg");
    let mut ppm = b"P6\n48 32\n255\n".to_vec();
    ppm.extend(helpers::generate_gradient(48, 32));
    let jpeg = helpers::encode_with_c_cjpeg(&cjpeg, &ppm, &["-progressive"], "marker_progressive");
    let first_sos = jpeg.windows(2).position(|b| b == [0xFF, 0xDA]).unwrap();
    let dht = jpeg
        .windows(2)
        .enumerate()
        .find(|(p, b)| *p > first_sos && *b == [0xFF, 0xC4])
        .unwrap()
        .0;
    let at = dht + 2 + usize::from(u16::from_be_bytes([jpeg[dht + 2], jpeg[dht + 3]]));
    assert_eq!(jpeg[at], 0xFF);
    let mut damaged = jpeg[..at].to_vec();
    damaged.extend_from_slice(&[0x12, 0x34]);
    damaged.extend_from_slice(&jpeg[at..]);
    let input = helpers::TempFile::new("late_marker.jpg");
    let output = helpers::TempFile::new("late_marker.ppm");
    input.write_bytes(&damaged);
    let oracle = std::process::Command::new(&djpeg)
        .arg("-outfile")
        .arg(output.path())
        .arg(input.path())
        .output()
        .unwrap();
    assert_eq!(oracle.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&oracle.stderr).contains("2 extraneous bytes"));
    let image = decompress(&damaged).unwrap();
    let (w, h, pixels) = helpers::parse_ppm_file(output.path());
    helpers::assert_pixels_identical(&image.data, &pixels, w, h, 3, "late marker recovery vs C");
    assert_eq!(image.warnings.len(), 1);
    let buffered = libjpeg_turbo_rs::ProgressiveDecoder::new(&damaged)
        .unwrap()
        .finish()
        .unwrap();
    assert_eq!(buffered.warnings, image.warnings);
    helpers::assert_pixels_identical(
        &buffered.data,
        &pixels,
        w,
        h,
        3,
        "buffered marker recovery vs C",
    );
    let strict = std::process::Command::new(&djpeg)
        .arg("-strict")
        .arg(input.path())
        .output()
        .unwrap();
    assert_eq!(strict.status.code(), Some(1));
    let mut handle = TjHandle::new();
    handle.set(TjParam::StopOnWarning, 1).unwrap();
    handle
        .decompress_header(&damaged)
        .expect("header ends before the damaged later scan");
    assert!(handle.decompress(&damaged).is_err());
    handle.set(TjParam::StopOnWarning, 0).unwrap();
    let recovered = handle.decompress(&damaged).unwrap();
    assert_eq!(recovered.warnings, image.warnings);
    assert_eq!(recovered.data, pixels);
    let twice = with_stray_bytes_after_first_segment(&damaged, &[0x00, 0xFF, 0x00]);
    input.write_bytes(&twice);
    // C prints only its first warning unless trace_level is at least three.
    let oracle = std::process::Command::new(&djpeg)
        .args(["-verbose", "-verbose", "-verbose"])
        .arg("-outfile")
        .arg(output.path())
        .arg(input.path())
        .output()
        .unwrap();
    assert_eq!(oracle.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&oracle.stderr);
    assert!(stderr.contains("3 extraneous bytes") && stderr.contains("2 extraneous bytes"));
    let twice_image = decompress(&twice).unwrap();
    let (w, h, twice_pixels) = helpers::parse_ppm_file(output.path());
    helpers::assert_pixels_identical(
        &twice_image.data,
        &twice_pixels,
        w,
        h,
        3,
        "two marker recoveries vs C",
    );
    assert_eq!(
        twice_image.warnings,
        vec![libjpeg_turbo_rs::DecodeWarning::ExtraneousData { bytes: 5 }]
    );
}

/// PR #678: high-precision handles must inspect warnings after the first SOS too.
#[test]
fn high_precision_strict_decode_checks_later_scan_markers() {
    use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
    let cjpeg = require_c_tool!("cjpeg");
    let djpeg = require_c_tool!("djpeg");
    let mut ppm = b"P6\n48 32\n255\n".to_vec();
    ppm.extend(helpers::generate_gradient(48, 32));
    let jpeg = helpers::encode_with_c_cjpeg(
        &cjpeg,
        &ppm,
        &["-progressive", "-precision", "12"],
        "marker_12bit",
    );
    let first_sos = jpeg.windows(2).position(|b| b == [0xFF, 0xDA]).unwrap();
    let dht = jpeg
        .windows(2)
        .enumerate()
        .find(|(p, b)| *p > first_sos && *b == [0xFF, 0xC4])
        .unwrap()
        .0;
    let at = dht + 2 + usize::from(u16::from_be_bytes([jpeg[dht + 2], jpeg[dht + 3]]));
    let mut damaged = jpeg[..at].to_vec();
    damaged.extend_from_slice(&[0x12, 0x34]);
    damaged.extend_from_slice(&jpeg[at..]);
    let input = helpers::TempFile::new("late_marker_12.jpg");
    input.write_bytes(&damaged);
    let oracle = std::process::Command::new(djpeg)
        .arg("-strict")
        .arg(input.path())
        .output()
        .unwrap();
    assert_eq!(oracle.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&oracle.stderr).contains("2 extraneous bytes"));
    let mut handle = TjHandle::new();
    handle.set(TjParam::StopOnWarning, 1).unwrap();
    handle.decompress_header(&damaged).unwrap();
    let error = handle.decompress_12bit(&damaged).unwrap_err();
    assert!(error.to_string().contains("stop_on_warning"), "{error}");
}

/// PR #678: FF fill before a stuffed entropy byte is not discarded marker data.
#[test]
fn entropy_fill_before_stuffed_byte_is_warning_free() {
    let cjpeg = require_c_tool!("cjpeg");
    let djpeg = require_c_tool!("djpeg");
    let mut ppm = b"P6\n128 128\n255\n".to_vec();
    let mut state: u32 = 42;
    for _ in 0..128 * 128 * 3 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        ppm.push(state as u8);
    }
    let jpeg = helpers::encode_with_c_cjpeg(
        &cjpeg,
        &ppm,
        &["-quality", "95", "-progressive"],
        "entropy_fill",
    );
    let sos = jpeg.windows(2).position(|b| b == [0xFF, 0xDA]).unwrap();
    let at = sos
        + jpeg[sos..]
            .windows(2)
            .position(|b| b == [0xFF, 0])
            .expect("noise fixture has a stuffed entropy byte");
    let mut padded = jpeg[..at].to_vec();
    padded.push(0xFF);
    padded.extend_from_slice(&jpeg[at..]);
    let input = helpers::TempFile::new("entropy_fill.jpg");
    let output = helpers::TempFile::new("entropy_fill.ppm");
    input.write_bytes(&padded);
    let oracle = std::process::Command::new(djpeg)
        .arg("-strict")
        .arg("-outfile")
        .arg(output.path())
        .arg(input.path())
        .output()
        .unwrap();
    assert_eq!(
        oracle.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&oracle.stderr)
    );
    let image = decompress(&padded).unwrap();
    let (w, h, pixels) = helpers::parse_ppm_file(output.path());
    helpers::assert_pixels_identical(&image.data, &pixels, w, h, 3, "entropy fill vs strict C");
    assert!(image.warnings.is_empty(), "{:?}", image.warnings);
    let mut decoder = libjpeg_turbo_rs::Decoder::new(&padded).unwrap();
    decoder.set_stop_on_warning(true);
    assert_eq!(decoder.decode_image().unwrap().data, pixels);
}

/// The TurboJPEG YUV-plane decode builds its own `Decoder`, so it must refuse
/// stray bytes before a later scan's marker under STOPONWARNING as
/// `djpeg -strict` does, and still decode them without it.
#[test]
fn stop_on_warning_rejects_late_marker_bytes_in_yuv_plane_decode() {
    use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
    let cjpeg = require_c_tool!("cjpeg");
    let djpeg = require_c_tool!("djpeg");
    let mut ppm = b"P6\n48 32\n255\n".to_vec();
    ppm.extend(helpers::generate_gradient(48, 32));
    let jpeg = helpers::encode_with_c_cjpeg(&cjpeg, &ppm, &["-progressive"], "marker_yuv_late");
    let first_sos = jpeg.windows(2).position(|b| b == [0xFF, 0xDA]).unwrap();
    let dht = jpeg
        .windows(2)
        .enumerate()
        .find(|(p, b)| *p > first_sos && *b == [0xFF, 0xC4])
        .unwrap()
        .0;
    let at = dht + 2 + usize::from(u16::from_be_bytes([jpeg[dht + 2], jpeg[dht + 3]]));
    assert_eq!(jpeg[at], 0xFF);
    let mut damaged = jpeg[..at].to_vec();
    damaged.extend_from_slice(&[0x12, 0x34]);
    damaged.extend_from_slice(&jpeg[at..]);
    let input = helpers::TempFile::new("late_marker_yuv.jpg");
    input.write_bytes(&damaged);
    let strict = std::process::Command::new(&djpeg)
        .arg("-strict")
        .arg(input.path())
        .output()
        .unwrap();
    assert_eq!(strict.status.code(), Some(1), "djpeg -strict refuses it");

    let mut handle = TjHandle::new();
    handle.set(TjParam::StopOnWarning, 1).unwrap();
    let refused = handle
        .decompress_to_yuv_planes(&damaged)
        .expect_err("STOPONWARNING must refuse late stray marker bytes");
    assert!(
        refused.to_string().contains("extraneous bytes"),
        "unexpected refusal: {refused}"
    );
    handle.set(TjParam::StopOnWarning, 0).unwrap();
    let (planes, ..) = handle
        .decompress_to_yuv_planes(&damaged)
        .expect("without STOPONWARNING the stray bytes are skipped");
    let (clean_planes, ..) = TjHandle::new().decompress_to_yuv_planes(&jpeg).unwrap();
    assert_eq!(planes, clean_planes, "recovery decodes as the clean stream");
}
