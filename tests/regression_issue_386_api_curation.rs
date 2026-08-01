//! Issue #386: API curation — one-call `probe`, chainable decoder
//! configuration, `Image` conveniences (`Clone`/`PartialEq`/`as_bytes`/
//! `into_vec`), and integer-width `FrameHeader` dimension accessors.

mod helpers;

use libjpeg_turbo_rs::{probe, Decoder, Image, JpegError, PixelFormat, Subsampling};

const FIXTURE: &[u8] = include_bytes!("fixtures/photo_640x480_420.jpg");

fn djpeg_grayscale_pixels(djpeg: &std::path::Path, jpeg: &[u8]) -> Vec<u8> {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("grayscale_source.jpg");
    std::fs::write(&src, jpeg).expect("write grayscale source");
    let out = std::process::Command::new(djpeg)
        .arg("-grayscale")
        .arg("-pnm")
        .arg(&src)
        .output()
        .expect("run djpeg -grayscale");
    assert!(
        out.status.success(),
        "djpeg -grayscale failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // PGM: `P5\n<w> <h>\n255\n` followed by the grayscale samples.
    let header_end: usize = out
        .stdout
        .iter()
        .enumerate()
        .filter(|&(_, byte)| *byte == b'\n')
        .map(|(index, _)| index)
        .nth(2)
        .expect("PGM header")
        + 1;
    out.stdout[header_end..].to_vec()
}

fn djpeg_grayscale_rejection(djpeg: &std::path::Path, jpeg: &[u8]) -> String {
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("invalid_grayscale_source.jpg");
    std::fs::write(&src, jpeg).expect("write invalid grayscale source");
    let out = std::process::Command::new(djpeg)
        .arg("-grayscale")
        .arg("-pnm")
        .arg(&src)
        .output()
        .expect("run djpeg -grayscale on invalid source");
    assert!(
        !out.status.success(),
        "djpeg must reject the same malformed grayscale source"
    );
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn cjpeg_with_sampling(
    cjpeg: &std::path::Path,
    sampling: &str,
    width: usize,
    height: usize,
    rgb: &[u8],
) -> Vec<u8> {
    assert_eq!(rgb.len(), width * height * 3);
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("sampled_source.ppm");
    let mut ppm: Vec<u8> = format!("P6\n{width} {height}\n255\n").into_bytes();
    ppm.extend_from_slice(rgb);
    std::fs::write(&src, ppm).expect("write sampled PPM source");
    let out = std::process::Command::new(cjpeg)
        .arg("-quality")
        .arg("95")
        .arg("-sample")
        .arg(sampling)
        .arg(&src)
        .output()
        .expect("run cjpeg with explicit sampling");
    assert!(
        out.status.success(),
        "cjpeg -sample {sampling} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// Splice an EXIF APP1 segment (orientation tag only) right after SOI,
/// same layout the issue-#391 regression tests use.
fn with_exif_orientation(jpeg: &[u8], orientation: u8) -> Vec<u8> {
    let mut tiff: Vec<u8> = vec![
        0x4D,
        0x4D,
        0x00,
        0x2A, // big-endian TIFF header
        0x00,
        0x00,
        0x00,
        0x08, // IFD0 offset
        0x00,
        0x01, // 1 entry
        0x01,
        0x12, // tag 0x0112 (Orientation)
        0x00,
        0x03, // SHORT
        0x00,
        0x00,
        0x00,
        0x01, // count 1
        0x00,
        orientation,
        0x00,
        0x00, // value + pad
        0x00,
        0x00,
        0x00,
        0x00, // next IFD: none
    ];
    let mut payload: Vec<u8> = b"Exif\0\0".to_vec();
    payload.append(&mut tiff);
    let seg_len: u16 = (payload.len() + 2) as u16;
    let mut out: Vec<u8> = Vec::with_capacity(jpeg.len() + payload.len() + 4);
    out.extend_from_slice(&jpeg[..2]); // SOI
    out.extend_from_slice(&[0xFF, 0xE1]);
    out.extend_from_slice(&seg_len.to_be_bytes());
    out.extend_from_slice(&payload);
    out.extend_from_slice(&jpeg[2..]);
    out
}

#[test]
fn probe_reports_header_essentials_in_one_call() {
    let info = probe(FIXTURE).expect("probe must parse the fixture header");

    // Cross-checked against the long-form Decoder path below.
    assert_eq!(info.width, 640usize);
    assert_eq!(info.height, 480usize);
    assert_eq!(info.components, 3);
    assert_eq!(info.precision, 8);
    assert!(!info.progressive);
    assert!(!info.lossless);
    assert!(!info.arithmetic);
    assert_eq!(info.subsampling, Subsampling::S420);

    let decoder = Decoder::new(FIXTURE).expect("decoder");
    let header = decoder.header();
    assert_eq!(info.width, usize::from(header.width));
    assert_eq!(info.height, usize::from(header.height));
    assert_eq!(info.color_space, decoder.jpeg_color_space());
    assert_eq!(info.subsampling, decoder.jpeg_subsampling());
    assert_eq!(info.density, *decoder.density());
}

#[test]
fn probe_reports_metadata_presence() {
    let info = probe(FIXTURE).expect("probe");
    assert_eq!(info.exif_orientation, None);
    assert!(!info.has_exif);
    assert!(!info.has_icc);
    assert!(!info.has_xmp);
    assert!(!info.has_iptc);
    assert_eq!(info.comment, None);

    let with_exif: Vec<u8> = with_exif_orientation(FIXTURE, 6);
    let info6 = probe(&with_exif).expect("probe exif");
    assert!(info6.has_exif);
    assert_eq!(info6.exif_orientation, Some(6));
    // Header-only additions must not disturb the frame facts.
    assert_eq!(info6.width, info.width);
    assert_eq!(info6.height, info.height);
}

#[test]
fn probe_rejects_invalid_input() {
    assert!(probe(&[]).is_err());
    assert!(probe(&FIXTURE[..2]).is_err());
    assert!(probe(&[0x00, 0x01, 0x02, 0x03]).is_err());
}

/// C cross-validation: djpeg's decoded PPM header carries the same
/// dimensions probe reports.
#[test]
fn probe_dimensions_match_djpeg() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");

    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("probe_fixture.jpg");
    std::fs::write(&src, FIXTURE).expect("write fixture");
    let out = std::process::Command::new(&djpeg)
        .arg(&src)
        .output()
        .expect("run djpeg");
    assert!(out.status.success(), "djpeg failed on the fixture");
    // PPM header: "P6\n<width> <height>\n255\n"
    let header: &[u8] = out.stdout.get(..32).unwrap_or(&out.stdout);
    let text: String = String::from_utf8_lossy(header).into_owned();
    let mut fields = text.split_ascii_whitespace();
    assert_eq!(fields.next(), Some("P6"));
    let w: usize = fields.next().unwrap().parse().unwrap();
    let h: usize = fields.next().unwrap().parse().unwrap();

    let info = probe(FIXTURE).expect("probe");
    assert_eq!((info.width, info.height), (w, h));
}

#[test]
fn frame_header_exposes_usize_dimensions() {
    let decoder = Decoder::new(FIXTURE).expect("decoder");
    let header = decoder.header();
    // The accessors exist so callers stop casting the raw u16 fields.
    let w: usize = header.width();
    let h: usize = header.height();
    assert_eq!(w, usize::from(header.width));
    assert_eq!(h, usize::from(header.height));
    assert_eq!((w, h), header.dimensions());
}

#[test]
fn image_is_cloneable_and_comparable() {
    let a: Image = libjpeg_turbo_rs::decompress(FIXTURE).expect("decode a");
    let b: Image = libjpeg_turbo_rs::decompress(FIXTURE).expect("decode b");
    assert_eq!(a, b, "two decodes of the same input must compare equal");

    let c: Image = a.clone();
    assert_eq!(a, c);

    let mut d: Image = a.clone();
    d.data[0] = d.data[0].wrapping_add(1);
    assert_ne!(a, d, "a pixel difference must break equality");
}

#[test]
fn image_byte_accessors_match_data_field() {
    let img: Image = libjpeg_turbo_rs::decompress(FIXTURE).expect("decode");
    assert_eq!(img.as_bytes(), &img.data[..]);
    let expected_len: usize = img.width * img.height * img.pixel_format.bytes_per_pixel();
    assert_eq!(img.as_bytes().len(), expected_len);

    let copy: Vec<u8> = img.data.clone();
    let owned: Vec<u8> = img.into_vec();
    assert_eq!(owned, copy, "into_vec must hand back the pixel buffer");
}

#[test]
fn chainable_configuration_matches_setter_path() {
    // Chained form...
    let chained: Image = Decoder::new(FIXTURE)
        .expect("decoder")
        .with_output_format(PixelFormat::Grayscale)
        .with_block_smoothing(false)
        .with_fast_upsample(true)
        .decode_image()
        .expect("chained decode");

    // ...must be byte-identical to the long-form setter path.
    let mut decoder = Decoder::new(FIXTURE).expect("decoder");
    decoder.set_output_format(PixelFormat::Grayscale);
    decoder.set_block_smoothing(false);
    decoder.set_fast_upsample(true);
    let long_form: Image = decoder.decode_image().expect("setter decode");

    assert_eq!(chained.pixel_format, PixelFormat::Grayscale);
    assert_eq!(chained, long_form);
}

/// Issue #386: `set_output_format(Grayscale)` on a colour JPEG used to
/// fail with `Unsupported("cannot convert color JPEG to grayscale")`
/// while `set_output_colorspace(Grayscale)` succeeded — an ergonomic
/// trap TurboJPEG doesn't have (TJPF_GRAY maps to JCS_GRAYSCALE). The
/// two routes must now produce identical pixels, cross-validated
/// against `djpeg -grayscale`.
#[test]
fn grayscale_output_format_matches_colorspace_route_and_djpeg() {
    let mut by_format = Decoder::new(FIXTURE).expect("decoder");
    by_format.set_output_format(PixelFormat::Grayscale);
    let img_format: Image = by_format.decode_image().expect("gray via output_format");
    assert_eq!(img_format.pixel_format, PixelFormat::Grayscale);
    assert_eq!(img_format.data.len(), img_format.width * img_format.height);

    let mut by_cs = Decoder::new(FIXTURE).expect("decoder");
    by_cs.set_output_colorspace(libjpeg_turbo_rs::ColorSpace::Grayscale);
    let img_cs: Image = by_cs.decode_image().expect("gray via output_colorspace");
    assert_eq!(
        img_format.data, img_cs.data,
        "output_format and output_colorspace gray routes must be identical"
    );

    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");
    let c_pixels: Vec<u8> = djpeg_grayscale_pixels(&djpeg, FIXTURE);
    assert_eq!(c_pixels.len(), img_format.data.len());
    assert_eq!(
        &c_pixels,
        &img_format.data[..],
        "gray output must be pixel-identical to djpeg -grayscale (diff=0)"
    );
}

/// P4-72: component plane 0 of a JCS_RGB stream is RED, not luma. Both
/// grayscale request routes must run libjpeg's RGB→gray conversion instead
/// of copying that plane.
#[test]
fn grayscale_output_converts_rgb_colorspace_source_like_djpeg() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");
    let width: usize = 33;
    let height: usize = 21;
    let rgb: Vec<u8> = (0..width * height)
        .flat_map(|index| {
            let x: usize = index % width;
            let y: usize = index / width;
            [
                (x * 13 + y * 29) as u8,
                (x * 31 + y * 7) as u8,
                (x * 3 + y * 17) as u8,
            ]
        })
        .collect();

    for subsampling in [Subsampling::S444, Subsampling::S420] {
        let jpeg: Vec<u8> = libjpeg_turbo_rs::Encoder::new(&rgb, width, height, PixelFormat::Rgb)
            .colorspace(libjpeg_turbo_rs::ColorSpace::Rgb)
            .subsampling(subsampling)
            .quality(95)
            .encode()
            .expect("encode JCS_RGB source");
        let info = probe(&jpeg).expect("probe");
        assert_eq!(
            info.color_space,
            libjpeg_turbo_rs::ColorSpace::Rgb,
            "fixture must really be a JCS_RGB stream"
        );
        assert_eq!(
            info.subsampling, subsampling,
            "fixture must retain the requested {subsampling:?} sampling"
        );
        let expected: Vec<u8> = djpeg_grayscale_pixels(&djpeg, &jpeg);

        for route in ["output_format", "output_colorspace"] {
            let mut decoder = Decoder::new(&jpeg).expect("decoder");
            match route {
                "output_format" => decoder.set_output_format(PixelFormat::Grayscale),
                _ => decoder.set_output_colorspace(libjpeg_turbo_rs::ColorSpace::Grayscale),
            }
            let image: Image = decoder.decode_image().unwrap_or_else(|error| {
                panic!("gray via {route} for {subsampling:?} failed: {error}")
            });
            assert_eq!(image.pixel_format, PixelFormat::Grayscale);
            assert_eq!(
                image.data, expected,
                "gray via {route} for {subsampling:?} must be pixel-identical to djpeg -grayscale"
            );
        }
    }
}

/// P4-72: component 0 is not required to carry the max sampling factor.
/// With comp0=1x1 and comp1=2x2, lenient decode must upsample plane 0 before
/// emitting gray. Strict mode must continue to enforce P4-21's
/// chroma-out-samples-luma guard.
#[test]
fn grayscale_output_upsamples_subsampled_component0_like_djpeg() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");
    let cjpeg: std::path::PathBuf = require_c_tool!("cjpeg");
    let width: usize = 33;
    let height: usize = 21;
    let rgb: Vec<u8> = (0..width * height)
        .flat_map(|index| {
            let x: usize = index % width;
            let y: usize = index / width;
            [
                (x * 17 + y * 3) as u8,
                (x * 5 + y * 19) as u8,
                (x * 11 + y * 7) as u8,
            ]
        })
        .collect();
    let jpeg: Vec<u8> = cjpeg_with_sampling(&cjpeg, "1x1,2x2,1x1", width, height, &rgb);

    let mut rgb_ok = Decoder::new(&jpeg).expect("decoder");
    rgb_ok.set_lenient(true);
    rgb_ok
        .decode_image()
        .expect("the unusual sampled stream itself must stay decodable to RGB");
    let expected: Vec<u8> = djpeg_grayscale_pixels(&djpeg, &jpeg);

    for route in ["output_format", "output_colorspace"] {
        let mut strict = Decoder::new(&jpeg).expect("strict decoder");
        match route {
            "output_format" => strict.set_output_format(PixelFormat::Grayscale),
            _ => strict.set_output_colorspace(libjpeg_turbo_rs::ColorSpace::Grayscale),
        }
        match strict.decode_image() {
            Err(JpegError::CorruptData(message)) => assert!(
                message.contains("chroma upsample factor zero")
                    && message.contains("out-samples luma"),
                "strict gray via {route} returned the wrong corruption reason: {message}"
            ),
            Err(error) => panic!("strict gray via {route} returned the wrong error: {error}"),
            Ok(_) => panic!("strict gray via {route} must retain the P4-21 rejection"),
        }

        let mut decoder = Decoder::new(&jpeg).expect("decoder");
        decoder.set_lenient(true);
        match route {
            "output_format" => decoder.set_output_format(PixelFormat::Grayscale),
            _ => decoder.set_output_colorspace(libjpeg_turbo_rs::ColorSpace::Grayscale),
        }
        let image: Image = decoder
            .decode_image()
            .unwrap_or_else(|error| panic!("lenient gray via {route} failed: {error}"));
        assert_eq!(image.pixel_format, PixelFormat::Grayscale);
        assert_eq!(
            image.data, expected,
            "lenient gray via {route} must upsample component 0 like djpeg"
        );
    }
}

/// P4-72's strict P4-21 recheck must not divide by zero while a legal
/// zero-height SOF is still waiting for a DNL marker to define its height.
#[test]
fn grayscale_output_zero_height_sof_never_panics() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");
    let mut jpeg = FIXTURE.to_vec();
    let sof = jpeg
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0xC0])
        .expect("baseline SOF0 marker");
    jpeg[sof + 5] = 0;
    jpeg[sof + 6] = 0;
    assert!(
        djpeg_grayscale_rejection(&djpeg, &jpeg).contains("Empty JPEG image"),
        "C must reject the unresolved-DNL source as an empty image"
    );

    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut decoder = Decoder::new(&jpeg).expect("zero-height SOF header is DNL-capable");
        decoder.set_output_format(PixelFormat::Grayscale);
        decoder.decode_image()
    }));
    assert!(
        decoded.is_ok(),
        "zero-height grayscale decode must not panic"
    );
    assert!(
        decoded.expect("panic checked").is_err(),
        "a stream without the required DNL must be rejected"
    );
}

/// Explicit grayscale output must reject a non-standard two-component source
/// without indexing a third YCbCr component that is not present.
#[test]
fn grayscale_output_two_component_frame_never_panics() {
    let djpeg: std::path::PathBuf = require_c_tool!("djpeg");
    let mut jpeg = FIXTURE.to_vec();
    let sof = jpeg
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0xC0])
        .expect("baseline SOF0 marker");
    jpeg[sof + 3] = 14;
    jpeg[sof + 5] = 0;
    jpeg[sof + 6] = 0;
    jpeg[sof + 9] = 2;
    jpeg.drain(sof + 16..sof + 19);

    let sos = jpeg
        .windows(2)
        .position(|bytes| bytes == [0xFF, 0xDA])
        .expect("baseline SOS marker");
    jpeg[sos + 3] = 10;
    jpeg[sos + 4] = 2;
    jpeg.drain(sos + 9..sos + 11);
    let c_error = djpeg_grayscale_rejection(&djpeg, &jpeg);
    assert!(
        !c_error.trim().is_empty(),
        "C rejection must include a diagnostic"
    );

    let decoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut decoder = Decoder::new(&jpeg)?;
        decoder.set_output_colorspace(libjpeg_turbo_rs::ColorSpace::Grayscale);
        decoder.decode_image()
    }));
    assert!(
        decoded.is_ok(),
        "two-component grayscale decode must not panic"
    );
    assert!(
        decoded.expect("panic checked").is_err(),
        "a two-component source must be rejected"
    );
}

#[test]
fn chainable_limits_still_enforced() {
    // with_max_pixels must behave exactly like set_max_pixels: the
    // 640x480 fixture exceeds a 1000-pixel budget and must error.
    let result = Decoder::new(FIXTURE)
        .expect("decoder")
        .with_max_pixels(1000)
        .decode_image();
    assert!(result.is_err(), "limit must reject the oversized decode");
}
