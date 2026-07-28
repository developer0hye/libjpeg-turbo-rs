//! Issue #386: API curation — one-call `probe`, chainable decoder
//! configuration, `Image` conveniences (`Clone`/`PartialEq`/`as_bytes`/
//! `into_vec`), and integer-width `FrameHeader` dimension accessors.

mod helpers;

use libjpeg_turbo_rs::{probe, Decoder, Image, PixelFormat, Subsampling};

const FIXTURE: &[u8] = include_bytes!("fixtures/photo_640x480_420.jpg");

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
    let dir = tempfile::tempdir().expect("tempdir");
    let src = dir.path().join("gray_fixture.jpg");
    std::fs::write(&src, FIXTURE).expect("write fixture");
    let out = std::process::Command::new(&djpeg)
        .arg("-grayscale")
        .arg("-pnm")
        .arg(&src)
        .output()
        .expect("run djpeg");
    assert!(out.status.success(), "djpeg -grayscale failed");
    // PGM: "P5\n<w> <h>\n255\n" then w*h luma bytes.
    let header_end: usize = out
        .stdout
        .iter()
        .enumerate()
        .filter(|&(_, b)| *b == b'\n')
        .map(|(i, _)| i)
        .nth(2)
        .expect("PGM header")
        + 1;
    let c_pixels: &[u8] = &out.stdout[header_end..];
    assert_eq!(c_pixels.len(), img_format.data.len());
    assert_eq!(
        c_pixels,
        &img_format.data[..],
        "gray output must be pixel-identical to djpeg -grayscale (diff=0)"
    );
}

/// Codex P1 on #386: the implied gray route must NOT trigger for
/// JCS_RGB sources — the override's gray path emits component plane 0,
/// which is RED there, not luma. Until a real RGB→gray conversion
/// exists (P4-72), the request must fail loudly instead of returning
/// wrong pixels.
#[test]
fn grayscale_output_format_rejects_rgb_colorspace_source() {
    let rgb: Vec<u8> = (0..32u32 * 32 * 3).map(|i| (i * 7) as u8).collect();
    let jpeg: Vec<u8> = libjpeg_turbo_rs::Encoder::new(&rgb, 32, 32, PixelFormat::Rgb)
        .colorspace(libjpeg_turbo_rs::ColorSpace::Rgb)
        .quality(95)
        .encode()
        .expect("encode JCS_RGB source");
    assert_eq!(
        probe(&jpeg).expect("probe").color_space,
        libjpeg_turbo_rs::ColorSpace::Rgb,
        "fixture must really be a JCS_RGB stream"
    );

    let mut decoder = Decoder::new(&jpeg).expect("decoder");
    decoder.set_output_format(PixelFormat::Grayscale);
    let result = decoder.decode_image();
    assert!(
        matches!(result, Err(libjpeg_turbo_rs::JpegError::Unsupported(_))),
        "gray from a JCS_RGB source must raise Unsupported, got {result:?}"
    );
}

/// Review P1 on #386: component 0 is not required to carry the max
/// sampling factor. With comp0=1x1 and comp1=2x2 the stream decodes to
/// RGB under `set_lenient(true)` (strict decode is rejected earlier by
/// P4-21's chroma-out-samples-luma guard), but plane 0 is quarter-size
/// — the gray output arm used to slice past it and panic (`range end
/// index 77120 out of range`). Both gray routes must return an error,
/// never panic.
#[test]
fn grayscale_output_rejects_subsampled_component0_without_panicking() {
    // Patch the SOF0 sampling bytes: comp0 2x2 -> 1x1, comp1 1x1 -> 2x2.
    let mut jpeg: Vec<u8> = FIXTURE.to_vec();
    let sof: usize = jpeg
        .windows(2)
        .position(|w| w == [0xFF, 0xC0])
        .expect("baseline SOF0 in fixture");
    // SOF0 layout: FF C0 len(2) prec(1) h(2) w(2) ncomp(1), then per
    // component: id(1) sampling(1) qtbl(1).
    let comp0_sampling: usize = sof + 11;
    let comp1_sampling: usize = comp0_sampling + 3;
    assert_eq!(jpeg[comp0_sampling], 0x22, "fixture must start as 4:2:0");
    jpeg[comp0_sampling] = 0x11;
    jpeg[comp1_sampling] = 0x22;

    let mut rgb_ok = Decoder::new(&jpeg).expect("decoder");
    rgb_ok.set_lenient(true);
    rgb_ok
        .decode_image()
        .expect("the patched stream itself must stay decodable to RGB");

    for route in ["output_format", "output_colorspace"] {
        let mut decoder = Decoder::new(&jpeg).expect("decoder");
        decoder.set_lenient(true);
        match route {
            "output_format" => decoder.set_output_format(PixelFormat::Grayscale),
            _ => decoder.set_output_colorspace(libjpeg_turbo_rs::ColorSpace::Grayscale),
        }
        let result = decoder.decode_image();
        assert!(
            matches!(result, Err(libjpeg_turbo_rs::JpegError::Unsupported(_))),
            "gray via {route} with quarter-size comp0 must raise Unsupported \
             (the stream is valid; the conversion is missing, P4-72), got {result:?}"
        );
    }
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
