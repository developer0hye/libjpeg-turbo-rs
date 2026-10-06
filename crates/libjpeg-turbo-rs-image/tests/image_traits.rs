//! The adapter through the `image` traits, as an application uses it (#637).
//!
//! Three oracles, each for what it is authoritative on:
//! * pixels — the core crate's own decode/encode of the same stream, which the
//!   root crate cross-validates against C libjpeg-turbo, plus one direct
//!   `djpeg` comparison here so the chain is not only transitive;
//! * metadata and orientation — `image 0.25`'s built-in JPEG decoder, whose
//!   semantics an application switching backends expects to keep;
//! * error categories — the `ImageError` variants `image` documents for each
//!   kind of refusal.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use image::error::{LimitErrorKind, ParameterErrorKind};
use image::metadata::Orientation;
use image::{
    ColorType, DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, ImageError, Limits,
};
use libjpeg_turbo_rs::{compress, decompress_to, Encoder, PixelFormat, Subsampling};
use libjpeg_turbo_rs_image::{JpegDecoder, JpegEncoder};

fn workspace_file(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn read_workspace_file(relative: &str) -> Vec<u8> {
    let path: PathBuf = workspace_file(relative);
    std::fs::read(&path).unwrap_or_else(|error| panic!("missing fixture {path:?}: {error}"))
}

fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push((x * 255 / width) as u8);
            pixels.push((y * 255 / height) as u8);
            pixels.push(((x * 7 + y * 3) % 256) as u8);
        }
    }
    pixels
}

/// A little-endian TIFF payload carrying only the Orientation tag.
fn tiff_with_orientation(orientation: u16) -> Vec<u8> {
    let mut data: Vec<u8> = Vec::new();
    data.extend_from_slice(b"II");
    data.extend_from_slice(&42u16.to_le_bytes());
    data.extend_from_slice(&8u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&0x0112u16.to_le_bytes());
    data.extend_from_slice(&3u16.to_le_bytes());
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&orientation.to_le_bytes());
    data.extend_from_slice(&0u16.to_le_bytes());
    data.extend_from_slice(&0u32.to_le_bytes());
    data
}

/// Read every pixel through `ImageDecoder::read_image`.
fn read_all(decoder: JpegDecoder) -> Vec<u8> {
    let mut pixels: Vec<u8> = vec![0u8; decoder.total_bytes() as usize];
    decoder.read_image(&mut pixels).expect("read_image");
    pixels
}

/// The streams the pixel-parity test walks, with the format each must decode to.
fn parity_cases() -> Vec<(&'static str, Vec<u8>, ColorType, PixelFormat)> {
    let (width, height): (usize, usize) = (67, 45);
    let rgb: Vec<u8> = gradient_rgb(width, height);
    let gray: Vec<u8> = rgb.iter().step_by(3).copied().collect();
    let encode = |subsampling: Subsampling, progressive: bool| -> Vec<u8> {
        Encoder::new(&rgb, width, height, PixelFormat::Rgb)
            .quality(85)
            .subsampling(subsampling)
            .progressive(progressive)
            .encode()
            .expect("encode fixture")
    };
    vec![
        (
            "testorig.jpg (4:2:0 baseline)",
            read_workspace_file("references/libjpeg-turbo/testimages/testorig.jpg"),
            ColorType::Rgb8,
            PixelFormat::Rgb,
        ),
        (
            "testimgint.jpg",
            read_workspace_file("references/libjpeg-turbo/testimages/testimgint.jpg"),
            ColorType::Rgb8,
            PixelFormat::Rgb,
        ),
        (
            "testimgari.jpg (arithmetic)",
            read_workspace_file("references/libjpeg-turbo/testimages/testimgari.jpg"),
            ColorType::Rgb8,
            PixelFormat::Rgb,
        ),
        (
            "4:2:2 baseline",
            encode(Subsampling::S422, false),
            ColorType::Rgb8,
            PixelFormat::Rgb,
        ),
        (
            "4:4:4 baseline",
            encode(Subsampling::S444, false),
            ColorType::Rgb8,
            PixelFormat::Rgb,
        ),
        (
            "4:2:0 progressive",
            encode(Subsampling::S420, true),
            ColorType::Rgb8,
            PixelFormat::Rgb,
        ),
        (
            "grayscale",
            compress(
                &gray,
                width,
                height,
                PixelFormat::Grayscale,
                85,
                Subsampling::S444,
            )
            .expect("encode gray fixture"),
            ColorType::L8,
            PixelFormat::Grayscale,
        ),
        (
            "CMYK (Adobe, four components)",
            read_workspace_file("tests/fixtures/cmyk_scanner/scanner_64x64.jpg"),
            ColorType::Rgb8,
            PixelFormat::Rgb,
        ),
    ]
}

/// Every case decodes through the trait to exactly what the core decode
/// produces for the same stream and format.
#[test]
fn read_image_matches_the_core_decode() {
    for (label, jpeg, color_type, format) in parity_cases() {
        let decoder: JpegDecoder =
            JpegDecoder::new(&jpeg).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(decoder.color_type(), color_type, "{label}");
        let expected = decompress_to(&jpeg, format).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(
            decoder.dimensions(),
            (expected.width as u32, expected.height as u32),
            "{label}"
        );
        assert_eq!(read_all(decoder), expected.data, "{label}: pixels differ");
    }
}

/// `original_color_type` names the stream's own layout, which `color_type`
/// cannot for a converted four-component stream.
#[test]
fn original_color_type_reports_the_stream_layout() {
    let cmyk: Vec<u8> = read_workspace_file("tests/fixtures/cmyk_scanner/scanner_64x64.jpg");
    let decoder: JpegDecoder = JpegDecoder::new(&cmyk).expect("cmyk");
    assert_eq!(decoder.original_color_type(), ExtendedColorType::Cmyk8);
    assert_eq!(decoder.color_type(), ColorType::Rgb8);
    let gray: Vec<u8> = compress(
        &[90u8; 64],
        8,
        8,
        PixelFormat::Grayscale,
        90,
        Subsampling::S444,
    )
    .expect("gray");
    assert_eq!(
        JpegDecoder::new(&gray).expect("gray").original_color_type(),
        ExtendedColorType::L8
    );
}

/// Locate `djpeg`: the pinned CI oracle first (`/opt/libjpeg-turbo`, which
/// the Integration Tests job installs), then Homebrew, then PATH.
fn djpeg() -> Option<PathBuf> {
    for candidate in ["/opt/libjpeg-turbo/bin/djpeg", "/opt/homebrew/bin/djpeg"] {
        let path: PathBuf = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    let output: std::process::Output = Command::new("which").arg("djpeg").output().ok()?;
    let path: String = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!path.is_empty()).then(|| PathBuf::from(path))
}

/// The binary PPM body (after the three header fields and their whitespace).
fn ppm_pixels(ppm: &[u8]) -> Vec<u8> {
    let mut fields: usize = 0;
    let mut index: usize = 0;
    while fields < 4 {
        while ppm[index].is_ascii_whitespace() {
            index += 1;
        }
        while !ppm[index].is_ascii_whitespace() {
            index += 1;
        }
        fields += 1;
    }
    ppm[index + 1..].to_vec()
}

/// The direct C check: the trait's pixels equal `djpeg -ppm` byte for byte.
#[test]
fn read_image_matches_c_djpeg() {
    let Some(djpeg) = djpeg() else {
        assert!(std::env::var_os("CI").is_none(), "CI requires djpeg");
        eprintln!("SKIP: djpeg not found");
        return;
    };
    let path: PathBuf = workspace_file("references/libjpeg-turbo/testimages/testorig.jpg");
    let output: std::process::Output = Command::new(&djpeg)
        .arg("-ppm")
        .arg(&path)
        .output()
        .expect("run djpeg");
    assert!(output.status.success(), "djpeg failed");
    let decoder: JpegDecoder = JpegDecoder::new(&std::fs::read(&path).expect("read")).expect("new");
    assert_eq!(read_all(decoder), ppm_pixels(&output.stdout));
}

fn jpeg_with_all_metadata(orientation: u16) -> Vec<u8> {
    let (width, height): (usize, usize) = (40, 24);
    let rgb: Vec<u8> = gradient_rgb(width, height);
    let icc: Vec<u8> = read_workspace_file("references/libjpeg-turbo/testimages/test1.icc");
    let exif: Vec<u8> = tiff_with_orientation(orientation);
    let xmp: &[u8] =
        b"<?xpacket begin=\"\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/><?xpacket end=\"w\"?>";
    // One IIM dataset: record 2, tag 120 (caption), length 5.
    let iptc: &[u8] = &[0x1C, 0x02, 0x78, 0x00, 0x05, b'h', b'e', b'l', b'l', b'o'];
    Encoder::new(&rgb, width, height, PixelFormat::Rgb)
        .quality(90)
        .icc_profile(&icc)
        .exif_data(&exif)
        .xmp_data(xmp)
        .iptc_data(iptc)
        .encode()
        .expect("encode with metadata")
}

/// `jpeg` with one more marker segment spliced in right after SOI, so it
/// precedes every segment the encoder wrote.
fn with_segment_after_soi(jpeg: &[u8], marker: u8, payload: &[u8]) -> Vec<u8> {
    let length: u16 = u16::try_from(payload.len() + 2).expect("segment fits");
    let mut out: Vec<u8> = jpeg[..2].to_vec();
    out.extend_from_slice(&[0xFF, marker]);
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(payload);
    out.extend_from_slice(&jpeg[2..]);
    out
}

/// The repeated-segment shapes on which the core's own accessors and
/// `image`'s built-in decoder disagree, so the adapter has to pick a side.
fn edge_case_metadata_streams() -> Vec<(&'static str, Vec<u8>)> {
    let base: Vec<u8> = jpeg_with_all_metadata(6);
    let mut early_exif: Vec<u8> = b"Exif\0\0".to_vec();
    early_exif.extend_from_slice(&tiff_with_orientation(3));
    // An Extended XMP chunk: signature, 32-byte GUID, full length, offset, data.
    let mut extension: Vec<u8> = b"http://ns.adobe.com/xmp/extension/\0".to_vec();
    extension.extend_from_slice(&[b'A'; 32]);
    extension.extend_from_slice(&4u32.to_be_bytes());
    extension.extend_from_slice(&0u32.to_be_bytes());
    extension.extend_from_slice(b"<x/>");
    vec![
        (
            "two EXIF segments: orientation 3 first, 6 last",
            with_segment_after_soi(&base, 0xE1, &early_exif),
        ),
        (
            "an Extended XMP chunk",
            with_segment_after_soi(&base, 0xE1, &extension),
        ),
        (
            "an APP13 with the signature and no payload",
            with_segment_after_soi(&base, 0xED, b"Photoshop 3.0\0"),
        ),
    ]
}

/// Every metadata accessor agrees with `image`'s own JPEG decoder — with
/// metadata present, with none, and on the repeated-segment edge cases.
#[test]
fn metadata_matches_image_builtin_decoder() {
    let plain: Vec<u8> = compress(
        &gradient_rgb(16, 16),
        16,
        16,
        PixelFormat::Rgb,
        80,
        Subsampling::S420,
    )
    .expect("plain");
    let mut streams: Vec<(&'static str, Vec<u8>)> = vec![
        ("all metadata", jpeg_with_all_metadata(6)),
        ("no metadata", plain),
    ];
    streams.extend(edge_case_metadata_streams());
    for (label, jpeg) in streams {
        let mut ours: JpegDecoder = JpegDecoder::new(&jpeg).expect("ours");
        let mut builtin: image::codecs::jpeg::JpegDecoder<Cursor<&Vec<u8>>> =
            image::codecs::jpeg::JpegDecoder::new(Cursor::new(&jpeg)).expect("builtin");
        assert_eq!(
            ours.icc_profile().expect("icc"),
            builtin.icc_profile().expect("icc"),
            "{label}: icc"
        );
        assert_eq!(
            ours.exif_metadata().expect("exif"),
            builtin.exif_metadata().expect("exif"),
            "{label}: exif"
        );
        assert_eq!(
            ours.xmp_metadata().expect("xmp"),
            builtin.xmp_metadata().expect("xmp"),
            "{label}: xmp"
        );
        assert_eq!(
            ours.iptc_metadata().expect("iptc"),
            builtin.iptc_metadata().expect("iptc"),
            "{label}: iptc"
        );
        assert_eq!(
            ours.orientation().expect("orientation"),
            builtin.orientation().expect("orientation"),
            "{label}: orientation"
        );
    }
    let mut ours: JpegDecoder = JpegDecoder::new(&jpeg_with_all_metadata(6)).expect("ours");
    assert!(
        ours.icc_profile().expect("icc").is_some(),
        "the fixture must carry an ICC profile"
    );
    assert!(
        ours.iptc_metadata().expect("iptc").is_some(),
        "the fixture must carry IPTC"
    );
    assert_eq!(
        ours.orientation().expect("orientation"),
        Orientation::Rotate90
    );
}

/// Orientation is reported, not applied: `read_image` returns the stored
/// raster, so the application's one `apply_orientation` rotates it once.
#[test]
fn orientation_is_applied_once_by_the_application() {
    let jpeg: Vec<u8> = jpeg_with_all_metadata(6);
    let mut decoder: JpegDecoder = JpegDecoder::new(&jpeg).expect("new");
    let orientation: Orientation = decoder.orientation().expect("orientation");
    let mut image: DynamicImage = DynamicImage::from_decoder(decoder).expect("from_decoder");
    assert_eq!(
        (image.width(), image.height()),
        (40, 24),
        "stored raster, unrotated"
    );
    let stored: DynamicImage = image.clone();
    image.apply_orientation(orientation);
    assert_eq!((image.width(), image.height()), (24, 40));
    assert_eq!(image, stored.rotate90());
}

fn limit_kind(error: ImageError) -> LimitErrorKind {
    match error {
        ImageError::Limits(limit) => limit.kind(),
        other => panic!("expected ImageError::Limits, got {other:?}"),
    }
}

fn limits(width: Option<u32>, height: Option<u32>, alloc: Option<u64>) -> Limits {
    let mut limits: Limits = Limits::no_limits();
    limits.max_image_width = width;
    limits.max_image_height = height;
    limits.max_alloc = alloc;
    limits
}

/// `set_limits` refuses before any decode, with `image`'s limit categories.
#[test]
fn limits_are_enforced_at_set_limits() {
    let jpeg: Vec<u8> = read_workspace_file("references/libjpeg-turbo/testimages/testorig.jpg");
    let (width, height): (u32, u32) = JpegDecoder::new(&jpeg).expect("new").dimensions();

    let mut decoder: JpegDecoder = JpegDecoder::new(&jpeg).expect("new");
    let refused: ImageError = decoder
        .set_limits(limits(Some(width - 1), None, None))
        .unwrap_err();
    assert_eq!(limit_kind(refused), LimitErrorKind::DimensionError);
    let refused: ImageError = decoder
        .set_limits(limits(None, Some(height - 1), None))
        .unwrap_err();
    assert_eq!(limit_kind(refused), LimitErrorKind::DimensionError);

    // The output alone is width * height * 3 bytes, and max_alloc counts it.
    let output_bytes: u64 = u64::from(width) * u64::from(height) * 3;
    let refused: ImageError = decoder
        .set_limits(limits(None, None, Some(output_bytes - 1)))
        .unwrap_err();
    assert_eq!(limit_kind(refused), LimitErrorKind::InsufficientMemory);

    // Refusals left the decoder usable with its previous (default) limits.
    let expected: Vec<u8> = decompress_to(&jpeg, PixelFormat::Rgb).expect("core").data;
    assert_eq!(read_all(decoder), expected);

    // Exact dimensions and a generous budget are accepted and decode.
    let mut decoder: JpegDecoder = JpegDecoder::new(&jpeg).expect("new");
    decoder
        .set_limits(limits(Some(width), Some(height), Some(64 * 1024 * 1024)))
        .expect("limits that fit");
    assert_eq!(read_all(decoder), expected);
}

/// `image::Limits::default()` (512 MiB `max_alloc`) refuses a header bomb at
/// `set_limits`, from the header alone: a 16384 x 16384 frame with no scan
/// data needs more than 1 GiB by the core's estimate. (`read_image` runs the
/// same checks again, but only after the caller has allocated a destination of
/// that size, so that path is structural, not something a test can drive.)
#[test]
fn default_limits_refuse_a_header_bomb_at_set_limits() {
    // A 4:4:4 baseline header for 16384 x 16384 with no scan data: the
    // memory estimate (> 1 GiB) is computed from the header alone.
    let small: Vec<u8> = compress(
        &gradient_rgb(8, 8),
        8,
        8,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
    )
    .expect("small");
    let sof: usize = small
        .windows(2)
        .position(|pair: &[u8]| pair == [0xFF, 0xC0])
        .expect("SOF0");
    let mut huge: Vec<u8> = small.clone();
    huge[sof + 5..sof + 7].copy_from_slice(&16384u16.to_be_bytes());
    huge[sof + 7..sof + 9].copy_from_slice(&16384u16.to_be_bytes());

    let mut decoder: JpegDecoder = JpegDecoder::new(&huge).expect("header parses");
    assert_eq!(decoder.dimensions(), (16384, 16384));
    let refused: ImageError = decoder.set_limits(Limits::default()).unwrap_err();
    assert_eq!(limit_kind(refused), LimitErrorKind::InsufficientMemory);
}

fn decoding_error(error: ImageError) -> String {
    match error {
        ImageError::Decoding(decoding) => decoding.to_string(),
        other => panic!("expected ImageError::Decoding, got {other:?}"),
    }
}

/// Malformed and truncated input are decoding errors, never panics; the
/// lenient switch turns the truncated case into a filled image.
#[test]
fn malformed_and_truncated_input() {
    let not_jpeg: Vec<u8> = b"definitely not a JPEG stream".to_vec();
    let refused: ImageError = JpegDecoder::new(&not_jpeg).err().expect("must refuse");
    decoding_error(refused);

    let jpeg: Vec<u8> = read_workspace_file("references/libjpeg-turbo/testimages/testorig.jpg");
    let truncated: Vec<u8> = jpeg[..jpeg.len() * 2 / 3].to_vec();
    let decoder: JpegDecoder = JpegDecoder::new(&truncated).expect("the header is intact");
    let mut pixels: Vec<u8> = vec![0u8; decoder.total_bytes() as usize];
    decoding_error(decoder.read_image(&mut pixels).unwrap_err());

    let mut decoder: JpegDecoder = JpegDecoder::new(&truncated).expect("header");
    decoder.set_lenient(true);
    let mut pixels: Vec<u8> = vec![0u8; decoder.total_bytes() as usize];
    decoder
        .read_image(&mut pixels)
        .expect("lenient decode fills the rest");
    let full: Vec<u8> = decompress_to(&jpeg, PixelFormat::Rgb).expect("core").data;
    let row_bytes: usize = JpegDecoder::new(&jpeg).expect("new").dimensions().0 as usize * 3;
    assert_eq!(
        pixels[..row_bytes * 8],
        full[..row_bytes * 8],
        "the decoded top rows match"
    );
}

/// The destination must be exactly `total_bytes()` long.
#[test]
fn destination_size_must_match_total_bytes() {
    let jpeg: Vec<u8> = read_workspace_file("references/libjpeg-turbo/testimages/testorig.jpg");
    for delta in [-1i64, 1] {
        let decoder: JpegDecoder = JpegDecoder::new(&jpeg).expect("new");
        let mut pixels: Vec<u8> = vec![0u8; (decoder.total_bytes() as i64 + delta) as usize];
        match decoder.read_image(&mut pixels).unwrap_err() {
            ImageError::Parameter(parameter) => {
                assert_eq!(parameter.kind(), ParameterErrorKind::DimensionMismatch)
            }
            other => panic!("delta {delta}: expected a parameter error, got {other:?}"),
        }
    }
}

/// Formats `image` cannot represent are refused at construction.
#[test]
fn unrepresentable_output_formats_are_unsupported() {
    let jpeg: Vec<u8> = read_workspace_file("references/libjpeg-turbo/testimages/testorig.jpg");
    for format in [PixelFormat::Bgr, PixelFormat::Bgra, PixelFormat::Cmyk] {
        match JpegDecoder::new_with_format(&jpeg, format)
            .err()
            .expect("must refuse")
        {
            ImageError::Unsupported(_) => {}
            other => panic!("{format:?}: expected Unsupported, got {other:?}"),
        }
    }
    let rgba: JpegDecoder = JpegDecoder::new_with_format(&jpeg, PixelFormat::Rgba).expect("rgba");
    assert_eq!(rgba.color_type(), ColorType::Rgba8);
    let expected: Vec<u8> = decompress_to(&jpeg, PixelFormat::Rgba).expect("core").data;
    assert_eq!(read_all(rgba), expected);
}

/// The core has no CMYK -> grayscale conversion, so asking for `L8` from a
/// four-component stream is refused at construction, not in `read_image`.
#[test]
fn grayscale_from_a_four_component_stream_is_refused_at_construction() {
    let cmyk: Vec<u8> = read_workspace_file("tests/fixtures/cmyk_scanner/scanner_64x64.jpg");
    match JpegDecoder::new_with_format(&cmyk, PixelFormat::Grayscale)
        .err()
        .expect("must refuse")
    {
        ImageError::Unsupported(_) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}

/// `from_vec` and `new` decode the same stream identically.
#[test]
fn from_vec_matches_new() {
    let jpeg: Vec<u8> = read_workspace_file("references/libjpeg-turbo/testimages/testimgint.jpg");
    let borrowed: Vec<u8> = read_all(JpegDecoder::new(&jpeg).expect("new"));
    let owned: Vec<u8> = read_all(JpegDecoder::from_vec(jpeg.clone()).expect("from_vec"));
    assert_eq!(borrowed, owned);
}

/// The encoder through `DynamicImage::write_with_encoder` writes exactly what
/// the core `compress` writes for the same pixels and settings, and a short
/// buffer is a parameter error.
#[test]
fn encoder_through_the_trait_matches_core_compress() {
    let (width, height): (usize, usize) = (48, 32);
    let rgb: Vec<u8> = gradient_rgb(width, height);
    let image: DynamicImage = DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(width as u32, height as u32, rgb.clone()).expect("raw"),
    );
    let mut through_trait: Vec<u8> = Vec::new();
    let mut encoder: JpegEncoder<&mut Vec<u8>> =
        JpegEncoder::new_with_quality(&mut through_trait, 82);
    encoder.set_subsampling(Subsampling::S422);
    image.write_with_encoder(encoder).expect("encode");
    let direct: Vec<u8> =
        compress(&rgb, width, height, PixelFormat::Rgb, 82, Subsampling::S422).expect("core");
    assert_eq!(through_trait, direct);

    // `write_image` takes exactly width * height pixels: one byte short and
    // one byte over are both parameter errors.
    let mut longer: Vec<u8> = rgb.clone();
    longer.push(0);
    for buffer in [&rgb[..rgb.len() - 1], &longer[..]] {
        let mut sink: Vec<u8> = Vec::new();
        let refused: ImageError = JpegEncoder::new(&mut sink)
            .write_image(buffer, width as u32, height as u32, ExtendedColorType::Rgb8)
            .unwrap_err();
        match refused {
            ImageError::Parameter(parameter) => {
                assert_eq!(parameter.kind(), ParameterErrorKind::DimensionMismatch)
            }
            other => panic!(
                "{} bytes: expected a parameter error, got {other:?}",
                buffer.len()
            ),
        }
    }
}
