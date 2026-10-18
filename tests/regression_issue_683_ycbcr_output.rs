//! Issue #683 (P4-244): `Decoder::set_output_colorspace(ColorSpace::YCbCr)`
//! must return what libjpeg returns for `out_color_space = JCS_YCbCr`.
//!
//! libjpeg chooses its upsampler from the component geometry alone
//! (`jdsample.c::jinit_upsampler`), independent of the output colour space,
//! and then `jdcolor.c::null_convert` copies the planes out. YCbCr output is
//! therefore the RGB pipeline stopped before the colour transform. The Rust
//! override used to bypass the upsampler: it replicated chroma, divided by
//! zero when chroma out-samples luma, and read Cr with Cb's geometry. Neither
//! `djpeg` nor TurboJPEG can request YCbCr output, so the reference is
//! `examples/ycbcr_decode_c_oracle.c`, which drives libjpeg directly.

mod helpers;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use helpers::c_oracle::{decode_with_ycbcr_c_oracle, ycbcr_decode_c_oracle, CYcbcrRequest};
use libjpeg_turbo_rs::{
    ColorSpace, Decoder, Encoder, Image, JpegError, PixelFormat, ScalingFactor, ScanlineDecoder,
};

/// Odd dimensions, so every layout has a partial MCU at the right and bottom
/// edges and the upsamplers' edge cases are exercised.
const WIDTH: usize = 61;
const HEIGHT: usize = 45;

/// Layouts a strict decode accepts: the standard subsamplings, a 3:1 ratio
/// that only libjpeg's generic `int_upsample` handles, and two where Cb and
/// Cr are sampled differently.
const STRICT_LAYOUTS: &[&str] = &[
    "1x1",
    "2x1",
    "2x2",
    "1x2",
    "4x1",
    "4x2",
    "3x1",
    "2x2,2x1,1x1",
    "2x2,1x2,2x1",
];

/// Legal T.81 layouts in which a chroma component out-samples luma in at
/// least one direction. P4-21 refuses them unless the decode is lenient;
/// libjpeg decodes them.
const CHROMA_OUTSAMPLES_LUMA_LAYOUTS: &[&str] = &["1x1,2x2,2x2", "1x1,2x2,1x1", "2x1,1x2,1x1"];

/// Smooth gradients plus a fine texture, so interpolated and replicated chroma
/// disagree on almost every sample.
fn textured_rgb(width: usize, height: usize) -> Vec<u8> {
    (0..width * height)
        .flat_map(|index| {
            let x: usize = index % width;
            let y: usize = index / width;
            [
                ((x * 255 / (width - 1) + (x * y) % 7 * 9) % 256) as u8,
                ((y * 255 / (height - 1) + (x + y) % 5 * 11) % 256) as u8,
                (((x + 2 * y) * 3 + (x * 3) % 13 * 7) % 256) as u8,
            ]
        })
        .collect()
}

fn encode_with_cjpeg(cjpeg: &Path, extra_args: &[&str], label: &str) -> Vec<u8> {
    let ppm: Vec<u8> = helpers::build_ppm(&textured_rgb(WIDTH, HEIGHT), WIDTH, HEIGHT);
    let mut args: Vec<&str> = vec!["-quality", "92"];
    args.extend_from_slice(extra_args);
    helpers::encode_with_c_cjpeg(cjpeg, &ppm, &args, label)
}

fn encode_sampled(cjpeg: &Path, sampling: &str) -> Vec<u8> {
    encode_with_cjpeg(
        cjpeg,
        &["-sample", sampling],
        &format!("issue683_{}", sampling.replace(',', "_")),
    )
}

/// The YCbCr oracle, or `None` (after a local SKIP) when no libjpeg
/// development install can build it. CI provisions one, so there a missing
/// oracle fails the test.
fn ycbcr_oracle() -> Option<PathBuf> {
    let oracle: Option<PathBuf> = ycbcr_decode_c_oracle();
    if oracle.is_none() {
        assert!(
            !helpers::is_ci(),
            "CI provisions a libjpeg development install, so the YCbCr oracle \
             must be buildable there"
        );
        eprintln!(
            "SKIP: no libjpeg development install (headers + library) found; \
             cannot build the YCbCr oracle"
        );
    }
    oracle
}

fn request(scale: ScalingFactor, fancy: bool) -> CYcbcrRequest {
    CYcbcrRequest {
        scale: (scale.num(), scale.denom()),
        fancy,
        crop: None,
    }
}

fn configured_decoder<'a>(
    jpeg: &'a [u8],
    request: &CYcbcrRequest,
    lenient: bool,
) -> libjpeg_turbo_rs::Result<Decoder<'a>> {
    let mut decoder: Decoder<'a> = Decoder::new(jpeg)?;
    decoder.set_scale(ScalingFactor::try_new(request.scale.0, request.scale.1)?);
    decoder.set_fast_upsample(!request.fancy);
    decoder.set_lenient(lenient);
    if let Some((x, y, width, height)) = request.crop {
        decoder.set_crop_region(x, y, width, height);
    }
    decoder.set_output_colorspace(ColorSpace::YCbCr);
    Ok(decoder)
}

/// Decode with YCbCr output; `Err` carries the panic message if the decoder
/// panicked, so every caller can assert "never panics" as well as the result.
fn rust_ycbcr(
    jpeg: &[u8],
    request: &CYcbcrRequest,
    lenient: bool,
) -> Result<libjpeg_turbo_rs::Result<Image>, String> {
    catch_unwind(AssertUnwindSafe(|| {
        configured_decoder(jpeg, request, lenient)?.decode_image()
    }))
    .map_err(|payload| {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|text| text.to_string()))
            .unwrap_or_else(|| "non-string panic payload".to_string())
    })
}

/// Byte-for-byte comparison that names the channel and position of the first
/// differences, since a chroma-only defect leaves Y identical.
fn assert_ycbcr_identical(rust: &[u8], c: &[u8], width: usize, label: &str) {
    assert_eq!(rust.len(), c.len(), "{label}: output length");
    let mut mismatches: [usize; 3] = [0; 3];
    let mut max_diff: [u8; 3] = [0; 3];
    let mut first: Vec<String> = Vec::new();
    for (index, (&ours, &theirs)) in rust.iter().zip(c).enumerate() {
        if ours != theirs {
            let channel: usize = index % 3;
            mismatches[channel] += 1;
            max_diff[channel] = max_diff[channel].max(ours.abs_diff(theirs));
            if first.len() < 5 {
                let pixel: usize = index / 3;
                first.push(format!(
                    "({}, {}) {}: rust {ours} vs libjpeg {theirs}",
                    pixel % width,
                    pixel / width,
                    ["Y", "Cb", "Cr"][channel]
                ));
            }
        }
    }
    assert_eq!(
        mismatches, [0; 3],
        "{label}: Y/Cb/Cr samples differing from libjpeg {mismatches:?} (max {max_diff:?}); \
         first: {first:?}"
    );
}

fn assert_matches_libjpeg(
    oracle: &Path,
    jpeg: &[u8],
    request: &CYcbcrRequest,
    lenient: bool,
    label: &str,
) {
    let (c_width, c_height, c_samples) = decode_with_ycbcr_c_oracle(oracle, jpeg, request)
        .unwrap_or_else(|stderr| panic!("{label}: libjpeg refused the decode: {stderr}"));
    let image: Image = rust_ycbcr(jpeg, request, lenient)
        .unwrap_or_else(|message| panic!("{label}: YCbCr decode panicked: {message}"))
        .unwrap_or_else(|error| panic!("{label}: YCbCr decode failed: {error}"));
    assert_eq!(
        (image.width, image.height),
        (c_width, c_height),
        "{label}: dimensions"
    );
    // Three samples per pixel. `PixelFormat` has no YCbCr variant, and adding
    // one is a semver-major change, so the override keeps the 3-byte `Rgb`
    // label and documents that the bytes are Y, Cb, Cr.
    assert_eq!(
        image.pixel_format,
        PixelFormat::Rgb,
        "{label}: pixel format"
    );
    assert_ycbcr_identical(&image.data, &c_samples, c_width, label);
}

/// Issue #683: every strict layout, at every scale libjpeg supports, with
/// fancy and with box upsampling, is byte-identical to libjpeg's YCbCr output.
/// Before the fix 4:2:2, 4:2:0 and 4:4:0 differed on 85–95 % of chroma
/// samples, and the Cb ≠ Cr layouts panicked or returned wrong Cr.
#[test]
fn ycbcr_output_matches_libjpeg_for_every_layout_scale_and_upsampler() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let Some(oracle) = ycbcr_oracle() else {
        return;
    };
    let mut tally = helpers::ComparisonTally::new(
        "ycbcr_output_matches_libjpeg_for_every_layout_scale_and_upsampler",
        STRICT_LAYOUTS.len() * ScalingFactor::SUPPORTED.len() * 2,
    );
    for &sampling in STRICT_LAYOUTS {
        let jpeg: Vec<u8> = encode_sampled(&cjpeg, sampling);
        for scale in ScalingFactor::SUPPORTED {
            for fancy in [true, false] {
                let label: String = format!(
                    "-sample {sampling} at {}/{} fancy={fancy}",
                    scale.num(),
                    scale.denom()
                );
                assert_matches_libjpeg(&oracle, &jpeg, &request(scale, fancy), false, &label);
                tally.compared();
            }
        }
    }
    tally.finish();
}

/// Issue #683: luma sampled below chroma used to panic with "attempt to
/// divide by zero". Strict decodes keep P4-21's refusal, as RGB output does;
/// lenient decodes upsample every component, luma included, like libjpeg.
#[test]
fn ycbcr_output_when_chroma_outsamples_luma_never_panics() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let Some(oracle) = ycbcr_oracle() else {
        return;
    };
    let mut tally = helpers::ComparisonTally::new(
        "ycbcr_output_when_chroma_outsamples_luma_never_panics",
        CHROMA_OUTSAMPLES_LUMA_LAYOUTS.len() * ScalingFactor::SUPPORTED.len() * 2,
    );
    for &sampling in CHROMA_OUTSAMPLES_LUMA_LAYOUTS {
        let jpeg: Vec<u8> = encode_sampled(&cjpeg, sampling);
        let full_scale: CYcbcrRequest = request(ScalingFactor::try_new(1, 1).unwrap(), true);
        match rust_ycbcr(&jpeg, &full_scale, false) {
            Err(message) => panic!("strict -sample {sampling} panicked: {message}"),
            Ok(Err(JpegError::CorruptData(message))) => assert!(
                message.contains("chroma upsample factor zero")
                    && message.contains("out-samples luma"),
                "strict -sample {sampling}: wrong corruption reason: {message}"
            ),
            Ok(Err(error)) => panic!("strict -sample {sampling}: wrong error: {error}"),
            Ok(Ok(_)) => panic!("strict -sample {sampling} must keep the P4-21 refusal"),
        }
        let strict_rgb = Decoder::new(&jpeg)
            .expect("header")
            .decode_image()
            .expect_err("strict RGB output refuses the same layout");
        assert!(
            matches!(strict_rgb, JpegError::CorruptData(_)),
            "strict RGB and YCbCr output must refuse alike, got {strict_rgb}"
        );
        for scale in ScalingFactor::SUPPORTED {
            for fancy in [true, false] {
                let label: String = format!(
                    "lenient -sample {sampling} at {}/{} fancy={fancy}",
                    scale.num(),
                    scale.denom()
                );
                assert_matches_libjpeg(&oracle, &jpeg, &request(scale, fancy), true, &label);
                tally.compared();
            }
        }
    }
    tally.finish();
}

/// Crop regions in the output's own coordinates, chosen to hit an unaligned
/// start, a start past the first iMCU column, the right and bottom edges,
/// single rows and columns, and crops that leave a 2:1 chroma component 2 and
/// 1 samples wide — `jpeg_crop_scanline` keeps the fancy filter for the
/// first and re-initialises the upsampler to the box filter for the second.
fn crop_regions(width: usize, height: usize) -> Vec<(usize, usize, usize, usize)> {
    vec![
        (width / 7, height / 5, width / 2, height / 2),
        (
            width / 3 + 1,
            height / 3,
            width - (width / 3 + 1),
            height - height / 3,
        ),
        (width - 1, height - 1, 1, 1),
        (2, 0, 1, height),
        (1, 1, 1, 1),
        (0, height / 2, width, 1),
    ]
}

/// Issue #683: the cropped path used to shift cloned planes and then
/// replicate chroma; it must match `jpeg_crop_scanline` + `jpeg_skip_scanlines`
/// in libjpeg, including the fancy filters' context at the crop edges.
#[test]
fn ycbcr_output_with_crop_matches_libjpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let Some(oracle) = ycbcr_oracle() else {
        return;
    };
    // `4x1,2x1,1x1` and `4x1,1x1,2x1`: a narrow crop takes the 4:1 component
    // below two samples while the 2:1 one keeps two, and libjpeg's
    // re-initialisation then drops the 2:1 component to the box filter too.
    let layouts: &[&str] = &[
        "2x1",
        "2x2",
        "1x2",
        "4x1",
        "2x2,2x1,1x1",
        "4x1,2x1,1x1",
        "4x1,1x1,2x1",
    ];
    let scales: [ScalingFactor; 3] = [
        ScalingFactor::try_new(1, 1).unwrap(),
        ScalingFactor::try_new(1, 2).unwrap(),
        ScalingFactor::try_new(3, 4).unwrap(),
    ];
    let regions_per_case: usize = crop_regions(WIDTH, HEIGHT).len();
    let mut tally = helpers::ComparisonTally::new(
        "ycbcr_output_with_crop_matches_libjpeg",
        layouts.len() * scales.len() * 2 * regions_per_case,
    );
    for &sampling in layouts {
        let jpeg: Vec<u8> = encode_sampled(&cjpeg, sampling);
        for scale in scales {
            for fancy in [true, false] {
                let full: CYcbcrRequest = request(scale, fancy);
                let (out_width, out_height, _) = decode_with_ycbcr_c_oracle(&oracle, &jpeg, &full)
                    .unwrap_or_else(|stderr| panic!("libjpeg refused {sampling}: {stderr}"));
                let regions = crop_regions(out_width, out_height);
                assert_eq!(regions.len(), regions_per_case);
                for crop in regions {
                    let label: String = format!(
                        "-sample {sampling} at {}/{} fancy={fancy} crop {crop:?}",
                        scale.num(),
                        scale.denom()
                    );
                    let cropped = CYcbcrRequest {
                        crop: Some(crop),
                        ..request(scale, fancy)
                    };
                    assert_matches_libjpeg(&oracle, &jpeg, &cropped, false, &label);
                    tally.compared();
                }
            }
        }
    }
    tally.finish();
}

/// `ScanlineDecoder::set_output_colorspace` reaches the same decode as
/// `Decoder`, so it too returns libjpeg's YCbCr output.
#[test]
fn scanline_decoder_ycbcr_output_matches_libjpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let Some(oracle) = ycbcr_oracle() else {
        return;
    };
    for sampling in ["2x1", "2x2"] {
        let jpeg: Vec<u8> = encode_sampled(&cjpeg, sampling);
        let full: CYcbcrRequest = request(ScalingFactor::try_new(1, 1).unwrap(), true);
        let (c_width, c_height, c_samples) = decode_with_ycbcr_c_oracle(&oracle, &jpeg, &full)
            .unwrap_or_else(|stderr| panic!("libjpeg refused {sampling}: {stderr}"));
        let mut scanline: ScanlineDecoder = ScanlineDecoder::new(&jpeg).expect("header");
        scanline.set_output_colorspace(ColorSpace::YCbCr);
        let image: Image = scanline.finish().expect("scanline YCbCr decode");
        assert_eq!((image.width, image.height), (c_width, c_height));
        assert_ycbcr_identical(
            &image.data,
            &c_samples,
            c_width,
            &format!("ScanlineDecoder -sample {sampling}"),
        );
    }
}

/// libjpeg permits YCbCr output only as a null conversion from a YCbCr
/// source (`jdcolor.c`, default case); anything else is
/// `JERR_CONVERSION_NOTIMPL`. The override used to hand back an RGB or
/// CMYK/YCCK source's samples relabelled as YCbCr.
#[test]
fn ycbcr_output_refuses_non_ycbcr_sources_like_libjpeg() {
    let cjpeg: PathBuf = require_c_tool!("cjpeg");
    let Some(oracle) = ycbcr_oracle() else {
        return;
    };
    // CMYK and YCCK come from this crate's encoder (cjpeg cannot read
    // four-channel input); libjpeg's refusal below is the reference.
    let cmyk_pixels: Vec<u8> = (0..WIDTH * HEIGHT * 4)
        .map(|index| (index * 7 % 251) as u8)
        .collect();
    let cmyk: Vec<u8> = Encoder::new(&cmyk_pixels, WIDTH, HEIGHT, PixelFormat::Cmyk)
        .quality(92)
        .encode()
        .expect("encode CMYK");
    let ycck: Vec<u8> = Encoder::new(&cmyk_pixels, WIDTH, HEIGHT, PixelFormat::Cmyk)
        .quality(92)
        .colorspace(ColorSpace::Ycck)
        .encode()
        .expect("encode YCCK");
    let sources: [(&str, Vec<u8>); 5] = [
        (
            "RGB (cjpeg -rgb)",
            encode_with_cjpeg(&cjpeg, &["-rgb"], "issue683_rgb_source"),
        ),
        (
            "RGB, 2x2 sampled (cjpeg -rgb -sample 2x2)",
            encode_with_cjpeg(
                &cjpeg,
                &["-rgb", "-sample", "2x2"],
                "issue683_rgb_source_420",
            ),
        ),
        (
            "grayscale (cjpeg -grayscale)",
            encode_with_cjpeg(&cjpeg, &["-grayscale"], "issue683_gray_source"),
        ),
        ("CMYK", cmyk),
        ("YCCK", ycck),
    ];
    for (name, jpeg) in sources {
        let full: CYcbcrRequest = request(ScalingFactor::try_new(1, 1).unwrap(), true);
        let c_error: String = decode_with_ycbcr_c_oracle(&oracle, &jpeg, &full)
            .err()
            .unwrap_or_else(|| panic!("libjpeg must refuse YCbCr output from a {name} source"));
        assert!(
            c_error.contains("Unsupported color conversion request"),
            "{name}: libjpeg refused for another reason: {c_error}"
        );
        match rust_ycbcr(&jpeg, &full, false) {
            Err(message) => panic!("{name}: YCbCr decode panicked: {message}"),
            Ok(Err(JpegError::Unsupported(message))) => assert!(
                message.contains("JERR_CONVERSION_NOTIMPL"),
                "{name}: refusal must name libjpeg's error: {message}"
            ),
            Ok(Err(error)) => panic!("{name}: wrong error: {error}"),
            Ok(Ok(_)) => panic!("{name}: YCbCr output must be refused, as libjpeg refuses it"),
        }
    }
}

/// Call the public legacy `decode_with_colorspace_override` YCbCr arm on
/// hand-built planes, one MCU column wide, catching any panic.
fn legacy_ycbcr(
    sampling: [(u8, u8); 3],
    planes: &[Vec<u8>],
    out_width: usize,
    out_height: usize,
) -> Result<libjpeg_turbo_rs::Result<Image>, String> {
    legacy_override(ColorSpace::YCbCr, &sampling, planes, out_width, out_height)
}

fn legacy_override(
    target: ColorSpace,
    sampling: &[(u8, u8)],
    planes: &[Vec<u8>],
    out_width: usize,
    out_height: usize,
) -> Result<libjpeg_turbo_rs::Result<Image>, String> {
    let frame = libjpeg_turbo_rs::FrameHeader {
        precision: 8,
        height: out_height as u16,
        width: out_width as u16,
        components: sampling
            .iter()
            .enumerate()
            .map(|(index, &(h, v))| libjpeg_turbo_rs::ComponentInfo {
                id: index as u8 + 1,
                horizontal_sampling: h,
                vertical_sampling: v,
                quant_table_index: 0,
            })
            .collect(),
        is_progressive: false,
        is_lossless: false,
    };
    catch_unwind(AssertUnwindSafe(|| {
        libjpeg_turbo_rs::decode::toggles::decode_with_colorspace_override(
            target,
            planes,
            &frame,
            out_width,
            out_height,
            1,
            &[8, 8, 8],
            None,
            None,
            None,
            None,
            None,
            libjpeg_turbo_rs::DensityInfo::default(),
            Vec::new(),
            Vec::new(),
        )
    }))
    .map_err(|_| "panicked".to_string())
}

fn assert_legacy_refuses(result: Result<libjpeg_turbo_rs::Result<Image>, String>, case: &str) {
    match result {
        Err(message) => panic!("legacy override {case}: {message}"),
        Ok(Err(JpegError::Unsupported(message))) => assert!(
            message.contains("legacy YCbCr override") || message.contains("3+ components"),
            "legacy override {case}: unexpected refusal {message}"
        ),
        Ok(Err(error)) => panic!("legacy override {case}: wrong error {error}"),
        Ok(Ok(_)) => panic!("legacy override {case}: must be refused"),
    }
}

/// Issue #683 / P4-80: the public low-level override must refuse, never
/// panic on, geometry its replicating YCbCr arm cannot handle. It used to
/// divide by zero when chroma out-sampled luma and index out of bounds when
/// Cb and Cr differed or a plane was short.
#[test]
fn legacy_ycbcr_override_refuses_unsupported_geometry_without_panicking() {
    let plane = |len: usize| vec![128u8; len];
    // Luma 1x1, chroma 2x2: luma 8 wide, chroma 16 wide.
    assert_legacy_refuses(
        legacy_ycbcr(
            [(1, 1), (2, 2), (2, 2)],
            &[plane(64), plane(256), plane(256)],
            8,
            8,
        ),
        "with chroma sampled above luma",
    );
    // Luma 2x2, Cb 2x1, Cr 1x1: Cr is 8 wide, Cb 16.
    assert_legacy_refuses(
        legacy_ycbcr(
            [(2, 2), (2, 1), (1, 1)],
            &[plane(256), plane(128), plane(64)],
            16,
            16,
        ),
        "with Cb and Cr sampled differently",
    );
    // A luma plane shorter than the output.
    assert_legacy_refuses(
        legacy_ycbcr(
            [(2, 2), (1, 1), (1, 1)],
            &[plane(100), plane(64), plane(64)],
            16,
            16,
        ),
        "with a short luma plane",
    );
    // Chroma planes without a single full row.
    assert_legacy_refuses(
        legacy_ycbcr(
            [(2, 2), (1, 1), (1, 1)],
            &[plane(256), plane(4), plane(64)],
            16,
            16,
        ),
        "with a chroma plane shorter than one row",
    );
    // Fewer planes than the frame has components.
    assert_legacy_refuses(
        legacy_ycbcr([(2, 2), (1, 1), (1, 1)], &[plane(256), plane(64)], 16, 16),
        "with a missing plane",
    );
    // The grayscale arm, given no component at all.
    match legacy_override(ColorSpace::Grayscale, &[], &[], 8, 8) {
        Err(message) => panic!("legacy grayscale override with no component: {message}"),
        Ok(Err(JpegError::Unsupported(_))) => {}
        Ok(Err(error)) => {
            panic!("legacy grayscale override with no component: wrong error {error}")
        }
        Ok(Ok(_)) => panic!("legacy grayscale override with no component must be refused"),
    }
}

/// The legacy arm keeps its documented behaviour for geometry it supports:
/// each chroma sample is replicated across the luma samples it covers. Only
/// `Decoder` upsamples like libjpeg.
#[test]
fn legacy_ycbcr_override_still_replicates_supported_chroma() {
    let luma: Vec<u8> = (0..256).map(|index| index as u8).collect();
    let cb: Vec<u8> = (0..64).map(|index| 100 + index as u8).collect();
    let cr: Vec<u8> = (0..64).map(|index| 200 - index as u8).collect();
    let image: Image = legacy_ycbcr(
        [(2, 2), (1, 1), (1, 1)],
        &[luma.clone(), cb.clone(), cr.clone()],
        16,
        16,
    )
    .expect("no panic")
    .expect("4:2:0 is supported");
    assert_eq!((image.width, image.height), (16, 16));
    for y in 0..16 {
        for x in 0..16 {
            let pixel: &[u8] = &image.data[(y * 16 + x) * 3..][..3];
            let chroma: usize = (y / 2) * 8 + x / 2;
            assert_eq!(
                pixel,
                [luma[y * 16 + x], cb[chroma], cr[chroma]],
                "({x}, {y})"
            );
        }
    }
}
