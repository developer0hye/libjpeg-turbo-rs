//! P4-197 (#618): a cropping region that does not fit inside the *scaled*
//! output is refused rather than decoded to a degenerate image.
//!
//! The oracle is C `djpeg -crop`, which refuses exactly this shape after
//! `jpeg_start_decompress` has fixed the scaled output size
//! (`references/libjpeg-turbo/src/djpeg.c:854-858`: `crop_x + crop_width >
//! output_width || crop_y + crop_height > output_height`). TurboJPEG's
//! `tj3SetCroppingRegion` applies the same bound (`turbojpeg.c:2106-2109`), and
//! its message is the one `Decoder` reports. The C-ABI half — set-time
//! validation, the iMCU-divisibility rule and the error strings — is
//! cross-validated against stock `libturbojpeg` by the `cropping_region` case of
//! `crates/libjpeg-turbo-rs-capi/examples/cabi_misuse_harness.c`.

mod helpers;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use libjpeg_turbo_rs::{Decoder, Image, JpegError, PixelFormat, ScalingFactor};

/// `tj3SetCroppingRegion`'s refusal, verbatim (`turbojpeg.c:2109`).
const UPSTREAM_MESSAGE: &str = "The cropping region exceeds the scaled image dimensions";

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> Vec<u8> {
    let path: PathBuf = fixture_path(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

/// The crop error's reason, or a panic naming what came back instead.
fn crop_refusal(result: Result<Image, JpegError>, label: &str) -> String {
    match result {
        Err(JpegError::InvalidCropRegion { reason }) => reason,
        Err(other) => panic!("{label}: expected InvalidCropRegion, got {other:?}"),
        Ok(image) => panic!(
            "{label}: expected InvalidCropRegion, got Ok({}x{}, {} bytes)",
            image.width,
            image.height,
            image.data.len()
        ),
    }
}

/// `djpeg [-scale N/D] -crop WxH+X+Y -pnm`: `Some((width, height, samples))`
/// when it decodes, `None` when it refuses the region. A refusal for any other
/// reason is a test failure, so the comparison cannot pass on an unrelated
/// error.
fn djpeg_crop(
    djpeg: &Path,
    jpeg: &Path,
    scale: Option<(u32, u32)>,
    region: (usize, usize, usize, usize),
) -> Option<(usize, usize, Vec<u8>)> {
    let (x, y, w, h): (usize, usize, usize, usize) = region;
    let out: helpers::TempFile = helpers::TempFile::new("crop_region_bounds.pnm");
    let mut command: Command = Command::new(djpeg);
    if let Some((num, denom)) = scale {
        command.arg("-scale").arg(format!("{num}/{denom}"));
    }
    let output: Output = command
        .arg("-crop")
        .arg(format!("{w}x{h}+{x}+{y}"))
        .arg("-pnm")
        .arg("-outfile")
        .arg(out.path())
        .arg(jpeg)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {djpeg:?}: {e}"));
    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(
            stderr.contains("crop dimensions exceed image dimensions"),
            "djpeg refused {w}x{h}+{x}+{y} on {jpeg:?} for a reason other than \
             the crop bound: {stderr}"
        );
        return None;
    }
    let raw: Vec<u8> = std::fs::read(out.path()).expect("read djpeg output");
    let parsed: Option<(usize, usize, Vec<u8>)> =
        helpers::parse_ppm(&raw).or_else(|| helpers::parse_pgm(&raw));
    Some(parsed.unwrap_or_else(|| panic!("djpeg wrote an unparseable PNM for {jpeg:?}")))
}

fn rust_crop(
    jpeg: &[u8],
    scale: Option<(u32, u32)>,
    region: (usize, usize, usize, usize),
) -> Result<Image, JpegError> {
    let (x, y, w, h): (usize, usize, usize, usize) = region;
    let mut decoder: Decoder<'_> = Decoder::new(jpeg).expect("fixture header parses");
    if let Some((num, denom)) = scale {
        decoder.set_scale(ScalingFactor::new(num, denom));
    }
    decoder.set_crop_region(x, y, w, h);
    decoder.decode_image()
}

/// Issue #618: `set_crop_region(10, 1, 16, 2)` on an 8x8 image put the left
/// boundary past the right edge. The decode aligned it down to column 8,
/// clamped the width to the zero columns left, and then either tripped a
/// `debug_assert!` whose comment called empty output unreachable (debug) or
/// returned `Ok` with `width = 0`, `height = 8` and the requested crop height
/// silently dropped (release). It must be refused, in both profiles, by every
/// entry point that can see the region.
#[test]
fn issue_618_left_boundary_past_scaled_width_is_refused() {
    let jpeg: Vec<u8> = read_fixture("gray_8x8.jpg");
    let mut decoder: Decoder<'_> = Decoder::new(&jpeg).expect("fixture header parses");
    decoder.set_crop_region(10, 1, 16, 2);

    let reason: String = crop_refusal(decoder.decode_image(), "decode_image");
    assert_eq!(reason, UPSTREAM_MESSAGE);

    let mut out: Vec<u8> = vec![0u8; 4096];
    match decoder.decode_image_into(&mut out) {
        Err(JpegError::InvalidCropRegion { reason }) => assert_eq!(reason, UPSTREAM_MESSAGE),
        other => panic!("decode_image_into: expected InvalidCropRegion, got {other:?}"),
    }

    // Size -> allocate -> decode: the size query must refuse what the decode
    // refuses, or a caller allocates for a clamped region that never decodes.
    match decoder.output_buffer_size() {
        Err(JpegError::InvalidCropRegion { reason }) => assert_eq!(reason, UPSTREAM_MESSAGE),
        other => panic!("output_buffer_size: expected InvalidCropRegion, got {other:?}"),
    }

    let Some(djpeg) = helpers::optional_c_tool("djpeg") else {
        eprintln!("SKIP: djpeg not found; the Rust assertions above still ran");
        return;
    };
    assert_eq!(
        djpeg_crop(&djpeg, &fixture_path("gray_8x8.jpg"), None, (10, 1, 16, 2)),
        None,
        "djpeg accepts -crop 16x2+10+1 on an 8x8 image; the oracle no longer \
         refuses what this test says it refuses"
    );
}

/// Accept/refuse must agree with `djpeg -crop` across the edges of the scaled
/// output, at 1/1 and 1/2, for a grayscale, a 4:4:4 and a 4:2:0 frame — and an
/// accepted region must produce djpeg's dimensions. Pixels are compared where
/// this port's cropped decode is pixel-exact against djpeg's (no chroma
/// upsampling: grayscale and 4:4:4); `cross_check_crop_scale.rs` records why
/// the subsampled crop boundary is not.
#[test]
fn crop_bounds_agree_with_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let fixtures: [(&str, bool); 3] = [
        ("gray_8x8.jpg", true),
        ("cjpeg_33x31_444.jpg", true),
        ("cjpeg_33x31_420.jpg", false),
    ];
    let scales: [Option<(u32, u32)>; 2] = [None, Some((1, 2))];
    let mut accepted: usize = 0;
    let mut refused: usize = 0;

    for (name, pixel_exact) in fixtures {
        let jpeg: Vec<u8> = read_fixture(name);
        for scale in scales {
            let mut probe: Decoder<'_> = Decoder::new(&jpeg).expect("fixture header parses");
            if let Some((num, denom)) = scale {
                probe.set_scale(ScalingFactor::new(num, denom));
            }
            let (out_w, out_h): (usize, usize) = (probe.output_width(), probe.output_height());
            let regions: [(usize, usize, usize, usize); 9] = [
                (0, 0, out_w, out_h),                 // the whole output
                (0, 0, out_w.div_ceil(2), out_h / 2), // interior
                (out_w - 1, out_h - 1, 1, 1),         // last pixel: fits exactly
                (0, 0, out_w + 1, out_h),             // one column too wide
                (out_w, 0, 1, 1),                     // starts at the right edge
                (out_w + 2, 1, 16, 2),                // the #618 shape
                (0, 1, out_w, out_h),                 // one row too tall
                (0, out_h, 1, 1),                     // starts at the bottom edge
                (1, 1, out_w, 1),                     // x + w one past the edge
            ];
            for region in regions {
                let label: String = format!("{name} scale={scale:?} region={region:?}");
                let theirs: Option<(usize, usize, Vec<u8>)> =
                    djpeg_crop(&djpeg, &fixture_path(name), scale, region);
                let ours: Result<Image, JpegError> = rust_crop(&jpeg, scale, region);
                match theirs {
                    None => {
                        let reason: String = crop_refusal(ours, &label);
                        assert_eq!(reason, UPSTREAM_MESSAGE, "{label}");
                        refused += 1;
                    }
                    Some((c_w, c_h, c_pixels)) => {
                        let image: Image = ours
                            .unwrap_or_else(|e| panic!("{label}: djpeg decodes, we refuse: {e}"));
                        assert_eq!((image.width, image.height), (c_w, c_h), "{label}");
                        if pixel_exact {
                            let expected_format: PixelFormat = if c_pixels.len() == c_w * c_h {
                                PixelFormat::Grayscale
                            } else {
                                PixelFormat::Rgb
                            };
                            assert_eq!(image.pixel_format, expected_format, "{label}");
                            assert_eq!(
                                helpers::pixel_max_diff(&image.data, &c_pixels),
                                0,
                                "{label}: cropped pixels differ from djpeg"
                            );
                            assert_eq!(image.data.len(), c_pixels.len(), "{label}");
                        }
                        accepted += 1;
                    }
                }
            }
        }
    }
    // Both arms must have run, or the agreement above is one-sided.
    assert_eq!(accepted, 18, "accepted regions");
    assert_eq!(refused, 36, "refused regions");
}

/// A zero *height* is accepted: `StreamingDecoder::skip_scanlines` skips to
/// the bottom by setting a zero-height vertical crop at `y = output_height`,
/// as `jpeg_skip_scanlines` may skip every row, and `djpeg -skip 0,7` writes
/// the 8x0 image this must equal. With the false invariant gone the
/// vertical-crop step has to handle the empty result, not assert it away.
#[test]
fn zero_height_crop_at_bottom_edge_decodes_empty() {
    let jpeg: Vec<u8> = read_fixture("gray_8x8.jpg");
    let mut decoder: Decoder<'_> = Decoder::new(&jpeg).expect("fixture header parses");
    decoder.set_crop_region(0, 8, 8, 0);
    let image: Image = decoder.decode_image().expect("a zero-height crop fits");
    assert_eq!((image.width, image.height), (8, 0));
    assert!(image.data.is_empty());

    let Some(djpeg) = helpers::optional_c_tool("djpeg") else {
        eprintln!("SKIP: djpeg not found; the Rust assertions above still ran");
        return;
    };
    let out: helpers::TempFile = helpers::TempFile::new("crop_region_bounds_skip.pgm");
    helpers::run_c_djpeg(
        &djpeg,
        &["-skip", "0,7", "-pnm"],
        &fixture_path("gray_8x8.jpg"),
        out.path(),
    );
    let raw: Vec<u8> = std::fs::read(out.path()).expect("read djpeg output");
    let (c_w, c_h, c_pixels): (usize, usize, Vec<u8>) =
        helpers::parse_pgm(&raw).expect("djpeg wrote a PGM");
    assert_eq!((c_w, c_h, c_pixels.len()), (8, 0, 0));
}

/// A zero *width* is refused: aligning its left boundary down would widen it
/// into columns nobody asked for (`set_crop(5, 0)` decoded five columns before
/// P4-197). C refuses it twice over — `jpeg_crop_scanline` raises
/// `JERR_WIDTH_OVERFLOW` for `*width == 0` (`jdapistd.c:213-216`) and
/// `djpeg -crop 0x8+5+0` does not parse — so the oracle is the source, not a
/// run.
#[test]
fn zero_width_crop_is_refused() {
    let jpeg: Vec<u8> = read_fixture("cjpeg_33x31_444.jpg");
    for x in [0usize, 5, 8] {
        let mut decoder: Decoder<'_> = Decoder::new(&jpeg).expect("fixture header parses");
        decoder.set_crop(x, 0);
        let reason: String = crop_refusal(decoder.decode_image(), &format!("set_crop({x}, 0)"));
        assert_eq!(reason, UPSTREAM_MESSAGE);
    }
}
