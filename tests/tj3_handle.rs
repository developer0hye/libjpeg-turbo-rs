mod helpers;

use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn handle_default_values() {
    let handle = TjHandle::new();
    // Quality and subsampling default to *unset*, exactly as upstream's
    // handle does (`tjInit`: quality -1, subsamp TJSAMP_UNKNOWN) — a lossy
    // compress refuses until the caller supplies them (P4-155, #539).
    assert_eq!(handle.get(TjParam::Quality), -1);
    assert_eq!(handle.get(TjParam::Subsampling), -1);
    // Default precision = 8
    assert_eq!(handle.get(TjParam::Precision), 8);
    // Default colorspace = TJCS_DEFAULT = -1 (auto-detect)
    assert_eq!(handle.get(TjParam::ColorSpace), -1);
    // Boolean defaults: all false (0)
    assert_eq!(handle.get(TjParam::FastUpSample), 0);
    assert_eq!(handle.get(TjParam::FastDct), 0);
    assert_eq!(handle.get(TjParam::Optimize), 0);
    assert_eq!(handle.get(TjParam::Progressive), 0);
    assert_eq!(handle.get(TjParam::Arithmetic), 0);
    assert_eq!(handle.get(TjParam::Lossless), 0);
    assert_eq!(handle.get(TjParam::BottomUp), 0);
    assert_eq!(handle.get(TjParam::NoRealloc), 0);
    assert_eq!(handle.get(TjParam::StopOnWarning), 0);
    // Default density = 72 DPI
    assert_eq!(handle.get(TjParam::XDensity), 1);
    assert_eq!(handle.get(TjParam::YDensity), 1);
    assert_eq!(handle.get(TjParam::DensityUnits), 0); // DPI
                                                      // Width/Height default 0 (not yet decompressed)
    assert_eq!(handle.get(TjParam::Width), 0);
    assert_eq!(handle.get(TjParam::Height), 0);
    // Lossless params
    assert_eq!(handle.get(TjParam::LosslessPsv), 1);
    assert_eq!(handle.get(TjParam::LosslessPt), 0);
    // Restart defaults
    assert_eq!(handle.get(TjParam::RestartBlocks), 0);
    assert_eq!(handle.get(TjParam::RestartRows), 0);
    // Scan limit default
    assert_eq!(handle.get(TjParam::ScanLimit), 0);
    // MaxMemory/MaxPixels defaults (0 = unlimited)
    assert_eq!(handle.get(TjParam::MaxMemory), 0);
    assert_eq!(handle.get(TjParam::MaxPixels), 0);
    // SaveMarkers default = 2 (All, matching C TJSM_ALL)
    assert_eq!(handle.get(TjParam::SaveMarkers), 2);
}

#[test]
fn handle_set_get_quality() {
    let mut handle = TjHandle::new();
    handle.set(TjParam::Quality, 90).unwrap();
    assert_eq!(handle.get(TjParam::Quality), 90);
}

#[test]
fn handle_set_get_subsampling() {
    let mut handle = TjHandle::new();
    // Set to S444 = index 0
    handle.set(TjParam::Subsampling, 0).unwrap();
    assert_eq!(handle.get(TjParam::Subsampling), 0);
    // Set to S422 = index 1
    handle.set(TjParam::Subsampling, 1).unwrap();
    assert_eq!(handle.get(TjParam::Subsampling), 1);
}

#[test]
fn handle_set_get_boolean_params() {
    let mut handle = TjHandle::new();
    handle.set(TjParam::Optimize, 1).unwrap();
    assert_eq!(handle.get(TjParam::Optimize), 1);
    handle.set(TjParam::Progressive, 1).unwrap();
    assert_eq!(handle.get(TjParam::Progressive), 1);
    handle.set(TjParam::Arithmetic, 1).unwrap();
    assert_eq!(handle.get(TjParam::Arithmetic), 1);
    handle.set(TjParam::Lossless, 1).unwrap();
    assert_eq!(handle.get(TjParam::Lossless), 1);
    handle.set(TjParam::BottomUp, 1).unwrap();
    assert_eq!(handle.get(TjParam::BottomUp), 1);
    handle.set(TjParam::StopOnWarning, 1).unwrap();
    assert_eq!(handle.get(TjParam::StopOnWarning), 1);
    handle.set(TjParam::FastUpSample, 1).unwrap();
    assert_eq!(handle.get(TjParam::FastUpSample), 1);
    handle.set(TjParam::FastDct, 1).unwrap();
    assert_eq!(handle.get(TjParam::FastDct), 1);
    handle.set(TjParam::NoRealloc, 1).unwrap();
    assert_eq!(handle.get(TjParam::NoRealloc), 1);
}

#[test]
fn handle_set_get_density() {
    let mut handle = TjHandle::new();
    handle.set(TjParam::XDensity, 300).unwrap();
    handle.set(TjParam::YDensity, 600).unwrap();
    handle.set(TjParam::DensityUnits, 2).unwrap(); // DPCM
    assert_eq!(handle.get(TjParam::XDensity), 300);
    assert_eq!(handle.get(TjParam::YDensity), 600);
    assert_eq!(handle.get(TjParam::DensityUnits), 2);
}

#[test]
fn handle_set_get_lossless_params() {
    let mut handle = TjHandle::new();
    handle.set(TjParam::LosslessPsv, 5).unwrap();
    handle.set(TjParam::LosslessPt, 8).unwrap();
    assert_eq!(handle.get(TjParam::LosslessPsv), 5);
    assert_eq!(handle.get(TjParam::LosslessPt), 8);
}

#[test]
fn handle_set_get_restart_params() {
    let mut handle = TjHandle::new();
    handle.set(TjParam::RestartBlocks, 50).unwrap();
    assert_eq!(handle.get(TjParam::RestartBlocks), 50);
    handle.set(TjParam::RestartRows, 3).unwrap();
    assert_eq!(handle.get(TjParam::RestartRows), 3);
}

#[test]
fn handle_set_get_limits() {
    let mut handle = TjHandle::new();
    handle.set(TjParam::ScanLimit, 200).unwrap();
    assert_eq!(handle.get(TjParam::ScanLimit), 200);
    handle.set(TjParam::MaxMemory, 1_000_000).unwrap();
    assert_eq!(handle.get(TjParam::MaxMemory), 1_000_000);
    handle.set(TjParam::MaxPixels, 500_000).unwrap();
    assert_eq!(handle.get(TjParam::MaxPixels), 500_000);
}

#[test]
fn handle_set_get_save_markers() {
    let mut handle = TjHandle::new();
    // C-compatible range: 0-4
    for level in 0..=4 {
        handle.set(TjParam::SaveMarkers, level).unwrap();
        assert_eq!(handle.get(TjParam::SaveMarkers), level);
    }
    // Out-of-range must fail
    assert!(handle.set(TjParam::SaveMarkers, 5).is_err());
    assert!(handle.set(TjParam::SaveMarkers, -1).is_err());
}

#[test]
fn handle_invalid_quality_returns_error() {
    let mut handle = TjHandle::new();
    // Quality must be 1-100
    assert!(handle.set(TjParam::Quality, 0).is_err());
    assert!(handle.set(TjParam::Quality, 101).is_err());
}

#[test]
fn handle_invalid_subsampling_returns_error() {
    let mut handle = TjHandle::new();
    // Valid subsampling indices (libjpeg-turbo 3.x): 0-8.
    assert!(handle.set(TjParam::Subsampling, -1).is_err());
    assert!(handle.set(TjParam::Subsampling, 9).is_err());
}

#[test]
fn handle_invalid_lossless_psv_returns_error() {
    let mut handle = TjHandle::new();
    // PSV must be 1-7
    assert!(handle.set(TjParam::LosslessPsv, 0).is_err());
    assert!(handle.set(TjParam::LosslessPsv, 8).is_err());
}

#[test]
fn handle_invalid_lossless_pt_returns_error() {
    let mut handle = TjHandle::new();
    // PT must be 0-15
    assert!(handle.set(TjParam::LosslessPt, -1).is_err());
    assert!(handle.set(TjParam::LosslessPt, 16).is_err());
}

#[test]
fn handle_invalid_density_units_returns_error() {
    let mut handle = TjHandle::new();
    // DensityUnits 0-2 are valid
    assert!(handle.set(TjParam::DensityUnits, -1).is_err());
    assert!(handle.set(TjParam::DensityUnits, 3).is_err());
}

#[test]
fn handle_icc_profile() {
    let mut handle = TjHandle::new();
    assert!(handle.icc_profile().is_none());
    let profile = vec![1u8, 2, 3, 4, 5];
    handle.set_icc_profile(Some(profile.clone()));
    assert_eq!(handle.icc_profile(), Some(profile.as_slice()));
    handle.set_icc_profile(None);
    assert!(handle.icc_profile().is_none());
}

#[test]
fn handle_scaling_factor() {
    let mut handle = TjHandle::new();
    // Valid scaling factors (standard JPEG IDCT scales)
    handle.set_scaling_factor(1, 1).unwrap();
    handle.set_scaling_factor(1, 2).unwrap();
    handle.set_scaling_factor(1, 4).unwrap();
    handle.set_scaling_factor(1, 8).unwrap();
    handle.set_scaling_factor(2, 1).unwrap();
    handle.set_scaling_factor(3, 2).unwrap();
    handle.set_scaling_factor(3, 4).unwrap();
    handle.set_scaling_factor(3, 8).unwrap();
    // Invalid scaling factor
    assert!(handle.set_scaling_factor(1, 3).is_err());
    assert!(handle.set_scaling_factor(0, 1).is_err());
    assert!(handle.set_scaling_factor(4, 1).is_err());
}

/// The cropping rules of `tj3SetCroppingRegion`
/// (`references/libjpeg-turbo/src/turbojpeg.c:2068-2115`): a header must have
/// been read, a left boundary off the scaled iMCU grid and a region past the
/// scaled image are refused, a zero width/height runs to the edge (P4-197,
/// #618). `resolve_cropping_region` applies them at set time, as the C ABI
/// does; `decompress` applies them to the image it decodes. The C-ABI twin is
/// compared against stock TurboJPEG by the `cropping_region` case of
/// `crates/libjpeg-turbo-rs-capi/examples/cabi_misuse_harness.c`; here the
/// decoded pixels of the accepted regions are compared against `djpeg -crop`.
#[test]
fn handle_cropping_region() {
    use libjpeg_turbo_rs::{CropRegion, JpegError};

    fn reason<T: std::fmt::Debug>(result: libjpeg_turbo_rs::Result<T>) -> String {
        match result {
            Err(JpegError::InvalidCropRegion { reason }) => reason,
            other => panic!("expected InvalidCropRegion, got {other:?}"),
        }
    }
    let region = |x: usize, y: usize, width: usize, height: usize| -> CropRegion {
        CropRegion {
            x,
            y,
            width,
            height,
        }
    };
    let exceeds: &str = "The cropping region exceeds the scaled image dimensions";
    let off_grid: &str =
        "The left boundary of the cropping region (4) is not\ndivisible by the scaled iMCU width (8)";

    // 33x31, 4:4:4: an 8-pixel iMCU at 1/1, 4 at 1/2.
    let fixture: std::path::PathBuf = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/cjpeg_33x31_444.jpg");
    let jpeg: Vec<u8> = std::fs::read(&fixture).expect("read fixture");
    let mut handle: TjHandle = TjHandle::new();

    // Set-time validation needs a header.
    assert_eq!(
        reason(handle.resolve_cropping_region(region(8, 0, 8, 8))),
        "JPEG header has not yet been read"
    );
    handle.decompress_header(&jpeg).expect("header");
    assert_eq!(
        reason(handle.resolve_cropping_region(region(4, 0, 8, 8))),
        off_grid
    );
    assert_eq!(
        reason(handle.resolve_cropping_region(region(32, 0, 8, 8))),
        exceeds
    );
    assert_eq!(
        reason(handle.resolve_cropping_region(region(40, 0, 0, 0))),
        exceeds
    );
    assert_eq!(
        reason(handle.resolve_cropping_region(region(0, 24, 8, 8))),
        exceeds
    );
    assert_eq!(
        handle
            .resolve_cropping_region(region(16, 5, 0, 0))
            .expect("fits"),
        region(16, 5, 17, 26),
        "a zero width/height is filled in to the edge"
    );

    // Accepted regions decode to djpeg's pixels. The second and third are
    // stored with a zero extent, resolved by the decode.
    let cases: [(Option<(u32, u32)>, CropRegion, (usize, usize)); 3] = [
        (None, region(8, 3, 17, 20), (17, 20)),
        (None, region(16, 5, 0, 0), (17, 26)),
        (Some((1, 2)), region(4, 2, 9, 0), (9, 14)),
    ];
    let djpeg: Option<std::path::PathBuf> = helpers::optional_c_tool("djpeg");
    for (scale, crop, (want_w, want_h)) in cases {
        let (num, denom): (u32, u32) = scale.unwrap_or((1, 1));
        handle
            .set_scaling_factor(num, denom)
            .expect("scaling factor");
        handle.set_cropping_region(Some(crop));
        let image: libjpeg_turbo_rs::Image = handle.decompress(&jpeg).expect("cropped decode");
        assert_eq!(
            (image.width, image.height),
            (want_w, want_h),
            "{scale:?} {crop:?}"
        );

        let Some(djpeg) = djpeg.as_ref() else {
            eprintln!("SKIP: djpeg not found; pixel comparison for {crop:?}");
            continue;
        };
        // djpeg is asked for the filled-in region.
        let out: helpers::TempFile = helpers::TempFile::new("tj3_handle_crop.ppm");
        let mut args: Vec<String> = Vec::new();
        if let Some((num, denom)) = scale {
            args.push("-scale".to_string());
            args.push(format!("{num}/{denom}"));
        }
        args.push("-crop".to_string());
        args.push(format!("{want_w}x{want_h}+{}+{}", crop.x, crop.y));
        args.push("-ppm".to_string());
        let arg_refs: Vec<&str> = args.iter().map(String::as_str).collect();
        helpers::run_c_djpeg(djpeg, &arg_refs, &fixture, out.path());
        let (c_w, c_h, c_pixels): (usize, usize, Vec<u8>) = helpers::parse_ppm_file(out.path());
        assert_eq!((c_w, c_h), (want_w, want_h), "djpeg {args:?}");
        assert_eq!(image.data.len(), c_pixels.len(), "{crop:?}");
        assert_eq!(
            helpers::pixel_max_diff(&image.data, &c_pixels),
            0,
            "{crop:?}"
        );
    }

    // Each rule refuses at decode time too, instead of clamping.
    let refused: [(Option<(u32, u32)>, CropRegion, &str); 4] = [
        // Fitted at 1/2; at 1/1 it is off the 8-pixel grid.
        (None, region(4, 2, 9, 14), off_grid),
        (None, region(32, 0, 8, 8), exceeds),
        (None, region(40, 0, 0, 0), exceeds),
        (Some((1, 2)), region(0, 10, 4, 7), exceeds),
    ];
    for (scale, crop, want) in refused {
        let (num, denom): (u32, u32) = scale.unwrap_or((1, 1));
        handle
            .set_scaling_factor(num, denom)
            .expect("scaling factor");
        handle.set_cropping_region(Some(crop));
        assert_eq!(reason(handle.decompress(&jpeg)), want, "{scale:?} {crop:?}");
    }

    // Reading a header ignores the stored region, as tj3DecompressHeader does
    // — the region stored last does not fit, and the read still succeeds.
    handle
        .decompress_header(&jpeg)
        .expect("header read ignores the crop");
    assert_eq!(handle.get(TjParam::Width), 33);
    assert_eq!(handle.get(TjParam::Height), 31);

    // TJUNCROPPED clears.
    handle.set_cropping_region(Some(region(0, 0, 0, 0)));
    handle.set_scaling_factor(1, 1).expect("scaling factor");
    let image: libjpeg_turbo_rs::Image = handle.decompress(&jpeg).expect("uncropped decode");
    assert_eq!((image.width, image.height), (33, 31));
}

/// A left boundary that clears TurboJPEG's iMCU rule but not the decoder's is
/// refused at decode time with upstream's message, never decoded wider than
/// the region. `cjpeg -sample 2x1,2x1,2x1` writes a frame upstream's
/// `getSubsamp` classifies as TJSAMP_444 (an 8-pixel iMCU, `turbojpeg.c:491-505`)
/// that libjpeg decodes in 16-pixel iMCU columns. Stock TurboJPEG 3.2.0,
/// measured with a C probe on the same command's output: `tj3SetCroppingRegion`
/// accepts `{8, 0, 32, 16}` and `tj3Decompress8` fails with
/// "Unexplained mismatch between specified (8) and\nactual (0) cropping region
/// left boundary" (`turbojpeg-mp.c:217-221`); `{16, 0, 32, 16}` decodes.
/// Before the check, ours returned a 40-column image for the 32-column
/// region — through the C ABI, past the end of the caller's buffer.
#[test]
fn handle_cropping_region_refuses_a_decoder_imcu_mismatch() {
    use libjpeg_turbo_rs::{CropRegion, JpegError};

    let cjpeg: std::path::PathBuf = require_c_tool!("cjpeg");
    let source: std::path::PathBuf = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("references/libjpeg-turbo/testimages/testorig.ppm");
    let out: helpers::TempFile = helpers::TempFile::new("tj3_handle_2x1.jpg");
    helpers::run_c_cjpeg(&cjpeg, &["-sample", "2x1,2x1,2x1"], &source, out.path());
    let jpeg: Vec<u8> = std::fs::read(out.path()).expect("read cjpeg output");

    let mut handle: TjHandle = TjHandle::new();
    handle.decompress_header(&jpeg).expect("header");
    let misaligned: CropRegion = CropRegion {
        x: 8,
        y: 0,
        width: 32,
        height: 16,
    };
    let stored: CropRegion = handle
        .resolve_cropping_region(misaligned)
        .expect("x = 8 clears TJSAMP_444's 8-pixel iMCU, as upstream's set does");
    handle.set_cropping_region(Some(stored));
    match handle.decompress(&jpeg) {
        Err(JpegError::InvalidCropRegion { reason }) => assert_eq!(
            reason,
            "Unexplained mismatch between specified (8) and\nactual (0) cropping region left boundary"
        ),
        Ok(image) => panic!("decoded {}x{} for a 32x16 region", image.width, image.height),
        Err(other) => panic!("expected InvalidCropRegion, got {other:?}"),
    }

    handle.set_cropping_region(Some(CropRegion {
        x: 16,
        ..misaligned
    }));
    let image: libjpeg_turbo_rs::Image = handle.decompress(&jpeg).expect("x = 16 is on both grids");
    assert_eq!((image.width, image.height), (32, 16));
}

/// The 12- and 16-bit decompress paths take no region yet (P4-219), so a stored
/// one is refused instead of being ignored: ignoring it returned the whole
/// frame to a C caller whose buffer was sized for the region. Upstream crops at
/// 12 bits (`turbojpeg-mp.c:211-279` is compiled for every
/// `BITS_IN_JSAMPLE != 16`); until P4-219 lands, refusing is the safe half.
#[test]
fn twelve_and_sixteen_bit_decompress_refuse_a_region_they_cannot_apply() {
    use libjpeg_turbo_rs::{CropRegion, JpegError, Subsampling};

    let samples: Vec<i16> = (0..32 * 32)
        .map(|i: i32| ((i * 13) % 4096) as i16)
        .collect();
    let twelve: Vec<u8> =
        libjpeg_turbo_rs::precision::compress_12bit(&samples, 32, 32, 1, 90, Subsampling::S444)
            .expect("12-bit encode");
    let region: CropRegion = CropRegion {
        x: 8,
        y: 4,
        width: 16,
        height: 8,
    };

    let mut handle: TjHandle = TjHandle::new();
    let whole: libjpeg_turbo_rs::precision::Image12 = handle
        .decompress_12bit(&twelve)
        .expect("no region: decodes");
    assert_eq!((whole.width, whole.height), (32, 32));

    handle.set_cropping_region(Some(region));
    match handle.decompress_12bit(&twelve) {
        Err(JpegError::Unsupported(message)) => assert!(message.contains("P4-219"), "{message}"),
        Ok(image) => panic!("decoded {}x{} for a 16x8 region", image.width, image.height),
        Err(other) => panic!("expected Unsupported, got {other:?}"),
    }
    match handle.decompress_16bit(&twelve) {
        Err(JpegError::Unsupported(message)) => assert!(message.contains("P4-219"), "{message}"),
        Ok(image) => panic!("decoded {}x{} for a 16x8 region", image.width, image.height),
        Err(other) => panic!("expected Unsupported, got {other:?}"),
    }
}

#[test]
fn handle_scaling_factors_list() {
    let factors = TjHandle::scaling_factors();
    assert!(factors.contains(&(1, 1)));
    assert!(factors.contains(&(1, 2)));
    assert!(factors.contains(&(1, 4)));
    assert!(factors.contains(&(1, 8)));
    assert_eq!(factors.len(), 16);
}

#[test]
fn handle_compress_roundtrip() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels = vec![128u8; width * height * 3];
    let mut handle = TjHandle::new();
    handle.set(TjParam::Quality, 85).unwrap();
    handle.set(TjParam::Subsampling, 0).unwrap(); // S444

    let jpeg = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // Verify it's valid JPEG: starts with FFD8, ends with FFD9
    assert!(jpeg.len() > 4);
    assert_eq!(jpeg[0], 0xFF);
    assert_eq!(jpeg[1], 0xD8);
    assert_eq!(jpeg[jpeg.len() - 2], 0xFF);
    assert_eq!(jpeg[jpeg.len() - 1], 0xD9);

    // Decompress and verify dimensions
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
}

#[test]
fn handle_compress_progressive() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![100u8; width * height * 3];
    let mut handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle.set(TjParam::Subsampling, 2).unwrap();
    handle.set(TjParam::Quality, 75).unwrap();
    handle.set(TjParam::Progressive, 1).unwrap();

    let jpeg = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
}

#[test]
fn handle_compress_arithmetic() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![100u8; width * height * 3];
    let mut handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle.set(TjParam::Subsampling, 2).unwrap();
    handle.set(TjParam::Quality, 80).unwrap();
    handle.set(TjParam::Arithmetic, 1).unwrap();

    let jpeg = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
}

#[test]
fn handle_compress_optimized() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![100u8; width * height * 3];
    let mut handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle.set(TjParam::Subsampling, 2).unwrap();
    handle.set(TjParam::Quality, 75).unwrap();
    handle.set(TjParam::Optimize, 1).unwrap();

    let jpeg = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
}

#[test]
fn handle_compress_lossless() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![42u8; width * height];
    let mut handle = TjHandle::new();
    handle.set(TjParam::Lossless, 1).unwrap();
    handle.set(TjParam::LosslessPsv, 1).unwrap();
    handle.set(TjParam::LosslessPt, 0).unwrap();

    let jpeg = handle
        .compress(&pixels, width, height, PixelFormat::Grayscale)
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    // Lossless: pixel data should match exactly
    assert_eq!(img.data, pixels);
}

#[test]
fn handle_decompress() {
    // Create a valid JPEG first
    let width: usize = 24;
    let height: usize = 24;
    let pixels = vec![200u8; width * height * 3];
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        85,
        Subsampling::S444,
    )
    .unwrap();

    let mut handle = TjHandle::new();
    let img = handle.decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
}

#[test]
fn handle_decompress_updates_width_height() {
    let width: usize = 32;
    let height: usize = 24;
    let pixels = vec![128u8; width * height * 3];
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let mut handle = TjHandle::new();
    assert_eq!(handle.get(TjParam::Width), 0);
    assert_eq!(handle.get(TjParam::Height), 0);

    let _img = handle.decompress(&jpeg).unwrap();
    assert_eq!(handle.get(TjParam::Width), width as i32);
    assert_eq!(handle.get(TjParam::Height), height as i32);
}

#[test]
fn handle_decompress_with_icc_profile() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];
    let icc = vec![0xAAu8; 64];
    let jpeg = libjpeg_turbo_rs::compress_with_metadata(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
        Some(&icc),
        None,
    )
    .unwrap();

    let mut handle = TjHandle::new();
    handle.set_icc_profile(Some(vec![0xBB; 10])); // should be overwritten by decompress
    let _img = handle.decompress(&jpeg).unwrap();
    // Verify handle ICC profile is updated from decoded image (not the old value)
    assert_eq!(handle.icc_profile(), Some(icc.as_slice()));
}

#[test]
fn handle_decompress_with_scaling() {
    let width: usize = 64;
    let height: usize = 64;
    let pixels = vec![128u8; width * height * 3];
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let mut handle = TjHandle::new();
    handle.set_scaling_factor(1, 2).unwrap();
    let img = handle.decompress(&jpeg).unwrap();
    assert_eq!(img.width, 32);
    assert_eq!(img.height, 32);
}

#[test]
fn handle_decompress_with_stop_on_warning() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let mut handle = TjHandle::new();
    handle.set(TjParam::StopOnWarning, 1).unwrap();
    let img = handle.decompress(&jpeg).unwrap();
    assert_eq!(img.width, width);
}

#[test]
fn handle_decompress_with_max_pixels() {
    let width: usize = 64;
    let height: usize = 64;
    let pixels = vec![128u8; width * height * 3];
    let jpeg = compress(
        &pixels,
        width,
        height,
        PixelFormat::Rgb,
        75,
        Subsampling::S444,
    )
    .unwrap();

    let mut handle = TjHandle::new();
    handle.set(TjParam::MaxPixels, 32 * 32).unwrap();
    let result = handle.decompress(&jpeg);
    assert!(result.is_err());
}

// === Behavioral tests: verify params actually affect output ===

#[test]
fn handle_quality_affects_output_size() {
    let width: usize = 64;
    let height: usize = 64;
    // Use varied pixel data to make quality effect visible
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();

    let mut handle_low = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle_low.set(TjParam::Subsampling, 2).unwrap();
    handle_low.set(TjParam::Quality, 10).unwrap();
    let jpeg_low = handle_low
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    let mut handle_high = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle_high.set(TjParam::Subsampling, 2).unwrap();
    handle_high.set(TjParam::Quality, 95).unwrap();
    let jpeg_high = handle_high
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // Higher quality must produce larger output
    assert!(
        jpeg_high.len() > jpeg_low.len(),
        "q95 ({}) should be larger than q10 ({})",
        jpeg_high.len(),
        jpeg_low.len()
    );
}

#[test]
fn handle_subsampling_affects_output() {
    let width: usize = 64;
    let height: usize = 64;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();

    let mut handle_444 = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle_444.set(TjParam::Quality, 75).unwrap();
    handle_444.set(TjParam::Subsampling, 0).unwrap(); // S444
    let jpeg_444 = handle_444
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    let mut handle_420 = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle_420.set(TjParam::Quality, 75).unwrap();
    handle_420.set(TjParam::Subsampling, 2).unwrap(); // S420
    let jpeg_420 = handle_420
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // S444 (no chroma subsampling) must produce larger output than S420
    assert!(
        jpeg_444.len() > jpeg_420.len(),
        "S444 ({}) should be larger than S420 ({})",
        jpeg_444.len(),
        jpeg_420.len()
    );
}

#[test]
fn handle_bottom_up_affects_pixel_order() {
    let width: usize = 8;
    let height: usize = 8;
    // Create a gradient: top row = dark, bottom row = bright
    let mut pixels = vec![0u8; width * height * 3];
    for y in 0..height {
        let val: u8 = (y * 255 / (height - 1)) as u8;
        for x in 0..width {
            let offset: usize = (y * width + x) * 3;
            pixels[offset] = val;
            pixels[offset + 1] = val;
            pixels[offset + 2] = val;
        }
    }

    // Encode without bottom-up
    let mut handle_normal = TjHandle::new();
    handle_normal.set(TjParam::Quality, 100).unwrap();
    handle_normal.set(TjParam::Subsampling, 0).unwrap();
    let jpeg_normal = handle_normal
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // Encode with bottom-up (rows are read in reverse order)
    let mut handle_bu = TjHandle::new();
    handle_bu.set(TjParam::Quality, 100).unwrap();
    handle_bu.set(TjParam::Subsampling, 0).unwrap();
    handle_bu.set(TjParam::BottomUp, 1).unwrap();
    let jpeg_bu = handle_bu
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // The two JPEGs should differ (flipped input = different DCT coefficients)
    assert_ne!(
        jpeg_normal, jpeg_bu,
        "BottomUp should produce different output"
    );
}

#[test]
fn handle_density_roundtrip_through_compress_decompress() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];

    let mut handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle.set(TjParam::Quality, 75).unwrap();
    handle.set(TjParam::Subsampling, 2).unwrap();
    handle.set(TjParam::XDensity, 300).unwrap();
    handle.set(TjParam::YDensity, 600).unwrap();
    handle.set(TjParam::DensityUnits, 1).unwrap(); // DPI

    let jpeg = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // Verify JFIF density in raw bytes (bytes 13-17 of APP0)
    assert_eq!(jpeg[13], 1, "density unit should be DPI (1)");
    assert_eq!(
        u16::from_be_bytes([jpeg[14], jpeg[15]]),
        300,
        "x_density should be 300"
    );
    assert_eq!(
        u16::from_be_bytes([jpeg[16], jpeg[17]]),
        600,
        "y_density should be 600"
    );

    // Decompress and verify handle captures density from JFIF
    let mut handle2 = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle2.set(TjParam::Quality, 75).unwrap();
    handle2.set(TjParam::Subsampling, 2).unwrap();
    let _img = handle2.decompress(&jpeg).unwrap();
    assert_eq!(handle2.get(TjParam::DensityUnits), 1);
    assert_eq!(handle2.get(TjParam::XDensity), 300);
    assert_eq!(handle2.get(TjParam::YDensity), 600);
}

#[test]
fn handle_decompress_updates_colorspace_and_subsampling() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels = vec![128u8; width * height * 3];

    // Compress with S422
    let mut enc_handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality is *unset*.
    enc_handle.set(TjParam::Quality, 75).unwrap();
    enc_handle.set(TjParam::Subsampling, 1).unwrap(); // S422
    let jpeg = enc_handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // Decompress and verify handle is updated from JPEG header
    let mut dec_handle = TjHandle::new();
    let _img = dec_handle.decompress(&jpeg).unwrap();

    // ColorSpace should be YCbCr (1) for standard JFIF
    assert_eq!(
        dec_handle.get(TjParam::ColorSpace),
        1,
        "color space should be YCbCr (1)"
    );
    // Subsampling should be S422 (1)
    assert_eq!(
        dec_handle.get(TjParam::Subsampling),
        1,
        "subsampling should be S422 (1)"
    );
}

#[test]
fn handle_decompress_grayscale_updates_colorspace() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height];

    let mut enc_handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling are
    // *unset*, as upstream's are, and a lossy compress refuses them.
    enc_handle.set(TjParam::Quality, 75).unwrap();
    enc_handle.set(TjParam::Subsampling, 3).unwrap(); // TJSAMP_GRAY
    let jpeg = enc_handle
        .compress(&pixels, width, height, PixelFormat::Grayscale)
        .unwrap();

    let mut dec_handle = TjHandle::new();
    let _img = dec_handle.decompress(&jpeg).unwrap();

    // ColorSpace should be Grayscale (2)
    assert_eq!(
        dec_handle.get(TjParam::ColorSpace),
        2,
        "color space should be Grayscale (2)"
    );
    // Subsampling should be TJSAMP_GRAY (3) — matches C libjpeg-turbo
    assert_eq!(
        dec_handle.get(TjParam::Subsampling),
        3,
        "subsampling should be TJSAMP_GRAY (3)"
    );
}

#[test]
fn handle_progressive_produces_multiple_sos_markers() {
    let width: usize = 32;
    let height: usize = 32;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 251) as u8).collect();

    let mut handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle.set(TjParam::Quality, 75).unwrap();
    handle.set(TjParam::Subsampling, 2).unwrap();
    handle.set(TjParam::Progressive, 1).unwrap();
    let jpeg = handle
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // Count SOS (0xFFDA) markers — progressive should have multiple scans
    let sos_count = jpeg
        .windows(2)
        .filter(|w| w[0] == 0xFF && w[1] == 0xDA)
        .count();
    assert!(
        sos_count > 1,
        "progressive JPEG should have multiple SOS markers, got {sos_count}"
    );
}

#[test]
fn handle_tj_param_enum_all_variants() {
    // Ensure all 26 variants exist and are distinct
    let params = [
        TjParam::Quality,
        TjParam::Subsampling,
        TjParam::Width,
        TjParam::Height,
        TjParam::Precision,
        TjParam::ColorSpace,
        TjParam::FastUpSample,
        TjParam::FastDct,
        TjParam::Optimize,
        TjParam::Progressive,
        TjParam::ScanLimit,
        TjParam::Arithmetic,
        TjParam::Lossless,
        TjParam::LosslessPsv,
        TjParam::LosslessPt,
        TjParam::RestartBlocks,
        TjParam::RestartRows,
        TjParam::XDensity,
        TjParam::YDensity,
        TjParam::DensityUnits,
        TjParam::MaxMemory,
        TjParam::MaxPixels,
        TjParam::BottomUp,
        TjParam::NoRealloc,
        TjParam::StopOnWarning,
        TjParam::SaveMarkers,
    ];
    assert_eq!(params.len(), 26);
    // Each param should be unique
    for (i, a) in params.iter().enumerate() {
        for (j, b) in params.iter().enumerate() {
            if i != j {
                assert_ne!(a, b);
            }
        }
    }
}

// === SaveMarkers behavioral tests ===

/// Helper: create a JPEG with ICC profile and COM marker embedded.
fn make_jpeg_with_icc_and_com() -> (Vec<u8>, Vec<u8>) {
    use libjpeg_turbo_rs::Encoder;
    let width: usize = 16;
    let height: usize = 16;
    let pixels = vec![128u8; width * height * 3];
    let icc = vec![0xAAu8; 64];
    let jpeg = Encoder::new(&pixels, width, height, PixelFormat::Rgb)
        .quality(75)
        .subsampling(Subsampling::S444)
        .icc_profile(&icc)
        .comment("test comment")
        .encode()
        .unwrap();
    (jpeg, icc)
}

#[test]
fn handle_save_markers_level0_no_markers_no_icc() {
    let (jpeg, _icc) = make_jpeg_with_icc_and_com();

    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 0).unwrap();
    let img = handle.decompress(&jpeg).unwrap();

    // Level 0: no ICC on handle
    assert!(
        handle.icc_profile().is_none(),
        "level 0: handle ICC should be None"
    );
    // Level 0: no ICC on image
    assert!(
        img.icc_profile().is_none(),
        "level 0: image ICC should be None"
    );
    // Level 0: no saved markers
    assert!(
        img.markers().is_empty(),
        "level 0: saved markers should be empty"
    );
}

#[test]
fn handle_save_markers_level1_com_only() {
    let (jpeg, _icc) = make_jpeg_with_icc_and_com();

    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 1).unwrap();
    let img = handle.decompress(&jpeg).unwrap();

    // Level 1: no ICC
    assert!(
        handle.icc_profile().is_none(),
        "level 1: handle ICC should be None"
    );
    // Level 1: only COM markers saved
    assert!(
        !img.markers().is_empty(),
        "level 1: should have saved COM marker"
    );
    for m in img.markers() {
        assert_eq!(
            m.code, 0xFE,
            "level 1: only COM (0xFE) markers should be saved"
        );
    }
}

#[test]
fn handle_save_markers_level2_all_with_icc() {
    let (jpeg, icc) = make_jpeg_with_icc_and_com();

    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 2).unwrap();
    let img = handle.decompress(&jpeg).unwrap();

    // Level 2: ICC extracted to handle
    assert_eq!(
        handle.icc_profile(),
        Some(icc.as_slice()),
        "level 2: handle ICC should match embedded profile"
    );
    // Level 2: all markers saved (COM + APP markers)
    assert!(
        !img.markers().is_empty(),
        "level 2: should have saved markers"
    );
    let has_com = img.markers().iter().any(|m| m.code == 0xFE);
    assert!(has_com, "level 2: should include COM marker");
}

#[test]
fn handle_save_markers_level3_all_except_icc() {
    let (jpeg, _icc) = make_jpeg_with_icc_and_com();

    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 3).unwrap();
    let img = handle.decompress(&jpeg).unwrap();

    // Level 3: no ICC
    assert!(
        handle.icc_profile().is_none(),
        "level 3: handle ICC should be None"
    );
    assert!(
        img.icc_profile().is_none(),
        "level 3: image ICC should be None"
    );
    // Level 3: saved markers should NOT contain ICC APP2
    for m in img.markers() {
        if m.code == 0xE2 {
            assert!(
                !m.data.starts_with(b"ICC_PROFILE\0"),
                "level 3: ICC APP2 markers should be filtered out"
            );
        }
    }
    // Level 3: COM should still be present
    let has_com = img.markers().iter().any(|m| m.code == 0xFE);
    assert!(has_com, "level 3: should include COM marker");
}

#[test]
fn handle_save_markers_level4_icc_only() {
    let (jpeg, icc) = make_jpeg_with_icc_and_com();

    let mut handle = TjHandle::new();
    handle.set(TjParam::SaveMarkers, 4).unwrap();
    let img = handle.decompress(&jpeg).unwrap();

    // Level 4: ICC extracted to handle
    assert_eq!(
        handle.icc_profile(),
        Some(icc.as_slice()),
        "level 4: handle ICC should match embedded profile"
    );
    // Level 4: only APP2 (ICC) markers saved, no COM
    let has_com = img.markers().iter().any(|m| m.code == 0xFE);
    assert!(!has_com, "level 4: should NOT include COM marker");
}

// === TJCS_DEFAULT and ColorSpace wiring tests ===

#[test]
fn handle_colorspace_default_is_negative_one() {
    let handle = TjHandle::new();
    assert_eq!(handle.get(TjParam::ColorSpace), -1);
    // -1 is valid (TJCS_DEFAULT)
    let mut h = TjHandle::new();
    h.set(TjParam::ColorSpace, -1).unwrap();
    assert_eq!(h.get(TjParam::ColorSpace), -1);
    // -2 is invalid
    assert!(h.set(TjParam::ColorSpace, -2).is_err());
}

#[test]
fn handle_colorspace_rgb_override_in_compress() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = (0..width * height * 3).map(|i| (i % 200) as u8).collect();

    // Default colorspace (-1 = auto = YCbCr for RGB input)
    let mut handle_default = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle_default.set(TjParam::Quality, 75).unwrap();
    handle_default.set(TjParam::Subsampling, 2).unwrap();
    let jpeg_default = handle_default
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // Explicit RGB colorspace (0) - should produce different output (no color conversion)
    let mut handle_rgb = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle_rgb.set(TjParam::Quality, 75).unwrap();
    handle_rgb.set(TjParam::Subsampling, 2).unwrap();
    handle_rgb.set(TjParam::ColorSpace, 0).unwrap(); // TJCS_RGB
    let jpeg_rgb = handle_rgb
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .unwrap();

    // RGB-direct vs YCbCr should produce different JPEG data
    assert_ne!(
        jpeg_default, jpeg_rgb,
        "RGB colorspace override should produce different output than auto/YCbCr"
    );
}

// === Multi-precision via TjHandle ===

#[test]
fn handle_compress_decompress_16bit_lossless() {
    let width: usize = 8;
    let height: usize = 8;
    let pixels: Vec<u16> = (0..width * height).map(|i| (i * 100) as u16).collect();

    let mut handle = TjHandle::new();
    handle.set(TjParam::LosslessPsv, 1).unwrap();
    handle.set(TjParam::LosslessPt, 0).unwrap();

    let jpeg = handle.compress_16bit(&pixels, width, height, 1).unwrap();
    let img = handle.decompress_16bit(&jpeg).unwrap();

    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(handle.get(TjParam::Precision), img.precision as i32);
    assert_eq!(img.data, pixels, "16-bit lossless should be pixel-exact");
}

#[test]
fn handle_compress_decompress_12bit() {
    let width: usize = 8;
    let height: usize = 8;
    let num_components: usize = 1;
    // 12-bit grayscale pixels (0-4095)
    let pixels: Vec<i16> = (0..width * height).map(|i| (i * 50) as i16).collect();

    let mut handle = TjHandle::new();
    // Explicit since P4-155 (#539): a fresh handle's quality/subsampling
    // are *unset*, as upstream's are, and a lossy compress refuses them.
    handle.set(TjParam::Quality, 75).unwrap();
    handle.set(TjParam::Subsampling, 2).unwrap();

    let jpeg = handle
        .compress_12bit(&pixels, width, height, num_components)
        .unwrap();
    let img = handle.decompress_12bit(&jpeg).unwrap();

    assert_eq!(img.width, width);
    assert_eq!(img.height, height);
    assert_eq!(handle.get(TjParam::Precision), 12);
    assert_eq!(img.num_components, num_components);
}

/// P4-155 (#539): the native gates themselves, independent of any C oracle.
///
/// The oracle matrix in `capi_compress_precision.rs` cross-validates the
/// C-ABI shape but returns early without a TurboJPEG install; this pins the
/// `TjHandle` layer everywhere: a fresh handle's lossy compress refuses with
/// upstream's message for each missing parameter, and a lossless compress
/// consults neither (`turbojpeg-mp.c:95-98`).
#[test]
fn handle_lossy_compress_refuses_unset_params() {
    let width: usize = 16;
    let height: usize = 16;
    let pixels: Vec<u8> = vec![128u8; width * height * 3];

    let fresh = TjHandle::new();
    let err = fresh
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect_err("unset quality must refuse");
    assert!(
        err.to_string()
            .contains("TJPARAM_QUALITY must be specified"),
        "got: {err}"
    );

    let mut with_quality = TjHandle::new();
    with_quality.set(TjParam::Quality, 75).unwrap();
    let err = with_quality
        .compress(&pixels, width, height, PixelFormat::Rgb)
        .expect_err("unset subsampling must refuse");
    assert!(
        err.to_string()
            .contains("TJPARAM_SUBSAMP must be specified"),
        "got: {err}"
    );

    // The lossless bypass: neither parameter is consulted.
    let gray: Vec<u8> = vec![128u8; width * height];
    let mut lossless = TjHandle::new();
    lossless.set(TjParam::Lossless, 1).unwrap();
    lossless
        .compress(&gray, width, height, PixelFormat::Grayscale)
        .expect("lossless compress consults neither parameter");
}
