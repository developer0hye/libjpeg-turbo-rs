use libjpeg_turbo_rs::tj3::{TjHandle, TjParam};
use libjpeg_turbo_rs::{compress, decompress, PixelFormat, Subsampling};

#[test]
fn handle_default_values() {
    let handle = TjHandle::new();
    // Default quality = 75
    assert_eq!(handle.get(TjParam::Quality), 75);
    // Default subsampling = S420 = index 2
    assert_eq!(handle.get(TjParam::Subsampling), 2);
    // Default precision = 8
    assert_eq!(handle.get(TjParam::Precision), 8);
    // Default colorspace = YCbCr = 1
    assert_eq!(handle.get(TjParam::ColorSpace), 1);
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
    // SaveMarkers default = 0 (None)
    assert_eq!(handle.get(TjParam::SaveMarkers), 0);
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
    // 0 = None, 1 = All
    handle.set(TjParam::SaveMarkers, 1).unwrap();
    assert_eq!(handle.get(TjParam::SaveMarkers), 1);
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
    // Valid subsampling indices: 0-5
    assert!(handle.set(TjParam::Subsampling, -1).is_err());
    assert!(handle.set(TjParam::Subsampling, 7).is_err());
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
    // Valid scaling factors
    handle.set_scaling_factor(1, 1).unwrap();
    handle.set_scaling_factor(1, 2).unwrap();
    handle.set_scaling_factor(1, 4).unwrap();
    handle.set_scaling_factor(1, 8).unwrap();
    // Invalid scaling factor
    assert!(handle.set_scaling_factor(1, 3).is_err());
    assert!(handle.set_scaling_factor(0, 1).is_err());
    assert!(handle.set_scaling_factor(2, 1).is_err());
}

#[test]
fn handle_cropping_region() {
    use libjpeg_turbo_rs::CropRegion;
    let mut handle = TjHandle::new();
    handle.set_cropping_region(Some(CropRegion {
        x: 10,
        y: 20,
        width: 100,
        height: 200,
    }));
    handle.set_cropping_region(None);
}

#[test]
fn handle_scaling_factors_list() {
    let factors = TjHandle::scaling_factors();
    assert!(factors.contains(&(1, 1)));
    assert!(factors.contains(&(1, 2)));
    assert!(factors.contains(&(1, 4)));
    assert!(factors.contains(&(1, 8)));
    assert_eq!(factors.len(), 4);
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
    let img = handle.decompress(&jpeg).unwrap();
    assert_eq!(img.icc_profile(), Some(icc.as_slice()));
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
