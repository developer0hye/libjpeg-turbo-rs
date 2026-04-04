//! Cross-validation: scaling factors for decode vs C djpeg -scale.
//!
//! Gaps addressed:
//! - Scaling factors 1/1, 1/2, 1/4, 1/8 tested across ALL subsamplings
//!   (previously only 1/2 and 1/4 with S444/S420)
//! - Odd dimensions at all scales matching tjunittest.c
//! - Per-subsampling factor restrictions matching C (tjunittest.c:672-681)
//!
//! Missing (requires 12 new IDCT kernels, ~2000 LOC feature work):
//! - Scaling factors: 2/1, 15/8, 7/4, 13/8, 3/2, 11/8, 5/4, 9/8, 7/8, 3/4, 5/8, 3/8
//!
//! All tests gracefully skip if djpeg is not found.

mod helpers;

use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::{compress, PixelFormat, ScalingFactor, Subsampling};

// ===========================================================================
// Constants
// ===========================================================================

const QUALITY: u8 = 90;

/// The 4 scaling factors supported by the Rust decoder.
const SUPPORTED_SCALES: &[(u32, u32, &str)] =
    &[(1, 1, "1_1"), (1, 2, "1_2"), (1, 4, "1_4"), (1, 8, "1_8")];

/// Per-subsampling factor restrictions (from tjunittest.c:672-681):
/// - S444: all factors
/// - S422, S440: 1/1, 1/2, 1/4
/// - S420: 1/1, 1/2, 1/4, 1/8
/// - S411, S441: 1/1, 1/2
fn is_valid_scale_for_subsamp(num: u32, denom: u32, subsamp: Subsampling) -> bool {
    match subsamp {
        Subsampling::S444 => true,
        Subsampling::S420 => true, // all 4 supported factors valid for 420
        Subsampling::S422 | Subsampling::S440 => {
            // 1/1, 1/2, 1/4 only (no 1/8)
            !(num == 1 && denom == 8)
        }
        Subsampling::S411 | Subsampling::S441 => {
            // 1/1, 1/2 only
            (num == 1 && denom == 1) || (num == 1 && denom == 2)
        }
        _ => num == 1 && denom == 1,
    }
}

const ALL_SUBSAMPLINGS: &[(Subsampling, &str)] = &[
    (Subsampling::S444, "444"),
    (Subsampling::S422, "422"),
    (Subsampling::S420, "420"),
    (Subsampling::S440, "440"),
    (Subsampling::S411, "411"),
    (Subsampling::S441, "441"),
];

// ===========================================================================
// Helpers
// ===========================================================================

fn make_test_jpeg(w: usize, h: usize, subsamp: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = helpers::generate_gradient(w, h);
    compress(&pixels, w, h, PixelFormat::Rgb, QUALITY, subsamp).expect("compress must succeed")
}

fn decode_scaled_rust(jpeg: &[u8], num: u32, denom: u32) -> (usize, usize, Vec<u8>) {
    let mut decoder = StreamingDecoder::new(jpeg).expect("StreamingDecoder::new");
    decoder.set_scale(ScalingFactor::new(num, denom));
    decoder.set_output_format(PixelFormat::Rgb);
    let img = decoder.decode().expect("decode");
    (img.width, img.height, img.data)
}

fn decode_scaled_c(
    djpeg: &std::path::Path,
    jpeg: &[u8],
    num: u32,
    denom: u32,
    label: &str,
) -> (usize, usize, Vec<u8>) {
    let jpeg_file = helpers::TempFile::new(&format!("{}.jpg", label));
    let ppm_file = helpers::TempFile::new(&format!("{}.ppm", label));
    jpeg_file.write_bytes(jpeg);

    let scale_arg: String = format!("{}/{}", num, denom);
    let output = std::process::Command::new(djpeg)
        .arg("-scale")
        .arg(&scale_arg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(ppm_file.path())
        .arg(jpeg_file.path())
        .output()
        .unwrap_or_else(|e| panic!("{}: djpeg failed: {:?}", label, e));

    assert!(
        output.status.success(),
        "{}: djpeg -scale {}/{} failed: {}",
        label,
        num,
        denom,
        String::from_utf8_lossy(&output.stderr)
    );

    let ppm_data = std::fs::read(ppm_file.path()).expect("read PPM");
    helpers::parse_ppm(&ppm_data).unwrap_or_else(|| panic!("{}: parse PPM failed", label))
}

// ===========================================================================
// All supported scaling factors x all subsamplings
// ===========================================================================

#[test]
fn c_xval_scaling_all_factors_all_subsamplings() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let w: usize = 48;
    let h: usize = 48;

    for &(subsamp, sname) in ALL_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);

        for &(num, denom, scale_name) in SUPPORTED_SCALES {
            if !is_valid_scale_for_subsamp(num, denom, subsamp) {
                continue;
            }

            let label: String = format!("scale_{}_{}_{}", scale_name, sname, "48x48");

            let (rw, rh, rust_rgb) = decode_scaled_rust(&jpeg, num, denom);
            let (cw, ch, c_rgb) = decode_scaled_c(&djpeg, &jpeg, num, denom, &label);

            assert_eq!(rw, cw, "{}: width (rust={}, c={})", label, rw, cw);
            assert_eq!(rh, ch, "{}: height (rust={}, c={})", label, rh, ch);
            helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);
        }
    }
}

// ===========================================================================
// Odd dimensions matching tjunittest.c
// ===========================================================================

#[test]
fn c_xval_scaling_odd_dimensions() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let odd_dims: &[(usize, usize)] = &[(35, 39), (39, 41), (41, 35)];

    for &(w, h) in odd_dims {
        for &(subsamp, sname) in ALL_SUBSAMPLINGS {
            let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);

            for &(num, denom, scale_name) in SUPPORTED_SCALES {
                if !is_valid_scale_for_subsamp(num, denom, subsamp) {
                    continue;
                }

                let label: String = format!("scale_{}_{}_{}x{}", scale_name, sname, w, h);

                let (rw, rh, rust_rgb) = decode_scaled_rust(&jpeg, num, denom);
                let (cw, ch, c_rgb) = decode_scaled_c(&djpeg, &jpeg, num, denom, &label);

                assert_eq!(rw, cw, "{}: width", label);
                assert_eq!(rh, ch, "{}: height", label);
                helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);
            }
        }
    }
}

// ===========================================================================
// Scale decode with different content types (photo, graphic, checker)
// ===========================================================================

#[test]
fn c_xval_scaling_fixture_images() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let fixtures: &[(&str, &str)] = &[
        ("tests/fixtures/photo_320x240_420.jpg", "photo_420"),
        ("tests/fixtures/photo_320x240_422.jpg", "photo_422"),
        ("tests/fixtures/photo_320x240_444.jpg", "photo_444"),
    ];

    for &(path, name) in fixtures {
        let jpeg = match std::fs::read(path) {
            Ok(data) => data,
            Err(_) => {
                eprintln!("SKIP: fixture {} not found", path);
                continue;
            }
        };

        for &(num, denom, scale_name) in SUPPORTED_SCALES {
            let label: String = format!("fixture_{}_{}", name, scale_name);

            let (rw, rh, rust_rgb) = decode_scaled_rust(&jpeg, num, denom);
            let (cw, ch, c_rgb) = decode_scaled_c(&djpeg, &jpeg, num, denom, &label);

            assert_eq!(rw, cw, "{}: width", label);
            assert_eq!(rh, ch, "{}: height", label);
            helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);
        }
    }
}
