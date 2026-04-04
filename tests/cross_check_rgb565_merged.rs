//! Cross-validation: RGB565, merged upsample, and fast upsample paths vs C djpeg.
//!
//! Gaps addressed:
//! - RGB565 decode with S420/S422/S440/S411/S441 (only S444 was tested)
//! - RGB565 dithered decode for subsampled images
//! - Fast upsample (nosmooth) for all subsamplings vs C djpeg -nosmooth
//! - Merged upsample with all eligible subsamplings vs C djpeg
//!
//! All tests gracefully skip if djpeg is not found.

mod helpers;

use libjpeg_turbo_rs::{compress, decompress_to, PixelFormat, Subsampling};

// ===========================================================================
// Constants
// ===========================================================================

const TEST_WIDTH: usize = 48;
const TEST_HEIGHT: usize = 48;
const QUALITY: u8 = 90;

/// Subsamplings to test. C libjpeg-turbo tests all 7 modes.
const ALL_SUBSAMPLINGS: &[(Subsampling, &str)] = &[
    (Subsampling::S444, "444"),
    (Subsampling::S422, "422"),
    (Subsampling::S420, "420"),
    (Subsampling::S440, "440"),
    (Subsampling::S411, "411"),
    (Subsampling::S441, "441"),
];

/// Subsamplings eligible for merged upsample (H2V1 or H2V2 only).
const MERGED_SUBSAMPLINGS: &[(Subsampling, &str)] =
    &[(Subsampling::S422, "422"), (Subsampling::S420, "420")];

// ===========================================================================
// Helpers
// ===========================================================================

fn make_test_jpeg(width: usize, height: usize, subsamp: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = helpers::generate_gradient(width, height);
    compress(&pixels, width, height, PixelFormat::Rgb, QUALITY, subsamp)
        .expect("compress must succeed")
}

/// Decode JPEG to RGB565 with Rust, returning raw 16-bit pixel data.
fn rust_decode_rgb565(jpeg_data: &[u8]) -> Vec<u8> {
    let img = decompress_to(jpeg_data, PixelFormat::Rgb565).expect("Rust RGB565 decode failed");
    img.data
}

/// Decode JPEG to RGB with Rust for reference comparison.
fn rust_decode_rgb(jpeg_data: &[u8]) -> (usize, usize, Vec<u8>) {
    let img = decompress_to(jpeg_data, PixelFormat::Rgb).expect("Rust RGB decode failed");
    (img.width, img.height, img.data)
}

/// Verify RGB565 pixels are correct quantization of RGB pixels.
/// Each RGB565 pixel packs R(5 bits), G(6 bits), B(5 bits) into 2 bytes (LE).
fn verify_rgb565_quantization(rgb: &[u8], rgb565: &[u8], width: usize, height: usize, label: &str) {
    let pixel_count: usize = width * height;
    assert_eq!(
        rgb.len(),
        pixel_count * 3,
        "{}: RGB data length mismatch",
        label
    );
    assert_eq!(
        rgb565.len(),
        pixel_count * 2,
        "{}: RGB565 data length mismatch",
        label
    );

    let mut max_r_diff: u8 = 0;
    let mut max_g_diff: u8 = 0;
    let mut max_b_diff: u8 = 0;

    for i in 0..pixel_count {
        let r: u8 = rgb[i * 3];
        let g: u8 = rgb[i * 3 + 1];
        let b: u8 = rgb[i * 3 + 2];

        let word: u16 = u16::from_le_bytes([rgb565[i * 2], rgb565[i * 2 + 1]]);
        let r565: u8 = ((word >> 11) & 0x1F) as u8;
        let g565: u8 = ((word >> 5) & 0x3F) as u8;
        let b565: u8 = (word & 0x1F) as u8;

        // Expected quantized values (truncation, matching C libjpeg-turbo)
        let expected_r5: u8 = r >> 3;
        let expected_g6: u8 = g >> 2;
        let expected_b5: u8 = b >> 3;

        let rd: u8 = (r565 as i16 - expected_r5 as i16).unsigned_abs() as u8;
        let gd: u8 = (g565 as i16 - expected_g6 as i16).unsigned_abs() as u8;
        let bd: u8 = (b565 as i16 - expected_b5 as i16).unsigned_abs() as u8;

        if rd > max_r_diff {
            max_r_diff = rd;
        }
        if gd > max_g_diff {
            max_g_diff = gd;
        }
        if bd > max_b_diff {
            max_b_diff = bd;
        }
    }

    // Allow ±1 for dithering/rounding in the 5-6-5 quantization
    assert!(
        max_r_diff <= 1 && max_g_diff <= 1 && max_b_diff <= 1,
        "{}: RGB565 quantization mismatch: max_r_diff={}, max_g_diff={}, max_b_diff={} (tolerance <=1)",
        label,
        max_r_diff,
        max_g_diff,
        max_b_diff
    );
}

// ===========================================================================
// RGB565 decode for all subsamplings (C cross-validated via RGB path)
// ===========================================================================

#[test]
fn c_xval_rgb565_decode_all_subsamplings() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    for &(subsamp, name) in ALL_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(TEST_WIDTH, TEST_HEIGHT, subsamp);
        let label: &str = &format!("rgb565_{}", name);

        // 1. Verify Rust RGB decode matches C djpeg (diff=0)
        let (rust_w, rust_h, rust_rgb) = rust_decode_rgb(&jpeg);
        let (c_w, c_h, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &jpeg, label);
        assert_eq!(rust_w, c_w, "{}: width mismatch", label);
        assert_eq!(rust_h, c_h, "{}: height mismatch", label);
        helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rust_w, rust_h, 3, label);

        // 2. Verify Rust RGB565 output is correct quantization of Rust RGB
        let rust_565: Vec<u8> = rust_decode_rgb565(&jpeg);
        verify_rgb565_quantization(&rust_rgb, &rust_565, rust_w, rust_h, label);
    }
}

#[test]
fn c_xval_rgb565_odd_dimensions() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    // Odd dimensions matching tjunittest.c: 35x39, 39x41, 41x35
    let odd_dims: &[(usize, usize)] = &[(35, 39), (39, 41), (41, 35)];
    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S444, "444"),
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
    ];

    for &(w, h) in odd_dims {
        for &(subsamp, sname) in subsamplings {
            let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);
            let label: String = format!("rgb565_odd_{}x{}_{}", w, h, sname);

            // RGB path C cross-validation
            let (rw, rh, rust_rgb) = rust_decode_rgb(&jpeg);
            let (cw, ch, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &jpeg, &label);
            assert_eq!(rw, cw, "{}: width mismatch", label);
            assert_eq!(rh, ch, "{}: height mismatch", label);
            helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);

            // RGB565 quantization correctness
            let rust_565: Vec<u8> = rust_decode_rgb565(&jpeg);
            verify_rgb565_quantization(&rust_rgb, &rust_565, rw, rh, &label);
        }
    }
}

// ===========================================================================
// RGB565 dithered decode for subsampled images
// ===========================================================================

#[test]
fn c_xval_rgb565_dithered_subsampled() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let subsamplings: &[(Subsampling, &str)] = &[
        (Subsampling::S420, "420"),
        (Subsampling::S422, "422"),
        (Subsampling::S444, "444"),
    ];

    for &(subsamp, sname) in subsamplings {
        let jpeg: Vec<u8> = make_test_jpeg(TEST_WIDTH, TEST_HEIGHT, subsamp);
        let label: String = format!("rgb565_dither_{}", sname);

        // Verify RGB565 decode produces correct output for subsampled images
        let (rw, rh, rust_rgb) = rust_decode_rgb(&jpeg);
        let rust_565: Vec<u8> = rust_decode_rgb565(&jpeg);
        let pixel_count: usize = rw * rh;
        assert_eq!(rust_565.len(), pixel_count * 2, "{}: RGB565 length", label);

        // C cross-validate the RGB path
        let (cw, ch, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &jpeg, &label);
        assert_eq!(rw, cw, "{}: width", label);
        assert_eq!(rh, ch, "{}: height", label);
        helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);

        // Verify RGB565 is correct quantization of the C-validated RGB
        verify_rgb565_quantization(&rust_rgb, &rust_565, rw, rh, &label);
    }
}

// ===========================================================================
// Fast upsample (nosmooth) for all subsamplings vs C djpeg -nosmooth
// ===========================================================================

#[test]
fn c_xval_fast_upsample_all_subsamplings() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    for &(subsamp, name) in ALL_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(TEST_WIDTH, TEST_HEIGHT, subsamp);
        let label: String = format!("nosmooth_{}", name);

        // Rust: decode with fast_upsample=true
        let mut dec =
            libjpeg_turbo_rs::ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new failed");
        dec.set_fast_upsample(true);
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img = dec.finish().expect("fast upsample decode failed");

        // C: djpeg -nosmooth -ppm
        let jpeg_file = helpers::TempFile::new(&format!("{}.jpg", label));
        let ppm_file = helpers::TempFile::new(&format!("{}.ppm", label));
        jpeg_file.write_bytes(&jpeg);

        let output = std::process::Command::new(&djpeg)
            .arg("-nosmooth")
            .arg("-ppm")
            .arg("-outfile")
            .arg(ppm_file.path())
            .arg(jpeg_file.path())
            .output()
            .expect("djpeg failed");
        assert!(
            output.status.success(),
            "djpeg -nosmooth failed for {}: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );
        let ppm_data = std::fs::read(ppm_file.path()).expect("read PPM");
        let (c_w, c_h, c_rgb) = helpers::parse_ppm(&ppm_data).expect("parse PPM");
        assert_eq!(rust_img.width, c_w, "{}: width", label);
        assert_eq!(rust_img.height, c_h, "{}: height", label);
        helpers::assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, &label);
    }
}

#[test]
fn c_xval_fast_upsample_odd_dimensions() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let odd_dims: &[(usize, usize)] = &[(35, 39), (39, 41), (41, 35)];
    let subsampled: &[(Subsampling, &str)] = &[
        (Subsampling::S422, "422"),
        (Subsampling::S420, "420"),
        (Subsampling::S440, "440"),
        (Subsampling::S411, "411"),
        (Subsampling::S441, "441"),
    ];

    for &(w, h) in odd_dims {
        for &(subsamp, sname) in subsampled {
            let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);
            let label: String = format!("nosmooth_odd_{}x{}_{}", w, h, sname);

            // Rust: fast upsample
            let mut dec =
                libjpeg_turbo_rs::ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new failed");
            dec.set_fast_upsample(true);
            dec.set_output_format(PixelFormat::Rgb);
            let rust_img = dec.finish().expect("fast upsample decode failed");

            // C: djpeg -nosmooth
            let jpeg_file = helpers::TempFile::new(&format!("{}.jpg", label));
            let ppm_file = helpers::TempFile::new(&format!("{}.ppm", label));
            jpeg_file.write_bytes(&jpeg);

            let output = std::process::Command::new(&djpeg)
                .arg("-nosmooth")
                .arg("-ppm")
                .arg("-outfile")
                .arg(ppm_file.path())
                .arg(jpeg_file.path())
                .output()
                .expect("djpeg failed");
            assert!(
                output.status.success(),
                "djpeg -nosmooth failed for {}: {}",
                label,
                String::from_utf8_lossy(&output.stderr)
            );
            let ppm_data = std::fs::read(ppm_file.path()).expect("read PPM");
            let (c_w, c_h, c_rgb) = helpers::parse_ppm(&ppm_data).expect("parse PPM");
            assert_eq!(rust_img.width, c_w, "{}: width", label);
            assert_eq!(rust_img.height, c_h, "{}: height", label);
            helpers::assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, &label);
        }
    }
}

// ===========================================================================
// Merged upsample C cross-validation (S422, S420)
// ===========================================================================

#[test]
fn c_xval_merged_upsample_vs_c_djpeg() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    for &(subsamp, name) in MERGED_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(TEST_WIDTH, TEST_HEIGHT, subsamp);
        let label: String = format!("merged_{}", name);

        // Rust: decode with merged upsample enabled
        let mut dec =
            libjpeg_turbo_rs::ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new failed");
        dec.set_merged_upsample(true);
        dec.set_output_format(PixelFormat::Rgb);
        let rust_img = dec.finish().expect("merged upsample decode failed");

        // C: djpeg -nosmooth (C's merged path uses box-filter like nosmooth)
        let jpeg_file = helpers::TempFile::new(&format!("{}.jpg", label));
        let ppm_file = helpers::TempFile::new(&format!("{}.ppm", label));
        jpeg_file.write_bytes(&jpeg);

        let output = std::process::Command::new(&djpeg)
            .arg("-nosmooth")
            .arg("-ppm")
            .arg("-outfile")
            .arg(ppm_file.path())
            .arg(jpeg_file.path())
            .output()
            .expect("djpeg failed");
        assert!(
            output.status.success(),
            "djpeg failed for {}: {}",
            label,
            String::from_utf8_lossy(&output.stderr)
        );
        let ppm_data = std::fs::read(ppm_file.path()).expect("read PPM");
        let (c_w, c_h, c_rgb) = helpers::parse_ppm(&ppm_data).expect("parse PPM");

        assert_eq!(rust_img.width, c_w, "{}: width", label);
        assert_eq!(rust_img.height, c_h, "{}: height", label);
        helpers::assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, &label);
    }
}

#[test]
fn c_xval_merged_upsample_odd_dimensions() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let odd_dims: &[(usize, usize)] = &[(35, 39), (39, 41), (41, 35), (31, 33), (63, 127)];

    for &(subsamp, name) in MERGED_SUBSAMPLINGS {
        for &(w, h) in odd_dims {
            let jpeg: Vec<u8> = make_test_jpeg(w, h, subsamp);
            let label: String = format!("merged_odd_{}x{}_{}", w, h, name);

            // Rust: merged upsample
            let mut dec =
                libjpeg_turbo_rs::ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new failed");
            dec.set_merged_upsample(true);
            dec.set_output_format(PixelFormat::Rgb);
            let rust_img = dec.finish().expect("merged upsample decode failed");

            // C: djpeg -nosmooth
            let jpeg_file = helpers::TempFile::new(&format!("{}.jpg", label));
            let ppm_file = helpers::TempFile::new(&format!("{}.ppm", label));
            jpeg_file.write_bytes(&jpeg);

            let output = std::process::Command::new(&djpeg)
                .arg("-nosmooth")
                .arg("-ppm")
                .arg("-outfile")
                .arg(ppm_file.path())
                .arg(jpeg_file.path())
                .output()
                .expect("djpeg failed");
            assert!(
                output.status.success(),
                "djpeg failed for {}: {}",
                label,
                String::from_utf8_lossy(&output.stderr)
            );
            let ppm_data = std::fs::read(ppm_file.path()).expect("read PPM");
            let (c_w, c_h, c_rgb) = helpers::parse_ppm(&ppm_data).expect("parse PPM");

            assert_eq!(rust_img.width, c_w, "{}: width", label);
            assert_eq!(rust_img.height, c_h, "{}: height", label);
            helpers::assert_pixels_identical(&rust_img.data, &c_rgb, c_w, c_h, 3, &label);
        }
    }
}

// ===========================================================================
// Merged upsample: verify merged and fast_upsample produce identical output
// ===========================================================================

#[test]
fn merged_equals_fast_upsample_all_eligible() {
    for &(subsamp, name) in MERGED_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(TEST_WIDTH, TEST_HEIGHT, subsamp);
        let label: String = format!("merged_eq_fast_{}", name);

        // Merged path
        let mut dec1 =
            libjpeg_turbo_rs::ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new failed");
        dec1.set_merged_upsample(true);
        dec1.set_output_format(PixelFormat::Rgb);
        let merged_img = dec1.finish().expect("merged decode failed");

        // Fast upsample path
        let mut dec2 =
            libjpeg_turbo_rs::ScanlineDecoder::new(&jpeg).expect("ScanlineDecoder::new failed");
        dec2.set_fast_upsample(true);
        dec2.set_output_format(PixelFormat::Rgb);
        let fast_img = dec2.finish().expect("fast upsample decode failed");

        assert_eq!(
            merged_img.data.len(),
            fast_img.data.len(),
            "{}: data length",
            label
        );
        let max_diff: u8 = helpers::pixel_max_diff(&merged_img.data, &fast_img.data);
        assert_eq!(
            max_diff, 0,
            "{}: merged and fast_upsample should produce identical output, got max_diff={}",
            label, max_diff
        );
    }
}

// ===========================================================================
// Fancy upsample (default) for all subsamplings vs C djpeg (default)
// ===========================================================================

#[test]
fn c_xval_fancy_upsample_all_subsamplings() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    for &(subsamp, name) in ALL_SUBSAMPLINGS {
        let jpeg: Vec<u8> = make_test_jpeg(TEST_WIDTH, TEST_HEIGHT, subsamp);
        let label: String = format!("fancy_{}", name);

        // Rust: default decode (fancy upsample)
        let (rust_w, rust_h, rust_rgb) = rust_decode_rgb(&jpeg);

        // C: djpeg default (fancy upsample)
        let (c_w, c_h, c_rgb) = helpers::decode_with_c_djpeg(&djpeg, &jpeg, &label);

        assert_eq!(rust_w, c_w, "{}: width", label);
        assert_eq!(rust_h, c_h, "{}: height", label);
        helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rust_w, rust_h, 3, &label);
    }
}
