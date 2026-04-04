//! Cross-validation: all 16 scaling factors vs C djpeg -scale.

mod helpers;

use libjpeg_turbo_rs::api::streaming::StreamingDecoder;
use libjpeg_turbo_rs::{compress, PixelFormat, ScalingFactor, Subsampling};

const QUALITY: u8 = 90;

const ALL_SCALES: &[(u32, u32, &str)] = &[
    (2, 1, "2_1"),
    (15, 8, "15_8"),
    (7, 4, "7_4"),
    (13, 8, "13_8"),
    (3, 2, "3_2"),
    (11, 8, "11_8"),
    (5, 4, "5_4"),
    (9, 8, "9_8"),
    (1, 1, "1_1"),
    (7, 8, "7_8"),
    (3, 4, "3_4"),
    (5, 8, "5_8"),
    (1, 2, "1_2"),
    (3, 8, "3_8"),
    (1, 4, "1_4"),
    (1, 8, "1_8"),
];

fn make_test_jpeg(w: usize, h: usize, subsamp: Subsampling) -> Vec<u8> {
    let pixels: Vec<u8> = helpers::generate_gradient(w, h);
    compress(&pixels, w, h, PixelFormat::Rgb, QUALITY, subsamp).expect("compress")
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
    let jpeg_file = helpers::TempFile::new(&format!("{label}.jpg"));
    let ppm_file = helpers::TempFile::new(&format!("{label}.ppm"));
    jpeg_file.write_bytes(jpeg);
    let output = std::process::Command::new(djpeg)
        .arg("-scale")
        .arg(format!("{num}/{denom}"))
        .arg("-ppm")
        .arg("-outfile")
        .arg(ppm_file.path())
        .arg(jpeg_file.path())
        .output()
        .unwrap_or_else(|e| panic!("{label}: djpeg failed: {e:?}"));
    assert!(
        output.status.success(),
        "{label}: djpeg failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let ppm_data = std::fs::read(ppm_file.path()).expect("read PPM");
    helpers::parse_ppm(&ppm_data).unwrap_or_else(|| panic!("{label}: parse PPM failed"))
}

#[test]
fn c_xval_extended_scaling_444() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(48, 48, Subsampling::S444);
    for &(num, denom, name) in ALL_SCALES {
        let label = format!("ext_scale_{name}_444_48x48");
        let (rw, rh, rust_rgb) = decode_scaled_rust(&jpeg, num, denom);
        let (cw, ch, c_rgb) = decode_scaled_c(&djpeg, &jpeg, num, denom, &label);
        assert_eq!(rw, cw, "{label}: width");
        assert_eq!(rh, ch, "{label}: height");
        helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);
    }
}

#[test]
fn c_xval_extended_scaling_420() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let jpeg: Vec<u8> = make_test_jpeg(48, 48, Subsampling::S420);
    for &(num, denom, name) in ALL_SCALES {
        let bs: u32 = (num * 8).div_ceil(denom);
        if bs % 2 != 0 && bs != 1 {
            continue;
        }
        let label = format!("ext_scale_{name}_420_48x48");
        let (rw, rh, rust_rgb) = decode_scaled_rust(&jpeg, num, denom);
        let (cw, ch, c_rgb) = decode_scaled_c(&djpeg, &jpeg, num, denom, &label);
        assert_eq!(rw, cw, "{label}: width");
        assert_eq!(rh, ch, "{label}: height");
        helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);
    }
}

#[test]
fn c_xval_extended_scaling_odd_dimensions() {
    let djpeg = match helpers::djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    for &(w, h) in &[(35usize, 39usize), (39, 41)] {
        let jpeg: Vec<u8> = make_test_jpeg(w, h, Subsampling::S444);
        for &(num, denom, name) in ALL_SCALES {
            let label = format!("ext_scale_{name}_444_{w}x{h}");
            let (rw, rh, rust_rgb) = decode_scaled_rust(&jpeg, num, denom);
            let (cw, ch, c_rgb) = decode_scaled_c(&djpeg, &jpeg, num, denom, &label);
            assert_eq!(rw, cw, "{label}: width");
            assert_eq!(rh, ch, "{label}: height");
            helpers::assert_pixels_identical(&rust_rgb, &c_rgb, rw, rh, 3, &label);
        }
    }
}
